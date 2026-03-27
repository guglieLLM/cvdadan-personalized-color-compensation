"""
CVD-AdaIN: Adaptive Instance Normalization condizionato dal profilo CVD.

Questo modulo implementa la sostituzione dei LayerNorm con AdaIN modulato
dal vettore CVD [θ, C, S] in modo che la normalizzazione sia DINAMICA
e dipendente dal profilo di daltonismo dell'utente.

Architettura:
    LayerNorm statico (γ, β fissi) → CVDAdaIN dinamico (γ, β = f(cvd_profile))

Formula:
    x_norm = (x - μ) / σ                    (Instance Normalization)
    γ, β = Linear(cvd_profile)              (Proiezione lineare CVD → params)
    output = γ * x_norm + β                 (Affine transform dinamico)

Uso:
    - Sostituisce LayerNorm nei blocchi ConvNeXt dell'encoder
    - Sostituisce ChannelLayerNorm nei blocchi del decoder
    - Il cvd_profile viene passato attraverso tutta la rete

References:
    - AdaIN: "Arbitrary Style Transfer in Real-time" (Huang & Belongie, ICCV 2017)
    - ConvNeXt: "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVDAdaIN(nn.Module):
    """
    Adaptive Instance Normalization condizionato dal profilo CVD.
    
    Sostituisce LayerNorm nei blocchi ConvNeXt e nel decoder.
    Supporta sia formato channels_last [B,H,W,C] che channels_first [B,C,H,W].
    
    Il vettore CVD [θ, C, S] modula dinamicamente γ e β:
    - Visione normale (0,0,0) → γ≈1, β≈0 (quasi identity)
    - CVD severo → γ, β modulati per compensazione
    
    Args:
        num_channels: Numero di canali delle feature
        cvd_dim: Dimensione del vettore CVD (default: 3 per [θ_norm, C_norm, S_norm])
        eps: Epsilon per stabilità numerica
        data_format: "channels_last" [B,H,W,C] o "channels_first" [B,C,H,W]
    """
    
    def __init__(self, 
                 num_channels: int,
                 cvd_dim: int = 3,
                 eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super().__init__()
        
        self.num_channels = num_channels
        self.cvd_dim = cvd_dim
        self.eps = eps
        self.data_format = data_format
        
        # Proiezione lineare: cvd [3] → [γ, β] [2C]
        # Questo è il cuore dell'AdaIN: genera γ e β dal profilo CVD
        self.style_projection = nn.Linear(cvd_dim, num_channels * 2)
        
        # Inizializzazione per comportamento identity iniziale
        # γ=1, β=0 → output = x_normalized (nessuna modulazione all'inizio)
        self._init_weights_identity()
    
    def _init_weights_identity(self):
        """Inizializza per comportamento identity: γ=1, β=0."""
        # Pesi a zero → output della Linear dipende solo dal bias
        nn.init.zeros_(self.style_projection.weight)
        
        # Bias: [γ₁, γ₂, ..., γ_C, β₁, β₂, ..., β_C]
        # Prima metà (γ) = 1.0, Seconda metà (β) = 0.0
        with torch.no_grad():
            self.style_projection.bias[:self.num_channels].fill_(1.0)   # γ = 1
            self.style_projection.bias[self.num_channels:].fill_(0.0)   # β = 0
    
    def forward(self, 
                x: torch.Tensor,           # [B,H,W,C] o [B,C,H,W]
                cvd_profile: torch.Tensor  # [B, 3]
               ) -> torch.Tensor:
        """
        Forward pass con normalizzazione condizionata da CVD.
        
        Args:
            x: Feature tensor [B,H,W,C] (channels_last) o [B,C,H,W] (channels_first)
            cvd_profile: Profilo CVD normalizzato [B, 3] (θ, C, S)
        
        Returns:
            Feature tensor modulato, stessa shape dell'input
        """
        
        if self.data_format == "channels_last":
            return self._forward_channels_last(x, cvd_profile)
        else:
            return self._forward_channels_first(x, cvd_profile)
    
    def _forward_channels_last(self, x: torch.Tensor, cvd_profile: torch.Tensor) -> torch.Tensor:
        """Forward per tensori [B, H, W, C] (formato ConvNeXt interno)."""
        # x: [B, H, W, C]
        
        # Instance Normalization sull'ultima dimensione (C)
        mu = x.mean(dim=-1, keepdim=True)      # [B, H, W, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # SAFETY: Clamp variance to prevent extreme gradients during backward
        var = torch.clamp(var, min=self.eps)
        
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        
        # SAFETY: Clamp normalized output to prevent explosion
        x_norm = torch.clamp(x_norm, min=-10.0, max=10.0)
        
        # Genera γ, β dal CVD profile
        style = self.style_projection(cvd_profile)  # [B, 2C]
        gamma, beta = style.chunk(2, dim=1)          # [B, C] each
        
        # Reshape per broadcasting: [B, C] → [B, 1, 1, C]
        gamma = gamma.unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)
        
        # Affine transform
        return gamma * x_norm + beta
    
    def _forward_channels_first(self, x: torch.Tensor, cvd_profile: torch.Tensor) -> torch.Tensor:
        """Forward per tensori [B, C, H, W] (formato decoder)."""
        B, C, H, W = x.shape
        
        # Instance Normalization: normalizza su (H, W) per ogni canale
        mu = x.mean(dim=[2, 3], keepdim=True)      # [B, C, 1, 1]
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        var_stable = torch.clamp(var, min=1e-8)  # Prevent division by near-zero variance
        x_norm = (x - mu) / torch.sqrt(var_stable + self.eps)
        
        # Genera γ, β dal CVD profile
        style = self.style_projection(cvd_profile)  # [B, 2C]
        gamma, beta = style.chunk(2, dim=1)          # [B, C] each
        
        # Reshape per broadcasting: [B, C] → [B, C, 1, 1]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        # Affine transform
        return gamma * x_norm + beta
    
    def extra_repr(self) -> str:
        return f'num_channels={self.num_channels}, cvd_dim={self.cvd_dim}, format={self.data_format}'


class CVDAdaINChannelFirst(CVDAdaIN):
    """
    Alias per CVDAdaIN con data_format="channels_first".
    Usato nel decoder per tensori [B, C, H, W].
    """
    def __init__(self, num_channels: int, cvd_dim: int = 3, eps: float = 1e-6):
        super().__init__(num_channels, cvd_dim, eps, data_format="channels_first")


class CVDAdaINChannelLast(CVDAdaIN):
    """
    Alias per CVDAdaIN con data_format="channels_last".
    Usato nell'encoder ConvNeXt per tensori [B, H, W, C].
    """
    def __init__(self, num_channels: int, cvd_dim: int = 3, eps: float = 1e-6):
        super().__init__(num_channels, cvd_dim, eps, data_format="channels_last")


# =============================================================================
# BLOCCHI CONVNEXT CON CVD-ADAIN INTEGRATO
# =============================================================================

class StochasticDepth(nn.Module):
    """Drop path (stochastic depth) per training regularization."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class CVDConvNeXtBlock(nn.Module):
    """
    Blocco ConvNeXt con LayerNorm sostituito da CVDAdaIN.
    
    Struttura originale:
        DWConv → Permute → LayerNorm → Linear → GELU → Linear → Scale → Permute → + Residual
    
    Struttura modificata:
        DWConv → Permute → CVDAdaIN(cvd) → Linear → GELU → Linear → Scale → Permute → + Residual
    
    Args:
        dim: Numero di canali input/output
        cvd_dim: Dimensione vettore CVD [θ_norm, C_norm, S_norm]
        drop_path: Probabilità drop path
        layer_scale_init: Valore iniziale per layer scale
    """
    
    def __init__(self,
                 dim: int,
                 cvd_dim: int = 3,
                 drop_path: float = 0.0,
                 layer_scale_init: float = 1e-6):
        super().__init__()
        
        self.dim = dim
        
        # Depthwise conv 7×7
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # CVDAdaIN al posto di LayerNorm (opera su channels_last dopo permute)
        self.norm = CVDAdaINChannelLast(dim, cvd_dim=cvd_dim)
        
        # MLP: expand → GELU → squeeze
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale (come in ConvNeXt originale)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        
        # Drop Path (stochastic depth)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, 
                x: torch.Tensor,           # [B, C, H, W]
                cvd_profile: torch.Tensor  # [B, 3]
               ) -> torch.Tensor:
        """
        Forward pass con CVDAdaIN condizionato.
        
        Args:
            x: Feature map [B, C, H, W]
            cvd_profile: Profilo CVD normalizzato [B, 3]
        
        Returns:
            Feature map [B, C, H, W]
        """
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)                          # [B, C, H, W]
        
        # Permute to channels-last (come ConvNeXt originale)
        x = x.permute(0, 2, 3, 1)                   # [B, H, W, C]
        
        # CVDAdaIN (sostituisce LayerNorm)
        x = self.norm(x, cvd_profile)               # [B, H, W, C]
        
        # MLP
        x = self.pwconv1(x)                         # [B, H, W, 4C]
        x = self.act(x)
        x = self.pwconv2(x)                         # [B, H, W, C]
        
        # Layer Scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back to channels-first
        x = x.permute(0, 3, 1, 2)                   # [B, C, H, W]
        
        # Residual + Drop Path
        x = residual + self.drop_path(x)
        
        return x


class CVDConvNeXtDownsample(nn.Module):
    """
    Downsample layer per ConvNeXt con LayerNorm → CVDAdaIN.
    
    Struttura: LayerNorm → Conv2d(stride=2)
    Modificata: CVDAdaIN → Conv2d(stride=2)
    """
    
    def __init__(self, in_channels: int, out_channels: int, cvd_dim: int = 3):
        super().__init__()
        
        # NOTA: Il downsample usa LayerNorm in formato channels_first
        # Dobbiamo convertire a channels_last, applicare norm, poi tornare
        self.norm = CVDAdaINChannelFirst(in_channels, cvd_dim=cvd_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor, cvd_profile: torch.Tensor) -> torch.Tensor:
        x = self.norm(x, cvd_profile)
        x = self.conv(x)
        return x


class CVDConvNeXtStage(nn.Module):
    """
    Stage di ConvNeXt con tutti i blocchi che usano CVDAdaIN.
    
    Args:
        in_channels: Canali input (per downsample)
        out_channels: Canali output
        num_blocks: Numero di blocchi ConvNeXt
        cvd_dim: Dimensione vettore CVD
        downsample: Se True, applica downsample 2x all'inizio
        drop_path_rates: Lista di probabilità drop path per ogni blocco
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 cvd_dim: int = 3,
                 downsample: bool = False,
                 drop_path_rates: list = None):
        super().__init__()
        
        self.has_downsample = downsample
        
        # Downsample layer (se necessario)
        if downsample:
            self.downsample = CVDConvNeXtDownsample(in_channels, out_channels, cvd_dim)
        else:
            self.downsample = None
            
        # Blocchi ConvNeXt con CVDAdaIN
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        
        self.blocks = nn.ModuleList([
            CVDConvNeXtBlock(
                dim=out_channels,
                cvd_dim=cvd_dim,
                drop_path=drop_path_rates[i]
            )
            for i in range(num_blocks)
        ])
    
    def forward(self, 
                x: torch.Tensor,           # [B, C_in, H, W]
                cvd_profile: torch.Tensor  # [B, 3]
               ) -> torch.Tensor:
        """
        Forward pass dello stage.
        
        Args:
            x: Feature map [B, C_in, H, W]
            cvd_profile: Profilo CVD normalizzato [B, 3]
        
        Returns:
            Feature map [B, C_out, H_out, W_out]
        """
        # Downsample (se presente)
        if self.downsample is not None:
            x = self.downsample(x, cvd_profile)
        
        # Blocchi
        for block in self.blocks:
            x = block(x, cvd_profile)
        
        return x


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def make_cvd_ln(num_channels: int, cvd_dim: int = 3, data_format: str = "channels_first"):
    """
    Factory function per creare CVDAdaIN.
    Drop-in replacement per make_ln() nel decoder.
    
    Args:
        num_channels: Numero di canali
        cvd_dim: Dimensione vettore CVD
        data_format: "channels_first" o "channels_last"
    
    Returns:
        CVDAdaIN module
    """
    return CVDAdaIN(num_channels, cvd_dim=cvd_dim, data_format=data_format)


def count_cvd_adain_parameters(model: nn.Module) -> dict:
    """
    Conta i parametri CVDAdaIN nel modello.
    
    Returns:
        Dict con conteggi per tipo di parametro
    """
    cvd_params = 0
    other_params = 0
    
    for name, param in model.named_parameters():
        if 'style_projection' in name:
            cvd_params += param.numel()
        else:
            other_params += param.numel()
    
    return {
        'cvd_adain_params': cvd_params,
        'other_params': other_params,
        'total': cvd_params + other_params,
        'cvd_percentage': 100 * cvd_params / (cvd_params + other_params) if (cvd_params + other_params) > 0 else 0
    }
