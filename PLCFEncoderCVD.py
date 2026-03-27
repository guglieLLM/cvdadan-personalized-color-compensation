"""
PLCFEncoderCVD — Encoder ConvNeXt-Tiny con CVDAdaIN interno ai blocchi.

Sostituisce tutti i LayerNorm interni ai blocchi ConvNeXt con CVDAdaIN
(modulazione affine condizionata dal profilo CVD 3D [θ, C, S]).
Lo stem rimane congelato; il resto dell'encoder è fine-tuned con lr dedicato.

Architettura ConvNeXt-Tiny modificata (input 256×256):

    Input [B, 3, 256, 256]
        ↓
    Stem (Conv 4×4 stride 4 + LayerNorm) → [B, 96, 64, 64]   (congelato)
        ↓
    Stage 0: 3 blocchi con CVDAdaIN → [B, 96, 64, 64]
        ↓
    Stage 1: Downsample + 3 blocchi con CVDAdaIN → [B, 192, 32, 32]
        ↓
    Stage 2: Downsample + 9 blocchi con CVDAdaIN → [B, 384, 16, 16]
        ↓
    Output (Bottleneck)

Punti CVDAdaIN nell'encoder: **17 totali**
    - 15 nei blocchi ConvNeXt (3+3+9, uno per blocco)
    - 2 nei layer di downsample (Stage 1 e Stage 2)

Ogni blocco ConvNeXt:
    DWConv → Permute → CVDAdaIN(cvd) → Linear → GELU → Linear → Scale → Permute → + Residual

Dipendenze:
    cvd_adain_modules (CVDConvNeXtBlock, CVDConvNeXtStage,
    CVDConvNeXtDownsample, CVDAdaINChannelFirst).
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from typing import Tuple, Optional

from cvd_adain_modules import (
    CVDConvNeXtBlock, 
    CVDConvNeXtStage,
    CVDConvNeXtDownsample,
    CVDAdaINChannelFirst
)


class CVDConvNeXtStem(nn.Module):
    """
    Stem di ConvNeXt: patchify + norm.
    
    Il LayerNorm dello stem NON viene sostituito con CVDAdaIN
    perché opera sull'input grezzo (non ha senso modulare l'input).
    """
    
    def __init__(self, pretrained_stem: nn.Module = None):
        super().__init__()
        
        if pretrained_stem is not None:
            self.stem = pretrained_stem
        else:
            # Stem from scratch
            self.stem = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                nn.GroupNorm(1, 96)  # LayerNorm equivalent
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PLCFEncoderCVD(nn.Module):
    """Encoder ConvNeXt-Tiny con CVDAdaIN in ogni blocco e downsample.

    Sostituisce i LayerNorm nei blocchi ConvNeXt con CVDAdaIN, consentendo
    normalizzazione dinamica condizionata dal profilo CVD 3D [θ, C, S].
    Lo stem viene congelato; i restanti parametri sono fine-tuned con lr
    dedicato (``encoder_learning_rate`` nel config YAML).

    Args:
        cvd_dim: Dimensione vettore CVD (default 3: θ, C, S).
        pretrained: Se True carica pesi ImageNet-1k da ConvNeXt-Tiny.
        freeze_stem: Se True congela il patchify layer iniziale.
        stop_at_stage: Ultimo stage da usare (0, 1 o 2). Default 2 → 384 ch.
        drop_path_rate: Probabilità base per stochastic depth.
    """
    
    # Configurazione ConvNeXt-Tiny
    DIMS = [96, 192, 384, 768]      # Canali per stage
    DEPTHS = [3, 3, 9, 3]            # Blocchi per stage
    
    def __init__(self,
                 cvd_dim: int = 3,
                 pretrained: bool = True,
                 freeze_stem: bool = True,
                 stop_at_stage: int = 2,
                 drop_path_rate: float = 0.1):
        super().__init__()
        
        self.cvd_dim = cvd_dim
        self.stop_at_stage = stop_at_stage
        self.freeze_stem = freeze_stem
        
        # Calcola drop path rates con incremento lineare
        total_blocks = sum(self.DEPTHS[:stop_at_stage + 1])
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # ═══════════════════════════════════════════════════════════════
        # CARICA MODELLO PRETRAINED (se richiesto)
        # ═══════════════════════════════════════════════════════════════
        pretrained_model = None
        if pretrained:
            print("[PLCFEncoderCVD] Caricamento ConvNeXt-Tiny pretrained...")
            pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # ═══════════════════════════════════════════════════════════════
        # STEM (lo teniamo come nell'originale, senza CVDAdaIN)
        # ═══════════════════════════════════════════════════════════════
        if pretrained_model is not None:
            self.stem = CVDConvNeXtStem(pretrained_model.features[0])
        else:
            self.stem = CVDConvNeXtStem(None)
        
        if freeze_stem:
            for param in self.stem.parameters():
                param.requires_grad = False
        
        # ═══════════════════════════════════════════════════════════════
        # STAGES CON CVD-ADAIN
        # ═══════════════════════════════════════════════════════════════
        self.stages = nn.ModuleList()
        block_idx = 0
        
        for stage_idx in range(stop_at_stage + 1):
            in_ch = self.DIMS[0] if stage_idx == 0 else self.DIMS[stage_idx - 1]
            out_ch = self.DIMS[stage_idx]
            num_blocks = self.DEPTHS[stage_idx]
            downsample = (stage_idx > 0)
            
            # Drop path rates per questo stage
            stage_drop_rates = drop_path_rates[block_idx:block_idx + num_blocks]
            block_idx += num_blocks
            
            # Crea stage
            stage = CVDConvNeXtStage(
                in_channels=in_ch,
                out_channels=out_ch,
                num_blocks=num_blocks,
                cvd_dim=cvd_dim,
                downsample=downsample,
                drop_path_rates=stage_drop_rates
            )
            
            # Copia pesi pretrained (se disponibili)
            if pretrained_model is not None:
                self._copy_stage_weights(stage, pretrained_model, stage_idx)
            
            self.stages.append(stage)
        
        # Output channels
        self._out_channels = self.DIMS[stop_at_stage]
        
        # Log info
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[PLCFEncoderCVD] Inizializzato:")
        print(f"  - Stop at stage: {stop_at_stage}")
        print(f"  - Output channels: {self._out_channels}")
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Stem frozen: {freeze_stem}")
    
    def _copy_stage_weights(self, stage: CVDConvNeXtStage, pretrained: nn.Module, stage_idx: int):
        """
        Copia i pesi pretrained dallo stage originale.
        
        Struttura pretrained.features:
            [0]: stem
            [1]: stage0 blocks
            [2]: stage1 downsample
            [3]: stage1 blocks
            [4]: stage2 downsample
            [5]: stage2 blocks
            [6]: stage3 downsample
            [7]: stage3 blocks
        """
        with torch.no_grad():
            # Indici in pretrained.features
            if stage_idx == 0:
                pretrained_blocks = pretrained.features[1]
                pretrained_downsample = None
            else:
                downsample_idx = stage_idx * 2
                blocks_idx = stage_idx * 2 + 1
                pretrained_downsample = pretrained.features[downsample_idx]
                pretrained_blocks = pretrained.features[blocks_idx]
            
            # Copia pesi downsample (se presente)
            if stage.downsample is not None and pretrained_downsample is not None:
                # Il downsample pretrained ha: LayerNorm + Conv2d
                # Il nostro ha: CVDAdaIN + Conv2d
                # Copiamo solo la Conv2d
                stage.downsample.conv.weight.copy_(pretrained_downsample[1].weight)
                stage.downsample.conv.bias.copy_(pretrained_downsample[1].bias)
            
            # Copia pesi blocchi
            for block_idx, (cvd_block, pretrained_block) in enumerate(zip(stage.blocks, pretrained_blocks)):
                # Struttura pretrained_block.block:
                # [0]: dwconv (Conv2d)
                # [1]: Permute
                # [2]: LayerNorm (non copiamo - sostituito con CVDAdaIN)
                # [3]: pwconv1 (Linear)
                # [4]: GELU
                # [5]: grn (Global Response Norm, opzionale in alcune versioni)
                # [6]: pwconv2 (Linear) 
                # [7]: Layer scale
                
                # dwconv
                cvd_block.dwconv.weight.copy_(pretrained_block.block[0].weight)
                cvd_block.dwconv.bias.copy_(pretrained_block.block[0].bias)
                
                # pwconv1 (Linear) - indice può variare
                pwconv1_idx = 3
                cvd_block.pwconv1.weight.copy_(pretrained_block.block[pwconv1_idx].weight)
                cvd_block.pwconv1.bias.copy_(pretrained_block.block[pwconv1_idx].bias)
                
                # pwconv2 (Linear)
                # Cerca il secondo Linear
                linear_count = 0
                for i, layer in enumerate(pretrained_block.block):
                    if isinstance(layer, nn.Linear):
                        linear_count += 1
                        if linear_count == 2:
                            cvd_block.pwconv2.weight.copy_(layer.weight)
                            cvd_block.pwconv2.bias.copy_(layer.bias)
                            break
                
                # Layer scale (gamma)
                # ConvNeXt usa layer_scale con shape [C, 1, 1], il nostro gamma è [C]
                if hasattr(pretrained_block, 'layer_scale') and pretrained_block.layer_scale is not None:
                    # Squeeze per rimuovere le dimensioni 1
                    ls = pretrained_block.layer_scale.squeeze()  # [C, 1, 1] -> [C]
                    cvd_block.gamma.copy_(ls)
                else:
                    # Cerca nell'ultimo elemento del block
                    for layer in reversed(list(pretrained_block.block)):
                        if hasattr(layer, 'gamma'):
                            gamma = layer.gamma
                            if gamma.dim() > 1:
                                gamma = gamma.squeeze()
                            cvd_block.gamma.copy_(gamma)
                            break
    
    @property
    def out_channels(self) -> int:
        """Numero di canali output."""
        return self._out_channels
    
    def forward(self,
                x: torch.Tensor,           # [B, 3, H, W]
                cvd_profile: torch.Tensor  # [B, 3]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass dell'encoder.

        Args:
            x: Immagine input [B, 3, H, W] (normalizzata ImageNet).
            cvd_profile: Profilo CVD normalizzato [B, 3] (θ, C, S).

        Returns:
            (latent, skip_feat):
                latent — feature map [B, C, H', W'] (es. [B, 384, 16, 16]
                    per input 256×256 con stop_at_stage=2).
                skip_feat — output Stage 0 [B, 96, H/4, W/4] per eventuale
                    skip connection nel decoder.
        """
        # Stem
        x = self.stem(x)  # [B, 96, H/4, W/4]
        
        # Stage 0 (salva per skip connection)
        x = self.stages[0](x, cvd_profile)
        skip_feat = x  # [B, 96, H/4, W/4]
        
        # Stage successivi
        for stage in self.stages[1:]:
            x = stage(x, cvd_profile)
        
        return x, skip_feat
    
    def freeze_encoder_except_adain(self):
        """
        Congela tutti i parametri ECCETTO quelli dei CVDAdaIN.
        Utile per fine-tuning solo della parte CVD.
        """
        for name, param in self.named_parameters():
            if 'style_projection' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[PLCFEncoderCVD] Encoder congelato eccetto CVDAdaIN. Trainable: {trainable:,}")
    
    def unfreeze_all(self):
        """Sblocca tutti i parametri."""
        for param in self.parameters():
            param.requires_grad = True


class PLCFEncoderCVDMinimal(PLCFEncoderCVD):
    """
    Versione minimale che si ferma a Stage 0.
    Output: [B, 96, 32, 32]
    """
    
    def __init__(self, cvd_dim: int = 3, pretrained: bool = True, freeze_stem: bool = True):
        super().__init__(
            cvd_dim=cvd_dim,
            pretrained=pretrained,
            freeze_stem=freeze_stem,
            stop_at_stage=0
        )


# =============================================================================
# TESTING
# =============================================================================

def test_encoder():
    """Test dell'encoder con CVDAdaIN."""
    print("=" * 60)
    print("TEST: PLCFEncoderCVD")
    print("=" * 60)
    
    # Crea encoder
    encoder = PLCFEncoderCVD(
        cvd_dim=3,  # Profilo 3D: [θ_norm, C_norm, S_norm]
        pretrained=True,
        freeze_stem=True,
        stop_at_stage=2
    )
    
    # Input
    x = torch.randn(2, 3, 128, 128)
    cvd = torch.rand(2, 3)  # Profilo 3D
    
    # Forward
    encoder.eval()
    with torch.no_grad():
        latent, skip_feat = encoder(x, cvd)
    
    print(f"\nInput shape: {x.shape}")
    print(f"CVD shape: {cvd.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Skip feat shape: {skip_feat.shape}")
    
    # Test freeze
    encoder.freeze_encoder_except_adain()
    
    # Verifica che CVDAdaIN sia trainabile
    adain_trainable = 0
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            adain_trainable += param.numel()
    print(f"\nParametri CVDAdaIN trainabili: {adain_trainable:,}")
    
    print("\n[OK] Test passed!")


if __name__ == "__main__":
    test_encoder()
