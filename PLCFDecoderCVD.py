"""
PLCFDecoderCVD — Decoder PLCF con CVDAdaIN al posto di ChannelLayerNorm.

Ogni normalizzazione è sostituita da CVDAdaIN (InstanceNorm + modulazione
affine condizionata dal profilo CVD 3D [θ, C, S]).

Architettura (per input 256×256, bottleneck [B, 384, 16, 16]):

    Input latent [B, 384, 16, 16]
        ↓
    Bottleneck: Conv3×3 + GELU + CVDAdaIN                          (1 CVDAdaIN)
        ↓ [B, 384, 16, 16]
    Block 3: Upsample + Conv → CVDAdaIN → GELU → Conv → CVDAdaIN → GELU
        ↓ [B, 192, 32, 32]                                        (2 CVDAdaIN)
    Block 2: Upsample + Conv → CVDAdaIN → GELU → Conv → CVDAdaIN → GELU
        ↓ [B, 96, 64, 64]                                         (2 CVDAdaIN)
    Block 1: Upsample + Conv → CVDAdaIN → GELU → Conv → CVDAdaIN → GELU
        ↓ [B, 64, 128, 128]                                       (2 CVDAdaIN)
    Block 0: Upsample + Conv → CVDAdaIN → GELU
        ↓ [B, 64, 256, 256]                                       (1 CVDAdaIN)
    Head: Conv3×3(64→32) → GELU → CVDAdaIN(32) → Conv1×1(32→out) → Tanh
        ↓ [B, out, 256, 256]                                      (1 CVDAdaIN)

    Totale CVDAdaIN nel decoder: **9**  (1 bottleneck + 6 blocks + 1 block0 + 1 head)

Output:
    - y_preserving=True  → out=2 (ΔCb, ΔCr in [-1, 1])
    - y_preserving=False → out=3 (ΔRGB in [-1, 1])

Il tipo di normalizzazione è InstanceNorm (channel-first), distinto
dall'encoder che usa LayerNorm (channel-last).

Dipendenze:
    cvd_adain_modules.CVDAdaINChannelFirst.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from cvd_adain_modules import CVDAdaINChannelFirst


class CVDDecoderBlock(nn.Module):
    """
    Blocco decoder con CVDAdaIN.
    
    Struttura:
        Upsample → Conv → CVDAdaIN(cvd) → GELU → Conv → CVDAdaIN(cvd) → GELU
    
    Args:
        in_channels: Canali input
        out_channels: Canali output
        cvd_dim: Dimensione vettore CVD
        upsample_mode: Modalità upsampling ('nearest' o 'bilinear')
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cvd_dim: int = 3,
                 upsample_mode: str = 'nearest'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Upsample + prima Conv
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = CVDAdaINChannelFirst(out_channels, cvd_dim=cvd_dim)
        self.act1 = nn.GELU()
        
        # Seconda Conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = CVDAdaINChannelFirst(out_channels, cvd_dim=cvd_dim)
        self.act2 = nn.GELU()
    
    def forward(self, 
                x: torch.Tensor,           # [B, C_in, H, W]
                cvd_profile: torch.Tensor  # [B, 3]
               ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature map [B, C_in, H, W]
            cvd_profile: Profilo CVD normalizzato [B, 3]
        
        Returns:
            Feature map [B, C_out, 2H, 2W]
        """
        # Upsample + Conv1 + Norm1 + Act1
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm1(x, cvd_profile)
        x = self.act1(x)
        
        # Conv2 + Norm2 + Act2
        x = self.conv2(x)
        x = self.norm2(x, cvd_profile)
        x = self.act2(x)
        
        return x


class CVDDecoderBlockSimple(nn.Module):
    """
    Blocco decoder semplificato con singolo CVDAdaIN.
    
    Struttura:
        Upsample → Conv → CVDAdaIN(cvd) → GELU
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cvd_dim: int = 3,
                 upsample_mode: str = 'nearest'):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = CVDAdaINChannelFirst(out_channels, cvd_dim=cvd_dim)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor, cvd_profile: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x, cvd_profile)
        x = self.act(x)
        return x


class PLCFDecoderCVD(nn.Module):
    """Decoder PLCF con 9 punti CVDAdaIN (InstanceNorm channel-first).

    Upsampling progressivo nearest-neighbor da bottleneck a target_resolution.
    La head finale usa tanh → output in [-1, 1].

    Args:
        in_channels: Canali dal bottleneck (default 384 per Stage 2).
        cvd_dim: Dimensione vettore CVD (default 3).
        output_channels: Canali output (2 per ΔCbCr Y'-preserving, 3 per ΔRGB).
        output_mode: ``"rgb"`` (unico supportato).
        target_resolution: Risoluzione spaziale di output (default 256).
    """
    
    def __init__(self,
                 in_channels: int = 384,
                 cvd_dim: int = 3,
                 output_channels: int = 3,
                 output_mode: Literal["rgb"] = "rgb",
                 target_resolution: int = 256):
        super().__init__()
        
        self.in_channels = in_channels
        self.cvd_dim = cvd_dim
        self.output_mode = output_mode
        self.output_channels = output_channels
        self.target_resolution = target_resolution
        
        # ═══════════════════════════════════════════════════════════════
        # BOTTLENECK: Conv opzionale per processare latent
        # ═══════════════════════════════════════════════════════════════
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.bottleneck_norm = CVDAdaINChannelFirst(in_channels, cvd_dim=cvd_dim)
        
        # ═══════════════════════════════════════════════════════════════
        # DECODER BLOCKS
        # Block 3: 384→192, 8→16
        # Block 2: 192→96, 16→32
        # Block 1: 96→64, 32→64
        # Block 0: 64→64, 64→128
        # ═══════════════════════════════════════════════════════════════
        
        self.block3 = CVDDecoderBlock(384, 192, cvd_dim=cvd_dim)
        self.block2 = CVDDecoderBlock(192, 96, cvd_dim=cvd_dim)
        self.block1 = CVDDecoderBlock(96, 64, cvd_dim=cvd_dim)
        self.block0 = CVDDecoderBlockSimple(64, 64, cvd_dim=cvd_dim)
        
        # ═══════════════════════════════════════════════════════════════
        # HEAD (scomposto per permettere CVDAdaIN)
        # ═══════════════════════════════════════════════════════════════
        self.head_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.head_act1 = nn.GELU()
        self.head_norm = CVDAdaINChannelFirst(32, cvd_dim=cvd_dim)  # CVD conditioning finale
        self.head_conv2 = nn.Conv2d(32, output_channels, kernel_size=1)
        self.head_act2 = nn.Tanh()  # Output in [-1, 1]
        
        # Scaling per LAB output
        self.ab_scale = 127.5  # [-1, 1] → [-127.5, 127.5] ≈ range a*b*
        
        # Log info
        print(f"[PLCFDecoderCVD] Inizializzato:")
        print(f"  - Input channels: {in_channels}")
        print(f"  - Output mode: {output_mode}")
        print(f"  - Output channels: {output_channels}")
        print(f"  - Target resolution: {target_resolution}")
        print(f"  - CVD dimension: {cvd_dim}")
    
    def forward(self,
                z: torch.Tensor,           # [B, 384, H, W]
                cvd_profile: torch.Tensor, # [B, 3]
                skip_feat: Optional[torch.Tensor] = None  # non usato
               ) -> torch.Tensor:
        """Forward pass del decoder.

        Args:
            z: Latent features [B, in_channels, H, W] dal bottleneck encoder.
            cvd_profile: Profilo CVD normalizzato [B, 3] (θ, C, S).
            skip_feat: Opzionale, non usato in questa versione.

        Returns:
            Delta output [B, output_channels, target_resolution, target_resolution]
            in [-1, 1] (tanh).
        """
        # Bottleneck
        x = self.bottleneck(z)
        x = self.bottleneck_norm(x, cvd_profile)
        
        # Decoder blocks
        x = self.block3(x, cvd_profile)  # [B, 192, 16, 16]
        x = self.block2(x, cvd_profile)  # [B, 96, 32, 32]
        x = self.block1(x, cvd_profile)  # [B, 64, 64, 64]
        x = self.block0(x, cvd_profile)  # [B, 64, 128, 128]
        
        # Resize se necessario
        if x.shape[2] != self.target_resolution:
            x = F.interpolate(x, size=(self.target_resolution, self.target_resolution), 
                            mode='nearest')
        
        # Head con CVDAdaIN finale
        x = self.head_conv1(x)              # [B, 32, H, W]
        x = self.head_act1(x)               # GELU
        x = self.head_norm(x, cvd_profile)  # CVD conditioning finale
        x = self.head_conv2(x)              # [B, output_channels, H, W]
        x = self.head_act2(x)               # Tanh → [-1, 1]
        
        return x
    
    def get_output_info(self) -> dict:
        """Ritorna informazioni sull'output."""
        return {
            "mode": self.output_mode,
            "channels": self.output_channels,
            "resolution": self.target_resolution,
            "range": "[-1, 1]"
        }


class PLCFDecoderCVDWithSkip(PLCFDecoderCVD):
    """
    Decoder con CVD-AdaIN e supporto per skip connection.
    
    Aggiunge la possibilità di usare skip features dall'encoder.
    """
    
    def __init__(self,
                 in_channels: int = 384,
                 skip_channels: int = 96,
                 cvd_dim: int = 3,
                 output_channels: int = 3,
                 output_mode: Literal["rgb"] = "rgb",
                 target_resolution: int = 256):
        super().__init__(
            in_channels=in_channels,
            cvd_dim=cvd_dim,
            output_channels=output_channels,
            output_mode=output_mode,
            target_resolution=target_resolution
        )
        
        self.skip_channels = skip_channels
        
        # Fusione skip connection (dopo Block 2, a risoluzione 32×32)
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(96 + skip_channels, 96, kernel_size=1),
        )
        self.skip_fusion_norm = CVDAdaINChannelFirst(96, cvd_dim=cvd_dim)
    
    def forward(self,
                z: torch.Tensor,           # [B, 384, 8, 8]
                cvd_profile: torch.Tensor, # [B, 3]
                skip_feat: Optional[torch.Tensor] = None  # [B, 96, 32, 32]
               ) -> torch.Tensor:
        """
        Forward pass con skip connection.
        """
        # Bottleneck
        x = self.bottleneck(z)
        x = self.bottleneck_norm(x, cvd_profile)
        
        # Block 3
        x = self.block3(x, cvd_profile)  # [B, 192, 16, 16]
        
        # Block 2
        x = self.block2(x, cvd_profile)  # [B, 96, 32, 32]
        
        # Skip connection (se presente)
        if skip_feat is not None:
            # Assicura che le dimensioni siano compatibili
            if skip_feat.shape[2:] != x.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatena e fonde
            x = torch.cat([x, skip_feat], dim=1)
            x = self.skip_fusion(x)
            x = self.skip_fusion_norm(x, cvd_profile)
        
        # Block 1 e 0
        x = self.block1(x, cvd_profile)  # [B, 64, 64, 64]
        x = self.block0(x, cvd_profile)  # [B, 64, 128, 128]
        
        # Resize se necessario
        if x.shape[2] != self.target_resolution:
            x = F.interpolate(x, size=(self.target_resolution, self.target_resolution), 
                            mode='nearest')
        
        # Head con CVDAdaIN finale
        x = self.head_conv1(x)              # [B, 32, H, W]
        x = self.head_act1(x)               # GELU
        x = self.head_norm(x, cvd_profile)  # CVD conditioning finale
        x = self.head_conv2(x)              # [B, output_channels, H, W]
        x = self.head_act2(x)               # Tanh → [-1, 1]
        
        return x


# =============================================================================
# TESTING
# =============================================================================

def test_decoder():
    """Test del decoder con CVDAdaIN."""
    print("=" * 60)
    print("TEST: PLCFDecoderCVD (RGB-only)")
    print("=" * 60)
    
    # Crea decoder
    decoder = PLCFDecoderCVD(
        in_channels=384,
        cvd_dim=3,  # Profilo 3D: [θ_norm, C_norm, S_norm]
        output_channels=3,
        output_mode="rgb"
    )
    
    # Input
    z = torch.randn(2, 384, 8, 8)
    cvd = torch.rand(2, 3)  # Profilo 3D
    
    # Forward
    decoder.eval()
    with torch.no_grad():
        out = decoder(z, cvd)
    
    print(f"\nInput shape: {z.shape}")
    print(f"CVD shape: {cvd.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.2f}, {out.max():.2f}]")
    
    # Conta parametri
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n[OK] Test passed!")


def test_decoder_with_skip():
    """Test del decoder con skip connection."""
    print("=" * 60)
    print("TEST: PLCFDecoderCVDWithSkip")
    print("=" * 60)
    
    decoder = PLCFDecoderCVDWithSkip(
        in_channels=384,
        skip_channels=96,
        cvd_dim=3  # Profilo 3D: [θ_norm, C_norm, S_norm]
    )
    
    z = torch.randn(2, 384, 8, 8)
    skip = torch.randn(2, 96, 32, 32)
    cvd = torch.rand(2, 3)  # Profilo 3D
    
    decoder.eval()
    with torch.no_grad():
        out = decoder(z, cvd, skip)
    
    print(f"\nInput z shape: {z.shape}")
    print(f"Skip feat shape: {skip.shape}")
    print(f"CVD shape: {cvd.shape}")
    print(f"Output shape: {out.shape}")
    
    print("\n[OK] Test passed!")


if __name__ == "__main__":
    test_decoder()
    print()
    test_decoder_with_skip()
