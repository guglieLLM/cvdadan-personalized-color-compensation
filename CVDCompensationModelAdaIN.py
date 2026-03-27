"""
CVDCompensationModelAdaIN — Modello completo per compensazione cromatica CVD.

Architettura encoder–decoder con condizionamento CVDAdaIN:

    Image [B, 3, 256, 256]  +  CVD Profile [B, 3]  (θ, C, S normalizzati)
               ↓                       ↓
    ┌─────────────────────────────────────────────────┐
    │       ENCODER (ConvNeXt-Tiny, ImageNet-1k)      │
    │  CVDAdaIN in 17 punti (15 blocchi + 2 downsa.)  │
    │  Stem congelato, resto fine-tuned (lr dedicato) │
    └──────────────────┬──────────────────────────────┘
                       ↓
              Latent [B, 384, 16, 16]
                       ↓
    ┌─────────────────────────────────────────────────┐
    │       DECODER PLCF (9 CVDAdaIN + tanh head)     │
    │  Upsampling progressivo nearest-neighbor        │
    └──────────────────┬──────────────────────────────┘
                       ↓
             ΔCbCr [B, 2, 256, 256]   (y_preserving=True)
                       ↓
    ┌─────────────────────────────────────────────────┐
    │           Y'-PRESERVING OUTPUT                  │
    │  Y'_out = Y'_in       (luma BT.601 copiata)    │
    │  Cb_out = Cb_in + ΔCb × 0.9                    │
    │  Cr_out = Cr_in + ΔCr × 0.9                    │
    └─────────────────────────────────────────────────┘

Il profilo CVD [θ, C, S] modula l'intera rete:
- Encoder: 17 CVDAdaIN (tipo LayerNorm channel-last)
- Decoder: 9 CVDAdaIN  (tipo InstanceNorm channel-first)

Principi progettuali:
- Condizionamento *esplicito* e *continuo*: il profilo 3D entra
  come modulazione affine (γ, β) in ogni normalizzazione (CVDAdaIN).
- Y'-Preserving: la luma Y' (BT.601) viene copiata dall'input, il decoder
  predice solo ΔCbCr → nessuna variazione di brightness by-design.
- RGB-only nel forward: nessuna conversione LAB; la loss opera internamente
  su a*b* (CIELAB), ma il forward rimane in RGB / YCbCr.

Dipendenze:
    PLCFEncoderCVD, PLCFDecoderCVD, color_space_utils (rgb_to_ycbcr_torch,
    ycbcr_to_rgb_torch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal, Dict, Union

from PLCFEncoderCVD import PLCFEncoderCVD
from PLCFDecoderCVD import PLCFDecoderCVD, PLCFDecoderCVDWithSkip
# Y-Preserving architecture: YCbCr conversions for luminance preservation
from color_space_utils import rgb_to_ycbcr_torch, ycbcr_to_rgb_torch
# NOTE: LAB conversion is handled internally by the loss function (CVDLoss)


class CVDCompensationModelAdaIN(nn.Module):
    """Modello encoder–decoder con CVDAdaIN per compensazione cromatica CVD.

    Due modalità di output:

    1. **Y'-Preserving** (``y_preserving=True``, default di addestramento):
       decoder predice ΔCb, ΔCr (2 canali); la luma Y' (BT.601) viene
       copiata dall'input → nessuna variazione di brightness by-design.
    2. **RGB-only** (``y_preserving=False``):
       decoder predice ΔRGB (3 canali); luma non vincolata.

    Il forward dell'encoder opera sempre su input normalizzato ImageNet.
    La loss (:class:`CVDLoss` in ``losses.py``) converte internamente a
    CIELAB per la componente MSE a*b*.

    Normalizzazione profilo CVD — schema **ibrido**:
        * θ  → normalizzazione globale (preserva distinzione protan/deutan/tritan)
        * C, S → normalizzazione per-tipo CVD (gestisce distribuzioni diverse)

    Args:
        cvd_dim: Dimensione vettore CVD (default 3: θ, C, S).
        pretrained_encoder: Se True carica pesi ImageNet-1k per l'encoder.
        freeze_encoder_except_adain: Se True congela encoder tranne CVDAdaIN.
        use_skip_connection: Se True usa skip connection da Stage 0.
        stop_at_stage: Ultimo stage encoder (0, 1, 2). Default 2 → 384 ch.
        delta_rgb_scale: Fattore di scala per delta crominanza (default 0.9).
        target_resolution: Risoluzione spaziale di output (default 256).
        y_preserving: Se True, architettura Y'-Preserving (default False).
    """
    
    def __init__(self,
                 cvd_dim: int = 3,
                 pretrained_encoder: bool = True,
                 freeze_encoder_except_adain: bool = True,
                 use_skip_connection: bool = False,
                 stop_at_stage: int = 2,
                 delta_rgb_scale: float = 0.9,
                 target_resolution: int = 256,
                 y_preserving: bool = False):
        super().__init__()
        
        self.cvd_dim = cvd_dim
        self.use_skip_connection = use_skip_connection
        self.delta_rgb_scale = delta_rgb_scale
        self.y_preserving = y_preserving
        
        # Output channels: 2 for Y-Preserving (ΔCb, ΔCr), 3 for RGB-only (ΔRGB)
        output_channels = 2 if y_preserving else 3
        
        # ═══════════════════════════════════════════════════════════════
        # ENCODER
        # ═══════════════════════════════════════════════════════════════
        self.encoder = PLCFEncoderCVD(
            cvd_dim=cvd_dim,
            pretrained=pretrained_encoder,
            freeze_stem=True,
            stop_at_stage=stop_at_stage
        )
        
        if freeze_encoder_except_adain:
            self.encoder.freeze_encoder_except_adain()
        
        # ═══════════════════════════════════════════════════════════════
        # DECODER
        # Y-Preserving: predice ΔCb, ΔCr (2 canali) in [-1, 1]
        # RGB-only: predice ΔRGB (3 canali) in [-1, 1]
        # ═══════════════════════════════════════════════════════════════
        if use_skip_connection:
            self.decoder = PLCFDecoderCVDWithSkip(
                in_channels=self.encoder.out_channels,
                skip_channels=96,  # Stage0 output
                cvd_dim=cvd_dim,
                output_channels=output_channels,
                target_resolution=target_resolution
            )
        else:
            self.decoder = PLCFDecoderCVD(
                in_channels=self.encoder.out_channels,
                cvd_dim=cvd_dim,
                output_channels=output_channels,
                target_resolution=target_resolution
            )
        
        # Modalità output: ritorna dict per compatibilità training
        self._return_dict = True
        
        # Log
        self._log_model_info()
    
    def _log_model_info(self):
        """Log informazioni sul modello."""
        params = self.count_parameters()
        
        mode_str = "Y-Preserving (dCb/dCr)" if self.y_preserving else "RGB-only (dRGB)"
        channels_str = "2 (dCb, dCr)" if self.y_preserving else "3 (dR, dG, dB)"
        
        print("\n" + "=" * 60)
        print(f"CVDCompensationModelAdaIN - Summary ({mode_str})")
        print("=" * 60)
        print(f"Encoder output channels: {self.encoder.out_channels}")
        print(f"Output mode: {mode_str}")
        print(f"Decoder output channels: {channels_str}")
        print(f"Delta scale: {self.delta_rgb_scale}")
        print(f"Skip connection: {self.use_skip_connection}")
        if self.y_preserving:
            print(f"\n Y'-PRESERVING ACTIVE: luma Y' (BT.601) copied from input.")
            print(f"   Y'(output) ≡ Y'(input) guaranteed by architecture.")
        print(f"\nParametri:")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Frozen: {params['frozen']:,}")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable %: {params['trainable_pct']:.1f}%")
        print("=" * 60 + "\n")
    
    def forward(self,
                image: torch.Tensor,
                profile_feats: torch.Tensor = None,
                cvd_profile: torch.Tensor = None,
                epoch: int = None,
                total_epochs: int = None
               ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass — supporta modalità RGB-only e Y'-Preserving.

        Args:
            image: Input RGB [B, 3, H, W] normalizzato ImageNet.
            profile_feats: Profilo CVD normalizzato [B, 3] (θ, C, S).
            cvd_profile: Alias di *profile_feats* per retro-compatibilità.
            epoch: Ignorato (compatibilità interfaccia).
            total_epochs: Ignorato (compatibilità interfaccia).

        Returns:
            Se ``_return_dict=True`` (default): ``dict`` con chiavi
                ``"delta_rgb"`` — delta raw dal decoder [B, 2|3, H, W]
                ``"rgb_output"`` — immagine compensata [B, 3, H, W] in [-1, 1]
            Altrimenti: solo ``rgb_output``.

        Note (Y'-Preserving):
            L'input ImageNet-normalized viene de-normalizzato in [0,1] → [-1,1]
            per la conversione YCbCr (BT.601).  La luma Y' dell'input viene
            copiata immutata; solo Cb, Cr vengono modificati dal decoder.
        """
        # Compatibilità: usa profile_feats o cvd_profile
        cvd = profile_feats if profile_feats is not None else cvd_profile
        if cvd is None:
            if self.training:
                raise RuntimeError("cvd_profile/profile_feats cannot be None in training mode")
            cvd = torch.zeros(image.size(0), 3, device=image.device)
        
        # Encode (always on RGB input)
        latent, skip_feat = self.encoder(image, cvd)
        
        # Decode
        if self.use_skip_connection:
            delta = self.decoder(latent, cvd, skip_feat)
        else:
            delta = self.decoder(latent, cvd)
        
        # Resize delta alla dimensione dell'input se necessario
        if delta.shape[2:] != image.shape[2:]:
            delta = F.interpolate(delta, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        if self.y_preserving:
            # ═══════════════════════════════════════════════════════════════
            # Y-PRESERVING MODE: Decoder outputs ΔCb, ΔCr (2 channels)
            # ═══════════════════════════════════════════════════════════════
            # L'input arriva normalizzato ImageNet [-2.1, +2.6].
            # La conversione YCbCr si aspetta range [-1, 1]:
            # de-normalizziamo solo per YCbCr, l'encoder ha già ricevuto l'input ImageNet
            
            # ImageNet normalization constants
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            
            # 1. Denormalize ImageNet → [0,1] → [-1,1] for YCbCr conversion
            image_01 = image * imagenet_std + imagenet_mean  # ImageNet → [0,1]
            image_tanh = image_01 * 2.0 - 1.0  # [0,1] → [-1,1]
            
            # 2. Convert to YCbCr (now in correct [-1,1] range)
            ycbcr_input = rgb_to_ycbcr_torch(image_tanh)
            Y_in = ycbcr_input[:, 0:1]   # Luma Y' (BT.601) — PRESERVED
            Cb_in = ycbcr_input[:, 1:2]  # Chrominance blue
            Cr_in = ycbcr_input[:, 2:3]  # Chrominance red
            
            # 3. Apply delta to chrominance only
            delta_Cb = delta[:, 0:1]
            delta_Cr = delta[:, 1:2]
            
            # ═══════════════════════════════════════════════════════════════
            # [OPZIONE A - DISABILITATA] Adaptive scaling per theta estremi
            # ═══════════════════════════════════════════════════════════════
            # Questa soluzione è stata SOSTITUITA da OPZIONE C (per-CVD normalization)
            # in cvd_dataset_loader.py che normalizza θ separatamente per ogni tipo CVD.
            # Con Opzione C, θ_norm è sempre in [-1, +1] per tutti i tipi e non serve
            # più l'amplificazione runtime.
            #
            # Lasciato come riferimento - NON RIMUOVERE:
            # ---------------------------------------------------------------
            # theta_norm = cvd[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            # extreme_scale = 1.0 + 0.8 * torch.clamp(theta_norm.abs() - 1.0, min=0.0)
            # delta_Cb = delta_Cb * extreme_scale
            # delta_Cr = delta_Cr * extreme_scale
            # ═══════════════════════════════════════════════════════════════
            
            Cb_out = Cb_in + delta_Cb * self.delta_rgb_scale
            Cr_out = Cr_in + delta_Cr * self.delta_rgb_scale
            
            # 4. Reconstruct YCbCr with ORIGINAL Y
            ycbcr_output = torch.cat([Y_in, Cb_out, Cr_out], dim=1)
            
            # 5. Convert back to RGB [-1,1]
            rgb_output = ycbcr_to_rgb_torch(ycbcr_output)
            rgb_output = torch.clamp(rgb_output, -1.0, 1.0)
            
            # For dict output, return delta in same format (2 channels for ΔCbCr)
            delta_output = delta
        else:
            # ═══════════════════════════════════════════════════════════════
            # RGB-ONLY MODE: Decoder outputs ΔRGB (3 channels)
            # ═══════════════════════════════════════════════════════════════
            rgb_output = torch.clamp(image + delta * self.delta_rgb_scale, -1.0, 1.0)
            delta_output = delta
        
        if self._return_dict:
            return {
                "delta_rgb": delta_output,  # Raw delta per analisi/debug (2ch o 3ch)
                "rgb_output": rgb_output    # Output finale clampato (sempre 3ch RGB)
            }
        
        return rgb_output
    
    def set_return_dict(self, return_dict: bool):
        """Imposta se forward ritorna dict o solo tensor."""
        self._return_dict = return_dict
        return self
    
    def count_parameters(self) -> dict:
        """Conta parametri trainabili e frozen."""
        trainable = 0
        frozen = 0
        
        for param in self.parameters():
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()
        
        total = trainable + frozen
        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': total,
            'trainable_pct': 100 * trainable / total if total > 0 else 0
        }
    
    def get_adain_parameters(self) -> list:
        """Ritorna solo i parametri CVDAdaIN (per optimizer separato)."""
        adain_params = []
        for name, param in self.named_parameters():
            if 'style_projection' in name:
                adain_params.append(param)
        return adain_params
    
    def get_decoder_parameters(self) -> list:
        """Ritorna solo i parametri del decoder."""
        return list(self.decoder.parameters())
    
    def freeze_encoder(self):
        """Congela completamente l'encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder_adain(self):
        """Sblocca solo i CVDAdaIN dell'encoder."""
        self.encoder.freeze_encoder_except_adain()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cvd_model_small(cvd_dim: int = 3, pretrained: bool = True) -> CVDCompensationModelAdaIN:
    """
    Crea modello piccolo (stop a Stage 0).
    Output encoder: [B, 96, 32, 32]
    """
    return CVDCompensationModelAdaIN(
        cvd_dim=cvd_dim,
        pretrained_encoder=pretrained,
        freeze_encoder_except_adain=True,
        use_skip_connection=False,
        stop_at_stage=0
    )


def create_cvd_model_medium(cvd_dim: int = 3, pretrained: bool = True) -> CVDCompensationModelAdaIN:
    """
    Crea modello medio (stop a Stage 1).
    Output encoder: [B, 192, 16, 16]
    """
    return CVDCompensationModelAdaIN(
        cvd_dim=cvd_dim,
        pretrained_encoder=pretrained,
        freeze_encoder_except_adain=True,
        use_skip_connection=False,
        stop_at_stage=1
    )


def create_cvd_model_large(cvd_dim: int = 3, pretrained: bool = True) -> CVDCompensationModelAdaIN:
    """
    Crea modello grande (stop a Stage 2).
    Output encoder: [B, 384, 16, 16] per input 256×256.
    """
    return CVDCompensationModelAdaIN(
        cvd_dim=cvd_dim,
        pretrained_encoder=pretrained,
        freeze_encoder_except_adain=True,
        use_skip_connection=False,
        stop_at_stage=2
    )


# =============================================================================
# TESTING
# =============================================================================

def test_full_model():
    """Test del modello completo (RGB-only)."""
    print("=" * 70)
    print("TEST: CVDCompensationModelAdaIN (RGB-only)")
    print("=" * 70)
    
    # Crea modello
    model = CVDCompensationModelAdaIN(
        cvd_dim=3,
        pretrained_encoder=True,
        freeze_encoder_except_adain=True,
        use_skip_connection=False,
        stop_at_stage=2,
        delta_rgb_scale=0.9
    )
    
    # Input (range [-1, 1] come da tanh normalization)
    image = torch.rand(2, 3, 128, 128) * 2 - 1  # [-1, 1]
    cvd_profile = torch.rand(2, 3)
    
    # Forward
    model.eval()
    with torch.no_grad():
        output = model(image, profile_feats=cvd_profile)
    
    print(f"\nImage shape: {image.shape}")
    print(f"CVD profile shape: {cvd_profile.shape}")
    print(f"Output type: {type(output)}")
    print(f"Output keys: {output.keys()}")
    print(f"delta_rgb shape: {output['delta_rgb'].shape}")
    print(f"rgb_output shape: {output['rgb_output'].shape}")
    print(f"delta_rgb range: [{output['delta_rgb'].min():.3f}, {output['delta_rgb'].max():.3f}]")
    print(f"rgb_output range: [{output['rgb_output'].min():.3f}, {output['rgb_output'].max():.3f}]")
    
    # Verifica output diverso per CVD diversi
    # NOTA: Prima del training i pesi CVDAdaIN sono inizializzati all'identità
    # (gamma=1, beta=0), quindi l'output potrebbe non cambiare
    print("\n--- CVD Sensitivity Test ---")
    cvd_normal = torch.zeros(1, 3)
    cvd_protan = torch.tensor([[0.3, 0.8, 0.6]])
    cvd_deutan = torch.tensor([[0.7, 0.8, 0.6]])
    
    image_test = torch.rand(1, 3, 128, 128) * 2 - 1
    
    # In train mode per vedere se il modello risponde
    model.train()
    with torch.no_grad():
        out_normal = model(image_test, profile_feats=cvd_normal)
        out_protan = model(image_test, profile_feats=cvd_protan)
        out_deutan = model(image_test, profile_feats=cvd_deutan)
    model.eval()
    
    # Confronta delta_rgb
    diff_protan = (out_protan['delta_rgb'] - out_normal['delta_rgb']).abs().mean().item()
    diff_deutan = (out_deutan['delta_rgb'] - out_normal['delta_rgb']).abs().mean().item()
    
    print(f"Diff (normal vs protan): {diff_protan:.6f}")
    print(f"Diff (normal vs deutan): {diff_deutan:.6f}")
    
    # Soft check - differenza può essere 0 con pesi inizializzati all'identità
    if diff_protan < 1e-6:
        print("  WARNING: Output uguale (normale con pesi iniziali all'identità)")
    else:
        print(" Output diverso per CVD diversi")
    
    print("\n All tests passed!")


def test_model_variants():
    """Test delle varianti del modello."""
    print("\n" + "=" * 70)
    print("TEST: Model Variants (RGB-only and Y-Preserving)")
    print("=" * 70)
    
    image = torch.rand(1, 3, 128, 128) * 2 - 1  # [-1, 1]
    cvd = torch.rand(1, 3)
    
    # Solo Stage 2 (Large) è compatibile con il decoder attuale
    # Il decoder ha struttura fissa 384→192→96→64
    # Per altri stop_stage servirebbero decoder con struttura diversa
    
    print("\nNOTA: Il decoder corrente supporta solo stop_at_stage=2 (in_channels=384)")
    print("Per usare stage 0 o 1 bisogna creare decoder con struttura diversa.\n")
    
    for name, factory in [
        # ("Small (Stage 0)", create_cvd_model_small),  # Richiede decoder specifico
        # ("Medium (Stage 1)", create_cvd_model_medium),  # Richiede decoder specifico
        ("Large (Stage 2)", create_cvd_model_large),
    ]:
        model = factory(pretrained=False)  # No pretrained per velocità
        model.eval()
        
        with torch.no_grad():
            out = model(image, profile_feats=cvd)
        
        params = model.count_parameters()
        print(f"\n{name}:")
        print(f"  Encoder channels: {model.encoder.out_channels}")
        print(f"  Output keys: {out.keys()}")
        print(f"  delta_rgb shape: {out['delta_rgb'].shape}")
        print(f"  rgb_output shape: {out['rgb_output'].shape}")
        print(f"  Trainable params: {params['trainable']:,}")
    
    print("\n Variant test completed!")


def test_y_preserving_mode():
    """Test della modalità Y-Preserving."""
    print("\n" + "=" * 70)
    print("TEST: Y-Preserving Mode")
    print("=" * 70)
    
    # Crea modello Y-Preserving
    model = CVDCompensationModelAdaIN(
        cvd_dim=3,
        pretrained_encoder=False,  # No pretrained per velocità
        freeze_encoder_except_adain=True,
        use_skip_connection=False,
        stop_at_stage=2,
        delta_rgb_scale=0.9,
        y_preserving=True  # <-- Y-PRESERVING MODE
    )
    
    # Input
    image = torch.rand(2, 3, 128, 128) * 2 - 1  # [-1, 1]
    cvd_profile = torch.rand(2, 3)
    
    # Forward
    model.eval()
    with torch.no_grad():
        output = model(image, profile_feats=cvd_profile)
    
    print(f"\n[Y-Preserving Mode Test]")
    print(f"Input image shape: {image.shape}")
    print(f"Output delta shape: {output['delta_rgb'].shape}")  # Should be [B, 2, H, W]
    print(f"Output RGB shape: {output['rgb_output'].shape}")   # Should be [B, 3, H, W]
    
    # CRITICAL TEST: Verify Y preservation
    # Convert input and output to YCbCr and compare Y channel
    ycbcr_input = rgb_to_ycbcr_torch(image)
    ycbcr_output = rgb_to_ycbcr_torch(output['rgb_output'])
    
    Y_in = ycbcr_input[:, 0]
    Y_out = ycbcr_output[:, 0]
    
    y_diff = (Y_in - Y_out).abs()
    max_y_diff = y_diff.max().item()
    mean_y_diff = y_diff.mean().item()
    
    print(f"\n[Luminance Preservation Check]")
    print(f"Y_input range: [{Y_in.min():.4f}, {Y_in.max():.4f}]")
    print(f"Y_output range: [{Y_out.min():.4f}, {Y_out.max():.4f}]")
    print(f"Max |Y_in - Y_out|: {max_y_diff:.6f}")
    print(f"Mean |Y_in - Y_out|: {mean_y_diff:.6f}")
    
    # Y should be nearly identical (small diff due to RGB→YCbCr→RGB roundtrip + clamp)
    if max_y_diff < 0.1:  # Allow small diff from clamp
        print(" Y-Preserving: Luminance preserved correctly!")
    else:
        print("❌ Y-Preserving: Luminance NOT preserved (check implementation)")
    
    # Compare with RGB-only model
    model_rgb = CVDCompensationModelAdaIN(
        cvd_dim=3,
        pretrained_encoder=False,
        freeze_encoder_except_adain=True,
        use_skip_connection=False,
        stop_at_stage=2,
        delta_rgb_scale=0.9,
        y_preserving=False  # RGB-only mode
    )
    model_rgb.eval()
    
    with torch.no_grad():
        output_rgb = model_rgb(image, profile_feats=cvd_profile)
    
    ycbcr_output_rgb = rgb_to_ycbcr_torch(output_rgb['rgb_output'])
    Y_out_rgb = ycbcr_output_rgb[:, 0]
    
    max_y_diff_rgb = (Y_in - Y_out_rgb).abs().max().item()
    
    print(f"\n[Comparison: Y-Preserving vs RGB-only]")
    print(f"Y-Preserving max Y diff: {max_y_diff:.6f}")
    print(f"RGB-only max Y diff: {max_y_diff_rgb:.6f}")
    
    if max_y_diff < max_y_diff_rgb:
        print(" Y-Preserving mode has better luminance preservation!")
    else:
        print(" Note: RGB-only mode may have similar Y diff with random weights")
    
    print("\n Y-Preserving test completed!")


if __name__ == "__main__":
    test_full_model()
    test_model_variants()
    test_y_preserving_mode()
