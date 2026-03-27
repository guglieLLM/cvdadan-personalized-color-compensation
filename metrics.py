"""
Metriche di valutazione per la pipeline di correzione cromatica personalizzata.

Questo modulo contiene implementazioni di metriche di valutazione utilizzate
durante la validazione dei modelli nelle diverse fasi della pipeline.
A differenza delle funzioni di loss in losses.py, queste metriche sono ottimizzate
per la valutazione e non per il training (no backpropagation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

class SSIMMetric(nn.Module):
    """Metrica SSIM calcolata su RGB (media dei 3 canali).
    
    Questa implementazione calcola SSIM su tutti e 3 i canali RGB
    per avere valori più comparabili con la letteratura.
    
    Args:
        window_size: Dimensione finestra SSIM (default: 11)
    """
    def __init__(self, window_size=11):
        super().__init__()
        
        # SSIM su RGB con range [0,1] (dopo conversione da [-1,1])
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0,  # RGB normalizzato [0,1]
            kernel_size=window_size,
            sigma=1.5,
            k1=0.01,
            k2=0.03,
            return_full_image=False,
            reduction='elementwise_mean'
        )
        
        # Parametri ImageNet per denormalizzazione (non più usati ma mantenuti per compatibilità)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.verbose = False


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcola SSIM metrica tra predizione e target su RGB.
        
        Args:
            pred: Tensor RGB [B, 3, H, W] con valori in [-1,1]
            target: Tensor RGB [B, 3, H, W] con valori in [-1,1]
            
        Returns:
            ssim: Tensor scalare con valore SSIM in [0,1]
            dove 1 = perfetta similarità, 0 = massima differenza
        """
        
        # CRITICAL FP16 PROTECTION
        original_dtype = pred.dtype
        needs_fp32_protection = original_dtype in [torch.float16, torch.bfloat16]
        
        if needs_fp32_protection:
            pred = pred.float()
            target = target.float()
            
        # Sposta metrica sul device corretto
        self.ssim_metric = self.ssim_metric.to(pred.device)
        
        # Converti da [-1,1] a [0,1] per SSIM standard
        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        
        # Clamp per sicurezza numerica
        pred_01 = torch.clamp(pred_01, 0.0, 1.0)
        target_01 = torch.clamp(target_01, 0.0, 1.0)
        
        if self.verbose:
            print(f"Range RGB pred: [{pred_01.min().item():.4f}, {pred_01.max().item():.4f}]")
            print(f"Range RGB target: [{target_01.min().item():.4f}, {target_01.max().item():.4f}]")

        # Calcola SSIM su RGB (media dei 3 canali)
        with torch.no_grad():
            ssim = self.ssim_metric(pred_01, target_01)
        
        if self.verbose:
            print(f"SSIM value: {ssim.item():.6f}")
        
        # Sanity check
        if ssim.item() < 0 or ssim.item() > 1:
            print(f"[WARNING] SSIM value fuori range [0,1]: {ssim.item():.6f}")
        
        return ssim

    def _normalize(self, x):
        """
        Normalizzazione ottimizzata per evitare NaN e preservare la stabilità numerica.
        """
        min_val = x.amin(dim=[1,2,3], keepdim=True)
        max_val = x.amax(dim=[1,2,3], keepdim=True)
        range_val = max_val - min_val
        
        # Evita divisione per zero: se range≈0 -> mappa a 0.5
        norm = torch.where(
            range_val < 1e-6,
            torch.full_like(x, 0.5),
            (x - min_val) / (range_val + 1e-6)
        )
        
        return norm

    def _rgb_to_L(self, rgb: torch.Tensor) -> torch.Tensor:
        """Conversione ottimizzata RGB -> L* (luminanza CIELAB).
        
        Implementazione ottimizzata che:
        1. Converte i valori RGB da range [-1,1] a [0,1] 
        2. Calcola direttamente Y senza passare per XYZ completo
        3. Usa matrice pre-calcolata come buffer
        4. Minimizza allocazioni di memoria
        
        Args:
            rgb: Tensor RGB [B, 3, H, W] con valori in range [-1,1] (da denormalize_imagenet_to_tanh_range)
            
        Returns:
            L: Tensor [B, 1, H, W] con valori in [0,100]
        """
        # Costanti
        threshold = 0.008856  # (6/29)^3
        
        # Converte da [-1,1] a [0,1] invece di denormalizzare ImageNet
        rgb_01 = (rgb + 1.0) / 2.0  # [-1,1] -> [0,1]
        rgb_01 = torch.clamp(rgb_01, 0.0, 1.0)  # Sicurezza numerica
        
        # 1. RGB -> RGB lineare (resto invariato - già ottimizzato)
        rgb_linear = torch.where(
            rgb_01 <= 0.04045,
            rgb_01 / 12.92,
            ((rgb_01 + 0.055) / 1.055).pow(2.4)
        )

        # 2. Calcolo diretto di Y (evita XYZ completo) - invariato
        # Reshape per efficienza
        B, _, H, W = rgb_linear.shape
        rgb_reshaped = rgb_linear.permute(0, 2, 3, 1)  # [B, H, W, 3]
        
        # Moltiplica per matrice Y pre-calcolata
        xyz_matrix = self.xyz_matrix.to(rgb.device, rgb.dtype)
        Y = torch.sum(rgb_reshaped * xyz_matrix, dim=-1, keepdim=True)  # [B, H, W, 1]
        
        # Riporta alla forma originale
        Y = Y.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        # 3. Calcolo L* ottimizzato - invariato
        # Nota: non necessitiamo della normalizzazione D65 completa
        # poiché usiamo solo Y e il suo white point è 1.0
        
        # Calcolo L* con clamp per stabilità numerica
        L = torch.where(
            Y > threshold,
            116.0 * torch.pow(Y.clamp(min=threshold), 1/3.0) - 16.0,
            903.3 * Y  # Coefficiente pre-calcolato per Y <= threshold
        )

        return L

    
    
    def set_verbose(self, verbose=True):
        """Attiva/disattiva i messaggi di debug."""
        self.verbose = verbose


class PSNRMetric(nn.Module):
    """Metrica PSNR (Peak Signal-to-Noise Ratio) per valutazione di qualità dell'immagine.
    
    Implementazione ottimizzata per la validazione dei modelli.
    
    Args:
        max_val: Valore massimo del segnale (default: 1.0 per RGB normalizzato)
        convert_to_grayscale: Se True, converte le immagini in grayscale prima del calcolo
    """
    def __init__(self, max_val=1.0, convert_to_grayscale=False):
        super().__init__()
        self.max_val = max_val
        self.convert_to_grayscale = convert_to_grayscale
        
        # Pesi RGB->grayscale (ITU-R BT.601)
        self.register_buffer('rgb_weights', 
                           torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        
        self.verbose = False
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcola PSNR tra predizione e target.
        
        Args:
            pred: Tensor [B, C, H, W] con valori in [0, max_val]
            target: Tensor [B, C, H, W] con valori in [0, max_val]
            
        Returns:
            psnr: Valore scalare in decibel (dB)
        """
        # CRITICAL FP16 PROTECTION: Store original dtype and convert if needed
        original_dtype = pred.dtype
        needs_fp32_protection = original_dtype in [torch.float16, torch.bfloat16]
        
        if needs_fp32_protection:
            # Convert to FP32 for stable PSNR calculations
            pred = pred.float()
            target = target.float()
            
        if self.convert_to_grayscale and pred.size(1) == 3:
            # Converti a grayscale - FIX: Usa le variabili convertite
            pred_gray = (pred * self.rgb_weights.to(pred.device)).sum(dim=1, keepdim=True)
            target_gray = (target * self.rgb_weights.to(target.device)).sum(dim=1, keepdim=True)
            mse = F.mse_loss(pred_gray, target_gray)
        else:
            mse = F.mse_loss(pred, target)
            
        # Evita problemi numerici con MSE molto piccoli
        if mse < 1e-10:
            if self.verbose:
                print("MSE near zero, capping PSNR")
            return torch.tensor(100.0, device=pred.device)
            
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        
        if self.verbose:
            print(f"PSNR: {psnr.item():.2f} dB")
            
        return psnr
    
    def set_verbose(self, verbose=True):
        """Attiva/disattiva i messaggi di debug."""
        self.verbose = verbose
