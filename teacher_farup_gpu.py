"""
Teacher Farup 2018/2020 + GDIP 2021: GPU-accelerated PyTorch implementation with FM100

CORE ALGORITHM (Farup / Yoshi-II):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Costruzione campo di gradienti target:
    G = ∇u₀ + (∇u₀ · e_d) e_c
    
Dove:
    e_l : asse di luminanza [1,1,1]/√3
    e_d : asse di confusione cromatica (da FM100 Vingrys–King-Smith)
    e_c : direzione compensazione (ortogonale a e_l ed e_d)
    
Ricostruzione via GDIP anisotropic solver (Farup 2021):
    - Minimizza difference structure tensor: S^Δ = (∇u - G)(∇u - G)ᵀ
    - Evoluzione diffusiva anisotropa batch su GPU

FM100/VINGRYS LAYER (MODELING):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- FM100 Hue Test → (θ, C_index, S_index)
- θ → confusion_vector 3D (usato come e_d)
-  NON fa parte del metodo Farup originale
- È uno strato di personalizzazione clinica che sostituisce PCA

MATHEMATICAL FIDELITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Exact kernel values from diff_filters() 
- Reflection padding matching scipy 'symm' boundary condition
- Structure tensor eigenvalue decomposition
- Anisotropic diffusion with linearized equation

PERFORMANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target: 67x speedup via batch processing (batch_size=32 on RTX 3090)

REFERENCES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Farup (2018): "Yoshi-II", JIST
- Farup (2020): "Individualised halo-free gradient-domain daltonisation"
- Farup (2021): "Gradient domain image processing" (eq. 40)
- Vingrys & King-Smith (1988): FM-100 Hue Test
- Machado et al. (2009): CVD simulation

Copyright (C) 2025 - GPU acceleration for thesis research
Original algorithm: Copyright (C) 2021 Ivar Farup (GPL v3)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import warnings

# Import for geometric theta → confusion vector conversion (Vingrys 1988 FM100)
from color_space_utils import theta_to_confusion_vector_gpu


class DiffFilters:
    """
    Finite Difference Method (FDM) convolution filters for gradient computation.
    
    Provides exact kernel values matching Farup's gradient.py diff_filters() function.
    All kernels are 3x3 and support 7 different methods.
    """
    
    # Exact kernel definitions from Farup's gradient.py (lines 24-98)
    KERNELS = {
        'FB': {
            'F_x': torch.tensor([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]),
            'F_y': torch.tensor([[0., 1., 0.], [0., -1., 0.], [0., 0., 0.]]),
            'B_x': torch.tensor([[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]]),
            'B_y': torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., -1., 0.]]),
        },
        'cent': {
            'F_x': 0.5 * torch.tensor([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]),
            'F_y': 0.5 * torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., -1., 0.]]),
            'B_x': 0.5 * torch.tensor([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]),
            'B_y': 0.5 * torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., -1., 0.]]),
        },
        'Sobel': {
            'F_x': 0.125 * torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]),
            'F_y': 0.125 * torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]),
            'B_x': 0.125 * torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]),
            'B_y': 0.125 * torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]),
        },
        'SobelFB': {
            'F_x': 0.25 * torch.tensor([[0., -1., 1.], [0., -2., 2.], [0., -1., 1.]]),
            'F_y': 0.25 * torch.tensor([[1., 2., 1.], [-1., -2., -1.], [0., 0., 0.]]),
            'B_x': 0.25 * torch.tensor([[-1., 1., 0.], [-2., 2., 0.], [-1., 1., 0.]]),
            'B_y': 0.25 * torch.tensor([[0., 0., 0.], [1., 2., 1.], [-1., -2., -1.]]),
        },
        'Feldman': {
            'F_x': (1/32) * torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]),
            'F_y': (1/32) * torch.tensor([[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]]),
            'B_x': (1/32) * torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]),
            'B_y': (1/32) * torch.tensor([[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]]),
        },
        'FeldmanFB': {
            'F_x': (1/16) * torch.tensor([[0., -3., 3.], [0., -10., 10.], [0., -3., 3.]]),
            'F_y': (1/16) * torch.tensor([[3., 10., 3.], [-3., -10., -3.], [0., 0., 0.]]),
            'B_x': (1/16) * torch.tensor([[-3., 3., 0.], [-10., 10., 0.], [-3., 3., 0.]]),
            'B_y': (1/16) * torch.tensor([[0., 0., 0.], [3., 10., 3.], [-3., -10., -3.]]),
        },
        'circFB': None,  # Computed dynamically
    }
    
    @staticmethod
    def get_filters(diff: str, device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, ...]:
        """
        Get forward and backward FDM filters for specified method.
        
        Args:
            diff: Method name ('FB', 'cent', 'Sobel', 'SobelFB', 'Feldman', 'FeldmanFB', 'circFB')
            device: Target device (cuda/cpu)
            dtype: Data type (default: float32 for numerical stability)
            
        Returns:
            Tuple of (F_x, F_y, B_x, B_y) tensors with shape (1, 1, 3, 3)
        """
        if diff == 'circFB':
            # Dynamic computation for circular forward-backward
            x = (np.sqrt(2) - 1) / 2
            norm = 2 * x + 1
            F_x = torch.tensor([[0., -x, x], [0., -1., 1.], [0., -x, x]], dtype=dtype, device=device) / norm
            F_y = torch.tensor([[x, 1., x], [-x, -1., -x], [0., 0., 0.]], dtype=dtype, device=device) / norm
            B_x = torch.tensor([[-x, x, 0.], [-1., 1., 0.], [-x, x, 0.]], dtype=dtype, device=device) / norm
            B_y = torch.tensor([[0., 0., 0.], [x, 1., x], [-x, -1., -x]], dtype=dtype, device=device) / norm
        else:
            kernels = DiffFilters.KERNELS[diff]
            F_x = kernels['F_x'].to(device=device, dtype=dtype)
            F_y = kernels['F_y'].to(device=device, dtype=dtype)
            B_x = kernels['B_x'].to(device=device, dtype=dtype)
            B_y = kernels['B_y'].to(device=device, dtype=dtype)
        
        # Reshape for conv2d: (out_channels=1, in_channels=1, H=3, W=3)
        return (F_x.view(1, 1, 3, 3), F_y.view(1, 1, 3, 3), 
                B_x.view(1, 1, 3, 3), B_y.view(1, 1, 3, 3))


# ==============================================================================
# Farup 2018/2020 Gradient Field Construction (Batch/GPU)
# ==============================================================================

def compute_farup_gradient_field_batch(
    u0: torch.Tensor,
    u_cvd: torch.Tensor,
    confusion_vectors: Optional[torch.Tensor] = None,
    theta_batch: Optional[torch.Tensor] = None,
    el: Optional[torch.Tensor] = None,
    diff: str = "FB",
    gradient_ed_source: str = "fm100",
    use_pca: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Costruisce campo di gradienti target G secondo Farup 2018/2020 (batch GPU).
    
    CORE ALGORITHM (Farup / Yoshi-II):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Formula:
        G = ∇u₀ + (∇u₀ · e_d) e_c
        
    Batch implementation con PyTorch per elaborazione parallela su GPU.
    
    Args:
        u0: Batch immagini originali (B, H, W, 3) in RGB lineare [0,1]
        u_cvd: Batch immagini CVD simulate (B, H, W, 3) in RGB lineare [0,1]
        confusion_vectors: Vettori confusione pre-calcolati (B, 3) opzionale
        theta_batch: Angoli FM100 in gradi (B,) opzionale - usa geometric CIELUV→RGB
        el: Asse luminanza 3D (default: [1,1,1]/√3)
        diff: Tipo differenze finite ('FB', 'C', 'SB')
        gradient_ed_source: "fm100" (primario) o "pca" (fallback)
        use_pca: Se True, forza PCA anche se confusion_vectors/theta_batch presente
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (Gx, Gy) campi gradiente (B, H, W, 3)
        
    Note:
        Priority: confusion_vectors > theta_batch > PCA
        theta_batch usa conversione geometrica CIELUV→RGB Linear (Vingrys 1988)
        
    References:
        - Farup (2018/2020/2021): CVD-specific gradient construction
        - Vingrys & King-Smith (1988): FM-100 confusion angles
    """
    device = u0.device
    dtype = u0.dtype
    B, H, W, C = u0.shape
    
    # Validazione
    if u0.shape != u_cvd.shape:
        raise ValueError(f"Shape mismatch: u0 {u0.shape} vs u_cvd {u_cvd.shape}")
    
    # Calcola differenza colore
    d0 = u0 - u_cvd  # (B, H, W, 3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: Determina e_d_batch (confusion axis per ogni sample)
    # Priority: confusion_vectors > theta_batch > PCA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if gradient_ed_source == "fm100" and confusion_vectors is not None and not use_pca:
        # ── FM100 MODE (pre-computed): usa vettori confusione pre-calcolati ──
        ed_batch = F.normalize(confusion_vectors.to(device=device, dtype=dtype), dim=1)  # (B, 3)
        
    elif gradient_ed_source == "fm100" and theta_batch is not None and not use_pca:
        # ── FM100 MODE (from theta): geometric CIELUV→RGB Linear conversion ──
        # Usa Vingrys 1988 FM100 theta angles → RGB Linear confusion vectors
        ed_batch = theta_to_confusion_vector_gpu(
            theta_batch.to(device=device, dtype=dtype),
            device=device,
            dtype=dtype
        )  # (B, 3) già normalizzato
        
    else:
        # ── PCA MODE: estrai da color difference (Farup originale) ──
        d0_flat = d0.reshape(B, -1, 3)  # (B, H*W, 3)
        
        # Covariance matrix per ogni sample: cov = d0^T @ d0
        cov = torch.bmm(d0_flat.transpose(1, 2), d0_flat)  # (B, 3, 3)
        
        # Eigendecomposition (complex output)
        eigenvalues, eigenvectors = torch.linalg.eig(cov)  # eigenvalues: (B, 3), eigenvectors: (B, 3, 3)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        # First principal component (massimo autovalore)
        idx_max = eigenvalues.argmax(dim=1)  # (B,)
        ed_batch = eigenvectors[torch.arange(B), :, idx_max]  # (B, 3)
        ed_batch = F.normalize(ed_batch, dim=1)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 2: Definisci e_l (lightness axis)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if el is None:
        el = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
    el = F.normalize(el, dim=0)
    el_batch = el.unsqueeze(0).expand(B, -1)  # (B, 3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3: Gram-Schmidt orthogonalization (CRITICAL) - Batch version
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    dot_ed_el = (ed_batch * el_batch).sum(dim=1, keepdim=True)  # (B, 1)
    
    # Degeneracy check per sample
    degenerate = dot_ed_el.abs() > 0.99  # (B, 1)
    
    if degenerate.any():
        warnings.warn(
            f"Degeneracy detected in {degenerate.sum().item()}/{B} samples. "
            f"Using fallback base gradients.",
            UserWarning
        )
    
    # Gram-Schmidt: rimuovi componente lungo e_l
    ed_batch = ed_batch - dot_ed_el * el_batch  # (B, 3)
    ed_batch = F.normalize(ed_batch, dim=1)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 4: Calcola e_c (compensation direction) - Batch cross product
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ec_batch = torch.cross(ed_batch, el_batch, dim=1)  # (B, 3)
    ec_batch = F.normalize(ec_batch, dim=1)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 5: Calcola gradienti immagini originali
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    gx, gy = scale_gradient_linear_batch(u0, factor=1.0, diff=diff)  # (B, H, W, 3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 6: Proietta gradienti su e_d con broadcasting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ed_exp = ed_batch[:, None, None, :]  # (B, 1, 1, 3)
    sx = (gx * ed_exp).sum(dim=3)  # (B, H, W)
    sy = (gy * ed_exp).sum(dim=3)  # (B, H, W)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 7: Costruisci correzione lungo e_c
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ec_exp = ec_batch[:, None, None, :]  # (B, 1, 1, 3)
    cx = sx.unsqueeze(3) * ec_exp  # (B, H, W, 3)
    cy = sy.unsqueeze(3) * ec_exp  # (B, H, W, 3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 8: Somma per ottenere G = ∇u₀ + (∇u₀ · e_d) e_c
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Gx = gx + cx
    Gy = gy + cy
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 9: Fallback per casi degeneri (usa gradienti base)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if degenerate.any():
        mask = degenerate[:, :, None, None].expand_as(Gx)  # (B, H, W, 3)
        Gx = torch.where(mask, gx, Gx)
        Gy = torch.where(mask, gy, Gy)
    
    return Gx, Gy


def scale_gradient_linear_batch(
    u: torch.Tensor,
    factor: float,
    diff: str = 'FB'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled gradient for batch of images using PyTorch convolution.
    
    Matches scipy.signal.correlate2d(mode='same', boundary='symm') behavior
    by using symmetric padding (replicate edge pixels).
    
    Args:
        u: Input images (B, H, W, 3) in [0, 1] range
        factor: Gradient scaling factor
        diff: Finite difference method
        
    Returns:
        (ux, uy): Scaled gradients, both (B, H, W, 3)
    """
    device = u.device
    dtype = u.dtype
    B, H, W, C = u.shape
    
    # Get filters (ensure same dtype as input)
    fx, fy, _, _ = DiffFilters.get_filters(diff, device, dtype)
    
    # Permute to (B, C, H, W) for conv2d
    u_conv = u.permute(0, 3, 1, 2)  # (B, 3, H, W)
    
    # Symmetric padding to match scipy 'symm' boundary
    # For 3x3 kernel, pad by 1 on each side using replicate (matches scipy 'symm')
    u_padded = F.pad(u_conv, (1, 1, 1, 1), mode='replicate')
    
    # Compute gradients (channel-wise convolution)
    ux_list = []
    uy_list = []
    for c in range(C):
        ux_c = F.conv2d(u_padded[:, c:c+1, :, :], fx, padding=0)  # (B, 1, H, W)
        uy_c = F.conv2d(u_padded[:, c:c+1, :, :], fy, padding=0)  # (B, 1, H, W)
        ux_list.append(ux_c)
        uy_list.append(uy_c)
    
    # Stack channels and permute back to (B, H, W, C)
    ux = torch.cat(ux_list, dim=1).permute(0, 2, 3, 1) * factor  # (B, H, W, 3)
    uy = torch.cat(uy_list, dim=1).permute(0, 2, 3, 1) * factor  # (B, H, W, 3)
    
    return ux, uy


def structure_tensor_batch(
    u: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    kappa: float,
    diff: str = 'FB',
    isotropic: bool = False,
    diff_struct: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute (difference) diffusion tensor for batch of images.
    
    Implements eigenvalue decomposition of structure tensor for anisotropic diffusion.
    Matches Farup's diffusion_tensor() function (gradient.py lines 138-193).
    
    Args:
        u: Current images (B, H, W, 3)
        vx: Target x-gradients (B, H, W, 3)
        vy: Target y-gradients (B, H, W, 3)
        kappa: Diffusion parameter (smaller = more anisotropic)
        diff: Finite difference method
        isotropic: Use isotropic instead of anisotropic
        diff_struct: Use difference structure tensor
        
    Returns:
        (D11, D22, D12): Diffusion tensor components, all (B, H, W)
    """
    device = u.device
    dtype = u.dtype
    B, H, W, C = u.shape
    
    # Compute current gradients
    fx, fy, _, _ = DiffFilters.get_filters(diff, device, dtype)
    u_conv = u.permute(0, 3, 1, 2)  # (B, 3, H, W)
    u_padded = F.pad(u_conv, (1, 1, 1, 1), mode='replicate')
    
    gx_list = []
    gy_list = []
    for c in range(C):
        gx_c = F.conv2d(u_padded[:, c:c+1, :, :], fx, padding=0)  # (B, 1, H, W)
        gy_c = F.conv2d(u_padded[:, c:c+1, :, :], fy, padding=0)  # (B, 1, H, W)
        gx_list.append(gx_c)
        gy_list.append(gy_c)
    
    gx = torch.cat(gx_list, dim=1).permute(0, 2, 3, 1)  # (B, H, W, 3)
    gy = torch.cat(gy_list, dim=1).permute(0, 2, 3, 1)  # (B, H, W, 3)
    
    # Difference structure tensor
    if diff_struct:
        vvx = vx
        vvy = vy
    else:
        vvx = torch.zeros_like(vx)
        vvy = torch.zeros_like(vy)
    
    if isotropic:
        # Isotropic diffusion
        gradsq = ((gx - vvx)**2 + (gy - vvy)**2).sum(dim=3)  # (B, H, W)
        D11 = 1.0 / (1.0 + gradsq**2 / kappa)
        D22 = D11.clone()
        D12 = torch.zeros_like(D11)
    else:
        # Anisotropic diffusion with structure tensor
        S11 = ((gx - vvx)**2).sum(dim=3)  # (B, H, W)
        S12 = ((gx - vvx) * (gy - vvy)).sum(dim=3)  # (B, H, W)
        S22 = ((gy - vvy)**2).sum(dim=3)  # (B, H, W)
        
        # Eigenvalues of structure tensor
        sqrt_term = torch.sqrt((S11 - S22)**2 + 4 * S12**2)
        lambda1 = 0.5 * (S11 + S22 + sqrt_term)
        lambda2 = 0.5 * (S11 + S22 - sqrt_term)
        
        # Eigenvectors
        theta1 = 0.5 * torch.atan2(2 * S12, S11 - S22)
        theta2 = theta1 + np.pi / 2
        
        v1x = torch.cos(theta1)
        v1y = torch.sin(theta1)
        v2x = torch.cos(theta2)
        v2y = torch.sin(theta2)
        
        # Diffusion coefficients
        Dlambda1 = 1.0 / (1.0 + lambda1**2 / kappa)
        Dlambda2 = 1.0 / (1.0 + lambda2**2 / kappa)
        
        # Diffusion tensor
        D11 = Dlambda1 * v1x**2 + Dlambda2 * v2x**2
        D22 = Dlambda1 * v1y**2 + Dlambda2 * v2y**2
        D12 = Dlambda1 * v1x * v1y + Dlambda2 * v2x * v2y
    
    return D11, D22, D12


def gdip_anisotropic_batch(
    u0: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    nit: int = 501,
    kappa: float = 1e-2,
    diff: str = 'FB',
    isotropic: bool = False,
    linear: bool = True,
    diff_struct: bool = True,
    debug: bool = False
) -> torch.Tensor:
    """
    Gradient domain image processing with anisotropic diffusion (batch version).
    
    Faithful PyTorch implementation of Farup's gdip_anisotropic() (gradient.py lines 254-335).
    
    Args:
        u0: Original images (B, H, W, 3) in [0, 1] range
        vx: Target x-gradients (B, H, W, 3)
        vy: Target y-gradients (B, H, W, 3)
        nit: Number of iterations (default: 501, Farup's default)
        kappa: Anisotropy parameter (smaller = more anisotropic)
        diff: Finite difference method ('FB' default)
        isotropic: Use isotropic instead of anisotropic diffusion
        linear: Use linearized equation (True for stability)
        diff_struct: Use difference structure tensor
        debug: Print progress every 10 iterations
        
    Returns:
        u: Processed images (B, H, W, 3) in [0, 1] range
    """
    device = u0.device
    dtype = u0.dtype
    B, H, W, C = u0.shape
    
    # Get filters (ensure same dtype as input)
    fx, fy, bx, by = DiffFilters.get_filters(diff, device, dtype)
    
    # Initialize diffusion tensor (linearized: compute once from u0)
    D11, D22, D12 = structure_tensor_batch(u0, vx, vy, kappa, diff, isotropic, diff_struct)
    
    # Initialize result
    u = u0.clone()
    
    # Allocate gradient buffers
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gxx = torch.zeros_like(u)
    gyy = torch.zeros_like(u)
    
    # Main iteration loop
    for i in range(nit):
        if debug and (i % 10 == 0):
            print(f"Iteration {i}/{nit}")
        
        # Update diffusion tensor if non-linear
        if not linear:
            D11, D22, D12 = structure_tensor_batch(u, vx, vy, kappa, diff, isotropic, diff_struct)
        
        # Compute gradients
        u_conv = u.permute(0, 3, 1, 2)  # (B, 3, H, W)
        u_padded = F.pad(u_conv, (1, 1, 1, 1), mode='replicate')
        
        for c in range(C):
            # Forward gradients
            gx_c = F.conv2d(u_padded[:, c:c+1, :, :], fx, padding=0)  # (B, 1, H, W)
            gy_c = F.conv2d(u_padded[:, c:c+1, :, :], fy, padding=0)  # (B, 1, H, W)
            gx[:, :, :, c] = gx_c.squeeze(1)
            gy[:, :, :, c] = gy_c.squeeze(1)
            
            # Anisotropic diffusion terms
            diff_x = D11 * (gx[:, :, :, c] - vx[:, :, :, c]) + D12 * (gy[:, :, :, c] - vy[:, :, :, c])
            diff_y = D12 * (gx[:, :, :, c] - vx[:, :, :, c]) + D22 * (gy[:, :, :, c] - vy[:, :, :, c])
            
            # Ensure same dtype for convolution
            diff_x = diff_x.to(dtype)
            diff_y = diff_y.to(dtype)
            
            # Backward divergence
            diff_x_padded = F.pad(diff_x.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
            diff_y_padded = F.pad(diff_y.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
            
            gxx_c = F.conv2d(diff_x_padded, bx, padding=0)  # (B, 1, H, W)
            gyy_c = F.conv2d(diff_y_padded, by, padding=0)  # (B, 1, H, W)
            
            gxx[:, :, :, c] = gxx_c.squeeze(1)
            gyy[:, :, :, c] = gyy_c.squeeze(1)
        
        # Update with fixed step size (0.24 from Farup)
        u += 0.24 * (gxx + gyy)
        
        # Clamp to [0, 1]
        u = torch.clamp(u, 0.0, 1.0)
    
    return u


def daltonize_farup_batch_gpu(
    images: torch.Tensor,
    cvd_simulated: torch.Tensor,
    config: Optional[Dict] = None
) -> torch.Tensor:
    """
    Batch GPU-accelerated Farup 2018/2020 + GDIP 2021 con FM100.
    
    CORE ALGORITHM (Farup / Yoshi-II):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Costruzione campo di gradienti target:
        G = ∇u₀ + (∇u₀ · e_d) e_c
        
    Ricostruzione via GDIP anisotropic solver (Farup 2021).
    
    FM100/VINGRYS LAYER (MODELING):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - FM100 → confusion_vectors → e_d (opzionale, via config)
    -  NON fa parte del metodo Farup originale
    
    Args:
        images: Original images (B, H, W, 3) in [0, 1] range, RGB, float32
        cvd_simulated: CVD-simulated versions (B, H, W, 3) in [0, 1] range
        config: Optional configuration dict with keys:
            - nit: int (default 501)
            - kappa: float (default 1e-2)
            - diff: str (default 'FB')
            - isotropic: bool (default False)
            - linear: bool (default True)
            - diff_struct: bool (default True)
            - debug: bool (default False)
            - confusion_vectors: torch.Tensor (B, 3) opzionale (FM100)
            - gradient_ed_source: str "fm100" o "pca" (default "fm100")
            
    Returns:
        compensated: Daltonized images (B, H, W, 3) in [0, 1] range
        
    References:
        - Farup (2018/2020/2021): CVD-specific gradient construction
        - Vingrys & King-Smith (1988): FM-100 Hue Test
    """
    # Default configuration (Farup 2020 official)
    default_config = {
        'nit': 501,
        'kappa': 1e-2,
        'diff': 'FB',
        'isotropic': False,
        'linear': True,
        'diff_struct': True,
        'debug': False,
        'gradient_ed_source': 'fm100',  # FM100 as primary source
        'confusion_vectors': None,      # Provided by dataset generator
    }
    
    if config is not None:
        default_config.update(config)
    cfg = default_config
    
    # CHECK: Identity teacher for normal vision profiles (gradient_ed_source='none')
    gradient_ed_source = cfg.get('gradient_ed_source', 'fm100')
    if gradient_ed_source == 'none':
        # Normal vision: return original images (no compensation)
        return images.clone()
    
    # Ensure images on GPU
    device = images.device
    if not images.is_cuda:
        warnings.warn("Images not on CUDA device. Consider moving to GPU for acceleration.")
    
    # Compute target gradients G via Farup 2018/2020 formula
    # With FM100 as primary source of e_d, PCA as fallback
    vx, vy = compute_farup_gradient_field_batch(
        u0=images,
        u_cvd=cvd_simulated,
        confusion_vectors=cfg.get('confusion_vectors', None),
        el=None,  # Default: [1,1,1]/√3
        diff=cfg['diff'],
        gradient_ed_source=gradient_ed_source,
        use_pca=False  # PCA solo se gradient_ed_source="pca"
    )
    
    # Run GDIP anisotropic diffusion
    compensated = gdip_anisotropic_batch(
        u0=images,
        vx=vx,
        vy=vy,
        nit=cfg['nit'],
        kappa=cfg['kappa'],
        diff=cfg['diff'],
        isotropic=cfg['isotropic'],
        linear=cfg['linear'],
        diff_struct=cfg['diff_struct'],
        debug=cfg['debug']
    )
    
    return compensated


# Hardware detection and optimization
def get_optimal_batch_size(device: torch.device, image_size: Tuple[int, int] = (256, 256)) -> int:
    """
    Determine optimal batch size based on GPU memory.
    
    Args:
        device: Target device
        image_size: (H, W) of images
        
    Returns:
        Optimal batch size
    """
    if not device.type == 'cuda':
        return 1  # CPU fallback
    
    # Get GPU memory
    total_mem = torch.cuda.get_device_properties(device).total_memory
    total_mem_gb = total_mem / (1024**3)
    
    # Empirical estimates for RTX 3090 (24GB)
    # Each 256x256 image with nit=501 needs ~750MB per image in batch
    H, W = image_size
    mem_per_image_mb = (H * W * 3 * 4 * 501 * 2) / (1024**2)  # float32, gradients, buffers
    
    # Conservative batch size (use 80% of memory)
    usable_mem_mb = total_mem_gb * 1024 * 0.8
    batch_size = max(1, int(usable_mem_mb / mem_per_image_mb))
    
    # Cap at 32 for RTX 3090 as recommended
    if total_mem_gb >= 20:
        batch_size = min(batch_size, 32)
    elif total_mem_gb >= 10:
        batch_size = min(batch_size, 16)
    else:
        batch_size = min(batch_size, 8)
    
    return batch_size


def print_gpu_info():
    """Print GPU hardware information for debugging."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(device)
        print(f"GPU Detected: {props.name}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Optimal Batch Size: {get_optimal_batch_size(device)}")
    else:
        print("No CUDA device available. Using CPU fallback.")


if __name__ == "__main__":
    # Quick hardware check
    print_gpu_info()
    
    # Numerical validation against scipy reference
    print("\nNumerical validation tests should be run separately.")
    print("Use test_farup_gpu_validation.py for comprehensive testing.")
