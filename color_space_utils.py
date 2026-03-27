"""
color_space_utils — Conversioni spazi colore per la pipeline CVD compensation.

Implementa: sRGB ↔ XYZ ↔ CIELUV e RGB ↔ YCbCr (BT.601).

La conversione YCbCr è utilizzata dal forward Y'-Preserving del modello
(CVDCompensationModelAdaIN) per separare luma Y' e crominanza Cb/Cr.
Le trasformazioni Lab sono usate internamente dalla loss (CVDLoss).

Riferimenti:
- IEC 61966-2-1:1999 (sRGB standard)
- CIE 1976 (CIELUV color space)
- CIE 1931 (XYZ color space, D65 illuminant)
- ITU-R BT.601 (YCbCr for Y'-Preserving architecture)
"""

from __future__ import annotations

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

# ============ MATRICES (RIUTILIZZATE DA CODICE ESISTENTE) ============
# Fonte: dataset_generator_cvd.py linee 427-438
# Standard: IEC 61966-2-1:1999 (sRGB)

M_sRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float32)

M_XYZ_TO_sRGB = np.linalg.inv(M_sRGB_TO_XYZ).astype(np.float32)

# White Point D65 (CIE 1931 2° observer)
XN, YN, ZN = 95.047, 100.000, 108.883

# u'_n, v'_n chromaticity coordinates for D65
U_PRIME_N = 4 * XN / (XN + 15*YN + 3*ZN)  # ≈ 0.2009
V_PRIME_N = 9 * YN / (XN + 15*YN + 3*ZN)  # ≈ 0.4610

# ============ RGB ↔ XYZ ============
def inverse_srgb_companding(c):
    """
    Inverse gamma (sRGB -> linear RGB)
    IEC 61966-2-1:1999 standard
    
    Args:
        c: sRGB values in [0, 1]
    
    Returns:
        Linear RGB values
    """
    return np.where(
        c <= 0.04045,
        c / 12.92,
        np.power((c + 0.055) / 1.055, 2.4)
    )

def srgb_companding(c):
    """
    Gamma companding (linear RGB -> sRGB)
    IEC 61966-2-1:1999 standard
    
    Uses explicit masking instead of np.where to avoid RuntimeWarning.
    The issue: np.where evaluates BOTH branches before applying the mask,
    so np.power(c, 1/2.4) gets computed even for values where c <= 0.0031308,
    causing warnings if any floating-point edge cases exist.
    
    IMPORTANT: This function expects input in [0, 1].
    For out-of-gamut values, apply perceptual_gamut_clip BEFORE calling this.
    
    Args:
        c: Linear RGB values in [0, 1] (after perceptual gamut mapping)
    
    Returns:
        sRGB gamma-corrected values in [0, 1]
    
    References:
        IEC 61966-2-1:1999 - sRGB standard
    """
    result = np.empty_like(c)
    
    # Linear segment for dark values (c <= 0.0031308)
    mask_linear = (c <= 0.0031308)
    result[mask_linear] = 12.92 * c[mask_linear]
    
    # Gamma segment for normal values (c > 0.0031308)
    mask_gamma = ~mask_linear
    result[mask_gamma] = 1.055 * np.power(c[mask_gamma], 1/2.4) - 0.055
    
    return result

def rgb_to_xyz(rgb_image):
    """
    sRGB [0,1] -> XYZ (D65)
    
    Args:
        rgb_image: (H, W, 3) in [0, 1] sRGB
    
    Returns:
        xyz_image: (H, W, 3) in XYZ (scaled to [0, 100] for Y)
    """
    # Step 1: Inverse gamma (sRGB -> linear RGB)
    rgb_linear = inverse_srgb_companding(rgb_image)
    
    # Step 2: Matrix transformation
    h, w, c = rgb_linear.shape
    rgb_flat = rgb_linear.reshape(-1, 3)
    xyz_flat = rgb_flat @ M_sRGB_TO_XYZ.T
    
    # Scale to standard range (Y in [0, 100])
    xyz_image = xyz_flat.reshape(h, w, c) * 100.0
    
    return xyz_image

def xyz_to_rgb(xyz_image):
    """
    XYZ (D65) -> sRGB [0,1]
    
    Args:
        xyz_image: (H, W, 3) in XYZ (Y scaled to [0, 100])
    
    Returns:
        rgb_image: (H, W, 3) in [0, 1] sRGB (clipped)
    """
    # Get linear RGB (may be out-of-gamut)
    rgb_linear = xyz_to_rgb_linear(xyz_image)
    
    # Apply gamma companding
    rgb_srgb = srgb_companding(rgb_linear)
    
    # Clip to valid range (may have out-of-gamut values)
    return np.clip(rgb_srgb, 0, 1)

def xyz_to_rgb_linear(xyz_image):
    """
    XYZ (D65) -> Linear RGB (NO gamma companding)
    
    Returns RGB in LINEAR space, which may contain out-of-gamut values
    (negatives or >1). Caller is responsible for gamut mapping.
    
    Args:
        xyz_image: (H, W, 3) in XYZ (Y scaled to [0, 100])
    
    Returns:
        rgb_linear: (H, W, 3) in linear RGB (may be outside [0,1])
    """
    # Normalize XYZ to [0, 1] range
    xyz_normalized = xyz_image / 100.0
    
    # Matrix transformation
    h, w, c = xyz_normalized.shape
    xyz_flat = xyz_normalized.reshape(-1, 3)
    rgb_linear_flat = xyz_flat @ M_XYZ_TO_sRGB.T
    rgb_linear = rgb_linear_flat.reshape(h, w, c)
    
    return rgb_linear

# ============ XYZ ↔ CIELUV ============
def xyz_to_luv(xyz_image):
    """
    XYZ -> CIELUV (L*, u*, v*)
    CIE 1976 standard
    
    Args:
        xyz_image: (H, W, 3) in XYZ (Y in [0, 100])
    
    Returns:
        luv_image: (H, W, 3) [L*, u*, v*]
                   L* in [0, 100], u* in [-100, 100], v* in [-100, 100]
    """
    X, Y, Z = xyz_image[..., 0], xyz_image[..., 1], xyz_image[..., 2]
    
    # === L* calculation ===
    # CIE 1976: L* = 116 * f(Y/Yn) - 16
    # where f(t) = t^(1/3) if t > (6/29)^3, else t/(3*(6/29)^2) + 4/29
    
    y_ratio = Y / YN
    
    # Threshold for linear vs cubic regime
    threshold = (6/29)**3  # ≈ 0.008856
    
    # f(Y/Yn) with conditional
    fy = np.where(
        y_ratio > threshold,
        np.power(y_ratio, 1/3),
        y_ratio / (3 * (6/29)**2) + 4/29
    )
    
    L_star = 116 * fy - 16
    L_star = np.clip(L_star, 0, 100)  # Ensure valid range
    
    # === u', v' chromaticity coordinates ===
    denom = X + 15*Y + 3*Z
    denom = np.clip(denom, 1e-10, None)  # Avoid division by zero
    
    u_prime = 4 * X / denom
    v_prime = 9 * Y / denom
    
    # === u*, v* calculation ===
    # u* = 13 * L* * (u' - u'_n)
    # v* = 13 * L* * (v' - v'_n)
    
    u_star = 13 * L_star * (u_prime - U_PRIME_N)
    v_star = 13 * L_star * (v_prime - V_PRIME_N)
    
    return np.stack([L_star, u_star, v_star], axis=-1)

def luv_to_xyz(luv_image):
    """
    CIELUV -> XYZ (inverse transformation)
    
    Args:
        luv_image: (H, W, 3) [L*, u*, v*]
    
    Returns:
        xyz_image: (H, W, 3) in XYZ (Y in [0, 100])
    """
    L_star, u_star, v_star = luv_image[..., 0], luv_image[..., 1], luv_image[..., 2]
    
    # === Recover u', v' from u*, v* ===
    # u' = u* / (13 * L*) + u'_n
    # v' = v* / (13 * L*) + v'_n
    
    # Avoid division by zero for L* = 0
    L_safe = np.clip(L_star, 1e-10, None)
    
    u_prime = u_star / (13 * L_safe) + U_PRIME_N
    v_prime = v_star / (13 * L_safe) + V_PRIME_N
    
    # === Recover Y from L* ===
    # L* = 116 * f(Y/Yn) - 16
    # f(Y/Yn) = (L* + 16) / 116
    
    fy = (L_star + 16) / 116
    
    # Inverse of f: Y/Yn = f^(-1)(fy)
    threshold = 6/29
    y_ratio = np.where(
        fy > threshold,
        np.power(fy, 3),
        3 * (6/29)**2 * (fy - 4/29)
    )
    
    Y = YN * y_ratio
    
    # === Recover X, Z from u', v', Y ===
    # u' = 4X / (X + 15Y + 3Z)
    # v' = 9Y / (X + 15Y + 3Z)
    
    # From v': (X + 15Y + 3Z) = 9Y / v'
    # From u': X = (u' * 9Y) / (4 * v')
    # From v': Z = (9Y / v' - X - 15Y) / 3
    
    v_prime_safe = np.clip(v_prime, 1e-10, None)
    
    X = Y * (9 * u_prime) / (4 * v_prime_safe)
    Z = Y * (12 - 3*u_prime - 20*v_prime) / (4*v_prime_safe)
    
    return np.stack([X, Y, Z], axis=-1)

# ============ CONVENIENCE FUNCTIONS ============
def rgb_to_luv(rgb_image):
    """
    sRGB -> CIELUV (one-step)
    
    Args:
        rgb_image: (H, W, 3) in [0, 1] sRGB
    
    Returns:
        luv_image: (H, W, 3) [L*, u*, v*]
    """
    xyz = rgb_to_xyz(rgb_image)
    return xyz_to_luv(xyz)

def luv_to_rgb(luv_image):
    """
    CIELUV -> sRGB (one-step)
    
    Args:
        luv_image: (H, W, 3) [L*, u*, v*]
    
    Returns:
        rgb_image: (H, W, 3) in [0, 1] sRGB (clipped)
    """
    xyz = luv_to_xyz(luv_image)
    return xyz_to_rgb(xyz)


# ============ VALIDATION TESTS ============
def test_round_trip():
    """Test RGB -> CIELUV -> RGB round-trip accuracy"""
    np.random.seed(42)
    
    # Generate 1000 random RGB pixels
    rgb_test = np.random.rand(32, 32, 3)
    
    # Round-trip
    luv = rgb_to_luv(rgb_test)
    rgb_recovered = luv_to_rgb(luv)
    
    # Calculate error
    error = np.abs(rgb_recovered - rgb_test)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"[Color Space Test] Round-trip RGB -> CIELUV -> RGB")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    
    if max_error < 0.01:
        print(f"   PASS (error < 0.01)")
        return True
    else:
        print(f"   FAIL (error >= 0.01)")
        return False

# ============ PYTORCH IMPLEMENTATIONS (for Deep Learning) ============
def rgb_to_lab_torch(rgb_tensor, input_range='unit'):
    """
    PyTorch-based RGB to LAB conversion (differentiable, GPU-compatible)
    
    Consolidates conversion logic from CVDLoss and train_phase_CONVNEXT.py
    to provide a single, tested, centralized implementation.
    
    Args:
        rgb_tensor: torch.Tensor, shape (B, 3, H, W)
        input_range: str, either 'tanh' or 'unit'
            - 'tanh': input in [-1, 1] (standard for tanh activations)
            - 'unit': input in [0, 1] (standard for sigmoid/no activation)
    
    Returns:
        lab_tensor: torch.Tensor, shape (B, 3, H, W) in standard LAB format
            - L: [0, 100]
            - a: typically [-128, 127]
            - b: typically [-128, 127]
    
    Scientific References:
        - IEC 61966-2-1:1999 (sRGB standard)
        - CIE 1976 (CIELAB color space)
        - CIE 1931 (XYZ color space, D65 illuminant)
    
    Notes:
        - Uses float32 for numerical stability
        - Applies perceptual gamut clipping to prevent out-of-gamut issues
        - Maintains differentiability for backpropagation
    """
    import torch
    
    # PROTEZIONE: Usa sempre float32 per stabilità numerica
    original_dtype = rgb_tensor.dtype
    rgb_float32 = rgb_tensor.float()
    
    # Step 1: Converti al range [0,1] basato su input_range
    if input_range == 'tanh':
        # Input in [-1, 1] → converti a [0, 1]
        rgb = (rgb_float32 + 1.0) / 2.0
    elif input_range == 'unit':
        # Input già in [0, 1]
        rgb = rgb_float32
    else:
        raise ValueError(f"input_range deve essere 'tanh' o 'unit', ricevuto: {input_range}")
    
    # CRITICAL: Clamp to [0, 1] to prevent NaN in srgb_to_linear
    # Values slightly outside range can occur due to numerical precision
    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    # Step 2: Linearizzazione sRGB (BEFORE gamut clipping for correctness)
    def srgb_to_linear(x):
        """Inverse gamma: sRGB [0,1] → linear RGB"""
        # Clamp x to prevent pow() on negative values which produces NaN
        x_safe = torch.clamp(x, min=0.0, max=1.0)
        return torch.where(x_safe <= 0.04045, x_safe / 12.92, torch.pow((x_safe + 0.055) / 1.055, 2.4))
    
    rgb_linear = srgb_to_linear(rgb)
    
    # Step 3: Clamp su RGB lineare per RGB->LAB
    # NOTA: Nella direzione RGB->LAB, input RGB dovrebbe già essere in [0,1]
    # quindi il clamp qui è solo per sicurezza numerica.
    # Il perceptual gamut clip è necessario nella direzione OPPOSTA (LAB->RGB)
    # dove valori LAB possono produrre RGB fuori gamut.
    rgb_linear = torch.clamp(rgb_linear, 0.0, 1.0)
    
    # Step 4: Conversione RGB lineare -> XYZ usando matrice standard sRGB D65
    # Matrice sempre in float32 per stabilità
    rgb_to_xyz_matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb_tensor.device, dtype=torch.float32)
    
    # Reshape per matrix multiplication: (B, 3, H, W) -> (B, H*W, 3)
    B, C, H, W = rgb_linear.shape
    rgb_reshaped = rgb_linear.permute(0, 2, 3, 1).reshape(B, H*W, 3)
    
    # Moltiplicazione matriciale: (B, H*W, 3) @ (3, 3) -> (B, H*W, 3)
    xyz_reshaped = torch.matmul(rgb_reshaped, rgb_to_xyz_matrix.T)
    
    # Reshape back: (B, H*W, 3) -> (B, 3, H, W)
    xyz = xyz_reshaped.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    # Step 5: Normalizzazione XYZ usando illuminant D65
    # Valori sempre in float32
    xyz_n = torch.tensor([0.95047, 1.00000, 1.08883], 
                        device=rgb_tensor.device, dtype=torch.float32)
    xyz_n = xyz_n.view(1, 3, 1, 1)  # Broadcast shape
    
    xyz_normalized = xyz / xyz_n
    
    # Step 6: Conversione XYZ -> LAB
    def f_xyz_to_lab(t):
        """Funzione di trasformazione XYZ->LAB (CIE 1976)"""
        delta = 6.0 / 29.0
        # NUMERICAL STABILITY: Clamp t to prevent pow() on negative values
        # which would produce NaN in gradient computation
        t_safe = torch.clamp(t, min=1e-8)
        return torch.where(t > delta**3, 
                          torch.pow(t_safe, 1.0/3.0),
                          t / (3.0 * delta**2) + 4.0/29.0)
    
    f_xyz = f_xyz_to_lab(xyz_normalized)
    
    # Calcolo delle componenti LAB
    L = 116.0 * f_xyz[:, 1:2, :, :] - 16.0  # L component
    a = 500.0 * (f_xyz[:, 0:1, :, :] - f_xyz[:, 1:2, :, :])  # a component  
    b = 200.0 * (f_xyz[:, 1:2, :, :] - f_xyz[:, 2:3, :, :])  # b component
    
    # Combina le componenti: (B, 1, H, W) -> (B, 3, H, W)
    lab = torch.cat([L, a, b], dim=1)
    
    # PROTEZIONE: Controlla valori validi prima di restituire
    if torch.isnan(lab).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("[rgb_to_lab_torch] NaN rilevato in conversione LAB")
        logger.debug(f"[rgb_to_lab_torch] RGB input range: [{rgb_float32.min():.4f}, {rgb_float32.max():.4f}]")
        # Fallback sicuro: mantieni il range LAB valido
        lab = torch.nan_to_num(lab, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Ritorna nel dtype originale se non era problematico
    if original_dtype not in [torch.float16, torch.bfloat16]:
        lab = lab.to(original_dtype)
    
    return lab


# ==============================================================================
# THETA -> CONFUSION VECTOR (GEOMETRIC CIELUV METHOD)
# ==============================================================================
# FIX per bug Tritan: sostituisce interpolazione circolare con derivazione
# geometrica rigorosa dall'angolo θ in CIELUV u*v* space.
#
# Metodo "Due Punti" (Differenziale):
#   P1 = (L=50, u*=0, v*=0)           # Grigio medio
#   P2 = (L=50, u*=ε·cos(θ), v*=ε·sin(θ))  # Perturbazione in direzione θ
#   confusion_vector = normalize(P2_rgb_linear - P1_rgb_linear)
#
# Riferimenti:
#   - Vingrys & King-Smith 1988: θ definito in piano CIELUV u*v*
#   - Farup 2018/2020: opera in RGB Lineare
# ==============================================================================

def theta_to_confusion_vector_luv(theta_deg: float) -> np.ndarray:
    """
    Converte confusion angle θ (FM100) in vettore 3D RGB Lineare via CIELUV.
    
    Usa il metodo geometrico "due punti" per derivare il vettore di confusione
    direttamente dalla direzione θ nel piano u*v* di CIELUV.
    
     IMPORTANTE: Output in RGB LINEARE (non gamma-corrected).
    Farup opera in spazio lineare, quindi NO srgb_companding().
    
    Args:
        theta_deg: Confusion angle in degrees [0, 360] o negativi
                   (θ da FM100 Hue Test, Vingrys 1988)
    
    Returns:
        np.ndarray: Vettore (3,) normalizzato in spazio RGB Lineare
                    rappresentante l'asse di confusione cromatica
    
    Examples:
        >>> theta_to_confusion_vector_luv(9.5)    # Protan: R-G dominante
        >>> theta_to_confusion_vector_luv(280.0)  # Tritan: B dominante (FIX!)
    
    Scientific Method:
        1. θ definisce direzione in piano CIELUV u*v*
        2. P1 = grigio medio (L=50, u*=0, v*=0)
        3. P2 = P1 + ε·(cos(θ), sin(θ)) in u*v*
        4. Converti P1, P2 in RGB Lineare via LUV→XYZ→RGB
        5. confusion_vec = normalize(P2 - P1)
    
    References:
        - Vingrys A.J., King-Smith P.E. (1988): FM-100 Hue Test
        - Farup (2018/2020): Gradient-domain daltonisation in Linear RGB
    """
    # Normalize theta to [0, 360]
    theta_normalized = theta_deg % 360.0
    
    # SPECIAL CASE: Normal vision (θ ≈ 0°)
    # Vingrys 1988: Normal subjects have θ=0°, no confusion axis
    if abs(theta_normalized) < 1e-3 or abs(theta_normalized - 360.0) < 1e-3:
        # Return neutral vector (handled by degeneracy check in Farup)
        return np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
    
    # Convert to radians
    theta_rad = np.radians(theta_normalized)
    
    # Direction in u*v* plane
    du = np.cos(theta_rad)
    dv = np.sin(theta_rad)
    
    # Epsilon for differential (small but numerically stable)
    epsilon = 0.1
    
    # Reference luminance (mid-gray)
    L_ref = 50.0
    
    # P1: Gray point at origin of u*v* plane
    P1_luv = np.array([[[L_ref, 0.0, 0.0]]], dtype=np.float64)  # Shape (1,1,3) for luv_to_xyz
    
    # P2: Perturbed point in direction θ
    P2_luv = np.array([[[L_ref, epsilon * du, epsilon * dv]]], dtype=np.float64)
    
    # Convert LUV -> XYZ
    P1_xyz = luv_to_xyz(P1_luv)  # Uses existing function
    P2_xyz = luv_to_xyz(P2_luv)
    
    # Convert XYZ -> RGB Linear (NO gamma companding!)
    # XYZ is scaled to [0, 100], normalize to [0, 1]
    P1_xyz_norm = P1_xyz[0, 0, :] / 100.0
    P2_xyz_norm = P2_xyz[0, 0, :] / 100.0
    
    # Matrix multiplication: XYZ -> RGB Linear
    P1_rgb_linear = P1_xyz_norm @ M_XYZ_TO_sRGB.T
    P2_rgb_linear = P2_xyz_norm @ M_XYZ_TO_sRGB.T
    
    # Differential vector in RGB Linear space
    diff_rgb = P2_rgb_linear - P1_rgb_linear
    
    # Normalize to unit vector
    norm = np.linalg.norm(diff_rgb)
    if norm < 1e-10:
        # Fallback for degenerate case
        return np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
    
    confusion_vec = diff_rgb / norm
    
    return confusion_vec.astype(np.float64)


def theta_to_confusion_vector_gpu(
    theta_batch: 'torch.Tensor',
    device: 'torch.device' = None,
    dtype: 'torch.dtype' = None
) -> 'torch.Tensor':
    """
    GPU batch version: Converte batch di θ in confusion vectors RGB Lineare.
    
    Usa Kornia per conversione LUV→sRGB, poi linearizza con rgb_to_linear_rgb().
    
     IMPORTANTE: Output in RGB LINEARE (non gamma-corrected).
    
    Args:
        theta_batch: Tensor (B,) di confusion angles in gradi
        device: torch device (default: same as theta_batch)
        dtype: torch dtype (default: float32)
    
    Returns:
        torch.Tensor: (B, 3) confusion vectors normalizzati in RGB Lineare
    
    Note:
        - AMP-compatible (Kornia è nativo)
        - Richiede kornia installato
    """
    import torch
    import torch.nn.functional as F
    
    try:
        import kornia.color
    except ImportError:
        raise ImportError(
            "Kornia non installato. Installa con: pip install kornia"
        )
    
    # Setup device/dtype
    if device is None:
        device = theta_batch.device
    if dtype is None:
        dtype = torch.float32
    
    B = theta_batch.shape[0]
    
    # Normalize theta to [0, 360]
    theta_normalized = theta_batch % 360.0
    
    # Convert to radians
    theta_rad = torch.deg2rad(theta_normalized)
    
    # Direction in u*v* plane
    du = torch.cos(theta_rad)  # (B,)
    dv = torch.sin(theta_rad)  # (B,)
    
    # Parameters
    epsilon = 0.1
    L_ref = 50.0
    
    # Build LUV tensors with shape (B, 3, 1, 1) for Kornia
    # P1: Gray point (L=50, u*=0, v*=0)
    P1_L = torch.full((B, 1, 1, 1), L_ref, device=device, dtype=dtype)
    P1_u = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)
    P1_v = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)
    P1_luv = torch.cat([P1_L, P1_u, P1_v], dim=1)  # (B, 3, 1, 1)
    
    # P2: Perturbed point
    P2_L = torch.full((B, 1, 1, 1), L_ref, device=device, dtype=dtype)
    P2_u = (epsilon * du).view(B, 1, 1, 1)
    P2_v = (epsilon * dv).view(B, 1, 1, 1)
    P2_luv = torch.cat([P2_L, P2_u, P2_v], dim=1)  # (B, 3, 1, 1)
    
    # Kornia LUV -> sRGB (gamma-corrected)
    P1_srgb = kornia.color.luv_to_rgb(P1_luv)  # (B, 3, 1, 1)
    P2_srgb = kornia.color.luv_to_rgb(P2_luv)  # (B, 3, 1, 1)
    
    # CRITICAL: Convert sRGB -> RGB Linear (remove gamma)
    # Farup operates in Linear RGB space
    P1_linear = kornia.color.rgb_to_linear_rgb(P1_srgb)  # (B, 3, 1, 1)
    P2_linear = kornia.color.rgb_to_linear_rgb(P2_srgb)  # (B, 3, 1, 1)
    
    # Squeeze to (B, 3)
    P1_linear = P1_linear.squeeze(-1).squeeze(-1)  # (B, 3)
    P2_linear = P2_linear.squeeze(-1).squeeze(-1)  # (B, 3)
    
    # Differential vector
    diff_rgb = P2_linear - P1_linear  # (B, 3)
    
    # Normalize to unit vectors
    confusion_vecs = F.normalize(diff_rgb, p=2, dim=1)  # (B, 3)
    
    # Handle θ ≈ 0° cases (normal vision)
    normal_mask = (theta_normalized.abs() < 1e-3) | ((theta_normalized - 360.0).abs() < 1e-3)
    if normal_mask.any():
        neutral_vec = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype) / (3.0 ** 0.5)
        confusion_vecs[normal_mask] = neutral_vec
    
    return confusion_vecs


# ============ YCbCr CONVERSION FUNCTIONS (Y-PRESERVING ARCHITECTURE) ============
# BT.601 Standard - for Y-Preserving CVD compensation
# Y preserves luminance, Cb/Cr encode chrominance
# Used internally to guarantee Y(output) == Y(input)

def rgb_to_ycbcr_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to YCbCr (BT.601 standard).
    
    This is a LINEAR transformation (differentiable), used internally
    in the Y-Preserving architecture to separate luminance from chrominance.
    
    BT.601 Matrix:
        Y  =  0.299*R + 0.587*G + 0.114*B
        Cb = -0.169*R - 0.331*G + 0.500*B + 128
        Cr =  0.500*R - 0.419*G - 0.081*B + 128
    
    For normalized input [-1, 1], we use the normalized version (no +128 offset).
    
    Args:
        rgb: Input tensor [B, 3, H, W] in range [-1, 1] (tanh normalized)
        
    Returns:
        ycbcr: Output tensor [B, 3, H, W] where:
            - Y channel: luminance in [-1, 1]
            - Cb, Cr channels: chrominance in [-1, 1]
    
    Note:
        This transformation preserves gradients and is fully differentiable.
        Y correlates strongly with L* (~95% correlation).
    """
    # BT.601 coefficients
    # For normalized RGB in [-1, 1], the matrix multiplication gives YCbCr in similar range
    Kr, Kg, Kb = 0.299, 0.587, 0.114
    
    R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    
    # Y: weighted sum (luminance)
    Y = Kr * R + Kg * G + Kb * B
    
    # Cb, Cr: chrominance differences (scaled to maintain range)
    # Cb = (B - Y) / (2 * (1 - Kb)) = (B - Y) / 1.772
    # Cr = (R - Y) / (2 * (1 - Kr)) = (R - Y) / 1.402
    Cb = (B - Y) / (2.0 * (1.0 - Kb))
    Cr = (R - Y) / (2.0 * (1.0 - Kr))
    
    return torch.cat([Y, Cb, Cr], dim=1)


def ycbcr_to_rgb_torch(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Convert YCbCr to RGB (BT.601 inverse).
    
    This is the inverse of rgb_to_ycbcr_torch, also linear and differentiable.
    
    Inverse formulas:
        R = Y + Cr * 2 * (1 - Kr) = Y + Cr * 1.402
        G = Y - Cb * 2 * (1 - Kb) * Kb/Kg - Cr * 2 * (1 - Kr) * Kr/Kg
        B = Y + Cb * 2 * (1 - Kb) = Y + Cb * 1.772
    
    Args:
        ycbcr: Input tensor [B, 3, H, W] with Y, Cb, Cr channels
        
    Returns:
        rgb: Output tensor [B, 3, H, W] in range [-1, 1]
    
    Note:
        No clamping is applied here - caller should clamp after if needed.
        This maintains gradient flow during training.
    """
    # BT.601 coefficients
    Kr, Kg, Kb = 0.299, 0.587, 0.114
    
    Y, Cb, Cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    
    # Inverse transformation
    R = Y + Cr * 2.0 * (1.0 - Kr)
    B = Y + Cb * 2.0 * (1.0 - Kb)
    G = Y - Cb * 2.0 * (1.0 - Kb) * (Kb / Kg) - Cr * 2.0 * (1.0 - Kr) * (Kr / Kg)
    
    return torch.cat([R, G, B], dim=1)


def test_ycbcr_roundtrip():
    """Test YCbCr conversion round-trip accuracy."""
    print("\n" + "=" * 60)
    print("TEST: YCbCr Round-Trip (BT.601)")
    print("=" * 60)
    
    # Test tensor
    rgb = torch.rand(2, 3, 64, 64) * 2 - 1  # [-1, 1]
    
    # Forward
    ycbcr = rgb_to_ycbcr_torch(rgb)
    
    # Inverse
    rgb_reconstructed = ycbcr_to_rgb_torch(ycbcr)
    
    # Error
    max_error = (rgb - rgb_reconstructed).abs().max().item()
    mean_error = (rgb - rgb_reconstructed).abs().mean().item()
    
    print(f"Input RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    print(f"YCbCr Y range: [{ycbcr[:, 0].min():.4f}, {ycbcr[:, 0].max():.4f}]")
    print(f"YCbCr Cb range: [{ycbcr[:, 1].min():.4f}, {ycbcr[:, 1].max():.4f}]")
    print(f"YCbCr Cr range: [{ycbcr[:, 2].min():.4f}, {ycbcr[:, 2].max():.4f}]")
    print(f"Round-trip max error: {max_error:.2e}")
    print(f"Round-trip mean error: {mean_error:.2e}")
    
    if max_error < 1e-5:
        print("✅ YCbCr round-trip: PASSED")
    else:
        print("❌ YCbCr round-trip: FAILED (error too high)")
    
    return max_error < 1e-5


# ==============================================================================
# SCOTERYTHROPY COMPENSATION (Post-Processing for Protanopia)
# ==============================================================================
# PROTANOPES hanno una ridotta percezione della luminanza nei rossi (30-50%),
# dovuta alla mancanza del cono L. Questo fenomeno è chiamato "scoterythropy"
# e NON è presente nei deuteranopi.
#
# Compensazione: Aumentiamo L* per i pixel "rossi" SOLO per profili PROTAN.
# Questo differenzia PROTAN da DEUTAN anche se la compensazione cromatica
# (Farup) è identica per entrambi (dato che i confusion vectors sono paralleli).
#
# Riferimenti Scientifici:
#   - Birch J. (2012): "Worldwide prevalence of red-green color deficiency"
#   - Cole et al. (2006): "Detection of defective colour vision"
#   - Wyszecki & Stiles (2000): "Color Science" - Section on scotopic vision
# ==============================================================================

def lab_to_rgb_torch(lab_tensor, output_range='unit'):
    """
    PyTorch-based LAB to RGB conversion (differentiable, GPU-compatible)
    
    Inverse of rgb_to_lab_torch().
    
    Args:
        lab_tensor: torch.Tensor, shape (B, 3, H, W) in standard LAB format
            - L: [0, 100]
            - a: typically [-128, 127]
            - b: typically [-128, 127]
        output_range: str, either 'tanh' or 'unit'
            - 'tanh': output in [-1, 1]
            - 'unit': output in [0, 1]
    
    Returns:
        rgb_tensor: torch.Tensor, shape (B, 3, H, W)
    
    Notes:
        - Uses Kornia for efficiency and correctness
        - Applies gamut clipping to [0, 1] before output transformation
    """
    import torch
    try:
        import kornia.color
    except ImportError:
        raise ImportError(
            "Kornia non installato. Installa con: pip install kornia"
        )
    
    # Ensure float32 for numerical stability
    original_dtype = lab_tensor.dtype
    lab_float32 = lab_tensor.float()
    
    # Kornia expects LAB with L in [0, 100], a/b in [-128, 127]
    # Our format is already correct!
    rgb_srgb = kornia.color.lab_to_rgb(lab_float32)  # Output in [0, 1]
    
    # Clamp to valid sRGB gamut
    rgb_srgb = torch.clamp(rgb_srgb, 0.0, 1.0)
    
    # Convert to requested output range
    if output_range == 'tanh':
        rgb_out = rgb_srgb * 2.0 - 1.0
    elif output_range == 'unit':
        rgb_out = rgb_srgb
    else:
        raise ValueError(f"output_range deve essere 'tanh' o 'unit', ricevuto: {output_range}")
    
    # Restore original dtype if safe
    if original_dtype not in [torch.float16, torch.bfloat16]:
        rgb_out = rgb_out.to(original_dtype)
    
    return rgb_out


def compute_severity_from_profile(profile_tensor, C_index=1, S_index=2):
    """
    Compute CVD severity from 3D HYBRID profile vector.
    
    The profile format (HYBRID 3D) is:
        [θ_norm, C_norm, S_norm]
    
    Severity is computed as: severity = sigmoid(C_norm) * sigmoid(S_norm)
    where C_norm and S_norm are z-scores (can be negative or > 1).
    
    Args:
        profile_tensor: torch.Tensor, shape (B, 3) or (3,) - normalized profile
        C_index: int, index of C_norm in profile (default: 1)
        S_index: int, index of S_norm in profile (default: 2)
    
    Returns:
        severity: torch.Tensor, shape (B,) or scalar - values in [0, 1]
    """
    import torch
    
    # Handle both batched and unbatched profiles
    if profile_tensor.dim() == 1:
        C_norm = profile_tensor[C_index]
        S_norm = profile_tensor[S_index]
    else:
        C_norm = profile_tensor[:, C_index]
        S_norm = profile_tensor[:, S_index]
    
    # C_norm and S_norm are z-scores, map to [0, 1] using sigmoid
    C_severity = torch.sigmoid(C_norm)
    S_severity = torch.sigmoid(S_norm)
    
    # Severity = C_severity * S_severity
    # Scale by 4.0 because sigmoid(0)*sigmoid(0) = 0.25
    severity = C_severity * S_severity * 4.0
    severity = torch.clamp(severity, 0.0, 1.0)
    
    return severity


def apply_scoterythropy_compensation(
    rgb_output,
    cvd_profile,
    max_boost_L=25.0,
    red_hue_tolerance=30.0,
    min_chroma_threshold=20.0,
    theta_index=0,
    C_index=1,
    S_index=2
):
    """
    Apply scoterythropy compensation for PROTAN profiles.
    
    Protanopes perceive reds as 30-50% darker due to missing L-cone.
    This function boosts luminance (L*) for red-ish pixels ONLY when
    the CVD profile indicates PROTAN type (determined by θ_norm).
    
    The boost is proportional to:
        1. How "red" the pixel is (hue near 0° in LAB a*-b* plane)
        2. How saturated (chromatic) the pixel is
        3. The severity of the CVD (C * S from profile)
        4. Whether it's PROTAN (determined from θ_norm)
    
    Args:
        rgb_output: torch.Tensor, shape (B, 3, H, W) in [0, 1] range
                    Output from the model AFTER standard inference
        cvd_profile: torch.Tensor, shape (B, 3) - normalized 3D HYBRID profile
                     Format: [θ_norm, C_norm, S_norm]
                     θ_norm discriminates CVD types:
                       - Protan: θ_norm > +0.3 (θ ≈ +8° to +11°)
                       - Deutan: θ_norm ≈ +0.1 to +0.3 (θ ≈ -9° to -3°)
                       - Tritan: θ_norm < -0.3 (θ ≈ -81° to -39°)
        max_boost_L: float, maximum L* boost (default: 25.0)
                     Scientific range: 10-30 based on 30-50% luminance reduction
        red_hue_tolerance: float, degrees from 0° to consider "red" (default: 30°)
        min_chroma_threshold: float, minimum chroma to apply boost (default: 20.0)
        theta_index: int, index of θ_norm in profile (default: 0)
        C_index: int, index of C_norm in profile (default: 1)
        S_index: int, index of S_norm in profile (default: 2)
    
    Returns:
        rgb_compensated: torch.Tensor, shape (B, 3, H, W) in [0, 1] range
                         With scoterythropy compensation applied for PROTAN
    
    Technical Notes:
        - Operates in LAB space for perceptual correctness
        - Only modifies L* channel, preserving chrominance
        - Smooth falloff from red hue center using cosine weighting
        - No effect on DEUTAN or TRITAN profiles
        - PROTAN detection: θ_norm > +0.3 (positive θ values ≈ +8° to +11°)
    
    Scientific Basis:
        - Protanopes have ~50% reduced L-cone sensitivity at 560nm (peak)
        - This causes reds to appear darker (scoterythropy)
        - Compensation: brighten reds proportionally to redness and severity
    """
    import torch
    import math
    
    # Ensure float32 for LAB conversion stability
    device = rgb_output.device
    rgb_float = rgb_output.float()
    
    # Extract θ_norm and severity parameters
    if cvd_profile.dim() == 1:
        theta_norm = cvd_profile[theta_index]
        C_norm = cvd_profile[C_index]
        S_norm = cvd_profile[S_index]
    else:
        theta_norm = cvd_profile[:, theta_index]  # (B,)
        C_norm = cvd_profile[:, C_index]          # (B,)
        S_norm = cvd_profile[:, S_index]          # (B,)
    
    # PROTAN detection from θ_norm (3D HYBRID format)
    # Protan has θ ≈ +8° to +11°, which normalizes to θ_norm > +0.3
    # Using smooth transition instead of hard threshold
    # θ_norm: Protan ≈ +0.85, Deutan ≈ +0.42, Tritan ≈ -0.6
    PROTAN_THRESHOLD = 0.5  # Threshold between Protan (+0.85) and Deutan (+0.42)
    is_protan = torch.sigmoid((theta_norm - PROTAN_THRESHOLD) * 10.0)  # Smooth 0-1 transition
    
    # C_norm e S_norm sono z-scores (possono essere >> 1 o negativi)
    # Convertiamo in un range [0, 1] usando sigmoid-like mapping
    # Prima clamp a range ragionevole, poi mappa a [0, 1]
    # z-score di 2 -> ~0.88, z-score di 0 -> 0.5, z-score di -2 -> ~0.12
    C_severity = torch.sigmoid(C_norm)  # Mappa z-score a [0, 1]
    S_severity = torch.sigmoid(S_norm)  # Mappa z-score a [0, 1]
    severity = C_severity * S_severity  # Prodotto in [0, 1] -> max ~0.25 per valori medi
    
    # Scala per avere range effettivo [0, 1] (sigmoid(0)*sigmoid(0) = 0.25)
    severity = severity * 4.0  # Ora [0, 1] per casi estremi
    severity = torch.clamp(severity, 0.0, 1.0)  # Safety clamp
    
    # Early exit if no PROTAN in batch
    if is_protan.max() < 0.1:
        return rgb_output  # No modification needed
    
    # Convert to LAB
    lab = rgb_to_lab_torch(rgb_float, input_range='unit')  # (B, 3, H, W)
    L = lab[:, 0:1, :, :]  # (B, 1, H, W)
    a = lab[:, 1:2, :, :]  # (B, 1, H, W)
    b = lab[:, 2:3, :, :]  # (B, 1, H, W)
    
    # Compute chroma in LAB: C_ab = sqrt(a² + b²)
    chroma_ab = torch.sqrt(a**2 + b**2 + 1e-8)  # (B, 1, H, W)
    
    # Compute hue angle in LAB a*-b* plane
    # IMPORTANT: In LAB, "red" is NOT at hue=0°!
    # - Red (RGB 1,0,0): hue ≈ 40° (a*=+80, b*=+67)
    # - Orange: hue ≈ 60-70°
    # - Yellow: hue ≈ 90-100°
    # - Green: hue ≈ 130-180°
    # - Cyan: hue ≈ 200-220°
    # - Blue: hue ≈ 270-300°
    # - Magenta: hue ≈ 320-340°
    hue_rad = torch.atan2(b, a + 1e-8)  # (B, 1, H, W)
    hue_deg = hue_rad * 180.0 / math.pi  # Convert to degrees [-180, 180]
    
    # Red weight: how "red" is this pixel?
    # Red is around hue ≈ 40° in LAB (not 0°!)
    # Include also orange (for stop signs, etc.) up to ~60°
    # Use range [0°, 60°] centered at 30° to capture reds and oranges
    RED_HUE_CENTER = 35.0  # degrees (between pure red ~40° and orange ~60°)
    
    # Calculate angular distance from red center (handle wrapping)
    hue_diff = torch.abs(hue_deg - RED_HUE_CENTER)
    # Handle wrap-around at ±180°
    hue_diff = torch.minimum(hue_diff, 360.0 - hue_diff)
    
    # Smooth falloff: 1.0 at center, 0.0 at tolerance boundary
    # Using cosine falloff for perceptual smoothness
    redness_weight = torch.clamp(1.0 - hue_diff / red_hue_tolerance, 0.0, 1.0)  # (B, 1, H, W)
    
    # Chroma weight: only apply to saturated colors
    # Smooth transition from 0 at min_chroma to 1 at 2*min_chroma
    chroma_weight = torch.clamp(
        (chroma_ab - min_chroma_threshold) / (min_chroma_threshold + 1e-8),
        0.0, 1.0
    )  # (B, 1, H, W)
    
    # Combined weight: redness × chroma
    pixel_weight = redness_weight * chroma_weight  # (B, 1, H, W)
    
    # Apply L* boost proportional to:
    # - pixel_weight (how red and saturated)
    # - severity (C * S from profile)
    # - is_protan flag (0 or 1)
    
    # Reshape scalars for broadcasting: (B,) -> (B, 1, 1, 1)
    if severity.dim() == 0:
        severity = severity.unsqueeze(0)
        is_protan = is_protan.unsqueeze(0)
    severity = severity.view(-1, 1, 1, 1)      # (B, 1, 1, 1)
    is_protan = is_protan.view(-1, 1, 1, 1)    # (B, 1, 1, 1)
    
    # Compute L* boost
    delta_L = max_boost_L * pixel_weight * severity * is_protan  # (B, 1, H, W)
    
    # Apply boost (clamp L* to valid range [0, 100])
    L_boosted = torch.clamp(L + delta_L, 0.0, 100.0)
    
    # Reconstruct LAB with boosted L*
    lab_boosted = torch.cat([L_boosted, a, b], dim=1)  # (B, 3, H, W)
    
    # Convert back to RGB
    rgb_compensated = lab_to_rgb_torch(lab_boosted, output_range='unit')
    
    # Restore original dtype
    rgb_compensated = rgb_compensated.to(rgb_output.dtype)
    
    return rgb_compensated


def test_scoterythropy_compensation():
    """Test scoterythropy compensation on synthetic data."""
    import torch
    
    print("\n" + "=" * 60)
    print("SCOTERYTHROPY COMPENSATION TEST")
    print("=" * 60)
    
    # Create a simple test image: 2x2 with [red, green, blue, gray]
    B, C, H, W = 1, 3, 2, 2
    
    # Pure red, pure green, pure blue, gray
    rgb_test = torch.zeros(B, C, H, W)
    rgb_test[0, 0, 0, 0] = 1.0  # Red: R=1, G=0, B=0
    rgb_test[0, 1, 0, 1] = 1.0  # Green: R=0, G=1, B=0
    rgb_test[0, 2, 1, 0] = 1.0  # Blue: R=0, G=0, B=1
    rgb_test[0, :, 1, 1] = 0.5  # Gray: R=0.5, G=0.5, B=0.5
    
    print(f"\nInput RGB (before compensation):")
    print(f"  Red pixel:   RGB = {rgb_test[0, :, 0, 0].tolist()}")
    print(f"  Green pixel: RGB = {rgb_test[0, :, 0, 1].tolist()}")
    print(f"  Blue pixel:  RGB = {rgb_test[0, :, 1, 0].tolist()}")
    print(f"  Gray pixel:  RGB = {rgb_test[0, :, 1, 1].tolist()}")
    
    # Test 1: PROTAN profile with high severity
    print("\n--- Test 1: PROTAN profile (severity=1.0) ---")
    # Profile: [θ_norm, C_norm, S_norm, is_protan, is_deutan, is_tritan]
    protan_profile = torch.tensor([[0.026, 1.0, 1.0, 1.0, 0.0, 0.0]])
    
    rgb_protan = apply_scoterythropy_compensation(rgb_test, protan_profile)
    
    print(f"Output RGB (PROTAN compensated):")
    print(f"  Red pixel:   RGB = {rgb_protan[0, :, 0, 0].tolist()} <- SHOULD BE BRIGHTER")
    print(f"  Green pixel: RGB = {rgb_protan[0, :, 0, 1].tolist()} <- Should be similar")
    print(f"  Blue pixel:  RGB = {rgb_protan[0, :, 1, 0].tolist()} <- Should be similar")
    print(f"  Gray pixel:  RGB = {rgb_protan[0, :, 1, 1].tolist()} <- Should be same")
    
    # Verify red is brighter
    red_original = rgb_test[0, :, 0, 0].mean().item()
    red_compensated = rgb_protan[0, :, 0, 0].mean().item()
    print(f"\n  Red luminance change: {red_original:.3f} -> {red_compensated:.3f} "
          f"(+{(red_compensated - red_original) * 100:.1f}%)")
    
    # Test 2: DEUTAN profile (should have NO effect)
    print("\n--- Test 2: DEUTAN profile (severity=1.0) ---")
    deutan_profile = torch.tensor([[0.99, 1.0, 1.0, 0.0, 1.0, 0.0]])
    
    rgb_deutan = apply_scoterythropy_compensation(rgb_test, deutan_profile)
    
    diff_deutan = (rgb_deutan - rgb_test).abs().max().item()
    print(f"  Max difference from original: {diff_deutan:.6f}")
    if diff_deutan < 1e-5:
        print("  ✅ DEUTAN: No modification (correct!)")
    else:
        print("  ❌ DEUTAN: Unexpected modification!")
    
    # Test 3: TRITAN profile (should have NO effect)
    print("\n--- Test 3: TRITAN profile (severity=1.0) ---")
    tritan_profile = torch.tensor([[0.78, 1.0, 1.0, 0.0, 0.0, 1.0]])
    
    rgb_tritan = apply_scoterythropy_compensation(rgb_test, tritan_profile)
    
    diff_tritan = (rgb_tritan - rgb_test).abs().max().item()
    print(f"  Max difference from original: {diff_tritan:.6f}")
    if diff_tritan < 1e-5:
        print("  ✅ TRITAN: No modification (correct!)")
    else:
        print("  ❌ TRITAN: Unexpected modification!")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if red_compensated > red_original and diff_deutan < 1e-5 and diff_tritan < 1e-5:
        print("✅ Scoterythropy compensation working correctly!")
        print("   - PROTAN: Red pixels brightened")
        print("   - DEUTAN: No modification")
        print("   - TRITAN: No modification")
        return True
    else:
        print("❌ Scoterythropy compensation has issues!")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("COLOR SPACE UTILITIES - VALIDATION TEST")
    print("=" * 60)
    test_round_trip()
    test_ycbcr_roundtrip()
    test_scoterythropy_compensation()
