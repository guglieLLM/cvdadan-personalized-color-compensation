"""
Simulatore CVD basato su Machado 2009 con conversioni sRGB standard IEC 61966-2-1.

Questo modulo implementa la simulazione di Color Vision Deficiency (CVD) utilizzando
ESCLUSIVAMENTE il modello Machado et al. 2009 via colour-science library.

CARATTERISTICHE:
- Backend OBBLIGATORIO: colour.matrix_cvd_Machado2009 (no alternative)
- Standard sRGB: IEC 61966-2-1:1999 (formule esatte, no approssimazioni)
- Cache pre-computata: 303 matrici (3 tipi x 101 severity levels)
- Interpolazione lineare per severity non-interi

WORKFLOW:
    1. sRGB [0,1] -> Linear RGB (inverse companding IEC 61966-2-1)
    2. Applica matrice Machado 3x3 in spazio lineare
    3. Linear RGB -> sRGB [0,1] (companding IEC 61966-2-1)

Riferimenti:
- Machado et al. 2009: "A Physiologically-based Model for Simulation of Color Vision Deficiency"
- IEC 61966-2-1:1999: Multimedia systems and equipment - Colour measurement and management
"""

import numpy as np
import warnings
from typing import Union, Tuple, Literal

# Torch import opzionale (richiesto solo per batch GPU processing)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from cvd_constants import (
    CVD_TYPE_TO_DEFICIENCY,
    VALID_CVD_TYPES,
    SRGB_TO_LINEAR_THRESHOLD,
    LINEAR_TO_SRGB_THRESHOLD,
    SRGB_GAMMA,
)
from cvd_shared_cache import get_cvd_matrix, get_cache_stats


# ==============================================================================
# IEC 61966-2-1 sRGB ↔ Linear RGB Conversions
# ==============================================================================

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    Converti sRGB [0,1] a linear RGB usando standard IEC 61966-2-1.
    
    Formula (inverse companding):
        linear = srgb / 12.92                          se srgb ≤ 0.04045
        linear = ((srgb + 0.055) / 1.055)^2.4         se srgb > 0.04045
    
    Args:
        srgb: Array numpy con valori sRGB in [0, 1]
              Shape: (..., 3) o qualsiasi shape compatibile
    
    Returns:
        np.ndarray: Linear RGB in [0, 1], stessa shape di input
    
    Examples:
        >>> img_srgb = np.random.rand(256, 256, 3)
        >>> img_linear = srgb_to_linear(img_srgb)
        >>> img_linear.shape
        (256, 256, 3)
    
    Note:
        - Usa soglia esatta 0.04045 (IEC 61966-2-1)
        - Gamma esatto 2.4 (non 2.2 o altre approssimazioni)
        - Preserva dtype float (conversione element-wise)
    """
    srgb = np.asarray(srgb, dtype=np.float32)
    
    # Inverse companding secondo IEC 61966-2-1
    linear = np.where(
        srgb <= SRGB_TO_LINEAR_THRESHOLD,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, SRGB_GAMMA)
    )
    
    return linear


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Converti linear RGB a sRGB [0,1] usando standard IEC 61966-2-1.
    
    Formula (companding):
        srgb = 12.92 * linear                         se linear ≤ 0.0031308
        srgb = 1.055 * linear^(1/2.4) - 0.055        se linear > 0.0031308
    
    Args:
        linear: Array numpy con valori linear RGB in [0, 1]
                Shape: (..., 3) o qualsiasi shape compatibile
    
    Returns:
        np.ndarray: sRGB in [0, 1], stessa shape di input
    
    Examples:
        >>> img_linear = np.random.rand(256, 256, 3)
        >>> img_srgb = linear_to_srgb(img_linear)
        >>> img_srgb.shape
        (256, 256, 3)
    
    Note:
        - Usa soglia esatta 0.0031308 (IEC 61966-2-1)
        - Esponente esatto 1/2.4 ≈ 0.4166667 (non 1/2.2)
        - Clipping a [0,1] applicato per gestire fuori-gamut
    """
    linear = np.asarray(linear, dtype=np.float32)
    
    # Companding secondo IEC 61966-2-1
    srgb = np.where(
        linear <= LINEAR_TO_SRGB_THRESHOLD,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / SRGB_GAMMA) - 0.055
    )
    
    # Clip a [0, 1] per gestire out-of-gamut dopo trasformazione CVD
    srgb = np.clip(srgb, 0.0, 1.0)
    
    return srgb


# ==============================================================================
# Applicazione Matrice CVD in Spazio Lineare
# ==============================================================================

def apply_cvd_matrix(image_linear: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Applica matrice CVD Machado 3x3 a immagine in spazio RGB lineare.
    
    Trasformazione:
        RGB_cvd = matrix @ RGB_linear
    
    Args:
        image_linear: Immagine in linear RGB [0, 1]
                     Shape: (H, W, 3) o (N, 3)
        matrix: Matrice Machado 3x3 (dtype float32)
    
    Returns:
        np.ndarray: Immagine CVD simulata in linear RGB [0, 1]
    
    Note:
        - Usa Einstein summation per efficienza
        - Preserva shape originale (H, W, 3)
        - Clipping a [0, 1] applicato dopo trasformazione
    """
    # Validazione input
    if image_linear.shape[-1] != 3:
        raise ValueError(
            f"Image must have 3 channels (RGB), got shape {image_linear.shape}"
        )
    
    if matrix.shape != (3, 3):
        raise ValueError(
            f"CVD matrix must be 3x3, got shape {matrix.shape}"
        )
    
    # Applica matrice: (H, W, 3) @ (3, 3)^T = (H, W, 3)
    # Einstein notation:ijk,kl->ijl dove k=3 (RGB channels)
    original_shape = image_linear.shape
    image_flat = image_linear.reshape(-1, 3)  # (H*W, 3)
    
    # Matrix multiplication: (H*W, 3) @ (3, 3)^T
    image_cvd_flat = image_flat @ matrix.T
    
    # Reshape to original
    image_cvd = image_cvd_flat.reshape(original_shape)
    
    # Clip per gestire out-of-gamut (alcune combinazioni CVD possono eccedere [0,1])
    image_cvd = np.clip(image_cvd, 0.0, 1.0)
    
    return image_cvd.astype(np.float32)


# ==============================================================================
# Simulazione CVD End-to-End
# ==============================================================================

def simulate_cvd_machado(
    image_srgb: np.ndarray,
    cvd_type: str,
    severity: float
) -> np.ndarray:
    """
    Simula Color Vision Deficiency usando Machado 2009 via colour-science.
    
    Pipeline completa:
        1. Converti sRGB -> linear RGB (IEC 61966-2-1 inverse companding)
        2. Ottieni matrice Machado dalla cache (con interpolazione se necessario)
        3. Applica matrice in spazio lineare: RGB_cvd = matrix @ RGB
        4. Converti linear RGB -> sRGB (IEC 61966-2-1 companding)
    
    Args:
        image_srgb: Immagine in sRGB [0, 1]
                   Shape: (H, W, 3) - array numpy o compatibile
        cvd_type: Tipo CVD - uno di {'protan', 'deutan', 'tritan'}
        severity: Severità CVD in [0, 100]
                 - 0 = visione normale (no CVD)
                 - 100 = CVD completo (dichromacy)
                 - Valori intermedi: anomalous trichromacy
                 - Può essere float (es. 50.3) -> interpolazione lineare
    
    Returns:
        np.ndarray: Immagine CVD simulata in sRGB [0, 1], shape (H, W, 3)
    
    Raises:
        ValueError: Se cvd_type non valido o severity fuori range
        RuntimeError: Se cache Machado non popolata
    
    Examples:
        >>> import numpy as np
        >>> from PIL import Image
        >>> 
        >>> # Carica immagine
        >>> img = Image.open('test.jpg').convert('RGB')
        >>> img_array = np.array(img) / 255.0  # Normalizza a [0,1]
        >>> 
        >>> # Simula protanopia severa
        >>> img_cvd = simulate_cvd_machado(img_array, 'protan', severity=80.0)
        >>> 
        >>> # Simula deuteranomalia lieve (con interpolazione)
        >>> img_cvd = simulate_cvd_machado(img_array, 'deutan', severity=25.5)
        >>> 
        >>> # Salva risultato
        >>> Image.fromarray((img_cvd * 255).astype(np.uint8)).save('cvd.jpg')
    
    Note:
        - Usa ESCLUSIVAMENTE colour.matrix_cvd_Machado2009 (no fallback)
        - Severity normalizzata internamente: [0,100] input -> [0,1] per colour-science
        - Cache hit rate target: >95% (interpolazione solo per severity non-interi)
        - Standard sRGB: IEC 61966-2-1 (gamma 2.4, soglie esatte)
    """
    # Validazione input
    image_srgb = np.asarray(image_srgb, dtype=np.float32)
    
    if image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError(
            f"Image must have shape (H, W, 3), got {image_srgb.shape}"
        )
    
    if not (0.0 <= image_srgb.min() and image_srgb.max() <= 1.0):
        raise ValueError(
            f"Image values must be in [0, 1]. "
            f"Got range [{image_srgb.min():.3f}, {image_srgb.max():.3f}]"
        )
    
    # Step 1: sRGB -> Linear RGB (IEC 61966-2-1)
    image_linear = srgb_to_linear(image_srgb)
    
    # Step 2: Ottieni matrice Machado dalla cache (con interpolazione se severity float)
    matrix = get_cvd_matrix(cvd_type, severity)
    
    # Step 3: Applica matrice CVD in spazio lineare
    image_cvd_linear = apply_cvd_matrix(image_linear, matrix)
    
    # Step 4: Linear RGB -> sRGB (IEC 61966-2-1)
    image_cvd_srgb = linear_to_srgb(image_cvd_linear)
    
    return image_cvd_srgb


# ==============================================================================
# Canonical Public API (severity in [0, 1])
# ==============================================================================

_SIMULATECVD_WARNED_AUTO_PERCENT = False


def _simulatecvd_severity_to_percent(
    severity: float,
    *,
    severity_input_mode: Literal['auto', 'unit', 'percent'] = 'auto',
    strict: bool = False,
) -> float:
    """Convert a public severity value to Machado percent scale [0, 100].

    Public canonical severity is T in [0, 1]. Machado lookup uses [0, 100].

    Modes:
      - unit: expects [0, 1]
      - percent: expects [0, 100]
      - auto: accepts either; in STRICT mode, auto is forbidden

    Returns:
      float severity_percent in [0, 100] (float preserved to keep interpolation).
    """
    global _SIMULATECVD_WARNED_AUTO_PERCENT

    try:
        s = float(severity)
    except Exception as e:
        raise ValueError(f"Invalid severity value: {severity!r}") from e

    if strict and severity_input_mode == 'auto':
        raise ValueError(
            "simulatecvd(strict=True) forbids severity_input_mode='auto'. "
            "Pass severity_input_mode='unit' (T in [0,1]) or 'percent' (0-100)."
        )

    mode = severity_input_mode
    if mode not in ('auto', 'unit', 'percent'):
        raise ValueError(
            f"Invalid severity_input_mode: {mode!r}. Expected 'auto'|'unit'|'percent'."
        )

    if mode == 'unit':
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"severity (unit) out of range [0,1]: {s}")
        return float(np.clip(s * 100.0, 0.0, 100.0))

    if mode == 'percent':
        if not (0.0 <= s <= 100.0):
            raise ValueError(f"severity (percent) out of range [0,100]: {s}")
        return float(np.clip(s, 0.0, 100.0))

    # auto
    if 0.0 <= s <= 1.0:
        return float(np.clip(s * 100.0, 0.0, 100.0))
    if 0.0 <= s <= 100.0:
        if not _SIMULATECVD_WARNED_AUTO_PERCENT:
            _SIMULATECVD_WARNED_AUTO_PERCENT = True
            warnings.warn(
                "simulatecvd(): severity_input_mode='auto' interpreted a value in [0,100] as percent. "
                "For future compatibility, pass severity_input_mode='percent' explicitly (or 'unit' for T in [0,1]).",
                RuntimeWarning,
                stacklevel=3,
            )
        return float(np.clip(s, 0.0, 100.0))

    raise ValueError(
        f"severity out of supported ranges: {s}. Expected [0,1] (unit) or [0,100] (percent)."
    )


def simulatecvd(
    image_srgb: np.ndarray,
    cvd_type: str,
    severity: float,
    *,
    severity_input_mode: Literal['auto', 'unit', 'percent'] = 'auto',
    strict: bool = False,
) -> np.ndarray:
    """Canonical CVD simulation wrapper.

    This is the public API entrypoint.

    - Canonical severity is T in [0, 1] (severity_input_mode='unit').
    - Internally converts to Machado percent scale [0,100] and delegates to
      simulate_cvd_machado() (which supports float interpolation).
    """
    severity_percent = _simulatecvd_severity_to_percent(
        severity, severity_input_mode=severity_input_mode, strict=strict
    )
    return simulate_cvd_machado(image_srgb, cvd_type, severity_percent)


def simulatecvd_batch_torch(
    images_nchw: 'torch.Tensor',
    cvd_type: str,
    severity: float,
    *,
    severity_input_mode: Literal['auto', 'unit', 'percent'] = 'auto',
    strict: bool = False,
) -> 'torch.Tensor':
    """Canonical batch CVD simulation wrapper (torch).

    Delegates to simulate_cvd_machado_batch_torch().
    """
    severity_percent = _simulatecvd_severity_to_percent(
        severity, severity_input_mode=severity_input_mode, strict=strict
    )
    return simulate_cvd_machado_batch_torch(images_nchw, cvd_type, severity_percent)


# ==============================================================================
# Batch GPU Processing (PyTorch)
# ==============================================================================

def simulate_cvd_machado_batch_torch(
    images_nchw: 'torch.Tensor',
    cvd_type: str,
    severity: float
) -> 'torch.Tensor':
    """
    Batch GPU-accelerated CVD simulation usando Machado 2009.
    
    Applica la stessa trasformazione di simulate_cvd_machado() ma su un batch
    di immagini usando PyTorch GPU per massima efficienza.
    
    Pipeline:
        1. (N,C,H,W) -> (N,H,W,C) permute per sRGB->linear conversion
        2. sRGB -> linear RGB (IEC 61966-2-1 element-wise su GPU)
        3. Ottieni matrice Machado 3x3, converti a torch tensor GPU
        4. Reshape (N,H,W,3) -> (N*H*W, 3), matmul @ M.T, reshape back
        5. linear RGB -> sRGB (IEC 61966-2-1 element-wise su GPU)
        6. (N,H,W,C) -> (N,C,H,W) permute per output formato PyTorch
    
    Args:
        images_nchw: Batch immagini in formato PyTorch (N, C, H, W)
                     - C deve essere 3 (RGB)
                     - Valori in [0, 1] sRGB
                     - dtype: torch.float32
                     - device: preferibilmente cuda
        cvd_type: Tipo CVD - uno di {'protan', 'deutan', 'tritan'}
        severity: Severità CVD in [0, 100]
    
    Returns:
        torch.Tensor: Batch immagini CVD simulate (N, C, H, W)
                     - Stesso shape, dtype e device di input
                     - Valori in [0, 1] sRGB
    
    Raises:
        ValueError: Se input shape non valido o cvd_type/severity invalidi
        RuntimeError: Se cache Machado non popolata o PyTorch non disponibile
    
    Examples:
        >>> import torch
        >>> # Batch di 15 immagini 256x256
        >>> images = torch.rand(15, 3, 256, 256, device='cuda')
        >>> 
        >>> # Simula protanopia batch
        >>> images_cvd = simulate_cvd_machado_batch_torch(
        ...     images, 'protan', severity=80.0
        ... )
        >>> images_cvd.shape
        torch.Size([15, 3, 256, 256])
    
    Note:
        - Numericamente equivalente a simulate_cvd_machado() entro tolleranza
          GPU float32 (atol=1e-4, rtol=1e-4)
        - Usa stessa cache Machado (get_cvd_matrix) convertita a torch
        - Stessi standard sRGB IEC 61966-2-1 (formule identiche)
        - Performance: ~10-15x più veloce per batch=15 vs 15 chiamate single
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch not available. Install with: pip install torch"
        )
    
    # Validazione input
    if images_nchw.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor (N,C,H,W), got shape {images_nchw.shape}"
        )
    
    N, C, H, W = images_nchw.shape
    if C != 3:
        raise ValueError(
            f"Expected C=3 (RGB), got C={C}"
        )
    
    device = images_nchw.device
    dtype = images_nchw.dtype
    
    # Step 1: Permute (N,C,H,W) -> (N,H,W,C) per conversioni sRGB
    images_nhwc = images_nchw.permute(0, 2, 3, 1)  # (N, H, W, 3)
    
    # Step 2: sRGB -> linear RGB (IEC 61966-2-1) - element-wise GPU
    # Formula: linear = srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055)^2.4
    linear_nhwc = torch.where(
        images_nhwc <= SRGB_TO_LINEAR_THRESHOLD,
        images_nhwc / 12.92,
        torch.pow((images_nhwc + 0.055) / 1.055, SRGB_GAMMA)
    )
    
    # Step 3: Ottieni matrice Machado dalla cache (numpy) e converti a torch
    matrix_np = get_cvd_matrix(cvd_type, severity)  # (3, 3) numpy float64
    matrix_torch = torch.from_numpy(matrix_np).to(dtype).to(device)  # (3, 3) torch
    
    # Step 4: Applica matrice CVD - reshape, matmul, reshape back
    # (N, H, W, 3) -> (N*H*W, 3) @ (3, 3).T -> (N*H*W, 3) -> (N, H, W, 3)
    linear_flat = linear_nhwc.reshape(-1, 3)  # (N*H*W, 3)
    cvd_flat = torch.matmul(linear_flat, matrix_torch.T)  # (N*H*W, 3) @ (3, 3)
    cvd_nhwc = cvd_flat.reshape(N, H, W, 3)  # (N, H, W, 3)
    
    # Clipping a [0, 1] (alcuni CVD matrix possono dare valori leggermente fuori range)
    cvd_nhwc = torch.clamp(cvd_nhwc, 0.0, 1.0)
    
    # Step 5: linear RGB -> sRGB (IEC 61966-2-1) - element-wise GPU
    # Formula: srgb = 12.92 * linear if linear <= 0.0031308 else 1.055 * linear^(1/2.4) - 0.055
    cvd_srgb_nhwc = torch.where(
        cvd_nhwc <= LINEAR_TO_SRGB_THRESHOLD,
        12.92 * cvd_nhwc,
        1.055 * torch.pow(cvd_nhwc, 1.0 / SRGB_GAMMA) - 0.055
    )
    
    # Step 6: Permute (N,H,W,C) -> (N,C,H,W) per output formato PyTorch
    cvd_srgb_nchw = cvd_srgb_nhwc.permute(0, 3, 1, 2)  # (N, 3, H, W)
    
    return cvd_srgb_nchw


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_simulator_info() -> dict:
    """
    Restituisce informazioni sul simulatore CVD configurato.
    
    Returns:
        dict: Informazioni con chiavi:
            - 'backend': Nome backend ('colour.matrix_cvd_Machado2009')
            - 'cache_stats': Statistiche cache Machado
            - 'srgb_standard': Standard sRGB usato ('IEC 61966-2-1:1999')
            - 'valid_cvd_types': Lista CVD types supportati
            - 'severity_range': Range severity [min, max]
    """
    cache_stats = get_cache_stats()
    
    return {
        'backend': 'colour.matrix_cvd_Machado2009',
        'cache_stats': cache_stats,
        'srgb_standard': 'IEC 61966-2-1:1999',
        'valid_cvd_types': sorted(VALID_CVD_TYPES),
        'severity_range': [0, 100],
        'gamma': SRGB_GAMMA,
        'interpolation': 'linear',
    }


def validate_simulation(
    image_original: np.ndarray,
    image_cvd: np.ndarray,
    severity: float
) -> dict:
    """
    Valida risultato simulazione CVD (sanity checks).
    
    Args:
        image_original: Immagine originale sRGB [0,1]
        image_cvd: Immagine CVD simulata sRGB [0,1]
        severity: Severità CVD usata [0,100]
    
    Returns:
        dict: Risultati validazione:
            - 'shape_match': True se shape identiche
            - 'value_range_ok': True se valori in [0,1]
            - 'mean_difference': Differenza media pixel-wise
            - 'max_difference': Differenza massima pixel-wise
            - 'is_valid': True se tutte le validazioni passano
    """
    shape_match = image_original.shape == image_cvd.shape
    
    value_range_ok = (
        0.0 <= image_cvd.min() <= 1.0 and
        0.0 <= image_cvd.max() <= 1.0
    )
    
    diff = np.abs(image_original - image_cvd)
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())
    
    # Per severity = 0, differenza dovrebbe essere ~0 (identità)
    # Per severity > 0, differenza dovrebbe essere > 0
    expected_identity = (severity == 0.0)
    is_identity = (mean_diff < 1e-5)
    identity_check = (expected_identity == is_identity)
    
    is_valid = shape_match and value_range_ok and identity_check
    
    return {
        'shape_match': shape_match,
        'value_range_ok': value_range_ok,
        'mean_difference': mean_diff,
        'max_difference': max_diff,
        'identity_check': identity_check,
        'is_valid': is_valid,
    }


# ==============================================================================
# Module Info (printed at import if verbose)
# ==============================================================================

if __name__ != "__main__":
    # Print info only on direct import (not on submodule import)
    import sys
    if 'cvd_simulator' in sys.modules and __name__ == 'cvd_simulator':
        info = get_simulator_info()
        print(f"[OK] CVD Simulator ready: {info['backend']} "
              f"(cache: {info['cache_stats']['total_matrices']} matrices)")
