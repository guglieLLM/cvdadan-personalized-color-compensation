"""
Cache pre-computata delle matrici Machado 2009 per simulazione CVD.

Questo modulo gestisce la cache di 303 matrici CVD (3 tipi x 101 livelli di severity)
utilizzando ESCLUSIVAMENTE colour.matrix_cvd_Machado2009 come backend.

REQUISITI IMPERATIVI:
- colour-science >= 0.4.5 è OBBLIGATORIO (no fallback)
- ImportError deve essere BLOCKING (nessun silent fallback a .npy o altre librerie)
- Interpolazione lineare per severity non-interi
- Target cache hit rate: >95%

Riferimenti:
- Machado et al. 2009: "A Physiologically-based Model for Simulation of Color Vision Deficiency"
- Standard sRGB: IEC 61966-2-1:1999
"""

import numpy as np
from typing import Tuple, Optional
import warnings

from cvd_constants import (
    CVD_TYPE_TO_DEFICIENCY,
    VALID_CVD_TYPES,
    MACHADO_CACHE_SEVERITY_LEVELS,
    MACHADO_CACHE_SEVERITY_MIN,
    MACHADO_CACHE_SEVERITY_MAX,
)

# ==============================================================================
# Import colour-science (MANDATORY - no fallback)
# ==============================================================================

try:
    from colour import matrix_cvd_Machado2009
    import colour
    COLOUR_VERSION = colour.__version__
except ImportError as e:
    raise ImportError(
        "colour-science library is MANDATORY for CVD simulation (Machado 2009).\n"
        "This is a BLOCKING error - no fallback is available.\n\n"
        "Installation:\n"
        "  pip install colour-science==0.4.5\n\n"
        "If already installed, verify installation:\n"
        "  python -c \"import colour; print(colour.__version__)\"\n"
    ) from e

# ==============================================================================
# Global Cache (populated at module import)
# ==============================================================================

CVD_MATRIX_CACHE = {}
"""
Cache globale delle matrici Machado 2009.

Struttura:
    {
        ('protan', 0): np.ndarray (3, 3),
        ('protan', 1): np.ndarray (3, 3),
        ...
        ('tritan', 100): np.ndarray (3, 3)
    }

Total entries: 3 CVD types x 101 severity levels = 303 matrices
"""

# ==============================================================================
# Cache Population (executed at import time)
# ==============================================================================

def populate_cache() -> None:
    """
    Pre-computa tutte le 303 matrici Machado 2009 e le salva nella cache globale.
    
    Chiamata automaticamente all'import del modulo.
    Utilizza ESCLUSIVAMENTE colour.matrix_cvd_Machado2009 (no fallback).
    
    Raises:
        ImportError: Se colour-science non disponibile (già gestito all'import)
        RuntimeError: Se la popolazione della cache fallisce
    """
    global CVD_MATRIX_CACHE
    
    try:
        for cvd_type in VALID_CVD_TYPES:
            deficiency = CVD_TYPE_TO_DEFICIENCY[cvd_type]
            
            for severity_int in range(MACHADO_CACHE_SEVERITY_MIN, 
                                     MACHADO_CACHE_SEVERITY_MAX + 1):
                # Severity normalizzata [0, 1] per colour-science
                severity_normalized = severity_int / 100.0
                
                # Ottieni matrice da colour-science (UNICA fonte autorizzata)
                matrix = matrix_cvd_Machado2009(
                    deficiency=deficiency,
                    severity=severity_normalized
                )
                
                # Converti a numpy array e salva nella cache
                CVD_MATRIX_CACHE[(cvd_type, severity_int)] = np.array(matrix, dtype=np.float32)
        
        # Validazione cache popolata
        expected_size = len(VALID_CVD_TYPES) * MACHADO_CACHE_SEVERITY_LEVELS
        actual_size = len(CVD_MATRIX_CACHE)
        
        if actual_size != expected_size:
            raise RuntimeError(
                f"Cache population failed: expected {expected_size} matrices, "
                f"got {actual_size}"
            )
        
        print(f"[OK] CVD Matrix Cache populated: {actual_size} matrices "
              f"(colour-science {COLOUR_VERSION})")
    
    except Exception as e:
        # Se population fallisce, cache rimane vuota -> tutti i get_cvd_matrix falliranno
        CVD_MATRIX_CACHE.clear()
        raise RuntimeError(
            f"Failed to populate CVD matrix cache from colour-science.\n"
            f"Error: {e}\n"
            f"This is a BLOCKING error - pipeline cannot proceed."
        ) from e


# Popola cache all'import del modulo
populate_cache()


# ==============================================================================
# Cache Retrieval (con interpolazione lineare)
# ==============================================================================

def get_cvd_matrix(cvd_type: str, severity: float) -> np.ndarray:
    """
    Ottiene la matrice CVD Machado 2009 dalla cache con interpolazione lineare.
    
    Per severity interi (es. 50.0), restituisce la matrice cached direttamente.
    Per severity non-interi (es. 50.3), interpola linearmente tra floor e ceil.
    
    Args:
        cvd_type: Tipo CVD ('protan', 'deutan', 'tritan')
        severity: Severità CVD in [0, 100] (può essere float)
    
    Returns:
        np.ndarray: Matrice 3x3 per trasformazione RGB lineare
    
    Raises:
        ValueError: Se cvd_type non valido o severity fuori range
        RuntimeError: Se cache non popolata (non dovrebbe mai accadere)
    
    Examples:
        >>> matrix = get_cvd_matrix('protan', 50.0)  # Cache hit diretto
        >>> matrix = get_cvd_matrix('deutan', 50.3)  # Interpolazione 50-51
        >>> matrix.shape
        (3, 3)
    """
    # Validazione input
    if cvd_type not in VALID_CVD_TYPES:
        raise ValueError(
            f"Invalid cvd_type: '{cvd_type}'. "
            f"Valid types: {sorted(VALID_CVD_TYPES)}"
        )
    
    if not (MACHADO_CACHE_SEVERITY_MIN <= severity <= MACHADO_CACHE_SEVERITY_MAX):
        raise ValueError(
            f"Severity {severity} out of range "
            f"[{MACHADO_CACHE_SEVERITY_MIN}, {MACHADO_CACHE_SEVERITY_MAX}]"
        )
    
    # Check cache popolata
    if not CVD_MATRIX_CACHE:
        raise RuntimeError(
            "CVD matrix cache is empty. This should never happen - "
            "cache is populated at module import."
        )
    
    # Caso 1: Severity intero -> cache hit diretto
    if severity == int(severity):
        severity_int = int(severity)
        key = (cvd_type, severity_int)
        
        if key not in CVD_MATRIX_CACHE:
            raise RuntimeError(
                f"Cache miss for integer severity: {key}. "
                f"This indicates cache corruption."
            )
        
        return CVD_MATRIX_CACHE[key].copy()
    
    # Caso 2: Severity float -> interpolazione lineare
    severity_floor = int(np.floor(severity))
    severity_ceil = int(np.ceil(severity))
    
    # Ottieni matrici floor e ceil
    matrix_floor = CVD_MATRIX_CACHE.get((cvd_type, severity_floor))
    matrix_ceil = CVD_MATRIX_CACHE.get((cvd_type, severity_ceil))
    
    if matrix_floor is None or matrix_ceil is None:
        raise RuntimeError(
            f"Cache miss for interpolation: cvd_type={cvd_type}, "
            f"severity_floor={severity_floor}, severity_ceil={severity_ceil}"
        )
    
    # Interpolazione lineare
    alpha = severity - severity_floor  # Peso per ceil (in [0, 1])
    matrix_interp = (1 - alpha) * matrix_floor + alpha * matrix_ceil
    
    return matrix_interp.astype(np.float32)


# ==============================================================================
# Cache Statistics
# ==============================================================================

def get_cache_stats() -> dict:
    """
    Restituisce statistiche sulla cache delle matrici CVD.
    
    Returns:
        dict: Statistiche cache con chiavi:
            - 'total_matrices': Numero totale matrici in cache
            - 'cvd_types': Lista CVD types cached
            - 'severity_range': (min, max) severity
            - 'colour_version': Versione colour-science usata
            - 'expected_size': Dimensione attesa (303)
            - 'is_complete': True se cache completa
    """
    return {
        'total_matrices': len(CVD_MATRIX_CACHE),
        'cvd_types': sorted(VALID_CVD_TYPES),
        'severity_range': (MACHADO_CACHE_SEVERITY_MIN, MACHADO_CACHE_SEVERITY_MAX),
        'colour_version': COLOUR_VERSION,
        'expected_size': len(VALID_CVD_TYPES) * MACHADO_CACHE_SEVERITY_LEVELS,
        'is_complete': len(CVD_MATRIX_CACHE) == (len(VALID_CVD_TYPES) * MACHADO_CACHE_SEVERITY_LEVELS),
    }


# ==============================================================================
# Module-level validation
# ==============================================================================

def _validate_cache_on_import():
    """Validazione cache eseguita all'import del modulo."""
    stats = get_cache_stats()
    
    if not stats['is_complete']:
        warnings.warn(
            f"CVD matrix cache incomplete: {stats['total_matrices']}/{stats['expected_size']} matrices. "
            f"This may cause runtime errors.",
            RuntimeWarning
        )

_validate_cache_on_import()
