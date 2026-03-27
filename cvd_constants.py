"""
Costanti condivise per la pipeline CVD compensation.

Questo modulo centralizza tutte le costanti utilizzate nella pipeline di compensazione CVD,
garantendo coerenza tra i vari moduli e facilitando la manutenzione.

Riferimenti:
- ΔE2000 thresholds: Standard "Just Noticeable Difference" (JND) in color science
- CVD type mapping: Allineamento con colour.matrix_cvd_Machado2009()
- sRGB standard: IEC 61966-2-1:1999
"""

# ==============================================================================
# ΔE2000 Quality Thresholds
# ==============================================================================

DELTA_E_THRESHOLD_STRICT = 3.0
"""
Soglia ΔE2000 per qualità "strict_ok".
Basata su JND (Just Noticeable Difference) standard in color science.
Immagini con mean ΔE ≤ 3.0 sono considerate eccellenti.
"""

DELTA_E_THRESHOLD_ACCEPTABLE = 5.0
"""
Soglia ΔE2000 per qualità "recovered" dopo retry.
Immagini con 3.0 < mean ΔE ≤ 5.0 sono accettabili ma non ottimali.
Immagini con mean ΔE > 5.0 sono marcate come "failed".
"""

# ==============================================================================
# CVD Type Mapping (per Machado 2009)
# ==============================================================================

CVD_TYPE_TO_DEFICIENCY = {
    "protan": "Protanomaly",
    "deutan": "Deuteranomaly",
    "tritan": "Tritanomaly",
}
"""
Mapping tra cvd_type interno (pipeline) e deficiency string (colour-science).

La pipeline usa identificatori brevi ('protan', 'deutan', 'tritan') mentre
colour.matrix_cvd_Machado2009() richiede stringhe complete ('Protanomaly', etc.).

Usage:
    deficiency = CVD_TYPE_TO_DEFICIENCY[cvd_type]
    matrix = matrix_cvd_Machado2009(deficiency=deficiency, severity=0.8)
"""

VALID_CVD_TYPES = frozenset(CVD_TYPE_TO_DEFICIENCY.keys())
"""Set di cvd_type validi: {'protan', 'deutan', 'tritan'}"""

# ==============================================================================
# CVD Type Theta Bounds (per normalizzazione per-CVD - Opzione C)
# ==============================================================================

CVD_TYPE_THETA_BOUNDS = {
    # Range theta_deg dal dataset sintetico (generate_synthetic_dataset.py)
    # Basati su Vingrys & King-Smith 1988
    "protan": (8.0, 11.0),      # Protanomale [9,11] + Protanope [8,10] → unione [8,11]
    "deutan": (-9.0, -3.0),     # Deuteranope [-9,-7] + Deuteranomale [-5,-3] → unione [-9,-3]
    "tritan": (-81.0, -39.0),   # Tritanope [-81,-79] + Tritanomale [-41,-39] → unione [-81,-39]
}
"""
Range di theta_deg (gradi) per ciascun tipo CVD nel dataset sintetico.

Usati per:
1. Classificare il tipo CVD da theta_deg raw
2. Normalizzazione per-CVD (Opzione C): ogni tipo ha mean/std separati

Fonte: generate_synthetic_dataset.py (RANGES dict)
"""


def classify_cvd_type_from_theta(theta_deg: float) -> str:
    """
    Classifica il tipo CVD in base al valore raw di theta_deg.
    
    Soglie basate sui range del dataset sintetico:
    - Protan: θ > 5° (range reale: 8° a 11°)
    - Deutan: -30° < θ ≤ 5° (range reale: -9° a -3°)
    - Tritan: θ ≤ -30° (range reale: -81° a -39°)
    
    Args:
        theta_deg: Angolo di confusione in gradi (valore RAW, non normalizzato)
    
    Returns:
        str: Tipo CVD ('protan', 'deutan', 'tritan')
    
    Examples:
        >>> classify_cvd_type_from_theta(9.5)
        'protan'
        >>> classify_cvd_type_from_theta(-6.0)
        'deutan'
        >>> classify_cvd_type_from_theta(-80.0)
        'tritan'
    """
    if theta_deg > 5.0:
        return "protan"
    elif theta_deg > -30.0:
        return "deutan"
    else:
        return "tritan"


# ==============================================================================
# Profilo CVD 3D con Normalizzazione Ibrida
# ==============================================================================

PROFILE_DIM = 3
"""
Dimensione del vettore profilo CVD.

Struttura vettore 3D:
    [θ_norm, C_norm, S_norm]
    
    - θ_norm: Normalizzazione GLOBALE (preserva distinzione Protan/Deutan/Tritan)
    - C_norm: Normalizzazione PER-CVD TYPE 
    - S_norm: Normalizzazione PER-CVD TYPE

Normalizzazione Ibrida:
    - θ (confusion angle): stats GLOBALI su tutto il dataset
      → Protan (+8°/+11°), Deutan (-9°/-3°), Tritan (-81°/-39°) restano DISTINTI
    - C_index e S_index: stats PER-TIPO CVD
      → Evita outlier (es. S_index Tritan molto più alto)

Vantaggi:
    - θ_norm preserva le differenze naturali tra tipi CVD (gap ~15° tra P/D)
    - Non serve one-hot encoding (θ già discrimina i tipi)
    - C/S normalizzati per-tipo evitano dominanza di outlier
"""

# Alias per retrocompatibilità
CVD_DIM_OPTION_D = PROFILE_DIM


# NOTA: Le funzioni get_cvd_one_hot() e get_cvd_one_hot_from_theta() sono state rimosse.
# Con normalizzazione ibrida (θ globale), il one-hot encoding non è più necessario
# perché θ_norm già discrimina naturalmente i tipi CVD:
#   - Protan: θ_norm > 0.8
#   - Deutan: θ_norm ~ 0.3-0.5  
#   - Tritan: θ_norm < -0.5


# ==============================================================================
# IEC 61966-2-1 sRGB Standard Thresholds
# ==============================================================================

SRGB_TO_LINEAR_THRESHOLD = 0.04045
"""
Soglia per conversione sRGB -> linear RGB (inverse companding).
Valori sRGB ≤ 0.04045 usano divisione lineare, > 0.04045 usano gamma 2.4.
Standard: IEC 61966-2-1:1999
"""

LINEAR_TO_SRGB_THRESHOLD = 0.0031308
"""
Soglia per conversione linear RGB -> sRGB (companding).
Valori linear ≤ 0.0031308 usano moltiplicazione, > 0.0031308 usano gamma 1/2.4.
Standard: IEC 61966-2-1:1999
"""

SRGB_GAMMA = 2.4
"""
Gamma standard per conversioni sRGB secondo IEC 61966-2-1.
Usato sia in companding (esponente 1/2.4) che inverse companding (esponente 2.4).
"""

# ==============================================================================
# Machado 2009 Cache Configuration
# ==============================================================================

MACHADO_CACHE_SEVERITY_LEVELS = 101
"""
Numero di livelli di severity pre-computati nella cache (0-100 inclusi).
Total matrices cached: 3 CVD types x 101 severities = 303 matrices.
"""

MACHADO_CACHE_SEVERITY_MIN = 0
"""Severity minima nella cache (0 = visione normale)"""

MACHADO_CACHE_SEVERITY_MAX = 100
"""Severity massima nella cache (100 = CVD completo)"""

# ==============================================================================
# Validazione Costanti (eseguita all'import)
# ==============================================================================

def _validate_constants():
    """Validazione interna delle costanti (chiamata all'import del modulo)."""
    assert DELTA_E_THRESHOLD_STRICT > 0, "DELTA_E_THRESHOLD_STRICT deve essere > 0"
    assert DELTA_E_THRESHOLD_ACCEPTABLE > DELTA_E_THRESHOLD_STRICT, \
        "DELTA_E_THRESHOLD_ACCEPTABLE deve essere > DELTA_E_THRESHOLD_STRICT"
    
    assert len(CVD_TYPE_TO_DEFICIENCY) == 3, "Devono esserci esattamente 3 CVD types"
    assert all(deficiency.endswith("anomaly") for deficiency in CVD_TYPE_TO_DEFICIENCY.values()), \
        "Tutti i deficiency devono terminare con 'anomaly' (Machado 2009)"
    
    assert 0 < SRGB_TO_LINEAR_THRESHOLD < 1, "Soglia sRGB->linear deve essere in (0,1)"
    assert 0 < LINEAR_TO_SRGB_THRESHOLD < 1, "Soglia linear->sRGB deve essere in (0,1)"
    assert SRGB_GAMMA > 0, "Gamma sRGB deve essere > 0"
    
    assert MACHADO_CACHE_SEVERITY_LEVELS == 101, "Cache deve avere 101 livelli (0-100)"
    assert MACHADO_CACHE_SEVERITY_MIN == 0, "Severity minima deve essere 0"
    assert MACHADO_CACHE_SEVERITY_MAX == 100, "Severity massima deve essere 100"

# Esegui validazione all'import
_validate_constants()
