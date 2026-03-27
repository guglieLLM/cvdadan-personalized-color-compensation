"""
Mapping clinico x=(θ, C_index, S_index) -> {cvd_type, severity_T}.

Questo modulo implementa il mapping tra lo spazio clinico 3D dei profili CVD
(basato su Vingrys & King-Smith 1988) e i parametri del simulatore Machado 2009.

IMPORTANTE - NATURA MODELLISTICA:
Questo mapping NON è una calibrazione clinica validata, ma una modellizzazione
esplicita per scopi di ricerca. I parametri sono scelti per coprire lo spazio
CVD in modo uniforme, non per replicare diagnosi cliniche FM-100/D-15.

WORKFLOW:
    1. Confusion angle θ -> CVD type (protan/deutan/tritan) via range continui
    2. C_index (confusion index) -> severity_T via funzione lineare parametrizzata
    3. S_index (selectivity index) -> usato per validazione, non nel mapping base

Riferimenti:
- Vingrys A.J., King-Smith P.E., "A quantitative scoring technique for panel 
  tests of color vision", IOVS, 1988
- Machado et al. 2009: severity_T in [0, 1] per matrici CVD
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

from cvd_constants import VALID_CVD_TYPES

# Tolleranza per errori di precisione floating-point
EPSILON = 1e-6

# ==============================================================================
# Confusion Angle Ranges (Vingrys & King-Smith 1988)
# ==============================================================================

DEFAULT_THETA_RANGES = {
    # Ranges basati su letteratura FM-100 (Vingrys & King-Smith 1988)
    # UPDATED: Support negative angles via 360° normalization + multi-interval ranges
    # 
    # Ogni tipo CVD può avere MULTIPLI intervalli clinici distinti (anomalie + anopie):
    # - Formato: lista di tuple (min, max) per rappresentare unioni di intervalli
    # - Negative angles normalizzati a [0, 360]: -5° → 355°, -81° → 279°
    # 
    # Clinical CVD types and their confusion axes:
    "protan": [(8.0, 11.0)],                          # Protanomale/Protanope: θ ∈ [8°, 11°] (red-green, L-cone)
    "deutan": [(351.0, 353.0), (355.0, 357.0)],       # Deuteranope [-9°,-7°] → [351°,353°] + Deuteranomale [-5°,-3°] → [355°,357°]
    "tritan": [(279.0, 281.0), (319.0, 321.0)],       # Tritanope [-81°,-79°] → [279°,281°] + Tritanomale [-41°,-39°] → [319°,321°]
}
"""
Range di confusion angle (gradi) per ciascun tipo CVD.

Basati su Vingrys & King-Smith 1988 e letteratura FM-100 Hue Test.

IMPORTANT: Multi-interval structure per rappresentare cluster clinici distinti:
  - Protan: [8°, 11°] (singolo intervallo, copre Protanomale/Protanope)
  - Deutan: UNIONE di [351°, 353°] (Deuteranope) E [355°, 357°] (Deuteranomale)
  - Tritan: UNIONE di [279°, 281°] (Tritanope) E [319°, 321°] (Tritanomale)

Questo evita range continui troppo larghi (es. [279°, 321°] coprirebbe 42° fuori dagli assi clinici).

Note:
- Valori approssimativi per modeling, non diagnosi clinica
- Overlap gestiti con priorità nella funzione theta_to_cvd_type()
"""

# ==============================================================================
# C_index -> severity_T Mapping Parameters
# ==============================================================================

DEFAULT_MAPPING_PARAMS = {
    "type": "linear_C",
    "slope": 0.25,       # CORRETTO: Mappa C_index [0, 4.0] -> T [0, 1]
    "intercept": 0.0,
    "C_min": 0.0,        # C_index minimo (visione normale)
    "C_max": 4.0,        # C_index massimo REALE nei profili clinici (era 11.5, mai raggiunto)
    "T_min": 0.0,        # severity_T minimo (no CVD)
    "T_max": 1.0,        # severity_T massimo (CVD completo)
}

"""
Parametri per mapping lineare C_index -> severity_T.

Formula: severity_T = slope * C_index + intercept
Clipping: severity_T in [T_min, T_max] = [0, 1]

Calibrazione (CORRETTA 2025-01-25):
- I profili clinici reali hanno C_index nel range [1.78, 3.5]
- Il valore teorico C_max=11.5 NON viene mai raggiunto nei dati reali
- Con slope=0.087 (vecchio), C_index=3.5 -> T=0.30 (solo 30% severità!)
- Con slope=0.25 (nuovo), C_index=4.0 -> T=1.0 (range completo)

Nuovo mapping:
- slope = 0.25 = 1/4.0
- C_index = 0 -> T = 0 (visione normale)
- C_index = 4.0 -> T = 1.0 (dichromacy completa)
- C_index = 3.5 -> T = 0.875 (87.5% severità, realistico per profili estremi)

[NOTA]: Questi parametri sono MODELLISTICI, non clinicamente validati.

"""

# ==============================================================================
# Theta -> CVD Type Mapping (con gestione overlap)
# ==============================================================================

def theta_to_cvd_type(
    theta_deg: float,
    c_index: Optional[float] = None,
    theta_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> str:
    """
    Mappa confusion angle θ (gradi) al tipo CVD via range continui.
    
    UPDATED: Supports negative angles via 360° normalization + Normal vision (θ=0°).
    Clinical ranges (Vingrys 1988):
      - Normale:       θ = 0° (no confusion axis, C < 1.78)
      - Protanomale:   [8°, 11°]
      - Deuteranomale: [-5°, -3°]   = [355°, 357°] after normalization
      - Tritanope:     [-81°, -79°] = [279°, 281°] after normalization
    
    Args:
        theta_deg: Confusion angle in degrees [0, 360] (negative angles normalized in caller)
        c_index: Confusion index (optional, used to validate Normal vision with θ=0°)
        theta_ranges: Range custom (default: DEFAULT_THETA_RANGES)
    
    Returns:
        str: CVD type ('protan', 'deutan', 'tritan')
    
    Eccezioni:
        ValueError: Se theta_deg fuori range [0, 360]
    
    Esempi:
        >>> theta_to_cvd_type(0.0, c_index=1.2)   # Normale (θ=0°, C<1.78)
        'protan'  # Default for Normal (severity≈0 from C_index)
        >>> theta_to_cvd_type(10.0)   # Protanomale
        'protan'
        >>> theta_to_cvd_type(356.0)  # Deuteranomale (-4° normalized)
        'deutan'
        >>> theta_to_cvd_type(280.0)  # Tritanope (-80° normalized)
        'tritan'
    
    Note:
        - θ=0° indicates Normal vision: defaults to 'protan' with severity≈0
        - Angoli negativi devono essere normalizzati dal chiamante (map_x_to_cvd_params)
        - Range non-overlapping (nessuna priorità necessaria)
    """
    
    # SPECIAL CASE: θ ≈ 0° indicates Normal vision (no confusion axis)
    # Vingrys 1988: Normal subjects have θ=0°, C<1.78, no CVD simulation needed
    # Return 'normal' for identity teacher (no Farup daltonization)
    if abs(theta_deg) < 1e-3:  # θ ≈ 0° (tolerance for floating-point)
        # Normal vision: cvd_type='normal', dataset generator will use identity teacher
        return 'normal'
    
    # Validate theta in [0, 360] range
    if not (0.0 <= theta_deg <= 360.0):
        raise ValueError(
            f"Confusion angle must be in [0, 360] degrees (after normalization), got {theta_deg}"
        )
    
    ranges = theta_ranges or DEFAULT_THETA_RANGES
    
    # Check each CVD type - ora supporta liste di intervalli
    for cvd_type in ['protan', 'deutan', 'tritan']:
        intervals = ranges[cvd_type]
        # Supporta sia formato vecchio (singola tupla) che nuovo (lista di tuple)
        if isinstance(intervals[0], (int, float)):
            # Formato vecchio: (min, max) -> converti a [(min, max)]
            intervals = [intervals]
        
        # Controlla se theta_deg cade in ALMENO UNO degli intervalli per questo tipo
        for range_min, range_max in intervals:
            if range_min <= theta_deg <= range_max:
                return cvd_type
    
    # Fallback: theta in gap between ranges -> assign to nearest
    warnings.warn(
        f"Confusion angle {theta_deg}° not in standard ranges. "
        f"Assigning to nearest CVD type.",
        UserWarning
    )
    
    # Compute distance to center of each range (usa il primo intervallo per semplicità)
    distances = {}
    for cvd_type, intervals in ranges.items():
        if isinstance(intervals[0], (int, float)):
            intervals = [intervals]
        # Usa il centro del primo intervallo come riferimento
        range_min, range_max = intervals[0]
        center = (range_min + range_max) / 2
        distances[cvd_type] = abs(theta_deg - center)
    
    return min(distances, key=distances.get)


# ==============================================================================
# C_index -> severity_T Mapping (lineare parametrizzato)
# ==============================================================================

def c_index_to_severity(
    c_index: float,
    mapping_params: Optional[Dict] = None
) -> float:
    """
    Mappa C_index (confusion index) a severity_T via funzione lineare.
    
    Formula:
        severity_T = slope * C_index + intercept
        severity_T = clip(severity_T, T_min, T_max)
    
    Args:
        c_index: Confusion index (tipicamente [0, 4] nei profili clinici reali)
        mapping_params: Parametri custom (default: DEFAULT_MAPPING_PARAMS)
    
    Returns:
        float: Severity T in [0, 1] per Machado 2009
    
    Examples:
        >>> c_index_to_severity(0.0)      # Visione normale
        0.0
        >>> c_index_to_severity(2.0)      # Anomalia moderata
        0.5
        >>> c_index_to_severity(4.0)      # Dichromacy
        1.0
        >>> c_index_to_severity(5.0)      # Out-of-range -> clipped
        1.0
    
    Note:
        - Mapping MODELLISTICO (non clinicamente validato)
        - Clipping automatico a [0, 1]
        - Parametri salvati nei metadata JSON per tracciabilità
    """
    params = mapping_params or DEFAULT_MAPPING_PARAMS
    
    slope = params["slope"]
    intercept = params["intercept"]
    T_min = params["T_min"]
    T_max = params["T_max"]
    
    # Mapping lineare
    severity_T = slope * c_index + intercept
    
    # Clip a range valido Machado
    severity_T = np.clip(severity_T, T_min, T_max)
    
    return float(severity_T)


# ==============================================================================
# Mapping Completo x -> {cvd_type, severity_T}
# ==============================================================================

def map_x_to_cvd_params(
    x: Dict[str, float],
    theta_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    mapping_params: Optional[Dict] = None
) -> Dict:
    """
    Mappa profilo clinico 3D x a parametri simulatore Machado 2009.
    
    Pipeline:
        1. θ (confusion angle) -> cvd_type via theta_to_cvd_type()
        2. C_index -> severity_T via c_index_to_severity()
        3. S_index -> salvato per validazione (non usato nel mapping base)
    
    Args:
        x: Profilo 3D con chiavi:
            - 'theta_deg': Confusion angle [0, 180] gradi
            - 'C_index': Confusion index [0, ~15]
            - 'S_index': Selectivity index [1, ~5]
        theta_ranges: Range custom per θ->type
        mapping_params: Parametri custom per C->T
    
    Returns:
        dict: Parametri CVD con chiavi:
            - 'cvd_type': str ('protan'|'deutan'|'tritan')
            - 'severity_T': float [0, 1]
            - 'x_original': dict (copia di x per tracciabilità)
            - 'mapping_info': dict (parametri usati)
    
    Raises:
        ValueError: Se x manca chiavi richieste o valori invalidi
        KeyError: Se x non contiene 'theta_deg', 'C_index', 'S_index'
    
    Examples:
        --> x = {'theta_deg': 95.0, 'C_index': 5.75, 'S_index': 2.3}
        --> params = map_x_to_cvd_params(x)
        --> params['cvd_type']
        'protan'
        --> params['severity_T']
        0.5
        --> params['x_original']
        {'theta_deg': 95.0, 'C_index': 5.75, 'S_index': 2.3}
    
    Note:
        - Mapping ESPLICITO e MODELLISTICO (non clinico)
        - Parametri salvabili in metadata JSON per riproducibilità
        - S_index incluso per future estensioni (anisotropic mapping)
    """
    # Validazione input
    required_keys = {'theta_deg', 'C_index', 'S_index'}
    if not required_keys.issubset(x.keys()):
        missing = required_keys - set(x.keys())
        raise KeyError(
            f"Profile x missing required keys: {missing}. "
            f"Required: {required_keys}"
        )
    
    theta_deg = x['theta_deg']
    c_index = x['C_index']
    s_index = x['S_index']
    
    # Validazione range con tolleranza floating-point
    # EPSILON serve SOLO per assorbire errori numerici (es. S=0.9992 ≈ 1.0 da numpy save/load)
    # NON altera i vincoli clinici: i range scientifici (C≥1.78 Vingrys, S≈1.8 Veselý)
    # sono garantiti dal generatore profili, questi check intercettano solo valori impossibili
    EPSILON = 1e-6  # Floating-point tolerance, NON clinicamente rilevante (~0.0001%)
    
    # Normalizza angoli theta negativi (Deuteranomale/Tritanope usano range negativi)
    # Range clinici da Vingrys 1988:
    #   - Deuteranomale: θ ∈ [-5°, -3°]  (negative red-green)
    #   - Tritanope:     θ ∈ [-81°, -79°] (negative blue-yellow)
    # Convert to [0, 360] range for processing
    if theta_deg < 0:
        theta_deg = theta_deg + 360.0  # -5° -> 355°, -81° -> 279°
    
    # Validate normalized theta in [0, 360]
    if not (0.0 <= theta_deg <= 360.0):
        raise ValueError(f"theta_deg must be in [0, 360] (or negative equivalent), got {x['theta_deg']}")
    
    if c_index < -EPSILON:
        raise ValueError(f"C_index must be >= 0, got {c_index}")
    
    if s_index < 1.0 - EPSILON:
        raise ValueError(f"S_index must be >= 1, got {s_index}")
    
    # Step 1: θ -> cvd_type (pass C_index for Normal vision validation)
    cvd_type = theta_to_cvd_type(theta_deg, c_index, theta_ranges)
    
    # Step 2: C_index -> severity_T
    # SPECIAL: For normal vision (cvd_type='normal'), set severity=0.0
    if cvd_type == 'normal':
        severity_T = 0.0
        is_normal = True
    else:
        severity_T = c_index_to_severity(c_index, mapping_params)
        is_normal = False
    
    # Step 3: Costruisci output con metadata
    params = mapping_params or DEFAULT_MAPPING_PARAMS
    ranges = theta_ranges or DEFAULT_THETA_RANGES
    
    result = {
        'cvd_type': cvd_type,
        'severity_T': severity_T,
        'is_normal': is_normal,
        'x_original': x.copy(),
        'mapping_info': {
            'type': params['type'],
            'parameters': params.copy(),
            'theta_ranges_used': ranges.copy(),
            'S_index_unused': s_index,  # Salvato ma non usato nel mapping base
        }
    }
    
    return result


# ==============================================================================
# Validazione Mapping
# ==============================================================================

def validate_mapping(
    x: Dict[str, float],
    expected_cvd_type: Optional[str] = None,
    expected_severity_range: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Valida risultato mapping x -> CVD params (sanity checks).
    
    Args:
        x: Profilo 3D originale
        expected_cvd_type: CVD type atteso (opzionale)
        expected_severity_range: Range severity atteso (T_min, T_max)
    
    Returns:
        dict: Risultati validazione:
            - 'cvd_type_match': bool (se expected_cvd_type fornito)
            - 'severity_in_range': bool
            - 'is_valid': bool (tutte le validazioni passate)
            - 'warnings': list di warning messages
    """
    result = map_x_to_cvd_params(x)
    
    cvd_type = result['cvd_type']
    severity_T = result['severity_T']
    
    warnings_list = []
    
    # Check 1: CVD type atteso
    cvd_type_match = True
    if expected_cvd_type is not None:
        cvd_type_match = (cvd_type == expected_cvd_type)
        if not cvd_type_match:
            warnings_list.append(
                f"CVD type mismatch: expected {expected_cvd_type}, got {cvd_type}"
            )
    
    # Check 2: Severity in range
    severity_in_range = (0.0 <= severity_T <= 1.0)
    if not severity_in_range:
        warnings_list.append(
            f"Severity out of [0, 1]: {severity_T}"
        )
    
    # Check 3: Severity range custom
    if expected_severity_range is not None:
        s_min, s_max = expected_severity_range
        in_custom_range = (s_min <= severity_T <= s_max)
        if not in_custom_range:
            warnings_list.append(
                f"Severity {severity_T} not in expected range [{s_min}, {s_max}]"
            )
    
    is_valid = cvd_type_match and severity_in_range and (len(warnings_list) == 0)
    
    return {
        'cvd_type_match': cvd_type_match,
        'severity_in_range': severity_in_range,
        'is_valid': is_valid,
        'warnings': warnings_list,
        'mapped_result': result,
    }


# ==============================================================================
# Utility: Batch Mapping
# ==============================================================================

def map_x_batch(
    x_batch: np.ndarray,
    theta_ranges: Optional[Dict] = None,
    mapping_params: Optional[Dict] = None
) -> list:
    """
    Applica mapping a batch di profili 3D.
    
    Args:
        x_batch: Array (N, 3) con colonne [theta_deg, C_index, S_index]
        theta_ranges: Range custom θ->type
        mapping_params: Parametri custom C->T
    
    Returns:
        list: Lista di N dict con risultati map_x_to_cvd_params()
    
    Examples:
        --> x_batch = np.array([
        ...     [95.0, 5.75, 2.3],   # Protan moderato
        ...     [65.0, 8.2, 1.8],    # Deutan severo
        ...     [175.0, 3.1, 3.5],   # Tritan lieve
        ... ])
        --> results = map_x_batch(x_batch)
        --> len(results)
        3
        --> results[0]['cvd_type']
        'protan'
    """
    if x_batch.ndim != 2 or x_batch.shape[1] != 3:
        raise ValueError(
            f"x_batch must have shape (N, 3), got {x_batch.shape}"
        )
    
    results = []
    for i, x_row in enumerate(x_batch):
        x_dict = {
            'theta_deg': float(x_row[0]),
            'C_index': float(x_row[1]),
            'S_index': float(x_row[2]),
        }
        
        try:
            result = map_x_to_cvd_params(x_dict, theta_ranges, mapping_params)
            results.append(result)
        except Exception as e:
            # Log error e continua con batch
            warnings.warn(
                f"Mapping failed for row {i}: {x_dict}. Error: {e}",
                RuntimeWarning
            )
            results.append(None)
    
    return results


# ==============================================================================
# Confusion Vector Generation (GEOMETRIC CIELUV METHOD - TRITAN FIX)
# ==============================================================================

def theta_to_confusion_vector_rgb(theta_deg: float) -> np.ndarray:
    """
    Converte confusion angle θ (FM100) in vettore 3D RGB Lineare.
    
     FIX 2025-11-27: Sostituita interpolazione circolare buggy con metodo
    geometrico CIELUV. Risolve failure rate 20.6% per Tritanope (θ=280°).
    
    METODO GEOMETRICO (Due Punti):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    θ definisce direzione nel piano CIELUV u*v* (Vingrys 1988).
    1. P1 = grigio medio (L=50, u*=0, v*=0)
    2. P2 = P1 + ε·(cos(θ), sin(θ)) in u*v*
    3. Converti P1, P2 in RGB Lineare via LUV→XYZ→RGB
    4. confusion_vec = normalize(P2 - P1)
    
     IMPORTANTE: Output in RGB LINEARE (non gamma-corrected).
    Farup opera in spazio lineare.
    
    NON fa parte dell'algoritmo Farup 2018/2020/2021 core.
    È un'estensione per personalizzazione clinica che sostituisce la PCA
    nell'estrazione dell'asse di confusione e_d.
    
    Args:
        theta_deg: Confusion angle in degrees [0, 360] o negativi
                   (θ da FM100 Hue Test, Vingrys 1988)
    
    Returns:
        np.ndarray: Vettore (3,) normalizzato in spazio RGB Lineare
                    rappresentante l'asse di confusione cromatica
    
    Examples:
        >>> theta_to_confusion_vector_rgb(9.5)    # Protan: R-G dominante
        >>> theta_to_confusion_vector_rgb(356.0)  # Deutan: G dominante
        >>> theta_to_confusion_vector_rgb(280.0)  # Tritan: B dominante (FIXED!)
    
    Note:
        - Per Normal vision (θ≈0°), restituisce [1,1,1]/√3
        - Usa conversione geometrica CIELUV→RGB Lineare via color_space_utils
    
    References:
        - Vingrys A.J., King-Smith P.E. (1988): FM-100 Hue Test
        - Farup (2018/2020): Gradient-domain daltonisation
    """
    # Import lazy per evitare dipendenze circolari
    from color_space_utils import theta_to_confusion_vector_luv
    
    # Delega alla nuova implementazione geometrica
    return theta_to_confusion_vector_luv(theta_deg)


# ==============================================================================
# Module Info
# ==============================================================================

def get_mapping_info() -> Dict:
    """
    Restituisce informazioni sul mapping configurato.
    
    Returns:
        dict: Info con chiavi:
            - 'mapping_type': str ('linear_C')
            - 'default_params': dict (parametri C->T)
            - 'theta_ranges': dict (range θ->type)
            - 'disclaimer': str (natura modellistica)
    """
    return {
        'mapping_type': DEFAULT_MAPPING_PARAMS['type'],
        'default_params': DEFAULT_MAPPING_PARAMS.copy(),
        'theta_ranges': DEFAULT_THETA_RANGES.copy(),
        'disclaimer': (
            "MODELING ONLY - NOT CLINICALLY VALIDATED. "
            "This mapping is designed for research purposes to uniformly "
            "cover CVD parameter space, not to replicate clinical diagnoses."
        ),
    }


if __name__ == "__main__":
    # Test rapido
    print("Testing mapping_x_to_T...")
    
    test_profiles = [
        {'theta_deg': 95.0, 'C_index': 0.0, 'S_index': 1.0},      # Normale -> T=0
        {'theta_deg': 95.0, 'C_index': 2.0, 'S_index': 2.3},      # Protan moderato -> T=0.5
        {'theta_deg': 65.0, 'C_index': 3.5, 'S_index': 1.5},      # Deutan severo -> T=0.875
        {'theta_deg': 175.0, 'C_index': 2.8, 'S_index': 3.2},     # Tritan moderato -> T=0.7
    ]
    
    for x in test_profiles:
        result = map_x_to_cvd_params(x)
        print(f"x={x} -> type={result['cvd_type']}, T={result['severity_T']:.3f}")
    
    print("\n[OK] Mapping module loaded successfully")
