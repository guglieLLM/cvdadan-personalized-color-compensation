"""
Teacher Farup 2018/2020 + GDIP 2021 con profili FM100.

CORE ALGORITHM (Farup / Yoshi-II):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Costruzione campo di gradienti target:
    G = ∇u₀ + (∇u₀ · e_d) e_c
    
Dove:
    e_l : asse di luminanza [1,1,1]/√3
    e_d : asse di confusione cromatica (da FM100 θ → confusion_vector)
    e_c : direzione compensazione (ortogonale a e_l ed e_d)
    
Ricostruzione via GDIP anisotropic solver (Farup 2021):
    - Minimizza difference structure tensor: S^Δ = (∇u - G)(∇u - G)ᵀ
    - Evoluzione diffusiva anisotropa: nit=501, kappa=1e-2

FM100/VINGRYS LAYER (OBBLIGATORIO):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- FM100 Hue Test → (θ, C_index, S_index)
- θ → confusion_vector 3D (usato come e_d)
- confusion_vector è OBBLIGATORIO (no fallback PCA)
- Deriva da theta_to_confusion_vector_rgb() in mapping_x_to_T.py

REFERENCES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Farup (2018): "Yoshi-II", Journal of Imaging Science and Technology
- Farup (2020): "Individualised halo-free gradient-domain daltonisation"
- Farup (2021): "Gradient domain image processing" (eq. 40)
- Vingrys & King-Smith (1988): "A quantitative scoring technique for 
  panel tests of color vision", IOVS

WORKFLOW:
    1. Ricevi confusion_vector da profilo FM100 (θ → e_d)
    2. Costruisci G via Farup formula: G = ∇u₀ + (∇u₀ · e_d) e_c
    3. Applica gdip_anisotropic() per ricostruzione
    4. Ritorna immagine compensata [0,1] sRGB

NOTA: Machado 2009 NON è usato qui. È usato solo per metriche di
validazione post-generazione (ΔE_CVD) in generate_dataset_report.py.

Repository ufficiale:
    variational-anisotropic-gradient-domain-main/gradient.py
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional
import subprocess
import warnings

# ==============================================================================
# Import Farup Repository (Official)
# ==============================================================================

# Path al repository Farup nella workspace
FARUP_REPO_PATH = Path(__file__).parent / "variational-anisotropic-gradient-domain-main"

if not FARUP_REPO_PATH.exists():
    raise ImportError(
        f"Farup repository non trovato in {FARUP_REPO_PATH}. "
        f"Directory: variational-anisotropic-gradient-domain-main/"
    )

# Aggiungi al PYTHONPATH
if str(FARUP_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(FARUP_REPO_PATH))

try:    # metto type ignore per evitare warning, viene inserito in runtime
    from gradient import gdip_anisotropic, scale_gradient_linear  # type: ignore
except ImportError as e:
    raise ImportError(
        f"Import fallito del Farup repository. "
        f"Assicurati che gradient.py esiste in {FARUP_REPO_PATH}. "
        f"Errore: {e}"
    )

# ==============================================================================
# Default Configurazione (Farup 2020)
# ==============================================================================

DEFAULT_CONFIG_FARUP_FULL = {
    "nit": 501,                 # Iterazioni (convergenza empirica ~500)
    "kappa": 1e-2,              # Parametro anisotropia (Farup default)
    "diff": "FB",               # Forward-Backward differenze finite
    "isotropic": False,         # Anisotropic (preserva edge/dettagli)
    "linear": True,             # Linearized equation (stabilità)
    "diff_struct": True,        # Difference structure tensor (migliora contrasto)
    "debug": False,             # Stampa progresso ogni 10 iterazioni
    "save": None,               # Nessun salvataggio intermedio (None)
    "save_every": 100,          # Frequenza salvataggio (se save != None)
}
"""
Configurazione default per gdip_anisotropic basata su Farup 2020.

Parametri chiave:
    - nit=501: Convergenza tipica dopo 400-500 iterazioni
    - kappa=1e-2: Anisotropia ottimale per edge preservation
    - diff='FB': Forward-Backward (stabile, no artifacts)
    - linear=True: Linearizza diffusion tensor (faster, stabile)
    - diff_struct=True: Usa difference structure tensor (migliora contrasto)

Riferimento al paper: Farup I., 2020, Section 3.2 "Implementation Details"
"""

# ==============================================================================
# Farup 2018/2020 Gradient Field Construction
# ==============================================================================

def compute_farup_gradient_field(
    u0: np.ndarray,
    confusion_vector: np.ndarray,
    el: Optional[np.ndarray] = None,
    diff: str = "FB"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Costruisce campo di gradienti target G secondo Farup 2018/2020.
    
    CORE ALGORITHM (Farup / Yoshi-II):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Formula:
        G = ∇u₀ + (∇u₀ · e_d) e_c
        
    Dove:
        e_l : asse di luminanza (lightness) = [1,1,1]/√3
        e_d : asse di confusione cromatica (confusion axis) da FM100
        e_c : direzione di compensazione = e_d × e_l (cross product)
        
    La formula aggiunge alla gradiante dell'immagine originale una correzione
    lungo l'asse e_c, proporzionale alla proiezione del gradiente sull'asse
    di confusione e_d. Questo amplifica le differenze cromatiche lungo la
    direzione di compensazione, migliorando la distinguibilità per CVD.
    
    Args:
        u0: Immagine originale (H, W, 3) in RGB lineare [0,1]
        confusion_vector: Vettore di confusione 3D da FM100 (OBBLIGATORIO).
            Deriva da theta_to_confusion_vector_rgb(θ) in mapping_x_to_T.py.
            Se None o vettore nullo, solleva ValueError.
        el: Asse di luminanza 3D (default: [1,1,1]/√3)
        diff: Tipo di differenze finite ('FB', 'C', 'SB')
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Gx, Gy) campi di gradiente (H, W, 3)
        
    Raises:
        ValueError: Se confusion_vector è None, nullo, o shape invalida
        
    References:
        - Farup (2018): "Yoshi-II", J. Imaging Science and Technology
        - Farup (2020): "Individualised halo-free gradient-domain daltonisation"
        - Farup (2021): "Gradient domain image processing", eq. (40)
        - Vingrys & King-Smith (1988): FM-100 Hue Test (per confusion_vector)
    """
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDAZIONE INPUT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if u0.ndim != 3 or u0.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {u0.shape}")
    
    # confusion_vector è OBBLIGATORIO - no fallback PCA
    if confusion_vector is None:
        raise ValueError(
            "confusion_vector è OBBLIGATORIO. "
            "Deve derivare da theta_to_confusion_vector_rgb(θ) in mapping_x_to_T.py. "
            "Nessun fallback PCA disponibile."
        )
    
    if not isinstance(confusion_vector, np.ndarray):
        raise ValueError(
            f"confusion_vector deve essere np.ndarray, ricevuto {type(confusion_vector)}"
        )
    
    if confusion_vector.shape != (3,):
        raise ValueError(
            f"confusion_vector deve avere shape (3,), ricevuto {confusion_vector.shape}"
        )
    
    # Check vettore nullo (indicherebbe errore nel profilo FM100)
    norm_cv = np.linalg.norm(confusion_vector)
    if norm_cv < 1e-10:
        raise ValueError(
            "confusion_vector è nullo (norma < 1e-10). "
            "Profilo FM100 invalido o corrotto."
        )
    
    H, W, C = u0.shape
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: Normalizza e_d (confusion axis) da FM100
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ed = confusion_vector.astype(np.float64).copy()
    ed = ed / norm_cv  # Normalizza
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 2: Definisci e_l (lightness axis)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if el is None:
        el = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    el = el / np.linalg.norm(el)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3: Gram-Schmidt orthogonalization (CRITICAL)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Questo passo è OBBLIGATORIO per proiettare e_d nel sottospazio
    # cromatico opponente, rimuovendo la componente di luminanza.
    # Replica esattamente unit_vectors() in daltonise.py di Farup 2020.
    
    dot_ed_el = float(np.dot(ed, el))
    
    # Degeneracy check: e_d quasi parallelo a e_l
    if abs(dot_ed_el) > 0.99:
        warnings.warn(
            f"Degeneracy detected: |e_d · e_l| = {abs(dot_ed_el):.4f} > 0.99. "
            f"Confusion axis too close to lightness. Fallback to base gradients.",
            UserWarning
        )
        # Fallback: restituisci gradienti originali senza correzione
        # Usa diff='FB' come default se diff=None (scale_gradient_linear si aspetta una stringa)
        diff_method = diff if diff is not None else 'FB'
        gx, gy = scale_gradient_linear(u0, factor=1.0, diff=diff_method)
        return gx, gy
    
    # Gram-Schmidt: rimuovi componente lungo e_l
    ed = ed - dot_ed_el * el
    ed = ed / np.linalg.norm(ed)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 4: Calcola e_c (compensation direction)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ec = np.cross(ed, el)
    ec = ec / np.linalg.norm(ec)
    
    # Ora abbiamo base ortonormale (e_l, e_d, e_c)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 5: Calcola gradienti dell'immagine originale
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # diff dovrebbe essere 'FB' o altro metodo stringa, non intero
    diff_method = diff if diff is not None else 'FB'
    gx, gy = scale_gradient_linear(u0, factor=1.0, diff=diff_method)  # (H, W, 3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 6: Proietta gradienti su e_d (confusion axis)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # sx = g_x · e_d, sy = g_y · e_d
    
    sx = np.einsum('ijk,k->ij', gx, ed)  # (H, W)
    sy = np.einsum('ijk,k->ij', gy, ed)  # (H, W)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 7: Costruisci correzione lungo e_c
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Termine: (∇u₀ · e_d) e_c
    
    cx = sx[:, :, None] * ec[None, None, :]  # (H, W, 3)
    cy = sy[:, :, None] * ec[None, None, :]  # (H, W, 3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 8: Somma per ottenere G = ∇u₀ + (∇u₀ · e_d) e_c
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Gx = gx + cx
    Gy = gy + cy
    
    return Gx, Gy


# ==============================================================================
# Farup Teacher Wrapper
# ==============================================================================

def daltonize_farup_full(
    image_original: np.ndarray,
    confusion_vector: np.ndarray,
    config: Optional[Dict] = None,
    return_gradients: bool = False
) -> np.ndarray:
    """
    Applica daltonizzazione Farup 2018/2020 + GDIP 2021 con profili FM100.
    
    CORE ALGORITHM (Farup / Yoshi-II):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Costruzione campo di gradienti target:
        G = ∇u₀ + (∇u₀ · e_d) e_c
        
    Ricostruzione via GDIP anisotropic solver (Farup 2021):
        - Minimizza difference structure tensor: S^Δ = (∇u - G)(∇u - G)ᵀ
        - Evoluzione diffusiva anisotropa: nit=501, kappa=1e-2
    
    FM100/VINGRYS LAYER (OBBLIGATORIO):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - FM100 Hue Test → (θ, C_index, S_index)
    - θ → confusion_vector 3D (usato come e_d)
    - confusion_vector è OBBLIGATORIO (no fallback PCA)
    
    Pipeline:
        1. Ricevi confusion_vector da profilo FM100
        2. Computa gradienti target G via Farup formula
        3. Applica gdip_anisotropic() per matching gradiente
        4. Ritorna immagine compensata in [0, 1] sRGB
    
    Args:
        image_original: Immagine originale (MxNx3, float32, [0,1])
        confusion_vector: Vettore 3D da FM100 per e_d (OBBLIGATORIO).
            Deriva da theta_to_confusion_vector_rgb(θ) in mapping_x_to_T.py.
        config: Parametri custom (default: DEFAULT_CONFIG_FARUP_FULL)
        return_gradients: Se True, ritorna anche (Gx, Gy) target
    
    Returns:
        np.ndarray: Immagine compensata (MxNx3, float32, [0,1])
        Se return_gradients=True: tuple (image_compensated, Gx, Gy)
    
    Raises:
        ValueError: Se confusion_vector è None/invalido o image fuori range
        RuntimeError: Se gdip_anisotropic fallisce
    
    References:
        - Farup (2018): "Yoshi-II", JIST
        - Farup (2020): "Individualised halo-free gradient-domain daltonisation"
        - Farup (2021): "Gradient domain image processing" (eq. 40)
        - Vingrys & King-Smith (1988): FM-100 Hue Test
    
    Note:
        - Algoritmo BLOCCO (501 iterazioni, ~2-5s per 512x512)
        - Output clipped automaticamente a [0,1]
        - Machado 2009 NON è usato qui (è solo per metriche ΔE_CVD)
    """
    
    # Validazione image_original
    if image_original.ndim != 3 or image_original.shape[2] != 3:
        raise ValueError(
            f"Immagine prevista (M, N, 3), ottenuta {image_original.shape}"
        )
    
    # Se il valore dell'immagine originale è out range allora lancia un warning ed effettua un clipping
    if not (0.0 <= image_original.min() and image_original.max() <= 1.0):
        warnings.warn(
            f"image_original values out of [0,1]: [{image_original.min()}, {image_original.max()}]. "
            f"Clipping to [0,1].",
            UserWarning
        )
        image_original = np.clip(image_original, 0.0, 1.0)
    
    
    # Setup config preso dall'implementazione ufficiale
    cfg = DEFAULT_CONFIG_FARUP_FULL.copy()
    
    if config is not None:
        cfg.update(config)
    
    # CHECK: Identity teacher for normal vision profiles
    # Se confusion_vector è nullo, ritorna immagine originale (no compensazione)
    if confusion_vector is not None and np.allclose(confusion_vector, 0):
        # Normal vision: return original image (no Farup compensation)
        return image_original.copy()
    
    # Step 1: Computa gradienti target G via Farup 2018/2020 formula
    # confusion_vector da FM100 è OBBLIGATORIO - no fallback PCA
    Gx, Gy = compute_farup_gradient_field(
        u0=image_original,
        confusion_vector=confusion_vector,
        el=None,  # Default: [1,1,1]/√3
        diff=cfg["diff"]
    )
    
    # Step 2: Inizializza con immagine originale
    u0 = image_original.copy()
    
    # Step 3: Applica gdip_anisotropic
    try:
        image_compensated = gdip_anisotropic(
            u0=u0,
            vx=Gx,  # Usa campo G costruito con Farup formula
            vy=Gy,  # Usa campo G costruito con Farup formula
            nit=cfg["nit"],
            kappa=cfg["kappa"],
            diff=cfg["diff"],
            save=cfg["save"],
            save_every=cfg["save_every"],
            isotropic=cfg["isotropic"],
            debug=cfg["debug"],
            linear=cfg["linear"],
            diff_struct=cfg["diff_struct"],
        )
    except Exception as e:
        raise RuntimeError(
            f"gdip_anisotropic fallimento: {e}"
        )
    
    # Step 4: Clip finale (gdip già clippa, ma garantiamo per sicurezza)
    image_compensated = np.clip(image_compensated, 0.0, 1.0).astype(np.float32)
    
    if return_gradients:
        return image_compensated, Gx, Gy
    else:
        return image_compensated


# ==============================================================================
# Elaborazione dei BATCH
# ==============================================================================

def daltonize_batch_farup(
    images_original: np.ndarray,
    confusion_vectors: np.ndarray,
    config: Optional[Dict] = None,
    verbose: bool = False
) -> np.ndarray:
    
    """
    Applica daltonizzazione Farup a batch di immagini con profili FM100.
    
    Args:
        images_original: Batch originale (BxMxNx3, float32, [0,1])
        confusion_vectors: Batch di vettori di confusione (Bx3, float32).
            Ogni vettore deriva da theta_to_confusion_vector_rgb(θ).
        config: Parametri custom
        verbose: Se True, mostra progress bar
    
    Returns:
        np.ndarray: Batch compensato (BxMxNx3, float32, [0,1])
    
    Raises:
        ValueError: Se confusion_vectors è None o shape non corrisponde
    
    Examples:
        >>> batch_orig = np.random.rand(10, 100, 100, 3).astype(np.float32)
        >>> confusion_vecs = np.random.rand(10, 3).astype(np.float32)
        >>> batch_comp = daltonize_batch_farup(batch_orig, confusion_vecs)
        >>> batch_comp.shape
        (10, 100, 100, 3)
    
    Note:
        - Processing sequenziale (no parallelismo)
        - Progress bar opzionale (tqdm)
        - confusion_vectors è OBBLIGATORIO (no fallback)
    """
    
    if images_original.ndim != 4:
        raise ValueError(f"Valore atteso 4D batch (B, M, N, 3), ottenuto {images_original.ndim}D")
    
    if confusion_vectors is None:
        raise ValueError(
            "confusion_vectors è OBBLIGATORIO. "
            "Deve essere array (B, 3) da theta_to_confusion_vector_rgb()."
        )
    
    if confusion_vectors.ndim != 2 or confusion_vectors.shape[1] != 3:
        raise ValueError(
            f"confusion_vectors deve avere shape (B, 3), ottenuto {confusion_vectors.shape}"
        )
    
    batch_size = images_original.shape[0]
    
    if confusion_vectors.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: images={batch_size}, confusion_vectors={confusion_vectors.shape[0]}"
        )
    
    results = []
    
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(batch_size), desc="Daltonizzazione")
        except ImportError:
            iterator = range(batch_size)
            print(f"Elaborazione delle immagini {batch_size} ...")
    else:
        iterator = range(batch_size)
    
    
    for i in iterator:
        
        img_comp = daltonize_farup_full(
            image_original=images_original[i],
            confusion_vector=confusion_vectors[i],
            config=config
        )
        
        results.append(img_comp)
    
    return np.stack(results, axis=0)


# ==============================================================================
# Git ---> Versione del tracciamento
# ==============================================================================

def get_farup_repo_version() -> Dict:
    """
    Estrae versione Farup repo (git commit hash se disponibile).
    
    Returns:
        dict: Info repo con chiavi:
            - 'repo_path': str
            - 'git_commit': str (hash commit o 'unknown')
            - 'git_dirty': bool (modifiche non committed)
            - 'has_git': bool
    
    Examples:
        --> info = get_farup_repo_version()
        --> 'git_commit' in info
        True
    
    Note:
        - Richiede git installato e repo con .git/
        - Se git non disponibile, ritorna 'unknown'
    """
    repo_path = FARUP_REPO_PATH
    
    # Check se .git/ esiste
    git_dir = repo_path / ".git"
    has_git = git_dir.exists()
    
    if not has_git:
        return {
            'repo_path': str(repo_path),
            'git_commit': 'unknown',
            'git_dirty': False,
            'has_git': False,
        }
    
    # Estrai commit hash
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        commit_hash = 'unknown'
    
    # Check se modifiche uncommitted
    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        git_dirty = len(status) > 0
    except Exception:
        git_dirty = False
    
    return {
        'repo_path': str(repo_path),
        'git_commit': commit_hash,
        'git_dirty': git_dirty,
        'has_git': True,
    }


# ==============================================================================
# Modulo per le informazioni
# ==============================================================================

def get_teacher_info() -> Dict:
    """
    Restituisce informazioni su teacher Farup configurato.
    
    Returns:
        dict: Info con chiavi:
            - 'teacher_name': str ('farup_full')
            - 'algorithm': str ('gdip_anisotropic')
            - 'default_config': dict
            - 'version': dict (git info)
            - 'citation': str
    """
    return {
        'teacher_name': 'farup_full',
        'algorithm': 'gdip_anisotropic',
        'default_config': DEFAULT_CONFIG_FARUP_FULL.copy(),
        'version': get_farup_repo_version(),
        'citation': (
            "Farup, I. (2020). Gradient-based image recoloring for dichromats. "
            "Journal of Imaging Science and Technology, 64(5), 050402."
        ),
    }


# ==============================================================================
# Validazione
# ==============================================================================

def validate_farup_output(
    image_compensated: np.ndarray,
    image_original: np.ndarray,
    max_delta_e: float = 10.0
) -> Dict:
    """
    Valida output teacher Farup.
    
    Args:
        image_compensated: Output teacher (MxNx3)
        image_original: Input original (MxNx3)
        max_delta_e: Threshold ΔE massimo atteso
    
    Returns:
        dict: Risultati validazione:
            - 'shape_valid': bool
            - 'range_valid': bool
            - 'is_valid': bool
            - 'warnings': list
    
    Note:
        - ΔE validation opzionale (richiede delta_e_ciede2000_torch.py)
        - Shape e range sempre validati
    """
    warnings_list = []
    
    # Check 1: Shape
    shape_valid = (image_compensated.shape == image_original.shape)
    if not shape_valid:
        warnings_list.append(
            f"Shape mismatch: comp={image_compensated.shape}, "
            f"orig={image_original.shape}"
        )
    
    # Check 2: Range [0,1]
    range_valid = (
        0.0 <= image_compensated.min() and image_compensated.max() <= 1.0
    )
    if not range_valid:
        warnings_list.append(
            f"Immagine compensata out of range [0,1]: "
            f"[{image_compensated.min()}, {image_compensated.max()}]"
        )
    
    is_valid = shape_valid and range_valid
    
    return {
        'shape_valid': shape_valid,
        'range_valid': range_valid,
        'is_valid': is_valid,
        'warnings': warnings_list,
    }


if __name__ == "__main__":
    
    # Test rapido
    print("Testing teacher_farup_full (FM100-only mode)...")
    
    # Info teacher
    info = get_teacher_info()
    print(f"Teacher: {info['teacher_name']}")
    print(f"Algorithm: {info['algorithm']}")
    
    # Test su immagine random con confusion_vector
    print("\nTest daltonizzazione su immagine random...")
    img_orig = np.random.rand(64, 64, 3).astype(np.float32)
    
    # Simula confusion_vector da FM100 (es. protanopia, θ ≈ 0°)
    # In produzione deriva da: theta_to_confusion_vector_rgb(θ)
    confusion_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Protan axis
    confusion_vector = confusion_vector / np.linalg.norm(confusion_vector)
    
    print(f"confusion_vector (e_d): {confusion_vector}")
    
    # Daltonize (senza image_cvd!)
    img_comp = daltonize_farup_full(
        image_original=img_orig,
        confusion_vector=confusion_vector
    )
    
    print(f"Input originale: shape={img_orig.shape}, range=[{img_orig.min():.3f}, {img_orig.max():.3f}]")
    print(f"Output compensato: shape={img_comp.shape}, range=[{img_comp.min():.3f}, {img_comp.max():.3f}]")
    
    # Validazione
    validation = validate_farup_output(img_comp, img_orig)
    if validation['is_valid']:
        print("\n[OK] Validazione successo!")
    else:
        print("\n[ERR] Validazione fallita:")
        for w in validation['warnings']:
            print(f"  - {w}")
    
    # Test errore: confusion_vector None
    print("\nTest errore: confusion_vector=None (deve fallire)...")
    try:
        daltonize_farup_full(img_orig, None)  # type: ignore
        print("[ERR] Avrebbe dovuto sollevare ValueError!")
    except ValueError as e:
        print(f"[OK] ValueError correttamente sollevato: {str(e)[:60]}...")
    
    print("\n[OK] Modulo teacher caricato con successo (FM100-only).")
