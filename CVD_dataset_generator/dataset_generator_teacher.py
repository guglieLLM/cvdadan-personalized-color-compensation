"""
Dataset Generator con Teacher Farup 2020 (Pipeline Unificata 3D).

Questo modulo implementa il generatore dataset COMPLETO per CVD compensation:
    1. Carica profili 3D sintetici x=(θ, C_index, S_index)
    2. Mapping continuo x -> {cvd_type, severity_T}
    3. CVD simulation con Machado 2009 (colour-science)
    4. Teacher daltonization con Farup 2020 GDIP anisotropic
    5. Salvataggio con metadata JSON esteso + versioning

OUTPUT:
    dataset/synthetic_dataset/
        train/
            {profile_id}_{image_id}.png (compensato)
            {profile_id}_{image_id}_cvd.png (CVD simulato)
        val/
        test/
        metadata.json (schema esteso)

SCHEMA METADATA (per immagine):
    {
        "profile_x": {"theta_deg": float, "C_index": float, "S_index": float},
        "cvd_params": {"cvd_type": str, "severity_T": float},
        "teacher": {
            "algorithm": "farup_full",
            "version": git_commit,
            "config": {...},
            "delta_e_mean": null  # Computed in post-validation (post_validate_deltae.py)
        },
        "files": {
            "compensated": path,
            "cvd_simulated": path,
            "original": path
        },
        "validation_status": "strict_ok" | "acceptable" | "failed"
    }

Note:
    - ΔE thresholds: 3.0 (strict_ok), 5.0 (acceptable), >5.0 (failed)
    - Profile x IMMUTABILE durante retry (solo teacher config varia)
    - Git commit hash per riproducibilità teacher
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
from PIL import Image
from tqdm import tqdm
import hashlib
import sys
import torch

# Aggiungi parent directory al Python path per importare moduli dalla root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# GPU acceleration detection
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    try:
        from teacher_farup_gpu import daltonize_farup_batch_gpu, get_optimal_batch_size, print_gpu_info  # type: ignore
        GPU_BATCH_SIZE = 32
        print("[GPU ENABLED] Farup GDIP will use GPU acceleration")
        print_gpu_info()
        print(f"Batch Size (fixed): {GPU_BATCH_SIZE}")  # Override print_gpu_info display
    except ImportError as e:
        warnings.warn(f"GPU detected but teacher_farup_gpu not available: {e}. Falling back to CPU.")
        USE_GPU = False
        GPU_BATCH_SIZE = 1
else:
    GPU_BATCH_SIZE = 1
    print("[CPU MODE] No CUDA device detected. Using scipy reference implementation.")

# Import moduli pipeline (dalla root del progetto)
# NOTA: simulate_cvd_machado NON è più usato per generazione (FM100-only mode)
# Machado è usato solo per metriche ΔE_CVD in generate_dataset_report.py
from mapping_x_to_T import map_x_to_cvd_params, get_mapping_info, theta_to_confusion_vector_rgb
from teacher_farup_full import daltonize_farup_full, get_teacher_info
from delta_e_ciede2000_torch import delta_e_ciede2000_torch
from lab_rgb_converter import safe_rgb_to_lab  # For ΔE computation
from cvd_constants import (
    DELTA_E_THRESHOLD_STRICT,
    DELTA_E_THRESHOLD_ACCEPTABLE
)

# Import CVD cache management (local module)
from cvd_shared_cache import get_cache_stats

# ==============================================================================
# Default Configuration
# ==============================================================================

# Path assoluto alla root del progetto (già definito sopra per sys.path)
# Allineato con la configurazione dataset di launch_grid.py
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "dataset" / "places365" / "subsets" / "subsets_derived_recolored_CVD" / "subsets_teacher_compensated_t_0.15_v_1"

DEFAULT_DATASET_CONFIG = {
    "output_dir": str(_DEFAULT_DATASET_DIR),
    "profile_source": None,  # Auto-detect: usa output_dir se None
    "image_source": None,  # Path a directory immagini originali (da specificare)
    "splits": ["train", "val", "test"],
    "n_images_per_profile": 20,  # N immagini random per ciascun profilo
    "image_size": (256, 256),   # Resize standardizzato
    "save_cvd_simulated": True,  # Salva CVD intermedio (disabilita con --no-cvd-save)
    "cvd_subdir": "cvd_intermediate",  # Nome base cartella CVD (sarà cvd_intermediate_{split}), None = stessa cartella compensate
    "save_original": False,      # NON salvare copie - abilita con --original-save
    "compute_delta_e": False,    # ΔE calcolato in post-validazione batch (post_validate_deltae.py) - 10-100x più veloce
    "validate_on_generation": True,  # Valida subito (vs post-validate)
    "random_seed": 42,
    "metadata_filename": "metadata.json",
    "teacher_config": None,  # Custom config per teacher (None = default)
}
"""
Configurazione default per generazione dataset.

Note:
    - image_source: DEVE essere specificato (directory con immagini RGB)
    - n_images_per_profile: 20 immagini -> 10500 profili x 20 = 210000 campioni
    - image_size: Resize a (256,256) per uniformità
    - save_cvd_simulated: True per debug/validation
"""

# ==============================================================================
# Split Mapping JSON Writer (for cvd_dataset_loader.py)
# ==============================================================================

def write_split_mapping_json(output_dir: Path, split_name: str, pairs_data: list, teacher_info: dict):
    """
    Write split-specific mapping JSON file required by cvd_dataset_loader.py.
    
    Args:
        output_dir: Base output directory
        split_name: Split name ('train', 'val', 'test')
        pairs_data: List of dicts with keys: image_normal, image_compensated, profile_x, cvd_params
        teacher_info: Teacher algorithm metadata from get_teacher_info()
    """
    mapping_file = output_dir / f"mapping_{split_name}.json"
    
    mapping_data = {
        "meta": {
            "version": "1.0",
            "split": split_name,
            "algorithm": teacher_info["algorithm"],
            "total_pairs": len(pairs_data),
            "creation_date": str(datetime.now().isoformat())
        },
        "pairs": pairs_data
    }
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"[OK] Written {mapping_file.name} with {len(pairs_data)} pairs")


def _save_split_mapping_incremental(
    output_dir: Path,
    split_name: str,
    samples: List[Dict],
    verbose: bool = True
) -> None:
    """
    Salva mapping JSON incrementale dopo ogni split completato.
    
    Questa funzione viene chiamata subito dopo ogni split per garantire
    che anche se il processo viene interrotto, i dati siano salvati.
    
    Args:
        output_dir: Directory output dataset
        split_name: Nome split (train/val/test)
        samples: Lista metadata samples per questo split
        verbose: Print progress
    """
    output_dir = Path(output_dir)
    teacher_info = get_teacher_info()
    
    pairs_data = []
    skipped_count = 0
    
    for sample in samples:
        # Estrai profile_x
        profile_dict = {
            "theta_deg": float(sample["profile_x"]["theta_deg"]),
            "C_index": float(sample["profile_x"]["C_index"]),
            "S_index": float(sample["profile_x"]["S_index"])
        }
        
        # Estrai paths
        original_path = sample["files"].get("original")
        compensated_path = sample["files"].get("compensated")
        
        # Skip se manca original path
        if original_path is None:
            skipped_count += 1
            continue
        
        pair = {
            "image_normal": original_path,
            "image_compensated": compensated_path,
            "profile_x": profile_dict,
            "cvd_params": sample["cvd_params"],
            "severity": sample["cvd_params"]["severity_T"],
            "cvd_type": sample["cvd_params"]["cvd_type"]
        }
        pairs_data.append(pair)
    
    if pairs_data:
        write_split_mapping_json(output_dir, split_name, pairs_data, teacher_info)
        if verbose and skipped_count > 0:
            print(f"  [WARN] Skipped {skipped_count} samples without original path")
    elif verbose:
        print(f"  [WARN] No valid pairs for split '{split_name}' - mapping not saved")


# ==============================================================================
# Image Loading & Preprocessing
# ==============================================================================

def load_and_preprocess_image(
    image_path: Path,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Carica e preprocessa immagine RGB a formato pipeline.
    
    Args:
        image_path: Path a immagine (PNG/JPG)
        target_size: Dimensione target (H, W)
    
    Returns:
        np.ndarray: Immagine (H, W, 3) float32 [0,1] sRGB
    
    Raises:
        IOError: Se caricamento fallisce
        ValueError: Se formato invalido
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {e}")
    
    # Resize
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # To numpy float32 [0,1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Validate shape
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        raise ValueError(
            f"Invalid image shape: {img_array.shape}. Expected (H, W, 3)"
        )
    
    return img_array


def get_random_images(
    image_dir: Path,
    n_images: int,
    random_state: Optional[np.random.RandomState] = None
) -> List[Path]:
    """
    Seleziona n immagini random da directory.
    
    Args:
        image_dir: Directory contenente immagini
        n_images: Numero immagini da selezionare
        random_state: Generator random
    
    Returns:
        List[Path]: Lista di n path immagini
    
    Raises:
        ValueError: Se n_images > immagini disponibili
    """
    image_dir = Path(image_dir)
    
    # Glob immagini (PNG/JPG/JPEG) - RICORSIVO per sottocartelle Places365
    image_files = list(image_dir.glob("**/*.png")) + \
                  list(image_dir.glob("**/*.jpg")) + \
                  list(image_dir.glob("**/*.jpeg"))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    if n_images > len(image_files):
        raise ValueError(
            f"Requested {n_images} images but only {len(image_files)} available"
        )
    
    rng = random_state or np.random.RandomState()
    selected = rng.choice(image_files, size=n_images, replace=False)
    
    return list(selected)


# ==============================================================================
# Single Sample Generation
# ==============================================================================

def generate_single_sample(
    profile_x: Dict[str, float],
    image_original: np.ndarray,
    profile_id: str,
    image_id: str,
    output_dir: Path,
    config: Dict,
    split: str = "train",
    source_image_path: Optional[Path] = None,
    image_category: Optional[str] = None
) -> Dict:
    """
    Genera singolo campione dataset (CVD -> teacher -> salva + metadata).
    
    Pipeline:
        1. Map x -> {cvd_type, severity_T}
        2. Simulate CVD (Machado 2009)
        3. Daltonize (Farup 2020)
        4. Compute ΔE (CVD -> compensato)
        5. Validate (thresholds)
        6. Save images + metadata
    
    Args:
        profile_x: Profilo 3D {"theta_deg", "C_index", "S_index"}
        image_original: Immagine RGB (H,W,3) float32 [0,1]
        profile_id: ID profilo (es. "profile_0042")
        image_id: ID immagine (es. "img_003")
        output_dir: Directory output dataset
        config: Config generazione
        split: Split name ("train"|"val"|"test")
        source_image_path: Path originale dell'immagine sorgente (opzionale)
    
    Returns:
        dict: Metadata campione generato
    
    Raises:
        RuntimeError: Se generazione fallisce
    """
    # Step 1: Mapping x -> CVD params
    cvd_params = map_x_to_cvd_params(profile_x)
    cvd_type = cvd_params['cvd_type']
    severity_T = cvd_params['severity_T']
    is_normal = cvd_params.get('is_normal', False)
    
    # SKIP: Normal profiles (identity teacher, no Farup compensation needed)
    if is_normal or cvd_type == 'normal':
        warnings.warn(
            f"Skipping profile_x={profile_x}: Normal vision profile (cvd_type='normal'). "
            f"Farup+FM100 teacher is only for CVD profiles (protan/deutan/tritan)."
        )
        return None  # Signal to skip this profile
    
    # Step 2: FM100 Confusion Vector (OBBLIGATORIO)
    # Extract confusion axis e_d from FM100 profile θ angle
    theta_deg = profile_x.get("theta_deg", None)
    if theta_deg is None:
        raise RuntimeError(
            f"theta_deg mancante in profile_x={profile_x}. "
            f"Impossibile calcolare confusion_vector. "
            f"Assicurarsi che il profilo FM100 contenga theta_deg."
        )
    
    try:
        # Convert FM100 angle to 3D RGB confusion vector
        confusion_vector_np = theta_to_confusion_vector_rgb(theta_deg)
    except Exception as e:
        raise RuntimeError(
            f"Errore calcolo confusion_vector da theta_deg={theta_deg}: {e}. "
            f"Nessun fallback PCA disponibile."
        )
    
    # Step 3: Teacher Daltonization (GPU-accelerated if available)
    # NOTA: Non usa più image_cvd (Machado). Solo confusion_vector da FM100.
    try:
        teacher_cfg = config.get("teacher_config", None)
        
        if USE_GPU:
            # GPU batch processing (batch_size=1 for single image)
            device = torch.device('cuda')
            img_orig_torch = torch.from_numpy(image_original).unsqueeze(0).to(device)  # (1, H, W, 3)
            
            # Convert confusion vector to torch tensor (1, 3) for batch processing
            confusion_vec_torch = torch.from_numpy(confusion_vector_np).float().unsqueeze(0).to(device)  # (1, 3)
            
            # GPU daltonization (FM100-only, no image_cvd)
            img_comp_torch = daltonize_farup_batch_gpu(
                images=img_orig_torch,
                confusion_vectors=confusion_vec_torch,
                config=teacher_cfg
            )
            
            # Convert back to numpy
            image_compensated = img_comp_torch.squeeze(0).cpu().numpy()
        else:
            # CPU scipy fallback (FM100-only, no image_cvd)
            image_compensated = daltonize_farup_full(
                image_original=image_original,
                confusion_vector=confusion_vector_np,
                config=teacher_cfg
            )
    except Exception as e:
        raise RuntimeError(f"Teacher daltonization failed: {e}")
    
    # Step 4: Compute ΔE - RIMOSSO
    # NOTA: ΔE tra image_cvd e image_compensated non è più calcolabile
    # perché non generiamo più image_cvd (Machado rimosso).
    # La metrica ΔE_CVD va calcolata in post-processing usando:
    #   - generate_dataset_report.py (usa Machado per simulare CVD e calcolare ΔE)
    # Impostiamo delta_e=None per tutti i sample.
    delta_e = None
    
    # Step 5: Determine validation status (only if ΔE was computed)
    validation_status = "unknown"  # Default when compute_delta_e=False
    if delta_e is not None:
        if delta_e <= DELTA_E_THRESHOLD_STRICT:
            validation_status = "strict_ok"
        elif delta_e <= DELTA_E_THRESHOLD_ACCEPTABLE:
            validation_status = "acceptable"
        else:
            validation_status = "failed"
    
    # Step 6: Save images
    # PRESERVA STRUTTURA CATEGORIE PLACES365
    split_dir = output_dir / split
    
    # Se abbiamo la categoria (es. 'abbey'), crea sottocartella
    if image_category:
        category_dir = split_dir / image_category
        category_dir.mkdir(parents=True, exist_ok=True)
    else:
        category_dir = split_dir
        category_dir.mkdir(parents=True, exist_ok=True)
    
    sample_basename = f"{profile_id}_{image_id}"
    
    # Compensated (sempre salvato nella cartella categoria)
    # Usa JPEG se configurato (I/O più veloce)
    use_jpeg = config.get("use_jpeg", False)
    if use_jpeg:
        compensated_filename = f"{sample_basename}.jpg"
        compensated_path = category_dir / compensated_filename
        img_comp_uint8 = (image_compensated * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_comp_uint8).save(compensated_path, quality=95)
    else:
        compensated_filename = f"{sample_basename}.png"
        compensated_path = category_dir / compensated_filename
        img_comp_uint8 = (image_compensated * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_comp_uint8).save(compensated_path)
    
    # CVD simulated - RIMOSSO
    # NOTA: image_cvd non viene più generato (Machado rimosso dalla pipeline di generazione).
    # Se necessario per debug/analisi, usare generate_dataset_report.py che applica Machado
    # alle immagini compensate per calcolare metriche ΔE_CVD.
    cvd_filename = None
    cvd_path = None
    
    # Original (opzionale)
    original_filename = None
    original_path = None
    original_path_str = None
    
    if config["save_original"]:
        # JPEG anche per l'originale (consistente con le altre immagini)
        if use_jpeg:
            original_filename = f"{sample_basename}_original.jpg"
            original_path = split_dir / original_filename
            img_orig_uint8 = (image_original * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_orig_uint8).save(original_path, quality=95)
        else:
            original_filename = f"{sample_basename}_original.png"
            original_path = split_dir / original_filename
            img_orig_uint8 = (image_original * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_orig_uint8).save(original_path)
        original_path_str = str(output_dir.name) + "/" + str(original_path.relative_to(output_dir))
    elif source_image_path:
        # Se non salviamo copia, costruiamo percorso relativo dalla root del progetto
        # Cerchiamo "dataset/" nel path sorgente per costruire il percorso relativo
        source_str = str(source_image_path).replace('\\', '/')
        if '/dataset/' in source_str:
            # Estrai la parte dal "dataset/" in poi
            idx = source_str.index('/dataset/')
            original_path_str = source_str[idx+1:]  # Rimuovi il leading /
        elif 'dataset/' in source_str:
            # Path già relativo
            idx = source_str.index('dataset/')
            original_path_str = source_str[idx:]
        else:
            # Fallback: usa il path assoluto (WARNING: non ideale)
            original_path_str = source_str
            warnings.warn(f"Could not extract relative path from: {source_str}")
    
    # Step 7: Metadata
    teacher_info = get_teacher_info()
    
    # Prepare FM100 profile information for metadata
    fm100_profile = None
    confusion_vector_used = False
    if confusion_vector_np is not None:
        fm100_profile = {
            "theta_deg": profile_x.get("theta_deg", None),
            "confusion_vector_rgb": confusion_vector_np.tolist() if confusion_vector_np is not None else None
        }
        confusion_vector_used = True
    
    metadata = {
        "sample_id": sample_basename,
        "profile_id": profile_id,
        "image_id": image_id,
        "split": split,
        "profile_x": profile_x.copy(),
        "cvd_params": {
            "cvd_type": cvd_type,
            "severity_T": severity_T,
            "mapping_info": cvd_params['mapping_info'],
            "cvd_model": "Machado2009",  # CVD simulation model
        },
        "teacher": {
            "algorithm": teacher_info.get('algorithm', 'unknown'),
            "version": teacher_info.get('version', {'git_commit': 'unknown', 'has_git': False}),
            "config": teacher_info.get('default_config', {}),
            "delta_e_mean": delta_e,
            "gradient_ed_source": "fm100" if confusion_vector_used else "none",  # e_d source (FM100 only)
            "fm100_profile": fm100_profile,  # FM100 confusion axis details
            "confusion_vector_used": confusion_vector_used,  # Boolean flag
            "gdip_params": {  # GDIP solver parameters for reproducibility
                "nit": teacher_info.get('default_config', {}).get('nit', 501),
                "kappa": teacher_info.get('default_config', {}).get('kappa', 1e-2),
                "dt": teacher_info.get('default_config', {}).get('dt', 0.05),
            }
        },
        "files": {
            "compensated": str(compensated_path.relative_to(output_dir)),
            "cvd_simulated": str(cvd_path.relative_to(output_dir)) if cvd_path else None,
            "original": original_path_str,
        },
        "validation_status": validation_status,
        "generation_timestamp": datetime.now().isoformat(),
    }
    
    return metadata


# ==============================================================================
# GPU Batch Processing (per profile)
# ==============================================================================

def generate_profile_batch(
    profile_x: Dict[str, float],
    image_paths: List[Path],
    profile_id: str,
    output_dir: Path,
    config: Dict,
    split: str = "train",
    gpu_batch_size: int = 32
) -> List[Dict]:
    """
    Genera batch di campioni per singolo profilo CVD (GPU-accelerated).
    
    Processa N immagini (tipicamente 20) in un unico batch GPU per ridurre
    overhead di transfer CPU<->GPU e sfruttare parallelismo CVD simulation
    e Farup daltonization.
    
    Pipeline batch:
        1. Load N images -> stack (N, H, W, 3) numpy
        2. Permute -> (N, C, H, W) torch, single .to(device) GPU transfer
        3. CVD simulation batch GPU: simulatecvd_batch_torch()
        4. Daltonization batch GPU: daltonize_farup_batch_gpu()
        5. Transfer GPU->CPU, save N images, generate N metadata
    
    Args:
        profile_x: Profilo 3D {"theta_deg", "C_index", "S_index"}
        image_paths: Lista di N path immagini da processare
        profile_id: ID profilo (es. "profile_0042")
        output_dir: Directory output dataset
        config: Config generazione
        split: Split name ("train"|"val"|"test")
        gpu_batch_size: Batch size massimo (con OOM fallback)
    
    Returns:
        List[Dict]: Lista di N metadata (uno per immagine processata)
    
    Raises:
        RuntimeError: Se batch processing fallisce (fallback a single)
        torch.cuda.OutOfMemoryError: Se GPU OOM (gestito con retry batch size minore)
    
    Note:
        - Target speedup: 1.4-1.6x vs generate_single_sample() loop
        - Memory: ~26MB per batch=32 immagini 256x256 FP32
        - OOM graceful degradation: 32 -> 10 -> 1 (fallback single)
        - Preserva equivalenza numerica: atol=1e-4 RGB, <0.01 ΔE
    """
    import torch
    from cvd_simulator import simulatecvd_batch_torch
    
    if not USE_GPU:
        raise RuntimeError(
            "generate_profile_batch() requires GPU. "
            "Use generate_single_sample() for CPU processing."
        )
    
    device = torch.device('cuda')
    target_size = config["image_size"]
    n_images = len(image_paths)
    
    # Step 1: Batch load images -> (N, H, W, 3) numpy stack
    try:
        images_list = []
        categories = []
        
        for img_path in image_paths:
            img_array = load_and_preprocess_image(img_path, target_size)
            images_list.append(img_array)
            
            # Extract category from path
            img_path_obj = Path(img_path)
            if img_path_obj.parent.name != split:
                categories.append(img_path_obj.parent.name)
            else:
                categories.append(None)
        
        images_batch_np = np.stack(images_list, axis=0)  # (N, H, W, 3)
        
    except Exception as e:
        raise RuntimeError(f"Batch image loading failed: {e}")
    
    # Step 2: Mapping x -> CVD params (same for all images in profile)
    cvd_params = map_x_to_cvd_params(profile_x)
    cvd_type = cvd_params['cvd_type']
    severity_T = cvd_params['severity_T']
    is_normal = cvd_params.get('is_normal', False)
    
    # SKIP: Normal profiles (identity teacher, no Farup compensation needed)
    if is_normal or cvd_type == 'normal':
        warnings.warn(
            f"Skipping profile_x={profile_x}: Normal vision profile (cvd_type='normal'). "
            f"Batch processing is only for CVD profiles (protan/deutan/tritan)."
        )
        return []  # Return empty list to signal skip
    
    if cvd_type is None or severity_T is None:
        raise RuntimeError(
            f"Invalid CVD params from profile_x={profile_x}: "
            f"cvd_type={cvd_type}, severity_T={severity_T}"
        )
    
    # Step 3: Convert numpy -> torch (N,C,H,W), single GPU transfer
    try:
        # (N, H, W, 3) -> (N, 3, H, W) via permute
        images_torch = torch.from_numpy(images_batch_np).permute(0, 3, 1, 2).to(device)
        
    except torch.cuda.OutOfMemoryError:
        # OOM on initial transfer - batch too large, fallback to smaller batch
        raise  # Will be caught by caller for retry
    
    # Step 4: Batch CVD simulation GPU
    try:
        # Canonical API: severity_T in [0,1]
        images_cvd_torch = simulatecvd_batch_torch(
            images_torch,
            cvd_type,
            severity_T,
            severity_input_mode='unit',
            strict=True,
        )  # (N, 3, H, W)
        
    except Exception as e:
        raise RuntimeError(f"Batch CVD simulation failed: {e}")
    
    # Step 4.5: FM100 Confusion Vector (Farup 2020 Integration)
    # Extract confusion axis e_d from FM100 profile for CVD-specific gradient field
    # Replicate for all N images in batch (same profile -> same confusion vector)
    confusion_vectors_torch = None
    try:
        theta_deg = profile_x.get("theta_deg", None)
        if theta_deg is not None:
            # Convert FM100 angle to 3D RGB confusion vector
            confusion_vec_np = theta_to_confusion_vector_rgb(theta_deg)  # (3,)
            # Replicate for batch: (3,) -> (N, 3)
            confusion_vec_np_batch = np.tile(confusion_vec_np, (n_images, 1))  # (N, 3)
            confusion_vectors_torch = torch.from_numpy(confusion_vec_np_batch).float().to(device)  # (N, 3)
    except Exception as e:
        warnings.warn(f"FM100 confusion vector extraction failed: {e}. confusion_vectors will be None.")
        confusion_vectors_torch = None
    
    # Step 5: Batch Farup daltonization GPU
    try:
        teacher_cfg = config.get("teacher_config", None)
        
        # Enhanced teacher config with FM100 confusion vectors
        teacher_cfg_enhanced = teacher_cfg.copy() if teacher_cfg else {}
        if confusion_vectors_torch is not None:
            teacher_cfg_enhanced['confusion_vectors'] = confusion_vectors_torch
            teacher_cfg_enhanced['gradient_ed_source'] = 'fm100'
        else:
            teacher_cfg_enhanced['gradient_ed_source'] = 'none'
        
        # Permute back to (N, H, W, 3) for daltonize_farup_batch_gpu
        images_orig_nhwc = images_torch.permute(0, 2, 3, 1)  # (N, H, W, 3)
        images_cvd_nhwc = images_cvd_torch.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        images_comp_nhwc = daltonize_farup_batch_gpu(
            images=images_orig_nhwc,
            cvd_simulated=images_cvd_nhwc,
            config=teacher_cfg_enhanced
        )  # (N, H, W, 3)
        
    except torch.cuda.OutOfMemoryError:
        # OOM during GDIP - batch too large for nit iterations
        raise  # Will be caught by caller for retry
    except Exception as e:
        raise RuntimeError(f"Batch Farup daltonization failed: {e}")
    
    # Step 6: Transfer GPU -> CPU numpy
    try:
        images_cvd_np = images_cvd_nhwc.cpu().numpy()  # (N, H, W, 3)
        images_comp_np = images_comp_nhwc.cpu().numpy()  # (N, H, W, 3)
        
    except Exception as e:
        raise RuntimeError(f"GPU->CPU transfer failed: {e}")
    
    # Step 7: Save images and generate metadata (per-image loop)
    metadata_list = []
    teacher_info = get_teacher_info()
    
    for i_img in range(n_images):
        image_id = f"img_{i_img:03d}"
        sample_basename = f"{profile_id}_{image_id}"
        
        image_original = images_batch_np[i_img]  # (H, W, 3)
        image_cvd = images_cvd_np[i_img]  # (H, W, 3)
        image_compensated = images_comp_np[i_img]  # (H, W, 3)
        source_image_path = image_paths[i_img]
        image_category = categories[i_img]
        
        # Create category directory
        split_dir = output_dir / split
        if image_category:
            category_dir = split_dir / image_category
            category_dir.mkdir(parents=True, exist_ok=True)
        else:
            category_dir = split_dir
            category_dir.mkdir(parents=True, exist_ok=True)
        
        # Save compensated image
        use_jpeg = config.get("use_jpeg", False)
        if use_jpeg:
            compensated_filename = f"{sample_basename}.jpg"
            compensated_path = category_dir / compensated_filename
            img_comp_uint8 = (image_compensated * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_comp_uint8).save(compensated_path, quality=95)
        else:
            compensated_filename = f"{sample_basename}.png"
            compensated_path = category_dir / compensated_filename
            img_comp_uint8 = (image_compensated * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_comp_uint8).save(compensated_path)
        
        # Save CVD simulated (optional)
        cvd_path = None
        if config["save_cvd_simulated"]:
            cvd_subdir = config.get("cvd_subdir", None)
            if cvd_subdir:
                # Salva fuori da train/val/test: cvd_intermediate_{split}/category/
                cvd_split_dir = output_dir / f"{cvd_subdir}_{split}"
                if image_category:
                    cvd_category_dir = cvd_split_dir / image_category
                else:
                    cvd_category_dir = cvd_split_dir
                cvd_category_dir.mkdir(parents=True, exist_ok=True)
            else:
                cvd_category_dir = category_dir
            
            if use_jpeg:
                cvd_filename = f"{sample_basename}_cvd.jpg"
                cvd_path = cvd_category_dir / cvd_filename
                img_cvd_uint8 = (image_cvd * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_cvd_uint8).save(cvd_path, quality=95)
            else:
                cvd_filename = f"{sample_basename}_cvd.png"
                cvd_path = cvd_category_dir / cvd_filename
                img_cvd_uint8 = (image_cvd * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_cvd_uint8).save(cvd_path)
        
        # Original path handling
        original_path_str = None
        if config["save_original"]:
            if use_jpeg:
                original_filename = f"{sample_basename}_original.jpg"
                original_path = split_dir / original_filename
                img_orig_uint8 = (image_original * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_orig_uint8).save(original_path, quality=95)
            else:
                original_filename = f"{sample_basename}_original.png"
                original_path = split_dir / original_filename
                img_orig_uint8 = (image_original * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_orig_uint8).save(original_path)
            original_path_str = str(original_path.relative_to(output_dir))
        elif source_image_path:
            original_path_str = str(source_image_path.resolve())
        
        # Prepare FM100 profile information for metadata
        fm100_profile = None
        confusion_vector_used = False
        if confusion_vectors_torch is not None:
            confusion_vec_np = confusion_vectors_torch[0].cpu().numpy()  # All same in batch
            fm100_profile = {
                "theta_deg": profile_x.get("theta_deg", None),
                "confusion_vector_rgb": confusion_vec_np.tolist()
            }
            confusion_vector_used = True
        
        # Generate metadata (delta_e=None for batch mode, computed in post-validation)
        metadata = {
            "sample_id": sample_basename,
            "profile_id": profile_id,
            "image_id": image_id,
            "split": split,
            "profile_x": profile_x.copy(),
            "cvd_params": {
                "cvd_type": cvd_type,
                "severity_T": severity_T,
                "mapping_info": cvd_params['mapping_info'],
                "cvd_model": "Machado2009",  # CVD simulation model
            },
            "teacher": {
                "algorithm": teacher_info.get('algorithm', 'unknown'),
                "version": teacher_info.get('version', {'git_commit': 'unknown', 'has_git': False}),
                "config": teacher_info.get('default_config', {}),
                "delta_e_mean": None,  # Computed in post-validation
                "gradient_ed_source": "fm100" if confusion_vector_used else "none",  # e_d source (FM100 only)
                "fm100_profile": fm100_profile,  # FM100 confusion axis details
                "confusion_vector_used": confusion_vector_used,  # Boolean flag
                "gdip_params": {  # GDIP solver parameters for reproducibility
                    "nit": teacher_info.get('default_config', {}).get('nit', 501),
                    "kappa": teacher_info.get('default_config', {}).get('kappa', 1e-2),
                    "dt": teacher_info.get('default_config', {}).get('dt', 0.05),
                }
            },
            "files": {
                "compensated": str(compensated_path.relative_to(output_dir)),
                "cvd_simulated": str(cvd_path.relative_to(output_dir)) if cvd_path else None,
                "original": original_path_str,
            },
            "validation_status": "unknown",  # Computed in post-validation
            "generation_timestamp": datetime.now().isoformat(),
        }
        
        metadata_list.append(metadata)
    
    return metadata_list


# ==============================================================================
# Batch Generation (per split)
# ==============================================================================

def generate_split_dataset(
    profiles_x: np.ndarray,
    image_dir: Path,
    split: str,
    output_dir: Path,
    config: Dict,
    random_state: Optional[np.random.RandomState] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Genera dataset completo per un split (train/val/test).
    
    Args:
        profiles_x: Array (N_profiles, 3) con profili 3D
        image_dir: Directory sorgente immagini originali
        split: Nome split ("train"|"val"|"test")
        output_dir: Directory output dataset
        config: Config generazione
        random_state: Generator random
        verbose: Progress bar
    
    Returns:
        List[Dict]: Lista metadata per ciascun campione generato
    
    Note:
        - Genera n_images_per_profile immagini per ciascun profilo
        - Total samples = N_profiles x n_images_per_profile
        - Salva metadata incrementalmente (ogni 100 samples)
    """
    rng = random_state or np.random.RandomState()
    n_profiles = len(profiles_x)
    n_images_per_profile = config["n_images_per_profile"]
    target_size = config["image_size"]
    
    metadata_list = []
    
    if verbose:
        print(f"\nGenerating {split} split...")
        print(f"  Profiles: {n_profiles}")
        print(f"  Images/profile: {n_images_per_profile}")
        print(f"  Total samples: {n_profiles * n_images_per_profile}")
    
    # Pre-calcola i path di tutte le immagini disponibili
    if verbose:
        print("  Pre-computing available image paths...")
    all_images = list(image_dir.glob("**/*.png")) + list(image_dir.glob("**/*.jpg"))
    if verbose:
        print(f"  Found {len(all_images)} images available for sampling")
    
    # Determine batch mode (default enabled for GPU unless disabled via config)
    use_batch_mode = USE_GPU and not config.get('disable_batch', False)
    gpu_batch_size = config.get('gpu_batch_size', n_images_per_profile)
    
    if verbose:
        if use_batch_mode:
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
            print(f"  [BATCH MODE] GPU enabled, batch_size={gpu_batch_size}")
            print(f"               Device: {device_name}, CUDA: {cuda_version}")
        else:
            # Determine reason for single mode
            if not USE_GPU:
                reason = "GPU not available (torch.cuda.is_available()=False)"
            elif config.get('disable_batch', False):
                reason = "--disable-batch flag set"
            else:
                reason = "Unknown (check USE_GPU and config)"
            print(f"  [SINGLE MODE] Reason: {reason}")
            print(f"                Processing images individually (slower)")
    
    # Incremental save interval (every N profiles)
    INCREMENTAL_SAVE_INTERVAL = 50  # Save JSON every 50 profiles (~1000 samples)
    last_saved_count = 0
    
    # Progress bar
    if verbose:
        iterator = tqdm(
            range(n_profiles), 
            desc=f"Processing {split} profiles"
        )
    else:
        iterator = range(n_profiles)
    
    for i_profile in iterator:
        profile_x_row = profiles_x[i_profile]
        profile_x = {
            'theta_deg': float(profile_x_row[0]),
            'C_index': float(profile_x_row[1]),
            'S_index': float(profile_x_row[2]),
        }
        
        profile_id = f"profile_{i_profile:05d}"
        
        # Seleziona n immagini random per questo profilo (using pre-computed list)
        try:
            if len(all_images) < n_images_per_profile:
                raise ValueError(
                    f"Requested {n_images_per_profile} images but only {len(all_images)} available"
                )
            image_paths = rng.choice(all_images, size=n_images_per_profile, replace=False)
        except ValueError as e:
            warnings.warn(
                f"Skipping profile {profile_id}: {e}", 
                RuntimeWarning
            )
            continue
        
        # Process profile: batch mode (GPU) or single mode (CPU/fallback)
        if use_batch_mode:
            # GPU batch processing with OOM graceful degradation
            current_batch_size = gpu_batch_size
            batch_success = False
            
            while current_batch_size >= 1 and not batch_success:
                try:
                    # Try batch processing
                    profile_metadata = generate_profile_batch(
                        profile_x=profile_x,
                        image_paths=list(image_paths),
                        profile_id=profile_id,
                        output_dir=output_dir,
                        config=config,
                        split=split,
                        gpu_batch_size=current_batch_size
                    )
                    
                    metadata_list.extend(profile_metadata)
                    batch_success = True
                    
                except torch.cuda.OutOfMemoryError:
                    # OOM detected - retry with smaller batch
                    if current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 3)  # 32->10->3->1
                        if verbose:
                            print(f"\n[BATCH MODE] OOM at batch_size={current_batch_size}, "
                                  f"retrying with batch_size={new_batch_size}")
                        current_batch_size = new_batch_size
                        torch.cuda.empty_cache()  # Clear GPU memory
                    else:
                        # batch_size=1 still OOM - fallback to CPU single mode
                        if verbose:
                            print(f"\n[BATCH MODE] OOM even at batch_size=1, "
                                  f"falling back to single mode for profile {profile_id}")
                        batch_success = False
                        break
                
                except Exception as e:
                    # Other error - log and fallback to single mode
                    warnings.warn(
                        f"Batch processing failed for {profile_id}: {e}. "
                        f"Falling back to single mode.",
                        RuntimeWarning
                    )
                    batch_success = False
                    break
            
            # If batch mode failed completely, fallback to single mode for this profile
            if not batch_success:
                use_batch_mode = False  # Disable for remaining profiles
                # Fall through to single mode processing below
        
        # Single mode processing (CPU or batch fallback)
        if not use_batch_mode or (use_batch_mode and not batch_success):
            # Original single-image loop
            for i_img, img_path in enumerate(image_paths):
                image_id = f"img_{i_img:03d}"
                
                # DEBUG: Print compute_delta_e status for first sample only
                if i_profile == 0 and i_img == 0:
                    print(f"[DEBUG] compute_delta_e = {config.get('compute_delta_e', False)}")
                
                try:
                    # Load & preprocess
                    image_original = load_and_preprocess_image(
                        img_path, target_size
                    )
                    
                    # VALIDAZIONE: Verifica che immagine sia stata caricata correttamente
                    if image_original is None:
                        raise RuntimeError(f"load_and_preprocess_image returned None for {img_path}")
                    
                    if not isinstance(image_original, np.ndarray):
                        raise RuntimeError(
                            f"load_and_preprocess_image returned {type(image_original)} instead of np.ndarray"
                        )
                    
                    if image_original.size == 0:
                        raise RuntimeError(f"Loaded image has zero size: {img_path}")
                    
                    # Estrai categoria da path (es. train/abbey/001.jpg -> "abbey")
                    img_path_obj = Path(img_path)
                    # Risali di 1 livello per trovare categoria (parent della cartella categoria)
                    if img_path_obj.parent.name != split:  # Se non è direttamente in train/val/test
                        category = img_path_obj.parent.name
                    else:
                        category = None
                    
                    # Generate sample
                    sample_metadata = generate_single_sample(
                        profile_x=profile_x,
                        image_original=image_original,
                        profile_id=profile_id,
                        image_id=image_id,
                        output_dir=output_dir,
                        config=config,
                        split=split,
                        source_image_path=img_path,
                        image_category=category
                    )
                    
                    # SKIP: If sample_metadata is None (e.g., normal profile skipped)
                    if sample_metadata is None:
                        continue
                    
                    metadata_list.append(sample_metadata)
                    
                except Exception as e:
                    warnings.warn(
                        f"Failed to generate sample {profile_id}_{image_id}: {e}",
                        RuntimeWarning
                    )
                    continue
        
        # ================================================================
        # INCREMENTAL SAVE: Salva mapping JSON ogni N profili
        # Questo protegge da interruzioni durante split lunghi
        # Salva SUBITO al primo profilo, poi ogni INTERVAL
        # ================================================================
        should_save = (i_profile == 0) or ((i_profile + 1) % INCREMENTAL_SAVE_INTERVAL == 0)
        if should_save and len(metadata_list) > last_saved_count:
            _save_split_mapping_incremental(
                output_dir=output_dir,
                split_name=split,
                samples=metadata_list,
                verbose=False  # Silent per non sporcare output
            )
            last_saved_count = len(metadata_list)
            if verbose:
                tqdm.write(f"  [CHECKPOINT] Saved {len(metadata_list)} samples to mapping_{split}.json")
    
    # Final save per questo split (in caso non sia multiplo di INTERVAL)
    if len(metadata_list) > last_saved_count:
        _save_split_mapping_incremental(
            output_dir=output_dir,
            split_name=split,
            samples=metadata_list,
            verbose=verbose
        )
    
    if verbose:
        print(f"  Generated {len(metadata_list)} samples")
    
    return metadata_list


# ==============================================================================
# Full Dataset Generation
# ==============================================================================

def generate_full_dataset(
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Genera dataset COMPLETO con tutti gli splits (train/val/test).
    
    Pipeline:
        1. Carica profili 3D da .npy (generate_synthetic_dataset)
        2. Per ciascun split:
            - Genera N_profiles x n_images_per_profile campioni
            - Salva immagini compensate + CVD + metadata
        3. Aggrega metadata globale
        4. Salva metadata.json finale
    
    Args:
        config: Config custom (default: DEFAULT_DATASET_CONFIG)
        verbose: Progress messages
    
    Returns:
        dict: Global metadata con statistiche generazione
    
    Raises:
        FileNotFoundError: Se profili o immagini mancanti
        ValueError: Se config invalida
    
    Examples:
        >>> config = DEFAULT_DATASET_CONFIG.copy()
        >>> config["image_source"] = "path/to/images"
        >>> metadata = generate_full_dataset(config, verbose=True)
        >>> metadata['total_samples_generated']
        35000
    
    Note:
        - RICHIEDE profili 3D pre-generati (generate_synthetic_dataset.py)
        - RICHIEDE directory immagini originali (config["image_source"])
        - Output: dataset/synthetic_dataset/{train,val,test}/
    """
    # Setup config
    cfg = DEFAULT_DATASET_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    
    # CRITICAL: Popola cache CVD matrices PRIMA di qualsiasi simulazione
    # (La cache è popolata automaticamente all'import di cvd_shared_cache)
    if verbose:
        print("\n[CVD Cache] Verifying Machado 2009 matrices...")
    
    cache_stats = get_cache_stats()
    if not cache_stats['is_complete']:
        raise RuntimeError(
            f"CVD matrix cache incomplete: {cache_stats['total_matrices']}/{cache_stats['expected_size']} matrices. "
            "Ensure colour-science library is installed correctly."
        )
    
    if verbose:
        print(f"[CVD Cache] Populated: {cache_stats['total_matrices']} matrices "
              f"(colour-science {cache_stats['colour_version']})")
    
    # Validate config
    if cfg["image_source"] is None:
        raise ValueError(
            "config['image_source'] must be specified (directory with original images)"
        )
    
    image_source = Path(cfg["image_source"])
    if not image_source.exists():
        raise FileNotFoundError(f"Image source directory not found: {image_source}")
    
    # Auto-detect profile_source: usa output_dir se non specificato
    if cfg["profile_source"] is None:
        profile_source = Path(cfg["output_dir"])
    else:
        profile_source = Path(cfg["profile_source"])
    
    if not profile_source.exists():
        raise FileNotFoundError(f"Profile source directory not found: {profile_source}")
    
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.RandomState(cfg["random_seed"])
    
    # Fast test mode: limita samples totali
    fast_test_limit = cfg.get("_fast_test_total_samples", None)
    if fast_test_limit:
        if verbose:
            print(f"\n[FAST TEST] Limiting to {fast_test_limit} total samples across all splits")
    
    # Load metadata profili
    profile_metadata_path = profile_source / "profiles_metadata.json"
    if not profile_metadata_path.exists():
        raise FileNotFoundError(
            f"Profile metadata not found: {profile_metadata_path}. "
            f"Run generate_synthetic_dataset.py first."
        )
    
    with open(profile_metadata_path, 'r') as f:
        profile_metadata = json.load(f)
    
    if verbose:
        print("=" * 70)
        print("Dataset Generation with Teacher Farup 2020")
        print("=" * 70)
        print(f"Profile source: {profile_source}")
        print(f"Image source: {image_source}")
        print(f"Output: {output_dir}")
        print(f"Splits: {cfg['splits']}")
    
    # Generate per split
    all_metadata = {}
    total_samples = 0
    samples_generated = 0  # Counter per fast test limit
    
    for split in cfg["splits"]:
        # Check fast test limit
        if fast_test_limit and samples_generated >= fast_test_limit:
            if verbose:
                print(f"[FAST TEST] Reached limit of {fast_test_limit} samples, skipping remaining splits")
            break
        
        # Load profili split
        profiles_file = profile_source / f"profiles_x_{split}.npy"
        if not profiles_file.exists():
            warnings.warn(
                f"Skipping split {split}: file not found {profiles_file}",
                UserWarning
            )
            continue
        
        profiles_x = np.load(profiles_file)
        
        # ==================================================================
        # Validazione difensiva: vincoli clinici globali (Vingrys 1988)
        # C_index >= 1.78 (cut-off CVD), S_index >= 1.01 (minimo geometrico)
        # ==================================================================
        C_MIN_CVD = 1.78          # Vingrys 1988 conservative cut-off per difetti congeniti
        S_MIN_GEOMETRIC = 1.01    # Min across all clusters (Protanope S_min=1.01)
        C_values = profiles_x[:, 1]
        S_values = profiles_x[:, 2]
        n_invalid_C = np.sum(C_values < C_MIN_CVD)
        n_invalid_S = np.sum(S_values < S_MIN_GEOMETRIC)
        if n_invalid_C > 0 or n_invalid_S > 0:
            warnings.warn(
                f"[{split}] Profili CVD con valori fuori range clinico: "
                f"{n_invalid_C} con C<{C_MIN_CVD}, {n_invalid_S} con S<{S_MIN_GEOMETRIC}. "
                f"File: {profiles_file}",
                UserWarning
            )
        
        # Fast test: limita numero di profili CON RANDOM SAMPLING
        if fast_test_limit:
            remaining = fast_test_limit - samples_generated
            if remaining <= 0:
                continue
            # Calcola quanti profili servono (considerando n_images_per_profile)
            n_images_per_prof = cfg["n_images_per_profile"]
            max_profiles = max(1, remaining // n_images_per_prof)
            if len(profiles_x) > max_profiles:
                #  RANDOM sampling invece di prendere sempre i primi
                profile_indices = rng.choice(len(profiles_x), size=max_profiles, replace=False)
                profiles_x = profiles_x[profile_indices]
                if verbose:
                    print(f"[FAST TEST] Randomly sampled {max_profiles}/{len(profiles_x)} profiles from {split}")
        
        # Generate split dataset
        split_metadata = generate_split_dataset(
            profiles_x=profiles_x,
            image_dir=image_source,
            split=split,
            output_dir=output_dir,
            config=cfg,
            random_state=rng,
            verbose=verbose
        )
        
        all_metadata[split] = split_metadata
        total_samples += len(split_metadata)
        samples_generated += len(split_metadata)
        
        # ============================================================
        # INCREMENTAL SAVE: Salva mapping JSON dopo ogni split completato
        # Questo garantisce che anche se il processo viene interrotto,
        # i dati degli split completati siano salvati
        # ============================================================
        if len(split_metadata) > 0:
            _save_split_mapping_incremental(
                output_dir=output_dir,
                split_name=split,
                samples=split_metadata,
                verbose=verbose
            )
        
        # Check fast test limit after split
        if fast_test_limit and samples_generated >= fast_test_limit:
            if verbose:
                print(f"[FAST TEST] Reached {samples_generated}/{fast_test_limit} samples")
            break
    
    # Aggregate statistics
    validation_stats = {
        "strict_ok": 0,
        "acceptable": 0,
        "failed": 0,
        "unknown": 0,
    }
    
    delta_e_values = []
    
    for split, samples in all_metadata.items():
        for sample in samples:
            status = sample["validation_status"]
            validation_stats[status] = validation_stats.get(status, 0) + 1
            
            delta_e = sample["teacher"]["delta_e_mean"]
            if delta_e is not None:
                delta_e_values.append(delta_e)
    
    # Global metadata
    # NOTA: simulator_info rimosso - Machado non è più usato in generazione
    teacher_info = get_teacher_info()
    mapping_info = get_mapping_info()
    
    global_metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "random_seed": cfg["random_seed"],
        "total_samples_generated": total_samples,
        "splits": {
            split: len(samples) 
            for split, samples in all_metadata.items()
        },
        "validation_statistics": validation_stats,
        "delta_e_statistics": {
            "mean": float(np.mean(delta_e_values)) if delta_e_values else None,
            "std": float(np.std(delta_e_values)) if delta_e_values else None,
            "median": float(np.median(delta_e_values)) if delta_e_values else None,
            "min": float(np.min(delta_e_values)) if delta_e_values else None,
            "max": float(np.max(delta_e_values)) if delta_e_values else None,
        },
        "pipeline_components": {
            # NOTA: cvd_simulator rimosso - Machado non è più usato in generazione
            # (usato solo per metriche ΔE_CVD in generate_dataset_report.py)
            "teacher": teacher_info,
            "mapping": mapping_info,
            "profile_metadata": profile_metadata,
        },
        "configuration": cfg,
        "samples": all_metadata,  # Metadata completi per split
    }
    
    # Save global metadata
    metadata_file = output_dir / cfg["metadata_filename"]
    with open(metadata_file, 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    # Write split-specific mapping JSON files for training
    # NOTA: I mapping vengono già salvati incrementalmente dopo ogni split
    # Questo blocco finale serve come backup/consolidamento
    teacher_info_dict = get_teacher_info()
    for split_name in cfg["splits"]:
        if split_name not in all_metadata:
            continue
            
        pairs_data = []
        for sample in all_metadata[split_name]:
            # Convert profile_x from dict to required format
            profile_dict = {
                "theta_deg": float(sample["profile_x"]["theta_deg"]),
                "C_index": float(sample["profile_x"]["C_index"]),
                "S_index": float(sample["profile_x"]["S_index"])
            }
            
            # cvd_dataset_loader.py si aspetta: image_normal, image_compensated
            original_path = sample["files"].get("original")
            compensated_path = sample["files"].get("compensated")
            
            # Skip samples without original image (if save_original=False)
            if original_path is None:
                warnings.warn(
                    f"Sample {sample.get('profile_id', 'unknown')} has no original path - skipping from mapping",
                    RuntimeWarning
                )
                continue
                
            pair = {
                "image_normal": original_path,
                "image_compensated": compensated_path,
                "profile_x": profile_dict,
                "cvd_params": sample["cvd_params"],
                "severity": sample["cvd_params"]["severity_T"],
                "cvd_type": sample["cvd_params"]["cvd_type"]
            }
            pairs_data.append(pair)
        
        # Salva mapping finale (sovrascrive quello incrementale con versione completa)
        if pairs_data:
            write_split_mapping_json(output_dir, split_name, pairs_data, teacher_info_dict)
        else:
            warnings.warn(
                f"No valid pairs for split '{split_name}' - mapping file will be empty or not created",
                UserWarning
            )
    
    if verbose:
        print("\n" + "=" * 70)
        print("Generation Complete")
        print("=" * 70)
        print(f"Total samples: {total_samples}")
        print(f"Validation stats: {validation_stats}")
        if delta_e_values:
            print(f"ΔE stats: mean={np.mean(delta_e_values):.2f}, "
                  f"median={np.median(delta_e_values):.2f}, "
                  f"max={np.max(delta_e_values):.2f}")
        print(f"\nMetadata saved: {metadata_file}")
        print(f"Split mapping files: mapping_train.json, mapping_val.json, mapping_test.json")
    
    return global_metadata


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate CVD compensation dataset with Farup 2020 teacher"
    )
    parser.add_argument(
        "--image-source", 
        type=str, 
        required=True,
        help="Path to directory containing original RGB images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_DATASET_DIR),
        help="Output directory for generated dataset"
    )
    parser.add_argument(
        "--profile-source",
        type=str,
        default=None,
        help="Directory containing CVD profiles (profiles_x_*.npy). If not specified, uses --output-dir"
    )
    parser.add_argument(
        "--n-images-per-profile",
        type=int,
        default=20,
        help="Number of images to generate per CVD profile"
    )
    parser.add_argument(
        "--no-cvd-save",
        action="store_true",
        help="Skip saving CVD simulated images (default: save them)"
    )
    parser.add_argument(
        "--original-save",
        action="store_true",
        help="Save copies of original images (default: use absolute paths only)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Fast mode: reduce iterations to 100 (lower quality, 5x faster for testing)"
    )
    parser.add_argument(
        "--fast-test-samples",
        type=int,
        default=None,
        help="Fast test mode: generate only N random samples total (implies --fast-mode, auto-validates)"
    )
    parser.add_argument(
        "--use-jpeg",
        action="store_true",
        default=True,
        help="Save compensated images as JPEG (quality=95) instead of PNG for faster I/O (default: True)"
    )
    parser.add_argument(
        "--disable-batch",
        action="store_true",
        help="Disable GPU batch processing (process images individually, for debugging/testing)"
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=32,
        help="GPU batch size for processing images per profile (default: 32, auto-fallback on OOM)"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_DATASET_CONFIG.copy()
    config["image_source"] = args.image_source
    config["output_dir"] = args.output_dir
    config["profile_source"] = args.profile_source  # Aggiungi profile_source
    config["n_images_per_profile"] = args.n_images_per_profile
    config["save_cvd_simulated"] = not args.no_cvd_save
    config["save_original"] = args.original_save
    config["use_jpeg"] = args.use_jpeg
    config["disable_batch"] = args.disable_batch
    config["gpu_batch_size"] = args.gpu_batch_size
    
    # Fast test mode override
    fast_test_mode = args.fast_test_samples is not None
    if fast_test_mode:
        print(f"[FAST TEST MODE] Generating only {args.fast_test_samples} random samples")
        print("[FAST TEST MODE] Using 200 iterations (fast)")
        print("[FAST TEST MODE] Auto-validation enabled")
        config["teacher_config"] = {"nit": 200}
        # Override: genera solo N samples totali invece di N per profilo
        config["_fast_test_total_samples"] = args.fast_test_samples
    elif args.fast_mode:
        print("[FAST MODE] Using 200 iterations (5x faster, lower quality)")
        config["teacher_config"] = {"nit": 200}
    
    # Generate
    try:
        metadata = generate_full_dataset(config, verbose=True)
        print("\n[OK] Dataset generation successful")
        
        # Auto-validate in fast test mode
        if fast_test_mode:
            print("\n" + "="*70)
            print("RUNNING AUTO-VALIDATION")
            print("="*70)
            
            import sys
            from pathlib import Path
            
            # Import post_validate_deltae
            post_validate_path = Path(__file__).parent / "post_validate_deltae.py"
            if post_validate_path.exists():
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(post_validate_path), config["output_dir"]],
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
                if result.stderr:
                    print("[VALIDATION WARNINGS/ERRORS]")
                    print(result.stderr)
            else:
                print("[WARNING] post_validate_deltae.py not found, skipping validation")
                
    except Exception as e:
        print(f"\n[ERR] Dataset generation failed: {e}")
        raise
