"""
CVD Compensation Training Script
================================

Addestra il modello CVDCompensationModelAdaIN per compensazione cromatica
personalizzata per soggetti con deficit della visione dei colori (CVD).

Architettura:
    - Encoder: ConvNeXt-Tiny (ImageNet-1k) con 17 CVDAdaIN (stem congelato,
      resto fine-tuned con lr dedicato)
    - Decoder: PLCF con 9 CVDAdaIN + tanh head
    - Condizionamento: profilo CVD 3D [θ, C, S] via CVDAdaIN
    - Y'-Preserving: 2 ch ΔCbCr, luma Y' (BT.601) copiata dall'input

Loss (config_01_no_delta_e, configurazione effettivamente usata):
    - MSE a*b* (α=0.7): errore crominanza in CIELAB
    - MS-SSIM  (γ=0.3): struttura multi-scala in sRGB
    - ΔE2000 calcolata come metrica di validazione, NON nella loss

Usage:
    python train.py --config path/to/config.yaml
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════════
# CUDA HIGH PRIORITY STREAM - Vantaggio competitivo su GPU condivisa
# ═══════════════════════════════════════════════════════════════════════════════
def set_high_priority_cuda():
    """
    Imposta uno stream CUDA a priorità massima per tutto il training.
    Quando la GPU è condivisa, i kernel con priorità alta vengono schedulati prima.
    """
    if torch.cuda.is_available():
        try:
            # Recupera il range di priorità supportato dalla GPU
            # Di solito: least_prio=0 (bassa), greatest_prio=-1 (alta)
            least_prio, greatest_prio = torch.cuda.Stream.priority_range()
            
            # Crea uno stream con la MASSIMA priorità possibile
            high_prio_stream = torch.cuda.Stream(priority=greatest_prio)
            
            # Imposta questo stream come quello corrente per tutto PyTorch
            torch.cuda.set_stream(high_prio_stream)
            
            print(f" CUDA Stream impostato su PRIORITÀ MASSIMA (priority={greatest_prio})")
            print(f"   Range disponibile: [{greatest_prio}, {least_prio}] (più basso = più prioritario)")
            return high_prio_stream
        except Exception as e:
            print(f"  Impossibile impostare priorità CUDA: {e}")
    return None

# Attiva subito lo stream ad alta priorità
_high_priority_stream = set_high_priority_cuda()
import torch.nn.functional as F
import random
import argparse
import json
import yaml
import math
import time
import warnings
import multiprocessing
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.amp import autocast

# Architettura CVDCompensationModelAdaIN con CVDAdaIN integrato
from CVDCompensationModelAdaIN import CVDCompensationModelAdaIN, create_cvd_model_large
from losses import CVDLoss, setup_nan_logger
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from cvd_dataset_loader import create_cvd_dataloaders, get_optimal_num_workers
from simple_logger import SimpleTrainingLogger
from train_utility import (
    get_cpu_info, get_gpu_info, get_current_lr, gradient_clipping,
    load_full_checkpoint, rotate_checkpoints, safe_float_convert,
    check_loss_safety, check_for_nan_inf, load_encoder_only_from_checkpoint,
    save_checkpoint, ImagePreProcessedDataset, print_training_config
)
from precision_utils import (
    detect_device_and_precision, reset_precision_cache,
    optimize_cuda_memory, compile_model, is_cuda_12_or_newer, supports_torch_compile
)
from color_space_utils import rgb_to_lab_torch


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config robustly.

    - Detects obvious binary files (e.g., accidentally passing a .pth checkpoint)
    - Tries a few common encodings when UTF-8 decoding fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if config_file.suffix.lower() in {".pth", ".pt", ".ckpt", ".pkl", ".pickle", ".bin"}:
        raise ValueError(
            f"Config must be a YAML text file, but got '{config_file.suffix}'. "
            f"Did you accidentally pass a checkpoint/model file? Path: {config_file}"
        )

    raw = config_file.read_bytes()
    head = raw[:2048]
    if b"\x00" in head:
        raise ValueError(
            "Config file appears to be binary (contains NUL bytes). "
            f"Expected a YAML text file: {config_file}"
        )

    # Heuristic: lots of control chars usually means it's not a text YAML.
    ctrl = sum(1 for b in head if b < 9 or (13 < b < 32))
    if head and (ctrl / len(head)) > 0.05:
        raise ValueError(
            "Config file does not look like plain text (too many control bytes). "
            f"Expected a YAML text file: {config_file}"
        )

    decode_attempts = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_decode_error: Exception | None = None
    for encoding in decode_attempts:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as e:
            last_decode_error = e
            continue

        try:
            cfg = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {config_file}\n{e}") from e

        if cfg is None:
            return {}
        if not isinstance(cfg, dict):
            raise ValueError(
                f"YAML config must be a mapping/dict at the top level, got {type(cfg).__name__}: {config_file}"
            )
        return cfg

    raise ValueError(
        "Unable to decode config file as text. Expected UTF-8 YAML (or similar). "
        f"File: {config_file}\nLast decode error: {last_decode_error}"
    )
from delta_e_ciede2000_torch import delta_e_ciede2000_torch as deltaE00
from metrics import SSIMMetric, PSNRMetric

# Costanti per la normalizzazione ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ===== DEVICE AND PRECISION SETUP =====
device, amp_dtype, precision_key = detect_device_and_precision()


def train_cvd_model(config):
    """
    CVD Training: Machado CVD Simulations -> Normal Vision Color Correction
    
    Addestra il modello a trasformare immagini CVD-distorte in versioni 
    percettivamente corrette per la visione normale, usando il dataset Machado CVD
    con profili clinici 3D [theta_deg, C_index, S_index].
    
    Features:
    - CVD dataset loader con profili 3D
    - CVDLoss con variance normalization + severity weighting + edge-aware weighting
    - Gradient clipping robusto e vanishing gradient detection
    - Mixed precision training (fp16/bf16)
    - Resume da checkpoint
    - Metrics: ΔE2000, SSIM(L*), PSNR
    
    Config requirements:
    - dataset_path: path al dataset CVD
    - batch_size: tipicamente 16-32
    - epochs: tipicamente 80
    - learning_rate: 1e-4 ContentEncoder, 1e-3 Decoder
    
    Returns:
        dict: training results con metriche finali
    """
    
    # ===== SETUP INIZIALE =====
    seed = config.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    # HYBRID CONFIG: Performance + Stability
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # ===== DEVICE & PRECISION DETECTION =====
    # Returns 3 values: device, amp_dtype (bfloat16 or None), precision_key ("bf16" or "fp32")
    device, amp_dtype, precision_key = detect_device_and_precision(config)
    
    # CUDA 12.x Memory Optimizations
    if device.type == 'cuda':
        optimize_cuda_memory(verbose=True)
    
    # AMP (Automatic Mixed Precision) flag - True when using bf16
    # NOTE: GradScaler NOT needed for bf16 (wide dynamic range prevents underflow)
    use_amp = (amp_dtype is not None and device.type == 'cuda')
    
    try:
        keep_last_k = int(config.get("checkpoint_keep_last_k", 3) or 3)
    except Exception:
        keep_last_k = 3
    keep_last_k = max(3, keep_last_k)
    
    # Setup directory output
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoints directory (can be separate from output_dir)
    checkpoints_dir = Path(config.get("checkpoints_dir", output_dir / "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation metrics
    from metrics import SSIMMetric, PSNRMetric
    from delta_e_ciede2000_torch import delta_e_ciede2000_torch
    
    criterion_ssim = SSIMMetric().to(device)
    criterion_psnr = PSNRMetric(max_val=1.0).to(device)
    
    print(f"[INFO] Validation metrics initialized:")
    print(f"  - Primary: ΔE2000 (perceptual color difference)")
    print(f"  - Secondary: SSIM(L*), PSNR (structure/quality)")
    print(f"[INFO] CVD Training: Direct end-to-end correction with Machado simulations")
    
    # ===== MODEL SETUP =====
    print(f"\n[INFO] Setting up CVDCompensationModelAdaIN...")
    
    # FINE-TUNING MODE: encoder NON congelato, usa LR differenziato
    freeze_encoder = config.get("freeze_encoder", False)
    
    model = CVDCompensationModelAdaIN(
        cvd_dim=3,  # HYBRID 3D: [θ_norm (GLOBAL), C_norm (per-type), S_norm (per-type)]
        pretrained_encoder=True,
        freeze_encoder_except_adain=freeze_encoder,
        use_skip_connection=config.get("use_skip_connection", False),
        stop_at_stage=config.get("encoder_depth_level", 2),
        delta_rgb_scale=config.get("delta_rgb_scale", 0.9),
        target_resolution=config.get("target_resolution", 256),
        y_preserving=config.get("y_preserving", False)  # Y-Preserving architecture
    ).to(device)
    
    print(f"[INFO] Model configuration:")
    print(f"  - Architecture: CVDCompensationModelAdaIN (CVDAdaIN integrato)")
    print(f"  - Encoder: ConvNeXt-Tiny (ImageNet pretrained)")
    print(f"  - Conditioning: CVDAdaIN (sostituisce LayerNorm) - HYBRID (3D)")
    y_preserving_mode = config.get("y_preserving", False)
    if y_preserving_mode:
        print(f"  - Output format: Y-Preserving (2 canali ΔCbCr, scale={config.get('delta_rgb_scale', 0.9)})")
        print(f"  -  Luminance preservation GUARANTEED by architecture")
    else:
        print(f"  - Output format: RGB-only (3 canali ΔRGB, scale={config.get('delta_rgb_scale', 0.9)})")
    print(f"  - Fine-tuning encoder: {not freeze_encoder}")
    
    # Training mode info
    if freeze_encoder:
        print(f"\n[TRAINING MODE] Encoder congelato eccetto CVDAdaIN:")
        print(f"  [FROZEN] ConvNeXt backbone weights")
        print(f"  [OK] CVDAdaIN parameters: TRAINABLE")
        print(f"  [OK] Decoder: TRAINABLE")
    else:
        print(f"\n[TRAINING MODE] FINE-TUNING - Tutto trainabile:")
        print(f"  [OK] Encoder: TRAINABLE (LR basso)")
        print(f"  [OK] Decoder: TRAINABLE (LR alto)")
    
    # Count trainable parameters
    params = model.count_parameters()
    
    print(f"[INFO] Model parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
    print(f"  Frozen: {params['frozen']:,} ({100 - params['trainable_pct']:.1f}%)")
    
    # ===== CUDA 12.x torch.compile() OPTIMIZATION =====
    use_compile = config.get("use_torch_compile", False)  # Default OFF to save ~3.8GB VRAM
    if use_compile and supports_torch_compile():
        compile_mode = config.get("torch_compile_mode", "reduce-overhead")
        model = compile_model(model, mode=compile_mode, verbose=True)
        print(f"[INFO] torch.compile() enabled with mode='{compile_mode}'")
    elif use_compile and not supports_torch_compile():
        print(f"[INFO] torch.compile() requested but not supported (CUDA 12+ and PyTorch 2.0+ required)")
    
    # ===== OPTIMIZER CON LEARNING RATE DIFFERENZIATO =====
    encoder_lr = config.get("encoder_learning_rate", 1e-5)
    decoder_lr = config.get("learning_rate", 1e-3)
    
    # Costruisci gruppi di parametri
    optimizer_params = [
        {
            "params": model.decoder.parameters(),
            "lr": decoder_lr,
            "name": "decoder"
        }
    ]
    
    # Aggiungi encoder se non congelato
    if not freeze_encoder:
        optimizer_params.append({
            "params": model.encoder.parameters(),
            "lr": encoder_lr,
            "name": "encoder"
        })
    else:
        optimizer_params.append({
            "params": model.get_adain_parameters(),
            "lr": decoder_lr,
            "name": "encoder_adain"
        })
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        weight_decay=config.get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    print(f"\n[INFO] Optimizer AdamW configured:")
    print(f"  - Decoder: LR={decoder_lr:.2e}")
    if not freeze_encoder:
        print(f"  - Encoder (fine-tuning): LR={encoder_lr:.2e}")
    else:
        print(f"  - Encoder CVDAdaIN only: LR={decoder_lr:.2e}")
    print(f"  - Weight decay: {config.get('weight_decay', 1e-4):.2e}")
    
    # ===== DATASET & DATALOADER SETUP =====
    print(f"\n[INFO] Setting up CVD datasets...")

    # Usa 'dataset_path_recolored' che contiene i JSON mapping files e le immagini CVD generate
    # Fallback a 'dataset_path' per compatibilità con config vecchi
    dataset_path = config.get("dataset_path_recolored") or config.get("dataset_path")
    if not dataset_path:
        raise ValueError("No dataset_path_recolored or dataset_path specified in config YAML")
    
    print(f"[INFO] Using CVD Machado dataset path: {dataset_path}")
    
    from cvd_dataset_loader import create_cvd_dataloaders
    
    # DataLoader parameters from config
    num_workers = config.get("num_workers", None)  # None = auto-detect
    prefetch_factor = config.get("prefetch_factor", 2) if num_workers and num_workers > 0 else None
    pin_memory = config.get("pin_memory", True) if torch.cuda.is_available() else False
    persistent_workers = config.get("persistent_workers", True) if num_workers and num_workers > 0 else False
    
    print(f"[INFO] DataLoader config: workers={num_workers}, prefetch={prefetch_factor}, pin_memory={pin_memory}, persistent={persistent_workers}")
    
    results = create_cvd_dataloaders(
        dataset_base_path=dataset_path,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        cache_preprocessing=config.get("cache_preprocessing", False),
        include_test=config.get("use_test_set", False),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        image_size=config.get("target_resolution", 256)
    )
    
    if config.get("use_test_set", False):
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = results
        print(f"[CVD Dataset] Loaded with test set:")
        print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
        print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
        print(f"  Test: {len(test_dataset)} samples ({len(test_loader)} batches)")
    else:
        train_loader, val_loader, train_dataset, val_dataset = results
        test_loader = None
        print(f"[CVD Dataset] Loaded without test set:")
        print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
        print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    
    print(f"[INFO] HYBRID: 3D profiles [θ_norm (GLOBAL), C_norm (per-type), S_norm (per-type)]")
    print(f"[INFO] Machado severity: {config.get('cvd_severity', 0.15)}")
    
    # =========================================================================
    # HYBRID: Ottieni statistiche profili CVD (θ globale + C/S per-tipo) dal dataset
    # =========================================================================
    per_cvd_stats = train_dataset.per_cvd_stats if getattr(train_dataset, 'per_cvd_stats', None) is not None else None
    theta_global_stats = train_dataset.theta_global_stats if getattr(train_dataset, 'theta_global_stats', None) is not None else None
    
    if per_cvd_stats is None or theta_global_stats is None:
        # Fallback: ricalcola se non disponibili (ma non dovrebbe mai succedere)
        print("[WARNING] HYBRID stats not found in dataset, recomputing from scratch...")
        from cvd_dataset_loader import get_cvd_statistics
        cvd_stats = get_cvd_statistics(train_dataset, max_samples=None)  # USA TUTTO IL DATASET!
        theta_global_stats = cvd_stats.get('theta_global', None)
        per_cvd_stats = cvd_stats.get('per_cvd_type', {})
        # Fallback to global stats if per_cvd_type not available
        if not per_cvd_stats:
            print("[WARNING] per_cvd_type not available, building from global stats")
            for cvd_type in ['protan', 'deutan', 'tritan']:
                per_cvd_stats[cvd_type] = {
                    'mean': cvd_stats['profile_mean'],
                    'std': cvd_stats['profile_std']
                }
        if theta_global_stats is None:
            theta_global_stats = {
                'mean': float(cvd_stats['profile_mean'][0]) if hasattr(cvd_stats['profile_mean'], '__len__') else float(cvd_stats['profile_mean']),
                'std': float(cvd_stats['profile_std'][0]) if hasattr(cvd_stats['profile_std'], '__len__') else float(cvd_stats['profile_std']),
            }
    
    # HYBRID: Costruisci train_profile_stats per il checkpoint
    # Converti numpy arrays a tipi serializzabili
    per_cvd_stats_serializable = {}
    for cvd_type, stats in per_cvd_stats.items():
        per_cvd_stats_serializable[cvd_type] = {
            'C_mean': float(stats.get('C_mean')) if 'C_mean' in stats else None,
            'C_std': float(stats.get('C_std')) if 'C_std' in stats else None,
            'S_mean': float(stats.get('S_mean')) if 'S_mean' in stats else None,
            'S_std': float(stats.get('S_std')) if 'S_std' in stats else None,
            'count': int(stats.get('count', 0)) if stats.get('count', None) is not None else 0,
        }
        if 'mean' in stats and stats['mean'] is not None:
            per_cvd_stats_serializable[cvd_type]['mean'] = stats['mean'].tolist() if hasattr(stats['mean'], 'tolist') else list(stats['mean'])
        if 'std' in stats and stats['std'] is not None:
            per_cvd_stats_serializable[cvd_type]['std'] = stats['std'].tolist() if hasattr(stats['std'], 'tolist') else list(stats['std'])

    theta_global_stats_serializable = {
        'mean': float(theta_global_stats.get('mean')),
        'std': float(theta_global_stats.get('std'))
    }

    train_profile_stats = {
        'theta_global_stats': theta_global_stats_serializable,
        'per_cvd_stats': per_cvd_stats_serializable,
        'normalization_type': 'hybrid_theta_global_cs_per_type',
        'cvd_dim': 3
    }
    
    print(f"[INFO] HYBRID: Profile stats saved for checkpoint")
    print(f"[INFO]   θ GLOBAL: mean={theta_global_stats_serializable['mean']:.2f}°, std={theta_global_stats_serializable['std']:.2f}°")
    for cvd_type in ['protan', 'deutan', 'tritan']:
        if cvd_type in per_cvd_stats_serializable:
            ts = per_cvd_stats_serializable[cvd_type]
            if ts.get('C_mean') is not None:
                print(f"[INFO]   {cvd_type}: C={ts['C_mean']:.3f}±{ts['C_std']:.3f}, S={ts['S_mean']:.3f}±{ts['S_std']:.3f}")
    
    # ===== NaN LOGGER: Inizializza per questo esperimento =====
    experiment_name = config.get('experiment_name', f"cvd_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir = config.get('output_dir', os.getcwd())
    setup_nan_logger(experiment_name=experiment_name, log_dir=log_dir)
    print(f"[INIT] NaN Logger initialized for experiment: {experiment_name}")
    
    # ===== LOSS FUNCTION: CVDLoss (3-Component Modern Loss) =====
    # Static Statistics Normalization: i lambda sono pesi di priorità,
    # la normalizzazione delle scale è gestita da M_init (calibrate() pre-training)
    criterion_cvd = CVDLoss(
        lambda_mse=config.get("cvd_lambda_mse", 1.0),
        lambda_delta_e=config.get("cvd_lambda_delta_e", 0.0),  # DISABLED by default
        lambda_ssim=config.get("cvd_lambda_ssim", 1.0),
        warmup_samples=config.get("cvd_warmup_samples", 200),
        severity_dynamic_weighting=config.get("cvd_severity_dynamic_weighting", False),
        edge_aware_weighting=config.get("cvd_edge_aware_weighting", False),
        profile_stats=train_profile_stats
    ).to(device)
    
    print(f"[INIT] CVDLoss initialized (Static Statistics Normalization)")
    print(f"[INIT]   λ_mse (MSE a*b*): {config.get('cvd_lambda_mse', 1.0)}")
    print(f"[INIT]   λ_delta_e (Delta-E2000): {config.get('cvd_lambda_delta_e', 0.0)}")
    print(f"[INIT]   λ_ssim (MS-SSIM L*): {config.get('cvd_lambda_ssim', 1.0)}")
    print(f"[INIT]   Warmup samples: {config.get('cvd_warmup_samples', 200)}")
    print(f"[INIT]   Severity Weighting (B.2): {config.get('cvd_severity_dynamic_weighting', False)}")
    print(f"[INIT]   Edge-Aware Weighting (B.3): {config.get('cvd_edge_aware_weighting', False)}")
    
    # Flag per sapere se Delta-E è attivo nella loss (per progress bar)
    delta_e_in_loss = config.get('cvd_lambda_delta_e', 1.0) > 0
    if not delta_e_in_loss:
        print(f"[INIT]     Delta-E2000 DISABLED in loss (λ=0), will be shown only in validation")
    
    # ===== CALIBRAZIONE M_init (Pre-Training) =====
    # Calcola le costanti di normalizzazione prima del training
    calibration_path = os.path.join(log_dir, f"calibration_constants_{experiment_name}.json")
    
    # Controlla se esiste già un file di calibrazione da riutilizzare
    if os.path.exists(calibration_path) and config.get("reuse_calibration", False):
        print(f"[CALIBRATE] Caricamento costanti M_init da: {calibration_path}")
        criterion_cvd.load_normalization_constants(calibration_path)
    else:
        print(f"[CALIBRATE] Inizio calibrazione M_init su {config.get('cvd_warmup_samples', 200)} campioni...")
        calibration_result = criterion_cvd.calibrate(
            dataloader=train_loader,
            model=model,
            num_samples=config.get("cvd_warmup_samples", 200),
            save_path=calibration_path
        )
        print(f"[CALIBRATE] Completato! M_mse={calibration_result['M_mse']:.4f}, "
              f"M_delta_e={calibration_result['M_delta_e']:.4f}, M_ssim={calibration_result['M_ssim']:.4f}")
    
    # ===== SCHEDULER =====
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get("lr_factor", 0.8),
        patience=config.get("lr_patience", 7),
        min_lr=config.get("min_lr", 1e-7)
    )
    
    print(f"[INFO] LR Scheduler: ReduceLROnPlateau")
    print(f"  LR factor: {config.get('lr_factor', 0.8)}")
    print(f"  LR patience: {config.get('lr_patience', 7)}")
    print(f"  Total epochs: {config.get('epochs', 50)}")
    
    # ===== VANISHING GRADIENT CONTROL =====
    g_ema = None
    vanish_streak = 0
    gradient_health = "unknown"
    
    VANISH_BETA = 0.95
    VANISH_THRESHOLD = 5e-4
    VANISH_MAX_STREAK = 5
    
    print(f"[INFO] Vanishing gradient monitoring initialized:")
    print(f"  - EMA beta: {VANISH_BETA}")
    print(f"  - Vanish threshold: {VANISH_THRESHOLD}")
    print(f"  - Max streak: {VANISH_MAX_STREAK}")
    
    # ===== EARLY STOPPING SETUP =====
    best_val_loss = float('inf')
    best_delta_e00 = float('inf')
    patience_counter = 0
    quality_gates_ever_passed = False
    start_epoch = 1
    
    # ===== RESUME MANAGEMENT =====
    auto_resume = config.get("auto_resume", True)
    resume_from = config.get("resume_from", "").strip()
    strict_profile_stats_match_on_resume = config.get("strict_profile_stats_match_on_resume", False)
    
    resume_checkpoint_path = None
    if resume_from:
        resume_checkpoint_path = Path(resume_from)
        if not resume_checkpoint_path.exists():
            print(f"[ERROR] Resume checkpoint not found: {resume_from}")
            resume_checkpoint_path = None
    elif auto_resume:
        # Look for latest checkpoint (both old and new naming patterns)
        checkpoint_patterns = [
            f"{config['experiment_name']}_cvd_ep*.pth",
            f"{config['experiment_name']}_phase_1_cvd_ep*.pth"  # Legacy pattern
        ]
        
        potential_checkpoints = []
        for pattern in checkpoint_patterns:
            potential_checkpoints.extend(list(checkpoints_dir.glob(pattern)))
            # Also check output_dir for backward compatibility
            potential_checkpoints.extend(list(output_dir.glob(pattern)))
        
        if potential_checkpoints:
            latest_checkpoint = max(potential_checkpoints, key=lambda p: p.stat().st_mtime)
            resume_checkpoint_path = latest_checkpoint
            print(f"[AUTO RESUME] Found checkpoint: {latest_checkpoint}")
    
    if resume_checkpoint_path:
        print(f"[RESUME] Attempting to resume from: {resume_checkpoint_path}")
        resume_result = load_full_checkpoint(
            resume_checkpoint_path,
            model=model,
            optimizer=optimizer, 
            scheduler=scheduler,
            scaler=None,  # No GradScaler for bf16
            config=config
        )

        # OPTIONAL HARD-GUARD: verify that profile normalization stats match.
        # This prevents silent bugs where resume uses a different dataset or different
        # cached stats (which would directly affect the HYBRID 3D conditioning vectors).
        if strict_profile_stats_match_on_resume:
            try:
                ckpt_raw = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False)
                ckpt_stats = ckpt_raw.get('profile_normalization', None)

                if ckpt_stats is None:
                    raise RuntimeError("Checkpoint missing 'profile_normalization'.")
                if not isinstance(ckpt_stats, dict):
                    raise RuntimeError("Checkpoint 'profile_normalization' is not a dict.")

                # Compare only the critical HYBRID fields.
                def _as_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return None

                def _close(a, b, tol=1e-4):
                    a = _as_float(a); b = _as_float(b)
                    if a is None or b is None:
                        return False
                    return abs(a - b) <= tol

                expected = train_profile_stats
                if ckpt_stats.get('normalization_type') != expected.get('normalization_type'):
                    raise RuntimeError(
                        f"normalization_type mismatch: ckpt={ckpt_stats.get('normalization_type')} vs run={expected.get('normalization_type')}"
                    )
                if int(ckpt_stats.get('cvd_dim', -1)) != int(expected.get('cvd_dim', -2)):
                    raise RuntimeError(
                        f"cvd_dim mismatch: ckpt={ckpt_stats.get('cvd_dim')} vs run={expected.get('cvd_dim')}"
                    )

                ckpt_theta = (ckpt_stats.get('theta_global_stats') or {})
                run_theta = (expected.get('theta_global_stats') or {})
                if not _close(ckpt_theta.get('mean'), run_theta.get('mean')) or not _close(ckpt_theta.get('std'), run_theta.get('std')):
                    raise RuntimeError(
                        "theta_global_stats mismatch: "
                        f"ckpt(mean,std)=({ckpt_theta.get('mean')},{ckpt_theta.get('std')}) vs "
                        f"run(mean,std)=({run_theta.get('mean')},{run_theta.get('std')})"
                    )

                ckpt_per = ckpt_stats.get('per_cvd_stats') or {}
                run_per = expected.get('per_cvd_stats') or {}
                for cvd_type in ['protan', 'deutan', 'tritan']:
                    if cvd_type not in ckpt_per or cvd_type not in run_per:
                        raise RuntimeError(f"per_cvd_stats missing '{cvd_type}' in ckpt or run")
                    for k in ['C_mean', 'C_std', 'S_mean', 'S_std']:
                        if not _close(ckpt_per[cvd_type].get(k), run_per[cvd_type].get(k)):
                            raise RuntimeError(
                                f"per_cvd_stats mismatch for {cvd_type}.{k}: ckpt={ckpt_per[cvd_type].get(k)} vs run={run_per[cvd_type].get(k)}"
                            )

                print("[RESUME] [OK] strict_profile_stats_match_on_resume: checkpoint stats match current dataset stats")
            except Exception as e:
                print(f"[RESUME] [ERR] strict_profile_stats_match_on_resume failed: {e}")
                raise
        
        if resume_result[0] is not None:
            (resume_epoch, resume_best_loss, resume_best_delta_e00, resume_patience, 
             resume_gradient_health, resume_cvd_weights, resume_cvd_normalization) = resume_result
            start_epoch = resume_epoch
            best_val_loss = resume_best_loss if resume_best_loss is not None else best_val_loss
            best_delta_e00 = resume_best_delta_e00 if resume_best_delta_e00 is not None else best_delta_e00
            patience_counter = resume_patience if resume_patience is not None else patience_counter
            gradient_health = resume_gradient_health if resume_gradient_health is not None else gradient_health
            
            # Restore CVDLoss lambda weights
            if resume_cvd_weights is not None:
                criterion_cvd.set_weights_from_checkpoint(
                    lambda_mse=resume_cvd_weights['lambda_mse'],
                    lambda_delta_e=resume_cvd_weights['lambda_delta_e'],
                    lambda_ssim=resume_cvd_weights['lambda_ssim'],
                    severity=resume_cvd_weights.get('severity')
                )
                print(f"[RESUME] [OK] CVDLoss lambda weights applied to criterion")
            else:
                print(f"[RESUME] [WARNING] No CVDLoss weights to restore, using config defaults")
            
            # Restore M_init normalization constants (CRITICAL for correct loss scaling)
            if resume_cvd_normalization is not None and resume_cvd_normalization.get('is_calibrated', False):
                criterion_cvd.set_normalization_constants(
                    M_mse=resume_cvd_normalization['M_mse'],
                    M_delta_e=resume_cvd_normalization['M_delta_e'],
                    M_ssim=resume_cvd_normalization['M_ssim']
                )
                print(f"[RESUME] [OK] M_init normalization constants restored from checkpoint")
            else:
                print(f"[RESUME] [WARNING] No M_init in checkpoint - will use JSON or recalibrate")
            
            print(f"[RESUME] [OK] Successfully resumed from epoch {start_epoch}")
        else:
            print(f"[RESUME] [ERR] Resume failed, starting from scratch")
    else:
        print(f"[INFO] Starting training from scratch (no resume)")
    
    # LOGGER SETUP
    logger = SimpleTrainingLogger(
        config, 
        train_ds=train_dataset,  
        val_ds=val_dataset,      
        model=model,
        resume_from_epoch=start_epoch if start_epoch > 1 else None  
    )
    print(f"[INFO] Training logger initialized (resume_from_epoch={start_epoch if start_epoch > 1 else None})")
    
    # Print training configuration
    print_training_config(config, device, amp_dtype, model)
    
    # Early stopping parameters
    use_enhanced_early_stopping = config.get("use_enhanced_early_stopping", True)
    min_epochs = config.get("min_epochs", 20)
    max_patience = config.get("patience", 20)
    min_delta = config.get("min_improvement", 0.001)
    early_stopping_delta_e_threshold = config.get("early_stopping_delta_e_threshold", 5.0)  # Soglia ΔE00 clinica
    
    # Quality Gates Thresholds
    min_ssim_threshold = config.get("min_ssim_threshold", 0.6)
    min_psnr_threshold = config.get("min_psnr_threshold", 20.0)
    
    # Delta E00 Perceptual Validation
    delta_e00_threshold = config.get("delta_e00_threshold", 3.0)
    should_calculate_delta_e00 = config.get("should_calculate_delta_e00", True)
    delta_e00_calculation_frequency = config.get("delta_e00_calculation_frequency", 5)
    
    # ===== TRAINING LOOP =====
    for epoch in range(start_epoch, config.get("epochs", 80) + 1):
        epoch_start_time = time.time()
        
        model.train()
        criterion_cvd.set_phase('train')  # Set phase per NaN logger
        train_total_loss = 0.0
        valid_batches = 0
        
        # ═══════════════════════════════════════════════════════════════════
        # SKIPPED BATCH TRACKING: Count and log batches skipped due to NaN/errors
        # Threshold: if >5% batches fail, stop training (indicates systematic bug)
        # ═══════════════════════════════════════════════════════════════════
        skipped_batches = 0
        total_batches_attempted = 0
        
        # Gradient tracking
        total_grad_norm = 0.0
        clipped_steps = 0
        total_steps_epoch = 0
        num_batches = 0
        
        # Loss components accumulation for epoch-level logging
        epoch_mse_ab = 0.0
        epoch_delta_e00 = 0.0
        epoch_msssim_rgb = 0.0
        epoch_lambda_mse = 0.0
        epoch_lambda_delta_e = 0.0
        epoch_lambda_ssim = 0.0
        
        # Accumulatori per valori NORMALIZZATI (per plot sulla stessa scala)
        epoch_mse_ab_norm = 0.0
        epoch_delta_e00_norm = 0.0
        epoch_msssim_rgb_norm = 0.0
        
        gradient_health_ema = None
        training_crisis_count = 0
        last_lr_backoff_epoch = -999
        
        is_best = False
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            total_batches_attempted += 1
            try:
                normal_images = batch['input'].to(device, non_blocking=True)
                compensated_images = batch['target'].to(device, non_blocking=True)
                cvd_profile = batch['profile'].to(device, non_blocking=True)
                metadata = batch['metadata']
                
                batch_size_actual = normal_images.size(0)
                
                # ===== SAFETY CHECK: Validate batch data =====
                # Check for NaN/Inf in input data (can happen with corrupted images)
                if check_for_nan_inf(normal_images, "normal_images"):
                    print(f"[SKIP] Batch {batch_idx} - NaN/Inf in input images, skipping...")
                    continue
                if check_for_nan_inf(compensated_images, "compensated_images"):
                    print(f"[SKIP] Batch {batch_idx} - NaN/Inf in target images, skipping...")
                    continue
                if check_for_nan_inf(cvd_profile, "cvd_profile"):
                    print(f"[SKIP] Batch {batch_idx} - NaN/Inf in CVD profile, skipping...")
                    continue
                
                # Debug logging (first batch only)
                if batch_idx == 0 and epoch == start_epoch and config.get('debug_verbose', False):
                    print(f"\n[BATCH DEBUG] CVD compensation batch loaded successfully")
                    print(f"  normal_images shape: {normal_images.shape}, range: [{normal_images.min():.4f}, {normal_images.max():.4f}]")
                    print(f"  compensated_images shape: {compensated_images.shape}")
                    print(f"  cvd_profile shape: {cvd_profile.shape}")
                
                optimizer.zero_grad()
                
                # Forward pass with AMP (bf16 or fp16)
                # NOTA: use_amp è già definito a livello di funzione
                # amp_dtype può essere torch.bfloat16 (no scaler) o torch.float16 (con scaler)
                
                if use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                        model_output = model(normal_images, profile_feats=cvd_profile, epoch=epoch, total_epochs=config.get("epochs", 80))
                        
                        if isinstance(model_output, dict):
                            output_rgb = model_output.get('rgb_output', model_output)
                        else:
                            output_rgb = model_output
                        
                    # Loss computation
                    # IMPORTANT: Denormalize target from ImageNet range to [0, 1]
                    # Model output is ALREADY in [-1, 1] (tanh activation in decoder)
                    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    target_denorm = compensated_images * imagenet_std + imagenet_mean  # [0, 1] range
                    target_denorm = torch.clamp(target_denorm, 0.0, 1.0)  # Safety clamp
                    
                    # Convert target to [-1, 1] for loss (model output is already [-1, 1] from tanh)
                    output_for_loss = output_rgb  # ALREADY [-1, 1] from tanh - DO NOT transform!
                    target_for_loss = target_denorm * 2.0 - 1.0  # [0,1] -> [-1,1]
                    
                    # DEBUG: Test with simple MSE to isolate NaN source
                    USE_SIMPLE_MSE_DEBUG = False  # Set to False to use CVDLoss
                    
                    if USE_SIMPLE_MSE_DEBUG:
                        # Simple MSE loss - if this works, problem is in CVDLoss
                        total_loss = torch.nn.functional.mse_loss(output_for_loss, target_for_loss)
                        loss_components = {
                            'mse_ab': total_loss,
                            'delta_e00': torch.tensor(0.0, device=device),
                            'msssim_rgb_loss': torch.tensor(0.0, device=device),
                            'msssim_rgb_value': torch.tensor(1.0, device=device),
                        }
                    else:
                        try:
                            loss_dict = criterion_cvd(output_for_loss, target_for_loss, cvd_profile=cvd_profile)
                            total_loss = loss_dict['loss']
                            
                            loss_components = {
                                'mse_ab': loss_dict.get('mse_ab', torch.tensor(0.0, device=device)),
                                'delta_e00': loss_dict.get('delta_e00', torch.tensor(0.0, device=device)),
                                'msssim_rgb_loss': loss_dict.get('msssim_rgb_loss', torch.tensor(0.0, device=device)),
                                'msssim_rgb_value': loss_dict.get('msssim_rgb_value', torch.tensor(1.0, device=device)),
                                'weights': loss_dict.get('weights', {}),
                            }
                        except Exception as e:
                            print(f"[ERROR] CVDLoss computation failed: {e}")
                            total_loss = torch.nn.functional.mse_loss(output_for_loss, target_for_loss)
                            loss_components = {'mse_ab': total_loss, 'delta_e00': torch.tensor(0.0), 'msssim_rgb_loss': torch.tensor(0.0), 'msssim_rgb_value': torch.tensor(1.0)}
                    
                    # Safety checks
                    if check_for_nan_inf(output_rgb, "output_rgb") or not check_loss_safety(total_loss, "total_loss"):
                        print(f"[SKIP] Batch {batch_idx} - NaN/Inf detected, skipping...")
                        continue
                    
                    # DEBUG: Print loss components to find NaN source
                    if batch_idx < 10:
                        def get_val(x):
                            return x.item() if hasattr(x, 'item') else float(x)
                        print(f"[DEBUG] Batch {batch_idx}: loss={get_val(total_loss):.4f}, "
                              f"mse_ab={get_val(loss_components['mse_ab']):.4f}, "
                              f"delta_e00={get_val(loss_components['delta_e00']):.4f}, "
                              f"msssim_rgb={get_val(loss_components['msssim_rgb_loss']):.4f}")
                        print(f"[DEBUG] output_for_loss: min={output_for_loss.min().item():.4f}, max={output_for_loss.max().item():.4f}")
                        print(f"[DEBUG] target_for_loss: min={target_for_loss.min().item():.4f}, max={target_for_loss.max().item():.4f}")
                    
                    # Standard backward pass (bf16 doesn't need GradScaler)
                    total_loss.backward()
                    
                    # Check gradient NaN/Inf with detailed debug
                    bad_grad = False
                    nan_params = []
                    for name, p in model.named_parameters():
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            nan_params.append(name)
                            if not bad_grad:  # Only print first occurrence
                                grad_stats = f"grad: min={p.grad.min().item():.6f}, max={p.grad.max().item():.6f}"
                                print(f"[WARN] Epoch {epoch}, Batch {batch_idx}: grad NaN/Inf in {name} ({grad_stats}) -> skip step")
                            bad_grad = True
                    
                    if bad_grad and batch_idx < 5:
                        print(f"[DEBUG] Total params with NaN grad: {len(nan_params)}")
                        print(f"[DEBUG] First 5 NaN params: {nan_params[:5]}")
                    
                    if bad_grad:
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    grad_norm = gradient_clipping(model, epoch)
                    
                    optimizer.step()
                    
                else:
                    # No AMP (CPU mode or AMP disabled)
                    model_output = model(normal_images, profile_feats=cvd_profile, epoch=epoch, total_epochs=config.get("epochs", 80))
                    
                    if isinstance(model_output, dict):
                        output_rgb = model_output.get('rgb_output', model_output)
                    else:
                        output_rgb = model_output
                    
                    # IMPORTANT: Denormalize target from ImageNet range to [0, 1]
                    # Model output is ALREADY in [-1, 1] (tanh activation in decoder)
                    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    target_denorm = compensated_images * imagenet_std + imagenet_mean
                    target_denorm = torch.clamp(target_denorm, 0.0, 1.0)
                    
                    # Convert target to [-1, 1] for loss (model output is already [-1, 1] from tanh)
                    output_for_loss = output_rgb  # ALREADY [-1, 1] from tanh - DO NOT transform!
                    target_for_loss = target_denorm * 2.0 - 1.0
                    
                    try:
                        loss_dict = criterion_cvd(output_for_loss, target_for_loss, cvd_profile=cvd_profile)
                        total_loss = loss_dict['loss']
                        loss_components = {
                            'mse_ab': loss_dict.get('mse_ab', torch.tensor(0.0)),
                            'delta_e00': loss_dict.get('delta_e00', torch.tensor(0.0)),
                            'msssim_rgb_loss': loss_dict.get('msssim_rgb_loss', torch.tensor(0.0)),
                            'msssim_rgb_value': loss_dict.get('msssim_rgb_value', torch.tensor(1.0)),
                            'weights': loss_dict.get('weights', {}),
                        }
                    except Exception as e:
                        print(f"[ERROR] CVDLoss computation failed: {e}")
                        total_loss = torch.nn.functional.mse_loss(output_for_loss, target_for_loss)
                        loss_components = {'mse_ab': total_loss, 'delta_e00': torch.tensor(0.0), 'msssim_rgb_loss': torch.tensor(0.0), 'msssim_rgb_value': torch.tensor(1.0)}
                    
                    if check_for_nan_inf(output_rgb, "output_rgb") or not check_loss_safety(total_loss, "total_loss"):
                        print(f"[SKIP] Batch {batch_idx} - NaN/Inf detected, skipping...")
                        continue
                    
                    total_loss.backward()
                    
                    bad_grad = False
                    for name, p in model.named_parameters():
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            bad_grad = True
                            break
                    
                    if bad_grad:
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    optimizer.step()
                
                # Gradient monitoring
                current_grad_norm = grad_norm
                
                if gradient_health_ema is None:
                    gradient_health_ema = current_grad_norm
                else:
                    gradient_health_ema = 0.9 * gradient_health_ema + 0.1 * current_grad_norm
                
                if current_grad_norm < 1e-4:
                    gradient_health = "vanishing"
                elif current_grad_norm > 10.0:
                    gradient_health = "exploding"
                else:
                    gradient_health = "healthy"
                
                total_grad_norm += float(current_grad_norm) if current_grad_norm is not None else 0.0
                total_steps_epoch += 1
                
                # Batch logging
                batch_loss_dict = {
                    'total_loss': safe_float_convert(total_loss),
                    'mse_ab': safe_float_convert(loss_components.get('mse_ab', 0.0)),
                    'delta_e00': safe_float_convert(loss_components.get('delta_e00', 0.0)),
                    'msssim_rgb_loss': safe_float_convert(loss_components.get('msssim_rgb_loss', 0.0)),
                    'msssim_rgb_value': safe_float_convert(loss_components.get('msssim_rgb_value', 1.0))
                }
                
                grad_norm_val = current_grad_norm.item() if isinstance(current_grad_norm, torch.Tensor) else float(current_grad_norm) if current_grad_norm is not None else 0.0
                logger.log_batch(epoch, batch_idx, batch_loss_dict, grad_norm_val)
                
                num_batches += 1
                train_total_loss += safe_float_convert(total_loss)
                valid_batches += 1
                
                # Accumulate loss components for epoch-level logging (RAW values)
                epoch_mse_ab += safe_float_convert(loss_components.get('mse_ab', 0.0))
                epoch_delta_e00 += safe_float_convert(loss_components.get('delta_e00', 0.0))
                epoch_msssim_rgb += safe_float_convert(loss_components.get('msssim_rgb_loss', 0.0))
                
                # Accumulate NORMALIZED values for plotting on same scale
                # Usa M_init da CVDLoss per normalizzare
                M_mse = getattr(criterion_cvd, 'M_mse', None)
                M_delta_e = getattr(criterion_cvd, 'M_delta_e', None)
                M_ssim = getattr(criterion_cvd, 'M_ssim', None)
                
                mse_val = safe_float_convert(loss_components.get('mse_ab', 0.0))
                de_val = safe_float_convert(loss_components.get('delta_e00', 0.0))
                ssim_val = safe_float_convert(loss_components.get('msssim_rgb_loss', 0.0))
                
                epoch_mse_ab_norm += mse_val / float(M_mse) if M_mse is not None and float(M_mse) > 0 else mse_val
                epoch_delta_e00_norm += de_val / float(M_delta_e) if M_delta_e is not None and float(M_delta_e) > 0 else de_val
                epoch_msssim_rgb_norm += ssim_val / float(M_ssim) if M_ssim is not None and float(M_ssim) > 0 else ssim_val
                
                # Accumulate lambda weights (take from loss_components if available)
                # CVDCompensationLoss uses: recon, fidelity, gamut
                # Map to: alpha (recon/mse), beta (fidelity/delta_e), gamma (gamut/ssim)
                weights = loss_components.get('weights', {})
                epoch_lambda_mse += safe_float_convert(weights.get('recon', weights.get('lambda_mse', 0.0)))
                epoch_lambda_delta_e += safe_float_convert(weights.get('fidelity', weights.get('lambda_delta_e', 0.0)))
                epoch_lambda_ssim += safe_float_convert(weights.get('gamut', weights.get('lambda_ssim', 0.0)))
                
                # Progress bar - show ΔE00 only if it's part of the loss
                postfix_dict = {
                    'loss': f'{safe_float_convert(total_loss):.4f}',
                    'mse_ab': f'{safe_float_convert(loss_components.get("mse_ab", 0.0)):.4f}',
                    'MS-SSIM': f'{safe_float_convert(loss_components.get("msssim_rgb_value", 1.0)):.4f}'
                }
                # Add ΔE00 to progress bar only if it's part of the loss (lambda > 0)
                if delta_e_in_loss:
                    postfix_dict['ΔE00'] = f'{safe_float_convert(loss_components.get("delta_e00", 0.0)):.4f}'
                
                train_pbar.set_postfix(postfix_dict)
                
            except Exception as e:
                skipped_batches += 1
                # Log to NaN logger for post-mortem analysis
                from losses import log_nan_event
                log_nan_event('train', batch_idx, 'BATCH_SKIPPED', {
                    'error': str(e),
                    'epoch': epoch,
                    'skipped_so_far': skipped_batches,
                    'total_attempted': total_batches_attempted
                })
                print(f"[SKIP] Batch {batch_idx} failed ({skipped_batches}/{total_batches_attempted}): {e}")
                continue
        
        # ═══════════════════════════════════════════════════════════════════
        # THRESHOLD CHECK: Stop if >5% batches failed (indicates systematic bug)
        # ═══════════════════════════════════════════════════════════════════
        if total_batches_attempted > 0:
            skip_ratio = skipped_batches / total_batches_attempted
            if skip_ratio > 0.05:  # 5% threshold
                raise RuntimeError(
                    f"[CRITICAL] Too many batches skipped: {skipped_batches}/{total_batches_attempted} "
                    f"({skip_ratio*100:.1f}%) - this indicates a systematic bug, not random bad data. "
                    f"Check NaN logger for details."
                )
            elif skipped_batches > 0:
                print(f"[INFO] Epoch {epoch}: {skipped_batches}/{total_batches_attempted} batches skipped ({skip_ratio*100:.2f}%)")
        
        avg_train_loss = train_total_loss / valid_batches if valid_batches > 0 else 0.0
        
        # ===== EMA GRADIENT TRACKING =====
        avg_grad_norm = total_grad_norm / total_steps_epoch if total_steps_epoch > 0 else 0.0
        
        if g_ema is None:
            g_ema = avg_grad_norm
        else:
            g_ema = VANISH_BETA * g_ema + (1.0 - VANISH_BETA) * avg_grad_norm
        
        if g_ema < VANISH_THRESHOLD:
            vanish_streak += 1
            gradient_health = "vanishing"
            print(f"[WARNING] Vanishing gradient detected: EMA {g_ema:.6f} < {VANISH_THRESHOLD} (streak: {vanish_streak})")
        else:
            vanish_streak = 0
            gradient_health = "healthy"
        
        if vanish_streak >= VANISH_MAX_STREAK:
            print(f"[CRITICAL] Vanishing gradient streak {vanish_streak} >= {VANISH_MAX_STREAK}")
            print(f"[INTERVENTION] Boosting learning rates...")
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = min(old_lr * 2.0, 1e-2)
                print(f"[LR BOOST] {old_lr:.2e} -> {param_group['lr']:.2e}")
            vanish_streak = 0
        
        # ===== VALIDATION EPOCH =====
        avg_val_loss = 0.0
        first_val_input = None
        first_val_target = None 
        first_val_output = None
        
        if val_loader is not None:
            model.eval()
            criterion_cvd.set_phase('val')  # Set phase per NaN logger
            val_total_loss = 0.0
            valid_val_batches = 0
            
            delta_e00_primary_scores = []
            mse_lab_scores = []
            ssim_scores = []
            psnr_scores = []
            msssim_rgb_scores = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
                
                for batch_idx, batch in enumerate(val_pbar):
                    try:
                        normal_images = batch['input'].to(device, non_blocking=True)
                        compensated_images = batch['target'].to(device, non_blocking=True)
                        cvd_profile = batch['profile'].to(device, non_blocking=True)
                        
                        # Use AMP if available (bf16 or fp16)
                        if use_amp:
                            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                                model_output = model(normal_images, profile_feats=cvd_profile, epoch=epoch, total_epochs=config.get("epochs", 80))
                                
                                if isinstance(model_output, dict):
                                    output_rgb = model_output.get('rgb_output', model_output)
                                else:
                                    output_rgb = model_output
                                
                                # Denormalize for loss computation
                                # Model output is ALREADY in [-1, 1] (tanh activation)
                                imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                                imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                                target_denorm = compensated_images * imagenet_std + imagenet_mean
                                target_denorm = torch.clamp(target_denorm, 0.0, 1.0)
                                output_for_loss = output_rgb  # ALREADY [-1, 1] from tanh
                                target_for_loss = target_denorm * 2.0 - 1.0
                                
                                try:
                                    loss_dict = criterion_cvd(output_for_loss, target_for_loss, cvd_profile=cvd_profile)
                                    total_loss = loss_dict['loss']
                                    
                                    delta_e00_val = loss_dict.get('delta_e00', torch.tensor(0.0, device=device))
                                    delta_e00_primary_scores.append(safe_float_convert(delta_e00_val))
                                    
                                    mse_lab_val = loss_dict.get('mse_ab', torch.tensor(0.0, device=device))
                                    mse_lab_scores.append(safe_float_convert(mse_lab_val))
                                    
                                except Exception as e:
                                    total_loss = torch.nn.functional.mse_loss(output_for_loss, target_for_loss)
                                    delta_e00_primary_scores.append(0.0)
                                    mse_lab_scores.append(safe_float_convert(total_loss))
                        else:
                            model_output = model(normal_images, profile_feats=cvd_profile, epoch=epoch, total_epochs=config.get("epochs", 80))
                            
                            if isinstance(model_output, dict):
                                output_rgb = model_output.get('rgb_output', model_output)
                            else:
                                output_rgb = model_output
                            
                            # Denormalize for loss computation
                            # Model output is ALREADY in [-1, 1] (tanh activation)
                            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                            target_denorm = compensated_images * imagenet_std + imagenet_mean
                            target_denorm = torch.clamp(target_denorm, 0.0, 1.0)
                            output_for_loss = output_rgb  # ALREADY [-1, 1] from tanh
                            target_for_loss = target_denorm * 2.0 - 1.0
                            
                            try:
                                loss_dict = criterion_cvd(output_for_loss, target_for_loss, cvd_profile=cvd_profile)
                                total_loss = loss_dict['loss']
                                
                                delta_e00_val = loss_dict.get('delta_e00', torch.tensor(0.0))
                                delta_e00_primary_scores.append(safe_float_convert(delta_e00_val))
                                
                                mse_lab_val = loss_dict.get('mse_ab', torch.tensor(0.0))
                                mse_lab_scores.append(safe_float_convert(mse_lab_val))
                                
                            except Exception as e:
                                total_loss = torch.nn.functional.mse_loss(output_for_loss, target_for_loss)
                                delta_e00_primary_scores.append(0.0)
                                mse_lab_scores.append(safe_float_convert(total_loss))
                        
                        # Secondary metrics - use consistent ranges
                        # criterion_ssim si aspetta entrambi gli input in [-1,1]
                        # output_rgb is already [-1,1] (tanh), target_denorm is [0,1]
                        try:
                            target_for_ssim = target_denorm * 2.0 - 1.0  # [0,1] → [-1,1]
                            ssim_val = criterion_ssim(output_rgb, target_for_ssim)
                            ssim_scores.append(safe_float_convert(ssim_val))
                        except Exception:
                            ssim_scores.append(0.0)
                        
                        # Anche criterion_psnr necessita di range consistente
                        # Convert output_rgb from [-1,1] to [0,1] for PSNR
                        try:
                            output_for_psnr = (output_rgb + 1.0) / 2.0  # [-1,1] → [0,1]
                            psnr_val = criterion_psnr(output_for_psnr, target_denorm)
                            psnr_scores.append(safe_float_convert(psnr_val))
                        except Exception:
                            psnr_scores.append(0.0)
                        
                        try:
                            msssim_loss, msssim_value = criterion_cvd.compute_msssim_rgb(output_for_loss, target_for_loss)
                            msssim_rgb_scores.append(safe_float_convert(msssim_value))
                        except Exception:
                            msssim_rgb_scores.append(0.0)
                        
                        if batch_idx == 0:
                            try:
                                first_val_input = normal_images[:4].detach().clone()
                                first_val_target = target_denorm[:4].detach().clone()  # Use denorm
                                first_val_output = output_rgb[:4].detach().clone()
                            except Exception:
                                pass
                        
                        val_total_loss += safe_float_convert(total_loss)
                        valid_val_batches += 1
                        
                        val_pbar.set_postfix({'val_loss': f'{safe_float_convert(total_loss):.4f}'})
                        
                    except Exception as e:
                        print(f"[ERROR] Validation batch {batch_idx} failed: {e}")
                        continue
            
            avg_val_loss = val_total_loss / valid_val_batches if valid_val_batches > 0 else 0.0
            
            avg_delta_e00_primary = np.mean(delta_e00_primary_scores) if delta_e00_primary_scores else 999.0
            avg_mse_lab = np.mean(mse_lab_scores) if mse_lab_scores else 0.0
            avg_val_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
            avg_val_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
            avg_val_msssim_rgb = np.mean(msssim_rgb_scores) if msssim_rgb_scores else 0.0
            
            if config.get('debug_verbose', False):
                print(f"\n{'='*75}")
                print(f"[VALIDATION] Epoch {epoch}/{config.get('epochs', 80)}")
                print(f"{'='*75}")
                print(f"  Total Loss: {avg_val_loss:.6f}")
                print(f"  Delta-E2000: {avg_delta_e00_primary:.6f}")
                print(f"  SSIM: {avg_val_ssim:.4f}")
                print(f"  PSNR: {avg_val_psnr:.2f} dB")
                print(f"{'='*75}\n")
            else:
                print(f"[VAL E{epoch}] Loss: {avg_val_loss:.6f} | ΔE2000: {avg_delta_e00_primary:.4f} | SSIM: {avg_val_ssim:.4f}")
            
            primary_metric = avg_delta_e00_primary
            metric_name = "Delta-E2000"
            
        else:
            avg_delta_e00_primary = 0.0
            avg_mse_lab = avg_train_loss
            avg_val_ssim = 0.0
            avg_val_psnr = 0.0
            primary_metric = avg_train_loss
            metric_name = "TrainLoss"
            print(f"\n[WARNING] No validation loader - using training loss for early stopping")
        
        # LR backoff based on clip ratio
        clip_ratio = (clipped_steps / total_steps_epoch) if total_steps_epoch > 0 else 0.0
        
        print(f"[EPOCH {epoch}] Clip ratio: {clip_ratio:.2%} ({clipped_steps}/{total_steps_epoch})")
        if clip_ratio > 0.5:
            print(f"[LR BACKOFF] High gradient clipping detected: {clip_ratio:.1%} > 50%")
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = max(old_lr * 0.5, 1e-6)
                print(f"[LR BACKOFF] LR {old_lr:.2e} -> {param_group['lr']:.2e}")
        
        # Update scheduler
        if val_loader is not None:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step(avg_train_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # ===== LOGGING =====
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        current_wds = [param_group.get('weight_decay', 0.0) for param_group in optimizer.param_groups]
        
        logger.log(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_delta_e00_primary=avg_delta_e00_primary if 'avg_delta_e00_primary' in locals() else None,
            val_mse_lab=avg_mse_lab if 'avg_mse_lab' in locals() else None,
            val_delta_e00=avg_delta_e00_primary if 'avg_delta_e00_primary' in locals() else None,
            val_ssim=avg_val_ssim,
            val_psnr=avg_val_psnr,
            val_msssim_rgb=avg_val_msssim_rgb if 'avg_val_msssim_rgb' in locals() else None,
            loss_components={
                # Valori NORMALIZZATI per plot sulla stessa scala della Train Loss
                'mse_ab': epoch_mse_ab_norm / valid_batches if valid_batches > 0 else 0.0,
                'delta_e00': epoch_delta_e00_norm / valid_batches if valid_batches > 0 else 0.0,
                'msssim_rgb_loss': epoch_msssim_rgb_norm / valid_batches if valid_batches > 0 else 0.0,
                # Valori RAW per interpretabilita' (opzionali)
                'mse_ab_raw': epoch_mse_ab / valid_batches if valid_batches > 0 else 0.0,
                'delta_e00_raw': epoch_delta_e00 / valid_batches if valid_batches > 0 else 0.0,
                'msssim_rgb_loss_raw': epoch_msssim_rgb / valid_batches if valid_batches > 0 else 0.0,
                # Pesi lambda
                'alpha_weight': epoch_lambda_mse / valid_batches if valid_batches > 0 else 0.0,
                'beta_weight': epoch_lambda_delta_e / valid_batches if valid_batches > 0 else 0.0,
                'gamma_weight': epoch_lambda_ssim / valid_batches if valid_batches > 0 else 0.0,
            },
            grad_norm=avg_grad_norm,
            lr_current=current_lrs[0] if current_lrs else None,
            wd_current=current_wds[0] if current_wds else None,
            ema_grad_norm=float(g_ema) if g_ema is not None else None,
            vanish_streak=int(vanish_streak),
            gradient_health=gradient_health,
            clip_ratio=clip_ratio,
            clipped_steps=clipped_steps,
            total_steps=total_steps_epoch,
            scheduler_last_lr=get_current_lr(scheduler, optimizer),
            best_val_loss=best_val_loss,
            best_delta_e00=best_delta_e00,
            primary_metric=primary_metric if 'primary_metric' in locals() else None,
            metric_name="Delta-E2000",
            patience_counter=patience_counter,
            max_patience=max_patience,
            is_best=is_best if 'is_best' in locals() else False,
            quality_gates_passed=False,
            delta_e00=avg_delta_e00_primary if 'avg_delta_e00_primary' in locals() else float('inf'),
            epoch_time=epoch_time,
            input_tensor=first_val_input,
            target_tensor=first_val_target,
            output_tensor=first_val_output
        )
        
        # ===== EPOCH SUMMARY =====
        print(f"\nEpoch {epoch}/{config.get('epochs', 50)} completed in {epoch_time:.1f}s")
        print(f"  Train CVDLoss: {avg_train_loss:.6f}")
        
        # ===== LOSS BALANCE LOGGING =====
        # Usa i valori NORMALIZZATI per calcolare i contributi percentuali
        avg_mse_ab_norm_epoch = epoch_mse_ab_norm / valid_batches if valid_batches > 0 else 0.0
        avg_delta_e00_norm_epoch = epoch_delta_e00_norm / valid_batches if valid_batches > 0 else 0.0
        avg_msssim_norm_epoch = epoch_msssim_rgb_norm / valid_batches if valid_batches > 0 else 0.0
        
        # Valori RAW per logging interpretabile
        avg_mse_ab_raw_epoch = epoch_mse_ab / valid_batches if valid_batches > 0 else 0.0
        avg_delta_e00_raw_epoch = epoch_delta_e00 / valid_batches if valid_batches > 0 else 0.0
        avg_msssim_raw_epoch = epoch_msssim_rgb / valid_batches if valid_batches > 0 else 0.0
        
        # Recupera i lambda dal config
        lambda_mse = config.get('cvd_lambda_mse', 0.45)
        lambda_de = config.get('cvd_lambda_delta_e', 0.30)
        lambda_ssim = config.get('cvd_lambda_ssim', 0.25)
        
        # Contributi pesati usando valori NORMALIZZATI (ora corretti!)
        weighted_mse = lambda_mse * avg_mse_ab_norm_epoch
        weighted_de = lambda_de * avg_delta_e00_norm_epoch
        weighted_ssim = lambda_ssim * avg_msssim_norm_epoch
        total_weighted = weighted_mse + weighted_de + weighted_ssim
        
        if total_weighted > 0:
            pct_mse = 100.0 * weighted_mse / total_weighted
            pct_de = 100.0 * weighted_de / total_weighted
            pct_ssim = 100.0 * weighted_ssim / total_weighted
            print(f"  [LOSS BALANCE] MSE: {pct_mse:.1f}% | ΔE00: {pct_de:.1f}% | SSIM: {pct_ssim:.1f}%")
            
            # Warning se una componente domina troppo (>70%)
            if pct_mse > 70 or pct_de > 70 or pct_ssim > 70:
                dominant = "MSE" if pct_mse > 70 else ("ΔE00" if pct_de > 70 else "SSIM")
                print(f"  [WARNING] {dominant} domina la loss! Considera di aggiustare i pesi λ")
        
        if val_loader is not None:
            print(f"  Val CVDLoss: {avg_val_loss:.6f}")
            print(f"  Val Metrics: SSIM={avg_val_ssim:.4f}, PSNR={avg_val_psnr:.4f}dB")
        
        print(f"  LR: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"  Gradient: norm={avg_grad_norm:.6f}, EMA={g_ema:.6f}, health={gradient_health}")
        print(f"  Clipping: {clip_ratio:.2%} ({clipped_steps}/{total_steps_epoch})")
        if vanish_streak > 0:
            print(f"  Vanishing streak: {vanish_streak}/{VANISH_MAX_STREAK}")
        
        # ===== CHECKPOINT & EARLY STOPPING =====
        is_best = False
        if val_loader is not None:
            psnr_satisfied = avg_val_psnr >= min_psnr_threshold
            ssim_satisfied = avg_val_ssim >= min_ssim_threshold
            quality_gates_passed = psnr_satisfied and ssim_satisfied
            
            if metric_name == "Delta-E2000":
                improvement_amount = best_val_loss - avg_delta_e00_primary
                significant_improvement = improvement_amount >= min_delta
                metric_improved = avg_delta_e00_primary < best_val_loss
            else:
                improvement_amount = best_val_loss - avg_val_loss
                significant_improvement = improvement_amount >= min_delta
                metric_improved = avg_val_loss < best_val_loss
            
            if metric_improved and significant_improvement:
                if metric_name == "Delta-E2000":
                    best_val_loss = avg_delta_e00_primary
                    best_delta_e00 = avg_delta_e00_primary
                    print(f"  [OK] New best Delta-E2000: {best_val_loss:.6f}")
                else:
                    best_val_loss = avg_val_loss
                    print(f"  [OK] New best Loss: {best_val_loss:.6f}")
                
                patience_counter = 0
                is_best = True
                
            elif metric_improved:
                if metric_name == "Delta-E2000":
                    best_val_loss = avg_delta_e00_primary
                    best_delta_e00 = avg_delta_e00_primary
                else:
                    best_val_loss = avg_val_loss
                
                is_best = True
                patience_counter += 1
                print(f"    Patience: {patience_counter}/{max_patience}")
                    
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{max_patience}")
            
            if quality_gates_passed and not quality_gates_ever_passed:
                quality_gates_ever_passed = True
                patience_counter = 0
                print(f"  Quality gates PASSED for first time! Resetting patience counter.")
                
        else:
            is_best = avg_train_loss < best_val_loss
            if is_best:
                best_val_loss = avg_train_loss
                patience_counter = 0
                print(f"  [OK] New best training LOSS: {best_val_loss:.6f}")
            else:
                patience_counter += 1
        
        # ===== SAVE CHECKPOINT =====
        cvd_weights = criterion_cvd.get_current_weights()
        cvd_normalization = criterion_cvd.get_normalization_constants()
        
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # NOTE: scaler_state_dict removed - bf16 doesn't use GradScaler
            "best_val_loss": best_val_loss,
            "metric_name": "CVDLoss",
            "best_delta_e00": best_delta_e00,
            "val_delta_e00": avg_delta_e00_primary if 'avg_delta_e00_primary' in locals() else 0.0,
            "val_ssim": avg_val_ssim if val_loader is not None else 0.0,
            "val_psnr": avg_val_psnr if val_loader is not None else 0.0,
            "val_msssim_rgb": avg_val_msssim_rgb if 'avg_val_msssim_rgb' in locals() else 0.0,
            "patience_counter": patience_counter,
            "gradient_health": gradient_health,
            "g_ema": g_ema,
            "vanish_streak": vanish_streak,
            "config": config,
            "conditioning_type": config.get("conditioning_type", "adain"),
            # CVDLoss lambda weights (Static Statistics Normalization)
            "cvd_loss_weights": {
                "lambda_mse": cvd_weights['lambda_mse'],
                "lambda_delta_e": cvd_weights['lambda_delta_e'],
                "lambda_ssim": cvd_weights['lambda_ssim'],
                "severity": cvd_weights['severity'],
                # Backward compatibility aliases
                "alpha": cvd_weights['lambda_mse'],
                "beta": cvd_weights['lambda_delta_e'],
                "gamma": cvd_weights['lambda_ssim']
            },
            # M_init normalization constants (CRITICAL for resume)
            "cvd_normalization_constants": {
                "M_mse": cvd_normalization['M_mse'],
                "M_delta_e": cvd_normalization['M_delta_e'],
                "M_ssim": cvd_normalization['M_ssim'],
                "is_calibrated": cvd_normalization['is_calibrated']
            },
            # Profile normalization stats (CRITICAL for inference)
            "profile_normalization": train_profile_stats,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "precision_key": precision_key,
            "amp_dtype": str(amp_dtype) if amp_dtype else None
        }
        
        # ===== CHECKPOINT SAVING STRATEGY =====
        # Best model: always save when is_best
        if is_best:
            best_checkpoint_path = checkpoints_dir / f"{config['experiment_name']}_cvd_best.pth"
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"  [OK] Best model saved: {best_checkpoint_path}")
        
        # Periodic checkpoints: 
        # - Every epoch until epoch 20 (1-indexed: epochs 1-20)
        # - Every 2 epochs after epoch 20
        # - NO deletion (keep all checkpoints)
        save_periodic = False
        if epoch < 20:
            # Save every epoch for first 20 epochs
            save_periodic = True
        else:
            # After epoch 20, save every 2 epochs (epoch 20, 22, 24, ...)
            save_periodic = (epoch % 2 == 0)
        
        if save_periodic:
            periodic_checkpoint_path = checkpoints_dir / f"{config['experiment_name']}_cvd_ep{epoch:04d}.pth"
            torch.save(checkpoint_data, periodic_checkpoint_path)
            print(f"  [OK] Checkpoint saved: {periodic_checkpoint_path}")
            # NOTE: No rotation/deletion - keep all checkpoints
        
        # ===== EARLY STOPPING CHECK =====
        # LOGICA A 3 CONDIZIONI:
        # 1. Almeno min_epochs completate
        # 2. Patience esaurita (nessun miglioramento per max_patience epochs)
        # 3. Delta-E00 SOTTO la soglia clinica (early_stopping_delta_e_threshold)
        #
        # Se ΔE00 è SOPRA la soglia, il training NON si ferma MAI
        # (il modello deve continuare a migliorare fino a raggiungere la qualità clinica)
        
        delta_e_below_threshold = avg_delta_e00_primary < early_stopping_delta_e_threshold
        
        if epoch >= min_epochs and patience_counter >= max_patience and delta_e_below_threshold:
            psnr_satisfied = avg_val_psnr >= min_psnr_threshold
            ssim_satisfied = avg_val_ssim >= min_ssim_threshold
            
            print(f"\n{'='*70}")
            print(f"[EARLY STOP] Training terminated at epoch {epoch}")
            print(f"{'='*70}")
            print(f" Reason: Delta-E2000 stagnation AND below clinical threshold")
            print(f" Delta-E2000: {avg_delta_e00_primary:.4f} < {early_stopping_delta_e_threshold:.2f} (threshold)")
            print(f" Min epochs: {min_epochs} (satisfied)")
            print(f" Patience: {patience_counter}/{max_patience} (exhausted)")
            print(f"")
            print(f" BEST MODEL METRICS:")
            print(f"   Delta-E2000: {best_val_loss:.6f}")
            print(f"")
            print(f" CURRENT METRICS (epoch {epoch}):")
            print(f"   Delta-E2000: {avg_delta_e00_primary:.6f}")
            print(f"   SSIM: {avg_val_ssim:.4f} (quality gate: {'PASS' if ssim_satisfied else 'FAIL'})")
            print(f"   PSNR: {avg_val_psnr:.2f}dB (quality gate: {'PASS' if psnr_satisfied else 'FAIL'})")
            print(f"{'='*70}\n")
            break
        elif epoch >= min_epochs and patience_counter >= max_patience and not delta_e_below_threshold:
            # Patience esaurita ma Delta-E00 ancora sopra soglia: continua
            print(f"  [] Patience exhausted but ΔE00 ({avg_delta_e00_primary:.4f}) > threshold ({early_stopping_delta_e_threshold:.2f})")
            print(f"      Training continues until ΔE00 < {early_stopping_delta_e_threshold:.2f}")
    
    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print(f" CVD TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best CVD Loss: {best_val_loss:.6f}")
    print(f"Final epoch: {epoch}")
    print(f"Model saved in: {checkpoints_dir}")
    
    # Save final checkpoint
    final_checkpoint_path = checkpoints_dir / f"{config['experiment_name']}_cvd_final.pth"
    
    cvd_weights = criterion_cvd.get_current_weights()
    cvd_normalization = criterion_cvd.get_normalization_constants()
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "final_val_loss": avg_val_loss,
        "config": config,
        # CVDLoss lambda weights (Static Statistics Normalization)
        "cvd_loss_weights": {
            "lambda_mse": cvd_weights['lambda_mse'],
            "lambda_delta_e": cvd_weights['lambda_delta_e'],
            "lambda_ssim": cvd_weights['lambda_ssim'],
            "severity": cvd_weights['severity'],
            # Backward compatibility aliases
            "alpha": cvd_weights['lambda_mse'],
            "beta": cvd_weights['lambda_delta_e'],
            "gamma": cvd_weights['lambda_ssim']
        },
        # M_init normalization constants
        "cvd_normalization_constants": {
            "M_mse": cvd_normalization['M_mse'],
            "M_delta_e": cvd_normalization['M_delta_e'],
            "M_ssim": cvd_normalization['M_ssim'],
            "is_calibrated": cvd_normalization['is_calibrated']
        }
    }, final_checkpoint_path)
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    print(f"  CVDLoss λ: λ_mse={cvd_weights['lambda_mse']:.3f}, λ_de={cvd_weights['lambda_delta_e']:.3f}, λ_ssim={cvd_weights['lambda_ssim']:.3f}")
    print(f"  M_init: M_mse={cvd_normalization['M_mse']:.4f}, M_delta_e={cvd_normalization['M_delta_e']:.4f}, M_ssim={cvd_normalization['M_ssim']:.4f}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_val_loss': avg_val_loss,
        'best_checkpoint_path': str(output_dir / f"{config['experiment_name']}_cvd_best.pth"),
        'final_checkpoint_path': str(final_checkpoint_path),
        'epochs_trained': epoch
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CVD Compensation Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    
    # Hardware info and final configuration summary
    cpu_cores = os.cpu_count()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    print(f"\n{'='*80}")
    print(f" FINAL TRAINING CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    
    # Hardware Summary
    print(f" HARDWARE DETECTED:")
    print(f"   - CPU: {cpu_cores} cores")
    print(f"   - GPU: {gpu_name}")
    print(f"   - GPU Memory: {gpu_memory:.1f} GB")
    hardware_type = cfg.get('hardware_detected', 'Unknown')
    print(f"   - System Type: {hardware_type}")
    
    # Training Configuration
    print(f"\n  TRAINING CONFIGURATION:")
    print(f"   - Batch Size: {cfg.get('batch_size', 'Not set')}")
    print(f"   - Workers: {cfg.get('num_workers', 'auto-detect')}")
    print(f"   - Prefetch Factor: {cfg.get('prefetch_factor', 2)}")
    print(f"   - Mixed Precision (AMP): {cfg.get('use_amp', True)} (dtype: {cfg.get('amp_dtype', 'auto')})")
    print(f"   - Pin Memory: {cfg.get('pin_memory', True)}")
    print(f"   - Persistent Workers: {cfg.get('persistent_workers', True)}")
    
    # Dataset Configuration
    dataset_path_display = cfg.get('dataset_path_recolored') or cfg.get('dataset_path', 'Not set')
    print(f"\n DATASET CONFIGURATION:")
    print(f"   - Dataset Path (CVD): {dataset_path_display}")
    print(f"   - Train Fraction: {cfg.get('FRACTION_IMAGES_TRAIN', 'Not set')}")
    print(f"   - Val Fraction: {cfg.get('FRACTION_IMAGES_VAL', 'Not set')}")
    
    # Model Configuration
    print(f"\n MODEL CONFIGURATION:")
    print(f"   - Architecture: {cfg.get('architecture', 'CVDCompensationModelAdaIN')}")
    print(f"   - Encoder: {cfg.get('encoder_type', 'convnext_tiny')}")
    print(f"   - Target Resolution: {cfg.get('target_resolution', 256)}")
    print(f"   - Learning Rate: {cfg.get('learning_rate', 'Not set')}")
    print(f"   - Encoder LR: {cfg.get('encoder_learning_rate', 'Not set')}")
    print(f"   - Weight Decay: {cfg.get('weight_decay', 'Not set')}")

    # Output Configuration
    print(f"\n OUTPUT CONFIGURATION:")
    print(f"   - Experiment: {cfg.get('experiment_name', 'Not set')}")
    print(f"   - Output Dir: {cfg.get('output_dir', 'Not set')}")
    print(f"   - Epochs: {cfg.get('epochs', 'Not set')}")
    print(f"   - Patience: {cfg.get('patience', 'Not set')}")
    
    print(f"{'='*80}")
    print(f" Starting training with above configuration...")
    print(f"{'='*80}\n")

    # Run training
    result = train_cvd_model(cfg)

    # Print final summary
    print("--- Final Summary ---")
    for k, v in result.items():
        print(f"{k}: {v}")
