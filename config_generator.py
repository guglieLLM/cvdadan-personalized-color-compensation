"""
config_generator — Generazione di configurazioni YAML per il training.

Genera configurazioni per CVDCompensationModelAdaIN con CVDLoss.
La configurazione predefinita (``config_01_no_delta_e``) usa loss a 2
componenti: MSE a*b* (λ=0.7) + MS-SSIM (λ=0.3), ΔE2000 è disabilitata
(λ=0.0) e viene calcolata solo come metrica di validazione.
"""

import os
import csv
import yaml
from pathlib import Path
import datetime
import argparse

# ======================================= PATHS (HARDCODED) =======================================
# Dataset paths per CVD training (Teacher-based Farup 2020 GDIP)
# ORIGINAL: immagini Places365 originali
DATASET_PATH_ORIGINAL = "dataset/places365/subsets/subsets_derived/subsets_t_0.15_v_1"
# RECOLORED: immagini CVD generate + JSON mapping files
DATASET_PATH_RECOLORED = "dataset/places365/subsets/subsets_derived_recolored_CVD/subsets_teacher_compensated_t_0.15_v_1"

# ======================================= CONFIGURATION =======================================

timestamp_sec = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"\n{'='*80}")
print(f"CVD Training Configuration Generator")
print(f"Architecture: CVDCompensationModelAdaIN (ConvNeXt-Tiny + CVDAdaIN)")
print(f"{'='*80}\n")

# Shared configuration (common to all experiments)
shared_config = {
    # Architecture
    "architecture": "CVDCompensationModelAdaIN",
    "encoder_type": "convnext_tiny",
    "conditioning_type": "cvdadain",
    
    # Y-Preserving Architecture (luminance preservation)
    # When True: decoder outputs ΔCb, ΔCr (2 channels), Y copied from input
    # When False: decoder outputs ΔRGB (3 channels), standard mode
    "y_preserving": True,  # RECOMMENDED: prevents white image problem
    
    # Dataset paths
    "dataset_path_original": DATASET_PATH_ORIGINAL,
    "dataset_path_recolored": DATASET_PATH_RECOLORED,
    
    # Training parameters
    "epochs": 2500,
    "patience": 20,
    "min_epochs": 20,
    "early_stopping_delta_e_threshold": 5.0,  # Soglia ΔE00: sotto questa E no improvement → STOP
    "warmup_epochs": 5,
    "test_every": 5,
    
    # Dataset fractions
    "FRACTION_IMAGES_TRAIN": 0.15,
    "FRACTION_IMAGES_VAL": 1.0,
    "FRACTION_IMAGES_TEST": 0.0005,
    
    # DataLoader performance
    "num_workers": 4,           # Parallel data loading workers
    "prefetch_factor": 2,       # Batches prefetched per worker
    "pin_memory": True,         # Pin memory for faster GPU transfer
    "persistent_workers": True, # Keep workers alive between epochs
    
    # Mixed Precision Training (bf16 for RTX 3090 Ampere)
    "use_amp": True,            # Automatic Mixed Precision
    "amp_dtype": "bfloat16",    # bf16 for Ampere+ GPUs (no GradScaler needed)
    
    # torch.compile (DISABLED - saves ~3.8GB VRAM from CUDA Graphs)
    "use_torch_compile": False,
    
    # Resolution
    "target_resolution": 256,  # Image resolution (256x256 matches dataset generation)
    
    # Reproducibility
    "seed": 42,
    "debug_verbose": False,
    
    # Checkpointing
    "checkpoint_every": 1,
    "checkpoint_keep_last_k": 40,
    "auto_resume": True,
    "resume_from": "",
    "resume_restore_rng": True,
    
    # Scheduler (force new params on resume)
    "force_new_scheduler_params": False,
}


# ======================================= EXPERIMENT CONFIGURATIONS =======================================
# CVDCompensationModelAdaIN with CVDLoss (MSE a*b*, Delta-E2000, MS-SSIM RGB)

experiment_configs = [
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURATION 1: NO DELTA-E (2-Component Loss) - PRIORITARIO
    # ═══════════════════════════════════════════════════════════════════
    # CVDLoss weights: α'=0.7 (MSE a*b*), β=0.0 (ΔE2000 disabled), γ'=0.3 (MS-SSIM RGB)
    # ΔE2000 rimane calcolato solo come metrica di validazione, non contribuisce alla loss
    # Riferimento: Standard colorization/recoloring setup (L1/L2 LAB + SSIM)
    {
        "name": "no_delta_e",
        "description": "2-component loss without Delta-E2000 (MSE a*b* + MS-SSIM RGB only)",
        
        # Learning rates (same as balanced)
        "learning_rate": 3e-5,
        "encoder_learning_rate": 1e-4,
        
        # Scheduler (ReduceLROnPlateau)
        "lr_factor": 0.7,
        "lr_patience": 15,
        "min_lr": 5e-6,
        
        # Training
        "weight_decay": 1e-4,
        "batch_size": 32,
        "max_norm": 0.5,
        
        # CVDLoss lambda weights (2-component: MSE + SSIM only)
        # ΔE2000 è escluso dalla loss ma calcolato per validazione/logging
        "cvd_lambda_mse": 0.7,            # α' - MSE a*b* (driver principale cromaticità)
        "cvd_lambda_delta_e": 0.0,        # β = 0 (ΔE2000 solo come metrica, non in loss)
        "cvd_lambda_ssim": 0.3,           # γ' - MS-SSIM RGB (struttura/qualità visiva)
        "cvd_warmup_samples": 200,
        
        # Early stopping
        "patience": 20,
        "min_improvement": 0.001,
        "min_epochs": 20,
        "early_stopping_delta_e_threshold": 5.0,  # Soglia clinica: STOP solo se ΔE00 < 2.0
        
        # Quality gates (ΔE threshold usato solo per validazione)
        "min_ssim_threshold": 0.60,
        "min_psnr_threshold": 20.0,
        "delta_e00_threshold": 3.0,
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURATION 2: BALANCED (3-Component Loss)
    # ═══════════════════════════════════════════════════════════════════
    # CVDLoss weights: α=0.45 (MSE a*b*), β=0.30 (ΔE2000), γ=0.25 (MS-SSIM RGB)
    {
        "name": "balanced",
        "description": "Balanced CVDLoss with equal emphasis on color and structure",
        
        # Learning rates (differential for encoder fine-tuning)
        "learning_rate": 3e-5,
        "encoder_learning_rate": 1e-4,
        
        # Scheduler (ReduceLROnPlateau)
        "lr_factor": 0.7,
        "lr_patience": 15,
        "min_lr": 5e-6,
        
        # Training
        "weight_decay": 1e-4,
        "batch_size": 32,
        "max_norm": 0.5,
        
        # CVDLoss lambda weights (Static Statistics Normalization)
        # I lambda sono pesi di priorità, la normalizzazione delle scale è automatica (M_init)
        "cvd_lambda_mse": 0.45,           # α - MSE a*b* (fedeltà cromatica pixel-wise)
        "cvd_lambda_delta_e": 0.30,       # β - ΔE2000 (fedeltà percettiva)
        "cvd_lambda_ssim": 0.25,          # γ - MS-SSIM RGB (struttura/qualità visiva)
        "cvd_warmup_samples": 200,
        
        # Early stopping
        "patience": 20,
        "min_improvement": 0.001,
        "min_epochs": 20,
        "early_stopping_delta_e_threshold": 5.0,  # Soglia clinica: STOP solo se ΔE00 < 2.0
        
        # Quality gates
        "min_ssim_threshold": 0.60,
        "min_psnr_threshold": 20.0,
        "delta_e00_threshold": 3.0,
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURATION 3: PERCEPTUAL FOCUS (Delta-E2000 Emphasis)
    # ═══════════════════════════════════════════════════════════════════
    # CVDLoss weights: α=0.40, β=0.35, γ=0.25
    {
        "name": "perceptual",
        "description": "Delta-E2000 emphasis for maximum perceptual color accuracy",
        
        # Learning rates
        "learning_rate": 5e-5,
        "encoder_learning_rate": 2e-4,
        
        # Scheduler
        "lr_factor": 0.6,
        "lr_patience": 10,
        "min_lr": 3e-6,
        
        # Training
        "weight_decay": 8e-5,
        "batch_size": 32,
        "max_norm": 0.7,
        
        # CVDLoss lambda weights (perceptual focus - Delta-E slightly higher)
        "cvd_lambda_mse": 0.40,           # α - MSE a*b*
        "cvd_lambda_delta_e": 0.35,       # β - ΔE2000 (leggermente più alto)
        "cvd_lambda_ssim": 0.25,          # γ - MS-SSIM RGB
        "cvd_warmup_samples": 200,
        
        # Early stopping (more reactive)
        "patience": 15,
        "min_improvement": 0.002,
        "min_epochs": 15,
        "early_stopping_delta_e_threshold": 5.0,  # Soglia clinica: STOP solo se ΔE00 < 2.0
        
        # Quality gates (tighter)
        "min_ssim_threshold": 0.60,
        "min_psnr_threshold": 20.0,
        "delta_e00_threshold": 2.5,
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURATION 4: CONSERVATIVE (Stability Focus)
    # ═══════════════════════════════════════════════════════════════════
    # CVDLoss weights: α=0.32, β=0.36, γ=0.32
    {
        "name": "conservative",
        "description": "Conservative learning for maximum stability",
        
        # Learning rates (conservative)
        "learning_rate": 2e-5,
        "encoder_learning_rate": 5e-5,
        
        # Scheduler (patient)
        "lr_factor": 0.75,
        "lr_patience": 18,
        "min_lr": 7e-6,
        
        # Training
        "weight_decay": 1.2e-4,
        "batch_size": 24,
        "max_norm": 0.3,
        
        # CVDLoss lambda weights (balanced/conservative)
        "cvd_lambda_mse": 0.32,
        "cvd_lambda_delta_e": 0.36,
        "cvd_lambda_ssim": 0.32,
        "cvd_warmup_samples": 200,
        
        # Early stopping (patient)
        "patience": 25,
        "min_improvement": 0.0015,
        "min_epochs": 25,
        "early_stopping_delta_e_threshold": 5.0,  # Soglia clinica: STOP solo se ΔE00 < 2.0
        
        # Quality gates (permissive)
        "min_ssim_threshold": 0.65,
        "min_psnr_threshold": 22.0,
        "delta_e00_threshold": 3.5,
    },
]


def generate_configs():
    """
    Generate YAML configuration files for each experiment.
    
    Directory structure (like old lanch_grid.py):
        configs/grid_cvd_<timestamp>/           <- YAML config files
        results/cvd/grid_cvd_<timestamp>/       <- Results per experiment
            └── <experiment_name>/
                ├── config_copy_<experiment>.yaml
                ├── checkpoints/
                └── ...
        logs/grid_cvd_<timestamp>/              <- Logs per experiment
            └── <experiment_name>/
                └── training.log
    """
    
    # ====== DIRECTORY STRUCTURE (phase-independent) ======
    configs_dir = Path(f"configs/grid_cvd_{timestamp_sec}")
    results_dir = Path(f"results/cvd/grid_cvd_{timestamp_sec}")
    logs_dir_base = Path(f"logs/grid_cvd_{timestamp_sec}")
    
    # Create base directories
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir_base.mkdir(parents=True, exist_ok=True)
    
    # CSV summary file
    csv_path = results_dir / "experiments_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_name", "config_type", "learning_rate", "encoder_lr", 
            "batch_size", "weight_decay", "lambda_mse", "lambda_delta_e", "lambda_ssim",
            "patience", "delta_e00_threshold", "seed"
        ])
    
    generated_files = []
    
    for i, exp_config in enumerate(experiment_configs, 1):
        # Merge shared config with experiment-specific config
        config = {**shared_config, **exp_config}
        
        # Generate experiment name with full details
        exp_name = f"cvd_{exp_config['name']}_lr{exp_config['learning_rate']:.0e}_bs{exp_config['batch_size']}_{timestamp_sec}"
        experiment_name_safe = exp_name.replace("/", "_").replace("\\", "_")[:200]
        config["experiment_name"] = experiment_name_safe
        
        # ====== PER-EXPERIMENT DIRECTORIES ======
        # Output directory for results (checkpoints, images, etc.)
        output_dir = results_dir / experiment_name_safe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoints subdirectory
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Log directory for this experiment
        log_dir_experiment = logs_dir_base / experiment_name_safe
        log_dir_experiment.mkdir(parents=True, exist_ok=True)
        
        # Plots subdirectory
        plots_dir = log_dir_experiment / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set paths in config
        config["output_dir"] = str(output_dir)
        config["checkpoints_dir"] = str(checkpoints_dir)
        config["log_file"] = str(log_dir_experiment / "training.log")
        config["plots_dir"] = str(plots_dir)
        
        # ====== WRITE YAML CONFIG ======
        yaml_path = configs_dir / f"config_{i:02d}_{exp_config['name']}.yaml"
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"# CVD Training Configuration\n")
            f.write(f"# {exp_config['description']}\n")
            f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"#\n")
            f.write(f"# Architecture: CVDCompensationModelAdaIN (ConvNeXt-Tiny + CVDAdaIN)\n")
            # Dynamic loss description based on lambda_delta_e
            if config['cvd_lambda_delta_e'] == 0:
                f.write(f"# Loss: CVDLoss 2-component (MSE a*b* + MS-SSIM RGB, ΔE2000 only for validation)\n")
            else:
                f.write(f"# Loss: CVDLoss 3-component (MSE a*b*, Delta-E2000, MS-SSIM RGB)\n")
            f.write(f"# Lambda weights: λ_mse={config['cvd_lambda_mse']}, λ_de={config['cvd_lambda_delta_e']}, λ_ssim={config['cvd_lambda_ssim']}\n")
            f.write(f"# Normalization: Static Statistics (M_init calibration)\n")
            f.write(f"#\n\n")
            
            output_config = {k: v for k, v in config.items() if k not in ['name', 'description']}
            yaml.dump(output_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Save config copy in output directory (for traceability)
        yaml_copy_path = output_dir / f"config_copy_{experiment_name_safe}.yaml"
        with open(yaml_copy_path, "w", encoding="utf-8") as f:
            yaml.dump(output_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # ====== UPDATE CSV SUMMARY ======
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                experiment_name_safe,
                exp_config['name'],
                f"{exp_config['learning_rate']:.0e}",
                f"{exp_config['encoder_learning_rate']:.0e}",
                exp_config['batch_size'],
                f"{exp_config['weight_decay']:.0e}",
                config['cvd_lambda_mse'],
                config['cvd_lambda_delta_e'],
                config['cvd_lambda_ssim'],
                config['patience'],
                config.get('delta_e00_threshold', 3.0),
                config['seed']
            ])
        
        generated_files.append(yaml_path)
        print(f"[{i}/{len(experiment_configs)}] Generated: {yaml_path}")
        print(f"    Output: {output_dir}")
        print(f"    Logs:   {log_dir_experiment}")
    
    print(f"\n{'='*80}")
    print(f"Generated {len(generated_files)} configuration files")
    print(f"{'='*80}")
    print(f"  Configs:  {configs_dir}")
    print(f"  Results:  {results_dir}")
    print(f"  Logs:     {logs_dir_base}")
    print(f"  Summary:  {csv_path}")
    print(f"{'='*80}\n")
    
    print("Usage:")
    print(f"  python train.py --config {generated_files[0]}")
    print()
    
    return configs_dir, generated_files


def create_single_config(output_path=None, **overrides):
    """
    Create a single configuration file with optional overrides.
    
    Creates proper directory structure:
        results/cvd/<experiment_name>/
            ├── config_copy_<experiment>.yaml
            ├── checkpoints/
            └── ...
        logs/cvd/<experiment_name>/
            ├── training.log
            └── plots/
    
    Args:
        output_path: Path for output YAML file (default: configs/config_cvd.yaml)
        **overrides: Override any configuration parameter
    
    Returns:
        Path to generated config file
    """
    # Start with balanced config as default
    config = {**shared_config, **experiment_configs[0]}  # Use balanced as base
    
    # Apply overrides
    config.update(overrides)
    
    # Generate experiment name if not provided
    if "experiment_name" not in overrides:
        config["experiment_name"] = f"cvd_custom_{timestamp_sec}"
    
    experiment_name_safe = config["experiment_name"].replace("/", "_").replace("\\", "_")[:200]
    config["experiment_name"] = experiment_name_safe
    
    # ====== CREATE DIRECTORY STRUCTURE ======
    # Results directory
    output_dir = Path("results/cvd") / experiment_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoints subdirectory
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Logs directory
    log_dir = Path("logs/cvd") / experiment_name_safe
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Plots subdirectory
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set paths in config
    config["output_dir"] = str(output_dir)
    config["checkpoints_dir"] = str(checkpoints_dir)
    config["log_file"] = str(log_dir / "training.log")
    config["plots_dir"] = str(plots_dir)
    
    # ====== WRITE YAML CONFIG ======
    # Config file path
    if output_path is None:
        configs_dir = Path("configs/cvd")
        configs_dir.mkdir(parents=True, exist_ok=True)
        output_path = configs_dir / f"config_{experiment_name_safe}.yaml"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write main config
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# CVD Training Configuration\n")
        f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"#\n")
        f.write(f"# Architecture: CVDCompensationModelAdaIN (ConvNeXt-Tiny + CVDAdaIN)\n")
        f.write(f"# Loss: CVDLoss 3-component (MSE a*b*, Delta-E2000, MS-SSIM L*)\n")
        f.write(f"#\n\n")
        
        output_config = {k: v for k, v in config.items() if k not in ['name', 'description']}
        yaml.dump(output_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Save config copy in output directory (for traceability)
    yaml_copy_path = output_dir / f"config_copy_{experiment_name_safe}.yaml"
    with open(yaml_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(output_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"\n{'='*80}")
    print(f"Generated configuration: {output_path}")
    print(f"{'='*80}")
    print(f"  Config:       {output_path}")
    print(f"  Output:       {output_dir}")
    print(f"  Checkpoints:  {checkpoints_dir}")
    print(f"  Logs:         {log_dir}")
    print(f"  Plots:        {plots_dir}")
    print(f"{'='*80}\n")
    
    print("Usage:")
    print(f"  python train.py --config {output_path}")
    print()
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CVD training configurations')
    parser.add_argument('--single', action='store_true', help='Generate single config file')
    parser.add_argument('--output', type=str, default=None, help='Output path for single config')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    if args.single:
        # Generate single config with optional overrides
        overrides = {}
        if args.batch_size:
            overrides['batch_size'] = args.batch_size
        if args.lr:
            overrides['learning_rate'] = args.lr
        if args.experiment_name:
            overrides['experiment_name'] = args.experiment_name
        
        create_single_config(args.output, **overrides)
    else:
        # Generate all experiment configs
        generate_configs()
