# PROJECT STRUCTURE — CVD Chromatic Compensation

> Struttura completa dei file nel repository.

## Root — Entry Points & Core

| File | Descrizione |
|------|-------------|
| `z__pipeline.py` | Menu interattivo CLI: Dataset → Training → Inference |
| `z__inference_gui.py` | GUI Tkinter per compensazione immagine singola (4 tab: Profilo, Immagine, Compensazione, Risultato) |
| `inference_core.py` | Modulo condiviso di inferenza (load model, load profile, compensate) |
| `train.py` | Script di training (`--config <yaml>`) |
| `config_generator.py` | Generazione automatica di configurazioni YAML |
| `FM_TEST.py` | Farnsworth-Munsell 100 Hue Test (PyQt5) |
| `get_profile_feats.py` | Estrazione features del profilo cromatico dall'FM100 |

## Root — Model Architecture

| File | Descrizione |
|------|-------------|
| `CVDCompensationModelAdaIN.py` | Modello principale: encoder-decoder + CVDAdaIN |
| `PLCFEncoderCVD.py` | Encoder ConvNeXt-Tiny con CVDAdaIN |
| `PLCFDecoderCVD.py` | Decoder PLCF con CVDAdaIN |
| `cvd_adain_modules.py` | Moduli CVD Adaptive Instance Normalization |

## Root — Training & Loss

| File | Descrizione |
|------|-------------|
| `losses.py` | CVDLoss: MSE a\*b\* + MS-SSIM RGB + ΔE2000 opzionale |
| `metrics.py` | SSIM, PSNR (torchmetrics) |
| `delta_e_ciede2000_torch.py` | CIEDE2000 implementata in PyTorch |
| `train_utility.py` | Utilità di training (scheduler, logging, ecc.) |
| `simple_logger.py` | Logger CSV + generazione plot |
| `precision_utils.py` | Rilevamento precisione CUDA (bf16/fp16/fp32) |

## Root — Data Pipeline

| File | Descrizione |
|------|-------------|
| `cvd_dataset_loader.py` | PyTorch Dataset per coppie (originale, compensata, profilo) |
| `cvd_cache_optimizer.py` | Caching ottimizzato profili JSON |
| `cvd_shared_cache.py` | Cache matrici Machado 2009 (colour-science) |
| `cvd_simulator.py` | Simulazione CVD (Protan / Deutan / Tritan) |
| `cvd_constants.py` | Costanti condivise del progetto |
| `color_space_utils.py` | Conversioni spazi colore (RGB ↔ YCbCr ↔ Lab) |

## Root — Teacher & Dataset Generation Support

| File | Descrizione |
|------|-------------|
| `teacher_farup_full.py` | Algoritmo Teacher Farup (CPU) — compensazione ottimale |
| `teacher_farup_gpu.py` | Versione GPU del Teacher Farup |
| `mapping_x_to_T.py` | Mapping clinico (θ, C, S) → (tipo CVD, severità) |

---

## `configs/grid_cvd_20251212_215310/`

| File | Descrizione |
|------|-------------|
| `config_01_no_delta_e.yaml` | **Config usata** — MSE a\*b\* (α=0.7) + MS-SSIM (γ=0.3), ΔE off |

## `CVD_dataset_generator/`

| File | Descrizione |
|------|-------------|
| `dataset_generator_teacher.py` | Pipeline principale generazione dataset con Teacher Farup |

## `icons/`

| File | Descrizione |
|------|-------------|
| `daltonismo range colori.jpeg` | Immagine range colori daltonismo |
| `info_icon.png` | Icona info per GUI |

## `variational-anisotropic-gradient-domain-main/`

> Dipendenza esterna (GPL v3) — usata da `teacher_farup_full.py`

| File | Descrizione |
|------|-------------|
| `gradient.py` | Calcolo gradienti variazionali anisotropi (richiede scipy) |
| `README.md` | Readme originale |
| `LICENSE` | Licenza GPL v3 |

## `results/` — Esperimento di training (solo best checkpoint + config + calibration)

```
results/cvd/grid_cvd_20251212_215310/
└── cvd_no_delta_e_lr3e-05_bs32_20251212_215310/
    ├── config_copy_*.yaml                          (1.6 KB)
    ├── calibration_constants_*.json                (221 B)
    └── checkpoints/
        └── *_cvd_best.pth                          (173.1 MB — Git LFS)
```

> I checkpoint `*_ep0186.pth` e `*_final.pth` NON vengono pushati (ridondanti).  
> Le altre 3 cartelle esperimento (balanced, conservative, perceptual) contengono solo config_copy senza checkpoint.

---

## Documentazione

| File | Descrizione |
|------|-------------|
| `README.md` | Panoramica del progetto, setup, utilizzo |
| `ARCHITECTURE_OPTIONS.md` | Opzioni architetturali disponibili (varianti non usate, configurazioni alternative) |
| `PROJECT_STRUCTURE.md` | Questo file |
| `requirements.txt` | Dipendenze Python |
| `LICENSE` | Licenza del progetto |

## File **NON** pushati (in `.gitignore`)

| Path | Motivo |
|------|--------|
| `dataset/` | Immagini di training/validation (~94K + 20K immagini) |
| `logs/` | Log di training (riproducibili) |
| `Z_____ALTRO/` | File di supporto/archivio non parte del progetto |
| `__pycache__/`, `*.pyc` | Cache Python |
| `.vscode/` | Configurazione locale VS Code |
| `.conda/` | Ambiente conda locale |
