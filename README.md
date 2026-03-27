# CVD Chromatic Compensation

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-GPLv3-blue)

🇬🇧 English | [🇮🇹 Italiano](README.it.md)

Real-time chromatic compensation system for users with **Color Vision Deficiency** (CVD), based on deep learning with adaptive conditioning (CVD-AdaIN).

> **Master's Thesis in Computer Engineering** — University of Palermo

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Dataset Generation](#dataset-generation)
7. [Training](#training)
8. [Inference](#inference)
9. [Results](#results)
10. [License](#license)

---

## Overview

This project implements an end-to-end pipeline for:

1. **User profiling** — Farnsworth-Munsell 100 Hue Test to characterize CVD type and severity into a 3D vector `[θ, C, S]`
2. **Dataset generation** — Optimal offline compensation via Farup's Teacher algorithm (anisotropic variational in gradient domain)
3. **Training** — CVD profile-conditioned neural network that learns to replicate the Teacher in real time
4. **Inference** — Single-image compensation in ~45 ms on GPU (Tkinter GUI or CLI)

The model receives an RGB image and the user's CVD profile, and returns the compensated image while preserving the original Y' luma (Y'-preserving in YCbCr BT.601 space).

---

## Theoretical Background

### Color Vision Deficiency (CVD)

CVD affects approximately 8% of the male population and 0.5% of the female population. It is classified as:

| Type | Affected cone | Confusion axis |
|------|--------------|----------------|
| **Protanopia/Protanomaly** | L (long, red) | Red-green |
| **Deuteranopia/Deuteranomaly** | M (medium, green) | Red-green |
| **Tritanopia/Tritanomaly** | S (short, blue) | Blue-yellow |

Severity ranges from mild (anomaly) to total (anopia). CVD simulation uses **Machado et al. (2009) matrices**, which model the reduction in cone spectral sensitivity as a severity-parameterized linear transformation in RGB space.

### CVD Profile: vector `[θ, C, S]`

The individual profile is extracted from the **Farnsworth-Munsell 100 Hue Test** via Vingrys–King-Smith quantitative scoring and encoded as:

<p align="center">
  <img src="assets/fm100_gui.png" alt="Farnsworth-Munsell 100 Hue Test GUI" width="550">
</p>

- **θ (theta)** — Confusion axis angle in the CIELUV $(u^*,v^*)$ chromaticity plane (in degrees). Indicates the dominant direction of the error pattern: distinguishes protan, deutan, and tritan profiles
- **C (Confusion index)** — Measures the overall magnitude of the error (global severity)
- **S (Scatter index)** — Measures the selectivity and directionality of the error pattern

### Teacher Algorithm: Farup (anisotropic variational)

The gold standard for offline compensation uses Farup's algorithm, operating in **linear RGB** in the gradient domain:
- From the CVD profile, an orthonormal chromatic basis $(e_\ell, e_d, e_c)$ is derived in linear RGB, where $e_d$ is the confusion direction and $e_c$ the direction of maximum visibility
- A target gradient field $\mathbf{G} = \nabla u_0 + (\nabla u_0 \cdot e_d)\, e_c$ is constructed, redirecting contrast components from the "blind" direction to the "visible" one
- An anisotropic variational solver (GDIP) reconstructs the compensated image by minimizing the deviation from the target field, with edge-preserving regularization

**Limitation**: the Teacher takes seconds to minutes per image → not usable in real time. The neural network is trained to replicate its output in milliseconds.

---

## Model Architecture

> For all available but unused architectural options, see [ARCHITECTURE_OPTIONS.md](ARCHITECTURE_OPTIONS.md).

<p align="center">
  <img src="assets/architecture.png" alt="Full model architecture" width="750">
</p>

### `CVDCompensationModelAdaIN`

```
Image [B, 3, 256, 256]   +   CVD Profile [B, 3]
         ↓                          ↓
┌─────────────────────────────────────────────────┐
│       ENCODER (ConvNeXt-Tiny, ImageNet-1k)      │
│  CVDAdaIN at 17 points (15 blocks + 2 downsa.)  │
│  Stem frozen, rest fine-tuned (dedicated lr)     │
└──────────────────┬──────────────────────────────┘
                   ↓
          Latent [B, 384, 16, 16]
                   ↓
┌─────────────────────────────────────────────────┐
│       DECODER PLCF (9 CVDAdaIN + tanh head)     │
│  Progressive nearest-neighbor upsampling         │
└──────────────────┬──────────────────────────────┘
                   ↓
           ΔCbCr [B, 2, 256, 256]
                   ↓
┌─────────────────────────────────────────────────┐
│           Y'-PRESERVING OUTPUT                  │
│  Y'_out = Y'_in       (BT.601 luma copied)     │
│  Cb_out = Cb_in + ΔCb × 0.9                    │
│  Cr_out = Cr_in + ΔCr × 0.9                    │
└─────────────────────────────────────────────────┘
```

**CVDAdaIN** (CVD Adaptive Normalization): the 3D CVD profile is projected by a Linear layer into `(γ, β)` parameters that modulate normalizations in the encoder (17 points, LayerNorm type) and decoder (9 points, Instance Norm type). The projection is initialized near-identity (γ≈1, β≈0) so that conditioning emerges gradually during training. The network dynamically adapts to each user's deficiency type and severity.

<p align="center">
  <img src="assets/decoder_detail.png" alt="Detailed PLCF decoder architecture" width="650">
</p>

**Y'-Preserving**: the decoder outputs only 2 channels (ΔCb, ΔCr). The input Y' luma (BT.601) is copied unchanged to the output, limiting brightness variations in the image domain. Note: Y' is an operational quantity (luma), distinct from CIE colorimetric luminance Y and perceptual lightness L*.

### CVD Profile Normalization

The `[θ, C, S]` profile is normalized with a **hybrid** strategy:
- **θ**: **global** normalization — preserves the distinction between CVD types
- **C, S**: **per-CVD-type** normalization — handles the different severity distributions for Protan/Deutan/Tritan

Statistics are saved in the checkpoint and automatically applied at inference.

### Loss Function

Two-component loss with static normalization via calibration constants:

$$\mathcal{L} = 0.7 \cdot \frac{\text{MSE}_{a^{*}b^{*}}}{M_{\text{mse}}} + 0.3 \cdot \frac{(1 - \text{MS-SSIM}_{\text{RGB}})}{M_{\text{ssim}}}$$

| Component | Space | Weight | Description |
|-----------|-------|--------|-------------|
| MSE a\*b\* | CIELAB | 0.7 | Chrominance error in perceptual a\*, b\* channels |
| MS-SSIM | sRGB | 0.3 | Multi-scale structural preservation |

The $M$ constants are automatically calibrated on the first ~200 training samples and saved in `calibration_constants_*.json`. ΔE2000 (CIEDE2000) is computed as a **validation metric** but is not included in the loss.

---

## Project Structure

```
.
├── z__pipeline.py                # Interactive CLI menu: Dataset → Training → Inference
├── z__inference_gui.py           # Tkinter GUI for single-image compensation
├── inference_core.py             # Shared inference module
├── train.py                      # Training script (--config <yaml>)
├── FM_TEST.py                    # Farnsworth-Munsell 100 Hue Test (PyQt5)
├── get_profile_feats.py          # CVD profile extraction from FM100
├── config_generator.py           # YAML configuration generation
│
├── CVDCompensationModelAdaIN.py  # Main model
├── PLCFEncoderCVD.py             # ConvNeXt-Tiny encoder + CVD-AdaIN
├── PLCFDecoderCVD.py             # PLCF decoder + CVD-AdaIN
├── cvd_adain_modules.py          # CVD-AdaIN modules
│
├── losses.py                     # CVDLoss (MSE a*b* + MS-SSIM)
├── metrics.py                    # SSIM, PSNR
├── delta_e_ciede2000_torch.py    # CIEDE2000 in PyTorch
├── color_space_utils.py          # RGB ↔ YCbCr ↔ Lab
├── cvd_dataset_loader.py         # PyTorch Dataset
├── cvd_simulator.py              # CVD simulation (Machado 2009)
├── cvd_shared_cache.py           # Machado matrices cache
├── cvd_cache_optimizer.py        # JSON profile cache
├── cvd_constants.py              # Shared constants
├── teacher_farup_full.py         # Teacher algorithm (CPU)
├── teacher_farup_gpu.py          # Teacher algorithm (GPU)
├── mapping_x_to_T.py             # Clinical mapping (θ,C,S) → CVD type
├── train_utility.py              # Training utilities
├── simple_logger.py              # CSV logger + plots
├── precision_utils.py            # CUDA precision detection
│
├── configs/                      # YAML configurations
├── CVD_dataset_generator/        # Dataset generation pipeline
├── results/                      # Best checkpoint + calibration (Git LFS)
├── variational-anisotropic-gradient-domain-main/
│                                 # External dependency (GPL v3) for Teacher Farup
├── ARCHITECTURE_OPTIONS.md       # Available architectural options (documentation)
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.x with NVIDIA GPU
- ~200 MB disk space for the checkpoint (via Git LFS)

### Setup

```bash
# 1. Create conda environment
conda create -n cvd python=3.11 -y
conda activate cvd

# 2. Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

| Package | Ref. version | Usage |
|---------|-------------|-------|
| `torch` | 2.7.0+cu128 | Deep learning framework |
| `torchvision` | 0.22.0+cu128 | ConvNeXt-Tiny pretrained backbone |
| `timm` | 1.0.19 | Model registry |
| `numpy` | 2.3.2 | Numerical arrays |
| `colour-science` | 0.4+ | Machado 2009 matrices |
| `pytorch-msssim` | 1.0+ | MS-SSIM in loss |
| `torchmetrics` | 1.0+ | SSIM / PSNR |
| `scipy` | 1.16.1 | Variational solver (dataset generation only) |
| `matplotlib` | 3.9.1 | Plotting |
| `PyQt5` | 5.15+ | _Optional_ — FM100 Test GUI |

---

## Dataset Generation

The dataset is generated by compensating **Places365** images with the Teacher Farup for simulated CVD profiles.

```bash
python z__pipeline.py --dataset
```

For each source image and randomly generated CVD profile:
1. Simulates CVD on the image (Machado 2009)
2. Applies Farup's variational algorithm for optimal compensation
3. Saves the (original, compensated) pair with CVD profile metadata in JSON

Resulting dataset: **94,321 training images**, **20,048 validation images**.

> **Note:** The pre-generated dataset is not included in the repository as it exceeds 10 GB.
> To request the dataset, open an [Issue](https://github.com/googlielmo93/cvdadan-personalized-color-compensation/issues) on this repository.

---

## Training

### Hardware

| | Specification |
|--|---------------|
| **GPU** | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| **Precision** | bfloat16 (mixed precision) |

### Launch

```bash
# Interactive menu
python z__pipeline.py --training

# Direct
python train.py --config "configs/grid_cvd_20251212_215310/config_01_no_delta_e.yaml"
```

Training features: auto-resume from checkpoint, early stopping (patience 20 on Val ΔE00, with ΔE00 < 5.0 threshold and quality gates on SSIM/PSNR), ReduceLROnPlateau (factor 0.7, patience 15, min_lr 5×10⁻⁶), automatic calibration of loss constants on the first ~200 samples.

### Results

| Metric | Value |
|--------|-------|
| **Total epochs** | 187 (early stopping) |
| **Best epoch** | 185 |
| **Val ΔE00** | **1.27** (CIEDE2000) |
| **Val SSIM** | **0.989** |
| **Val PSNR** | **~37.2 dB** |

The checkpoint (epoch 185, 173.1 MB) is included in the repository via Git LFS.

---

## Inference

### GUI (recommended)

```bash
python z__inference_gui.py
```

Tkinter interface with 4 tabs:
1. **CVD Profile** — Manual input of θ, C, S or JSON loading from FM_TEST.py
2. **Image** — Image selection
3. **Compensation** — Checkpoint selection and processing
4. **Result** — Side-by-side comparison with save option

### CLI

```bash
python z__pipeline.py --inference
```

### User Profiling

To obtain a real user's CVD profile:

```bash
python FM_TEST.py
```

Requires PyQt5. Produces a JSON file with `[θ, C, S]` directly usable in the inference GUI.

---

## Results

The model achieves a **mean ΔE00 of 1.27** on validation (human perceptibility threshold ≈ 1.0), with SSIM of 0.989 and PSNR of ~37.2 dB.

Compensation preserves **Y' luma** (BT.601, copied identically from input) and modifies only chrominance (Cb, Cr), limiting brightness variations in the image domain.

---

## License

The `variational-anisotropic-gradient-domain-main/` folder is distributed under **GPL v3** license (see its LICENSE file). It is required only for dataset generation (Teacher Farup), not for inference.
