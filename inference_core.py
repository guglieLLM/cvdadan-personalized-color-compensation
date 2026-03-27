#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funzioni core di inferenza per CVD Compensation Model.
=====================================================

Modulo condiviso usato da:
    - z__inference_gui.py     (GUI Tkinter, compensazione singola immagine)
    - tools/test_inference_simple.py  (validazione batch, report, scientific mode)

NESSUNA dipendenza da matplotlib — sicuro da importare in contesti GUI.

Contenuto:
    Costanti:   IMAGENET_MEAN/STD, MODEL_DEFAULTS
    Config:     cfg()                               — lettura config con default
    Checkpoint: load_checkpoint()                    — metadati senza modello
    Modello:    create_model(), load_model()         — creazione + caricamento pesi
    Immagine:   load_image()                         — Resize + CenterCrop + ImageNet norm
    Inferenza:  run_inference()                      — forward pass + tanh→[0,1]
    Profilo:    normalize_profile()                  — normalizzazione ibrida θ/C/S
    Utility:    denormalize_imagenet(), tensor_to_numpy()
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from CVDCompensationModelAdaIN import CVDCompensationModelAdaIN
from cvd_constants import classify_cvd_type_from_theta


# ===========================================================================
# Costanti
# ===========================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default del costruttore CVDCompensationModelAdaIN (usati quando la config
# del checkpoint non specifica un valore).
MODEL_DEFAULTS = {
    "architecture": "CVDCompensationModelAdaIN",
    "y_preserving": True,
    "target_resolution": 256,
    "cvd_dim": 3,
    "delta_rgb_scale": 0.9,
    "pretrained_encoder": True,
    "freeze_encoder_except_adain": True,
    "use_skip_connection": False,
    "stop_at_stage": 2,
}


# ===========================================================================
# Config helper
# ===========================================================================

def cfg(config: dict, key: str):
    """Legge un valore dalla config, con fallback ai default del modello."""
    return config.get(key, MODEL_DEFAULTS.get(key))


# ===========================================================================
# Checkpoint + modello
# ===========================================================================

def load_checkpoint(checkpoint_path, device="cpu"):
    """Carica checkpoint ed estrae metadati SENZA istanziare il modello.

    Returns:
        ckpt, config, profile_stats, epoch, best_de
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    profile_stats = ckpt.get("profile_normalization", None)
    epoch = ckpt.get("epoch", "N/A")
    best_de = ckpt.get("best_delta_e00", "N/A")
    return ckpt, config, profile_stats, epoch, best_de


def create_model(config):
    """Crea CVDCompensationModelAdaIN da config dict (senza caricare pesi)."""
    return CVDCompensationModelAdaIN(
        cvd_dim=cfg(config, "cvd_dim"),
        pretrained_encoder=cfg(config, "pretrained_encoder"),
        freeze_encoder_except_adain=cfg(config, "freeze_encoder_except_adain"),
        use_skip_connection=cfg(config, "use_skip_connection"),
        stop_at_stage=cfg(config, "stop_at_stage"),
        delta_rgb_scale=cfg(config, "delta_rgb_scale"),
        target_resolution=cfg(config, "target_resolution"),
        y_preserving=cfg(config, "y_preserving"),
    )


def load_model(checkpoint_path, device="cuda"):
    """Carica modello completo da checkpoint (.pth).

    Returns:
        model (eval mode), config, profile_stats
    """
    ckpt, config, profile_stats, _, _ = load_checkpoint(checkpoint_path, device)
    model = create_model(config)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])

    model = model.to(device)
    model.eval()
    return model, config, profile_stats


# ===========================================================================
# Image loading
# ===========================================================================

def load_image(image_path, size=256, device="cuda"):
    """Carica e preprocessa un'immagine per il modello.

    Returns:
        img_tensor:  [1,3,H,W] normalizzato ImageNet
        img_raw_01:  [1,3,H,W] raw [0,1] (per visualizzazione)
    """
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    raw_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    img_raw_01 = raw_transform(img).unsqueeze(0).to(device)

    return img_tensor, img_raw_01


# ===========================================================================
# Inferenza
# ===========================================================================

def run_inference(model, input_tensor, cvd_profile, device="cuda"):
    """Forward pass del modello. Output in [0,1].

    Il modello ritorna SEMPRE [-1,1] (tanh) sia in y_preserving che RGB-only.
    Questa funzione converte: (output + 1) / 2 → [0,1].

    Returns:
        output_01: [1,3,H,W] in [0,1]
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, profile_feats=cvd_profile)
    if isinstance(output, dict):
        rgb_out = output["rgb_output"]
    else:
        rgb_out = output
    return torch.clamp((rgb_out + 1.0) / 2.0, 0.0, 1.0)


# ===========================================================================
# Normalizzazione profilo CVD
# ===========================================================================

def normalize_profile(theta, c_idx, s_idx, profile_stats, device="cpu",
                      verbose=False):
    """Normalizza profilo CVD raw [θ, C, S] con statistiche ibride.

    Ordine di priorità:
        1. Hybrid: θ globale + C/S per-tipo CVD
        2. Legacy per-CVD: per-type per tutti e 3
        3. Legacy globale: profile_mean/std
        4. Nessuna: valori raw

    Returns:
        tensor [1, 3] = [θ_norm, C_norm, S_norm]
    """
    cvd_type = classify_cvd_type_from_theta(theta)
    if verbose:
        print(f"    [NORM] CVD Type: {cvd_type}")

    if profile_stats and "theta_global_stats" in profile_stats and "per_cvd_stats" in profile_stats:
        tgs = profile_stats["theta_global_stats"]
        pcs = profile_stats["per_cvd_stats"].get(cvd_type, {})

        theta_norm = (theta - tgs["mean"]) / max(tgs["std"], 1e-8)

        C_mean = pcs.get("C_mean", pcs.get("mean", [0, 0, 0])[1] if "mean" in pcs else 0)
        C_std = max(pcs.get("C_std", pcs.get("std", [1, 1, 1])[1] if "std" in pcs else 1), 1e-8)
        S_mean = pcs.get("S_mean", pcs.get("mean", [0, 0, 0])[2] if "mean" in pcs else 0)
        S_std = max(pcs.get("S_std", pcs.get("std", [1, 1, 1])[2] if "std" in pcs else 1), 1e-8)
        C_norm = (c_idx - C_mean) / C_std
        S_norm = (s_idx - S_mean) / S_std

        if verbose:
            print(f"    [NORM HYBRID] θ: {theta:.2f}° → {theta_norm:.4f}")
            print(f"    [NORM HYBRID] C: {c_idx:.4f} → {C_norm:.4f}")
            print(f"    [NORM HYBRID] S: {s_idx:.4f} → {S_norm:.4f}")

        return torch.tensor([[theta_norm, C_norm, S_norm]], dtype=torch.float32, device=device)

    elif profile_stats and "per_cvd_stats" in profile_stats and cvd_type in profile_stats["per_cvd_stats"]:
        ts = profile_stats["per_cvd_stats"][cvd_type]
        mean = torch.tensor(ts["mean"], dtype=torch.float32)
        std = torch.clamp(torch.tensor(ts["std"], dtype=torch.float32), min=1e-8)
        raw = torch.tensor([theta, c_idx, s_idx], dtype=torch.float32)
        normed = ((raw - mean) / std).tolist()
        if verbose:
            print(f"    [NORM FALLBACK] Using per-CVD TYPE stats for '{cvd_type}'")
        return torch.tensor([normed], dtype=torch.float32, device=device)

    elif profile_stats and "profile_mean" in profile_stats:
        mean = np.array(profile_stats["profile_mean"])
        std = np.clip(np.array(profile_stats["profile_std"]), 1e-8, None)
        normed = ([theta, c_idx, s_idx] - mean) / std
        if verbose:
            print(f"    [NORM FALLBACK] Using GLOBAL stats")
        return torch.tensor([normed.tolist()], dtype=torch.float32, device=device)

    if verbose:
        print(f"    [NORM FALLBACK] NO STATS! Using raw values")
    return torch.tensor([[theta, c_idx, s_idx]], dtype=torch.float32, device=device)


# ===========================================================================
# Utility
# ===========================================================================

def denormalize_imagenet(tensor):
    """Denormalizza da ImageNet normalization a [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def tensor_to_numpy(tensor):
    """Converte tensor [B,C,H,W] o [C,H,W] a numpy HWC [0,1]."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
