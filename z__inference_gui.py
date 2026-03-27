#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVD Image Compensation — Inference GUI
=======================================

Script con interfaccia grafica (Tkinter) per compensare immagini singole
per utenti con deficit cromatico (CVD).

ARCHITETTURA DI RIFERIMENTO (da config YAML):
    architecture:           CVDCompensationModelAdaIN
    encoder_type:           convnext_tiny          (ConvNeXt-Tiny pretrained ImageNet)
    conditioning_type:      cvdadain               (CVD-AdaIN integrato in ogni blocco)
    y_preserving:           true                   (preservazione luma Y' BT.601)
    target_resolution:      256                    (Resize + CenterCrop)
    delta_rgb_scale:        0.9                    (fattore di scala per delta crominanza)

PIPELINE INFERENZA:
    1. Immagine RGB → Resize(target_resolution) → CenterCrop(target_resolution)
       → ToTensor [0,1] → Normalize ImageNet (mean=[.485,.456,.406], std=[.229,.224,.225])
       → tensor [B,3,H,W] in range circa [-2.1, +2.6]

    2. Profilo CVD raw [θ_deg, C_index, S_index] → Normalizzazione IBRIDA:
        - θ: normalizzazione GLOBALE con theta_global_stats dal checkpoint
             → preserva distinzione Protan(θ>5°) / Deutan(-30<θ≤5°) / Tritan(θ≤-30°)
        - C, S: normalizzazione PER-CVD TYPE con per_cvd_stats dal checkpoint
             → gestisce outlier (es. Tritan con C/S molto diversi)
       → tensor [1,3] = [θ_norm, C_norm, S_norm]

    3. Forward pass modello:
       ┌─────────────────────────────────────────────────────────────────────┐
       │  Encoder (ConvNeXt-Tiny + CVDAdaIN) → latent [B, 384, 16, 16]      │
       │  Decoder (PLCFDecoderCVD + CVDAdaIN) → delta [B, 2, H, W]         │
       │                                                                     │
       │  SE y_preserving=True:                                              │
       │    input ImageNet-norm → denorm [0,1] → tanh [-1,1] → YCbCr       │
       │    Y_output  = Y_input  (COPIATO: luma Y' perfettamente preservata)│
       │    Cb_output = Cb_input + ΔCb * delta_rgb_scale                    │
       │    Cr_output = Cr_input + ΔCr * delta_rgb_scale                    │
       │    YCbCr → RGB → clamp [-1, 1]                                     │
       │                                                                     │
       │  SE y_preserving=False:                                             │
       │    rgb_output = clamp(image + ΔRGB * delta_rgb_scale, -1, 1)       │
       └─────────────────────────────────────────────────────────────────────┘
       → rgb_output [B, 3, H, W] in range [-1, 1] (tanh) IN ENTRAMBI I CASI

    4. Conversione output: (rgb_output + 1) / 2  → [0, 1]  → uint8 PNG
       NOTA: Il modello ritorna SEMPRE [-1,1] indipendentemente da y_preserving.
             In y_preserving mode converte internamente ImageNet→[0,1]→[-1,1]
             prima di lavorare, quindi l'output è in range tanh, NON ImageNet.
             NON si deve usare denorm ImageNet sull'output.

PARAMETRI CONFIG NON USATI IN INFERENZA (solo training):
    use_amp / amp_dtype:    bfloat16 AMP solo durante train.py
    learning_rate:          ottimizzatore Adam
    batch_size:             solo DataLoader training
    cvd_lambda_*:           pesi loss (MSE a*b* + MS-SSIM, no ΔE in questa config)
    epochs / patience:      scheduler e early stopping
    FRACTION_IMAGES_*:      sottocampionamento dataset

PROFILO CVD (3D, normalizzazione ibrida):
    Il vettore profilo [θ, C, S] proviene dal test Farnsworth-Munsell 100 Hue:
    - θ (Confusion Angle): identifica il tipo di deficit cromatico
        Protan (θ > 5°): deficit coni L (rosso), asse confusione ~9-11°
        Deutan (-30° < θ ≤ 5°): deficit coni M (verde), asse ~-3° / -9°
        Tritan (θ ≤ -30°): deficit coni S (blu), asse ~-39° / -81°
    - C (C-index): indice di cromaticità, intensità dell'errore cromatico
    - S (S-index): indice di severità, rapporto asse maggiore/minore dell'ellisse

    La normalizzazione usa statistiche IBRIDE salvate nel checkpoint:
    - theta_global_stats: {mean, std} calcolate su TUTTI i profili del dataset
    - per_cvd_stats: {C_mean, C_std, S_mean, S_std} PER TIPO CVD (protan/deutan/tritan)

CATENA FM100 TEST:
    GUI tkinter (questo script)
      → get_profile_feats.get_profile_feats_from_test(json_path, parent_window)
        → nasconde finestra tkinter parent
        → FM_TEST.FarnsworthTestWrapper().run_test_and_get_correction()  (PyQt5)
        → salva JSON con {Confusion Angle, C-index, S-index, Classification, ...}
        → extract_profile_feats() → 6D array (per pipeline video, non usato qui)
        → ripristina finestra tkinter parent
    Questo script legge solo θ, C, S dal JSON salvato.

Usage:
    python z__inference_gui.py
"""

import sys
import os
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime

import numpy as np

from inference_core import (
    cfg, MODEL_DEFAULTS, IMAGENET_MEAN, IMAGENET_STD,
    load_checkpoint, load_model, load_image,
    run_inference, normalize_profile, tensor_to_numpy,
)

# ===========================================================================
# Costanti e path di progetto
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"
LOGS_DIR = REPO_ROOT / "logs"
TEST_RESULTS_DIR = REPO_ROOT / "test_results"
PROFILES_DIR = REPO_ROOT / "profiles_output"
OUTPUT_DIR = REPO_ROOT / "output_compensated"

SUPPORTED_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# Chiavi del profilo FM100 nel JSON di risultati
FM100_KEY_THETA = "Confusion Angle (degrees)"
FM100_KEY_C = "C-index"
FM100_KEY_S = "S-index"
FM100_KEY_CLASS = "Classification"


# ===========================================================================
# Backend: ricerca checkpoint, simulazione CVD, caricamento profilo FM100
# ===========================================================================

def find_checkpoints(base_dir: Path = RESULTS_DIR):
    """Cerca ricorsivamente tutti i file .pth sotto *base_dir*.

    Ordina per priorità:
        0 = "best"  (miglior ΔE00 su validation)
        1 = "final"  (ultimo epoch)
        2 = epoch intermedio
    A parità di priorità, ordina per data modifica decrescente (più recente prima).
    """
    found = []
    if not base_dir.exists():
        return found
    for pth in base_dir.rglob("*.pth"):
        name = pth.name.lower()
        priority = 2
        if "best" in name:
            priority = 0
        elif "final" in name:
            priority = 1
        found.append((priority, pth.stat().st_mtime, pth))
    found.sort(key=lambda x: (x[0], -x[1]))
    return [p for _, _, p in found]


def simulate_cvd_on_image(image_01_hwc, theta_deg):
    """Simula come un daltonico vede un'immagine usando il modello Machado 2009.

    Usa classify_cvd_type_from_theta() per determinare il tipo CVD
    (stesse soglie usate dalla normalizzazione del profilo) con severità 1.0
    (deficit completo) per la visualizzazione worst-case.

    Args:
        image_01_hwc: immagine [0,1] numpy HWC
        theta_deg:    angolo di confusione dalla profilazione FM100

    Returns:
        immagine simulata [0,1] numpy HWC
    """
    from cvd_constants import classify_cvd_type_from_theta
    import torch

    cvd_type = classify_cvd_type_from_theta(theta_deg)

    t = torch.tensor(image_01_hwc, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    try:
        from cvd_simulator import simulatecvd
        sim = simulatecvd(t, cvd_type, 1.0)
        return sim[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    except Exception:
        return image_01_hwc


def load_profile_from_json(json_path: str):
    """Carica θ, C, S da un file JSON di risultati FM100.

    Il JSON è prodotto da get_profile_feats.get_profile_feats_from_test()
    e contiene i risultati di calculate_error_basic_PCA() di FM_TEST.py.

    Chiavi lette (definite come costanti FM100_KEY_*):
        - "Confusion Angle (degrees)": θ in gradi, identifica tipo CVD
        - "C-index": indice di cromaticità
        - "S-index": indice di severità
        - "Classification": stringa descrittiva (es. "Protanope")

    Se il JSON è una lista (più test), prende il più recente per timestamp.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = max(data, key=lambda x: x.get("timestamp", ""))
    theta = data[FM100_KEY_THETA]
    c_idx = data[FM100_KEY_C]
    s_idx = data[FM100_KEY_S]
    classification = data.get(FM100_KEY_CLASS, "")
    return theta, c_idx, s_idx, classification


# ===========================================================================
# GUI
# ===========================================================================

class InferenceGUI:
    """Interfaccia grafica per inferenza CVD compensation su immagine singola.

    Wizard a 4 tab:
        1. Verifica  — dipendenze (PyTorch, CUDA) + selezione checkpoint + display config
        2. Profilo   — carica JSON esistente o esegui FM100 Hue test
        3. Immagine  — seleziona file, preview originale + simulazione CVD
        4. Risultato — originale | compensata | sim CVD della compensata

    Tutti i comportamenti condizionali sono guidati dalla config del checkpoint:
        - target_resolution  → dimensione crop input e canvas preview
        - y_preserving       → documentato nel pannello config (impatto su output)
        - delta_rgb_scale    → documentato nel pannello config
        - experiment_name    → mostrato come riferimento
        - loss weights       → mostrati per contesto (solo training)
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CVD Image Compensation")
        self.root.geometry("900x680")
        self.root.minsize(800, 600)

        # State
        self.checkpoint_path = None
        self.config = None
        self.profile_stats = None
        self.model = None
        self.device = "cpu"

        self.theta = None
        self.c_idx = None
        self.s_idx = None
        self.cvd_classification = ""
        self.profile_json_path = None

        self.image_path = None
        self.original_np = None
        self.compensated_np = None

        # Notebook (tabs = wizard steps)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_step4()

        # Disabilita tab 2-4 all'inizio
        for i in range(1, 4):
            self.notebook.tab(i, state="disabled")

        # Avvia verifica dipendenze
        self.root.after(300, self._auto_check)

    # ------------------------------------------------------------------
    # STEP 1 — Verifica dipendenze
    # ------------------------------------------------------------------
    def _build_step1(self):
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="1. Verifica")

        ttk.Label(frame, text="Verifica Dipendenze", font=("", 14, "bold")).pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=6)

        self.checks_frame = ttk.Frame(frame)
        self.checks_frame.pack(fill="x", pady=4)

        self.check_labels = {}
        for key, text in [
            ("torch", "PyTorch disponibile"),
            ("cuda", "CUDA / GPU"),
            ("checkpoint", "Checkpoint modello (.pth)"),
            ("architecture", "Architettura: CVDCompensationModelAdaIN"),
            ("profile_stats", "Statistiche normalizzazione profilo (hybrid)"),
        ]:
            row = ttk.Frame(self.checks_frame)
            row.pack(fill="x", pady=2)
            indicator = ttk.Label(row, text="  ?  ", width=6)
            indicator.pack(side="left")
            ttk.Label(row, text=text).pack(side="left", padx=4)
            self.check_labels[key] = indicator

        ttk.Separator(frame).pack(fill="x", pady=6)

        # Checkpoint selection
        ckpt_frame = ttk.Frame(frame)
        ckpt_frame.pack(fill="x", pady=4)
        ttk.Label(ckpt_frame, text="Checkpoint:").pack(side="left")
        self.ckpt_var = tk.StringVar()
        self.ckpt_combo = ttk.Combobox(ckpt_frame, textvariable=self.ckpt_var, width=70, state="readonly")
        self.ckpt_combo.pack(side="left", padx=4, fill="x", expand=True)
        self.ckpt_combo.bind("<<ComboboxSelected>>", self._on_ckpt_selected)
        ttk.Button(ckpt_frame, text="Sfoglia...", command=self._browse_checkpoint).pack(side="left")

        # Info label (multi-riga per mostrare config dettagliata)
        self.ckpt_info_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.ckpt_info_var, wraplength=800,
                  justify="left", font=("Consolas", 9)).pack(anchor="w", pady=4)

        # Avanti
        self.step1_next = ttk.Button(frame, text="Avanti  >>", command=self._step1_next, state="disabled")
        self.step1_next.pack(anchor="e", pady=8)

    def _auto_check(self):
        """Verifica automatica dipendenze."""
        # PyTorch
        try:
            import torch
            self._set_check("torch", True, f"v{torch.__version__}")
        except ImportError:
            self._set_check("torch", False, "Non installato")
            return

        # CUDA
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            self._set_check("cuda", True, f"{gpu} ({mem:.1f} GB)")
            self.device = "cuda"
        else:
            self._set_check("cuda", None, "Non disponibile, uso CPU")
            self.device = "cpu"

        # Cerca checkpoint
        ckpts = find_checkpoints()
        if ckpts:
            paths_str = [str(p) for p in ckpts]
            self.ckpt_combo["values"] = paths_str
            self.ckpt_combo.current(0)
            self._on_ckpt_selected(None)
        else:
            self._set_check("checkpoint", False, "Nessun .pth trovato in results/")

    def _set_check(self, key, ok, detail=""):
        """Imposta indicatore: True=verde, False=rosso, None=giallo."""
        lbl = self.check_labels[key]
        if ok is True:
            lbl.config(text="  OK ", foreground="green")
        elif ok is False:
            lbl.config(text=" FAIL", foreground="red")
        else:
            lbl.config(text=" WARN", foreground="orange")
        if detail:
            lbl.config(text=lbl.cget("text"))
            # Aggiungi dettaglio alla riga
            parent = lbl.master
            # Rimuovi dettagli vecchi se presenti
            for w in parent.winfo_children():
                if hasattr(w, "_detail"):
                    w.destroy()
            d = ttk.Label(parent, text=f"— {detail}", foreground="gray")
            d._detail = True
            d.pack(side="left", padx=4)

    def _on_ckpt_selected(self, _event):
        """Quando viene selezionato un checkpoint dal combo.

        Legge il checkpoint (senza caricare il modello) ed estrae:
        - config: tutti i parametri di addestramento
        - profile_normalization: statistiche per normalizzazione profilo
        - epoch, best_delta_e00: stato dell'addestramento

        Mostra nel pannello info i parametri rilevanti per l'inferenza.
        """
        path = self.ckpt_var.get()
        if not path or not Path(path).exists():
            return
        try:
            _, config, pstats, epoch, best_de = load_checkpoint(path, device="cpu")
            # Se il checkpoint è cambiato, invalida il modello cached
            if self.checkpoint_path != path:
                self.model = None
            self.checkpoint_path = path
            self.config = config
            self.profile_stats = pstats

            self._set_check("checkpoint", True, Path(path).name)

            # Verifica architettura
            arch = config.get("architecture", "CVDCompensationModelAdaIN")
            if arch == "CVDCompensationModelAdaIN":
                self._set_check("architecture", True, f"{arch} (ConvNeXt-Tiny + CVDAdaIN)")
            else:
                self._set_check("architecture", None, f"Architettura inattesa: {arch}")

            # Costruisci info dettagliata dalla config
            y_pres = cfg(config, "y_preserving")
            res = cfg(config, "target_resolution")
            scale = cfg(config, "delta_rgb_scale")
            dim = cfg(config, "cvd_dim")
            exp_name = config.get("experiment_name", "N/A")

            # Pesi loss (solo per contesto, non usati in inferenza)
            l_mse = config.get("cvd_lambda_mse", "?")
            l_de = config.get("cvd_lambda_delta_e", "?")
            l_ssim = config.get("cvd_lambda_ssim", "?")

            info_lines = [
                f"Esperimento: {exp_name}",
                f"Epoch: {epoch}  |  Best ΔE00: {best_de}",
                f"",
                f"─── Parametri inferenza (dalla config) ───",
                f"y_preserving: {y_pres}  (Y copiato, solo ΔCb/ΔCr)"
                    if y_pres else
                    f"y_preserving: {y_pres}  (ΔRGB 3ch, no preserv. luma)",
                f"target_resolution: {res}×{res} px",
                f"delta_rgb_scale: {scale}",
                f"cvd_dim: {dim}  (profilo 3D: [θ_norm, C_norm, S_norm])",
                f"",
                f"─── Solo training (non usati in inferenza) ───",
                f"Loss: λ_mse={l_mse}, λ_ΔE={l_de}, λ_ssim={l_ssim}",
                f"AMP: {config.get('use_amp', '?')} ({config.get('amp_dtype', '?')})",
            ]
            self.ckpt_info_var.set("\n".join(info_lines))

            if pstats:
                ntype = pstats.get("normalization_type", "legacy")
                has_hybrid = "theta_global_stats" in pstats and "per_cvd_stats" in pstats
                if has_hybrid:
                    self._set_check("profile_stats", True,
                                    f"Hybrid (θ globale + C/S per-tipo)")
                else:
                    self._set_check("profile_stats", True, f"Tipo: {ntype}")
            else:
                self._set_check("profile_stats", None,
                                "Non presenti nel checkpoint — profili non normalizzati")

            self.step1_next.config(state="normal")
        except Exception as e:
            self._set_check("checkpoint", False, str(e)[:80])

    def _browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Seleziona checkpoint .pth",
            initialdir=str(RESULTS_DIR),
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("Tutti", "*.*")],
        )
        if path:
            vals = list(self.ckpt_combo["values"]) if self.ckpt_combo["values"] else []
            if path not in vals:
                vals.insert(0, path)
                self.ckpt_combo["values"] = vals
            self.ckpt_var.set(path)
            self._on_ckpt_selected(None)

    def _step1_next(self):
        self.notebook.tab(1, state="normal")
        self.notebook.select(1)

    # ------------------------------------------------------------------
    # STEP 2 — Profilo CVD
    # ------------------------------------------------------------------
    def _build_step2(self):
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="2. Profilo CVD")

        ttk.Label(frame, text="Profilo CVD dell'utente", font=("", 14, "bold")).pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=6)

        # Opzioni
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=4)
        ttk.Button(btn_frame, text="Carica profilo JSON esistente", command=self._load_profile_json).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Esegui test FM100 Hue", command=self._run_fm100).pack(side="left", padx=4)

        ttk.Separator(frame).pack(fill="x", pady=6)

        # Card profilo
        self.profile_card = ttk.LabelFrame(frame, text="Profilo caricato", padding=8)
        self.profile_card.pack(fill="x", pady=4)
        self.profile_info_var = tk.StringVar(value="Nessun profilo caricato")
        ttk.Label(self.profile_card, textvariable=self.profile_info_var, wraplength=750, justify="left").pack(anchor="w")

        # Profili precedenti
        self.prev_profiles_frame = ttk.LabelFrame(frame, text="Profili precedenti salvati", padding=8)
        self.prev_profiles_frame.pack(fill="both", expand=True, pady=4)
        self.profile_listbox = tk.Listbox(self.prev_profiles_frame, height=6)
        self.profile_listbox.pack(fill="both", expand=True)
        self.profile_listbox.bind("<Double-1>", self._on_profile_double_click)
        self._refresh_profile_list()

        # Avanti
        nav = ttk.Frame(frame)
        nav.pack(fill="x", pady=8)
        ttk.Button(nav, text="<< Indietro", command=lambda: self.notebook.select(0)).pack(side="left")
        self.step2_next = ttk.Button(nav, text="Avanti  >>", command=self._step2_next, state="disabled")
        self.step2_next.pack(side="right")

    def _refresh_profile_list(self):
        self.profile_listbox.delete(0, tk.END)
        if TEST_RESULTS_DIR.exists():
            jsons = sorted(TEST_RESULTS_DIR.glob("fm100_test_*.json"), reverse=True)
            for j in jsons:
                self.profile_listbox.insert(tk.END, j.name)

    def _on_profile_double_click(self, _event):
        sel = self.profile_listbox.curselection()
        if not sel:
            return
        name = self.profile_listbox.get(sel[0])
        path = TEST_RESULTS_DIR / name
        self._set_profile_from_json(str(path))

    def _load_profile_json(self):
        path = filedialog.askopenfilename(
            title="Seleziona risultato FM100 (.json)",
            initialdir=str(TEST_RESULTS_DIR) if TEST_RESULTS_DIR.exists() else str(REPO_ROOT),
            filetypes=[("JSON", "*.json"), ("Tutti", "*.*")],
        )
        if path:
            self._set_profile_from_json(path)

    def _set_profile_from_json(self, path):
        try:
            theta, c_idx, s_idx, classification = load_profile_from_json(path)
            self.theta = theta
            self.c_idx = c_idx
            self.s_idx = s_idx
            self.cvd_classification = classification
            self.profile_json_path = path

            from cvd_constants import classify_cvd_type_from_theta
            cvd_type = classify_cvd_type_from_theta(theta)

            info = (
                f"Tipo CVD:  {cvd_type.upper()}  ({classification})\n"
                f"Angolo confusione (theta):  {theta:.2f} deg\n"
                f"C-index:  {c_idx:.4f}\n"
                f"S-index:  {s_idx:.4f}\n"
                f"File:  {Path(path).name}"
            )
            self.profile_info_var.set(info)
            self.step2_next.config(state="normal")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare profilo:\n{e}")

    def _run_fm100(self):
        """Lancia il test FM100 Hue (PyQt5) tramite get_profile_feats."""
        TEST_RESULTS_DIR.mkdir(exist_ok=True)
        PROFILES_DIR.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = str(TEST_RESULTS_DIR / f"fm100_test_{timestamp}.json")

        # Avvia in thread separato per non bloccare tkinter
        threading.Thread(target=self._fm100_worker, args=(json_path, timestamp), daemon=True).start()

    def _fm100_worker(self, json_path, timestamp):
        try:
            from get_profile_feats import get_profile_feats_from_test

            profile_feats = get_profile_feats_from_test(
                save_path_json=json_path,
                parent_window=self.root,
            )

            if profile_feats is not None:
                # Salva anche il .npy 6D (compatibilita con pipeline video)
                npy_path = PROFILES_DIR / f"profile_7d_{timestamp}.npy"
                np.save(npy_path, profile_feats)

                # Carica theta/C/S dal JSON appena salvato
                self.root.after(0, lambda: self._set_profile_from_json(json_path))
                self.root.after(0, self._refresh_profile_list)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Errore FM100", str(e)))

    def _step2_next(self):
        self.notebook.tab(2, state="normal")
        self.notebook.select(2)

    # ------------------------------------------------------------------
    # STEP 3 — Selezione immagine + compensazione
    # ------------------------------------------------------------------
    def _build_step3(self):
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="3. Immagine")

        ttk.Label(frame, text="Seleziona immagine da compensare", font=("", 14, "bold")).pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=6)

        # Selezione file
        sel_frame = ttk.Frame(frame)
        sel_frame.pack(fill="x", pady=4)
        ttk.Button(sel_frame, text="Scegli immagine...", command=self._browse_image).pack(side="left")
        self.img_path_var = tk.StringVar(value="Nessuna immagine selezionata")
        ttk.Label(sel_frame, textvariable=self.img_path_var, wraplength=600).pack(side="left", padx=8)

        # Preview: originale + simulazione CVD
        preview_frame = ttk.Frame(frame)
        preview_frame.pack(fill="both", expand=True, pady=4)

        # Canvas originale
        left = ttk.LabelFrame(preview_frame, text="Originale", padding=4)
        left.pack(side="left", fill="both", expand=True, padx=4)
        self.canvas_orig = tk.Canvas(left, width=256, height=256, bg="#2b2b2b")
        self.canvas_orig.pack()

        # Canvas simulazione
        right = ttk.LabelFrame(preview_frame, text="Simulazione CVD (come vede il daltonico)", padding=4)
        right.pack(side="left", fill="both", expand=True, padx=4)
        self.canvas_sim = tk.Canvas(right, width=256, height=256, bg="#2b2b2b")
        self.canvas_sim.pack()

        # Compensazione
        ttk.Separator(frame).pack(fill="x", pady=6)
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="<< Indietro", command=lambda: self.notebook.select(1)).pack(side="left")
        self.btn_compensate = ttk.Button(btn_row, text="Compensa immagine", command=self._compensate, state="disabled")
        self.btn_compensate.pack(side="right")
        self.progress_var = tk.StringVar(value="")
        ttk.Label(btn_row, textvariable=self.progress_var).pack(side="right", padx=8)

    def _browse_image(self):
        ext = " ".join(f"*{e}" for e in SUPPORTED_IMAGE_EXT)
        path = filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[("Immagini", ext), ("Tutti", "*.*")],
        )
        if path:
            self.image_path = path
            self.img_path_var.set(Path(path).name)
            self._show_preview(path)
            self.btn_compensate.config(state="normal")

    def _show_preview(self, path):
        """Mostra preview originale + simulazione CVD nei canvas.

        Il Resize + CenterCrop usa target_resolution dalla config del checkpoint
        (valore letto durante la selezione del checkpoint nello Step 1).
        La simulazione CVD usa il θ dal profilo caricato nello Step 2 per
        determinare il tipo di deficit (Protan/Deutan/Tritan) e viene
        mostrata a severità 1.0 (worst-case) via Machado 2009.
        """
        try:
            from PIL import Image, ImageTk
            import torchvision.transforms as T

            res = cfg(self.config, "target_resolution") if self.config else 256
            img = Image.open(path).convert("RGB")
            transform = T.Compose([T.Resize(res), T.CenterCrop(res)])
            img_256 = transform(img)

            # Originale
            self._photo_orig = ImageTk.PhotoImage(img_256)
            self.canvas_orig.create_image(128, 128, image=self._photo_orig)

            # Simulazione CVD
            if self.theta is not None:
                img_np = np.array(img_256).astype(np.float32) / 255.0
                sim_np = simulate_cvd_on_image(img_np, self.theta)
                sim_pil = Image.fromarray((sim_np * 255).astype(np.uint8))
                self._photo_sim = ImageTk.PhotoImage(sim_pil)
                self.canvas_sim.create_image(128, 128, image=self._photo_sim)
        except Exception as e:
            self.img_path_var.set(f"Errore preview: {e}")

    def _compensate(self):
        """Avvia la compensazione in un thread separato."""
        self.btn_compensate.config(state="disabled")
        self.progress_var.set("Caricamento modello...")
        threading.Thread(target=self._compensate_worker, daemon=True).start()

    def _compensate_worker(self):
        """Worker thread per la compensazione.

        Sequenza:
            1. Carica modello (lazy: solo al primo compenso, poi riusa)
               Parametri dal checkpoint config: cvd_dim, y_preserving,
               target_resolution, delta_rgb_scale, stop_at_stage, ecc.
            2. Normalizza profilo [θ, C, S] → [θ_norm, C_norm, S_norm]
               con statistiche ibride dal checkpoint
            3. Forward pass: image ImageNet-norm + profilo → rgb_output [-1,1]
            4. Conversione tanh→[0,1] e salvataggio PNG + JSON metadati
        """
        try:
            import torch

            # Carica modello (se non già caricato o se checkpoint è cambiato)
            if self.model is None:
                self.root.after(0, lambda: self.progress_var.set("Caricamento modello..."))
                self.model, self.config, self.profile_stats = load_model(
                    self.checkpoint_path, self.device
                )

            self.root.after(0, lambda: self.progress_var.set("Normalizzazione profilo..."))
            profile_3d = normalize_profile(
                self.theta, self.c_idx, self.s_idx,
                self.profile_stats, self.device
            )

            self.root.after(0, lambda: self.progress_var.set("Inferenza in corso..."))
            res = cfg(self.config, "target_resolution")
            img_tensor, img_raw_01 = load_image(
                self.image_path, size=res, device=self.device
            )
            output_01 = run_inference(
                self.model, img_tensor, profile_3d, self.device
            )
            from PIL import Image as _PIL
            orig_size = _PIL.open(self.image_path).size
            orig_np = tensor_to_numpy(img_raw_01)
            comp_np = tensor_to_numpy(output_01)
            self.original_np = orig_np
            self.compensated_np = comp_np

            # Salva immagine compensata
            OUTPUT_DIR.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"compensated_{ts}.png"
            from PIL import Image
            Image.fromarray((comp_np * 255).astype(np.uint8)).save(out_path)

            # Salva metadati completi (config-driven)
            meta = {
                "timestamp": ts,
                "source_image": str(self.image_path),
                "source_image_original_size": list(orig_size),
                "checkpoint": str(self.checkpoint_path),
                "device": self.device,
                "output_path": str(out_path),
                # Profilo CVD raw (prima della normalizzazione)
                "profile_raw": {
                    "theta_deg": self.theta,
                    "C_index": self.c_idx,
                    "S_index": self.s_idx,
                    "classification": self.cvd_classification,
                },
                # Parametri config rilevanti per inferenza
                "config_inference": {
                    "architecture": self.config.get("architecture", "CVDCompensationModelAdaIN"),
                    "y_preserving": cfg(self.config, "y_preserving"),
                    "target_resolution": cfg(self.config, "target_resolution"),
                    "delta_rgb_scale": cfg(self.config, "delta_rgb_scale"),
                    "cvd_dim": cfg(self.config, "cvd_dim"),
                    "experiment_name": self.config.get("experiment_name", ""),
                },
                # Info training (per tracciabilità)
                "config_training": {
                    "cvd_lambda_mse": self.config.get("cvd_lambda_mse"),
                    "cvd_lambda_delta_e": self.config.get("cvd_lambda_delta_e"),
                    "cvd_lambda_ssim": self.config.get("cvd_lambda_ssim"),
                    "use_amp": self.config.get("use_amp"),
                    "amp_dtype": self.config.get("amp_dtype"),
                },
                # Normalizzazione usata
                "normalization_type": (
                    self.profile_stats.get("normalization_type", "unknown")
                    if self.profile_stats else "none"
                ),
            }
            meta_path = out_path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            self.root.after(0, lambda: self._show_results(orig_np, comp_np, str(out_path)))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Errore compensazione", str(e)))
            self.root.after(0, lambda: self.progress_var.set(""))
            self.root.after(0, lambda: self.btn_compensate.config(state="normal"))

    # ------------------------------------------------------------------
    # STEP 4 — Risultati
    # ------------------------------------------------------------------
    def _build_step4(self):
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="4. Risultato")

        ttk.Label(frame, text="Risultato Compensazione", font=("", 14, "bold")).pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=6)

        # 3 canvas: orig | compensata | sim CVD della compensata
        row = ttk.Frame(frame)
        row.pack(fill="both", expand=True)

        lf1 = ttk.LabelFrame(row, text="Originale", padding=4)
        lf1.pack(side="left", fill="both", expand=True, padx=2)
        self.res_canvas_orig = tk.Canvas(lf1, width=220, height=220, bg="#2b2b2b")
        self.res_canvas_orig.pack()

        lf2 = ttk.LabelFrame(row, text="Compensata", padding=4)
        lf2.pack(side="left", fill="both", expand=True, padx=2)
        self.res_canvas_comp = tk.Canvas(lf2, width=220, height=220, bg="#2b2b2b")
        self.res_canvas_comp.pack()

        lf3 = ttk.LabelFrame(row, text="Sim. CVD della compensata", padding=4)
        lf3.pack(side="left", fill="both", expand=True, padx=2)
        self.res_canvas_sim = tk.Canvas(lf3, width=220, height=220, bg="#2b2b2b")
        self.res_canvas_sim.pack()

        # Info
        self.result_info_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.result_info_var, wraplength=800, justify="left").pack(anchor="w", pady=6)

        # Azioni
        ttk.Separator(frame).pack(fill="x", pady=4)
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Nuova immagine", command=self._new_image).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Cambio profilo", command=self._change_profile).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Esci", command=self.root.quit).pack(side="right", padx=4)

    def _show_results(self, orig_np, comp_np, out_path):
        """Mostra i risultati nello step 4.

        Pannello con 3 immagini affiancate:
        - Originale: come la vede un normotipico
        - Compensata: immagine modificata dal modello per compensare il deficit
        - Sim. CVD della compensata: come il daltonico vedrà l'immagine
          compensata (dovrebbe essere più simile all'originale)

        La simulazione CVD usa Machado 2009 con il tipo CVD determinato
        dal θ del profilo e severità 1.0 (worst-case).
        """
        from PIL import Image, ImageTk

        # Attiva tab risultati
        self.notebook.tab(3, state="normal")
        self.notebook.select(3)
        self.progress_var.set("")
        self.btn_compensate.config(state="normal")

        def np_to_photo(arr, size=220):
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            pil = pil.resize((size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(pil)

        # Originale
        self._res_photo_orig = np_to_photo(orig_np)
        self.res_canvas_orig.delete("all")
        self.res_canvas_orig.create_image(110, 110, image=self._res_photo_orig)

        # Compensata
        self._res_photo_comp = np_to_photo(comp_np)
        self.res_canvas_comp.delete("all")
        self.res_canvas_comp.create_image(110, 110, image=self._res_photo_comp)

        # Simulazione CVD sulla compensata
        sim_comp = simulate_cvd_on_image(comp_np, self.theta)
        self._res_photo_sim = np_to_photo(sim_comp)
        self.res_canvas_sim.delete("all")
        self.res_canvas_sim.create_image(110, 110, image=self._res_photo_sim)

        self.result_info_var.set(
            f"Salvata in: {out_path}\n"
            f"Config: y_preserving={cfg(self.config, 'y_preserving')}, "
            f"res={cfg(self.config, 'target_resolution')}, "
            f"scale={cfg(self.config, 'delta_rgb_scale')}"
        )

    def _new_image(self):
        """Torna allo step 3 per una nuova immagine."""
        self.notebook.select(2)

    def _change_profile(self):
        """Torna allo step 2 per cambiare profilo.

        Il modello NON viene invalidato perché l'architettura è
        profile-agnostic: il profilo CVD [θ, C, S] entra come input
        condizionale via CVDAdaIN, non modifica i pesi.
        Basta ricalcolare normalize_profile_3d() con i nuovi valori.
        """
        self.notebook.select(1)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    gui = InferenceGUI()
    gui.run()


if __name__ == "__main__":
    main()
