#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVD Pipeline - Gestione centralizzata del workflow

Comandi disponibili:
    python z__pipeline.py                  # Menu interattivo
    python z__pipeline.py --dataset        # Solo generazione dataset
    python z__pipeline.py --training       # Solo training
    python z__pipeline.py --inference      # Solo test inferenza
    python z__pipeline.py --gui            # Avvia GUI video recoloring
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# ============================================================================
# Costanti di progetto
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent

# Path dataset di default
DEFAULT_IMAGE_SOURCE = "dataset/places365/subsets/subsets_derived/subsets_t_0.15_v_1"
DEFAULT_DATASET_OUTPUT = "dataset/places365/subsets/subsets_derived_recolored_CVD/subsets_teacher_compensated_t_0.15_v_1"

# ============================================================================
# Utility
# ============================================================================

def run(cmd, description):
    """Esegue un comando con logging."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  CMD: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"\n  FALLITO (exit code {result.returncode})")
            return False
        print(f"\n  OK")
        return True
    except FileNotFoundError as e:
        print(f"\n  Comando non trovato: {e}")
        return False


def ask(prompt, default=""):
    """Input con default."""
    try:
        suffix = f" [{default}]" if default else ""
        val = input(f"{prompt}{suffix}: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print("\nAnnullato.")
        return None


def ask_choice(prompt, valid):
    """Input con scelta validata."""
    try:
        val = input(f"{prompt}: ").strip()
        return val if val in valid else None
    except (EOFError, KeyboardInterrupt):
        print("\nAnnullato.")
        return None


def check_script(path):
    """Verifica che uno script esista, stampa warning se no."""
    full = REPO_ROOT / path
    if not full.exists():
        print(f"  ATTENZIONE: {path} non trovato!")
        return False
    return True


# ============================================================================
# FASE A — Generazione Dataset
# ============================================================================

def phase_dataset_menu():
    """Menu generazione dataset CVD."""
    while True:
        print(f"""
{'='*60}
  GENERAZIONE DATASET CVD
{'='*60}

  Pipeline completa generazione dataset teacher-compensated.

  [1] Step 1 — Genera profili sintetici 3D
  [2] Step 2 — Genera dataset CVD-compensated (teacher)
  [3] Step 3 — Post-validazione Delta-E (tutti gli split)
  [4] Step 4 — Rigenera immagini fallite
  [5] Step 5 — Plot validazione
  [6] Step 6 — Report dataset
  [7] Step 7 — Grafici aggiuntivi dai JSON
  [A] Esegui TUTTI gli step in sequenza (1-7)
  [T] Test veloce (30 campioni)
  [0] Indietro
""")
        c = ask_choice(">> Scelta", ["1","2","3","4","5","6","7","A","a","T","t","0"])
        if c is None or c == "0":
            break
        elif c == "1":
            step_generate_profiles()
        elif c == "2":
            step_generate_dataset()
        elif c == "3":
            step_post_validate()
        elif c == "4":
            step_regenerate_failed()
        elif c == "5":
            step_validation_plots()
        elif c == "6":
            step_dataset_report()
        elif c == "7":
            step_report_charts()
        elif c.upper() == "A":
            run_all_dataset_steps()
        elif c.upper() == "T":
            step_fast_test()


def step_generate_profiles():
    """Step 1 — Genera profili sintetici 3D."""
    if not check_script("CVD_dataset_generator/generate_synthetic_dataset.py"):
        return
    run("python CVD_dataset_generator/generate_synthetic_dataset.py",
        "Step 1: Generazione profili sintetici 3D")


def step_generate_dataset():
    """Step 2 — Genera dataset CVD-compensated con teacher Farup."""
    if not check_script("CVD_dataset_generator/dataset_generator_teacher.py"):
        return

    img_src = ask("  Path immagini sorgente", DEFAULT_IMAGE_SOURCE)
    if img_src is None:
        return

    extra = ""
    n_img = ask("  Immagini per profilo (default=15)", "15")
    if n_img and n_img != "15":
        extra += f" --n-images-per-profile {n_img}"

    no_cvd = ask("  Salvare immagini CVD intermedie? (s/n)", "s")
    if no_cvd and no_cvd.lower() == "n":
        extra += " --no-cvd-save"

    save_orig = ask("  Salvare copie originali? (s/n)", "n")
    if save_orig and save_orig.lower() == "s":
        extra += " --original-save"

    use_png = ask("  Formato output (jpeg/png)", "jpeg")
    if use_png and use_png.lower() == "png":
        extra += " --no-use-jpeg"

    cmd = f'python CVD_dataset_generator/dataset_generator_teacher.py --image-source "{img_src}"{extra}'
    run(cmd, "Step 2: Generazione dataset CVD-compensated (teacher)")


def step_post_validate():
    """Step 3 — Post-validazione Delta-E su tutti gli split."""
    if not check_script("CVD_dataset_generator/post_validate_deltae_v2.py"):
        return

    dataset = ask("  Path dataset compensato", DEFAULT_DATASET_OUTPUT)
    originals = ask("  Path originali", DEFAULT_IMAGE_SOURCE)
    if dataset is None or originals is None:
        return

    cmd = (f'python CVD_dataset_generator/post_validate_deltae_v2.py '
           f'"{dataset}" --originals-dir "{originals}" --all-splits')
    run(cmd, "Step 3: Post-validazione Delta-E (tutti gli split)")


def step_regenerate_failed():
    """Step 4 — Rigenera immagini fallite dalla validazione."""
    if not check_script("CVD_dataset_generator/regenerate_failed_images_optimized.py"):
        return

    dataset = ask("  Path dataset compensato", DEFAULT_DATASET_OUTPUT)
    if dataset is None:
        return

    cmd = f'python CVD_dataset_generator/regenerate_failed_images_optimized.py "{dataset}"'
    run(cmd, "Step 4: Rigenerazione immagini fallite")


def step_validation_plots():
    """Step 5 — Genera plot di validazione."""
    if not check_script("CVD_dataset_generator/generate_validation_best_plots.py"):
        return

    dataset = ask("  Path dataset compensato", DEFAULT_DATASET_OUTPUT)
    if dataset is None:
        return

    cmd = f'python CVD_dataset_generator/generate_validation_best_plots.py "{dataset}"'
    run(cmd, "Step 5: Generazione plot di validazione")


def step_dataset_report():
    """Step 6 — Report dataset (con benchmark opzionale)."""
    if not check_script("CVD_dataset_generator/generate_dataset_report.py"):
        return

    bench = ask("  Eseguire benchmark CPU vs GPU? (s/n)", "n")
    extra = " --run-benchmark" if bench and bench.lower() == "s" else ""

    cmd = f"python CVD_dataset_generator/generate_dataset_report.py{extra}"
    run(cmd, "Step 6: Report dataset")


def step_report_charts():
    """Step 7 — Grafici aggiuntivi dai JSON esistenti."""
    if not check_script("CVD_dataset_generator/generate_all_report_charts.py"):
        return

    cmd = "python CVD_dataset_generator/generate_all_report_charts.py"
    run(cmd, "Step 7: Grafici aggiuntivi dai JSON")


def run_all_dataset_steps():
    """Esegue tutti gli step del dataset in sequenza."""
    print("\n  Esecuzione sequenziale di tutti gli step dataset...\n")
    steps = [
        step_generate_profiles,
        step_generate_dataset,
        step_post_validate,
        step_regenerate_failed,
        step_validation_plots,
        step_dataset_report,
        step_report_charts,
    ]
    for i, step_fn in enumerate(steps, 1):
        print(f"\n--- Step {i}/{len(steps)} ---")
        step_fn()


def step_fast_test():
    """Test veloce: 30 campioni GPU."""
    if not check_script("CVD_dataset_generator/generate_synthetic_dataset.py"):
        return
    if not check_script("CVD_dataset_generator/dataset_generator_teacher.py"):
        return

    print("\n  Test veloce: profili + 30 campioni\n")
    ok = run("python CVD_dataset_generator/generate_synthetic_dataset.py",
             "Fast test — Step 1: Profili sintetici")
    if not ok:
        return

    cmd = (
        f'python CVD_dataset_generator/dataset_generator_teacher.py'
        f' --profile-source "{DEFAULT_DATASET_OUTPUT}"'
        f' --image-source "{DEFAULT_IMAGE_SOURCE}"'
        f' --output-dir "test_fix_random_sampling"'
        f' --fast-test-samples 30'
        f' --n-images-per-profile 1'
        f' --no-cvd-save'
    )
    run(cmd, "Fast test — Step 2: 30 campioni GPU")


# ============================================================================
# FASE B — Configurazione e Training
# ============================================================================

def phase_training_menu():
    """Menu configurazione e training."""
    while True:
        print(f"""
{'='*60}
  CONFIGURAZIONE E TRAINING
{'='*60}

  [1] Genera TUTTE le configurazioni (balanced, perceptual, conservative)
  [2] Genera configurazione singola
  [3] Avvia training da config YAML
  [4] Resume training da checkpoint specifico
  [0] Indietro
""")
        c = ask_choice(">> Scelta", ["1","2","3","4","0"])
        if c is None or c == "0":
            break
        elif c == "1":
            step_generate_all_configs()
        elif c == "2":
            step_generate_single_config()
        elif c == "3":
            step_train()
        elif c == "4":
            step_train_resume()


def step_generate_all_configs():
    """Genera tutte le configurazioni."""
    if not check_script("config_generator.py"):
        return
    run("python config_generator.py", "Generazione tutte le configurazioni")


def step_generate_single_config():
    """Genera configurazione singola."""
    if not check_script("config_generator.py"):
        return

    name = ask("  Nome esperimento")
    if not name:
        print("  Nome non fornito, annullato.")
        return

    cmd = f'python config_generator.py --single --experiment-name "{name}"'
    run(cmd, f"Generazione config: {name}")


def step_train():
    """Avvia training da un file YAML."""
    if not check_script("train.py"):
        return

    config_path = ask("  Path config YAML (es. configs/cvd/config_balanced.yaml)")
    if not config_path:
        print("  Path non fornito, annullato.")
        return

    if not (REPO_ROOT / config_path).exists():
        print(f"  File non trovato: {config_path}")
        return

    extra = ""
    batch_override = ask("  Override batch size? (vuoto = no)")
    if batch_override:
        extra += f" --batch-size {batch_override}"

    debug = ask("  Modalita debug? (s/n)", "n")
    if debug and debug.lower() == "s":
        extra += " --debug"

    cmd = f'python train.py --config "{config_path}"{extra}'
    run(cmd, f"Training: {config_path}")


def step_train_resume():
    """Resume training da checkpoint specifico."""
    if not check_script("train.py"):
        return

    config_path = ask("  Path config YAML")
    ckpt_path = ask("  Path checkpoint .pth")
    if not config_path or not ckpt_path:
        print("  Path non forniti, annullato.")
        return

    cmd = f'python train.py --config "{config_path}" --resume "{ckpt_path}"'
    run(cmd, f"Resume training da {Path(ckpt_path).name}")


# ============================================================================
# FASE C — Inferenza
# ============================================================================

def phase_inference_menu():
    """Menu test inferenza."""
    while True:
        print(f"""
{'='*60}
  TEST INFERENZA
{'='*60}

  [1] Test inferenza su checkpoint (CLI)
  [2] GUI compensazione immagine singola
  [3] FM100 test
  [0] Indietro
""")
        c = ask_choice(">> Scelta", ["1","2","3","0"])
        if c is None or c == "0":
            break
        elif c == "1":
            step_test_inference()
        elif c == "2":
            step_launch_inference_gui()
        elif c == "3":
            step_fm100_test()


def step_test_inference():
    """Test inferenza su un checkpoint."""
    if not check_script("tools/test_inference_simple.py"):
        return

    ckpt = ask("  Path checkpoint .pth")
    if not ckpt:
        print("  Path non fornito, annullato.")
        return

    cmd = f'python tools/test_inference_simple.py --checkpoint "{ckpt}"'
    run(cmd, f"Test inferenza: {Path(ckpt).name}")


def step_launch_inference_gui():
    """Avvia la GUI compensazione immagine singola."""
    if not check_script("z__inference_gui.py"):
        return

    print("  Avvio GUI compensazione immagine...")
    try:
        subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "z__inference_gui.py")],
            cwd=str(REPO_ROOT)
        )
        print("  GUI avviata in background.")
    except Exception as e:
        print(f"  Errore avvio GUI: {e}")
        print("  Prova manualmente: python z__inference_gui.py")


def step_fm100_test():
    """Esegue FM100 test."""
    if not check_script("FM_TEST.py"):
        return
    run("python FM_TEST.py", "FM100 Test")


# ============================================================================
# Menu principale
# ============================================================================

def main_menu():
    """Menu interattivo principale."""
    while True:
        print(f"""
{'='*60}
  CVD PIPELINE — Menu Principale
{'='*60}
  Workflow: Dataset -> Config -> Training -> Inference -> GUI

  [1] DATASET    — Generazione dataset CVD (teacher-compensated)
  [2] TRAINING   — Configurazione e addestramento modello
  [3] INFERENCE  — Test inferenza e GUI compensazione
  [0] Esci
""")
        c = ask_choice(">> Scelta", ["1","2","3","0"])
        if c is None or c == "0":
            print("Arrivederci!")
            break
        elif c == "1":
            phase_dataset_menu()
        elif c == "2":
            phase_training_menu()
        elif c == "3":
            phase_inference_menu()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CVD Pipeline — Gestione centralizzata del workflow"
    )
    parser.add_argument("--dataset",   action="store_true", help="Menu generazione dataset")
    parser.add_argument("--training",  action="store_true", help="Menu training")
    parser.add_argument("--inference", action="store_true", help="Menu test inferenza")
    parser.add_argument("--gui",       action="store_true", help="Avvia GUI compensazione immagine")
    args = parser.parse_args()

    if args.dataset:
        phase_dataset_menu()
    elif args.training:
        phase_training_menu()
    elif args.inference:
        phase_inference_menu()
    elif args.gui:
        step_launch_inference_gui()
    else:
        main_menu()


if __name__ == "__main__":
    main()
