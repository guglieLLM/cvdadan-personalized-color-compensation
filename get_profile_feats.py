"""
get_profile_feats — Estrazione del profilo CVD 3D da FM100 Hue Test o JSON.

Questo modulo estrae il profilo percettivo individuale dal test
Farnsworth-Munsell 100 Hue (scoring Vingrys–King-Smith) e lo codifica
come vettore 3D [θ, C, S] utilizzabile dal modello CVDCompensationModelAdaIN
tramite condizionamento CVDAdaIN.

Representazione intermedia 6D (legacy, per compatibilità cache):
    [Rmaj, Rmin, dU, dV, S, C]  — calcolata dallo scoring FM100 nel piano
    cromatico CIELUV (u*, v*).

Il modello di training utilizza solo il vettore 3D [θ, C, S]:
    - θ: angolo asse confusione (gradi, nel piano CIELUV u*, v*)
    - C: Confusion index — ampiezza complessiva dell'errore
    - S: Scatter index  — selettività / direzionalità del pattern

Dipendenze:
    FM_TEST (GUI PyQt5 per somministrazione test).
"""

import json
import os
import numpy as np
import sys
from FM_TEST import (FarnsworthTestWrapper as FM_W, unique_identify_path as u_id_path, on_load)
from datetime import datetime


# ------------------------------------------------------------
# Funzione principale per eseguire il test FM100 Hue
# ------------------------------------------------------------
def get_profile_feats_from_test(save_path_json="results.json", parent_window=None):
    """
    Esegue il test FM100 Hue e genera il vettore profile_feats (6D normalizzato).
    
    Args:
        save_path_json: Percorso per salvare risultati JSON del test
        parent_window: Finestra tkinter principale (per evitare conflitti GUI)
    """
    # Salva lo stato della finestra principale se fornita
    saved_geometry = None
    saved_state = None
    if parent_window:
        try:
            saved_geometry = parent_window.geometry()
            saved_state = parent_window.state()
            # Nascondi temporaneamente la finestra principale
            parent_window.withdraw()
        except Exception as e:
            print(f"Warning: Errore nel salvare stato finestra: {e}")
    
    try:
        wrapper = FM_W()
        results = wrapper.run_test_and_get_correction()

        if results is None:
            raise ValueError("Errore: risultati del test non disponibili")

        # Salva i risultati
        with open(save_path_json, 'w') as f:
            json.dump(results, f, indent=4, default=to_serializable)

        # Genera il profilo
        profile_feats, _, _ = extract_profile_feats(results)
        
        return profile_feats
        
    except Exception as e:
        print(f"Errore durante il test FM100: {e}")
        raise
    finally:
        # Ripristina sempre la finestra principale
        if parent_window and saved_geometry:
            try:
                # Usa after per ripristinare dopo un breve delay
                parent_window.after(500, lambda: _restore_parent_window(parent_window, saved_geometry, saved_state))
            except Exception as e:
                print(f"Warning: Errore nel ripristino finestra: {e}")

def _restore_parent_window(parent, geometry, state):
    """Ripristina lo stato della finestra principale dopo il test PyQt5."""
    try:
        parent.deiconify()  # Mostra la finestra
        parent.geometry(geometry)
        if state and state != 'withdrawn':
            parent.state(state)
        parent.lift()
        parent.focus_force()
        # Forza topmost momentaneamente per assicurarsi che sia visibile
        parent.attributes('-topmost', True)
        parent.after(200, lambda: parent.attributes('-topmost', False))
        print(" Finestra principale ripristinata con successo")
    except Exception as e:
        print(f"Warning: Errore ripristino finestra: {e}")


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

# ------------------------------------------------------------
# Funzione per estrarre il vettore da risultati già caricati
# ------------------------------------------------------------
def get_profile_feats_from_file(results):
    """
    Estrae il vettore profile_feats (6D normalizzato) da risultati JSON.
    """
    profile_feats, _, _ = extract_profile_feats(results)
    return profile_feats


# ------------------------------------------------------------
# Funzione che effettua la conversione dei risultati in vettore
# ------------------------------------------------------------
def extract_profile_feats(results):
    """
    Converte i risultati del FM100 Hue in un vettore 6D normalizzato + metadati.

    Returns:
        profile_feats (np.ndarray): vettore 6D normalizzato
        p_raw (np.ndarray): vettore 6D non normalizzato
        theta_deg (float): angolo di confusione in gradi
    """
    if isinstance(results, list):
        results = max(results, key=lambda x: datetime.fromisoformat(x["timestamp"]))

    theta_deg = results["Confusion Angle (degrees)"]
    C = results["C-index"]
    S = results["S-index"]

    # Controlli clinici
    if not -90 <= theta_deg <= 90:
        print(f"[WARNING] θ fuori range clinico: {theta_deg}°")
    if not 0 <= C <= 2.5:
        print(f"[WARNING] C-index fuori range clinico: {C}")
    if not 0 <= S <= 3.0:
        print(f"[WARNING] S-index fuori range clinico: {S}")

    # Costruzione parametri clinici
    Rmaj = C * 9.2
    Rmin = Rmaj / max(S, 1e-6)
    theta_rad = np.deg2rad(theta_deg)
    dU = Rmaj * np.cos(theta_rad)
    dV = Rmaj * np.sin(theta_rad)

    p_raw = np.array([Rmaj, Rmin, dU, dV, S, C], dtype=np.float32)
    profile_feats = p_raw / (np.linalg.norm(p_raw) + 1e-12)

    return profile_feats, p_raw, theta_deg


# ------------------------------------------------------------
# Funzione per caricare e salvare profili esistenti
# ------------------------------------------------------------
def extract_profile_feats_on_existing_profile(name, output_file_path=None, created_id=True, path=None):
    """
    Carica un profilo utente esistente (JSON) e salva il vettore profile_feats (6D) come .npy
    + un JSON con metadati (theta_deg e p_raw).
    """
    path_load_JSON = path
    if path is None:
        path_load_JSON, unique_id = u_id_path(
            name=name, type_file=".json", created_id=created_id, folder_name="dataset_test"
        )

    results = on_load(gui=False, path=path_load_JSON)
    if results is None:
        raise FileNotFoundError(f"Nessun profilo JSON trovato per l'utente '{name}'")

    feats, p_raw, theta_deg = extract_profile_feats(results)

    # Salvataggio .npy
    if output_file_path is None:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_directory, "profiles_feats_users")
        os.makedirs(output_dir, exist_ok=True)
        # fall back per unique-id
        if 'unique_id' not in locals():
            unique_id = os.path.splitext(os.path.basename(path_load_JSON))[0]

        output_file_path = os.path.join(output_dir, unique_id + "-profile_feats6.npy")

    np.save(output_file_path, feats)

    # Salvataggio JSON metadati
    meta_path = output_file_path.replace(".npy", ".json")
    with open(meta_path, "w") as f:
        json.dump({
            "theta_deg": float(theta_deg),
            "p_raw": p_raw.tolist(),
            "profile_feats_norm": feats.tolist()
        }, f, indent=4)

    print(f"[INFO] Vettore profile_feats salvato in {output_file_path}")
    print(f"[INFO] Metadati salvati in {meta_path}")
    print(f"[INFO] Contenuto vettore:", feats)

    return output_file_path, unique_id


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]

    if any(arg.startswith("-utente=") for arg in args):
        user_flag = [arg for arg in args if arg.startswith("-utente=")][0]
        user_name = user_flag.split("=", 1)[1]
        extract_profile_feats_on_existing_profile(user_name)
    else:
        feats = get_profile_feats_from_test()
        print("profile_feats generato:", feats)
