"""
Ottimizzazioni per il caricamento del CVD dataset
"""

import json
import pickle
import time
from pathlib import Path
import numpy as np
import math


def create_lightweight_cache(json_path, cache_path=None):
    """
    Crea una versione cache più leggera del JSON
    Mantiene i dati essenziali per il training (immagini + profili sintetici)
    """
    if cache_path is None:
        cache_path = Path(str(json_path).replace('.json', '_cache.pkl'))

    print(f"[CACHE] Creating lightweight cache: {cache_path}")
    start_time = time.time()

    with open(json_path, 'r') as f:
        full_data = json.load(f)

    # Normalize top-level structure: accept {'meta':..., 'pairs': [...]} or plain list
    if isinstance(full_data, dict) and "pairs" in full_data:
        items = full_data["pairs"]
    else:
        items = full_data

    # Estrai solo i dati essenziali
    lightweight_data = []
    for item in items:
        # Costruisci il profilo 6D robusto
        profile_6d = build_profile_6d(item)

        # Calcola theta_deg dai dU, dV per compatibility con severity
        dU, dV = profile_6d[2], profile_6d[3]
        theta_deg = math.degrees(math.atan2(dV, dU))

        # Supporta entrambi i formati di path
        # Nuovo formato (Teacher Farup): image_normal, image_compensated
        # Vecchio formato: original_path, simulated_path
        original_path = item.get('image_normal') or item.get('original_path', '')
        compensated_path = item.get('image_compensated') or item.get('simulated_path', '')
        
        # Normalizza percorsi (Unix-style per consistenza)
        if original_path:
            original_path = original_path.replace('\\', '/')
        if compensated_path:
            compensated_path = compensated_path.replace('\\', '/')

        essential_item = {
            'original_path': original_path,
            'compensated_path': compensated_path,
            # Legacy compatibility
            'simulated_path': compensated_path,
            # Anche con le nuove chiavi per compatibilità con cvd_dataset_loader
            'image_normal': original_path,
            'image_compensated': compensated_path,
            'profile_6d': profile_6d,
            'cvd_type': item.get('cvd_type', 'unknown'),
            'severity': item.get('severity', theta_deg),  # Usa severity dal JSON se presente
            'qa_passed': item.get('qa_passed', True),  # Per filtraggio QA
            # PRESERVE profile_x for 3D profile extraction in cvd_dataset_loader
            'profile_x': item.get('profile_x'),
            # Also preserve cvd_params for fallback
            'cvd_params': item.get('cvd_params'),
        }
        lightweight_data.append(essential_item)

    # Salva cache binaria
    with open(cache_path, 'wb') as f:
        pickle.dump(lightweight_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    load_time = time.time() - start_time
    original_size = Path(json_path).stat().st_size / 1024 / 1024
    cache_size = cache_path.stat().st_size / 1024 / 1024

    print(f"[CACHE] Created in {load_time:.2f}s")
    print(f"[CACHE] Size reduction: {original_size:.1f}MB -> {cache_size:.1f}MB ({cache_size/original_size:.1%})")

    return cache_path


def build_profile_6d(item):
    """
    Ricostruisce un profilo 6D coerente a partire da vari formati:
    - Nuovo formato (Teacher Farup): profile_x con {theta_deg, C_index, S_index}
    - Vecchio formato: p_star / p_raw / p_norm
    
    Profilo 6D: [Rmaj, Rmin, dU, dV, S, C] - theta è ridondante (ricavabile da dU, dV)
    """
    # NUOVO FORMATO: profile_x dal Teacher Farup
    if "profile_x" in item and item["profile_x"] is not None:
        profile_x = item["profile_x"]
        theta_deg = profile_x.get("theta_deg", 0.0)
        C_index = profile_x.get("C_index", 0.0)
        S_index = profile_x.get("S_index", 0.0)
        
        # Converti theta_deg in dU, dV (vettore unitario sulla confusion line)
        theta_rad = math.radians(theta_deg)
        dU = math.cos(theta_rad)
        dV = math.sin(theta_rad)
        
        # Per il nuovo formato, non abbiamo Rmaj/Rmin, usiamo valori default
        # Il modello userà principalmente theta, S, C
        Rmaj = 1.0  # Valore di default
        Rmin = 0.5  # Valore di default
        
        return [Rmaj, Rmin, dU, dV, S_index, C_index]
    
    # VECCHIO FORMATO: p_raw (6D)
    if "p_raw" in item and item["p_raw"] is not None:
        p_raw = list(item["p_raw"])[:6]
        if len(p_raw) == 6:
            return p_raw

    # Fallback: p_norm (6D normalized)
    if "p_norm" in item and item["p_norm"] is not None:
        p_norm = list(item["p_norm"])[:6]
        if len(p_norm) == 6:
            return p_norm

    # Fallback: p_star (se è 6D usa direttamente, se è 7D rimuovi theta)
    if "p_star" in item and item["p_star"] is not None:
        p = list(item["p_star"])
        if len(p) == 6:
            return p
        elif len(p) == 7:
            # 7D: [Rmaj, Rmin, phi, dU, dV, S, C] -> rimuovi phi (indice 2)
            Rmaj, Rmin, phi, dU, dV, S, C = p
            return [Rmaj, Rmin, dU, dV, S, C]

    # Se tutto fallisce -> profilo nullo
    return [0.0] * 6


def load_dataset_smart(json_path, use_cache=True):
    """
    Carica il dataset in modo intelligente usando cache se disponibile
    """
    json_path = Path(json_path)
    cache_path = Path(str(json_path).replace('.json', '_cache.pkl'))

    # Se cache esiste ed è aggiornata -> usala
    if use_cache and cache_path.exists():
        json_time = json_path.stat().st_mtime
        cache_time = cache_path.stat().st_mtime

        if cache_time > json_time:
            print(f"[SMART LOAD] Using cache: {cache_path}")
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            load_time = time.time() - start_time
            print(f"[SMART LOAD] Loaded {len(data)} items in {load_time:.2f}s (cache)")
            return data

    # Altrimenti carica JSON e crea cache
    print(f"[SMART LOAD] Loading JSON: {json_path}")
    start_time = time.time()
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    
    # Normalize top-level structure for load_dataset_smart
    if isinstance(full_data, dict) and "pairs" in full_data:
        full_data = full_data["pairs"]
        
    load_time = time.time() - start_time
    print(f"[SMART LOAD] Loaded {len(full_data)} items in {load_time:.2f}s (JSON)")

    # Crea cache per la prossima volta
    if use_cache:
        create_lightweight_cache(json_path, cache_path)

    return full_data


def get_dataset_stats(data):
    """
    Calcola statistiche rapide del dataset
    """
    if not data:
        return {}

    # Handle canonical format with 'pairs' key
    items = data.get('pairs', data) if isinstance(data, dict) else data

    # Supporto retrocompatibile per profili 6D e 7D
    profiles = []
    for item in items:
        if 'profile_6d' in item and item['profile_6d'] is not None:
            profiles.append(item['profile_6d'])
        elif 'profile_7d' in item and item['profile_7d'] is not None:
            # Conversione automatica da 7D a 6D (rimuovi θ_deg all'indice 2)
            p7d = item['profile_7d']
            if len(p7d) == 7:
                profiles.append([p7d[0], p7d[1], p7d[3], p7d[4], p7d[5], p7d[6]])  # Skip θ_deg
            else:
                profiles.append(p7d)  # Fallback
        elif 'p_star' in item and item['p_star'] is not None:
            # Usa sistema esistente di conversione
            profiles.append(build_profile_6d(item))
        else:
            print(f"[WARNING] Item senza profilo valido: {item.keys()}")
            
    cvd_types = [item['cvd_type'] for item in items]
    severities = [item['severity'] for item in items]

    profiles = np.array(profiles)

    return {
        'count': len(items),
        'cvd_types': list(set(cvd_types)),
        'severity_range': (min(severities), max(severities)),
        'profile_mean': profiles.mean(axis=0).tolist(),
        'profile_std': profiles.std(axis=0).tolist()
    }


if __name__ == "__main__":
    # Test del sistema di cache
    train_json = "dataset/.../mapping_train.json"
    val_json = "dataset/.../mapping_val.json"

    print("=== Testing Smart Loading System ===")

    # Train
    print("\n--- TRAIN SET ---")
    train_data = load_dataset_smart(train_json, use_cache=True)
    train_stats = get_dataset_stats(train_data)
    print(f"Train stats: {train_stats['count']} samples, CVD types: {train_stats['cvd_types']}")

    # Val
    print("\n--- VAL SET ---")
    val_data = load_dataset_smart(val_json, use_cache=True)
    val_stats = get_dataset_stats(val_data)
    print(f"Val stats: {val_stats['count']} samples, CVD types: {val_stats['cvd_types']}")

    print("\n=== Smart loading system ready! ===")
