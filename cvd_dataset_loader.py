"""
cvd_dataset_loader — PyTorch Dataset per il training di CVDCompensationModelAdaIN.

Carica coppie (immagine_normale, immagine_compensata) + profilo CVD 3D
dai file JSON di mapping generati dalla pipeline di dataset.

Dipendenze:
    mapping JSON (generato da ``CVD_dataset_generator/dataset_generator_teacher.py``),
    torchvision.transforms, cvd_constants.
"""

import json
import torch
import time
import sys
import re
import multiprocessing
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import os
import math


def get_optimal_num_workers():
    """
    Calcola automaticamente il numero ottimale di workers per DataLoader
    basato sui core CPU disponibili.
    
    Returns:
        int: Numero ottimale di workers
    """
    cpu_count = multiprocessing.cpu_count()
    
    # Strategia ottimizzata per diverse configurazioni hardware:
    if cpu_count >= 16:      # High-end workstation/server
        optimal = min(12, cpu_count * 2 // 3)
    elif cpu_count >= 8:     # Standard desktop/laptop 8+ core  
        optimal = min(6, cpu_count * 3 // 4)
    elif cpu_count >= 4:     # Mid-range 4-6 core
        optimal = min(4, cpu_count // 2)
    elif cpu_count >= 2:     # Low-end dual core
        optimal = 2
    else:                    # Single core (rare)
        optimal = 1
    
    print(f"[AUTO WORKERS] CPU cores: {cpu_count} -> Using {optimal} workers")
    return optimal


class CVDMappingDataset(Dataset):
    """Dataset per il training di CVDCompensationModelAdaIN.

    Carica coppie (image_normal, image_compensated) con profilo CVD 3D:
        - INPUT:  immagini normali (visione normovedente)
        - TARGET: immagini compensate dal Teacher Farup (gold standard offline)
        - CONDITIONING: profilo CVD 3D [θ, C, S] normalizzato (schema ibrido)

    Direzione Training: Normale → CVD-Friendly (daltonizzazione).

    QA Filtering:
        Se ``use_qa_filter=True``, scarta samples con ``qa_passed=False``.

    Normalizzazione profilo (schema ibrido):
        [θ_norm, C_norm, S_norm]
        - θ: normalizzazione GLOBALE (preserva distinzione protan/deutan/tritan)
        - C, S: normalizzazione PER-TIPO CVD (gestisce distribuzioni diverse)
        - Non serve one-hot encoding: θ_norm già discrimina i tipi CVD
    """
    
    def __init__(self, mapping_json_path, base_path=".", split="train", transform=None, 
                 per_cvd_stats=None, theta_global_stats=None, use_qa_filter=True):
        """
        Args:
            mapping_json_path (str): Path al file mapping JSON (train o val)
            base_path (str): Path base per i percorsi relativi
            split (str): "train" o "val"
            transform (callable): Trasformazioni per le immagini
            per_cvd_stats (dict): Statistiche PER-CVD TYPE per C/S (normalizzazione ibrida)
                {
                    'protan': {'C_mean': float, 'C_std': float, 'S_mean': float, 'S_std': float},
                    'deutan': {...},
                    'tritan': {...}
                }
            theta_global_stats (dict): Statistiche GLOBALI per θ (normalizzazione ibrida)
                {'mean': float, 'std': float}
            use_qa_filter (bool): Se True, filtra samples con qa_passed=False
        
        NOTE NORMALIZZAZIONE IBRIDA:
            - theta_global_stats contiene stats GLOBALI per θ → preserva distinzione tipi CVD
            - per_cvd_stats contiene stats per-tipo per C/S → gestisce outlier
            - Output profile shape: [3] = [θ_norm, C_norm, S_norm]
            - Non serve one-hot: θ normalizzato globalmente già discrimina i tipi
        """
        self.base_path = Path(base_path)
        self.split = split
        self.transform = transform
        self.use_qa_filter = use_qa_filter
        
        # NORMALIZZAZIONE IBRIDA: θ globale + C/S per-tipo CVD
        self.per_cvd_stats = per_cvd_stats        # Stats per C/S (per-tipo CVD)
        self.theta_global_stats = theta_global_stats  # Stats per θ (globali)
        
        # File di log per tenere traccia dei file mancanti
        self.missing_files_log = []
        
        print(f"[CVDMappingDataset] Loading {split} mapping from: {mapping_json_path}")
        print(f"[CVDMappingDataset] QA filter: {'ENABLED' if use_qa_filter else 'DISABLED'}")
        
        # Sistema di caricamento intelligente con cache opzionale
        # Per dataset grandi (>100MB), questo può ridurre significativamente i tempi di startup
        try:
            from cvd_cache_optimizer import load_dataset_smart
            
            start_time = time.time()
            self.mapping_data = load_dataset_smart(mapping_json_path, use_cache=True)
            load_time = time.time() - start_time
            
            print(f"[CVDMappingDataset] Loaded {len(self.mapping_data)} samples for {split} in {load_time:.2f}s")
            
        except ImportError:
            # Fallback al caricamento standard se il cache optimizer non è disponibile
            print(f"[CVDMappingDataset] Using standard JSON loading (cache optimizer not found)")
            start_time = time.time()
            
            with open(mapping_json_path, 'r') as f:
                self.mapping_data = json.load(f)
            
            load_time = time.time() - start_time
            print(f"[CVDMappingDataset] Loaded {len(self.mapping_data)} samples for {split} in {load_time:.2f}s")
        
        # ensure mapping_data is the list of pairs (support canonical mapping with top-level "pairs")
        if isinstance(self.mapping_data, dict) and "pairs" in self.mapping_data:
            self.mapping_data = self.mapping_data["pairs"]
        
        # =========================================================================
        # QA FILTER: Scarta samples con qa_passed=False (se use_qa_filter=True)
        # Il campo qa_passed viene aggiunto da post_validate_deltae_v2.py
        # Default qa_passed=True per retrocompatibilità con mapping senza campo qa
        # =========================================================================
        if self.use_qa_filter:
            pre_qa_count = len(self.mapping_data)
            self.mapping_data = [
                item for item in self.mapping_data 
                if item.get("qa_passed", True)  # Default True per retrocompatibilità
            ]
            qa_filtered_count = pre_qa_count - len(self.mapping_data)
            
            if qa_filtered_count > 0:
                print(f"[CVDMappingDataset] QA FILTER: {qa_filtered_count:,d} samples removed (qa_passed=False)")
                print(f"[CVDMappingDataset] QA FILTER: {len(self.mapping_data):,d} samples remaining")
        
        # Pre-calcola statistiche per monitoraggio
        # Supporta sia 'cvd_type' (English) che 'cluster' (Italian)
        cvd_types = []
        severities = []
        cvd_severity_pairs = []  # Traccia combinazioni tipo+severity
        
        for item in self.mapping_data:
            if 'cvd_type' in item:
                cvd_type = item['cvd_type']
            elif 'cluster' in item:
                cvd_type = item['cluster']
            else:
                cvd_type = 'unknown'
            
            cvd_types.append(cvd_type)
            
            # Estrai severity dal campo specifico o dal path completo
            severity = item.get('severity', None)
            if severity is None or severity == 0.0:
                # Prova a estrarre dal path completo (cartella o file)
                # Supporta sia nuovo schema (image_compensated) che legacy (simulated_path)
                sim_path = item.get('image_compensated') or item.get('simulated_path', '')
                # Pattern 1: cerca _t_<numero>_ nel path (es. machado_CVD_t_0.15_v_1)
                match = re.search(r'_t_([\d.]+)_', sim_path)
                if match:
                    severity = float(match.group(1))
                else:
                    # Pattern 2: cerca severity=<numero> o sev<numero>
                    match2 = re.search(r'(?:severity|sev)[_=]([\d.]+)', sim_path, re.IGNORECASE)
                    if match2:
                        severity = float(match2.group(1))
                    else:
                        # DEBUG: stampa il primo path che non matcha per capire il formato
                        if len(severities) == 0:
                            print(f"[DEBUG] First compensated_path sample: {sim_path}")
                        severity = 0.0
            
            severities.append(severity)
            cvd_severity_pairs.append((cvd_type, severity))
        
        unique_cvd = set(cvd_types)
        unique_combinations = set(cvd_severity_pairs)
        
        print(f"[CVDMappingDataset] CVD types: {unique_cvd}")
        print(f"[CVDMappingDataset] Unique CVD type+severity combinations: {len(unique_combinations)}")
        print(f"[CVDMappingDataset] Severity range: {min(severities):.2f} - {max(severities):.2f}")
        
        # Mostra le combinazioni uniche
        if len(unique_combinations) <= 20:  # Mostra solo se non troppe
            sorted_combinations = sorted(unique_combinations, key=lambda x: (x[0], x[1]))
            print(f"[CVDMappingDataset] Combinations: {sorted_combinations}")
        
        # Filtra i dati per rimuovere file mancanti
        original_count = len(self.mapping_data)
        self.mapping_data = self._filter_existing_files()
        filtered_count = len(self.mapping_data)
        
        if original_count != filtered_count:
            removed_count = original_count - filtered_count
            print(f"[CVDMappingDataset] WARNING: {removed_count}/{original_count} samples removed due to missing files")
            
            # Salva il log dei file mancanti
            if self.missing_files_log:
                log_path = f"missing_files_{split}.log"
                with open(log_path, 'w') as f:
                    f.write(f"Missing files log for {split} dataset\n")
                    f.write(f"Total missing: {len(self.missing_files_log)}\n\n")
                    for missing_file in self.missing_files_log:
                        f.write(f"{missing_file}\n")
                print(f"[CVDMappingDataset] Missing files logged to: {log_path}")
        
        print(f"[CVDMappingDataset] Final dataset size: {len(self.mapping_data)} samples")
        
        # Cache dei percorsi per validation startup più veloce
        self._validate_paths = True  # Opzione per validare i path al startup
        
    def _filter_existing_files(self):
        """
        Filtra il dataset rimuovendo elementi con file mancanti
        
        Returns:
            list: Dataset filtrato con solo file esistenti
        """
        filtered_data = []
        
        for idx, item in enumerate(self.mapping_data):
            try:
                # Determina le chiavi corrette per questo item
                # Nuovo schema: image_normal/image_compensated
                # Legacy schema: original_path/simulated_path
                if 'image_normal' in item and 'image_compensated' in item:
                    normal_key = 'image_normal'
                    compensated_key = 'image_compensated'
                else:
                    normal_key = 'original_path'
                    compensated_key = 'simulated_path'
                
                # Controlla se esistono tutti i file necessari
                compensated_exists = self._check_file_exists(item.get(compensated_key, ''))
                original_exists = self._check_file_exists(item.get(normal_key, ''))
                
                if compensated_exists and original_exists:
                    filtered_data.append(item)
                else:
                    # Log dei file mancanti
                    missing_info = {
                        'idx': idx,
                        'compensated_path': item.get(compensated_key, 'N/A'),
                        'compensated_exists': compensated_exists,
                        'original_path': item.get(normal_key, 'N/A'),
                        'original_exists': original_exists,
                        'cvd_type': item.get('cvd_type', item.get('cluster', 'unknown')),  # SMART: try both fields
                        'severity': item.get('severity', 'unknown')
                    }
                    self.missing_files_log.append(missing_info)
                    
            except Exception as e:
                # In caso di errore, logga e salta questo elemento
                error_info = {
                    'idx': idx,
                    'error': str(e),
                    'item_keys': list(item.keys()) if isinstance(item, dict) else 'invalid_item'
                }
                self.missing_files_log.append(error_info)
        
        return filtered_data
    
    def _check_file_exists(self, file_path):
        """
        Controlla se un file esiste utilizzando la stessa logica di __getitem__
        
        Args:
            file_path (str): Percorso del file da controllare
            
        Returns:
            bool: True se il file esiste, False altrimenti
        """
        try:
            path = Path(file_path)
            
            # Se il percorso non esiste, prova a combinarlo con base_path
            if not path.exists():
                path = self.base_path / file_path
            
            # Se ancora non esiste, prova il percorso come relativo dalla directory corrente
            if not path.exists():
                path = Path('.') / file_path
            
            return path.exists()
            
        except Exception:
            return False
        
    def __len__(self):
        return len(self.mapping_data)
    
    def __getitem__(self, idx):
        """
        Returns dict with:
            'input': Immagine NORMALE [3, H, W] (input per training)
            'target': Immagine COMPENSATA [3, H, W] (target per training)
            'profile': Profilo CVD 6D NORMALIZZATO [6] per conditioning
            'metadata': Dict con cvd_type, severity, ecc.
        """
        import math

        item = self.mapping_data[idx]

        # -------------------------------
        # 1. Verifica chiavi immagini (NUOVO schema)
        # -------------------------------
        # Supporta sia schema nuovo (image_normal/image_compensated) 
        # che schema legacy (original_path/simulated_path)
        
        if 'image_normal' in item and 'image_compensated' in item:
            # NUOVO schema daltonization
            normal_key = 'image_normal'
            compensated_key = 'image_compensated'
        else:
            # Legacy schema (per compatibilità)
            normal_key = 'original_path'
            compensated_key = 'simulated_path'
        
        if normal_key not in item or compensated_key not in item:
            raise KeyError(f"Item {idx} missing image keys. Available: {list(item.keys())}")

        # -------------------------------
        # 2. Caricamento immagini
        # -------------------------------
        # Logica di risoluzione percorsi - supporta:
        # 1. Percorsi relativi dalla root del progetto (es: dataset/places365/...)
        # 2. Percorsi relativi dal base_path (es: train/cafeteria/...)
        # 3. Percorsi assoluti (legacy)
        
        def resolve_image_path(path_str: str, base_path: Path) -> Path:
            """Risolve un percorso immagine provando diverse strategie."""
            path_str = path_str.replace('\\', '/')  # Normalizza separatori
            
            # 1. Prova come path relativo dalla root del progetto (nuovo formato)
            if path_str.startswith('dataset/'):
                resolved = Path('.') / path_str
                if resolved.exists():
                    return resolved
            
            # 2. Prova come path assoluto
            direct_path = Path(path_str)
            if direct_path.is_absolute() and direct_path.exists():
                return direct_path
            
            # 3. Prova relativo al base_path
            relative_to_base = base_path / path_str
            if relative_to_base.exists():
                return relative_to_base
            
            # 4. Prova dalla directory corrente
            from_cwd = Path('.') / path_str
            if from_cwd.exists():
                return from_cwd
            
            # 5. Se il path contiene 'dataset/', prova a estrarre e risolvere
            if '/dataset/' in path_str or path_str.startswith('dataset/'):
                if '/dataset/' in path_str:
                    idx = path_str.index('/dataset/')
                    rel_path = path_str[idx+1:]  # Rimuovi leading /
                else:
                    rel_path = path_str
                resolved = Path('.') / rel_path
                if resolved.exists():
                    return resolved
            
            # Fallback: ritorna il path relativo al base_path (per errore migliore)
            return relative_to_base
        
        # INPUT: Immagine normale
        normal_path = resolve_image_path(item[normal_key], self.base_path)
        normal_image = Image.open(normal_path).convert('RGB')

        # TARGET: Immagine compensata
        compensated_path = resolve_image_path(item[compensated_key], self.base_path)
        compensated_image = Image.open(compensated_path).convert('RGB')

        # -------------------------------
        # 3. Applica trasformazioni
        # -------------------------------
        if self.transform:
            normal_image = self.transform(normal_image)
            compensated_image = self.transform(compensated_image)

        # -------------------------------
        # 4. Estrazione profilo 3D NATIVO
        # -------------------------------
        # Profilo 3D: [theta_deg, C_index, S_index] - formato nativo da dataset_generator_teacher.py
        
        if "profile_x" in item and item["profile_x"] is not None:
            # NUOVO FORMATO 3D (dataset generato con profili clinici)
            profile_x = item["profile_x"]
            profile_3d = [
                float(profile_x.get("theta_deg", 0.0)),
                float(profile_x.get("C_index", 0.0)),
                float(profile_x.get("S_index", 0.0))
            ]
            theta_deg_raw = profile_3d[0]  # Per metadata
        elif "cvd_params" in item and "x_original" in item.get("cvd_params", {}):
            # FALLBACK: Estrai da cvd_params.x_original (dataset vecchio formato)
            # map_x_to_cvd_params() salva x_original = {theta_deg, C_index, S_index}
            x_orig = item["cvd_params"]["x_original"]
            profile_3d = [
                float(x_orig.get("theta_deg", 0.0)),
                float(x_orig.get("C_index", 0.0)),
                float(x_orig.get("S_index", 0.0))
            ]
            theta_deg_raw = profile_3d[0]
            if idx == 0:  # Info solo prima volta
                print(f"[INFO] Sample {idx} using profile from cvd_params.x_original (legacy format)")
        else:
            # Fallback finale: profilo zero per compatibilità
            profile_3d = [0.0, 0.0, 0.0]
            theta_deg_raw = 0.0
            if idx == 0:  # Warning solo prima volta
                print(f"[WARNING] Sample {idx} missing profile_x AND cvd_params.x_original, using zero profile")

        profile_3d = torch.tensor(profile_3d, dtype=torch.float32)
        
        # -------------------------------
        # 4.5. NORMALIZZAZIONE IBRIDA (3D vector)
        # -------------------------------
        # θ: normalizzazione GLOBALE → preserva distinzione Protan/Deutan/Tritan
        # C, S: normalizzazione PER-CVD TYPE → gestisce outlier
        # Output: [θ_norm, C_norm, S_norm] - NO one-hot (θ già discrimina i tipi)
        
        from cvd_constants import classify_cvd_type_from_theta
        
        # 1. Classifica CVD type dal theta RAW (per lookup stats C/S)
        cvd_type_classified = classify_cvd_type_from_theta(theta_deg_raw)
        
        # 2. Normalizzazione IBRIDA
        # Controlliamo se abbiamo le statistiche ibride (nuovo formato)
        if hasattr(self, 'theta_global_stats') and self.theta_global_stats is not None:
            # NUOVO FORMATO: Usa stats ibride
            theta_mean = self.theta_global_stats['mean']
            theta_std = max(self.theta_global_stats['std'], 1e-8)
            theta_norm = (profile_3d[0] - theta_mean) / theta_std
            
            # C e S con stats per-tipo CVD
            if self.per_cvd_stats is not None and cvd_type_classified in self.per_cvd_stats:
                type_stats = self.per_cvd_stats[cvd_type_classified]
                C_mean = type_stats.get('C_mean', type_stats.get('mean', [0, 0, 0])[1] if 'mean' in type_stats else 0)
                C_std = max(type_stats.get('C_std', type_stats.get('std', [1, 1, 1])[1] if 'std' in type_stats else 1), 1e-8)
                S_mean = type_stats.get('S_mean', type_stats.get('mean', [0, 0, 0])[2] if 'mean' in type_stats else 0)
                S_std = max(type_stats.get('S_std', type_stats.get('std', [1, 1, 1])[2] if 'std' in type_stats else 1), 1e-8)
                
                C_norm = (profile_3d[1] - C_mean) / C_std
                S_norm = (profile_3d[2] - S_mean) / S_std
            else:
                C_norm = profile_3d[1]
                S_norm = profile_3d[2]
            
            profile_3d_normalized = torch.tensor([theta_norm, C_norm, S_norm], dtype=torch.float32)
        elif self.per_cvd_stats is not None and cvd_type_classified in self.per_cvd_stats:
            # FALLBACK: vecchio formato (tutto per-tipo) - per retrocompatibilità temporanea
            type_stats = self.per_cvd_stats[cvd_type_classified]
            profile_mean_tensor = torch.tensor(type_stats['mean'], dtype=torch.float32)
            profile_std_tensor = torch.tensor(type_stats['std'], dtype=torch.float32)
            profile_std_tensor = torch.clamp(profile_std_tensor, min=1e-8)
            profile_3d_normalized = (profile_3d - profile_mean_tensor) / profile_std_tensor
        else:
            # No normalization (raw values)
            profile_3d_normalized = profile_3d

        # -------------------------------
        # 5. Metadati (severity da theta_deg RAW)
        # -------------------------------
        # Estrai cvd_type da cvd_params (nuovo schema)
        cvd_params = item.get('cvd_params', {})
        cvd_type = cvd_params.get('cvd_type', 'unknown')
        # Salva valori RAW (non normalizzati) per visualizzazione nei plot
        profile_x_raw = item.get("profile_x", {})
        
        metadata = {
            'cvd_type': cvd_type,
            'cvd_type_classified': cvd_type_classified,  # tipo classificato dal theta
            'severity': float(theta_deg_raw),  # theta_deg non normalizzato
            'image_name': item.get('files', {}).get('original', 'unknown'),
            'delta_e_mean': float(item.get('teacher', {}).get('delta_e', 0.0)),
            # Valori RAW per visualizzazione (NON per il modello)
            'theta_deg_raw': float(profile_x_raw.get('theta_deg', theta_deg_raw)),
            'C_index_raw': float(profile_x_raw.get('C_index', 0.0)),
            'S_index_raw': float(profile_x_raw.get('S_index', 0.0)),
        }

        # -------------------------------
        # 6. Return dict (profilo 3D con normalizzazione ibrida)
        # -------------------------------
        return {
            'input': normal_image,              # Immagine normale (INPUT)
            'target': compensated_image,        # Immagine compensata (TARGET)
            'profile': profile_3d_normalized,   # Profilo CVD 3D [θ_norm, C_norm, S_norm]
            'metadata': metadata                # Metadati (cvd_type, severity, ecc.)
        }



def create_cvd_dataloaders(dataset_base_path, batch_size=32, num_workers=None, pin_memory=True, 
                          cache_preprocessing=False, include_test=True, prefetch_factor=None,
                          persistent_workers=None, image_size=256):
    """
    Crea DataLoader per training e validation del modello CVD
    
    Args:
        dataset_base_path (str): Path base del dataset
        batch_size (int): Batch size
        num_workers (int): Worker threads (None = auto-detection)
        pin_memory (bool): Pin memory per GPU
        cache_preprocessing (bool): Se True, salva una versione pre-processata per caricamenti futuri
        include_test (bool): Se True, crea anche il test dataset
        prefetch_factor (int): Batches to prefetch per worker (None = default 2)
        persistent_workers (bool): Keep workers alive between epochs (None = auto based on num_workers)
        image_size (int): Risoluzione immagini (default 256 = dimensione dataset)
    
    Returns:
        se include_test=False: train_loader, val_loader, train_dataset, val_dataset
        se include_test=True: train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    
    # Auto-detection dei workers se non specificati
    if num_workers is None:
        num_workers = get_optimal_num_workers()
        # Su CPU a 4 core, limitiamo a 4 workers (test ha mostrato 28.03 batch/sec ottimali)
        cpu_count = os.cpu_count()
        if cpu_count <= 8:  # CPU limitata
            num_workers = min(4, num_workers)  # Max 4 workers per CPU piccole
            print(f"[CPU LIMITED] Detected {cpu_count} cores, limited to {num_workers} workers for optimal throughput")
        print(f"[AUTO WORKERS] Using auto-detected {num_workers} workers")
    else:
        print(f"[MANUAL WORKERS] Using manually specified {num_workers} workers")
    
    # Prefetch factor (default 2 if workers > 0, None otherwise)
    if prefetch_factor is None and num_workers is not None and num_workers > 0:
        prefetch_factor = 2
    elif num_workers == 0:
        prefetch_factor = None
    
    # Persistent workers (default True if workers > 0)
    if persistent_workers is None:
        persistent_workers = (num_workers is not None and num_workers > 0)
    
    
    # Usa direttamente il percorso fornito
    base_path = Path(dataset_base_path)
    
    # I mapping JSON dovrebbero essere direttamente nella directory fornita
    train_json = base_path / "mapping_train.json"
    val_json = base_path / "mapping_val.json"
    test_json = base_path / "mapping_test.json"
    
    print(f"train path:  {train_json}")
    print(f"val path:  {val_json}")
    
    if include_test:
        print(f"test path:  {test_json}")
    
    if not train_json.exists():
        raise FileNotFoundError(f"Training mapping not found: {train_json}")
    if not val_json.exists():
        raise FileNotFoundError(f"Validation mapping not found: {val_json}")
    if include_test and not test_json.exists():
        print(f"[WARNING] Test mapping not found: {test_json}")
        print(f"[WARNING] Proceeding without test dataset")
        include_test = False
    
    print(f"[CVD DataLoaders] Loading from: {base_path}")
    print(f"[CVD DataLoaders] Train JSON: {train_json.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"[CVD DataLoaders] Val JSON: {val_json.stat().st_size / 1024 / 1024:.1f} MB")
    
    if include_test and test_json.exists():
        print(f"[CVD DataLoaders] Test JSON: {test_json.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Trasformazioni (ImageNet normalization per compatibilità con modelli pre-trained)
    # Risoluzione configurabile via parametro image_size (default 256 = dimensione dataset)
    print(f"[CVD DataLoaders] Image resolution: {image_size}x{image_size}")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Risoluzione configurabile
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # Normalizzazione ImageNet standard
    ])
    
    # PASSO 1: Crea dataset training SENZA normalizzazione per calcolare statistiche
    print("[CVD DataLoaders] Step 1: Computing HYBRID profile normalization statistics...")
    print("[CVD DataLoaders]          (θ GLOBAL + C/S per-CVD type)")
    
    # Controlla se esistono statistiche salvate (nome file: profile_normalization_stats.npz)
    stats_cache_path = base_path / "profile_normalization_stats.npz"
    
    per_cvd_stats = None
    theta_global_stats = None
    cached = None
    
    if stats_cache_path.exists():
        try:
            cached = np.load(stats_cache_path, allow_pickle=True)
            # Carica per_cvd_stats (dict con stats per tipo CVD)
            if 'per_cvd_stats' in cached:
                per_cvd_stats = cached['per_cvd_stats'].item()  # .item() per convertire da numpy array a dict
                # Carica theta_global_stats se presente
                if 'theta_global_stats' in cached:
                    theta_global_stats = cached['theta_global_stats'].item()
                elif 'global_mean' in cached:
                    # Fallback: ricostruisci da global_mean/std
                    theta_global_stats = {
                        'mean': float(cached['global_mean'][0]),
                        'std': float(cached['global_std'][0])
                    }
                else:
                    # Nessun theta_global_stats → ricalcola tutto
                    print(f"[CVD DataLoaders] Cache missing theta_global_stats, recomputing...")
                    per_cvd_stats = None
            else:
                print(f"[CVD DataLoaders] Warning: Cache file exists but missing 'per_cvd_stats', recomputing...")
                per_cvd_stats = None
        except Exception as e:
            print(f"[CVD DataLoaders] Warning: Failed to load cached stats ({e}), recomputing...")
            per_cvd_stats = None
    
    # Calcola statistiche se non disponibili
    # NOTE: non sovrascrivere theta_global_stats se è stato caricato dalla cache.
    if per_cvd_stats is None:
        train_dataset_temp = CVDMappingDataset(
            mapping_json_path=train_json,
            base_path=base_path,
            split="train",
            transform=transform,
            per_cvd_stats=None,  # No normalization per il calcolo delle statistiche
            theta_global_stats=None
        )
        
        # PASSO 2: Calcola statistiche IBRIDE profili 3D dal training set
        # - θ: stats GLOBALI (preserva distinzione Protan/Deutan/Tritan)
        # - C/S: stats PER-CVD TYPE (gestisce outlier)
        profile_stats = get_cvd_statistics(train_dataset_temp, max_samples=None)  # None = usa tutto il dataset
        
        # Estrai theta_global_stats
        theta_global_stats = profile_stats.get('theta_global', {
            'mean': float(profile_stats['profile_mean'][0]),
            'std': float(profile_stats['profile_std'][0])
        })
        
        # Costruisci per_cvd_stats dict (solo C/S per-tipo)
        per_cvd_stats = {}
        for cvd_type in ['protan', 'deutan', 'tritan']:
            if cvd_type in profile_stats.get('per_cvd_type', {}):
                type_stats = profile_stats['per_cvd_type'][cvd_type]
                per_cvd_stats[cvd_type] = {
                    'C_mean': float(type_stats.get('C_mean', type_stats.get('mean', [0,0,0])[1])),
                    'C_std': float(type_stats.get('C_std', type_stats.get('std', [1,1,1])[1])),
                    'S_mean': float(type_stats.get('S_mean', type_stats.get('mean', [0,0,0])[2])),
                    'S_std': float(type_stats.get('S_std', type_stats.get('std', [1,1,1])[2])),
                    'count': type_stats.get('count', 1),
                    # Mantieni anche vecchio formato per retrocompatibilità
                    'mean': type_stats.get('mean', profile_stats['profile_mean']),
                    'std': type_stats.get('std', profile_stats['profile_std']),
                }
            else:
                # Fallback: usa statistiche globali se il tipo non è presente
                print(f"[WARNING] No samples found for {cvd_type}, using global stats as fallback")
                per_cvd_stats[cvd_type] = {
                    'C_mean': float(profile_stats['profile_mean'][1]),
                    'C_std': float(profile_stats['profile_std'][1]),
                    'S_mean': float(profile_stats['profile_mean'][2]),
                    'S_std': float(profile_stats['profile_std'][2]),
                    'count': 1,
                    'mean': profile_stats['profile_mean'],
                    'std': profile_stats['profile_std'],
                }
        
        # Salva statistiche IBRIDE per uso futuro
        try:
            np.savez(stats_cache_path, 
                     per_cvd_stats=per_cvd_stats,
                     theta_global_stats=theta_global_stats,  # Stats globali θ
                     global_mean=profile_stats['profile_mean'],
                     global_std=profile_stats['profile_std'])
            print(f"[CVD DataLoaders] HYBRID profile stats cached to {stats_cache_path}")
        except Exception as e:
            print(f"[CVD DataLoaders] Warning: Failed to save stats cache ({e})")
    # NOTE: else case è già gestito sopra nel blocco cache loading

    # SAFETY: theta_global_stats deve esistere anche quando per_cvd_stats è caricato da cache.
    # (evita crash in print e garantisce normalizzazione θ globale coerente)
    if theta_global_stats is None:
        # 1) prova a ricostruire da cache se disponibile
        try:
            if cached is not None and 'global_mean' in cached and 'global_std' in cached:
                theta_global_stats = {
                    'mean': float(cached['global_mean'][0]),
                    'std': float(cached['global_std'][0])
                }
        except Exception:
            theta_global_stats = None

    if theta_global_stats is None:
        # 2) fallback finale: usa le stats globali salvate dentro per_cvd_stats (retrocompatibilità)
        # scegli un tipo presente (tipicamente 'protan') e leggi mean/std[0]
        for _t in ['protan', 'deutan', 'tritan']:
            if per_cvd_stats is not None and _t in per_cvd_stats:
                m = per_cvd_stats[_t].get('mean', None)
                s = per_cvd_stats[_t].get('std', None)
                if m is not None and s is not None and len(m) >= 1 and len(s) >= 1:
                    theta_global_stats = {'mean': float(m[0]), 'std': float(s[0])}
                    break

    if theta_global_stats is None:
        raise RuntimeError(
            "[CVD DataLoaders] theta_global_stats is missing after cache/load and recompute. "
            "Delete profile_normalization_stats.npz to force full recompute."
        )
    
    print(f"[CVD DataLoaders] HYBRID NORMALIZATION - Profile stats:")
    print(f"   θ (GLOBAL): mean={theta_global_stats['mean']:.4f}, std={theta_global_stats['std']:.4f}")
    print(f"   C/S per-CVD type:")
    for cvd_type in ['protan', 'deutan', 'tritan']:
        if cvd_type in per_cvd_stats:
            stats = per_cvd_stats[cvd_type]
            C_mean = stats.get('C_mean', stats.get('mean', [0,0,0])[1])
            C_std = stats.get('C_std', stats.get('std', [1,1,1])[1])
            S_mean = stats.get('S_mean', stats.get('mean', [0,0,0])[2])
            S_std = stats.get('S_std', stats.get('std', [1,1,1])[2])
            print(f"      {cvd_type}: C={C_mean:.4f}±{C_std:.4f}, S={S_mean:.4f}±{S_std:.4f}")
    print(f"   => Output: 3D vector [θ_norm, C_norm, S_norm]")
    
    # PASSO 3: Crea dataset definitivi CON normalizzazione IBRIDA
    train_dataset = CVDMappingDataset(
        mapping_json_path=train_json,
        base_path=base_path,
        split="train",
        transform=transform,
        per_cvd_stats=per_cvd_stats,
        theta_global_stats=theta_global_stats
    )
    
    val_dataset = CVDMappingDataset(
        mapping_json_path=val_json,
        base_path=base_path,
        split="val", 
        transform=transform,
        per_cvd_stats=per_cvd_stats,
        theta_global_stats=theta_global_stats
    )
    
    # Crea test dataset se richiesto
    test_dataset = None
    if include_test and test_json.exists():
        test_dataset = CVDMappingDataset(
            mapping_json_path=test_json,
            base_path=base_path,
            split="test",
            transform=transform,
            per_cvd_stats=per_cvd_stats,
            theta_global_stats=theta_global_stats
        )
    
    # Crea DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        #drop_last=True  # Importante per batch consistency
        drop_last=(len(train_dataset) > batch_size * 2)  # Solo se abbiamo almeno 2 batch completi
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )
    
    # DEBUG CRITICAL: Check for empty loaders
    print(f"[CVD DataLoaders] Train dataset size: {len(train_dataset)}")
    print(f"[CVD DataLoaders] Val dataset size: {len(val_dataset)}")
    print(f"[CVD DataLoaders] Batch size: {batch_size}")
    print(f"[CVD DataLoaders] Train loader batches: {len(train_loader)}")
    print(f"[CVD DataLoaders] Val loader batches: {len(val_loader)}")
    
    if len(train_loader) == 0:
        print(f"[ERROR] Train loader is EMPTY! This will cause ZeroDivisionError!")
        print(f"[ERROR] Check: dataset size ({len(train_dataset)}) vs batch_size ({batch_size}) with drop_last=True")
        if len(train_dataset) < batch_size:
            print(f"[ERROR] Dataset too small for batch_size! Reduce batch_size to <= {len(train_dataset)}")
    
    if len(val_loader) == 0:
        print(f"[ERROR] Val loader is EMPTY!")
        if len(val_dataset) < batch_size:
            print(f"[ERROR] Val dataset too small for batch_size! Reduce batch_size to <= {len(val_dataset)}")
    
    # Crea test loader se richiesto
    test_loader = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False
        )
        
    
    print(f"[CVD DataLoaders] Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    if test_loader is not None:
        print(f"[CVD DataLoaders] Test: {len(test_loader)} batches")
    
    # Return condizionale
    if include_test and test_loader is not None:
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    else:
        return train_loader, val_loader, train_dataset, val_dataset


def get_cvd_statistics(dataset, max_samples=10000):
    """
    Analizza le statistiche dei profili CVD nel dataset.
    
    NORMALIZZAZIONE IBRIDA: 
    - θ (theta_deg): statistiche GLOBALI → preserva distinzione Protan/Deutan/Tritan
    - C_index, S_index: statistiche PER-CVD TYPE → gestisce outlier (es. S_index Tritan)
    
    Questa strategia permette al modello di distinguere i tipi CVD basandosi su θ_norm,
    senza bisogno di one-hot encoding.
    
    OTTIMIZZAZIONE: Estrae profili direttamente dal JSON mapping invece di caricare immagini!
    Questo riduce il tempo da ~3.75 ore a ~10-30 secondi per 270K samples.
    
    Args:
        dataset (CVDMappingDataset): Dataset da analizzare
        max_samples (int or None): Numero massimo di samples da analizzare 
                                   - Default 10000 per velocità
                                   - None = usa TUTTO il dataset (raccomandato per normalizzazione)
    
    Returns:
        dict: Statistiche sui profili CVD con struttura IBRIDA:
            {
                'theta_global': {'mean': float, 'std': float},  # θ GLOBALE
                'per_cvd_type': {       # C, S per-CVD TYPE
                    'protan': {'C_mean': float, 'C_std': float, 'S_mean': float, 'S_std': float},
                    'deutan': {...},
                    'tritan': {...}
                },
                'profile_mean': [...],  # Globale per riferimento
                'profile_std': [...],   # Globale per riferimento
                ...
            }
    """
    import math
    from cvd_constants import classify_cvd_type_from_theta
    
    profiles = []
    cvd_types = []
    severities = []
    
    # Per-CVD type tracking (solo per statistiche informative, non per normalizzazione)
    profiles_by_type = {'protan': [], 'deutan': [], 'tritan': []}
    
    # Gestisci max_samples=None (usa tutto il dataset)
    if max_samples is None:
        num_samples = len(dataset)
        print(f"[CVD Statistics] HYBRID: Computing stats on FULL dataset ({num_samples} samples)...")
    else:
        num_samples = min(len(dataset), max_samples)
        print(f"[CVD Statistics] HYBRID: Computing stats on {num_samples}/{len(dataset)} samples...")
    
    print(f"[CVD Statistics] HYBRID MODE: θ GLOBAL, C/S per-CVD type")
    print(f"[CVD Statistics] FAST MODE: Extracting profiles from JSON (no image loading)")
    
    # Progress bar usando tqdm se disponibile, altrimenti stampa ogni 1000 samples
    try:
        from tqdm import tqdm
        iterator = tqdm(range(num_samples), desc="Computing CVD stats")
    except ImportError:
        iterator = range(num_samples)
        print_interval = max(1, num_samples // 10)  # Stampa ogni 10%
    
    for i in iterator:
        # FAST PATH: Estrai profilo direttamente dal mapping JSON (no image loading!)
        item = dataset.mapping_data[i]
        
        # DEBUG: Print first item to see available fields
        if i == 0:
            print(f"\n[CVD Statistics DEBUG] First item keys: {list(item.keys())}")
            print(f"[CVD Statistics DEBUG] First item sample: {item}")
            print()
        
        # Estrai profilo 3D nativo (STESSO FORMATO di __getitem__)
        # Supporta sia formato originale (profile_x) che formato cache (profile_6d) che legacy (cvd_params.x_original)
        if "profile_x" in item and item["profile_x"] is not None:
            # Formato originale JSON: profile_x = {theta_deg, C_index, S_index}
            profile_x = item["profile_x"]
            theta_deg = float(profile_x.get("theta_deg", 0.0))
            C_index = float(profile_x.get("C_index", 0.0))
            S_index = float(profile_x.get("S_index", 0.0))
            profile_3d = [theta_deg, C_index, S_index]
        elif "cvd_params" in item and "x_original" in item.get("cvd_params", {}):
            # FALLBACK LEGACY: Estrai da cvd_params.x_original
            x_orig = item["cvd_params"]["x_original"]
            theta_deg = float(x_orig.get("theta_deg", 0.0))
            C_index = float(x_orig.get("C_index", 0.0))
            S_index = float(x_orig.get("S_index", 0.0))
            profile_3d = [theta_deg, C_index, S_index]
        elif "profile_6d" in item and item["profile_6d"] is not None:
            # Formato cache: profile_6d = [Rmaj, Rmin, dU, dV, S_index, C_index]
            p6d = item["profile_6d"]
            dU, dV = p6d[2], p6d[3]
            S_index = p6d[4]
            C_index = p6d[5]
            # Ricostruisci theta_deg da dU, dV
            theta_deg = math.degrees(math.atan2(dV, dU))
            profile_3d = [theta_deg, C_index, S_index]
        else:
            profile_3d = [0.0, 0.0, 0.0]
            theta_deg = 0.0
        
        # Converti a numpy array (no normalizzazione - vogliamo statistiche RAW)
        profile_3d_np = np.array(profile_3d, dtype=np.float32)
        profiles.append(profile_3d_np)
        
        # OPZIONE IBRIDA: Classifica CVD type per statistiche C/S per-tipo
        cvd_type_classified = classify_cvd_type_from_theta(theta_deg)
        profiles_by_type[cvd_type_classified].append(profile_3d_np)
        
        # Estrai metadata - supporta entrambi i formati
        if 'cvd_params' in item:
            cvd_type = item.get('cvd_params', {}).get('cvd_type', 'unknown')
        else:
            cvd_type = item.get('cvd_type', 'unknown')
        cvd_types.append(cvd_type)
        severities.append(theta_deg)
        
        # Stampa progresso se tqdm non disponibile
        if 'tqdm' not in sys.modules and i > 0 and i % print_interval == 0:
            print(f"[CVD Statistics] Progress: {i}/{num_samples} ({100*i/num_samples:.1f}%)")
    
    profiles = np.array(profiles)  # [N, 3]
    
    # =========================================================================
    # NORMALIZZAZIONE IBRIDA:
    # - θ (theta_deg): statistiche GLOBALI → preserva distinzione tra tipi CVD
    # - C_index, S_index: statistiche PER-CVD TYPE → gestisce outlier
    # =========================================================================
    
    # 1. Statistiche GLOBALI per θ (colonna 0)
    theta_global_mean = float(profiles[:, 0].mean())
    theta_global_std = float(profiles[:, 0].std())
    
    print(f"\n[CVD Statistics] HYBRID NORMALIZATION STATS:")
    print(f"  θ (GLOBAL): mean={theta_global_mean:.4f}°, std={theta_global_std:.4f}°")
    
    # 2. Statistiche PER-CVD TYPE per C e S (colonne 1 e 2)
    print(f"\n  C/S per-CVD type:")
    for cvd_type in ['protan', 'deutan', 'tritan']:
        type_profiles = np.array(profiles_by_type[cvd_type]) if profiles_by_type[cvd_type] else np.array([])
        if len(type_profiles) > 0:
            # θ per-tipo (solo per info, NON per normalizzazione)
            theta_mean = type_profiles[:, 0].mean()
            theta_std = type_profiles[:, 0].std()
            # C e S per-tipo (USATI per normalizzazione)
            C_mean = type_profiles[:, 1].mean()
            C_std = type_profiles[:, 1].std()
            S_mean = type_profiles[:, 2].mean()
            S_std = type_profiles[:, 2].std()
            print(f"    {cvd_type:6s}: n={len(type_profiles):5d}")
            print(f"             θ={theta_mean:.2f}±{theta_std:.2f}° (raw range)")
            print(f"             C={C_mean:.4f}±{C_std:.4f}, S={S_mean:.4f}±{S_std:.4f}")
        else:
            print(f"    {cvd_type:6s}: n=0 samples")
    
    # 3. Costruisci dict per_cvd_type con solo C/S stats (θ è globale)
    per_cvd_type = {}
    for cvd_type in ['protan', 'deutan', 'tritan']:
        type_profiles = np.array(profiles_by_type[cvd_type]) if profiles_by_type[cvd_type] else np.array([])
        if len(type_profiles) > 0:
            per_cvd_type[cvd_type] = {
                'C_mean': float(type_profiles[:, 1].mean()),
                'C_std': float(type_profiles[:, 1].std()),
                'S_mean': float(type_profiles[:, 2].mean()),
                'S_std': float(type_profiles[:, 2].std()),
                'count': len(type_profiles),
                # Mantieni anche vecchio formato per compatibilità con eventuale codice legacy
                'mean': np.array([theta_global_mean, type_profiles[:, 1].mean(), type_profiles[:, 2].mean()]),
                'std': np.array([theta_global_std, type_profiles[:, 1].std(), type_profiles[:, 2].std()]),
            }
        else:
            # Fallback: usa statistiche globali se non ci sono samples
            per_cvd_type[cvd_type] = {
                'C_mean': float(profiles[:, 1].mean()),
                'C_std': float(profiles[:, 1].std()),
                'S_mean': float(profiles[:, 2].mean()),
                'S_std': float(profiles[:, 2].std()),
                'count': 0,
                'mean': profiles.mean(axis=0),
                'std': profiles.std(axis=0),
            }
    
    # NORMALIZZAZIONE IBRIDA: Ritorna θ globale + C/S per-tipo
    stats = {
        # HYBRID: θ normalizzato globalmente
        'theta_global': {
            'mean': theta_global_mean,
            'std': theta_global_std
        },
        # HYBRID: C/S normalizzati per-tipo CVD
        'per_cvd_type': per_cvd_type,
        # Reference stats (solo per info/debug)
        'profile_mean': profiles.mean(axis=0),  # [3] - riferimento
        'profile_std': profiles.std(axis=0),    # [3] - riferimento
        'profile_min': profiles.min(axis=0),
        'profile_max': profiles.max(axis=0),
        'cvd_type_counts': {cvd_type: cvd_types.count(cvd_type) for cvd_type in set(cvd_types)},
        'severity_mean': np.mean(severities),
        'severity_std': np.std(severities),
        'severity_range': (min(severities), max(severities))
    }
    
    return stats



def build_profile6_from_item(it):
    """
    it: dict entry dal mapping JSON
    Restituisce: tensor torch.float32 shape (6,) => [Rmaj,Rmin,dU,dV,S,C]
    """
    # 1) Preferisci p_raw (se presente e 6D)
    if "p_raw" in it and it["p_raw"] is not None:
        p_raw = list(it["p_raw"])
        if len(p_raw) >= 6:
            Rmaj, Rmin, dU, dV, S, C = p_raw[:6]
            profile6 = [float(Rmaj), float(Rmin), float(dU), float(dV), float(S), float(C)]
            return torch.tensor(profile6, dtype=torch.float32)

    # 2) Fallback: p_norm (6D normalized)
    if "p_norm" in it and it["p_norm"] is not None:
        p_norm = list(it["p_norm"])
        if len(p_norm) >= 6:
            Rmaj_n, Rmin_n, dU_n, dV_n, S_n, C_n = p_norm[:6]
            profile6 = [float(Rmaj_n), float(Rmin_n), float(dU_n), float(dV_n), float(S_n), float(C_n)]
            return torch.tensor(profile6, dtype=torch.float32)

    # 3) Fallback: p_star (se 6D usa direttamente, se 7D rimuovi theta)
    if "p_star" in it and it["p_star"] is not None:
        p_star = list(it["p_star"])
        if len(p_star) == 6:
            # già 6D
            profile6 = [float(x) for x in p_star]
            return torch.tensor(profile6, dtype=torch.float32)
        elif len(p_star) == 7:
            # 7D: [Rmaj, Rmin, phi, dU, dV, S, C] -> rimuovi phi (indice 2)
            Rmaj, Rmin, phi, dU, dV, S, C = p_star
            profile6 = [float(Rmaj), float(Rmin), float(dU), float(dV), float(S), float(C)]
            return torch.tensor(profile6, dtype=torch.float32)

    # Final fallback: zeros
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)




if __name__ == "__main__":
    # Test del dataset loader
    dataset_path = "."  # Current directory
    
    try:
        # Test con include_test=True (ora è il default)
        results = create_cvd_dataloaders(
            dataset_base_path=dataset_path,
            batch_size=4,
            num_workers=0,  # Per debug
            include_test=True  # Usiamo il default, ora è True
        )
        
        # Unpacking condizionale in base al numero di valori restituiti
        if len(results) == 6:
            train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = results
            print(f"\n[TEST] Dataset creato con successo (con test dataset)!")
            print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds) if test_ds else 'N/A'}")
        else:
            train_loader, val_loader, train_ds, val_ds = results
            test_loader, test_ds = None, None
            print(f"\n[TEST] Dataset creato con successo (senza test dataset)!")
            print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        
        # Test di un batch
        print(f"\n[TEST] Testing first batch...")
        for cvd_batch, orig_batch, profiles_batch, metadata_batch in train_loader:
            print(f"CVD images shape: {cvd_batch.shape}")
            print(f"Original images shape: {orig_batch.shape}")
            print(f"Profiles shape: {profiles_batch.shape}")
            print(f"First profile: {profiles_batch[0]}")
            print(f"CVD types: {[meta['cvd_type'] for meta in metadata_batch]}")
            print(f"Severities: {[meta['severity'] for meta in metadata_batch]}")
            break
            
        # Statistiche sui profili
        print(f"\n[TEST] Computing dataset statistics...")
        stats = get_cvd_statistics(train_ds)
        print(f"Profile means: {stats['profile_mean']}")
        print(f"Profile stds: {stats['profile_std']}")
        print(f"CVD type distribution: {stats['cvd_type_counts']}")
        print(f"Severity: {stats['severity_mean']:.1f} ± {stats['severity_std']:.1f}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
