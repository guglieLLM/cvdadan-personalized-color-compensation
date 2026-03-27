"""
train_utility — Funzioni di utilità per il training loop.

Contiene:
    - Serializzazione checkpoint (save/load con auto-resume)
    - Denormalizzazione tensori ImageNet per visualizzazione
    - Info hardware (CPU/GPU)
    - Gradient clipping e debug gradienti decoder
    - :class:`ImagePreProcessedDataset` per cache pre-processata

Dipendenze:
    torch, torchvision.
"""
import torch
import os
import warnings
from datetime import datetime
from torchvision.datasets import ImageFolder
import torch.nn.utils


def denormalize_tensor(tensor):
    """
    Converte un tensore normalizzato (ImageNet) in [0,1] per visualizzazione --> DE-NORMALIZZIAMO IN USCITA DAL DECODER
    
    Gestisce automaticamente il trasferimento CPU/GPU per garantire compatibilità con matplotlib.
    
    Args:
        tensor: Tensore normalizzato [B, C, H, W] su GPU o CPU
        
    Returns:
        Tensore denormalizzato [B, C, H, W] sempre su CPU e pronto per visualizzazione
    """
    # Assicurati che il tensore sia su CPU e detached per visualizzazione
    if tensor.is_cuda:
        tensor_cpu = tensor.detach().cpu()
    else:
        tensor_cpu = tensor.detach()
    
    # Crea mean e std sempre su CPU
    mean = torch.tensor([0.485, 0.456, 0.406], device='cpu').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cpu').view(1, 3, 1, 1)
    
    # Denormalizza e clamp
    denormalized = (tensor_cpu * std + mean).clamp(0, 1)
    
    return denormalized


def tensor_to_cpu_safe(tensor):
    """
    Trasferisce in modo sicuro un tensore su CPU per visualizzazione/plotting.
    
    Args:
        tensor: Tensore PyTorch su GPU o CPU
        
    Returns:
        Tensore su CPU, detached dal grafo computazionale
    """
    if tensor is None:
        return None
        
    if tensor.is_cuda:
        return tensor.detach().cpu()
    else:
        return tensor.detach()


def save_checkpoint(model, path):
    """Salva il checkpoint del modello in modo sicuro con backup"""
    try:
        # Convert path to string if it's a Path object
        path_str = str(path)
        # Crea backup se esiste già un file
        if os.path.exists(path_str):
            backup_path = path_str + '.backup'
            os.replace(path_str, backup_path)
        
        # Salva il nuovo checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'save_time': datetime.now().isoformat()
        }, path_str)
        
        # Verifica che il file sia stato salvato correttamente
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Il file {path_str} non è stato creato")
            
        # Verifica che il checkpoint sia caricabile
        checkpoint = torch.load(path_str, weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint salvato non contiene model_state_dict")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Errore nel salvataggio del checkpoint: {str(e)}")
        # Ripristina il backup se presente
        if os.path.exists(backup_path):
            os.replace(backup_path, path_str)
        return False


class ImagePreProcessedDataset(ImageFolder):
    """
    Classe personalizzata per dataset autoencoder non supervisionato.

    Estende torchvision.datasets.ImageFolder ed è progettata per addestrare un autoencoder
    dove input e target coincidono.
    
    Usata nella Fase 1 della pipeline di color grading
    per daltonici, dove il compito consiste nella ricostruzione fedele di immagini RGB reali
    (senza etichette di classe).

    Durante l'inizializzazione, il dataset carica immagini da una directory strutturata
    in sottocartelle (formato compatibile con ImageFolder), come previsto per il dataset
    MIT Places365.

    Modifiche principali:
    - Le classi (etichette) sono ignorate.
    - Il metodo __getitem__ è sovrascritto per restituire solo l'immagine preprocessata.

    Vantaggi:
    - Nessuna ambiguità su input/target (sono identici)
    - Compatibile con altri dataset strutturati in formato ImageFolder
    - Pulizia semantica e flessibilità per futuri riutilizzi

    Returns:
        Tensor: immagine RGB preprocessata da usare sia come input che come target
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Debug: mostra il numero di file prima e dopo il filtraggio
        original_count = len(self.samples)
        
        # Filtra file realmente esistenti
        self.samples = [(p, l) for (p, l) in self.samples if os.path.exists(p)]
        
        filtered_count = len(self.samples)
        missing_files = original_count - filtered_count
        
        if missing_files > 0:
            print(f"[WARNING] Dataset {args[0] if args else 'unknown'}: {missing_files} file mancanti su {original_count} totali")
        
        print(f"[INFO] Dataset inizializzato: {filtered_count} immagini valide")
        
    
    def find_classes(self, directory):
        """
        Sovrascrive il comportamento di default di ImageFolder per identificare
        solo le classi (sottocartelle) valide nel dataset.

        In particolare, questa versione esclude automaticamente tutte le cartelle
        il cui nome contiene il carattere '~', che possono essere presenti su
        filesystem Windows come alias legacy (es. PLACES~1) e non sono categorie valide.

        Args:
            directory (str): percorso alla directory principale (es. train/ o val/)

        Returns:
            tuple:
                - classes (List[str]): nomi ordinati delle classi valide
                - class_to_idx (Dict[str, int]): mappatura classe -> indice numerico
        """

        classes =  [ 
                        d.name for d in os.scandir(directory)
                        if d.is_dir() and "~" not in d.name
                   ]
        
        classes.sort()
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        return classes, class_to_idx
    
    
    def __getitem__(self, index):
        """
        Restituisce una coppia (immagine, immagine) per l'autoencoder.
        """
        path, _ = self.samples[index]
        
        try:
            img, _ = super().__getitem__(index)
            return img, img  # input e target uguali
            
        except Exception as e:
            # Log dell'errore e rilancia (gestione normale del DataLoader)
            print(f"\n[ERROR] Errore caricamento immagine all'indice {index}: {path}")
            print(f"[ERROR] {type(e).__name__}: {str(e)}")
            raise
    
    def get_path(self, index):
        """
        Restituisce il percorso dell'immagine all'indice specificato.
        Utile per operazioni di copia/backup quando si creano subset del dataset.
        
        Args:
            index (int): Indice dell'immagine nel dataset
            
        Returns:
            str: Percorso assoluto del file immagine
        """
        return self.samples[index][0]  # samples è una lista di tuple (path, class)


def get_cpu_info():
    """
    Ottiene informazioni sulla CPU per ottimizzare l'utilizzo delle risorse.
    
    Returns:
        dict: Informazioni sulla CPU e configurazioni consigliate per dataloader
    """
    import multiprocessing
    
    # Numero totale di core CPU
    cpu_count = multiprocessing.cpu_count()
    
    # Calcolo workers ottimali per DataLoader
    optimal_workers = min(8, cpu_count // 2) if cpu_count > 4 else cpu_count
    
    # Determina se usare pin_memory (consigliato quando si usa GPU)
    pin_memory = torch.cuda.is_available()
    
    return {
        "cpu_count": cpu_count,
        "optimal_workers": optimal_workers,
        "pin_memory": pin_memory
    }
    
    
def get_gpu_info():
    """
    Ottiene informazioni sulla GPU e su CUDA per ottimizzare l'utilizzo delle risorse.
    
    Returns:
        dict: Informazioni sulla GPU, CUDA e configurazioni consigliate
    """

    info = {}
    
    # Disponibilità CUDA
    info["cuda_available"] = torch.cuda.is_available()
    
    if info["cuda_available"]:
        # Versione CUDA
        info["cuda_version"] = torch.version.cuda
        
        # Numero di GPU disponibili
        info["gpu_count"] = torch.cuda.device_count()
        
        # Nome della GPU principale
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # Memoria totale della GPU principale (in GB)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info["total_memory_gb"] = round(total_memory, 2)
        
        # Memoria attualmente allocata (in MB, dopo reset per accuratezza)
        torch.cuda.reset_peak_memory_stats()
        info["allocated_memory_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
        
        # Configurazioni consigliate
        # Batch size ottimale stimato (basato su memoria, regola per i tuoi modelli)
        info["optimal_batch_size"] = min(128, int(total_memory * 1024 / 512))  # Esempio: 512MB per sample
        
        # Usa pin_memory per DataLoader (sempre True se GPU disponibile)
        info["pin_memory"] = True
        
        # Suggerimento per mixed precision (AMP) per ottimizzare memoria e velocità
        info["use_mixed_precision"] = True  # Raccomandato per GPU moderne
        
    else:
        # Valori di fallback se GPU non disponibile
        info["cuda_version"] = "Non disponibile"
        info["gpu_count"] = 0
        info["gpu_name"] = "Nessuna GPU"
        info["total_memory_gb"] = 0
        info["allocated_memory_mb"] = 0
        info["optimal_batch_size"] = 0
        info["pin_memory"] = False
        info["use_mixed_precision"] = False
    
    return info


def debug_decoder_gradients(decoder, batch_num, component_name="decoder"):
    """
    Debug function robusta per controllare i gradienti di qualunque decoder.
    Compatibile con diverse architetture di decoder.
    """
    print(f"[GRADIENT-DEBUG Batch {batch_num}] Controllo gradienti {component_name}:")
    
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            grad_mean = param.grad.mean().item()
            grad_norm = param.grad.norm().item()
            grad_has_nan = torch.isnan(param.grad).any().item()
            grad_has_inf = torch.isinf(param.grad).any().item()
            
            print(f"  {name}: norm={grad_norm:.6f}, min={grad_min:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}, nan={grad_has_nan}, inf={grad_has_inf}")
        else:
            print(f"  {name}: No gradient available")


def check_specific_corrupted_file(file_path):
    """
    Controlla se uno specifico file è corrotto e fornisce dettagli diagnostici.
    
    Args:
        file_path (str): Percorso al file da controllare
    """
    import os
    from PIL import Image
    
    print(f"\n=== DIAGNOSTIC per {file_path} ===")
    
    # 1. Verifica esistenza
    if not os.path.exists(file_path):
        print(" File non esiste")
        return False
    
    print(" File esiste")
    
    # 2. Verifica dimensione
    file_size = os.path.getsize(file_path)
    print(f" Dimensione: {file_size} bytes")
    
    if file_size == 0:
        print(" File vuoto")
        return False
    
    # 3. Verifica permessi
    readable = os.access(file_path, os.R_OK)
    print(f" Leggibile: {readable}")
    
    if not readable:
        print(" File non leggibile")
        return False
    
    # 4. Prova ad aprire con PIL
    try:
        with Image.open(file_path) as img:
            print(f"  Formato: {img.format}")
            print(f"  Modalità: {img.mode}")
            print(f"  Dimensioni: {img.size}")
            
            # Prova a verificare l'integrità
            img.verify()
            print(" Verifica PIL passata")
            
        # Prova a caricare i dati reali
        with Image.open(file_path) as img:
            img.load()  # Carica effettivamente i pixel
            print(" Caricamento pixel riuscito")
            
        return True
        
    except Exception as e:
        print(f"[OK] Errore PIL: {type(e).__name__}: {str(e)}")
        return False


# ===== UTILITY FUNCTION PER EVITARE WARNING LR =====
def get_current_lr(scheduler, optimizer):
    """
    Ottiene il learning rate corrente evitando warning deprecati.
    Assicura che il valore restituito sia sempre un Python float.
    
    Args:
        scheduler: Il scheduler PyTorch
        optimizer: L'optimizer PyTorch
        
    Returns:
        float: Il learning rate corrente come Python float
    """
    try:
        # DEBUG: Log scheduler type and available attributes
        scheduler_type = type(scheduler).__name__
        
        # Per ReduceLROnPlateau e altri scheduler moderni
        if hasattr(scheduler, 'get_last_lr') and callable(getattr(scheduler, 'get_last_lr', None)):
            # Prova prima con _last_lr (più sicuro)
            if hasattr(scheduler, '_last_lr') and scheduler._last_lr:
                lr_val = scheduler._last_lr[0]
                # DEBUG: 
                # print(f"[DEBUG LR] Using _last_lr: {lr_val} (type: {type(lr_val)})")
                # Converte tensor 0-d in scalare se necessario
                return lr_val.item() if hasattr(lr_val, 'item') else lr_val
            
            # Se _last_lr è vuoto, prova get_last_lr con warning suppressed
            elif hasattr(scheduler, 'get_last_lr'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        last_lr_list = scheduler.get_last_lr()
                        if last_lr_list:  # Assicurati che la lista non sia vuota
                            lr_val = last_lr_list[0]
                            # DEBUG: 
                            # print(f"[DEBUG LR] Using get_last_lr(): {lr_val} (type: {type(lr_val)})")
                            # Converte tensor 0-d in scalare se necessario
                            return lr_val.item() if hasattr(lr_val, 'item') else lr_val
                        else:
                            # Lista vuota, vai al fallback optimizer
                            # print(f"[DEBUG LR] get_last_lr() returned empty list, using optimizer fallback")
                            pass
                    except Exception as e:
                        # print(f"[DEBUG LR] get_last_lr() failed: {e}, using optimizer fallback")
                        pass
        
        # Fallback sicuro all'optimizer (questo dovrebbe sempre funzionare)
        lr_val = optimizer.param_groups[0]['lr']
        # print(f"[DEBUG LR] Using optimizer fallback: {lr_val} (type: {type(lr_val)})")
        return lr_val.item() if hasattr(lr_val, 'item') else lr_val
        
    except Exception as e:
        # Se tutto fallisce, usa l'optimizer (ultima risorsa)
        print(f"[ERROR LR] All methods failed: {e}, using emergency optimizer fallback")
        lr_val = optimizer.param_groups[0]['lr']
        return lr_val.item() if hasattr(lr_val, 'item') else lr_val


def gradient_clipping(model, epoch, max_norm_initial=5.0, max_norm_later=3.0, switch_epoch=15):
    """
    Gradient clipping per CVDCompensationModelAdaIN con encoder e decoder.
    Applica il clipping su tutti i parametri trainabili del modello.
    
    Args:
        model: Il modello CVDCompensationModelAdaIN
        epoch: L'epoca corrente per determinare la soglia di clipping
        max_norm_initial: La norma massima per le epoche iniziali (default: 5.0)
        max_norm_later: La norma massima per le epoche successive (default: 3.0)
        switch_epoch: L'epoca alla quale cambiare da max_norm_initial a max_norm_later
    
    Returns:
        torch.Tensor: La norma dei gradienti prima del clipping
    """
    # Usa una soglia di clipping adeguata in base all'epoca
    max_norm = max_norm_initial if epoch <= switch_epoch else max_norm_later
    
    # Colleziona tutti i parametri trainabili del modello
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    try:
        # Applica gradient clipping standard di PyTorch
        return torch.nn.utils.clip_grad_norm_(trainable_params, max_norm, error_if_nonfinite=False)
    except Exception as e:
        print(f"[WARNING] Gradient clipping failed: {e}")
        device_ = next(model.parameters()).device
        return torch.tensor(0.0, device=device_, dtype=torch.float32)


# Alias for backward compatibility
phase_1_0_gradient_clipping = gradient_clipping


# ===== CHECKPOINT UTILITIES (GENERIC - NO PHASE SUFFIX) =====

def load_full_checkpoint(path, *, model, optimizer, scheduler, scaler=None, config):
    """
    Load full checkpoint with compatibility checks.
    
    Generic checkpoint loader that works across training phases.
    Supports cross-phase resume with compatibility warnings.
    
    NOTE: scaler parameter is kept for backward compatibility but is IGNORED.
    bf16 training does not use GradScaler (wide dynamic range prevents underflow).
    Old checkpoints with scaler_state_dict are safely ignored.
    
    Args:
        path: Path to checkpoint file
        model: Model instance to load state into
        optimizer: Optimizer instance
        scheduler: LR scheduler instance
        scaler: DEPRECATED - kept for backward compatibility, will be ignored
        config: Training configuration dict
        
    Returns:
        tuple: (resume_epoch, best_val_loss, best_delta_e00, patience_counter, 
                gradient_health, cvd_loss_weights, cvd_normalization_constants)
        Returns (None, None, None, None, None, None, None) on failure
    """
    import traceback
    
    try:
        print(f"[RESUME] Loading checkpoint from: {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Verify checkpoint structure
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'epoch']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            print(f"[WARNING] Checkpoint missing keys: {missing_keys}")
            
        # Check phase compatibility
        if 'phase' in checkpoint:
            checkpoint_phase = checkpoint['phase']
            if checkpoint_phase not in ['1_cvd', 'D', '1', 'cvd']:
                print(f"[WARNING] Checkpoint phase mismatch: {checkpoint_phase} not in ['1_cvd', 'D']")
                print("[WARNING] Proceeding anyway - ensure compatibility!")
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            try:
                model.load_state_dict(model_state, strict=False)
                print("[RESUME] [OK] Model state loaded")
            except Exception as e:
                print(f"[ERROR] Failed to load model state: {e}")
                return None, None, None, None, None, None, None
        
        # CONFIG FLAG: Check if we should force NEW config params (default: True)
        force_new_params = config.get('force_new_scheduler_params', True)
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("[RESUME] [OK] Optimizer state loaded")
                
                # If FORCE mode, override learning rate from NEW config
                if force_new_params and 'learning_rate' in config:
                    new_lr = config['learning_rate']
                    old_lr = optimizer.param_groups[0]['lr']
                    
                    # Apply new LR to all param groups
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    print(f"[RESUME FIX] [OK] Learning rate updated: {old_lr:.2e} -> {new_lr:.2e} (from new config)")
                
            except Exception as e:
                print(f"[WARNING] Failed to load optimizer state: {e}")
        
        # Load scheduler state with CONFIG-CONTROLLED behavior
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            try:
                old_scheduler_state = checkpoint['scheduler_state_dict']
                
                if force_new_params:
                    # FORCE MODE: Apply NEW config parameters, preserve only history
                    print(f"[RESUME MODE]  FORCE NEW SCHEDULER PARAMS (force_new_scheduler_params=True)")
                    print(f"  Old checkpoint: factor={old_scheduler_state.get('factor', 0.8)}, patience={old_scheduler_state.get('patience', 7)}, min_lr={old_scheduler_state.get('min_lrs', [1e-7])[0]}")
                    print(f"  New config:     factor={config.get('lr_factor', 0.7)}, patience={config.get('lr_patience', 15)}, min_lr={config.get('min_lr', 5e-06)}")
                    
                    # Preserve ONLY training history (not hyperparameters)
                    num_bad_epochs = old_scheduler_state.get('num_bad_epochs', 0)
                    best_loss = old_scheduler_state.get('best', float('inf'))
                    last_epoch = old_scheduler_state.get('last_epoch', -1)
                    
                    # Apply NEW hyperparameters from config
                    scheduler.factor = config.get('lr_factor', 0.7)
                    scheduler.patience = config.get('lr_patience', 15)
                    scheduler.min_lrs = [config.get('min_lr', 5e-06)] * len(optimizer.param_groups)
                    scheduler.num_bad_epochs = num_bad_epochs
                    scheduler.best = best_loss
                    scheduler.last_epoch = last_epoch
                    
                    print(f"[RESUME FIX] [OK] Scheduler updated: bad_epochs={num_bad_epochs}, best_loss={best_loss:.6f}")
                    print(f"[RESUME FIX] [OK] NEW hyperparameters applied from config YAML")
                else:
                    # LEGACY MODE: Load full scheduler state from checkpoint (old behavior)
                    print(f"[RESUME MODE] ℹ LEGACY MODE (force_new_scheduler_params=False)")
                    print(f"  Loading FULL scheduler state from checkpoint (including hyperparameters)")
                    scheduler.load_state_dict(old_scheduler_state)
                    print(f"[RESUME] [OK] Scheduler state loaded from checkpoint (factor={scheduler.factor}, patience={scheduler.patience})")
                    
            except Exception as e:
                print(f"[WARNING] Failed to load scheduler state: {e}")
        
        # NOTE: scaler_state_dict is IGNORED - bf16 doesn't use GradScaler
        # Old checkpoints with scaler_state_dict are safely skipped for backward compatibility
        if 'scaler_state_dict' in checkpoint:
            print("[RESUME] [SKIP] scaler_state_dict found but IGNORED (bf16 doesn't use GradScaler)")
        
        # Load RNG states for reproducibility
        if 'rng_state' in checkpoint:
            try:
                torch.set_rng_state(checkpoint['rng_state'])
                if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
                print("[RESUME] [OK] RNG states restored")
            except Exception as e:
                print(f"[WARNING] Failed to restore RNG states: {e}")
        
        # Extract training metadata
        resume_epoch = checkpoint.get('epoch', 0) + 1  # Next epoch to train
        
        # Look for best_val_loss first, fallback for compatibility
        best_val_loss = checkpoint.get('best_val_loss', None)
        if best_val_loss is None:
            # Backward compatibility: try old best_val_mse_ab key
            best_val_loss = checkpoint.get('best_val_mse_ab', float('inf'))
            if best_val_loss != float('inf'):
                print(f"[RESUME] Using legacy 'best_val_mse_ab' as best_val_loss: {best_val_loss}")
        
        # Load best_delta_e00 (SECONDARY metric - reporting only)
        best_delta_e00_resume = checkpoint.get('best_delta_e00', float('inf'))
        metric_name_resume = checkpoint.get('metric_name', 'Loss')
        
        patience_counter = checkpoint.get('patience_counter', 0)
        gradient_health = checkpoint.get('gradient_health', 'unknown')
        
        # RESTORE CVDLoss WEIGHTS (lambda) if present - with backward compatibility
        if 'cvd_loss_weights' in checkpoint:
            cvd_weights = checkpoint['cvd_loss_weights']
            # New names first, fallback to old alpha/beta/gamma
            lambda_mse_resume = cvd_weights.get('lambda_mse', cvd_weights.get('alpha', 1.0))
            lambda_delta_e_resume = cvd_weights.get('lambda_delta_e', cvd_weights.get('beta', 1.0))
            lambda_ssim_resume = cvd_weights.get('lambda_ssim', cvd_weights.get('gamma', 1.0))
            severity_resume = cvd_weights.get('severity', 1.0)
            
            print(f"[RESUME] [OK] CVDLoss weights restored from checkpoint:")
            print(f"  λ_mse (MSE a*b*):      {lambda_mse_resume:.3f}")
            print(f"  λ_delta_e (Delta-E):   {lambda_delta_e_resume:.3f}")
            print(f"  λ_ssim (MS-SSIM L*):   {lambda_ssim_resume:.3f}")
            print(f"  Severity:              {severity_resume:.3f}")
            
            # Return weights to be applied to criterion_cvd in training loop
            cvd_loss_weights_resume = {
                'lambda_mse': lambda_mse_resume,
                'lambda_delta_e': lambda_delta_e_resume,
                'lambda_ssim': lambda_ssim_resume,
                'severity': severity_resume
            }
        else:
            print(f"[RESUME] [WARNING] No CVDLoss weights found in checkpoint")
            print(f"[RESUME] Using default weights from config")
            cvd_loss_weights_resume = None
        
        # RESTORE M_init normalization constants if present
        cvd_normalization_resume = None
        if 'cvd_normalization_constants' in checkpoint:
            norm_consts = checkpoint['cvd_normalization_constants']
            cvd_normalization_resume = {
                'M_mse': norm_consts.get('M_mse', 1.0),
                'M_delta_e': norm_consts.get('M_delta_e', 1.0),
                'M_ssim': norm_consts.get('M_ssim', 1.0),
                'is_calibrated': norm_consts.get('is_calibrated', False)
            }
            print(f"[RESUME] [OK] M_init normalization constants restored:")
            print(f"  M_mse:     {cvd_normalization_resume['M_mse']:.6f}")
            print(f"  M_delta_e: {cvd_normalization_resume['M_delta_e']:.6f}")
            print(f"  M_ssim:    {cvd_normalization_resume['M_ssim']:.6f}")
        else:
            print(f"[RESUME] [WARNING] No M_init constants in checkpoint (will use JSON or recalibrate)")
        
        print(f"[RESUME] Checkpoint loaded successfully:")
        print(f"  - Resume from epoch: {resume_epoch}")
        print(f"  - Best val loss: {best_val_loss:.6f}")
        print(f"  - Best ΔE2000: {best_delta_e00_resume:.4f}" if best_delta_e00_resume != float('inf') else "  - Best ΔE2000: N/A")
        print(f"  - Primary metric: {metric_name_resume}")
        print(f"  - Patience counter: {patience_counter}")
        print(f"  - Gradient health: {gradient_health}")
        print(f"  - Logger will restore from CSV automatically")
        
        return resume_epoch, best_val_loss, best_delta_e00_resume, patience_counter, gradient_health, cvd_loss_weights_resume, cvd_normalization_resume
        
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint not found: {path}")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None


def rotate_checkpoints(ckpt_dir, keep_last_k, pattern="epoch_*.pth"):
    """
    Maintain only last k checkpoints to prevent storage overflow.
    
    Args:
        ckpt_dir: Path object - Directory containing checkpoints
        keep_last_k: int - Number of recent checkpoints to keep
        pattern: str - Glob pattern for checkpoint files
    """
    try:
        checkpoint_files = list(ckpt_dir.glob(pattern))
        if len(checkpoint_files) <= keep_last_k:
            return  # Nothing to rotate
            
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Delete old checkpoints
        for old_checkpoint in checkpoint_files[keep_last_k:]:
            print(f"[CLEANUP] Removing old checkpoint: {old_checkpoint.name}")
            old_checkpoint.unlink()
            
    except Exception as e:
        print(f"[WARNING] Checkpoint rotation failed: {e}")


# ===== SAFETY UTILITIES (GENERIC - NO PHASE SUFFIX) =====

def safe_float_convert(value, default_val=0.0):
    """
    Safely convert value to float with NaN/Inf handling.
    
    Args:
        value: Value to convert (tensor, int, float, or other)
        default_val: Default value if conversion fails
        
    Returns:
        float: Converted value or default_val
    """
    import math
    
    try:
        if value is None:
            return default_val
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result = float(value.item())
            else:
                print(f"[WARNING] Multi-element tensor in float conversion -> using {default_val}")
                return default_val
        elif isinstance(value, (int, float)):
            result = float(value)
        else:
            print(f"[WARNING] Unknown type in float conversion -> using {default_val}")
            return default_val
            
        if math.isnan(result) or math.isinf(result):
            print(f"[WARNING] NaN/Inf detected in float conversion -> using {default_val}")
            return default_val
        return result
    except (ValueError, TypeError):
        print(f"[WARNING] Failed to convert {value} to float, using {default_val}")
        return default_val


def check_loss_safety(loss_value, loss_name="loss"):
    """
    Check if loss value is safe (not NaN/Inf).
    
    Args:
        loss_value: Loss value to check (tensor or float)
        loss_name: Name for logging purposes
        
    Returns:
        bool: True if safe, False if NaN/Inf detected
    """
    import math
    
    if loss_value is None:
        print(f"[LOSS SAFETY] {loss_name} is None!")
        return False
    
    if torch.is_tensor(loss_value):
        if torch.isnan(loss_value).any() or torch.isinf(loss_value).any():
            print(f"[LOSS SAFETY] {loss_name} contains NaN/Inf: {loss_value}")
            return False
    else:
        if math.isnan(loss_value) or math.isinf(loss_value):
            print(f"[LOSS SAFETY] {loss_name} is NaN/Inf: {loss_value}")
            return False
    return True


def check_for_nan_inf(tensor, name="tensor"):
    """
    Check tensor for NaN/Inf with detailed logging.
    
    Args:
        tensor: Tensor to check
        name: Name for logging purposes
        
    Returns:
        bool: True if NaN/Inf found, False otherwise
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total_elements = tensor.numel()
        print(f"[NaN DETECTED] {name}: {nan_count}/{total_elements} NaN values")
        print(f"[NaN STATS] Min: {tensor[~torch.isnan(tensor)].min() if (~torch.isnan(tensor)).any() else 'ALL_NAN'}")
        print(f"[NaN STATS] Max: {tensor[~torch.isnan(tensor)].max() if (~torch.isnan(tensor)).any() else 'ALL_NAN'}")
        return True
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        total_elements = tensor.numel()
        print(f"[Inf DETECTED] {name}: {inf_count}/{total_elements} Inf values")
        return True
    return False


def load_encoder_only_from_checkpoint(model, checkpoint_path):
    """
    Load only ContentEncoder weights from checkpoint.
    
    Useful for initializing new training phases with pretrained encoder.
    
    Args:
        model: PerceptualLatentColorFlowAE model instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        bool: True if loading succeeded, False otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Filter only encoder parameters
            encoder_state_dict = {k: v for k, v in state_dict.items() if 'content_encoder' in k}
            
            if encoder_state_dict:
                # Verify architecture compatibility
                model_dict = model.state_dict()
                encoder_keys_in_model = [k for k in model_dict.keys() if 'content_encoder' in k]
                
                print(f"[DEBUG] Checkpoint has {len(encoder_state_dict)} ContentEncoder parameters")
                print(f"[DEBUG] Model expects {len(encoder_keys_in_model)} ContentEncoder parameters")
                
                # Load filtered parameters
                model_dict.update(encoder_state_dict)
                model.load_state_dict(model_dict, strict=False)
                
                # Verify parameters were loaded
                loaded_keys = set(encoder_state_dict.keys())
                expected_keys = set(encoder_keys_in_model)
                missing_keys = expected_keys - loaded_keys
                unexpected_keys = loaded_keys - expected_keys
                
                print(f"[INFO] [OK] ContentEncoder parameters loaded: {len(encoder_state_dict)} layers")
                if missing_keys:
                    print(f"[WARNING] Missing keys in checkpoint: {len(missing_keys)} parameters")
                    print(f"[WARNING] First 3 missing: {list(missing_keys)[:3]}")
                if unexpected_keys:
                    print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)} parameters")
                
                return True
            else:
                print(f"[ERROR] No ContentEncoder parameters found in checkpoint")
                print(f"[ERROR] Checkpoint keys: {list(state_dict.keys())[:5]}...")
                return False
        else:
            print(f"[ERROR] Invalid checkpoint format: missing 'model_state_dict'")
            return False
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")
        return False


def print_training_config(config, device, amp_dtype, model=None):
    """
    Print training configuration in minimal or verbose mode based on debug_verbose flag.
    
    Args:
        config (dict): Training configuration dictionary
        device (torch.device): Device being used
        amp_dtype (torch.dtype): AMP data type (bfloat16 or None for fp32)
        model (nn.Module, optional): Model for parameter counting
        
    Note: use_fp16 parameter removed - bf16 is the only mixed precision mode now.
    """
    debug_verbose = config.get('debug_verbose', False)
    
    # Determine precision string
    precision_str = "bf16 (bfloat16)" if amp_dtype is not None else "fp32 (float32)"
    use_amp = amp_dtype is not None
    
    if not debug_verbose:
        # Minimal mode: Only essential info
        print("\n" + "="*80)
        print(f"[CONFIG] Experiment: {config.get('experiment_name', 'unnamed')}")
        print(f"[CONFIG] Phase: {config.get('training_phase', 'unknown')} | Device: {device} | Precision: {precision_str}")
        print(f"[CONFIG] AMP Enabled: {use_amp} | GradScaler: NOT USED (bf16 doesn't need it)")
        print("="*80 + "\n")
        return
    
    # Verbose mode: Detailed configuration
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION - VERBOSE MODE")
    print("="*80)
    
    # Hardware info
    print(f"\n[HARDWARE]")
    print(f"  Device: {device}")
    print(f"  Precision: {precision_str}")
    print(f"  AMP Enabled: {use_amp}")
    print(f"  GradScaler: NOT USED (bf16 has wide dynamic range)")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  bf16 Support: {torch.cuda.is_bf16_supported()}")
    
    # Experiment info
    print(f"\n[EXPERIMENT]")
    print(f"  Name: {config.get('experiment_name', 'unnamed')}")
    print(f"  Phase: {config.get('training_phase', 'unknown')}")
    print(f"  Run ID: {config.get('run_id', 'N/A')}")
    
    # Model info
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[MODEL]")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Training hyperparameters
    print(f"\n[TRAINING]")
    print(f"  Epochs: {config.get('epochs', 'N/A')}")
    print(f"  Batch size: {config.get('batch_size', 'N/A')}")
    print(f"  Learning rate: {config.get('lr', 'N/A')}")
    print(f"  Weight decay: {config.get('weight_decay', 0.0)}")
    print(f"  Gradient clip: {config.get('gradient_clip_val', 'None')}")
    print(f"  Warmup epochs: {config.get('warmup_epochs', 0)}")
    
    # Optimizer
    optimizer_type = config.get('optimizer', 'AdamW')
    print(f"\n[OPTIMIZER]")
    print(f"  Type: {optimizer_type}")
    if optimizer_type == 'AdamW':
        print(f"  Beta1: {config.get('beta1', 0.9)}")
        print(f"  Beta2: {config.get('beta2', 0.999)}")
    
    # Scheduler
    print(f"\n[SCHEDULER]")
    print(f"  Type: {config.get('scheduler', 'CosineAnnealingLR')}")
    if config.get('scheduler') == 'CosineAnnealingLR':
        print(f"  T_max: {config.get('epochs', 'N/A')}")
        print(f"  Eta_min: {config.get('eta_min', 1e-6)}")
    
    # Loss configuration (phase-specific)
    training_phase = config.get('training_phase', '')
    if training_phase == '1_cvd':
        print(f"\n[LOSS - CVDLoss]")
        print(f"  lambda_mse_ab: {config.get('lambda_mse_ab', 'N/A')}")
        print(f"  lambda_delta_e: {config.get('lambda_delta_e', 'N/A')}")
        print(f"  lambda_ssim_l: {config.get('lambda_ssim_l', 'N/A')}")
        print(f"  CVD type: {config.get('cvd_type', 'N/A')}")
        print(f"  CVD severity: {config.get('cvd_severity', 'N/A')}")
    
    # Early stopping
    if config.get('early_stopping_patience'):
        print(f"\n[EARLY STOPPING]")
        print(f"  Patience: {config.get('early_stopping_patience')} epochs")
        print(f"  Min delta: {config.get('early_stopping_min_delta', 0.0)}")
    
    # Quality gates
    if config.get('enable_quality_gates', False):
        print(f"\n[QUALITY GATES]")
        print(f"  Min Delta-E improvement: {config.get('min_delta_e_improvement', 'N/A')}")
        print(f"  Max acceptable loss: {config.get('max_acceptable_loss', 'N/A')}")
        print(f"  Consecutive fails threshold: {config.get('quality_gate_patience', 'N/A')}")
    
    # Checkpointing
    print(f"\n[CHECKPOINTING]")
    print(f"  Save frequency: every {config.get('checkpoint_frequency', 1)} epoch(s)")
    print(f"  Keep last: {config.get('keep_last_n_checkpoints', 3)} checkpoint(s)")
    print(f"  Save best only: {config.get('save_best_only', False)}")
    
    # Dataset info
    if config.get('dataset_path'):
        print(f"\n[DATASET]")
        print(f"  Path: {config.get('dataset_path')}")
        print(f"  Train/Val split: {config.get('train_val_split', 'N/A')}")
        print(f"  Num workers: {config.get('num_workers', 'N/A')}")
    
    # Resume info
    if config.get('resume_from_checkpoint'):
        print(f"\n[RESUME]")
        print(f"  Checkpoint: {config.get('resume_from_checkpoint')}")
        print(f"  Resume training: {config.get('resume_training', False)}")
    
    print("\n" + "="*80 + "\n")
