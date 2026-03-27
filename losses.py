import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from datetime import datetime

# Setup del logger principale
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# NaN LOGGER SEPARATO - Raccoglie tutti i problemi NaN/Inf per ogni esperimento
# ════════════════════════════════════════════════════════════════════════════════
_nan_logger = None
_nan_log_file = None

def setup_nan_logger(experiment_name: str = None, log_dir: str = None):
    """
    Inizializza un logger separato per tracciare tutti i problemi NaN/Inf.
    Crea un file di log dedicato per ogni esperimento.
    
    Args:
        experiment_name: Nome dell'esperimento (default: timestamp)
        log_dir: Directory per i log (default: current dir)
    
    Returns:
        Logger configurato
    """
    global _nan_logger, _nan_log_file
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_dir is None:
        log_dir = os.getcwd()
    
    # Crea il logger dedicato
    _nan_logger = logging.getLogger(f"nan_tracker_{experiment_name}")
    _nan_logger.setLevel(logging.DEBUG)
    _nan_logger.handlers.clear()  # Rimuovi handler precedenti
    
    # File handler per il log NaN
    _nan_log_file = os.path.join(log_dir, f"nan_debug_{experiment_name}.log")
    file_handler = logging.FileHandler(_nan_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Formato dettagliato con timestamp
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    _nan_logger.addHandler(file_handler)
    
    # Header del file
    _nan_logger.info("=" * 80)
    _nan_logger.info(f"NaN/Inf Debug Log - Experiment: {experiment_name}")
    _nan_logger.info("=" * 80)
    _nan_logger.info("")
    
    logger.info(f"[NaN Tracker] Log file created: {_nan_log_file}")
    
    return _nan_logger

def get_nan_logger():
    """Restituisce il NaN logger (lo crea se non esiste)."""
    global _nan_logger
    if _nan_logger is None:
        setup_nan_logger()
    return _nan_logger

def log_nan_event(phase: str, batch_idx: int, component: str, details: dict):
    """
    Logga un evento NaN/Inf con tutti i dettagli.
    
    Args:
        phase: 'train' o 'val'
        batch_idx: Indice del batch
        component: Nome componente ('input', 'lab_conversion', 'mse_ab', 'delta_e', 'msssim', 'total')
        details: Dict con dettagli (range, nan_count, etc.)
    """
    nan_log = get_nan_logger()
    
    # Formatta il messaggio
    msg_parts = [f"[{phase.upper()}] Batch {batch_idx} | {component}"]
    for key, value in details.items():
        if isinstance(value, float):
            msg_parts.append(f"{key}={value:.6f}")
        else:
            msg_parts.append(f"{key}={value}")
    
    nan_log.warning(" | ".join(msg_parts))

# Importa la metrica delta_e per la valutazione colorimetrica
from delta_e_ciede2000_torch import delta_e_ciede2000_torch

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE LOSS (post-debug NaN fix)
# ════════════════════════════════════════════════════════════════════════════════
DEBUG_COMPONENT_ISOLATION = True   # Se True, stampa debug per ogni componente
DEBUG_DISABLE_DELTA_E = False      # Delta-E2000 ORA ATTIVO con versione stabile
DEBUG_DISABLE_MSSSIM = False       # MS-SSIM attivo  
DEBUG_DISABLE_MSE_AB = False       # MSE a*b* attivo
DEBUG_USE_RGB_INSTEAD_OF_LAB = False  # LAB attivo (gamut clip disabilitato in color_space_utils)
DEBUG_USE_SIMPLE_DELTA_E = False   # Non usato se DEBUG_DISABLE_DELTA_E=True


class CVDLoss(nn.Module):
    """Loss multi-componente per CVD compensation con calibrazione statica.

    Supporta fino a 3 componenti; nella configurazione finale del training
    (``config_01_no_delta_e``) ne sono attive **solo 2**:

    +-----------+--------+------+-------------------------------------+
    | Componente| Spazio | λ    | Note                                |
    +===========+========+======+=====================================+
    | MSE a*b*  | CIELAB | 0.7  | Errore crominanza percettiva        |
    +-----------+--------+------+-------------------------------------+
    | ΔE2000    | CIELAB | 0.0  | **OFF** — usata solo in validazione  |
    +-----------+--------+------+-------------------------------------+
    | MS-SSIM   | sRGB   | 0.3  | Struttura multi-scala               |
    +-----------+--------+------+-------------------------------------+

    Normalizzazione (Data-Driven Static, rif. Groenendijk WACV 2021):
        ``L_norm = L_raw / M_init``, dove ``M_init`` è la media delle loss
        raw sui primi ~200 campioni di training.

    Args:
        lambda_mse: Peso MSE a*b* (default 1.0, config 0.7).
        lambda_delta_e: Peso ΔE2000 (default 0.0 — disabilitata).
        lambda_ssim: Peso MS-SSIM (default 1.0, config 0.3).
        warmup_samples: Numero di campioni per calibrazione (~200).
        severity_dynamic_weighting: Peso dinamico per severità CVD (OFF).
        edge_aware_weighting: Peso spaziale su bordi Sobel (OFF).
    """
    
    def __init__(self, 
                 lambda_mse=1.0,              # Peso priorità MSE a*b* (default: neutro)
                 lambda_delta_e=0.0,          # Peso priorità Delta-E2000 (default: DISABLED - evita calcolo costoso)
                 lambda_ssim=1.0,             # Peso priorità MS-SSIM L* (default: neutro)
                 warmup_samples=200,          # Numero campioni per calibrazione M_init
                 severity_dynamic_weighting=False,  # B.2: disattiva per default
                 edge_aware_weighting=False,       # B.3: disattiva per default
                 profile_stats=None):              # MIGRAZIONE 3D: statistiche normalizzazione profili
        super().__init__()
        
        # Lambda: pesi priorità componenti (controllo manuale dell'utente)
        # NOTA: Questi NON sono per il bilanciamento delle scale (quello lo fa M_init),
        # ma per dare priorità relative alle componenti (es. λ_ssim=2.0 per enfatizzare struttura)
        self.lambda_mse = lambda_mse
        self.lambda_delta_e = lambda_delta_e
        self.lambda_ssim = lambda_ssim
        
        # Numero campioni per calibrazione
        self.warmup_samples = warmup_samples
        
        # Feature flags
        self.severity_dynamic_weighting = severity_dynamic_weighting
        self.edge_aware_weighting = edge_aware_weighting
        
        # MIGRAZIONE 3D: Statistiche per denormalizzazione profili
        # profile_stats = {'profile_mean': [3], 'profile_std': [3]} da cvd_dataset_loader
        # NOTA: Richiesto SOLO se severity_dynamic_weighting=True
        if severity_dynamic_weighting:
            if profile_stats is None or 'profile_mean' not in profile_stats or 'profile_std' not in profile_stats:
                raise RuntimeError(
                    "CVDLoss con severity_dynamic_weighting=True richiede 'profile_stats' con chiavi "
                    "'profile_mean' e 'profile_std' per denormalizzare C_index -> severity. "
                    "Calcolare le statistiche dal training dataset usando cvd_dataset_loader.get_cvd_statistics()."
                )
            
            # Validazione shape: profile_stats deve contenere [3] elementi per profili 3D
            profile_mean_tensor = torch.tensor(profile_stats['profile_mean'], dtype=torch.float32)
            profile_std_tensor = torch.tensor(profile_stats['profile_std'], dtype=torch.float32)
            
            if profile_mean_tensor.shape[0] != 3 or profile_std_tensor.shape[0] != 3:
                raise ValueError(
                    f"profile_stats deve contenere 3 elementi per profili 3D [theta, C, S]. "
                    f"Ricevuto: profile_mean shape={profile_mean_tensor.shape}, "
                    f"profile_std shape={profile_std_tensor.shape}"
                )
            
            self.register_buffer('profile_mean', profile_mean_tensor)
            self.register_buffer('profile_std', profile_std_tensor)
        else:
            # severity_dynamic_weighting=False: non serve denormalizzare, usa placeholder
            self.register_buffer('profile_mean', torch.zeros(3))
            self.register_buffer('profile_std', torch.ones(3))
        
        # ═══════════════════════════════════════════════════════════════════
        # STATIC STATISTICS NORMALIZATION (M_init buffers)
        # ═══════════════════════════════════════════════════════════════════
        # M_init = media delle loss raw calcolate durante calibrazione pre-training
        # Formula normalizzazione: L_norm = L_raw / M_init
        # IMPORTANTE: Chiamare calibrate() PRIMA del training!
        self.register_buffer('M_mse', torch.tensor(1.0))       # M_init per MSE a*b*
        self.register_buffer('M_delta_e', torch.tensor(1.0))   # M_init per Delta-E2000
        self.register_buffer('M_ssim', torch.tensor(1.0))      # M_init per MS-SSIM loss
        self.register_buffer('is_calibrated', torch.tensor(False))  # Flag di calibrazione
        
        # Trackers per i pesi correnti lambda (per checkpoint e logging)
        self.register_buffer('current_lambda_mse', torch.tensor(lambda_mse))
        self.register_buffer('current_lambda_delta_e', torch.tensor(lambda_delta_e))
        self.register_buffer('current_lambda_ssim', torch.tensor(lambda_ssim))
        self.register_buffer('last_severity', torch.tensor(0.0))
        
        # ===== NaN TRACKING =====
        self._batch_counter = 0  # Contatore batch per logging
        self._current_phase = 'train'  # 'train' o 'val'
        
        # ===== MS-SSIM FAILURE TRACKING =====
        self._msssim_failure_count = 0  # Contatore fallimenti consecutivi
        self._msssim_failure_warned = False  # Flag per warning una sola volta
        
        # ===== EDGE-AWARE WEIGHTING SETUP (B.3) =====
        # Sobel kernel per detección di bordi (usati solo se edge_aware_weighting=True)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        # Register come buffers (non parametri trainable, ma salvati in checkpoint)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Parametro controllo: quanto enfatizzare bordi
        self.edge_strength_factor = 2.0  # Range finale weight: [1.0, 3.0]
        
        # ===== END EDGE-AWARE SETUP =====
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # Pre-training: calcolo M_init
    # ═══════════════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def calibrate(self, dataloader, model, num_samples=None, force_shuffle_check=True, save_path=None):
        """
        Calibra le costanti M_init iterando su campioni del dataset.
        
        DEVE essere chiamato PRIMA del training!
        
        Args:
            dataloader: DataLoader con i dati di training
            model: Modello da usare per generare predizioni
            num_samples: Numero di campioni da usare (default: self.warmup_samples)
            force_shuffle_check: Se True, verifica che il dataloader abbia shuffle=True
                                 per evitare bias da campionamento ordinato
            save_path: Se specificato, salva le costanti M_init su file JSON
                       Es: "checkpoints/calibration_constants.json"
        
        Returns:
            dict con M_mse, M_delta_e, M_ssim calcolati
        
        Raises:
            RuntimeError: Se shuffle=False e force_shuffle_check=True
        
        Note:
            - Usa torch.no_grad() per non contaminare i gradienti
            - Itera su N campioni random (se shuffle=True) per rappresentatività
            - I valori M_init sono salvati come buffer persistenti
            - Se save_path è specificato, salva anche su file JSON per riproducibilità
        """
        if num_samples is None:
            num_samples = self.warmup_samples
        
        # ═══════════════════════════════════════════════════════════════════
        # PROTEZIONE BIAS CAMPIONAMENTO
        # ═══════════════════════════════════════════════════════════════════
        # Se il dataloader non ha shuffle=True, rischiamo di calibrare solo
        # sui primi N campioni (es. tutti Protanopia o tutti scuri)
        if force_shuffle_check:
            # Verifica se il dataloader ha un sampler con shuffle
            sampler = getattr(dataloader, 'sampler', None)
            is_shuffled = False
            
            if sampler is not None:
                # RandomSampler indica shuffle=True
                from torch.utils.data import RandomSampler
                is_shuffled = isinstance(sampler, RandomSampler)
            
            if not is_shuffled:
                logger.warning(
                    "[CVDLoss.calibrate] ATTENZIONE: Il dataloader non sembra avere shuffle=True! "
                    "Questo può causare bias nella calibrazione (es. primi N campioni tutti dello stesso tipo). "
                    "Passa force_shuffle_check=False per disabilitare questo controllo, "
                    "oppure usa un dataloader con shuffle=True."
                )
                # Non raise error, solo warning - l'utente può decidere
        
        logger.info(f"[CVDLoss.calibrate] Inizio calibrazione M_init con {num_samples} campioni...")
        
        # Accumulatori per le loss raw
        mse_sum = 0.0
        delta_e_sum = 0.0
        ssim_sum = 0.0
        count = 0
        
        # Salva lo stato del modello (eval mode)
        was_training = model.training
        model.eval()
        
        try:
            for batch in dataloader:
                if count >= num_samples:
                    break
                
                # Estrai dati dal batch (formato dataset CVD)
                if isinstance(batch, dict):
                    # Chiavi: 'source'/'normal', 'target', 'profile'
                    source = batch.get('source', batch.get('normal', batch.get('input')))
                    target = batch.get('target', batch.get('compensated', batch.get('output')))
                    profile = batch.get('profile', batch.get('cvd_profile'))
                elif isinstance(batch, (list, tuple)):
                    source, target, profile = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                else:
                    raise ValueError(f"Formato batch non supportato: {type(batch)}")
                
                # Sposta su device del modello
                device = next(model.parameters()).device
                source = source.to(device)
                target = target.to(device)
                if profile is not None:
                    profile = profile.to(device)
                
                # Forward pass - usa la stessa firma del training loop
                # CVDCompensationModelAdaIN: model(x, profile_feats=, epoch=, total_epochs=)
                if profile is not None:
                    model_output = model(source, profile_feats=profile, epoch=1, total_epochs=100)
                else:
                    model_output = model(source)
                
                # Il modello può restituire un dict o direttamente il tensore
                if isinstance(model_output, dict):
                    # 'rgb_output' è la chiave usata nel training loop
                    predicted = model_output.get('rgb_output',
                                model_output.get('output', 
                                model_output.get('predicted', 
                                model_output.get('recolored',
                                model_output.get('x_hat', None)))))
                    if predicted is None:
                        # Prova la prima chiave che contiene un tensore 4D
                        for key, val in model_output.items():
                            if isinstance(val, torch.Tensor) and val.dim() == 4:
                                predicted = val
                                logger.debug(f"[CVDLoss.calibrate] Usata chiave '{key}' per output")
                                break
                    if predicted is None:
                        raise ValueError(f"Impossibile trovare output tensore nel dict: {list(model_output.keys())}")
                else:
                    predicted = model_output
                
                # Verifica shape - deve essere [B, C, H, W]
                if predicted.dim() != 4 or predicted.shape[1] != 3:
                    raise ValueError(f"Output shape invalido: {predicted.shape}, expected [B, 3, H, W]")
                
                # Denormalizza target se in range ImageNet
                if target.min() < -0.5:  # ImageNet-normalized
                    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    target = target * imagenet_std + imagenet_mean  # -> [0, 1]
                
                # Converti in range [-1, 1] per rgb_to_lab con input_range='tanh'
                predicted_for_loss = predicted * 2.0 - 1.0  # [0,1] -> [-1,1]
                target_for_loss = target * 2.0 - 1.0        # [0,1] -> [-1,1]
                
                # Calcola le 3 loss raw
                # ── MSE a*b* ──
                pred_lab = self.rgb_to_lab(predicted_for_loss.float())
                target_lab = self.rgb_to_lab(target_for_loss.float())
                pred_ab = pred_lab[:, 1:3, :, :]
                target_ab = target_lab[:, 1:3, :, :]
                mse_loss = F.mse_loss(pred_ab, target_ab, reduction='mean')
                
                # ── Delta-E2000 ── (solo se contribuisce alla loss)
                # Se lambda_delta_e=0, non serve M_delta_e per normalizzazione
                if self.lambda_delta_e > 0:
                    delta_e_map = delta_e_ciede2000_torch(
                        pred_lab.permute(0, 2, 3, 1),
                        target_lab.permute(0, 2, 3, 1)
                    )
                    delta_e_loss = delta_e_map.mean()
                else:
                    delta_e_loss = torch.tensor(0.0, device=mse_loss.device)
                
                # ── MS-SSIM RGB ──
                # NOTA: compute_msssim_rgb si aspetta input in [-1, 1]
                msssim_loss, _ = self.compute_msssim_rgb(predicted_for_loss, target_for_loss)
                
                # Accumula solo se valori validi
                if not (torch.isnan(mse_loss) or torch.isinf(mse_loss)):
                    mse_sum += mse_loss.item()
                if self.lambda_delta_e > 0 and not (torch.isnan(delta_e_loss) or torch.isinf(delta_e_loss)):
                    delta_e_sum += delta_e_loss.item()
                if not (torch.isnan(msssim_loss) or torch.isinf(msssim_loss)):
                    ssim_sum += msssim_loss.item()
                
                count += source.shape[0]  # Conta i sample, non i batch
                
                if count % 50 == 0:
                    logger.info(f"[CVDLoss.calibrate] Processati {count}/{num_samples} campioni...")
        
        finally:
            # Ripristina lo stato del modello
            if was_training:
                model.train()
        
        # Calcola medie (M_init)
        n_batches = max(1, count // dataloader.batch_size)
        M_mse = mse_sum / n_batches if n_batches > 0 else 1.0
        # M_delta_e: usa 1.0 se lambda_delta_e == 0 (non usato nella loss)
        M_delta_e = delta_e_sum / n_batches if (n_batches > 0 and self.lambda_delta_e > 0) else 1.0
        M_ssim = ssim_sum / n_batches if n_batches > 0 else 1.0
        
        # Protezione da M=0 (causerebbe divisione per zero)
        M_mse = max(M_mse, 1e-6)
        M_delta_e = max(M_delta_e, 1e-6)
        M_ssim = max(M_ssim, 1e-6)
        
        # Salva nei buffer
        self.M_mse.copy_(torch.tensor(M_mse))
        self.M_delta_e.copy_(torch.tensor(M_delta_e))
        self.M_ssim.copy_(torch.tensor(M_ssim))
        self.is_calibrated.copy_(torch.tensor(True))
        
        logger.info(
            f"[CVDLoss.calibrate] Calibrazione completata!\n"
            f"  M_mse     = {M_mse:.6f}\n"
            f"  M_delta_e = {M_delta_e:.6f}\n"
            f"  M_ssim    = {M_ssim:.6f}\n"
            f"  Campioni usati: {count}"
        )
        
        # Risultato da restituire
        calibration_result = {
            'M_mse': M_mse,
            'M_delta_e': M_delta_e,
            'M_ssim': M_ssim,
            'samples_used': count,
            'lambda_mse': self.lambda_mse,
            'lambda_delta_e': self.lambda_delta_e,
            'lambda_ssim': self.lambda_ssim,
            'timestamp': datetime.now().isoformat()
        }
        
        # Salva su file JSON se richiesto
        if save_path is not None:
            import json
            # Crea directory se non esiste
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_result, f, indent=2)
            
            logger.info(f"[CVDLoss.calibrate] Costanti M_init salvate in: {save_path}")
        
        return calibration_result
    
    def get_normalization_constants(self):
        """Recupera le costanti M_init per checkpoint."""
        return {
            'M_mse': self.M_mse.item(),
            'M_delta_e': self.M_delta_e.item(),
            'M_ssim': self.M_ssim.item(),
            'is_calibrated': self.is_calibrated.item()
        }
    
    def set_normalization_constants(self, M_mse, M_delta_e, M_ssim):
        """Ripristina le costanti M_init da checkpoint."""
        self.M_mse.copy_(torch.tensor(M_mse))
        self.M_delta_e.copy_(torch.tensor(M_delta_e))
        self.M_ssim.copy_(torch.tensor(M_ssim))
        self.is_calibrated.copy_(torch.tensor(True))
        logger.info(
            f"[CVDLoss] Costanti M_init ripristinate da checkpoint: "
            f"M_mse={M_mse:.6f}, M_delta_e={M_delta_e:.6f}, M_ssim={M_ssim:.6f}"
        )
    
    def load_normalization_constants(self, json_path):
        """
        Carica le costanti M_init da un file JSON salvato con calibrate().
        
        Args:
            json_path: Percorso al file JSON con le costanti
            
        Returns:
            dict con i valori caricati
            
        Raises:
            FileNotFoundError: Se il file non esiste
            KeyError: Se il file non contiene le chiavi richieste
        """
        import json
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File di calibrazione non trovato: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verifica chiavi obbligatorie
        required_keys = ['M_mse', 'M_delta_e', 'M_ssim']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Chiave mancante nel file di calibrazione: {key}")
        
        # Imposta le costanti
        self.set_normalization_constants(
            M_mse=data['M_mse'],
            M_delta_e=data['M_delta_e'],
            M_ssim=data['M_ssim']
        )
        
        logger.info(
            f"[CVDLoss] Costanti M_init caricate da: {json_path}\n"
            f"  Calibrazione originale: {data.get('timestamp', 'N/A')}\n"
            f"  Campioni usati: {data.get('samples_used', 'N/A')}"
        )
        
        return data
        
    def rgb_to_lab(self, rgb_tensor):
        """
        Conversione RGB->LAB differenziabile usando implementazione centralizzata
        
        Usa rgb_to_lab_torch() da color_space_utils.py per evitare
        duplicazione e garantire consistenza.
        
        Input: rgb_tensor in range [-1, 1] con shape (B, 3, H, W)
        Output: lab_tensor in formato standard LAB con shape (B, 3, H, W)
        """
        from color_space_utils import rgb_to_lab_torch
        return rgb_to_lab_torch(rgb_tensor, input_range='tanh')

    def compute_edge_aware_weights(self, L_channel):
        """
        Compute spatial weight map basato su Sobel edge detection.
        
        INPUT:
            L_channel: torch.Tensor, shape [B, 1, H, W]
                      Canale L (luminanza) da LAB, range [0, 100]
        
        OUTPUT:
            weight_map: torch.Tensor, shape [B, 1, H, W]
                       Range [1.0, 3.0]
                       Bordi hanno weight > 1.0
                       Aree piatte hanno weight ≈ 1.0
        
        ALGORITMO:
        1. Applica Sobel X e Y convolutions
        2. Calcola magnitudine gradiente
        3. Normalizza a [0, 1]
        4. Scala a weight map [1.0, 1.0 + edge_strength_factor]
        """
        
        # Error handling: check NaN in L_channel
        if torch.isnan(L_channel).any():
            logger.warning("[CVDLoss] L_channel contains NaN, skipping edge-aware weighting")
            return torch.ones_like(L_channel)
        
        # Step 1: Applica Sobel kernels
        # F.conv2d(input, weight, padding)
        # input: [B, 1, H, W], weight: [1, 1, 3, 3]
        # output: [B, 1, H, W]
        
        L_grad_x = F.conv2d(L_channel, self.sobel_x, padding=1)  # [B, 1, H, W]
        L_grad_y = F.conv2d(L_channel, self.sobel_y, padding=1)  # [B, 1, H, W]
        
        # Step 2: Calcola magnitudine gradiente
        # ||∇L|| = √(∂L/∂x² + ∂L/∂y²)
        L_grad_mag = torch.sqrt(L_grad_x**2 + L_grad_y**2 + 1e-6)  # [B, 1, H, W]
        # +1e-6 per evitare sqrt(0) = NaN
        
        # Step 3: Normalizza magnitudine a [0, 1]
        # Usa min-max normalization per robustness
        B = L_grad_mag.size(0)
        L_grad_flat = L_grad_mag.view(B, -1)  # [B, H*W]
        
        L_grad_min = L_grad_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        L_grad_max = L_grad_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        
        # Error handling: check se gradiente è identicamente zero
        grad_range = L_grad_max - L_grad_min
        if (grad_range < 1e-6).any():
            logger.debug("[CVDLoss] Gradient is near zero, potential image-wide flatness")
            # Continue comunque, weight_map sarà uniforme ~1.0
        
        L_grad_norm = (L_grad_mag - L_grad_min) / (grad_range + 1e-6)
        # [B, 1, H, W] in range [0, 1]
        
        # Step 4: Scala a weight map [1.0, 1.0 + edge_strength_factor]
        # weight = 1.0 + 2.0 x normalized_gradient
        # Bordi (gradient ≈ 1.0) -> weight ≈ 3.0
        # Piatto (gradient ≈ 0.0) -> weight ≈ 1.0
        
        weight_map = 1.0 + self.edge_strength_factor * L_grad_norm
        
        return weight_map  # [B, 1, H, W]

    def compute_dynamic_weights(self, cvd_profile):
        """
        Calcola pesi alpha e beta dinamicamente in base alla severity del profilo CVD.
        
        INPUT:
            cvd_profile: torch.Tensor [B, 6] normalizzato
                        [Rmaj, Rmin, dU, dV, S, C]
                        Indice 5 = C = Confusion Index (range 0-2.5 clinicamente)
                        Dopo normalizzazione dataloader: range ~0-1
        
        OUTPUT:
            alpha: float, MSE weight (pesi MSE)
            beta: float, DeltaE00 weight
            
        LOGICA:
        ┌─────────────────────────────────────────────────────────────┐
        │ severity = C_index / 2.5  (denormalize, range 0-1)          │
        │                                                              │
        │ IF severity < 0.3:  # Mild anomalopia                        │
        │     alpha ranges [0.0 -> 0.1]                                │
        │     beta = 0.9                                              │
        │                                                              │
        │ ELIF 0.3 <= severity < 0.7:  # Moderate                     │
        │     alpha ranges [0.1 -> 0.2]  (linear interpolation)        │
        │     beta ranges [0.9 -> 0.8]  (linear interpolation)         │
        │                                                              │
        │ ELSE severity >= 0.7:  # Severe dichromia                   │
        │     alpha = 0.2                                             │
        │     beta = 0.8                                              │
        └─────────────────────────────────────────────────────────────┘
        """
        # Valida shape
        if cvd_profile.dim() != 2 or cvd_profile.shape[1] != 3:
            logger.warning(f"[CVDLoss] Expected [B, 3] profile, got {cvd_profile.shape}. Using default weights.")
            return self.lambda_mse, self.lambda_delta_e
        
        # Estrai C_index (INDICE 1 in profilo 3D)
        C_index_normalized = cvd_profile[:, 1]  # [B] - normalizzato (z-score)
        
        # Check NaN
        if torch.isnan(C_index_normalized).any():
            logger.warning("[CVDLoss] C_index contains NaN, using default weights")
            return self.lambda_mse, self.lambda_delta_e
        
        # DENORMALIZZAZIONE: z-score -> valore clinico [0, 11.5]
        # Formula inversa: x = z * σ + μ
        C_index_raw = C_index_normalized * self.profile_std[1] + self.profile_mean[1]  # [B]
        
        # Map C_index [0, 11.5] -> severity [0, 1]
        # C_index range clinico Vingrys & King-Smith: 0 (normal) - 11.5 (severe dichromat)
        severity = torch.clamp(C_index_raw / 11.5, 0.0, 1.0)  # [B]
        
        # Compute alpha e beta per ogni sample nel batch (vettorizzato)
        alpha = torch.zeros_like(severity)
        beta = torch.ones_like(severity)
        
        # Caso 1: severity < 0.3 (Mild)
        mask_mild = severity < 0.3
        alpha[mask_mild] = 0.1 * (severity[mask_mild] / 0.3)
        beta[mask_mild] = 0.9
        # Interpretazione: alpha varia [0 -> 0.1], beta fisso 0.9
        
        # Caso 2: 0.3 <= severity < 0.7 (Moderate)
        mask_moderate = (severity >= 0.3) & (severity < 0.7)
        t = (severity[mask_moderate] - 0.3) / 0.4  # parametro interpolazione [0, 1]
        alpha[mask_moderate] = 0.1 + 0.1 * t
        beta[mask_moderate] = 0.9 - 0.1 * t
        # Interpretazione: alpha varia [0.1 -> 0.2], beta varia [0.9 -> 0.8]
        
        # Caso 3: severity >= 0.7 (Severe)
        mask_severe = severity >= 0.7
        alpha[mask_severe] = 0.2
        beta[mask_severe] = 0.8
        # Interpretazione: alpha fisso 0.2, beta fisso 0.8 (baseline)
        
        # Ritorna media batch (per consistency con loss composita)
        alpha_mean = alpha.mean().item()
        beta_mean = beta.mean().item()
        
        return alpha_mean, beta_mean

    def compute_msssim_rgb(self, pred_rgb, target_rgb):
        """
        Calcola MS-SSIM direttamente sulle immagini RGB.
        
        MOTIVAZIONE TEORICA:
        Nel nostro approccio "delta", il modello predice Δa*b* mentre L* viene
        preservato dall'input. Calcolare MS-SSIM su L* non avrebbe senso perché
        L* output == L* input sempre (loss = 0 costante, nessun gradiente).
        
        MS-SSIM RGB invece misura:
        1. Qualità percettiva finale dell'immagine compensata
        2. Consistenza strutturale (bordi, texture, dettagli)
        3. Penalizza artefatti da gamut clipping (quando a*b* esce dal gamut RGB)
        4. Complementare a MSE a*b* (pixel-wise) e ΔE2000 (percettivo colore)
        
        VANTAGGI rispetto alla versione L*:
        - Misura qualità dell'output finale (ciò che l'utente vede)
        - Più veloce (elimina 2 conversioni RGB↔LAB)
        - Semanticamente corretto per il task di compensazione CVD
        
        Args:
            pred_rgb: (B, 3, H, W) in [-1, 1]
            target_rgb: (B, 3, H, W) in [-1, 1]
            
        Returns:
            msssim_loss: Scalar tensor (1 - MS-SSIM)
            msssim_value: Scalar tensor (raw MS-SSIM in [0,1] per logging)
        """
        try:
            from pytorch_msssim import ms_ssim, ssim
        except ImportError:
            raise ImportError(
                "pytorch-msssim non installato. Esegui: pip install pytorch-msssim"
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # FIX NaN: Forza float32 per evitare underflow/overflow in bf16/fp16
        # MS-SSIM è sensibile alla precisione numerica
        # ═══════════════════════════════════════════════════════════════════
        pred_rgb = pred_rgb.float()
        target_rgb = target_rgb.float()
        
        # ═══════════════════════════════════════════════════════════════════
        # NORMALIZZAZIONE: Da [-1, 1] a [0, 1] per MS-SSIM
        # MS-SSIM si aspetta valori in [0, 1] con data_range=1.0
        # ═══════════════════════════════════════════════════════════════════
        pred_rgb_01 = (pred_rgb + 1.0) / 2.0
        target_rgb_01 = (target_rgb + 1.0) / 2.0
        
        # Clamp per sicurezza (valori fuori range causano NaN in SSIM)
        pred_rgb_01 = torch.clamp(pred_rgb_01, 0.0, 1.0)
        target_rgb_01 = torch.clamp(target_rgb_01, 0.0, 1.0)
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK: Verifica input NaN
        # ═══════════════════════════════════════════════════════════════════
        if torch.isnan(pred_rgb_01).any() or torch.isnan(target_rgb_01).any():
            pred_nan_count = torch.isnan(pred_rgb_01).sum().item()
            target_nan_count = torch.isnan(target_rgb_01).sum().item()
            raise ValueError(
                f"[compute_msssim_rgb] NaN in input - "
                f"pred NaN count: {pred_nan_count}, target NaN count: {target_nan_count}"
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK: Verifica che ci sia varianza (evita div-by-zero in SSIM)
        # ═══════════════════════════════════════════════════════════════════
        pred_std = pred_rgb_01.std()
        target_std = target_rgb_01.std()
        if pred_std < 1e-6 or target_std < 1e-6:
            # Immagini costanti -> SSIM=1 per definizione, loss=0
            return torch.tensor(0.0, device=pred_rgb.device, dtype=torch.float32), \
                   torch.tensor(1.0, device=pred_rgb.device, dtype=torch.float32)
        
        # MS-SSIM richiede immagini ≥161x161 per 5 scale di default
        # Fallback a SSIM singola scala per immagini piccole
        _, _, H, W = pred_rgb_01.shape
        
        try:
            if H < 161 or W < 161:
                # SSIM singola scala (funziona da 11x11 in su)
                # win_size deve essere dispari e <= min(H,W)
                win_size = min(11, min(H, W))
                if win_size % 2 == 0:
                    win_size = max(3, win_size - 1)  # Rendi dispari
                
                msssim_val = ssim(
                    pred_rgb_01, target_rgb_01, 
                    data_range=1.0,        # RGB normalizzato a [0, 1]
                    size_average=True,     # Media spaziale
                    win_size=win_size,
                    channel=3              # 3 canali RGB
                )
            else:
                # MS-SSIM multi-scala completo
                msssim_val = ms_ssim(
                    pred_rgb_01, target_rgb_01, 
                    data_range=1.0,        # RGB normalizzato a [0, 1]
                    size_average=True      # Media spaziale
                )
            
            # ═══════════════════════════════════════════════════════════════════
            # FINAL CHECK: Verifica che il risultato non sia NaN
            # ═══════════════════════════════════════════════════════════════════
            if torch.isnan(msssim_val).any() or torch.isinf(msssim_val).any():
                raise ValueError(
                    f"[compute_msssim_rgb] MS-SSIM produced NaN/Inf - "
                    f"pred range: [{pred_rgb_01.min():.4f}, {pred_rgb_01.max():.4f}], "
                    f"target range: [{target_rgb_01.min():.4f}, {target_rgb_01.max():.4f}], "
                    f"pred std: {pred_std:.6f}, target std: {target_std:.6f}"
                )
                       
        except ValueError:
            # Re-raise ValueError (our explicit NaN exceptions)
            raise
        except Exception as e:
            # Wrap other exceptions with context
            raise ValueError(f"[compute_msssim_rgb] Exception in MS-SSIM calculation: {e}") from e
        
        # Converti a loss: MS-SSIM è similarità [0,1], vogliamo dissimilarità
        msssim_loss = 1.0 - msssim_val
        
        # msssim_value per logging (raw similarity)
        msssim_value = msssim_val.detach()
        
        return msssim_loss, msssim_value

    def set_phase(self, phase: str):
        """Imposta la fase corrente ('train' o 'val') per il logging."""
        self._current_phase = phase
        self._batch_counter = 0  # Reset counter per ogni fase
    
    def forward(self, predicted, target, cvd_profile):
        """
        Forward pass con 3 componenti: MSE a*b* + Delta-E2000 + MS-SSIM L*
        
        STATIC STATISTICS NORMALIZATION:
        1. Calcola loss raw per ogni componente
        2. Converti a loss (per MS-SSIM: 1 - MS-SSIM)
        3. Normalizza con divisione statica: L_norm = L_raw / M_init
        4. λ-weighted sum: L_tot = λ₁·L₁_norm + λ₂·L₂_norm + λ₃·L₃_norm
        
        MODALITÀ INPUT:
        - predicted/target: RGB [B,3,H,W] in [-1,1]
        - cvd_profile: (B, 3) [theta_deg, C_index, S_index]
        
        Returns:
            dict con 'loss' (scalar) + componenti raw/normalized per logging
        """
        try:
            # Incrementa batch counter
            self._batch_counter += 1
            batch_idx = self._batch_counter
            phase = getattr(self, '_current_phase', 'unknown')
            
            # ═══════════════════════════════════════════════════════════════════
            # CHECK INPUT: NaN/Inf nel predicted e target
            # ═══════════════════════════════════════════════════════════════════
            pred_has_nan = torch.isnan(predicted).any().item()
            pred_has_inf = torch.isinf(predicted).any().item()
            tgt_has_nan = torch.isnan(target).any().item()
            tgt_has_inf = torch.isinf(target).any().item()
            
            if pred_has_nan or pred_has_inf:
                pred_min = predicted[~torch.isnan(predicted)].min().item() if not pred_has_nan else float('nan')
                pred_max = predicted[~torch.isnan(predicted)].max().item() if not pred_has_nan else float('nan')
                log_nan_event(phase, batch_idx, 'INPUT_PREDICTED', {
                    'has_nan': pred_has_nan,
                    'has_inf': pred_has_inf,
                    'min': pred_min,
                    'max': pred_max,
                    'nan_count': torch.isnan(predicted).sum().item(),
                    'inf_count': torch.isinf(predicted).sum().item()
                })
            
            if tgt_has_nan or tgt_has_inf:
                log_nan_event(phase, batch_idx, 'INPUT_TARGET', {
                    'has_nan': tgt_has_nan,
                    'has_inf': tgt_has_inf,
                    'nan_count': torch.isnan(target).sum().item(),
                    'inf_count': torch.isinf(target).sum().item()
                })
            
            # ═══════════════════════════════════════════════════════════════════
            # PREPROCESSING: RGB -> LAB, FP32 protection
            # ═══════════════════════════════════════════════════════════════════
            original_dtype = predicted.dtype
            needs_fp32_protection = original_dtype in [torch.float16, torch.bfloat16]
            
            # DEBUG: Bypass LAB conversion per isolare il problema
            if DEBUG_USE_RGB_INSTEAD_OF_LAB:
                # Usa direttamente RGB invece di LAB (per test)
                if needs_fp32_protection:
                    pred_lab = predicted.float()  # Non è LAB, è RGB!
                    target_lab = target.float()
                    cvd_profile_fp32 = cvd_profile.float()
                else:
                    pred_lab = predicted
                    target_lab = target
                    cvd_profile_fp32 = cvd_profile
                if DEBUG_COMPONENT_ISOLATION:
                    logger.info("[CVDLoss DEBUG] BYPASS LAB: using RGB directly (DEBUG_USE_RGB_INSTEAD_OF_LAB=True)")
            else:
                # Converti a LAB (sempre necessario per tutte le componenti)
                if needs_fp32_protection:
                    pred_lab = self.rgb_to_lab(predicted.float())
                    target_lab = self.rgb_to_lab(target.float())
                    cvd_profile_fp32 = cvd_profile.float()
                else:
                    pred_lab = self.rgb_to_lab(predicted)
                    target_lab = self.rgb_to_lab(target)
                    cvd_profile_fp32 = cvd_profile
            
            # ═══════════════════════════════════════════════════════════════════
            # CHECK LAB: NaN/Inf dopo conversione
            # ═══════════════════════════════════════════════════════════════════
            pred_lab_nan = torch.isnan(pred_lab).any().item()
            pred_lab_inf = torch.isinf(pred_lab).any().item()
            tgt_lab_nan = torch.isnan(target_lab).any().item()
            tgt_lab_inf = torch.isinf(target_lab).any().item()
            
            if pred_lab_nan or pred_lab_inf:
                # Calcola range solo per valori validi
                valid_L = pred_lab[:,0][~torch.isnan(pred_lab[:,0]) & ~torch.isinf(pred_lab[:,0])]
                valid_a = pred_lab[:,1][~torch.isnan(pred_lab[:,1]) & ~torch.isinf(pred_lab[:,1])]
                valid_b = pred_lab[:,2][~torch.isnan(pred_lab[:,2]) & ~torch.isinf(pred_lab[:,2])]
                log_nan_event(phase, batch_idx, 'LAB_PRED', {
                    'has_nan': pred_lab_nan,
                    'has_inf': pred_lab_inf,
                    'nan_count': torch.isnan(pred_lab).sum().item(),
                    'L_range': f"[{valid_L.min().item():.2f}, {valid_L.max().item():.2f}]" if len(valid_L) > 0 else "ALL_NAN",
                    'a_range': f"[{valid_a.min().item():.2f}, {valid_a.max().item():.2f}]" if len(valid_a) > 0 else "ALL_NAN",
                    'b_range': f"[{valid_b.min().item():.2f}, {valid_b.max().item():.2f}]" if len(valid_b) > 0 else "ALL_NAN"
                })
            
            if tgt_lab_nan or tgt_lab_inf:
                log_nan_event(phase, batch_idx, 'LAB_TARGET', {
                    'has_nan': tgt_lab_nan,
                    'has_inf': tgt_lab_inf,
                    'nan_count': torch.isnan(target_lab).sum().item()
                })
            
            # ═══════════════════════════════════════════════════════════════════
            # COMPONENTE 1: MSE solo su a*b* (cromaticità)
            # ═══════════════════════════════════════════════════════════════════
            if DEBUG_DISABLE_MSE_AB:
                mse_ab_loss = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
                if DEBUG_COMPONENT_ISOLATION:
                    logger.info("[CVDLoss DEBUG] MSE a*b* DISABLED")
            else:
                pred_ab = pred_lab[:, 1:3, :, :]    # Solo a*b* [B, 2, H, W]
                target_ab = target_lab[:, 1:3, :, :]
                mse_ab_loss = F.mse_loss(pred_ab, target_ab, reduction='mean')
                
                # Check MSE NaN
                if torch.isnan(mse_ab_loss).item() or torch.isinf(mse_ab_loss).item():
                    log_nan_event(phase, batch_idx, 'COMP1_MSE_AB', {
                        'value': 'NaN' if torch.isnan(mse_ab_loss).item() else 'Inf',
                        'pred_ab_nan': torch.isnan(pred_ab).sum().item(),
                        'target_ab_nan': torch.isnan(target_ab).sum().item()
                    })
                elif DEBUG_COMPONENT_ISOLATION:
                    logger.info(f"[CVDLoss DEBUG] mse_ab_loss = {mse_ab_loss.item():.4f}")
            
            # ═══════════════════════════════════════════════════════════════════
            # COMPONENTE 2: Delta-E2000 su LAB completo (percezione colore)
            # ═══════════════════════════════════════════════════════════════════
            # IMPORTANTE: Delta-E2000 viene calcolato SEMPRE per validation/monitoring!
            # lambda_delta_e controlla SOLO se contribuisce alla loss totale.
            # Questo permette di usare Delta-E come metrica per early stopping
            # anche quando non è nella loss (lambda_delta_e=0).
            if DEBUG_DISABLE_DELTA_E:
                delta_e00_loss = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
                if DEBUG_COMPONENT_ISOLATION:
                    logger.info("[CVDLoss DEBUG] Delta-E2000 DISABLED")
            elif DEBUG_USE_SIMPLE_DELTA_E:
                # VERSIONE SEMPLIFICATA: sqrt(MSE LAB) ≈ Delta-E (Euclidean)
                # Numericamente stabile, concettualmente simile a Delta-E76
                # Formula: ΔE = sqrt(ΔL² + Δa² + Δb²) mediato su tutti i pixel
                diff_lab = pred_lab - target_lab  # [B, 3, H, W]
                delta_e_squared = (diff_lab ** 2).sum(dim=1)  # [B, H, W] - somma su L,a,b
                delta_e_map = torch.sqrt(delta_e_squared + 1e-8)  # [B, H, W] con eps per stabilità
                delta_e00_loss = delta_e_map.mean()  # scalar
                if DEBUG_COMPONENT_ISOLATION:
                    logger.info(f"[CVDLoss DEBUG] delta_e00_loss (SIMPLE) = {delta_e00_loss.item():.4f}")
            else:
                try:
                    # Delta-E2000 richiede (B, H, W, 3) invece di (B, 3, H, W)
                    delta_e00_map = delta_e_ciede2000_torch(
                        pred_lab.permute(0, 2, 3, 1),   # [B, H, W, 3]
                        target_lab.permute(0, 2, 3, 1)
                    )  # [B, H, W]
                    delta_e00_loss = delta_e00_map.mean()  # scalar
                    
                    # Check Delta-E NaN
                    de_has_nan = torch.isnan(delta_e00_map).any().item()
                    de_has_inf = torch.isinf(delta_e00_map).any().item()
                    if de_has_nan or de_has_inf:
                        nan_count = torch.isnan(delta_e00_map).sum().item()
                        inf_count = torch.isinf(delta_e00_map).sum().item()
                        total = delta_e00_map.numel()
                        # Trova valori validi per stats
                        valid_de = delta_e00_map[~torch.isnan(delta_e00_map) & ~torch.isinf(delta_e00_map)]
                        log_nan_event(phase, batch_idx, 'COMP2_DELTA_E', {
                            'nan_count': nan_count,
                            'inf_count': inf_count,
                            'total_pixels': total,
                            'nan_pct': f"{100*nan_count/total:.2f}%",
                            'valid_min': valid_de.min().item() if len(valid_de) > 0 else 'N/A',
                            'valid_max': valid_de.max().item() if len(valid_de) > 0 else 'N/A',
                            'valid_mean': valid_de.mean().item() if len(valid_de) > 0 else 'N/A'
                        })
                    elif DEBUG_COMPONENT_ISOLATION:
                        logger.info(f"[CVDLoss DEBUG] delta_e00_loss = {delta_e00_loss.item():.4f}")
                    
                except Exception as e:
                    log_nan_event(phase, batch_idx, 'COMP2_DELTA_E_ERROR', {'exception': str(e)})
                    delta_e00_loss = F.mse_loss(pred_lab, target_lab)
            
            # ═══════════════════════════════════════════════════════════════════
            # COMPONENTE 3: MS-SSIM su RGB (qualità strutturale immagine finale)
            # ═══════════════════════════════════════════════════════════════════
            if DEBUG_DISABLE_MSSSIM:
                msssim_rgb_loss = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
                msssim_rgb_value = torch.tensor(1.0, device=predicted.device, dtype=predicted.dtype)
                if DEBUG_COMPONENT_ISOLATION:
                    logger.info("[CVDLoss DEBUG] MS-SSIM DISABLED")
            else:
                # compute_msssim_rgb restituisce GIÀ la loss (1 - MS-SSIM)
                msssim_rgb_loss, msssim_rgb_value = self.compute_msssim_rgb(predicted, target)
                
                # Check MS-SSIM NaN
                if torch.isnan(msssim_rgb_loss).item() or torch.isinf(msssim_rgb_loss).item():
                    log_nan_event(phase, batch_idx, 'COMP3_MSSSIM', {
                        'loss_value': 'NaN' if torch.isnan(msssim_rgb_loss).item() else 'Inf',
                        'ssim_value': msssim_rgb_value.item() if not torch.isnan(msssim_rgb_value).item() else 'NaN'
                    })
                elif DEBUG_COMPONENT_ISOLATION:
                    logger.info(f"[CVDLoss DEBUG] msssim_rgb_loss = {msssim_rgb_loss.item():.4f}")
            
            # ═══════════════════════════════════════════════════════════════════
            # STATIC STATISTICS NORMALIZATION (sostituisce z-score)
            # ═══════════════════════════════════════════════════════════════════
            # Formula: L_norm = L_raw / M_init
            # M_init calcolato durante calibrate() pre-training
            # VANTAGGI: 
            #   - Loss SEMPRE >= 0 (no valori negativi)
            #   - Costanti FISSE (no instabilità running stats)
            #   - Scientificamente fondato (PINNs standard, Wang et al.)
            
            # Verifica che calibrate() sia stato chiamato
            if not self.is_calibrated.item():
                logger.warning(
                    "[CVDLoss] ATTENZIONE: calibrate() non è stato chiamato! "
                    "Le costanti M_init sono ai valori default (1.0). "
                    "Chiama loss.calibrate(dataloader, model) prima del training."
                )
            
            # Normalizzazione statica: L_norm = L_raw / M
            # Questo porta tutte le componenti sulla stessa scala (~1.0 quando L=M)
            mse_ab_norm = mse_ab_loss / self.M_mse
            delta_e00_norm = delta_e00_loss / self.M_delta_e
            msssim_rgb_norm = msssim_rgb_loss / self.M_ssim
            
            # Per logging (compatibilità con codice esistente)
            # Le "statistiche" sono ora le costanti M_init fisse
            mean_mse_ab = self.M_mse.item()
            std_mse_ab = 1.0  # Non usato nella divisione statica
            mean_delta_e00 = self.M_delta_e.item()
            std_delta_e00 = 1.0
            mean_msssim_rgb = self.M_ssim.item()
            std_msssim_rgb = 1.0
            
            # ═══════════════════════════════════════════════════════════════════
            # LAMBDA WEIGHTS (priorità componenti - controllo manuale)
            # ═══════════════════════════════════════════════════════════════════
            # I lambda NON servono per bilanciare le scale (quello lo fa M_init),
            # ma per dare priorità relative alle componenti secondo le esigenze dell'utente
            if self.severity_dynamic_weighting and cvd_profile is not None:
                try:
                    # Dynamic weights per MSE e Delta-E basati su severity CVD
                    lambda_mse_dyn, lambda_de_dyn = self.compute_dynamic_weights(cvd_profile_fp32)
                    lambda_ssim_dyn = self.lambda_ssim
                    
                    lambda_mse = torch.tensor(lambda_mse_dyn, device=pred_lab.device)
                    lambda_de = torch.tensor(lambda_de_dyn, device=pred_lab.device)
                    lambda_ssim = torch.tensor(lambda_ssim_dyn, device=pred_lab.device)
                except Exception as e:
                    logger.warning(f"[CVDLoss] Error computing dynamic weights: {e}, using static lambda")
                    lambda_mse = torch.tensor(self.lambda_mse, device=pred_lab.device)
                    lambda_de = torch.tensor(self.lambda_delta_e, device=pred_lab.device)
                    lambda_ssim = torch.tensor(self.lambda_ssim, device=pred_lab.device)
            else:
                # Lambda statici (default)
                lambda_mse = torch.tensor(self.lambda_mse, device=pred_lab.device)
                lambda_de = torch.tensor(self.lambda_delta_e, device=pred_lab.device)
                lambda_ssim = torch.tensor(self.lambda_ssim, device=pred_lab.device)
            
            # ═══════════════════════════════════════════════════════════════════
            # B.3: EDGE-AWARE WEIGHTING (opzionale, default=False)
            # ═══════════════════════════════════════════════════════════════════
            if self.edge_aware_weighting:
                logger.warning("[CVDLoss] edge_aware_weighting non ancora implementato nella versione tricomponente")
            
            # ═══════════════════════════════════════════════════════════════════
            # LOSS FINALE: λ-weighted sum delle componenti normalizzate
            # ═══════════════════════════════════════════════════════════════════
            # Formula: L_tot = λ₁·(L₁/M₁) + λ₂·(L₂/M₂) + λ₃·(L₃/M₃)
            # Se lambda_delta_e == 0, esclude ΔE dalla loss (ma resta calcolato per logging)
            # Con M_init correttamente calibrato, ogni L/M ≈ 1.0 quando L=M
            # I lambda controllano le priorità relative
            
            # Calcola loss totale: sempre MSE + SSIM, aggiungi Delta-E solo se lambda > 0
            total_loss = lambda_mse * mse_ab_norm + lambda_ssim * msssim_rgb_norm
            if lambda_de.item() > 0:
                total_loss = total_loss + lambda_de * delta_e00_norm
            
            # Con Static Normalization, la loss è SEMPRE >= 0 (no più blocco negative_loss)
            # Verifica di sicurezza solo per casi estremi
            if total_loss < 0:
                logger.error(f"[CVDLoss] UNEXPECTED: Loss negativa ({total_loss.item():.6f}) con Static Normalization!")
            
            # NaN PROTECTION: Se la loss è NaN/Inf, raise exception
            # Training loop will catch, skip batch, and log properly
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise ValueError(
                    f"[CVDLoss] NaN/Inf in total_loss - "
                    f"mse_ab: {mse_ab_loss.item() if not torch.isnan(mse_ab_loss).item() else 'NaN'}, "
                    f"delta_e00: {delta_e00_loss.item() if not torch.isnan(delta_e00_loss).item() else 'NaN'}, "
                    f"msssim_rgb: {msssim_rgb_loss.item() if not torch.isnan(msssim_rgb_loss).item() else 'NaN'}"
                )
            
            # Aggiorna buffer lambda per checkpoint (sempre in FP32)
            self.current_lambda_mse.copy_(lambda_mse.detach().float())
            self.current_lambda_delta_e.copy_(lambda_de.detach().float())
            self.current_lambda_ssim.copy_(lambda_ssim.detach().float())
            
            # Calcola severity per monitoring (se profile disponibile)
            # Profilo 3D: [theta_deg, C_index, S_index]
            if cvd_profile is not None and cvd_profile_fp32.shape[1] >= 1:
                # Usa theta_deg per calcolare la severity (indice di direzione confusione)
                theta_deg = cvd_profile_fp32[:, 0]  # Angolo in gradi
                # Severity basata su theta: più lontano da 0°, più severa la CVD
                severity = torch.abs(theta_deg).mean() / 90.0  # Normalizzato 0-1
                self.last_severity.copy_(severity.detach().float())
            else:
                severity = torch.tensor(0.0)
            
            # ═══════════════════════════════════════════════════════════════════
            # CHECK TOTAL LOSS: NaN/Inf finale
            # ═══════════════════════════════════════════════════════════════════
            if torch.isnan(total_loss).item() or torch.isinf(total_loss).item():
                log_nan_event(phase, batch_idx, 'TOTAL_LOSS', {
                    'value': 'NaN' if torch.isnan(total_loss).item() else 'Inf',
                    'mse_ab': mse_ab_loss.item() if not torch.isnan(mse_ab_loss).item() else 'NaN',
                    'delta_e00': delta_e00_loss.item() if not torch.isnan(delta_e00_loss).item() else 'NaN',
                    'msssim_rgb': msssim_rgb_loss.item() if not torch.isnan(msssim_rgb_loss).item() else 'NaN',
                    'lambda_mse': lambda_mse.item(),
                    'lambda_delta_e': lambda_de.item(),
                    'lambda_ssim': lambda_ssim.item()
                })
            
            # ═══════════════════════════════════════════════════════════════════
            # RETURN DICT: Loss totale + componenti per logging
            # ═══════════════════════════════════════════════════════════════════
            return {
                'loss': total_loss,  # Scalar loss per backward
                
                # Componenti raw (pre-normalization)
                'mse_ab': mse_ab_loss.item(),
                'delta_e00': delta_e00_loss.item(),
                'msssim_rgb_loss': msssim_rgb_loss.item(),
                'msssim_rgb_value': msssim_rgb_value.item(),  # Raw MS-SSIM similarity [0,1]
                
                # Componenti normalized (post Static Normalization)
                'mse_ab_norm': mse_ab_norm.item(),
                'delta_e00_norm': delta_e00_norm.item(),
                'msssim_rgb_norm': msssim_rgb_norm.item(),
                
                # Lambda (priorità componenti)
                'weights': {
                    'lambda_mse': lambda_mse.item(),
                    'lambda_delta_e': lambda_de.item(),
                    'lambda_ssim': lambda_ssim.item()
                },
                
                # Costanti M_init (per monitoring/debug)
                'normalization_constants': {
                    'M_mse': self.M_mse.item(),
                    'M_delta_e': self.M_delta_e.item(),
                    'M_ssim': self.M_ssim.item(),
                    'is_calibrated': self.is_calibrated.item()
                },
                
                # Severity (per monitoring)
                'severity': severity.item() if isinstance(severity, torch.Tensor) else severity
            }
            
        except Exception as e:
            logger.error(f"[CVDLoss] Critical error in forward: {e}")
            # Fallback: MSE semplice per evitare crash
            fallback_loss = F.mse_loss(predicted, target)
            return {
                'loss': fallback_loss,
                'mse_ab': fallback_loss.item(),
                'delta_e00': 0.0,
                'msssim_L_loss': 0.0,
                'msssim_L_value': 1.0,
                'mse_ab_norm': fallback_loss.item(),
                'delta_e00_norm': 0.0,
                'msssim_L_norm': 0.0,
                'weights': {
                    'lambda_mse': self.lambda_mse, 
                    'lambda_delta_e': self.lambda_delta_e, 
                    'lambda_ssim': self.lambda_ssim
                },
                'normalization_constants': {
                    'M_mse': self.M_mse.item(),
                    'M_delta_e': self.M_delta_e.item(),
                    'M_ssim': self.M_ssim.item(),
                    'is_calibrated': self.is_calibrated.item()
                },
                'severity': 0.0,
                'error': str(e)
            }
    
    def get_current_weights(self):
        """Recupera i pesi lambda attuali dai buffer (per logging/debug)"""
        return {
            'lambda_mse': self.current_lambda_mse.item(),
            'lambda_delta_e': self.current_lambda_delta_e.item(),
            'lambda_ssim': self.current_lambda_ssim.item(),
            'severity': self.last_severity.item()
        }
    
    def set_weights_from_checkpoint(self, lambda_mse, lambda_delta_e, lambda_ssim=None, severity=None):
        """Ripristina i pesi lambda da checkpoint"""
        logger.info(
            f"[CVDLoss] Restoring lambda weights from checkpoint: "
            f"λ_mse={lambda_mse:.4f}, λ_delta_e={lambda_delta_e:.4f}, "
            f"λ_ssim={lambda_ssim if lambda_ssim else 'N/A'}"
        )
        
        # Aggiorna buffer
        self.current_lambda_mse.copy_(torch.tensor(lambda_mse))
        self.current_lambda_delta_e.copy_(torch.tensor(lambda_delta_e))
        if lambda_ssim is not None:
            self.current_lambda_ssim.copy_(torch.tensor(lambda_ssim))
        
        if severity is not None:
            self.last_severity.copy_(torch.tensor(severity))
        
        logger.info(
            f"[CVDLoss] Lambda weights restored: "
            f"λ_mse={self.lambda_mse:.2f}, λ_delta_e={self.lambda_delta_e:.2f}, λ_ssim={self.lambda_ssim:.2f}"
        )

