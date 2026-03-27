"""
SimpleTrainingLogger — Logger di training per CVDCompensationModelAdaIN.

Registra metriche epoca/batch, genera plot e confronti visivi.
Configurato per la loss a 2 componenti effettivamente usata nel
training finale (MSE a*b* + MS-SSIM), con ΔE00 come metrica di
validazione.

Dipendenze:
    train_utility (denormalize_tensor, tensor_to_cpu_safe),
    matplotlib, torchvision.
"""

from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt
import torch
import traceback
import torchvision.utils as vutils
from datetime import datetime
from train_utility import denormalize_tensor, tensor_to_cpu_safe


class SimpleTrainingLogger:
    """Logger CSV + plot per il training di CVDCompensationModelAdaIN.

    Metriche registrate:
        - Train loss: CVDLoss (MSE a*b* + MS-SSIM; ΔE2000 opzionale)
        - Validazione: ΔE00 (primaria), SSIM, PSNR
        - Norme dei gradienti
        - Stato early stopping
    """
    
    def __init__(self, config, train_ds=None, val_ds=None, model=None, resume_from_epoch=None):
        # Setup output directories based on config
        self.output_dir = Path(config["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config["log_file"]).parent.resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.debug_verbose = config.get('debug_verbose', False)
        print(f"[LOGGER] Inizializzazione CVD Training Logger")
        
        # Loss label for plotting
        self.train_loss_label = "CVD Loss (MSE a*b* + MS-SSIM)"
        self.plot_title = "CVD Compensation Training"
        
        # Descriptions for visual comparisons
        self.descriptions = {
            "input": "CVD Input\n(Distorted)",
            "target": "Original\n(Ground Truth)",
            "output": "CVD Corrected\n(Model Output)"
        }

        # File paths
        self.log_path = self.log_dir / f"log_{config['experiment_name']}.txt"
        self.csv_path = self.log_dir / f"metrics_{config['experiment_name']}.csv"
        self.batch_log_path = self.log_dir / f"batch_metrics_{config['experiment_name']}.csv"
        
        # Plots directory: use explicit config or default to log_dir/plots
        if "plots_dir" in config and config["plots_dir"]:
            self.plots_dir = Path(config["plots_dir"]).resolve()
        else:
            self.plots_dir = self.log_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Check if we're resuming from a checkpoint
        self.is_resuming = resume_from_epoch is not None and resume_from_epoch > 1
        
        if self.is_resuming:
            last_completed = resume_from_epoch - 1
            print(f"[LOGGER] Resume dal checkpoint")
            print(f"[LOGGER] Ultima epoca completata: {last_completed}")
            print(f"[LOGGER] Prossima epoca da eseguire: {resume_from_epoch}")
            print(f"[LOGGER] Caricamento log esistenti...")
            
            self._restore_state_from_csv()
            
            if self.log_path.exists():
                with open(self.log_path, "a", encoding='utf-8') as f:
                    f.write(f"\n=== RESUME FROM CHECKPOINT ===\n")
                    f.write(f"Resume time: {datetime.now().isoformat()}\n")
                    f.write(f"Last completed epoch: {last_completed}\n")
                    f.write(f"Resuming from epoch: {resume_from_epoch}\n\n")
                    f.flush()
        else:
            print(f"[LOGGER] Nuovo training, inizializzazione log da zero...")
            
            # Init batch-level CSV with headers
            with open(self.batch_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                # CVDLoss 3-component headers
                writer.writerow([
                    "epoch", "batch", "train_cvd_loss",
                    "mse_ab_component", "delta_e00_component", "msssim_L_component",
                    "grad_norm"
                ])

            # Collect initial metadata
            gpu_info = "gpu not available -> CPU only"
            if torch.cuda.is_available():
                try:
                    gpu_info = torch.cuda.get_device_name(0)
                except:
                    gpu_info = "gpu (unknown model)"
                    
            meta = {
                "experiment": config.get("experiment_name"),
                "start_time": datetime.now().isoformat(),
                "architecture": "CVDCompensationModelAdaIN (ConvNeXt-Tiny + CVDAdaIN)",
                "total_params_decoder": self._get_decoder_params(model),
                "num_train_images": len(train_ds) if train_ds is not None else None,
                "num_val_images": len(val_ds) if val_ds is not None else None,
                "batch_size": config.get("batch_size"),
                "gpu": gpu_info,
                "fraction_train": config.get("FRACTION_IMAGES_TRAIN", None),
                "fraction_val": config.get("FRACTION_IMAGES_VAL", None)
            }

            # Write initial metadata to log.txt
            with open(self.log_path, "w", encoding='utf-8') as f:
                f.write("Experiment metadata:\n")
                for k, v in meta.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            # Initialize internal buffers
            self._init_empty_buffers()

            # Initialize epoch-level CSV with headers
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # CVDLoss 3-component + validation metrics + early stopping
                writer.writerow([
                    "epoch", "train_loss",
                    "mse_ab_component", "delta_e00_component", "msssim_L_component",
                    "alpha_weight", "beta_weight", "gamma_weight",
                    "val_delta_e00", "val_msssim_L", "val_ssim", "val_psnr",
                    "grad_norm",
                    "lr_current", "wd_current", "grad_clip_threshold",
                    "ema_grad_norm", "vanish_streak", "scheduler_last_lr",
                    "best_epoch", "counter_no_improve", "patience",
                    "significant_improvement", "improvement_threshold"
                ])

    def _init_empty_buffers(self):
        """Initialize empty internal buffers for new training."""
        # Core metrics
        self.epochs = []
        self.train_losses = []
        self.grad_norms = []
        
        # CVDLoss 3-component
        self.mse_ab_components = []
        self.delta_e00_components = []
        self.msssim_L_components = []
        
        # Weights
        self.alpha_weights = []
        self.beta_weights = []
        self.gamma_weights = []
        
        # Validation metrics
        self.val_delta_e00s = []
        self.val_msssim_L = []
        self.val_ssims = []
        self.val_psnrs = []
        
        # Early stopping tracking
        self.best_epochs = []
        self.counters_no_improve = []
        self.patience_values = []
        self.significant_improvements = []
        self.improvement_thresholds = []

    def _get_decoder_params(self, model):
        """
        Get decoder parameters count, handling torch.compile() wrapped models.
        
        torch.compile() returns an OptimizedModule that doesn't support len() or bool checks.
        We need to access the original model via _orig_mod attribute.
        """
        if model is None:
            return None
        try:
            # Handle torch.compile() wrapped models
            if hasattr(model, '_orig_mod'):
                actual_model = model._orig_mod
            else:
                actual_model = model
            
            if hasattr(actual_model, 'decoder'):
                return sum(p.numel() for p in actual_model.decoder.parameters())
            return None
        except Exception:
            return None

    def _restore_state_from_csv(self):
        """Ripristina lo stato interno del logger dai file CSV esistenti."""
        self._init_empty_buffers()
        csv_row_count = 0
        
        def safe_float(value, default=None):
            if not value or value.strip() == "":
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        try:
            if self.csv_path.exists():
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        csv_row_count += 1
                        try:
                            epoch = int(row["epoch"])
                            
                            # Parse CVDLoss components
                            train_loss = safe_float(row.get("train_loss", ""))
                            mse_ab = safe_float(row.get("mse_ab_component", ""))
                            delta_e00 = safe_float(row.get("delta_e00_component", ""))
                            msssim_L = safe_float(row.get("msssim_L_component", ""))
                            
                            # Parse weights
                            alpha = safe_float(row.get("alpha_weight", ""))
                            beta = safe_float(row.get("beta_weight", ""))
                            gamma = safe_float(row.get("gamma_weight", ""))
                            
                            # Parse validation metrics
                            val_delta_e00 = safe_float(row.get("val_delta_e00", ""))
                            val_msssim_L = safe_float(row.get("val_msssim_L", ""))
                            val_ssim = safe_float(row.get("val_ssim", ""))
                            val_psnr = safe_float(row.get("val_psnr", ""))
                            
                            grad_norm = safe_float(row.get("grad_norm", ""))
                            
                            # Parse early stopping
                            best_epoch = row.get("best_epoch", "")
                            best_epoch = int(best_epoch) if best_epoch and best_epoch.strip() else None
                            counter_no_improve = row.get("counter_no_improve", "")
                            counter_no_improve = int(counter_no_improve) if counter_no_improve and counter_no_improve.strip() else 0
                            patience = row.get("patience", "")
                            patience = int(patience) if patience and patience.strip() else None
                            significant_improvement = row.get("significant_improvement", "")
                            significant_improvement = significant_improvement == "1" if significant_improvement else False
                            improvement_threshold = safe_float(row.get("improvement_threshold", ""))
                            
                            # Append all data
                            self.epochs.append(epoch)
                            self.train_losses.append(train_loss)
                            self.mse_ab_components.append(mse_ab)
                            self.delta_e00_components.append(delta_e00)
                            self.msssim_L_components.append(msssim_L)
                            self.alpha_weights.append(alpha)
                            self.beta_weights.append(beta)
                            self.gamma_weights.append(gamma)
                            self.val_delta_e00s.append(val_delta_e00)
                            self.val_msssim_L.append(val_msssim_L)
                            self.val_ssims.append(val_ssim)
                            self.val_psnrs.append(val_psnr)
                            self.grad_norms.append(grad_norm)
                            self.best_epochs.append(best_epoch)
                            self.counters_no_improve.append(counter_no_improve)
                            self.patience_values.append(patience)
                            self.significant_improvements.append(significant_improvement)
                            self.improvement_thresholds.append(improvement_threshold)
                            
                        except (ValueError, KeyError) as e:
                            print(f"[WARNING] Errore nel parsing della riga CSV {csv_row_count}: {e}")
                            continue
                            
                print(f"[LOGGER] CSV conteneva {csv_row_count} righe di dati")
                print(f"[LOGGER] Ripristinati {len(self.epochs)} record")
                self._verify_lists_synchronization(expected_length=csv_row_count)
            else:
                print(f"[LOGGER] CSV non trovato, inizializzazione da zero")
        except Exception as e:
            print(f"[WARNING] Errore nel ripristino stato da CSV: {e}")
            self._init_empty_buffers()

    def _verify_lists_synchronization(self, expected_length=None):
        """Verifica che tutte le liste interne abbiano la stessa lunghezza."""
        if expected_length is None:
            expected_length = len(self.epochs)
            
        lists_info = {
            "epochs": len(self.epochs),
            "train_losses": len(self.train_losses),
            "grad_norms": len(self.grad_norms),
            "mse_ab_components": len(self.mse_ab_components),
            "delta_e00_components": len(self.delta_e00_components),
            "msssim_L_components": len(self.msssim_L_components),
            "val_delta_e00s": len(self.val_delta_e00s),
            "val_msssim_L": len(self.val_msssim_L),
            "val_ssims": len(self.val_ssims),
            "val_psnrs": len(self.val_psnrs),
            "best_epochs": len(self.best_epochs),
            "counters_no_improve": len(self.counters_no_improve),
        }
        
        mismatched = {name: length for name, length in lists_info.items() if length != expected_length}
        
        if mismatched:
            print(f"[WARNING] Liste non sincronizzate:")
            for name, length in mismatched.items():
                print(f"  {name}: {length} (attesi: {expected_length})")
        else:
            print(f"[INFO] Tutte le liste sono sincronizzate ({expected_length} elementi)")

    def get_state(self):
        """Restituisce lo stato attuale del logger per salvarlo nel checkpoint."""
        return {
            "epochs": self.epochs.copy(),
            "train_losses": self.train_losses.copy(),
            "grad_norms": self.grad_norms.copy(),
            "mse_ab_components": self.mse_ab_components.copy(),
            "delta_e00_components": self.delta_e00_components.copy(),
            "msssim_L_components": self.msssim_L_components.copy(),
            "alpha_weights": self.alpha_weights.copy(),
            "beta_weights": self.beta_weights.copy(),
            "gamma_weights": self.gamma_weights.copy(),
            "val_delta_e00s": self.val_delta_e00s.copy(),
            "val_msssim_L": self.val_msssim_L.copy(),
            "val_ssims": self.val_ssims.copy(),
            "val_psnrs": self.val_psnrs.copy(),
            "best_epochs": self.best_epochs.copy(),
            "counters_no_improve": self.counters_no_improve.copy(),
            "patience_values": self.patience_values.copy(),
            "significant_improvements": self.significant_improvements.copy(),
            "improvement_thresholds": self.improvement_thresholds.copy(),
            "is_resuming": self.is_resuming,
        }

    def restore_state(self, state):
        """Ripristina lo stato del logger da un checkpoint."""
        if state is not None:
            self.epochs = state.get("epochs", [])
            self.train_losses = state.get("train_losses", [])
            self.grad_norms = state.get("grad_norms", [])
            self.mse_ab_components = state.get("mse_ab_components", [])
            self.delta_e00_components = state.get("delta_e00_components", [])
            self.msssim_L_components = state.get("msssim_L_components", [])
            self.alpha_weights = state.get("alpha_weights", [])
            self.beta_weights = state.get("beta_weights", [])
            self.gamma_weights = state.get("gamma_weights", [])
            self.val_delta_e00s = state.get("val_delta_e00s", [])
            self.val_msssim_L = state.get("val_msssim_L", [])
            self.val_ssims = state.get("val_ssims", [])
            self.val_psnrs = state.get("val_psnrs", [])
            self.best_epochs = state.get("best_epochs", [])
            self.counters_no_improve = state.get("counters_no_improve", [])
            self.patience_values = state.get("patience_values", [])
            self.significant_improvements = state.get("significant_improvements", [])
            self.improvement_thresholds = state.get("improvement_thresholds", [])
            
            print(f"[LOGGER] Ripristinato stato con {len(self.epochs)} record dal checkpoint")
            self._verify_lists_synchronization()

    def log_batch(self, epoch, batch_idx, loss_data, grad_norm):
        """
        Log batch-level data for CVDLoss 3-component.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            loss_data: Dict with 'total_loss', 'mse_ab', 'delta_e00', 'msssim_L_loss'
            grad_norm: Gradient norm
        """
        # Convert tensors to scalars
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().cpu().item()
        
        # Extract loss components
        if isinstance(loss_data, dict):
            train_cvd_loss = loss_data.get('total_loss', 0.0)
            mse_ab_component = loss_data.get('mse_ab', 0.0)
            delta_e00_component = loss_data.get('delta_e00', 0.0)
            # Accept both old (msssim_L_loss) and new (msssim_rgb_loss) key names
            msssim_L_component = loss_data.get('msssim_L_loss', loss_data.get('msssim_rgb_loss', 0.0))
        else:
            train_cvd_loss = loss_data
            mse_ab_component = 0.0
            delta_e00_component = 0.0
            msssim_L_component = 0.0
        
        # Convert tensors
        def to_scalar(val):
            if isinstance(val, torch.Tensor):
                return val.detach().cpu().item()
            return val
        
        train_cvd_loss = to_scalar(train_cvd_loss)
        mse_ab_component = to_scalar(mse_ab_component)
        delta_e00_component = to_scalar(delta_e00_component)
        msssim_L_component = to_scalar(msssim_L_component)
        
        def safe_format(val, fmt):
            try:
                if math.isnan(val) or math.isinf(val):
                    return "nan"
                return fmt.format(val)
            except (ValueError, TypeError):
                return "nan"
        
        with open(self.batch_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                batch_idx,
                safe_format(train_cvd_loss, "{:.6f}"),
                safe_format(mse_ab_component, "{:.6f}"),
                safe_format(delta_e00_component, "{:.6f}"),
                safe_format(msssim_L_component, "{:.6f}"),
                safe_format(grad_norm, "{:.4f}")
            ])
            f.flush()

    def log(self, epoch, train_loss, **kwargs):
        """
        Main logging method for CVD training.
        
        Args:
            epoch: Current epoch number
            train_loss: Total CVDLoss or dict with components
            
        Kwargs:
            val_delta_e00: Primary validation metric (Delta-E2000)
            val_msssim_L: MS-SSIM L* validation
            val_ssim: SSIM validation
            val_psnr: PSNR validation
            loss_components: Dict with mse_ab, delta_e00, msssim_L_loss, weights
            grad_norm: Gradient norm
            lr_current, wd_current, grad_clip_threshold: Optimizer params
            ema_grad_norm, vanish_streak, scheduler_last_lr: Training state
            best_epoch, counter_no_improve, patience: Early stopping
            significant_improvement, improvement_threshold: Early stopping flags
            input_tensor, target_tensor, output_tensor: For visual comparison
        """
        # Extract parameters
        grad_norm = kwargs.get('grad_norm', 0.0)
        input_tensor = kwargs.get('input_tensor', None)
        target_tensor = kwargs.get('target_tensor', None)
        output_tensor = kwargs.get('output_tensor', None)
        
        # Adaptive parameters
        lr_current = kwargs.get('lr_current', None)
        wd_current = kwargs.get('wd_current', None)
        grad_clip_threshold = kwargs.get('grad_clip_threshold', None)
        ema_grad_norm = kwargs.get('ema_grad_norm', None)
        vanish_streak = kwargs.get('vanish_streak', None)
        scheduler_last_lr = kwargs.get('scheduler_last_lr', None)
        
        # Early stopping
        best_epoch = kwargs.get('best_epoch', None)
        counter_no_improve = kwargs.get('counter_no_improve', 0)
        patience = kwargs.get('patience', None)
        significant_improvement = kwargs.get('significant_improvement', False)
        improvement_threshold = kwargs.get('improvement_threshold', None)
        
        # Validation metrics
        val_delta_e00 = kwargs.get('val_delta_e00', None)
        val_msssim_L = kwargs.get('val_msssim_L', None)
        val_ssim = kwargs.get('val_ssim', None)
        val_psnr = kwargs.get('val_psnr', None)
        loss_components = kwargs.get('loss_components', {})
        
        # Helper to convert tensors
        def to_scalar(val):
            if isinstance(val, torch.Tensor):
                return val.cpu().item()
            return val
        
        train_loss = to_scalar(train_loss)
        grad_norm = to_scalar(grad_norm)
        val_delta_e00 = to_scalar(val_delta_e00) if val_delta_e00 is not None else None
        val_msssim_L = to_scalar(val_msssim_L) if val_msssim_L is not None else None
        val_ssim = to_scalar(val_ssim) if val_ssim is not None else None
        val_psnr = to_scalar(val_psnr) if val_psnr is not None else None
        
        # Extract loss components
        if isinstance(train_loss, dict):
            mse_ab_component = train_loss.get('mse_ab', 0.0)
            delta_e00_component = train_loss.get('delta_e00', 0.0)
            # Accept both old (msssim_L_loss) and new (msssim_rgb_loss) key names
            msssim_L_component = train_loss.get('msssim_L_loss', train_loss.get('msssim_rgb_loss', 0.0))
            alpha_weight = train_loss.get('alpha_weight', 0.0)
            beta_weight = train_loss.get('beta_weight', 0.0)
            gamma_weight = train_loss.get('gamma_weight', 0.0)
            train_loss_scalar = train_loss.get('total_loss', 0.0)
        else:
            train_loss_scalar = train_loss
            mse_ab_component = loss_components.get('mse_ab', 0.0)
            delta_e00_component = loss_components.get('delta_e00', 0.0)
            # Accept both old (msssim_L_loss) and new (msssim_rgb_loss) key names
            msssim_L_component = loss_components.get('msssim_L_loss', loss_components.get('msssim_rgb_loss', 0.0))
            alpha_weight = loss_components.get('alpha_weight', 0.0)
            beta_weight = loss_components.get('beta_weight', 0.0)
            gamma_weight = loss_components.get('gamma_weight', 0.0)
        
        # Convert component tensors
        mse_ab_component = to_scalar(mse_ab_component)
        delta_e00_component = to_scalar(delta_e00_component)
        msssim_L_component = to_scalar(msssim_L_component)
        alpha_weight = to_scalar(alpha_weight)
        beta_weight = to_scalar(beta_weight)
        gamma_weight = to_scalar(gamma_weight)
        train_loss_scalar = to_scalar(train_loss_scalar)
        
        # Validate values
        if math.isnan(train_loss_scalar) or math.isinf(train_loss_scalar):
            print(f"[WARNING] LOGGER Epoch {epoch}: Train loss non valido ({train_loss_scalar})")
            train_loss_scalar = float('nan')
        
        if val_delta_e00 is not None and (math.isnan(val_delta_e00) or math.isinf(val_delta_e00)):
            print(f"[WARNING] LOGGER Epoch {epoch}: Val DeltaE00 non valido ({val_delta_e00})")
            val_delta_e00 = float('nan')
        
        # Update internal buffers
        self.epochs.append(epoch)
        self.train_losses.append(train_loss_scalar)
        self.mse_ab_components.append(mse_ab_component)
        self.delta_e00_components.append(delta_e00_component)
        self.msssim_L_components.append(msssim_L_component)
        self.alpha_weights.append(alpha_weight)
        self.beta_weights.append(beta_weight)
        self.gamma_weights.append(gamma_weight)
        self.grad_norms.append(grad_norm)
        self.val_delta_e00s.append(val_delta_e00)
        self.val_msssim_L.append(val_msssim_L)
        self.val_ssims.append(val_ssim)
        self.val_psnrs.append(val_psnr)
        self.best_epochs.append(best_epoch)
        self.counters_no_improve.append(counter_no_improve)
        self.patience_values.append(patience)
        self.significant_improvements.append(significant_improvement)
        self.improvement_thresholds.append(improvement_threshold)
        
        # Write to text log
        try:
            with open(self.log_path, "a", encoding='utf-8') as f:
                def safe_fmt(val, fmt):
                    try:
                        if val is None:
                            return "None"
                        if math.isnan(val) or math.isinf(val):
                            return "nan"
                        return fmt.format(val)
                    except (ValueError, TypeError):
                        return "nan"
                
                line = f"Epoch {epoch:03d} | Train Loss: {safe_fmt(train_loss_scalar, '{:.6f}')}"
                line += f" | MSE a*b*: {safe_fmt(mse_ab_component, '{:.6f}')} (α={safe_fmt(alpha_weight, '{:.3f}')})"
                line += f" | ΔE00: {safe_fmt(delta_e00_component, '{:.6f}')} (β={safe_fmt(beta_weight, '{:.3f}')})"
                line += f" | MS-SSIM L*: {safe_fmt(msssim_L_component, '{:.6f}')} (γ={safe_fmt(gamma_weight, '{:.3f}')})"
                line += f" | Val ΔE00: {safe_fmt(val_delta_e00, '{:.4f}')}"
                line += f" | Grad Norm: {safe_fmt(grad_norm, '{:.4f}')}"
                
                if lr_current is not None:
                    line += f" | LR: {safe_fmt(lr_current, '{:.2e}')}"
                if best_epoch is not None:
                    line += f" | Best: Epoch {best_epoch}"
                if patience is not None:
                    line += f" | Counter: {counter_no_improve}/{patience}"
                if significant_improvement:
                    line += f" |  SIGNIFICANT"
                    
                line += "\n"
                f.write(line)
                f.flush()
                
        except Exception as e:
            print(f"[ERROR] Epoch {epoch}: Errore scrittura TXT log: {e}")
            traceback.print_exc()
        
        # Write to CSV
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                def safe_csv(val, fmt, fallback="nan"):
                    try:
                        if val is None:
                            return ""
                        if math.isnan(val) or math.isinf(val):
                            return fallback
                        return fmt.format(val)
                    except (ValueError, TypeError):
                        return fallback
                
                row = [
                    epoch,
                    safe_csv(train_loss_scalar, "{:.6f}"),
                    safe_csv(mse_ab_component, "{:.6f}"),
                    safe_csv(delta_e00_component, "{:.6f}"),
                    safe_csv(msssim_L_component, "{:.6f}"),
                    safe_csv(alpha_weight, "{:.3f}"),
                    safe_csv(beta_weight, "{:.3f}"),
                    safe_csv(gamma_weight, "{:.3f}"),
                    safe_csv(val_delta_e00, "{:.4f}"),
                    safe_csv(val_msssim_L, "{:.4f}") if val_msssim_L is not None else "",
                    safe_csv(val_ssim, "{:.4f}"),
                    safe_csv(val_psnr, "{:.4f}"),
                    safe_csv(grad_norm, "{:.4f}"),
                    safe_csv(lr_current, "{:.6e}") if lr_current is not None else "",
                    safe_csv(wd_current, "{:.6e}") if wd_current is not None else "",
                    safe_csv(grad_clip_threshold, "{:.4f}") if grad_clip_threshold is not None else "",
                    safe_csv(ema_grad_norm, "{:.6f}") if ema_grad_norm is not None else "",
                    vanish_streak if vanish_streak is not None else "",
                    safe_csv(scheduler_last_lr, "{:.6e}") if scheduler_last_lr is not None else "",
                    best_epoch if best_epoch is not None else "",
                    counter_no_improve,
                    patience if patience is not None else "",
                    "1" if significant_improvement else "0",
                    safe_csv(improvement_threshold, "{:.4f}") if improvement_threshold is not None else ""
                ]
                
                writer.writerow(row)
                f.flush()
                
        except Exception as e:
            print(f"[ERROR] Epoch {epoch}: Errore scrittura CSV: {e}")
            traceback.print_exc()
        
        # Generate plots
        self._save_plots(epoch)
        
        # Save visual comparison if provided
        if input_tensor is not None and output_tensor is not None:
            self._save_visual_comparison(epoch, input_tensor, target_tensor, output_tensor)
            self._save_labeled_visual_comparison(epoch, input_tensor, target_tensor, output_tensor)

    def _save_plots(self, epoch):
        """Salva i grafici delle metriche CVD training."""
        try:
            epochs = [int(e) if isinstance(e, torch.Tensor) else e for e in self.epochs]
            
            def safe_convert_list(values_list, name):
                converted = []
                invalid_count = 0
                for v in values_list:
                    try:
                        if v is None:
                            converted.append(None)
                            continue
                        if isinstance(v, torch.Tensor):
                            v = v.cpu().item()
                        v_float = float(v)
                        if not (math.isnan(v_float) or math.isinf(v_float)):
                            converted.append(v_float)
                        else:
                            converted.append(None)
                            invalid_count += 1
                    except:
                        converted.append(None)
                        invalid_count += 1
                
                valid_values = [v for v in converted if v is not None]
                if self.debug_verbose and invalid_count > 0:
                    print(f"[DEBUG] {name}: {invalid_count} valori non validi su {len(converted)}")
                
                return converted, valid_values
            
            # Convert all metrics
            train_losses, valid_train = safe_convert_list(self.train_losses, "Train Loss")
            val_delta_e00s, valid_delta_e00 = safe_convert_list(self.val_delta_e00s, "Val ΔE00")
            mse_ab, valid_mse_ab = safe_convert_list(self.mse_ab_components, "MSE a*b*")
            delta_e00_comp, valid_de_comp = safe_convert_list(self.delta_e00_components, "ΔE00 Component")
            msssim_L, valid_msssim = safe_convert_list(self.msssim_L_components, "MS-SSIM L*")
            grad_norms, valid_grad = safe_convert_list(self.grad_norms, "Grad Norm")
            val_ssims, valid_ssims = safe_convert_list(self.val_ssims, "Val SSIM")
            val_psnrs, valid_psnrs = safe_convert_list(self.val_psnrs, "Val PSNR")
            
            # Create 5-subplot figure (removed: Validation SSIM always ~0.99, ΔE00 Component always 0 when β=0)
            plt.figure(figsize=(15, 20))
            
            # Plot 1: Train CVD Loss
            plt.subplot(5, 1, 1)
            if valid_train:
                valid_epochs = [epochs[i] for i, v in enumerate(train_losses) if v is not None]
                plt.plot(valid_epochs, valid_train, 'b-', alpha=0.8, linewidth=1)
                # Titolo dinamico: 2 o 3 componenti in base a beta (ΔE00 nella loss)
                beta_val = self.beta_weights[-1] if self.beta_weights else 0.0
                n_components = 3 if beta_val > 0 else 2
                components_str = "MSE a*b* + MS-SSIM" if beta_val == 0 else "MSE a*b* + ΔE00 + MS-SSIM"
                plt.title(f'Train Loss (CVDLoss {n_components}-component: {components_str}) - Epoch {epoch}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid train loss data', ha='center', va='center', transform=plt.gca().transAxes)
            
            # Plot 2: Validation Delta-E2000 (PRIMARY)
            plt.subplot(5, 1, 2)
            if valid_delta_e00:
                valid_epochs = [epochs[i] for i, v in enumerate(val_delta_e00s) if v is not None]
                plt.plot(valid_epochs, valid_delta_e00, 'r-', alpha=0.8, linewidth=2, label='Val ΔE00')
                plt.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Clinical threshold (2.0)')
                plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent (1.0)')
                plt.title(f'Validation ΔE00 (PRIMARY - Early Stopping) - Epoch {epoch}')
                plt.xlabel('Epoch')
                plt.ylabel('ΔE00 (Lower is Better)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid ΔE00 data', ha='center', va='center', transform=plt.gca().transAxes)
            
            # Plot 3: MSE a*b* Component (NORMALIZED)
            plt.subplot(5, 1, 3)
            if valid_mse_ab:
                valid_epochs = [epochs[i] for i, v in enumerate(mse_ab) if v is not None]
                plt.plot(valid_epochs, valid_mse_ab, 'g-', alpha=0.8, linewidth=1, label='MSE a*b* (norm)')
                plt.title(f'MSE a*b* Component NORMALIZED (α={self.alpha_weights[-1] if self.alpha_weights else 0.3:.2f}) - Epoch {epoch}')
                plt.xlabel('Epoch')
                plt.ylabel('MSE a*b* (normalized)')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid MSE a*b* data', ha='center', va='center', transform=plt.gca().transAxes)
            
            # Plot 4: MS-SSIM RGB Loss Component (NORMALIZED)
            plt.subplot(5, 1, 4)
            if valid_msssim:
                valid_epochs = [epochs[i] for i, v in enumerate(msssim_L) if v is not None]
                plt.plot(valid_epochs, valid_msssim, 'c-', alpha=0.8, linewidth=1, label='MS-SSIM (norm)')
                gamma_val = self.gamma_weights[-1] if self.gamma_weights else 0.3
                plt.title(f'MS-SSIM RGB Loss NORMALIZED (γ={gamma_val:.2f}) - Epoch {epoch}')
                plt.xlabel('Epoch')
                plt.ylabel('MS-SSIM Loss (normalized)')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid MS-SSIM data', ha='center', va='center', transform=plt.gca().transAxes)
            
            # Plot 5: Gradient Norm
            plt.subplot(5, 1, 5)
            if valid_grad:
                valid_epochs = [epochs[i] for i, v in enumerate(grad_norms) if v is not None]
                plt.plot(valid_epochs, valid_grad, 'k-', alpha=0.6, linewidth=1)
                plt.title(f'Gradient Norm - Epoch {epoch}')
                plt.xlabel('Epoch')
                plt.ylabel('Grad Norm')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid gradient data', ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"epoch_{epoch:03d}_cvd_metrics.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            # Dedicated Delta-E2000 plot
            if valid_delta_e00:
                plt.figure(figsize=(10, 6))
                valid_epochs = [epochs[i] for i, v in enumerate(val_delta_e00s) if v is not None]
                plt.plot(valid_epochs, valid_delta_e00, 'r-', alpha=0.8, linewidth=2)
                plt.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Clinical threshold (ΔE00 < 2.0)')
                plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent (ΔE00 < 1.0)')
                plt.title(f'CVD Training: Validation ΔE00 Progress - Epoch {epoch}')
                plt.xlabel('Epoch')
                plt.ylabel('ΔE00 (Lower is Better)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(self.plots_dir / f"epoch_{epoch:03d}_delta_e00_dedicated.png", dpi=100, bbox_inches='tight')
                plt.close()
            
            if self.debug_verbose:
                print(f"[INFO] Epoch {epoch}: Plot delle metriche salvati")
            
        except Exception as e:
            print(f"[ERROR] Epoch {epoch}: Errore nel salvataggio plot: {str(e)}")
            traceback.print_exc()
            plt.close('all')

    def _save_visual_comparison(self, epoch, input_tensor, target_tensor, output_tensor):
        """Salva un confronto visivo tra input, target e output.
        
        NOTA: Usa lo stesso processing della versione labeled per coerenza.
        Applica gamut_clip per evitare artefatti di contrasto.
        """
        try:
            input_cpu = tensor_to_cpu_safe(input_tensor)
            output_cpu = tensor_to_cpu_safe(output_tensor)
            target_cpu = tensor_to_cpu_safe(target_tensor) if target_tensor is not None else None
            
            if input_cpu is None or output_cpu is None:
                print(f"[WARNING] Epoch {epoch}: Tensori vuoti per visualizzazione, skip")
                return
            
            input_denorm = denormalize_tensor(input_cpu)
            # Model output is in [-1, 1] (tanh) - convert to [0, 1] for visualization
            output_denorm = ((output_cpu + 1.0) / 2.0).clamp(0, 1)
            # target arrives already denormalized [0,1] from train.py (target_denorm)
            target_denorm = target_cpu.clamp(0, 1) if target_cpu is not None else None
            
            # Applica gamut_clip per coerenza con la versione labeled
            from perceptual_gamut_clip import gamut_clip
            
            def tensor_to_clipped(tensor):
                """Converte tensor [C,H,W] a numpy [H,W,C] con gamut clip."""
                if tensor.dtype in [torch.float16, torch.bfloat16]:
                    tensor = tensor.float()
                np_img = tensor.permute(1, 2, 0).numpy().astype('float32')
                clipped = gamut_clip(np_img, mode="full")
                # Riconverti a tensor [C,H,W]
                return torch.from_numpy(clipped).permute(2, 0, 1)
            
            img_in = tensor_to_clipped(input_denorm[0])
            img_out = tensor_to_clipped(output_denorm[0])
            
            if target_denorm is not None:
                img_target = tensor_to_clipped(target_denorm[0])
                comparison = torch.cat([img_in, img_target, img_out], dim=2)
            else:
                comparison = torch.cat([img_in, img_out], dim=2)
            
            output_path = self.plots_dir / f"epoch_{epoch:03d}_reconstruction.png"
            vutils.save_image(comparison, output_path)
            
            if epoch <= 3:
                print(f"[VISUAL] Epoch {epoch}: Confronto salvato in {output_path}")
                
        except Exception as e:
            print(f"[ERROR] Epoch {epoch}: Errore nel salvataggio confronto visivo: {str(e)}")

    def _save_labeled_visual_comparison(self, epoch, input_tensor, target_tensor, output_tensor):
        """Salva un confronto visivo CON LABEL chiare."""
        try:
            input_cpu = tensor_to_cpu_safe(input_tensor)
            output_cpu = tensor_to_cpu_safe(output_tensor)
            target_cpu = tensor_to_cpu_safe(target_tensor) if target_tensor is not None else None
            
            if input_cpu is None or output_cpu is None:
                return
            
            input_denorm = denormalize_tensor(input_cpu)
            # Model output is in [-1, 1] (tanh) - convert to [0, 1] for visualization
            output_denorm = ((output_cpu + 1.0) / 2.0).clamp(0, 1)
            # target arrives already denormalized [0,1] from train.py (target_denorm)
            target_denorm = target_cpu.clamp(0, 1) if target_cpu is not None else None
            
            from perceptual_gamut_clip import gamut_clip
            
            def safe_to_numpy(tensor):
                if tensor.dtype in [torch.float16, torch.bfloat16]:
                    tensor_np = tensor.float().permute(1, 2, 0).numpy().astype('float32')
                else:
                    tensor_np = tensor.permute(1, 2, 0).numpy().astype('float32')
                return gamut_clip(tensor_np, mode="full")
            
            img_in = safe_to_numpy(input_denorm[0])
            img_out = safe_to_numpy(output_denorm[0])
            
            if target_denorm is not None:
                img_target = safe_to_numpy(target_denorm[0])
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img_in)
                axes[0].set_title(self.descriptions["input"], fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(img_target)
                axes[1].set_title(self.descriptions["target"], fontsize=12, fontweight='bold', color='green')
                axes[1].axis('off')
                
                axes[2].imshow(img_out)
                axes[2].set_title(self.descriptions["output"], fontsize=12, fontweight='bold', color='blue')
                axes[2].axis('off')
                
                fig.suptitle(f'{self.plot_title} (Epoch {epoch})', fontsize=14, fontweight='bold')
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                axes[0].imshow(img_in)
                axes[0].set_title(self.descriptions["input"], fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(img_out)
                axes[1].set_title(self.descriptions["output"], fontsize=12, fontweight='bold', color='blue')
                axes[1].axis('off')
                
                fig.suptitle(f'{self.plot_title} (Epoch {epoch})', fontsize=14, fontweight='bold')
            
            output_path = self.plots_dir / f"epoch_{epoch:03d}_reconstruction_labeled.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            if epoch <= 3:
                print(f"[VISUAL LABELED] Epoch {epoch}: Confronto etichettato salvato")
                
        except Exception as e:
            print(f"[ERROR] Epoch {epoch}: Errore nel salvataggio confronto etichettato: {str(e)}")
            traceback.print_exc()

    def test_logger_functionality(self):
        """Metodo di test per verificare che il logger funzioni."""
        print(f"\n[LOGGER TEST] Inizio test funzionalità logger CVD")
        
        try:
            print(f"[TEST 1] Verifica esistenza file:")
            print(f"  TXT log: {self.log_path} -> {'EXISTS' if self.log_path.exists() else 'MISSING'}")
            print(f"  CSV metrics: {self.csv_path} -> {'EXISTS' if self.csv_path.exists() else 'MISSING'}")
            print(f"  CSV batch: {self.batch_log_path} -> {'EXISTS' if self.batch_log_path.exists() else 'MISSING'}")
            
            print(f"[TEST 2] Verifica permessi di scrittura:")
            test_content = f"# Test line {datetime.now().isoformat()}\n"
            
            with open(self.log_path, "a", encoding='utf-8') as f:
                f.write(test_content)
                f.flush()
            print(f"  TXT log: WRITE OK")
            
            print(f"[TEST 3] Test chiamata metodo log() con dati fittizi:")
            self.log(
                epoch=9999,
                train_loss={
                    'total_loss': 0.5,
                    'mse_ab': 0.15,
                    'delta_e00': 0.2,
                    'msssim_L_loss': 0.15,
                    'alpha_weight': 0.3,
                    'beta_weight': 0.4,
                    'gamma_weight': 0.3
                },
                val_delta_e00=1.5,
                val_msssim_L=0.92,
                val_ssim=0.88,
                val_psnr=28.5,
                grad_norm=0.05,
                best_epoch=9998,
                counter_no_improve=1,
                patience=10,
                significant_improvement=False,
                improvement_threshold=0.01
            )
            print(f"[TEST 3] Chiamata log() completata senza errori")
            
            print(f"[TEST 4] Test chiamata metodo log_batch():")
            self.log_batch(
                epoch=9999,
                batch_idx=999,
                loss_data={
                    'total_loss': 0.5,
                    'mse_ab': 0.15,
                    'delta_e00': 0.2,
                    'msssim_L_loss': 0.15
                },
                grad_norm=0.03
            )
            print(f"[TEST 4] Chiamata log_batch() completata senza errori")
            
            print(f"[LOGGER TEST]  Tutti i test completati con successo!")
            return True
            
        except Exception as e:
            print(f"[LOGGER TEST] ✗ Errore durante test: {e}")
            traceback.print_exc()
            return False
