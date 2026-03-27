# ================================================================================
# precision_utils.py - Advanced Precision Detection for CUDA 11.x/12.x
# ================================================================================
# This module detects device and precision settings optimized for NVIDIA GPUs.
# - bf16 (bfloat16): Primary precision for Ampere+ (RTX 30xx, 40xx, A100, H100)
# - fp32 (float32): Fallback for older GPUs or CPU
# 
# CUDA 12.x Features:
# - torch.compile() for 20-50% speedup (inductor backend)
# - Improved memory allocation and garbage collection
# - Flash Attention 2 support (if using attention layers)
# - Better TensorCore utilization
#
# IMPORTANT: GradScaler is NOT needed for bf16 (wide dynamic range prevents underflow)
# ================================================================================

import os
import torch
from typing import Tuple, Optional, Dict, Any


# ================================================================================
# CUDA Version Detection
# ================================================================================

def get_cuda_version() -> Optional[Tuple[int, int]]:
    """
    Get CUDA version as tuple (major, minor).
    
    Returns:
        Tuple[int, int] or None if CUDA not available
        
    Examples:
        (12, 1) for CUDA 12.1
        (11, 8) for CUDA 11.8
    """
    if not torch.cuda.is_available():
        return None
    
    cuda_version = torch.version.cuda
    if cuda_version:
        parts = cuda_version.split('.')
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
        except (ValueError, IndexError):
            return None
    return None


def is_cuda_12_or_newer() -> bool:
    """Check if CUDA 12.x or newer is available."""
    version = get_cuda_version()
    return version is not None and version[0] >= 12


def is_cuda_11_or_newer() -> bool:
    """Check if CUDA 11.x or newer is available."""
    version = get_cuda_version()
    return version is not None and version[0] >= 11


# ================================================================================
# torch.compile() Support (CUDA 12.x optimization)
# ================================================================================

def supports_torch_compile() -> bool:
    """
    Check if torch.compile() is available and beneficial.
    
    Best with CUDA 12.x + PyTorch 2.x for optimal inductor backend performance.
    
    Returns:
        bool: True if torch.compile should be used
    """
    # Check PyTorch version has compile
    if not hasattr(torch, 'compile'):
        return False
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        return False
    
    # CUDA 12.x has best support for torch.compile inductor backend
    cuda_version = get_cuda_version()
    if cuda_version is None:
        return False
    
    # Require CUDA 11.8+ minimum, optimal with 12.x
    if cuda_version[0] < 11 or (cuda_version[0] == 11 and cuda_version[1] < 8):
        return False
    
    # Check compute capability (Ampere+ recommended: >= 8.0)
    compute_cap = torch.cuda.get_device_properties(0).major
    if compute_cap < 7:  # Minimum Volta for decent performance
        return False
    
    return True


def compile_model(model: torch.nn.Module, 
                  mode: str = "reduce-overhead",
                  verbose: bool = True) -> torch.nn.Module:
    """
    Compile model with torch.compile() for CUDA 12.x optimization.
    
    Args:
        model: PyTorch model to compile
        mode: Compilation mode:
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Best for inference (reduces Python overhead)
            - "max-autotune": Maximum optimization (longer compile time)
        verbose: Print compilation status
    
    Returns:
        Compiled model (or original if compilation fails/unsupported)
    """
    if not supports_torch_compile():
        if verbose:
            print("[torch.compile] Not supported on this system - using eager mode")
        return model
    
    try:
        compiled_model = torch.compile(model, mode=mode)
        if verbose:
            cuda_ver = get_cuda_version()
            print(f"[torch.compile] Model compiled successfully!")
            print(f"[torch.compile] Mode: {mode}")
            print(f"[torch.compile] CUDA: {cuda_ver[0]}.{cuda_ver[1]}")
            print(f"[torch.compile] Expected speedup: 20-50%")
        return compiled_model
    except Exception as e:
        if verbose:
            print(f"[torch.compile] Compilation failed: {e}")
            print(f"[torch.compile] Falling back to eager mode")
        return model


# ================================================================================
# Memory Optimization (CUDA 12.x features)
# ================================================================================

def optimize_cuda_memory(verbose: bool = True) -> None:
    """
    Apply CUDA 12.x memory optimizations.
    
    Features:
    - Improved memory allocator settings
    - Garbage collection hints
    - Memory pool configuration
    """
    if not torch.cuda.is_available():
        return
    
    cuda_version = get_cuda_version()
    
    # CUDA 12.x specific optimizations
    if cuda_version and cuda_version[0] >= 12:
        # Enable memory efficient allocator (CUDA 12.x)
        try:
            # Set expandable segments for better memory reuse
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
                                  'expandable_segments:True,garbage_collection_threshold:0.8')
            if verbose:
                print("[CUDA 12.x] Memory optimizations enabled:")
                print("  - Expandable segments: ON")
                print("  - GC threshold: 0.8")
        except Exception as e:
            if verbose:
                print(f"[CUDA 12.x] Memory optimization warning: {e}")
    
    # General optimizations (CUDA 11.x+)
    if cuda_version and cuda_version[0] >= 11:
        try:
            # Enable cudnn benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for Ampere+ (faster matmul with minimal precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if verbose:
                print("[CUDA 11.x+] General optimizations enabled:")
                print("  - cuDNN benchmark: ON")
                print("  - TF32 matmul: ON")
        except Exception as e:
            if verbose:
                print(f"[CUDA 11.x+] Optimization warning: {e}")


# ================================================================================
# Device and Precision Detection (Main Function)
# ================================================================================

def detect_device_and_precision(config: dict = None, verbose: bool = True):
    """
    Detect optimal device and precision for training.
    
    Simplified for RTX 3090 (Ampere architecture):
    - bf16: Primary mode, uses autocast with bfloat16 (NO GradScaler needed)
    - fp32: Fallback for CPU or non-bf16 capable GPUs
    
    Args:
        config: Optional config dict with 'device' and 'amp_dtype' keys
        verbose: Print detection results (default True)
    
    Returns:
        Tuple of (device, amp_dtype, precision_key):
        - device: torch.device (cuda or cpu)
        - amp_dtype: torch.dtype or None (bfloat16 for AMP, None for fp32)
        - precision_key: str for logging ("bf16" or "fp32")
    """
    config = config or {}
    
    # Device detection
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    
    # Precision detection
    amp_dtype = None
    precision_key = "fp32"  # Default fallback
    
    if device.type == 'cuda':
        # Check bf16 support (Ampere and newer - compute capability >= 8.0)
        bf16_supported = torch.cuda.is_bf16_supported()
        
        # Get requested precision from config
        requested_dtype = config.get('amp_dtype', 'auto')
        
        if requested_dtype in ['bfloat16', 'bf16', 'auto'] and bf16_supported:
            amp_dtype = torch.bfloat16
            precision_key = "bf16"
        elif requested_dtype in ['float32', 'fp32', None]:
            amp_dtype = None
            precision_key = "fp32"
        elif not bf16_supported and requested_dtype in ['bfloat16', 'bf16']:
            # bf16 requested but not supported - fallback to fp32
            amp_dtype = None
            precision_key = "fp32"
            if verbose:
                print(f"[PRECISION] [WARNING] bf16 requested but NOT supported - falling back to fp32")
        else:
            # Unknown dtype - try auto
            if bf16_supported:
                amp_dtype = torch.bfloat16
                precision_key = "bf16"
            else:
                amp_dtype = None
                precision_key = "fp32"
        
        # SAFETY OVERRIDE: Force FP32 for debugging numerical issues
        force_fp32_debug = os.environ.get('FORCE_FP32_DEBUG', 'false').lower() == 'true'
        if force_fp32_debug:
            amp_dtype = None
            precision_key = "fp32"
            if verbose:
                print(f"[PRECISION OVERRIDE] FORCE_FP32_DEBUG=true detected - using FP32")
    else:
        # CPU mode - no AMP
        amp_dtype = None
        precision_key = "fp32"
    
    # Print results
    if verbose:
        cuda_ver = get_cuda_version()
        
        print("\n" + "="*70)
        print("PRECISION CONFIGURATION")
        print("="*70)
        print(f"  Device:         {device}")
        
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            compute_cap = torch.cuda.get_device_properties(0)
            print(f"  GPU:            {gpu_name}")
            print(f"  VRAM:           {gpu_mem:.1f} GB")
            print(f"  Compute Cap.:   {compute_cap.major}.{compute_cap.minor}")
            print(f"  CUDA Version:   {cuda_ver[0]}.{cuda_ver[1]}" if cuda_ver else "  CUDA Version:   N/A")
            print(f"  bf16 Support:   {'YES' if torch.cuda.is_bf16_supported() else 'NO'}")
            print(f"  torch.compile:  {'YES' if supports_torch_compile() else 'NO'}")
            
            # CUDA 12.x specific features
            if cuda_ver and cuda_ver[0] >= 12:
                print(f"  CUDA 12.x:       Advanced memory allocation")
                print(f"                   Improved TensorCore utilization")
                print(f"                   torch.compile inductor backend")
        
        print(f"  Precision:      {precision_key.upper()}")
        print(f"  AMP Enabled:    {'YES (autocast)' if amp_dtype is not None else 'NO'}")
        if amp_dtype is not None:
            print(f"  AMP dtype:      {amp_dtype}")
            print(f"  GradScaler:     NOT USED (bf16 has wide dynamic range)")
        print("="*70 + "\n")
    
    return device, amp_dtype, precision_key


def get_autocast_context(device, amp_dtype):
    """
    Get the appropriate autocast context manager.
    
    Args:
        device: torch.device
        amp_dtype: torch.dtype or None
    
    Returns:
        Context manager for use in with statement
    """
    if amp_dtype is not None and device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
    else:
        # Return null context manager
        return torch.amp.autocast(device_type='cuda', enabled=False)


def reset_precision_cache():
    """
    Legacy function - kept for backward compatibility.
    No longer uses caching, so this is a no-op.
    """
    print("[INFO] reset_precision_cache() called - no cache to reset (simplified mode)")
