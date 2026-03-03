import random
import numpy as np
import torch
import gc
import yaml
import os


def set_global_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(config: dict = None) -> torch.device:
    """
    Get compute device based on config and hardware availability.
    
    config.execution.device options:
        "auto"  -> GPU if available, else CPU (default)
        "cuda"  -> Force GPU (raises RuntimeError if unavailable)
        "cpu"   -> Force CPU

    config.execution.require_gpu:
        True -> Raise RuntimeError if CUDA is not available
    """
    if config is not None:
        exec_cfg = config.get("execution", {})
        device_pref = exec_cfg.get("device", "auto")
        require_gpu = exec_cfg.get("require_gpu", False)

        if device_pref == "cuda" or require_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU required (require_gpu=true or device=cuda) but CUDA is not available. "
                    "Install the correct CUDA-enabled PyTorch or set require_gpu: false."
                )
            device = torch.device("cuda")
        elif device_pref == "cpu":
            device = torch.device("cpu")
        else:  # "auto"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Fallback: auto-detect
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def print_device_info(config: dict = None):
    """Print GPU/CPU device information at startup."""
    device = get_device(config)
    print(f"  Device: {device}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"  GPU   : {name} ({total_mem:.1f} GB VRAM)")
    else:
        print("  GPU   : Not available -- running on CPU")
    return device


def clear_gpu_memory():
    """Clear GPU memory after each epoch."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE, and sMAPE.

    MAPE uses a small epsilon (1e-5) to avoid division by zero, but remains
    unreliable when |actual| << 1 (e.g. ETTm1 OT near-zero windows).

    sMAPE (symmetric MAPE) uses  2|y-ŷ|/(|y|+|ŷ|+ε) * 100  which is bounded
    [0, 200%] and handles near-zero targets gracefully -- use this as the
    primary accuracy metric for ETTm1.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    errors = actual - predicted
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / (np.abs(actual) + 1e-5)) * 100.0
    smape = np.mean(2.0 * abs_errors / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}
