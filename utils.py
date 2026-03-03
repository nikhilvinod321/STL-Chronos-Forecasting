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


def get_device() -> torch.device:
    """Get compute device with CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    """Compute MAE, RMSE, and MAPE with epsilon to prevent div-by-zero."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors) / (np.abs(actual) + 1e-5)) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
