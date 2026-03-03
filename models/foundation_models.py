"""
Foundation model wrappers for second foundation models.
Includes: Chronos-Bolt (architecturally distinct from Chronos-T5),
Lag-Llama, and TimesFM.
Provides a unified predict(context, horizon) interface.
"""

import torch
import numpy as np
from utils import get_device, clear_gpu_memory


class LagLlamaModel:
    """
    Lag-Llama zero-shot forecasting model.
    Decoder-only architecture (Meta LLaMA-style) trained on diverse time series.
    GitHub : https://github.com/time-series-foundation-models/lag-llama
    HF hub : time-series-foundation-models/Lag-Llama

    Install : pip install lag-llama
    Also installs gluonts, pytorch-lightning, and huggingface-hub as deps.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        device_obj = get_device(config)
        self.device_str = "cuda" if device_obj.type == "cuda" else "cpu"
        self.predictor = None
        self._prediction_length = None

    def load_model(self):
        """Download checkpoint from HF Hub and build a zero-shot GluonTS predictor."""
        try:
            import huggingface_hub
            import torch
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError as exc:
            raise ImportError(
                "Lag-Llama not installed. Run:  pip install lag-llama"
            ) from exc

        horizon         = self.config.get("evaluation", {}).get("horizon", 48)
        context_length  = self.config.get("evaluation", {}).get("context_length", 336)
        self._prediction_length = horizon

        print("  Downloading Lag-Llama checkpoint (time-series-foundation-models/Lag-Llama)...",
              flush=True)
        ckpt_path = huggingface_hub.hf_hub_download(
            "time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
        )
        ckpt = torch.load(ckpt_path, map_location=self.device_str, weights_only=False)
        model_kwargs = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})

        # Build estimator using architecture params stored in the checkpoint.
        # Skip distr_output (stored as object, estimator wants a string),
        # lags_seq (checkpoint stores computed integers; estimator expects freq
        # strings), and device/num_parallel_samples which we control explicitly.
        safe_keys = {"input_size", "n_layer", "n_embd_per_head", "n_head",
                     "scaling", "time_feat"}
        safe_kwargs = {k: v for k, v in model_kwargs.items() if k in safe_keys}
        import torch as _t
        device_obj = _t.device(self.device_str)
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=horizon,
            context_length=context_length,
            device=device_obj,
            num_parallel_samples=20,
            batch_size=1,
            nonnegative_pred_samples=False,
            aug_prob=0,
            lr=0,
            **safe_kwargs,
        )

        # Zero-shot: build predictor from checkpoint weights.
        # PyTorch >= 2.6 requires explicitly allowlisting non-tensor classes
        # (like GluonTS distribution objects) that are pickled in the ckpt.
        from gluonts.torch.modules.loss import NegativeLogLikelihood, DistributionLoss
        from gluonts.torch.distributions.studentT import StudentTOutput
        _t.serialization.add_safe_globals(
            [NegativeLogLikelihood, DistributionLoss, StudentTOutput]
        )
        lightning_module = estimator.create_lightning_module()
        transformation   = estimator.create_transformation()
        self.predictor   = estimator.create_predictor(
            transformation=transformation,
            module=lightning_module,
        )
        print(f"  Lag-Llama predictor ready (zero-shot, horizon={horizon}, "
              f"context={context_length}, device={self.device_str}).", flush=True)
        return self

    def predict(self, context: np.ndarray, horizon: int = 48) -> np.ndarray:
        """
        Zero-shot forecast using Lag-Llama.

        Args:
            context : 1-D numpy array of length context_length
            horizon : forecast horizon (steps)
        Returns:
            forecast : 1-D numpy array of length horizon (median over samples)
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        import pandas as pd
        from gluonts.dataset.common import ListDataset

        # Determine GluonTS period frequency from config
        freq_str = self.config.get("_freq_str", "h")

        ds = ListDataset(
            [{"start": pd.Period("2020-01-01 00", freq=freq_str),
              "target": context.tolist()}],
            freq=freq_str,
        )
        forecasts = list(self.predictor.predict(ds))
        if not forecasts:
            raise RuntimeError("Lag-Llama predictor returned no forecasts.")

        # forecasts[0].samples has shape (num_samples, horizon)
        samples = forecasts[0].samples
        median  = np.median(samples, axis=0)
        return median[:horizon]


class TimesFFMModel:
    """
    TimesFM foundation model for time series forecasting.
    GitHub: https://github.com/google-research/timesfm
    Paper: https://arxiv.org/abs/2310.10688
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = get_device()
        self.model = None

    def load_model(self):
        """Load TimesFM from HuggingFace."""
        try:
            from timesfm import TimesFM
        except ImportError:
            raise ImportError(
                "timesfm not installed. Install with: pip install timesfm"
            )

        self.model = TimesFM(
            context_len=512,
            prediction_len=128,
            use_gpu=self.device.type == "cuda",
        )
        print("TimesFM model loaded.")
        return self

    def predict(self, context: np.ndarray, horizon: int = 48) -> np.ndarray:
        """
        Zero-shot prediction using TimesFM.
        Args:
            context: np.ndarray of shape (context_length,)
            horizon: forecast length
        Returns:
            forecast: np.ndarray of shape (horizon,)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # TimesFM expects 2D input: (num_time_series, time_steps)
        context_2d = context.reshape(1, -1)

        with torch.no_grad():
            # TimesFM returns (num_time_series, horizon) with forecastand variance
            forecast, _ = self.model.forecast(
                context_2d,
                freq=None,  # None = assume uniform spacing
            )

        forecast_np = forecast.squeeze()
        
        # Ensure correct shape and horizon length
        if forecast_np.ndim == 0:
            forecast_np = forecast_np.reshape(-1)
        
        return forecast_np[:horizon]


class ChronosBoltModel:
    """
    Chronos-Bolt foundation model -- architecturally distinct from Chronos-T5.
    Uses an encoder-only PatchTST-based design instead of T5 seq2seq.
    Supported HuggingFace IDs:
        amazon/chronos-bolt-tiny   (~7 M params)
        amazon/chronos-bolt-mini   (~21 M params)
        amazon/chronos-bolt-small  (~48 M params)
        amazon/chronos-bolt-base   (~205 M params)
    Requires only the already-installed 'chronos-forecasting' package.
    """

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", config: dict = None):
        self.model_id = model_id
        self.config = config or {}
        self.device = get_device(config)
        self.pipeline = None

    def load_model(self):
        """Load Chronos-Bolt pipeline from HuggingFace."""
        from chronos import ChronosBoltPipeline
        from utils import print_device_info
        print_device_info(self.config)

        device_map = "cuda" if self.device.type == "cuda" else "cpu"
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.pipeline = ChronosBoltPipeline.from_pretrained(
            self.model_id,
            device_map=device_map,
            dtype=dtype,
        )
        print(f"Chronos-Bolt ({self.model_id}) loaded on {device_map.upper()}.")
        return self

    def predict(self, context: np.ndarray, horizon: int = 48) -> np.ndarray:
        """
        Zero-shot forecast using Chronos-Bolt.
        Args:
            context: 1D numpy array (context_length,)
            horizon: forecast steps
        Returns:
            forecast: 1D numpy array (horizon,)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_model() first.")

        context_tensor = torch.tensor(context, dtype=torch.float32)

        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=context_tensor,
                prediction_length=horizon,
                limit_prediction_length=False,
            )

        # forecast shape: (num_samples, horizon) or (horizon,) depending on version
        if forecast.dim() == 2:
            return torch.median(forecast, dim=0).values.cpu().numpy()
        return forecast.squeeze().cpu().numpy()[:horizon]


def get_foundation_model(model_name: str, config: dict = None):
    """
    Factory function to get a foundation model wrapper.
    Args:
        model_name: one of:
            'chronos-bolt-small' / 'chronos-bolt-base' / 'chronos-bolt-mini' / 'chronos-bolt-tiny'
            'lag-llama'
            'timesfm'
        config: optional config dict (passed through for horizon/context_length/freq_str)
    Returns:
        loaded model instance with a predict(context, horizon) method
    """
    model_name_lower = model_name.lower().strip()

    if "chronos-bolt" in model_name_lower:
        hf_id = f"amazon/{model_name_lower}"  # e.g. amazon/chronos-bolt-small
        model = ChronosBoltModel(model_id=hf_id, config=config)
        return model.load_model()
    elif model_name_lower == "lag-llama":
        model = LagLlamaModel(config=config)
        return model.load_model()
    elif model_name_lower == "timesfm":
        model = TimesFFMModel(config)
        return model.load_model()
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose: 'chronos-bolt-small', 'lag-llama', or 'timesfm'."
        )


if __name__ == "__main__":
    # Test forward pass (requires models to be installed)
    config = {}
    
    # Create dummy context
    context = np.random.randn(336) * 1000 + 30000
    
    print("Testing Lag-Llama (if installed)...")
    try:
        lag_llama = get_foundation_model("lag-llama", config)
        forecast = lag_llama.predict(context, horizon=48)
        print(f"  Lag-Llama forecast shape: {forecast.shape}, mean: {forecast.mean():.1f}")
    except ImportError as e:
        print(f"  Lag-Llama not available: {e}")
    
    print("Testing TimesFM (if installed)...")
    try:
        timesfm = get_foundation_model("timesfm", config)
        forecast = timesfm.predict(context, horizon=48)
        print(f"  TimesFM forecast shape: {forecast.shape}, mean: {forecast.mean():.1f}")
    except ImportError as e:
        print(f"  TimesFM not available: {e}")
