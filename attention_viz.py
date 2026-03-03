import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
Attention Weight Extraction and Visualization for Chronos-T5.

Hypothesis: STL decomposition flattens macro-structural context, causing
transformer attention heads to assign low-confidence, uniform weights to noise.

This script:
  1. Loads amazon/chronos-t5-small
  2. Hooks into attention layers of the final encoder layer
  3. Runs two forward passes:
      a) Raw load data (original context)
      b) STL-decomposed residual (flattened context)
  4. Plots attention weight heatmaps side-by-side at 300 DPI (IEEE quality)

Usage:
  python attention_viz.py [--window_idx 0] [--layer -1] [--output_dir plots]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import torch
from chronos import ChronosPipeline

from utils import load_config, set_global_seed, get_device
from data_loader import load_data, split_train_test, create_rolling_windows
from decomposition import STLDecomposer


# -----------------------------------------------------------------------------
# IEEE-quality style context
# -----------------------------------------------------------------------------
IEEE_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


class AttentionExtractor:
    """
    Hooks into a specific T5 encoder layer to extract self-attention weights.
    Cross-attention in decoder layer can also be extracted.
    """

    def __init__(self, model_id: str = "amazon/chronos-t5-small", config: dict = None):
        self.model_id = model_id
        self.config = config or {}
        self.device = get_device(config)
        self.pipeline = None
        self._hooks = []
        # attention_weights[layer_idx][(head_idx)] = tensor(seq_len, seq_len)
        self._attention_cache = {}

    def load(self):
        """Load the Chronos pipeline."""
        from utils import print_device_info
        print_device_info(self.config)

        device_map = "cuda" if self.device.type == "cuda" else "cpu"
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=device_map,
            dtype=dtype,
        )
        print(f"Loaded {self.model_id} on {device_map.upper()}")
        return self

    def _make_hook(self, layer_name: str):
        """Factory: returns a forward hook that stores attention weights."""
        def hook(module, input, output):
            # output is typically (attn_output, attn_weights_or_None, ...)
            # T5Attention returns (attn_output, position_bias, weights_if_output_requested)
            # We enable output_attentions in the forward call
            if isinstance(output, tuple) and len(output) >= 2:
                # The attention weights tensor is the one with shape (..., seq_q, seq_k)
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        # shape: (batch, heads, seq_q, seq_k)
                        self._attention_cache[layer_name] = item.detach().cpu().float()
                        break
        return hook

    def register_hooks(self, layer_indices: list = None):
        """
        Register forward hooks on T5 encoder self-attention modules.
        """
        t5_model = self.pipeline.model.model  # T5ForConditionalGeneration inner model
        encoder = t5_model.encoder

        # Get all attention layers
        encoder_layers = encoder.block
        n_layers = len(encoder_layers)

        if layer_indices is None:
            # Default: hook only the final encoder layer
            layer_indices = [n_layers - 1]

        for idx in layer_indices:
            if idx < 0:
                idx = n_layers + idx
            if 0 <= idx < n_layers:
                attn_module = encoder_layers[idx].layer[0].SelfAttention
                hook = attn_module.register_forward_hook(
                    self._make_hook(f"encoder_layer_{idx}")
                )
                self._hooks.append(hook)
                print(f"  Hooked encoder layer {idx}")

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def extract_attention(self, context: np.ndarray, horizon: int = 48) -> dict:
        """
        Run a forward pass with output_attentions=True and capture attention weights.

        Args:
            context: 1D numpy array (context_length,)
            horizon: forecast length (needed for pipeline call)

        Returns:
            dict: layer_name -> attention_tensor (batch, heads, seq_q, seq_k)
        """
        self._attention_cache.clear()
        context_tensor = torch.tensor(context, dtype=torch.float32)

        # Enable attention output for the forward pass
        t5_model = self.pipeline.model.model
        encoder = t5_model.encoder

        # patch config to output attentions temporarily
        original_output_attentions = t5_model.config.output_attentions
        t5_model.config.output_attentions = True

        with torch.no_grad():
            # Use the ChronosPipeline tokenizer to build proper inputs
            ctx_tensor_2d = context_tensor.unsqueeze(0)
            input_ids, attention_mask, _ = self.pipeline.tokenizer.context_input_transform(
                ctx_tensor_2d
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # Run only the encoder pass
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Restore config
        t5_model.config.output_attentions = original_output_attentions

        # If hooks caught something, use those; otherwise fall through encoder_outputs
        if not self._attention_cache and hasattr(encoder_outputs, "attentions") and \
                encoder_outputs.attentions is not None:
            for layer_idx, attn in enumerate(encoder_outputs.attentions):
                self._attention_cache[f"encoder_layer_{layer_idx}"] = attn.detach().cpu().float()

        return dict(self._attention_cache)


def aggregate_attention_map(attn: torch.Tensor) -> np.ndarray:
    """
    Aggregate attention tensor across all heads by averaging.
    Args:
        attn: (batch, heads, seq_q, seq_k) tensor
    Returns:
        2D numpy array (seq_q, seq_k)
    """
    # Mean over batch dim (usually 1) and head dim
    agg = attn.squeeze(0).mean(dim=0).numpy()  # (seq_q, seq_k)
    return agg


def compute_attention_entropy(attn_map: np.ndarray) -> float:
    """
    Compute mean Shannon entropy of attention distributions.
    Higher entropy -> more uniform (less focused) attention.
    """
    row_sums = attn_map.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-12, row_sums)
    p = attn_map / row_sums
    
    # Clip to avoid log(0)
    p = np.clip(p, 1e-12, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=-1)  # (seq_q,)
    return float(np.mean(entropy))


def plot_attention_heatmaps(
    raw_attn: np.ndarray,
    stl_attn: np.ndarray,
    save_path: str = "plots/attention_heatmaps.png",
    layer_name: str = "Final Encoder Layer",
    downsample_to: int = 96,
):
    """
    Plot attention heatmaps for raw vs STL-residual contexts side-by-side.
    IEEE print-quality at 300 DPI.

    Args:
        raw_attn: (seq_q, seq_k) aggregated attention from raw context
        stl_attn: (seq_q, seq_k) aggregated attention from STL residual context
        save_path: output path
        layer_name: label for plot title
        downsample_to: downsample sequence axis to this length for readability
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # Downsample for visual clarity
    def downsample(mat, target):
        if mat.shape[0] > target:
            step = max(1, mat.shape[0] // target)
            mat = mat[::step, ::step][:target, :target]
        return mat

    raw_ds = downsample(raw_attn, downsample_to)
    stl_ds = downsample(stl_attn, downsample_to)

    raw_entropy = compute_attention_entropy(raw_ds)
    stl_entropy = compute_attention_entropy(stl_ds)

    with plt.rc_context(IEEE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        
        # Shared vmax for fair comparison
        vmax = max(raw_ds.max(), stl_ds.max())
        vmin = 0.0

        cmap = "YlOrRd"
        kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, square=True,
                  linewidths=0.0, cbar=True, xticklabels=False, yticklabels=False)

        # -- Raw attention -----------------------------------------------------
        ax1 = axes[0]
        sns.heatmap(raw_ds, ax=ax1, **kw,
                    cbar_kws={"shrink": 0.8, "label": "Attention Weight"})
        ax1.set_title(f"Raw Load Context\n(H={raw_entropy:.3f} bits)")
        ax1.set_xlabel("Key Position (downsampled)")
        ax1.set_ylabel("Query Position (downsampled)")

        # -- STL Residual attention --------------------------------------------
        ax2 = axes[1]
        sns.heatmap(stl_ds, ax=ax2, **kw,
                    cbar_kws={"shrink": 0.8, "label": "Attention Weight"})
        ax2.set_title(f"STL Residual Context\n(H={stl_entropy:.3f} bits)")
        ax2.set_xlabel("Key Position (downsampled)")
        ax2.set_ylabel("Query Position (downsampled)")

        fig.suptitle(
            f"Attention Weight Heatmaps -- {layer_name}\n"
            f"ΔEntropy (STL − Raw) = {(stl_entropy - raw_entropy):+.3f} bits",
            fontsize=11,
            y=1.01,
        )

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"  Saved attention heatmaps -> {save_path}")
    print(f"  Raw entropy  : {raw_entropy:.4f} bits")
    print(f"  STL entropy  : {stl_entropy:.4f} bits")
    delta = stl_entropy - raw_entropy
    direction = "more uniform (SUPPORTS hypothesis)" if delta > 0 else "more focused (REFUTES hypothesis)"
    print(f"  Delta        : {delta:+.4f} bits  [{direction}]")

    return save_path, raw_entropy, stl_entropy


def plot_attention_per_head(
    raw_attn: torch.Tensor,
    stl_attn: torch.Tensor,
    save_path: str = "plots/attention_per_head.png",
    max_heads: int = 8,
    downsample_to: int = 64,
):
    """
    Plot a grid of per-head attention maps (raw vs STL) for the first N heads.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # raw_attn, stl_attn: (batch, heads, seq_q, seq_k)
    raw_np = raw_attn.squeeze(0).numpy()     # (heads, seq_q, seq_k)
    stl_np = stl_attn.squeeze(0).numpy()

    n_heads = min(raw_np.shape[0], max_heads)

    def ds(mat, t):
        if mat.shape[0] > t:
            step = max(1, mat.shape[0] // t)
            return mat[::step, ::step][:t, :t]
        return mat

    with plt.rc_context(IEEE_STYLE):
        fig, axes = plt.subplots(2, n_heads, figsize=(n_heads * 1.8, 4.5))
        if n_heads == 1:
            axes = axes.reshape(2, 1)

        for h in range(n_heads):
            raw_h = ds(raw_np[h], downsample_to)
            stl_h = ds(stl_np[h], downsample_to)
            vmax = max(raw_h.max(), stl_h.max())

            kw = dict(vmin=0, vmax=vmax, cmap="YlOrRd",
                      xticklabels=False, yticklabels=False, cbar=False, square=True)

            sns.heatmap(raw_h, ax=axes[0, h], **kw)
            axes[0, h].set_title(f"H{h+1}", fontsize=8)
            if h == 0:
                axes[0, h].set_ylabel("Raw", fontsize=8)

            sns.heatmap(stl_h, ax=axes[1, h], **kw)
            if h == 0:
                axes[1, h].set_ylabel("STL", fontsize=8)

        fig.suptitle(f"Per-Head Attention Maps -- Final Encoder Layer\n({n_heads} heads shown)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"  Saved per-head attention -> {save_path}")


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def run_attention_analysis(config: dict, window_idx: int = 0,
                           layer_idx: int = -1, output_dir: str = "plots"):
    """
    Full attention extraction and visualization pipeline.
    Args:
        config: experiment config dict
        window_idx: which test window to use (default 0)
        layer_idx: which encoder layer to visualize (-1 = final)
        output_dir: where to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(config["execution"]["random_seed"])

    # 1. Load data & create windows
    print("[Attn] Loading data...")
    df = load_data(config)
    train_df, test_df = split_train_test(df, config)

    context_length = config["evaluation"]["context_length"]
    horizon = config["evaluation"]["horizon"]
    stride = config["evaluation"]["stride"]

    windows = create_rolling_windows(
        df, test_df,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        dry_run=True,
        max_windows=max(window_idx + 1, 5),
    )

    if window_idx >= len(windows):
        window_idx = 0
        print(f"  [Attn] Requested window {window_idx} not available, using window 0.")

    w = windows[window_idx]
    raw_context = w["context"].values  # Shape: (context_length,)

    # 2. Compute STL residual context
    print("[Attn] Computing STL decomposition...")
    decomposer = STLDecomposer(period=168, robust=True)
    decomposer.fit_dow_adjustments(train_df)
    result = decomposer.decompose_and_project(
        w["context"], w["target"].index, horizon=horizon, use_dow=True
    )
    stl_residual = result["context_residual"]  # Shape: (context_length,)
    print(f"  Raw context   : mean={raw_context.mean():.1f}, std={raw_context.std():.1f}")
    print(f"  STL residual  : mean={stl_residual.mean():.4f}, std={stl_residual.std():.4f}")

    # 3. Load model and register hooks
    print("[Attn] Loading Chronos model with attention hooks...")
    extractor = AttentionExtractor(config["model"]["chronos_id"], config=config)
    extractor.load()

    # Determine number of encoder layers
    t5_model = extractor.pipeline.model.model
    n_encoder_layers = len(t5_model.encoder.block)
    final_layer_idx = n_encoder_layers - 1 if layer_idx == -1 else layer_idx
    print(f"  Model has {n_encoder_layers} encoder layers. Hooking layer {final_layer_idx}.")
    extractor.register_hooks(layer_indices=[final_layer_idx])

    # 4. Extract attention for raw context
    print("[Attn] Extracting attention -- raw context...")
    raw_attention_cache = extractor.extract_attention(raw_context, horizon=horizon)

    # 5. Extract attention for STL residual
    print("[Attn] Extracting attention -- STL residual...")
    stl_attention_cache = extractor.extract_attention(stl_residual, horizon=horizon)

    extractor.remove_hooks()

    layer_key = f"encoder_layer_{final_layer_idx}"

    if layer_key not in raw_attention_cache or layer_key not in stl_attention_cache:
        print(f"  [WARNING] Attention not found for key '{layer_key}'.")
        print(f"  Available keys: {list(raw_attention_cache.keys())}")
        return

    raw_attn_tensor = raw_attention_cache[layer_key]   # (1, heads, seq_q, seq_k)
    stl_attn_tensor = stl_attention_cache[layer_key]

    # 6. Aggregate and plot
    print("[Attn] Generating heatmap plots...")
    raw_agg = aggregate_attention_map(raw_attn_tensor)
    stl_agg = aggregate_attention_map(stl_attn_tensor)

    layer_label = f"Encoder Layer {final_layer_idx} (Final)"
    plot_attention_heatmaps(
        raw_agg, stl_agg,
        save_path=os.path.join(output_dir, "attention_heatmaps.png"),
        layer_name=layer_label,
        downsample_to=96,
    )

    plot_attention_per_head(
        raw_attn_tensor, stl_attn_tensor,
        save_path=os.path.join(output_dir, "attention_per_head.png"),
        max_heads=8,
        downsample_to=64,
    )

    # 7. Entropy bar chart
    _plot_entropy_comparison(
        raw_agg, stl_agg,
        save_path=os.path.join(output_dir, "attention_entropy.png"),
        layer_label=layer_label,
    )


def _plot_entropy_comparison(raw_agg, stl_agg, save_path, layer_label):
    """Bar chart comparing per-token attention entropy."""
    raw_ent = compute_attention_entropy(raw_agg)
    stl_ent = compute_attention_entropy(stl_agg)

    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=(4, 3))

        bars = ax.bar(
            ["Raw Context", "STL Residual"],
            [raw_ent, stl_ent],
            color=["#1f77b4", "#d62728"],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, v in zip(bars, [raw_ent, stl_ent]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax.set_ylabel("Mean Attention Entropy (bits)")
        ax.set_title(f"Attention Entropy -- {layer_label}\nHigher = More Uniform Attention")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"  Saved entropy comparison -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronos Attention Weight Extraction & Visualization")
    parser.add_argument("--window_idx", type=int, default=0,
                        help="Test window index to analyze (default: 0)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Encoder layer index (-1 = final, default: -1)")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Output directory for plots (default: plots/)")
    args = parser.parse_args()

    config = load_config()
    run_attention_analysis(
        config,
        window_idx=args.window_idx,
        layer_idx=args.layer,
        output_dir=args.output_dir,
    )
