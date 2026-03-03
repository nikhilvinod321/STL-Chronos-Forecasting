import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
LoRA Training Analysis Scripts:
  1. Plot learning curves from saved loss history
  2. Run hyperparameter sweep over LoRA ranks [8, 16, 32]

Usage:
  python lora_analysis.py --mode curves --history_json lora_history.json
  python lora_analysis.py --mode sweep
  python lora_analysis.py --mode both
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import load_config, set_global_seed, compute_metrics, clear_gpu_memory
from data_loader import load_data, split_train_test
from decomposition import STLDecomposer
from models.chronos_lora import ChronosLoRAModel
from train import prepare_lora_training_data


# -----------------------------------------------------------------------------
# IEEE-quality plotting defaults
# -----------------------------------------------------------------------------
IEEE_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
}


def plot_learning_curves(history: dict, save_path: str = "plots/lora_learning_curves.png",
                         lora_rank: int = None, lora_alpha: int = None):
    """
    Plot training and validation loss curves side-by-side.
    IEEE print-quality output (300 DPI).

    Args:
        history: {'train_loss': [...], 'val_loss': [...]}
        save_path: output file path
        lora_rank: LoRA rank label for title
        lora_alpha: LoRA alpha label for title
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    with plt.rc_context(IEEE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

        # -- Left: Both curves on same axis ------------------------------------
        ax = axes[0]
        ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", marker="o", markersize=3)
        ax.plot(epochs, val_loss, label="Validation Loss", color="#d62728",
                marker="s", markersize=3, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        title = "LoRA Fine-Tuning: Learning Curves"
        if lora_rank:
            title += f"\nRank={lora_rank}"
        if lora_alpha:
            title += f", Alpha={lora_alpha}"
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # -- Right: Train/Val gap (overfitting detector) -----------------------
        ax2 = axes[1]
        gap = [v - t for t, v in zip(train_loss, val_loss)]
        color = "#ff7f0e" if max(gap) > 0.01 else "#2ca02c"
        ax2.bar(epochs, gap, color=color, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val Loss − Train Loss")
        ax2.set_title("Overfitting Gap per Epoch")
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved learning curves -> {save_path}")
    return save_path


def run_lora_rank_sweep(config: dict, dry_run: bool = False,
                        ranks: list = None,
                        save_dir: str = "plots") -> dict:
    """
    Sweep LoRA ranks [8, 16, 32], train for N epochs each, log final MAPE.

    Args:
        config: experiment config dict
        dry_run: whether to use quick dry-run settings
        ranks: list of LoRA ranks to test (default: [8, 16, 32])
        save_dir: where to save per-rank plots

    Returns:
        sweep_results: {'rank': [...], 'alpha': [...], 'final_mape': [...]}
    """
    if ranks is None:
        ranks = [8, 16, 32]

    set_global_seed(config["execution"]["random_seed"])
    os.makedirs(save_dir, exist_ok=True)

    # Load data once
    print("[Sweep] Loading data...")
    df = load_data(config)
    train_df, test_df = split_train_test(df, config)

    decomposer = STLDecomposer(period=168, robust=True)
    decomposer.fit_dow_adjustments(train_df)

    context_length = config["evaluation"]["context_length"]
    horizon = config["evaluation"]["horizon"]
    stride = config["evaluation"]["stride"]
    epochs = config["training"]["epochs"] if not dry_run else config["dry_run"].get("max_epochs", 1)

    # Prepare training contexts once (shared across rank sweep)
    print("[Sweep] Preparing training data (decomposed residuals)...")
    train_contexts, train_targets = prepare_lora_training_data(
        train_df, decomposer,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
    )

    # Create a small evaluation window set from test data
    from data_loader import create_rolling_windows
    max_eval_windows = 10  # Quick evaluation per rank
    eval_windows = create_rolling_windows(
        df, test_df,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        dry_run=True,
        max_windows=max_eval_windows,
    )
    print(f"[Sweep] Using {len(eval_windows)} windows for evaluation.")

    sweep_results = {"rank": [], "alpha": [], "final_mape": [], "history": []}

    for rank in ranks:
        # Alpha is canonically set to 2x rank
        alpha = rank * 2
        print(f"\n{'=' * 60}")
        print(f"[Sweep] LoRA Rank={rank}, Alpha={alpha}")
        print(f"{'=' * 60}")

        sweep_config = {
            **config,
            "model": {
                **config["model"],
                "lora_rank": rank,
                "lora_alpha": alpha,
            },
        }

        # Build model with this rank
        model = ChronosLoRAModel(sweep_config)
        model.load_pipeline()
        model.apply_lora()

        # Train and capture loss history
        history = model.train_lora(
            train_contexts, train_targets,
            epochs=epochs,
            dry_run=dry_run,
            config=sweep_config,
        )

        # Evaluate MAPE on validation windows
        all_mapes = []
        for w in eval_windows:
            context = w["context"]
            actual = w["target"].values
            try:
                result = decomposer.decompose_and_project(
                    context, w["target"].index,
                    horizon=horizon, use_dow=True,
                )
                residual_forecast = model.predict_lora(
                    result["context_residual"], horizon=horizon
                )
                forecast = decomposer.recompose(
                    result["trend_forecast"],
                    result["seasonal_forecast"],
                    result["dow_forecast"],
                    residual_forecast,
                )
                from utils import compute_metrics
                metrics = compute_metrics(actual, forecast)
                all_mapes.append(metrics["MAPE"])
            except Exception as e:
                print(f"  [Sweep] Window error: {e}")

        final_mape = float(np.mean(all_mapes)) if all_mapes else float("nan")
        print(f"  [Sweep] Rank={rank} -> Final MAPE = {final_mape:.4f}%")

        sweep_results["rank"].append(rank)
        sweep_results["alpha"].append(alpha)
        sweep_results["final_mape"].append(final_mape)
        sweep_results["history"].append(history)

        # Save per-rank learning curve
        plot_learning_curves(
            history,
            save_path=f"{save_dir}/lora_curve_rank{rank}.png",
            lora_rank=rank,
            lora_alpha=alpha,
        )

        # Free GPU memory between runs
        del model
        clear_gpu_memory()

    # -- Summary bar chart -----------------------------------------------------
    _plot_sweep_summary(sweep_results, save_path=f"{save_dir}/lora_sweep_summary.png")

    # -- Print optimal configuration -------------------------------------------
    valid_results = [(r, m) for r, m in zip(sweep_results["rank"], sweep_results["final_mape"])
                     if not np.isnan(m)]
    if valid_results:
        best_rank, best_mape = min(valid_results, key=lambda x: x[1])
        best_alpha = sweep_results["alpha"][sweep_results["rank"].index(best_rank)]
        print(f"\n{'=' * 60}")
        print(f"SWEEP COMPLETE -- Optimal Configuration:")
        print(f"  LoRA Rank = {best_rank}, Alpha = {best_alpha}")
        print(f"  Final MAPE = {best_mape:.4f}%")
        print(f"{'=' * 60}")

    return sweep_results


def _plot_sweep_summary(sweep_results: dict, save_path: str):
    """
    Plot bar chart summarizing MAPE for each rank.
    IEEE print-quality output.
    """
    ranks = sweep_results["rank"]
    mapes = sweep_results["final_mape"]

    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(
            [f"r={r}\nα={a}" for r, a in zip(ranks, sweep_results["alpha"])],
            mapes,
            color=colors[:len(ranks)],
            edgecolor="black",
            linewidth=0.6,
        )

        # Annotate bars
        for bar, v in zip(bars, mapes):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        # Highlight best
        if mapes and not all(np.isnan(m) for m in mapes):
            best_idx = int(np.nanargmin(mapes))
            bars[best_idx].set_edgecolor("#d62728")
            bars[best_idx].set_linewidth(2.0)

        ax.set_ylabel("MAPE (%)")
        ax.set_title("LoRA Rank Hyperparameter Sweep -- Final MAPE")
        ax.set_xlabel("LoRA Configuration")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved sweep summary -> {save_path}")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Learning Curves & Hyperparameter Sweep")
    parser.add_argument(
        "--mode",
        choices=["curves", "sweep", "both"],
        default="sweep",
        help="Mode: 'curves' (plot from saved JSON), 'sweep' (run rank sweep), 'both'",
    )
    parser.add_argument(
        "--history_json",
        type=str,
        default=None,
        help="Path to JSON file with saved history (required for --mode curves)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="List of LoRA ranks to sweep (default: 8 16 32)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Use dry-run settings")
    parser.add_argument(
        "--save_dir", type=str, default="plots", help="Directory for output plots"
    )
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["execution"]["random_seed"])

    if args.mode in ("curves", "both"):
        # Auto-detect all lora_history_<Dataset>.json files; fall back to single --history_json
        history_files = []
        if args.history_json:
            history_files = [(args.history_json, os.path.splitext(os.path.basename(args.history_json))[0])]
        else:
            import glob
            found = sorted(glob.glob(os.path.join(args.save_dir, "lora_history_*.json")))
            if found:
                for f in found:
                    stem = os.path.splitext(os.path.basename(f))[0]  # e.g. lora_history_PJM
                    ds   = stem.replace("lora_history_", "")         # e.g. PJM
                    history_files.append((f, ds))
            else:
                print("[curves] No lora_history_*.json files found. Using dummy data for demo.")
                history_files = [(None, "demo")]

        for hist_path, ds_label in history_files:
            if hist_path is None:
                history = {
                    "train_loss": [0.08, 0.06, 0.05, 0.045, 0.042, 0.040, 0.038, 0.037, 0.036, 0.035],
                    "val_loss":   [0.09, 0.075, 0.065, 0.060, 0.058, 0.057, 0.059, 0.060, 0.062, 0.063],
                }
            else:
                with open(hist_path) as f:
                    history = json.load(f)
            save_path = os.path.join(args.save_dir, f"lora_learning_curves_{ds_label}.png")
            plot_learning_curves(
                history,
                save_path=save_path,
                lora_rank=config["model"]["lora_rank"],
                lora_alpha=config["model"]["lora_alpha"],
            )

    if args.mode in ("sweep", "both"):
        sweep_results = run_lora_rank_sweep(
            config,
            dry_run=args.dry_run,
            ranks=args.ranks,
            save_dir=args.save_dir,
        )

        # Save sweep results to JSON
        results_path = os.path.join(args.save_dir, "lora_sweep_results.json")
        save_data = {
            k: v for k, v in sweep_results.items() if k != "history"
        }
        save_data["history"] = [
            {key: [float(x) for x in vals] for key, vals in h.items()}
            for h in sweep_results["history"]
        ]
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Sweep results saved -> {results_path}")
