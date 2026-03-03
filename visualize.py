import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import os

from utils import load_config


# --- Short labels for plots ----------------------------------------------------
MODEL_ALIASES = {
    "SeasonalNaive":                            "SeasonalNaive",
    "AutoETS":                                  "AutoETS",
    "XGBoost":                                  "XGBoost",
    "Raw Chronos (Zero-shot)":                  "Raw Chronos",
    "STL + Chronos (Zero-shot)":                "STL+Chronos",
    "STL + DoW + Chronos (Zero-shot)":          "STL+DoW+Chronos",
    "STL + DoW + Chronos + LoRA":               "STL+DoW+Chronos+LoRA",
    "Raw CHRONOS-BOLT-SMALL (Zero-shot)":       "Raw Bolt-Small",
    "STL + DoW + CHRONOS-BOLT-SMALL (Zero-shot)": "STL+DoW+Bolt-Small",
    # Lag-Llama (decoder-only, non-Amazon)
    "Raw LAG-LLAMA (Zero-shot)":                "Raw Lag-Llama",
    "STL + DoW + LAG-LLAMA (Zero-shot)":        "STL+DoW+Lag-Llama",
}

# Color per model family
def _color(name: str) -> str:
    n_up = name.upper()
    if "LORA" in n_up:    return "#2196F3"   # blue
    if "BOLT" in n_up:    return "#9C27B0"   # purple
    if "LAG-LLAMA" in n_up or "LAG_LLAMA" in n_up:
                          return "#00BCD4"   # teal (non-Amazon model)
    if "Chronos" in name or "CHRONOS" in n_up:
                          return "#4CAF50"   # green
    if "XGBoost" in name: return "#FF9800"   # orange
    return "#9E9E9E"                          # grey


def _short(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def _bar_chart(ax, names, means, stds, ylabel, title):
    colors = [_color(n) for n in names]
    short  = [_short(n) for n in names]
    x = range(len(names))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def plot_per_dataset(results_df, ablation_df, output_dir, dataset_name):
    """Generate ablation bar chart, box plots, and ranked chart for one dataset."""
    abl = ablation_df[ablation_df["Dataset"] == dataset_name].copy()
    res = results_df[results_df["Dataset"] == dataset_name].copy()
    if abl.empty:
        print(f"  [WARN] No data for {dataset_name} in ablation.csv -- skipping")
        return

    names = abl["Model"].tolist()

    # -- Decide primary metric based on dataset --------------------------------
    # ETTm1 OT is near-zero -> MAPE is unreliable; use sMAPE (if available) or
    # fall back to MAE as the "accuracy" axis.
    has_smape = "sMAPE_mean" in abl.columns and not abl["sMAPE_mean"].isna().all()
    if dataset_name == "ETTm1":
        if has_smape:
            acc_col, acc_std, acc_label = "sMAPE_mean", "sMAPE_std", "sMAPE (%)"
        else:
            acc_col, acc_std, acc_label = "MAE_mean", "MAE_std", "MAE"
    else:
        acc_col, acc_std, acc_label = "MAPE_mean", "MAPE_std", "MAPE (%)"

    # -- 1. Three-panel ablation bar chart -------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Ablation Study -- {dataset_name}", fontsize=14, fontweight="bold")

    _bar_chart(axes[0], names,
               abl["MAE_mean"].values, abl["MAE_std"].values,
               "MAE", "MAE by Model")
    _bar_chart(axes[1], names,
               abl["RMSE_mean"].values, abl["RMSE_std"].values,
               "RMSE", "RMSE by Model")
    _bar_chart(axes[2], names,
               abl[acc_col].values, abl[acc_std].values,
               acc_label, f"{acc_label.split(' ')[0]} by Model")

    plt.tight_layout()
    fp = os.path.join(output_dir, f"ablation_bar_{dataset_name}.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")

    # -- 2. Error distribution box plots -------------------------------------
    metrics_to_box = (["MAE", "RMSE", acc_col.replace("_mean", "")]
                      if acc_col != "MAE_mean" else ["MAE", "RMSE"])
    n_panels = len(metrics_to_box)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    fig.suptitle(f"Error Distributions -- {dataset_name}", fontsize=14, fontweight="bold")
    if n_panels == 1:
        axes = [axes]

    model_order = names
    for ax, metric in zip(axes, metrics_to_box):
        if metric not in res.columns:
            continue
        data = [res[res["Model"] == m][metric].dropna().values for m in model_order]
        bp = ax.boxplot(data, tick_labels=[_short(m) for m in model_order],
                        patch_artist=True, notch=False)
        colors = [_color(m) for m in model_order]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_xticklabels([_short(m) for m in model_order],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"{metric} Distribution", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(output_dir, f"error_distribution_{dataset_name}.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")

    # -- 3. Ranked horizontal bar chart ---------------------------------------
    sorted_abl = abl.sort_values(acc_col)
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = sorted_abl[acc_col].values
    errs = sorted_abl[acc_std].values
    clrs = ["#4CAF50" if v == vals.min()
            else "#FFC107" if v < np.median(vals)
            else "#F44336"
            for v in vals]
    ax.barh(range(len(sorted_abl)), vals, xerr=errs, capsize=3,
            color=clrs, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(len(sorted_abl)))
    ax.set_yticklabels([_short(n) for n in sorted_abl["Model"].values], fontsize=9)
    ax.set_xlabel(acc_label, fontsize=11)
    ax.set_title(f"Models Ranked by {acc_label.split(' ')[0]} -- {dataset_name}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(output_dir, f"ranking_{dataset_name}.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")


def plot_cross_dataset_mae(ablation_df, output_dir):
    """
    Grouped bar chart comparing MAE across datasets for every model.
    MAE is on a comparable absolute scale within each dataset (not across datasets),
    so we normalise by each dataset's best-model MAE to get a relative scale.
    """
    abl = ablation_df.copy()
    datasets = abl["Dataset"].unique()
    if len(datasets) < 2:
        return

    # Normalise MAE per dataset: divide by minimum MAE in that dataset
    abl["MAE_norm"] = abl.groupby("Dataset")["MAE_mean"].transform(
        lambda x: x / x.min()
    )

    models = abl["Model"].unique()
    x = np.arange(len(models))
    width = 0.8 / len(datasets)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, ds in enumerate(datasets):
        ds_vals = []
        for m in models:
            row = abl[(abl["Dataset"] == ds) & (abl["Model"] == m)]
            ds_vals.append(row["MAE_norm"].values[0] if not row.empty else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, ds_vals, width,
               label=ds, alpha=0.8, edgecolor="black", linewidth=0.4)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Best model = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in models], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalised MAE  (1.0 = best per dataset)", fontsize=10)
    ax.set_title("Cross-Dataset Comparison (Normalised MAE)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(output_dir, "cross_dataset_comparison.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")


def plot_forecast_vs_actual(results_csv: str = "results.csv",
                             ablation_csv: str = "ablation.csv",
                             output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    results_df  = pd.read_csv(results_csv)
    ablation_df = pd.read_csv(ablation_csv)

    datasets = ablation_df["Dataset"].unique() if "Dataset" in ablation_df.columns else ["PJM"]

    # Per-dataset plots
    for ds in datasets:
        print(f"\n  --- {ds} ---")
        plot_per_dataset(results_df, ablation_df, output_dir, ds)

    # Cross-dataset comparison
    if len(datasets) > 1:
        print("\n  --- Cross-dataset ---")
        plot_cross_dataset_mae(ablation_df, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--results",    type=str, default="results.csv")
    parser.add_argument("--ablation",   type=str, default="ablation.csv")
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found. Run evaluate.py first.")
        return

    plot_forecast_vs_actual(args.results, args.ablation, args.output_dir)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
