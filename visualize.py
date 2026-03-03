import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import os

from utils import load_config


def plot_forecast_vs_actual(results_csv: str = "results.csv",
                             ablation_csv: str = "ablation.csv",
                             output_dir: str = "plots"):
    """
    Generate all visualizations:
    1. Ablation bar chart (MAE, RMSE, MAPE)
    2. Error distribution box plots
    3. Per-model metric comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.read_csv(results_csv)
    ablation_df = pd.read_csv(ablation_csv)

    # =========================================================
    # 1. Ablation Bar Chart
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (metric, ylabel) in enumerate([
        ("MAE", "MAE (MW)"),
        ("RMSE", "RMSE (MW)"),
        ("MAPE", "MAPE (%)"),
    ]):
        ax = axes[idx]
        models = ablation_df["Model"].values
        means = ablation_df[f"{metric}_mean"].values
        stds = ablation_df[f"{metric}_std"].values

        colors = []
        for m in models:
            if "LoRA" in m:
                colors.append("#2196F3")
            elif "Chronos" in m:
                colors.append("#4CAF50")
            elif "XGBoost" in m:
                colors.append("#FF9800")
            else:
                colors.append("#9E9E9E")

        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{metric} by Model", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "ablation_bar_chart.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")

    # =========================================================
    # 2. Error Distribution (Box Plots)
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, metric in enumerate(["MAE", "RMSE", "MAPE"]):
        ax = axes[idx]
        model_names = results_df["Model"].unique()
        data = [results_df[results_df["Model"] == m][metric].values for m in model_names]

        bp = ax.boxplot(data, labels=model_names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#B3E5FC")
            patch.set_alpha(0.7)

        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} Distribution", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")

    # =========================================================
    # 3. Forecast vs Actual Plot (first window, best ablation variant)
    # =========================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    ablation_models = [m for m in ablation_df["Model"].values
                       if "Chronos" in m or "LoRA" in m]
    baseline_models = [m for m in ablation_df["Model"].values
                       if m not in ablation_models]

    x = np.arange(len(ablation_df))
    width = 0.25

    mae_vals = ablation_df["MAE_mean"].values
    rmse_vals = ablation_df["RMSE_mean"].values

    ax.bar(x - width/2, mae_vals, width, label="MAE", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, rmse_vals, width, label="RMSE", color="#FF5722", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["Model"].values, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error (MW)", fontsize=11)
    ax.set_title("MAE vs RMSE Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "mae_rmse_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")

    # =========================================================
    # 4. MAPE Heatmap-style bar (sorted)
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_df = ablation_df.sort_values("MAPE_mean")
    colors_sorted = ["#4CAF50" if v == sorted_df["MAPE_mean"].min()
                     else "#FFC107" if v < sorted_df["MAPE_mean"].median()
                     else "#F44336"
                     for v in sorted_df["MAPE_mean"].values]

    ax.barh(range(len(sorted_df)), sorted_df["MAPE_mean"].values,
            xerr=sorted_df["MAPE_std"].values, capsize=3,
            color=colors_sorted, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["Model"].values, fontsize=9)
    ax.set_xlabel("MAPE (%)", fontsize=11)
    ax.set_title("Models Ranked by MAPE", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "mape_ranking.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--results", type=str, default="results.csv")
    parser.add_argument("--ablation", type=str, default="ablation.csv")
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
