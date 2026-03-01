import argparse
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

from utils import load_config, set_global_seed, compute_metrics, clear_gpu_memory
from data_loader import load_data, split_train_test, create_rolling_windows
from decomposition import STLDecomposer
from models.chronos_lora import ChronosLoRAModel
from models.xgboost_baseline import XGBoostBaseline
from models.statsforecast_baselines import StatsforecastBaselines
from train import train_pipeline


def evaluate_ablation(windows, decomposer, chronos_model, config, use_stl, use_dow, use_lora):
    """
    Evaluate a single ablation variant across all windows.

    Variants:
    1. Raw Chronos (Zero-shot): use_stl=False, use_dow=False, use_lora=False
    2. STL + Chronos (Zero-shot on residuals, NO DoW): use_stl=True, use_dow=False, use_lora=False
    3. STL + DoW + Chronos (Zero-shot on residuals): use_stl=True, use_dow=True, use_lora=False
    4. STL + DoW + Chronos + LoRA: use_stl=True, use_dow=True, use_lora=True
    """
    horizon = config["evaluation"]["horizon"]
    all_metrics = []
    all_forecasts = []

    for i, w in enumerate(tqdm(windows, desc="    Windows", unit="win", leave=False)):
        context = w["context"]
        actual = w["target"].values

        if not use_stl:
            # Raw Chronos: predict directly on raw context
            if use_lora:
                forecast = chronos_model.predict_lora(context.values, horizon=horizon)
            else:
                forecast = chronos_model.predict_zero_shot(context.values, horizon=horizon)
        else:
            # STL decomposition
            result = decomposer.decompose_and_project(
                context, w["target"].index, horizon=horizon, use_dow=use_dow
            )

            # Predict residual with Chronos
            if use_lora:
                residual_forecast = chronos_model.predict_lora(
                    result["context_residual"], horizon=horizon
                )
            else:
                residual_forecast = chronos_model.predict_zero_shot(
                    result["context_residual"], horizon=horizon
                )

            # Recompose
            dow_forecast = result["dow_forecast"] if use_dow else np.zeros(horizon)
            forecast = decomposer.recompose(
                result["trend_forecast"],
                result["seasonal_forecast"],
                dow_forecast,
                residual_forecast,
            )

        metrics = compute_metrics(actual, forecast)
        all_metrics.append(metrics)
        all_forecasts.append(forecast)

    return all_metrics, all_forecasts


def run_evaluation(config: dict, dry_run: bool = False):
    """
    Full evaluation pipeline including baselines and ablation study.
    """
    seed = config["execution"]["random_seed"]
    set_global_seed(seed)

    if dry_run:
        config["dry_run"]["enabled"] = True
    is_dry_run = config["dry_run"]["enabled"] or dry_run

    print("=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)

    # Train all models first
    artifacts = train_pipeline(config, dry_run=is_dry_run)

    decomposer = artifacts["decomposer"]
    xgb_model = artifacts["xgb_model"]
    chronos_model = artifacts["chronos_model"]
    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]
    df_full = artifacts["df_full"]

    # Create rolling windows
    print("\n[EVAL] Creating rolling windows...")
    windows = create_rolling_windows(
        df_full, test_df,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"],
        dry_run=is_dry_run,
        max_windows=config["dry_run"]["max_windows"],
    )
    print(f"  Total rolling windows: {len(windows)}")

    if not windows:
        print("ERROR: No rolling windows created. Check data availability for test year.")
        return

    results = {}
    actuals = [w["target"].values for w in windows]

    # =========================================================
    # BASELINE MODELS
    # =========================================================
    print("\n" + "-" * 40)
    print("BASELINE MODELS")
    print("-" * 40)

    # 1. Statsforecast baselines (SeasonalNaive, AutoETS)
    print("\n[EVAL] Running Statsforecast baselines...")
    sf_baselines = StatsforecastBaselines(
        horizon=config["evaluation"]["horizon"],
        season_length=168,
        dry_run=is_dry_run,
        config=config,
    )
    sf_results = sf_baselines.forecast_all_windows(windows)

    for model_name in ["SeasonalNaive", "AutoETS"]:
        if model_name in sf_results and sf_results[model_name]:
            metrics_list = []
            for pred, actual in zip(sf_results[model_name], actuals):
                metrics_list.append(compute_metrics(actual, pred))
            results[model_name] = metrics_list
            print(f"  {model_name}: {len(metrics_list)} windows evaluated")

    # 2. XGBoost baseline
    print("\n[EVAL] Running XGBoost baseline...")
    xgb_forecasts = xgb_model.forecast_all_windows(windows)
    xgb_metrics = []
    for pred, actual in zip(xgb_forecasts, actuals):
        xgb_metrics.append(compute_metrics(actual, pred))
    results["XGBoost"] = xgb_metrics

    # =========================================================
    # ABLATION STUDY
    # =========================================================
    print("\n" + "-" * 40)
    print("ABLATION STUDY")
    print("-" * 40)

    ablation_configs = [
        ("Raw Chronos (Zero-shot)", False, False, False),
        ("STL + Chronos (Zero-shot)", True, False, False),
        ("STL + DoW + Chronos (Zero-shot)", True, True, False),
        ("STL + DoW + Chronos + LoRA", True, True, True),
    ]

    for name, use_stl, use_dow, use_lora in ablation_configs:
        print(f"\n[EVAL] {name}...")
        t0 = time.time()
        metrics_list, forecasts = evaluate_ablation(
            windows, decomposer, chronos_model, config,
            use_stl=use_stl, use_dow=use_dow, use_lora=use_lora,
        )
        elapsed = time.time() - t0
        results[name] = metrics_list
        print(f"  Completed in {elapsed:.1f}s, {len(metrics_list)} windows")
        clear_gpu_memory()

    # =========================================================
    # AGGREGATE AND EXPORT RESULTS
    # =========================================================
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)

    # results.csv: per-window metrics
    rows = []
    for model_name, metrics_list in results.items():
        for i, m in enumerate(metrics_list):
            rows.append({
                "Model": model_name,
                "Window": i,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "MAPE": m["MAPE"],
            })
    results_df = pd.DataFrame(rows)
    results_df.to_csv("results.csv", index=False)
    print(f"\n  Per-window results saved to results.csv ({len(rows)} rows)")

    # ablation.csv: mean and std across windows
    summary_rows = []
    for model_name, metrics_list in results.items():
        maes = [m["MAE"] for m in metrics_list]
        rmses = [m["RMSE"] for m in metrics_list]
        mapes = [m["MAPE"] for m in metrics_list]
        summary_rows.append({
            "Model": model_name,
            "MAE_mean": np.mean(maes),
            "MAE_std": np.std(maes),
            "RMSE_mean": np.mean(rmses),
            "RMSE_std": np.std(rmses),
            "MAPE_mean": np.mean(mapes),
            "MAPE_std": np.std(mapes),
            "N_windows": len(metrics_list),
        })

    ablation_df = pd.DataFrame(summary_rows)
    ablation_df.to_csv("ablation.csv", index=False)
    print(f"  Summary results saved to ablation.csv")

    # Print summary table
    print(f"\n{'Model':<35} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10}")
    print("-" * 70)
    for _, row in ablation_df.iterrows():
        print(f"{row['Model']:<35} {row['MAE_mean']:>8.1f}±{row['MAE_std']:<5.1f} "
              f"{row['RMSE_mean']:>8.1f}±{row['RMSE_std']:<5.1f} "
              f"{row['MAPE_mean']:>6.2f}±{row['MAPE_std']:<5.2f}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results, windows, actuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate forecasting models")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    run_evaluation(config, dry_run=args.dry_run)
