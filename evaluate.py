import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

from utils import load_config, set_global_seed, compute_metrics, clear_gpu_memory
from data_loader import create_rolling_windows
from decomposition import STLDecomposer
from models.chronos_lora import ChronosLoRAModel
from models.xgboost_baseline import XGBoostBaseline
from models.statsforecast_baselines import StatsforecastBaselines
from train import train_pipeline, DATASET_META


# -----------------------------------------------------------------------------
# Progress helper
# -----------------------------------------------------------------------------

_EVAL_START = time.time()

def _stage(msg: str):
    """Print a timestamped stage banner for easy progress tracking."""
    elapsed = time.time() - _EVAL_START
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    ts = f"{h:02d}:{m:02d}:{s:02d}"
    print(f"\n[{ts}] >>> {msg}", flush=True)


def _done(model_num: int, n_models: int, name: str, metrics_list: list, elapsed: float):
    """Print a one-liner after each model completes."""
    if not metrics_list:
        return
    mapes = [m["MAPE"] for m in metrics_list]
    maes  = [m["MAE"]  for m in metrics_list]
    print(
        f"  [{model_num:>2}/{n_models}] ✓ {name:<42} "
        f"MAPE: {np.mean(mapes):5.2f}% +-{np.std(mapes):.2f}  "
        f"MAE: {np.mean(maes):>8,.0f}  "
        f"({elapsed:.0f}s)",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Per-model evaluation helpers
# -----------------------------------------------------------------------------

def evaluate_ablation(windows, decomposer, chronos_model, config, use_stl, use_dow, use_lora):
    """
    Evaluate one Chronos ablation variant across all windows.
    Returns (metrics_list, forecasts_list).
    """
    horizon = config["evaluation"]["horizon"]
    all_metrics, all_forecasts = [], []
    n = len(windows)

    for i, w in enumerate(windows):
        context = w["context"]
        actual  = w["target"].values

        if not use_stl:
            forecast = (chronos_model.predict_lora(context.values, horizon=horizon)
                        if use_lora
                        else chronos_model.predict_zero_shot(context.values, horizon=horizon))
        else:
            result = decomposer.decompose_and_project(
                context, w["target"].index, horizon=horizon, use_dow=use_dow
            )
            residual_forecast = (
                chronos_model.predict_lora(result["context_residual"], horizon=horizon)
                if use_lora
                else chronos_model.predict_zero_shot(result["context_residual"], horizon=horizon)
            )
            dow_fc = result["dow_forecast"] if use_dow else np.zeros(horizon)
            forecast = decomposer.recompose(
                result["trend_forecast"], result["seasonal_forecast"],
                dow_fc, residual_forecast,
            )

        m = compute_metrics(actual, forecast)
        all_metrics.append(m)
        all_forecasts.append(forecast)
        print(f"      win {i+1:>3}/{n} | MAPE={m['MAPE']:>5.2f}%  MAE={m['MAE']:>8,.0f}", flush=True)

    return all_metrics, all_forecasts


def evaluate_second_model(windows, decomposer, second_model, config,
                          use_stl: bool, use_dow: bool, model_label: str):
    """
    Evaluate a second foundation model (Lag-Llama / TimesFM) across all windows.
    Supports raw and STL+DoW variants.
    Returns (metrics_list, forecasts_list).
    """
    horizon = config["evaluation"]["horizon"]
    all_metrics, all_forecasts = [], []

    n = len(windows)
    for i, w in enumerate(windows):
        context = w["context"]
        actual  = w["target"].values
        try:
            if not use_stl:
                forecast = second_model.predict(context.values, horizon=horizon)
            else:
                result = decomposer.decompose_and_project(
                    context, w["target"].index, horizon=horizon, use_dow=use_dow
                )
                residual_forecast = second_model.predict(
                    result["context_residual"], horizon=horizon
                )
                dow_fc = result["dow_forecast"] if use_dow else np.zeros(horizon)
                forecast = decomposer.recompose(
                    result["trend_forecast"], result["seasonal_forecast"],
                    dow_fc, residual_forecast,
                )
            m = compute_metrics(actual, forecast)
            all_metrics.append(m)
            all_forecasts.append(forecast)
            print(f"      win {i+1:>3}/{n} | MAPE={m['MAPE']:>5.2f}%  MAE={m['MAE']:>8,.0f}", flush=True)
        except Exception as e:
            print(f"  [WARN] {model_label} window {i+1} skipped: {e}", flush=True)

    return all_metrics, all_forecasts


# -----------------------------------------------------------------------------
# Single-dataset evaluation
# -----------------------------------------------------------------------------

def evaluate_dataset(config: dict, dataset_name: str, is_dry_run: bool,
                     chronos_model: ChronosLoRAModel = None) -> tuple:
    """
    Train all models and run the full ablation study for one dataset.

    Returns:
        results      - {model_name: [metrics_dict, ...]}
        chronos_model  - the loaded/trained Chronos instance (for reuse)
        meta         - dataset metadata dict
    """
    # -- Train -----------------------------------------------------------------
    _stage(f"[{dataset_name}] STAGE 1/8 -- Training models (STL + XGBoost + LoRA)")
    artifacts = train_pipeline(
        config,
        dry_run=is_dry_run,
        dataset_name=dataset_name,
        chronos_model=chronos_model,
    )

    decomposer    = artifacts["decomposer"]
    xgb_model     = artifacts["xgb_model"]
    chronos_model = artifacts["chronos_model"]
    meta          = artifacts["dataset_meta"]
    test_df       = artifacts["test_df"]
    df_full       = artifacts["df_full"]

    # -- Rolling windows -------------------------------------------------------
    _stage(f"[{dataset_name}] STAGE 2/8 -- Building rolling evaluation windows")
    print(f"[EVAL:{dataset_name}] Creating rolling windows...")
    windows = create_rolling_windows(
        df_full, test_df,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"],
        dry_run=is_dry_run,
        max_windows=config["dry_run"]["max_windows"],
        freq_minutes=meta.get("freq_minutes", 60),
    )
    print(f"  Total rolling windows: {len(windows)}")

    if not windows:
        print(f"  ERROR: No windows for {dataset_name}. Skipping.")
        return {}, chronos_model, meta

    results = {}
    actuals = [w["target"].values for w in windows]

    # Work out total model count up-front for the counter
    second_model_name = config["model"].get("second_foundation_model")
    n_models = 7 + (2 if second_model_name else 0)   # 3 baselines + 4 Chronos + 0-2 second
    model_num = 0

    # -- Statsforecast baselines -----------------------------------------------
    _stage(f"[{dataset_name}] STAGE 3/{7 + (2 if second_model_name else 0)} -- SeasonalNaive + AutoETS")
    print(f"[EVAL:{dataset_name}] Statsforecast baselines ({len(windows)} windows)...")
    sf_baselines = StatsforecastBaselines(
        horizon=config["evaluation"]["horizon"],
        season_length=meta["sf_season"],
        dry_run=is_dry_run,
        config=config,
    )
    t0 = time.time()
    sf_results = sf_baselines.forecast_all_windows(windows)
    for name in ["SeasonalNaive", "AutoETS"]:
        if name in sf_results and sf_results[name]:
            model_num += 1
            ml = [compute_metrics(a, p) for p, a in zip(sf_results[name], actuals)]
            results[name] = ml
            _done(model_num, n_models, name, ml, time.time() - t0)

    # -- XGBoost ---------------------------------------------------------------
    _stage(f"[{dataset_name}] STAGE 4/{n_models} -- XGBoost baseline")
    t0 = time.time()
    xgb_preds = xgb_model.forecast_all_windows(windows)
    model_num += 1
    ml = [compute_metrics(a, p) for p, a in zip(xgb_preds, actuals)]
    results["XGBoost"] = ml
    _done(model_num, n_models, "XGBoost", ml, time.time() - t0)

    # -- Chronos ablation study ------------------------------------------------
    ablation_configs = [
        ("Raw Chronos (Zero-shot)",          False, False, False),
        ("STL + Chronos (Zero-shot)",         True,  False, False),
        ("STL + DoW + Chronos (Zero-shot)",   True,  True,  False),
        ("STL + DoW + Chronos + LoRA",        True,  True,  True),
    ]
    for i_abl, (name, use_stl, use_dow, use_lora) in enumerate(ablation_configs):
        model_num += 1
        _stage(f"[{dataset_name}] MODEL {model_num}/{n_models} -- {name}")
        t0 = time.time()
        metrics_list, _ = evaluate_ablation(
            windows, decomposer, chronos_model, config,
            use_stl=use_stl, use_dow=use_dow, use_lora=use_lora,
        )
        results[name] = metrics_list
        _done(model_num, n_models, name, metrics_list, time.time() - t0)
        clear_gpu_memory()

    # -- Second foundation model -----------------------------------------------
    if second_model_name:
        _stage(f"[{dataset_name}] MODEL {model_num+1}-{model_num+2}/{n_models} -- {second_model_name}")
        print(f"  Loading {second_model_name}...", flush=True)
        try:
            from models.foundation_models import get_foundation_model
            # Inject frequency string so Lag-Llama builds the correct GluonTS dataset
            config["_freq_str"] = meta.get("freq_str", "H")
            second_model = get_foundation_model(second_model_name, config)
            label = second_model_name.upper()

            # Raw zero-shot
            model_num += 1
            print(f"\n  [{model_num:>2}/{n_models}] ▶ Raw {label} (Zero-shot)...", flush=True)
            t0 = time.time()
            m, _ = evaluate_second_model(windows, decomposer, second_model, config,
                                         use_stl=False, use_dow=False,
                                         model_label=f"Raw {label}")
            results[f"Raw {label} (Zero-shot)"] = m
            _done(model_num, n_models, f"Raw {label} (Zero-shot)", m, time.time() - t0)

            # STL + DoW
            model_num += 1
            print(f"\n  [{model_num:>2}/{n_models}] ▶ STL + DoW + {label} (Zero-shot)...", flush=True)
            t0 = time.time()
            m, _ = evaluate_second_model(windows, decomposer, second_model, config,
                                         use_stl=True, use_dow=True,
                                         model_label=f"STL+DoW+{label}")
            results[f"STL + DoW + {label} (Zero-shot)"] = m
            _done(model_num, n_models, f"STL + DoW + {label} (Zero-shot)", m, time.time() - t0)

            del second_model
            clear_gpu_memory()

        except ImportError as e:
            print(f"  [SKIP] {second_model_name} not installed: {e}")
        except Exception as e:
            print(f"  [SKIP] {second_model_name} error: {e}")

    return results, chronos_model, meta


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------

def run_evaluation(config: dict, dry_run: bool = False):
    """
    Multi-dataset, multi-model evaluation pipeline.
    Loops over all datasets defined in config["data"]["datasets"].
    Chronos is loaded once and reused across datasets.
    Results saved to results.csv and ablation.csv with a Dataset column.
    """
    set_global_seed(config["execution"]["random_seed"])

    if dry_run:
        config["dry_run"]["enabled"] = True
    is_dry_run = config["dry_run"]["enabled"] or dry_run

    datasets = config["data"].get("datasets", ["PJM"])
    second_model_name = config["model"].get("second_foundation_model")
    n_models_per_ds   = 7 + (2 if second_model_name else 0)

    print("=" * 60)
    print(f"EVALUATION PIPELINE  --  Datasets: {datasets}")
    print("=" * 60)
    print(f"  Datasets  : {len(datasets)}  ({', '.join(datasets)})")
    print(f"  Models    : {n_models_per_ds} per dataset")
    print(f"  2nd model : {second_model_name or 'none'}")
    print(f"  Device    : {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")
    if dry_run:
        print(f"  Mode      : DRY RUN (max {config['dry_run']['max_windows']} windows)")
    print("=" * 60, flush=True)

    all_rows        = []   # per-window rows for results.csv
    all_summary     = []   # per-model summary rows for ablation.csv
    chronos_model   = None  # reused across datasets

    for dataset_name in datasets:
        print(f"\n{'#' * 60}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'#' * 60}")

        results, chronos_model, meta = evaluate_dataset(
            config, dataset_name, is_dry_run, chronos_model=chronos_model
        )

        if not results:
            continue

        # -- Per-window CSV rows -----------------------------------------------
        for model_name, metrics_list in results.items():
            for i, m in enumerate(metrics_list):
                all_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Window": i,
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "MAPE": m["MAPE"],
                    "sMAPE": m.get("sMAPE", float("nan")),
                })

        # -- Summary rows ------------------------------------------------------
        for model_name, metrics_list in results.items():
            if not metrics_list:
                continue
            maes   = [m["MAE"]   for m in metrics_list]
            rmses  = [m["RMSE"]  for m in metrics_list]
            mapes  = [m["MAPE"]  for m in metrics_list]
            smapes = [m.get("sMAPE", float("nan")) for m in metrics_list]
            all_summary.append({
                "Dataset":     dataset_name,
                "Model":       model_name,
                "MAE_mean":    np.mean(maes),
                "MAE_std":     np.std(maes),
                "RMSE_mean":   np.mean(rmses),
                "RMSE_std":    np.std(rmses),
                "MAPE_mean":   np.mean(mapes),
                "MAPE_std":    np.std(mapes),
                "sMAPE_mean":  np.nanmean(smapes),
                "sMAPE_std":   np.nanstd(smapes),
                "N_windows":   len(metrics_list),
            })

        # -- Per-dataset console summary ---------------------------------------
        print(f"\n{'-' * 80}")
        print(f"  RESULTS: {dataset_name}")
        print(f"{'-' * 80}")
        print(f"  {'Model':<38} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>12} {'sMAPE(%)':>12}")
        print(f"  {'-' * 78}")
        for row in all_summary:
            if row["Dataset"] != dataset_name:
                continue
            print(f"  {row['Model']:<38} "
                  f"{row['MAE_mean']:>7.1f}+-{row['MAE_std']:<5.1f} "
                  f"{row['RMSE_mean']:>7.1f}+-{row['RMSE_std']:<5.1f} "
                  f"{row['MAPE_mean']:>8.2f}+-{row['MAPE_std']:<6.2f} "
                  f"{row['sMAPE_mean']:>8.2f}+-{row['sMAPE_std']:<6.2f}")

    # -- Save combined CSVs ----------------------------------------------------
    if all_rows:
        results_df = pd.DataFrame(all_rows)
        results_df.to_csv("results.csv", index=False)
        print(f"\n  Per-window results  -> results.csv  ({len(all_rows)} rows)")

    if all_summary:
        ablation_df = pd.DataFrame(all_summary)
        ablation_df.to_csv("ablation.csv", index=False)
        print(f"  Summary results     -> ablation.csv ({len(all_summary)} rows)")

    # -- Multi-dataset cross-comparison ---------------------------------------
    if len(datasets) > 1 and all_summary:
        print(f"\n{'=' * 80}")
        print("  CROSS-DATASET SUMMARY  (sMAPE% -- bounded [0,200%], robust across all datasets)")
        print("  NOTE: ETTm1 MAPE is unreliable due to near-zero OT values; use sMAPE instead.")
        print(f"{'=' * 80}")
        abl = pd.DataFrame(all_summary)
        # Use sMAPE for all datasets (bounded 0-200%, robust to near-zero values)
        pivot_smape = abl.pivot_table(
            index="Model",
            columns="Dataset",
            values="sMAPE_mean",
            aggfunc="first",
        ).round(3)
        print("  sMAPE (%) -- lower is better:")
        print(pivot_smape.to_string())
        # Also show MAE for ETTm1 (absolute scale)
        pivot_mae = abl.pivot_table(
            index="Model",
            columns="Dataset",
            values="MAE_mean",
            aggfunc="first",
        ).round(4)
        print("\n  MAE -- lower is better:")
        print(pivot_mae.to_string())

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return all_rows, all_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate forecasting models")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset list: PJM, ETTm1, or PJM,ETTm1")
    args = parser.parse_args()

    config = load_config(args.config)

    # Allow CLI override of datasets
    if args.dataset:
        config["data"]["datasets"] = [d.strip() for d in args.dataset.split(",")]

    run_evaluation(config, dry_run=args.dry_run)

