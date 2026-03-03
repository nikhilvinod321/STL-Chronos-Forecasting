import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

from utils import load_config, set_global_seed, clear_gpu_memory
from data_loader import load_dataset, create_rolling_windows
from decomposition import STLDecomposer
from models.chronos_lora import ChronosLoRAModel
from models.xgboost_baseline import XGBoostBaseline


# -- Per-dataset structural constants -----------------------------------------
DATASET_META = {
    # dataset_name: (stl_period, xgb_weekly_lag, season_length_statsforecast, freq_minutes, freq_str)
    # PJM: 1-hourly  -> period=168 (1 week), weekly_lag=168, sf_season=168
    "PJM":   {"stl_period": 168, "xgb_seasonal_lag": 168, "sf_season": 168,
               "freq_minutes": 60,  "freq_str": "h"},
    # ETTm1: 15-min  -> period=96  (1 day),  weekly_lag=96,  sf_season=96
    "ETTm1": {"stl_period": 96,  "xgb_seasonal_lag": 96,  "sf_season": 96,
               "freq_minutes": 15,  "freq_str": "15min"},
    # UCI Electricity: 1-hourly, 321 Portuguese clients aggregated; same structure as PJM
    "UCI_Electricity": {"stl_period": 168, "xgb_seasonal_lag": 168, "sf_season": 168,
                        "freq_minutes": 60, "freq_str": "h"},
}


def prepare_lora_training_data(train_df: pd.DataFrame, decomposer: STLDecomposer,
                                context_length: int = 336, horizon: int = 48,
                                stride: int = 48):
    """
    Prepare (context_residual, target_residual) pairs from training data for LoRA fine-tuning.
    """
    contexts = []
    targets = []

    series = train_df["PJME_MW"]
    n = len(series)
    total_windows = max(0, (n - context_length - horizon) // stride + 1)

    start_idx = context_length
    pbar = tqdm(total=total_windows, desc="  Decomposing train windows", unit="win")
    while start_idx + horizon <= n:
        ctx_start = start_idx - context_length
        ctx_end = start_idx
        tgt_end = start_idx + horizon

        context = series.iloc[ctx_start:ctx_end]
        target = series.iloc[ctx_end:tgt_end]

        if len(context) == context_length and len(target) == horizon:
            try:
                result = decomposer.decompose_and_project(
                    context, target.index, horizon=horizon, use_dow=True
                )
                target_residual = (
                    target.values
                    - result["trend_forecast"]
                    - result["seasonal_forecast"]
                    - result["dow_forecast"]
                )
                contexts.append(result["context_residual"])
                targets.append(target_residual)
            except Exception as e:
                print(f"  Skipping training window at idx {start_idx}: {e}")

        start_idx += stride
        pbar.update(1)

    pbar.close()
    print(f"  Prepared {len(contexts)} training pairs for LoRA fine-tuning.")
    return contexts, targets


def train_pipeline(config: dict, dry_run: bool = False,
                   dataset_name: str = None,
                   chronos_model: ChronosLoRAModel = None):
    """
    Dataset-aware training pipeline.

    Args:
        config: experiment config dict
        dry_run: enables quick dry-run limits
        dataset_name: "PJM" or "ETTm1". Defaults to first in config["data"]["datasets"].
        chronos_model: pass an already-loaded ChronosLoRAModel to skip reloading
                       (useful when running multiple datasets back-to-back).

    Returns:
        artifacts dict containing all trained objects + metadata.
    """
    seed = config["execution"]["random_seed"]
    set_global_seed(seed)

    if dry_run:
        config["dry_run"]["enabled"] = True
    is_dry_run = config["dry_run"]["enabled"] or dry_run

    # Resolve dataset name
    if dataset_name is None:
        dataset_name = config["data"].get("datasets", ["PJM"])[0]
    meta = DATASET_META.get(dataset_name, DATASET_META["PJM"])

    _t0 = time.time()
    def _tstage(step, msg):
        elapsed = time.time() - _t0
        h, rem = divmod(int(elapsed), 3600)
        m, s   = divmod(rem, 60)
        print(f"[{h:02d}:{m:02d}:{s:02d}] TRAIN {step} | {msg}")

    print("=" * 60)
    print(f"TRAINING PIPELINE  --  Dataset: {dataset_name}")
    print("=" * 60)

    # -- Load dataset ----------------------------------------------------------
    _tstage("1/5", f"Loading {dataset_name} data")
    print(f"\n[1/5] Loading {dataset_name} data...")
    df, train_df, test_df, dataset_info = load_dataset(dataset_name, config)
    print(f"  Full dataset : {len(df)} rows  ({dataset_info['frequency']})")
    print(f"  Train        : {len(train_df)} rows  "
          f"({train_df.index.min()} -> {train_df.index.max()})")
    print(f"  Test         : {len(test_df)} rows  "
          f"({test_df.index.min()} -> {test_df.index.max()})")

    # -- 2. Fit STL Decomposer -------------------------------------------------
    stl_period = meta["stl_period"]
    _tstage("2/5", f"Fitting STL Decomposer (period={stl_period})")
    print(f"\n[2/5] Fitting STL Decomposer (period={stl_period}, DoW adjustments)...")
    decomposer = STLDecomposer(period=stl_period, robust=True)
    dow_adj = decomposer.fit_dow_adjustments(train_df)
    print(f"  DoW adjustments (Mon-Sun): "
          f"{[f'{dow_adj.get(i, 0):.1f}' for i in range(7)]}")

    # -- 3. Train XGBoost baseline ---------------------------------------------
    xgb_lags = list(range(1, 25)) + [meta["xgb_seasonal_lag"]]
    _tstage("3/5", f"Training XGBoost (lags 1-24 + {meta['xgb_seasonal_lag']})")
    print(f"\n[3/5] Training XGBoost (lags 1-24 + {meta['xgb_seasonal_lag']})...")
    xgb_model = XGBoostBaseline(
        horizon=config["evaluation"]["horizon"],
        lags=xgb_lags,
        random_seed=seed,
    )
    t0 = time.time()
    xgb_model.fit(train_df["PJME_MW"].values)
    print(f"  XGBoost trained in {time.time() - t0:.1f}s")

    # -- 4. Load Chronos (reuse if already provided) ---------------------------
    _tstage("4/5", "Loading / reusing Chronos pipeline")
    if chronos_model is None:
        print("\n[4/5] Loading Chronos pipeline...")
        chronos_model = ChronosLoRAModel(config)
        chronos_model.load_pipeline()
        print("  Chronos pipeline loaded.")
    else:
        print("\n[4/5] Reusing existing Chronos pipeline.")

    # -- 5. LoRA fine-tuning ---------------------------------------------------
    _tstage("5/5", f"LoRA fine-tuning on {dataset_name}")
    print(f"\n[5/5] LoRA fine-tuning on {dataset_name}...")
    # Re-apply LoRA fresh for each dataset (resets trained weights)
    chronos_model.apply_lora()

    print("  Preparing LoRA training data (decomposing training windows)...")
    lora_contexts, lora_targets = prepare_lora_training_data(
        train_df, decomposer,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"] * 4,
    )

    lora_history = {"train_loss": [], "val_loss": []}
    if lora_contexts:
        t0 = time.time()
        lora_history = chronos_model.train_lora(
            lora_contexts, lora_targets,
            epochs=config["training"]["epochs"],
            dry_run=is_dry_run,
            config=config,
        )
        elapsed = time.time() - t0
        print(f"  LoRA fine-tuning completed in {elapsed:.1f}s")

        os.makedirs("plots", exist_ok=True)
        history_path = f"plots/lora_history_{dataset_name}.json"
        with open(history_path, "w") as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in lora_history.items()},
                f, indent=2,
            )
        print(f"  Loss history saved -> {history_path}")
    else:
        print("  WARNING: No training data prepared for LoRA. Skipping fine-tuning.")

    clear_gpu_memory()

    artifacts = {
        "dataset_name": dataset_name,
        "dataset_meta": meta,
        "decomposer": decomposer,
        "xgb_model": xgb_model,
        "chronos_model": chronos_model,
        "config": config,
        "train_df": train_df,
        "test_df": test_df,
        "df_full": df,
        "lora_history": lora_history,
    }

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  --  {dataset_name}")
    print("=" * 60)

    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to train on: PJM or ETTm1 (default: first in config)")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = train_pipeline(config, dry_run=args.dry_run, dataset_name=args.dataset)

