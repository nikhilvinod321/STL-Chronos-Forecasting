import argparse
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

from utils import load_config, set_global_seed, clear_gpu_memory
from data_loader import load_data, split_train_test, create_rolling_windows
from decomposition import STLDecomposer
from models.chronos_lora import ChronosLoRAModel
from models.xgboost_baseline import XGBoostBaseline


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
                # Target residual = actual target - projected components
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


def train_pipeline(config: dict, dry_run: bool = False):
    """
    Main training pipeline.
    1. Load data and split
    2. Fit STL decomposer (DoW adjustments)
    3. Train XGBoost baseline
    4. Load Chronos, optionally apply LoRA and fine-tune
    """
    seed = config["execution"]["random_seed"]
    set_global_seed(seed)

    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    # Override dry_run from config if not explicitly set
    if dry_run:
        config["dry_run"]["enabled"] = True

    is_dry_run = config["dry_run"]["enabled"] or dry_run

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_data(config)
    train_df, test_df = split_train_test(df, config)
    print(f"  Full dataset: {len(df)} rows")
    print(f"  Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Test: {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

    # 2. Fit STL Decomposer (DoW adjustments from training data only)
    print("\n[2/5] Fitting STL Decomposer (DoW adjustments)...")
    decomposer = STLDecomposer(period=168, robust=True)
    dow_adj = decomposer.fit_dow_adjustments(train_df)
    print(f"  DoW adjustments (Mon-Sun): {[f'{v:.1f}' for v in [dow_adj[i] for i in range(7)]]}")

    # 3. Train XGBoost baseline
    print("\n[3/5] Training XGBoost baseline...")
    xgb_model = XGBoostBaseline(
        horizon=config["evaluation"]["horizon"],
        random_seed=seed,
    )
    t0 = time.time()
    xgb_model.fit(train_df["PJME_MW"].values)
    print(f"  XGBoost trained in {time.time() - t0:.1f}s")

    # 4. Load Chronos (zero-shot doesn't need training)
    print("\n[4/5] Loading Chronos pipeline...")
    chronos_model = ChronosLoRAModel(config)
    chronos_model.load_pipeline()
    print("  Chronos pipeline loaded.")

    # 5. LoRA fine-tuning
    print("\n[5/5] LoRA fine-tuning...")
    chronos_model.apply_lora()

    # Prepare training data for LoRA
    print("  Preparing LoRA training data (decomposing training windows)...")
    lora_contexts, lora_targets = prepare_lora_training_data(
        train_df, decomposer,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"] * 4,  # Larger stride for training efficiency
    )

    if lora_contexts:
        t0 = time.time()
        chronos_model.train_lora(
            lora_contexts, lora_targets,
            epochs=config["training"]["epochs"],
            dry_run=is_dry_run,
            config=config,
        )
        print(f"  LoRA fine-tuning completed in {time.time() - t0:.1f}s")
    else:
        print("  WARNING: No training data prepared for LoRA. Skipping fine-tuning.")

    clear_gpu_memory()

    # Save artifacts
    artifacts = {
        "decomposer": decomposer,
        "xgb_model": xgb_model,
        "chronos_model": chronos_model,
        "config": config,
        "train_df": train_df,
        "test_df": test_df,
        "df_full": df,
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = train_pipeline(config, dry_run=args.dry_run)
