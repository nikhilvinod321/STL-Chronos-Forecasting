import pandas as pd
import numpy as np
from utils import load_config


def load_data(config: dict = None) -> pd.DataFrame:
    """
    Load PJM East hourly dataset, sort chronologically, set DatetimeIndex.
    Maps columns to Datetime and PJME_MW.
    """
    if config is None:
        config = load_config()

    path = config["data"]["path"]
    df = pd.read_csv(path)

    # Map columns
    df.columns = ["Datetime", "PJME_MW"]
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Sort chronologically and drop duplicates
    df = df.sort_values("Datetime").drop_duplicates(subset="Datetime").reset_index(drop=True)
    df = df.set_index("Datetime")

    # Ensure hourly frequency by reindexing
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = "Datetime"

    # Forward fill small gaps (if any), then drop remaining NaNs
    df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear", limit=3)
    df = df.dropna()

    return df


def split_train_test(df: pd.DataFrame, config: dict = None):
    """
    Split data into train (2015-2017) and test (2018) sets.
    """
    if config is None:
        config = load_config()

    train_years = config["data"]["train_years"]
    test_year = config["data"]["test_year"]

    train = df[df.index.year.isin(train_years)]
    test = df[df.index.year == test_year]

    return train, test


def create_rolling_windows(
    df_full: pd.DataFrame,
    test_df: pd.DataFrame,
    context_length: int = 336,
    horizon: int = 48,
    stride: int = 48,
    dry_run: bool = False,
    max_windows: int = 2,
):
    """
    Create rolling window pairs (context -> target) from the test set.
    Context comes from df_full (may overlap into training data).
    Targets are strictly non-overlapping.
    Prevents future data leakage.

    Returns list of dicts: {context: pd.Series, target: pd.Series,
                            context_start, context_end, target_start, target_end}
    """
    windows = []
    test_start = test_df.index.min()
    test_end = test_df.index.max()

    target_start = test_start
    count = 0

    while target_start + pd.Timedelta(hours=horizon - 1) <= test_end:
        target_end = target_start + pd.Timedelta(hours=horizon - 1)
        context_start = target_start - pd.Timedelta(hours=context_length)
        context_end = target_start - pd.Timedelta(hours=1)

        # Extract context and target from the full dataframe
        context = df_full.loc[context_start:context_end, "PJME_MW"]
        target = df_full.loc[target_start:target_end, "PJME_MW"]

        # Validate lengths
        if len(context) == context_length and len(target) == horizon:
            windows.append({
                "context": context,
                "target": target,
                "context_start": context_start,
                "context_end": context_end,
                "target_start": target_start,
                "target_end": target_end,
            })
            count += 1

        if dry_run and count >= max_windows:
            break

        target_start += pd.Timedelta(hours=stride)

    return windows


if __name__ == "__main__":
    config = load_config()
    df = load_data(config)
    print(f"Full dataset: {df.shape}, {df.index.min()} to {df.index.max()}")

    train, test = split_train_test(df, config)
    print(f"Train: {train.shape}, Test: {test.shape}")

    windows = create_rolling_windows(
        df, test,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"],
        dry_run=config["dry_run"]["enabled"],
        max_windows=config["dry_run"]["max_windows"],
    )
    print(f"Rolling windows: {len(windows)}")
    if windows:
        w = windows[0]
        print(f"  Window 0: context {w['context_start']} to {w['context_end']}, "
              f"target {w['target_start']} to {w['target_end']}")
