import pandas as pd
import numpy as np
import os
import io
import zipfile
import urllib.request
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
    freq_minutes: int = 60,
):
    """
    Create rolling window pairs (context -> target) from the test set.
    Context comes from df_full (may overlap into training data).
    Targets are strictly non-overlapping.
    Prevents future data leakage.

    freq_minutes: minutes per time step (60 for hourly PJM, 15 for 15-min ETTm1).
    All step counts (context_length, horizon, stride) are in *steps*, not hours.

    Returns list of dicts: {context: pd.Series, target: pd.Series,
                            context_start, context_end, target_start, target_end}
    """
    def _td(steps):
        return pd.Timedelta(minutes=freq_minutes * steps)

    windows = []
    test_start = test_df.index.min()
    test_end = test_df.index.max()

    target_start = test_start
    count = 0

    while target_start + _td(horizon - 1) <= test_end:
        target_end = target_start + _td(horizon - 1)
        context_start = target_start - _td(context_length)
        context_end = target_start - _td(1)

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

        target_start += _td(stride)

    return windows


def load_ettm1_data(config: dict = None) -> pd.DataFrame:
    """
    Load ETTm1 (Electricity Transformer Temperature) dataset.
    Downloads from official source if not present locally.
    Returns DataFrame with Datetime index and 'PJME_MW' column (renamed from OT for consistency).
    """
    if config is None:
        config = load_config()

    path = "data/ETTm1.csv"
    
    # Download if not exists
    if not os.path.exists(path):
        print(f"  Downloading ETTm1 data to {path}...")
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
        urllib.request.urlretrieve(url, path)
        print(f"  Downloaded ETTm1 dataset.")

    df = pd.read_csv(path)
    
    # ETTm1 has columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    # We use OT (Oil Temperature) as the target, rename to PJME_MW for consistency
    df.columns = ["Datetime"] + list(df.columns[1:-1]) + ["PJME_MW"]
    df = df[["Datetime", "PJME_MW"]]  # Keep only target variable
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Sort and ensure 15-minute frequency
    df = df.sort_values("Datetime").drop_duplicates(subset="Datetime").reset_index(drop=True)
    df = df.set_index("Datetime")

    # Ensure 15-minute frequency by reindexing
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="15min")
    df = df.reindex(full_idx)
    df.index.name = "Datetime"

    # Forward fill small gaps, then drop remaining NaNs
    df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear", limit=3)
    df = df.dropna()

    return df


def split_train_test_by_ratio(df: pd.DataFrame, train_ratio: float = 0.6) -> tuple:
    """
    Split data by ratio (for datasets without explicit train/test years).
    Useful for ETTm1 and other datasets without yearly structure.
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def _parse_tsf(fileobj) -> list:
    """
    Parse a Monash TSF-format file object.
    Returns list of (series_name, start_str, np.ndarray) tuples.
    """
    series = []
    in_data = False
    for raw in fileobj:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if in_data and ":" in line:
            parts = line.split(":")
            if len(parts) < 3:
                continue
            name = parts[0]
            start_str = parts[1]
            values_str = parts[2]
            try:
                values = np.array([float(v) for v in values_str.split(",")
                                   if v.strip() not in ("", "NaN", "nan", "?")])
                series.append((name, start_str, values))
            except ValueError:
                pass
    return series


def load_uci_electricity_data(config: dict = None) -> pd.DataFrame:
    """
    Load UCI Electricity (Monash hourly edition, 321 Portuguese clients 2011-2014).
    Source: Monash Forecasting Repository – Zenodo record 4656140.

    Aggregates all 321 client series into one total-demand series so the
    task matches the univariate grid-load structure of PJM.  The result is
    stored in PJME_MW for pipeline consistency.

    Falls back to a local CSV ('data/UCI_electricity.csv') if the download
    fails – put a file there with columns ['Datetime','PJME_MW'].
    """
    if config is None:
        config = load_config()

    tsf_path  = "data/electricity_hourly_dataset.tsf"
    csv_cache = "data/UCI_electricity.csv"

    # ------------------------------------------------------------------ load -
    if os.path.exists(csv_cache):
        print(f"  Loading UCI Electricity from cache: {csv_cache}")
        df = pd.read_csv(csv_cache)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        df = df.reindex(full_idx)
        df.index.name = "Datetime"
        df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear", limit=3)
        return df.dropna()

    # Download TSF if missing
    if not os.path.exists(tsf_path):
        url = ("https://zenodo.org/record/4656140/files/"
               "electricity_hourly_dataset.zip")
        print(f"  Downloading UCI Electricity dataset from Zenodo...")
        try:
            os.makedirs("data", exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                raw_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                tsf_name = next(n for n in zf.namelist() if n.endswith(".tsf"))
                with zf.open(tsf_name) as f_in, open(tsf_path, "wb") as f_out:
                    f_out.write(f_in.read())
            print(f"  Saved: {tsf_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not download UCI Electricity dataset: {exc}\n"
                "Manual fix: place 'data/UCI_electricity.csv' with columns "
                "['Datetime','PJME_MW'] (hourly, no missing values)."
            ) from exc

    # Parse TSF
    print(f"  Parsing {tsf_path} ...")
    with open(tsf_path, "r", encoding="utf-8", errors="replace") as f:
        all_series = _parse_tsf(f)

    if not all_series:
        raise ValueError("TSF parsing yielded no series. Check the file format.")

    # Build a common hourly DatetimeIndex spanning 2011-2014
    # The start dates in the TSF use the format "1990-01-01 00-00-00" with
    # dashes instead of colons in the time part.
    def _parse_start(s: str) -> pd.Timestamp:
        parts = s.strip().split(" ")
        if len(parts) == 2:
            # date_part  time_part (dashes instead of colons)
            s = parts[0] + " " + parts[1].replace("-", ":")
        return pd.to_datetime(s, errors="coerce")

    # Find global date range across all series
    starts, lengths = [], []
    for _, start_str, vals in all_series:
        ts = _parse_start(start_str)
        if pd.isna(ts):
            continue
        starts.append(ts)
        lengths.append(len(vals))

    global_start = min(starts)
    max_len = max(lengths)
    global_end = global_start + pd.Timedelta(hours=max_len - 1)
    full_idx = pd.date_range(start=global_start, end=global_end, freq="h")

    # Align every series onto the common index and accumulate
    agg = np.zeros(len(full_idx), dtype=np.float64)
    n_valid = 0
    for _, start_str, vals in all_series:
        ts = _parse_start(start_str)
        if pd.isna(ts):
            continue
        offset = int((ts - global_start).total_seconds() // 3600)
        end_pos = min(offset + len(vals), len(agg))
        frag = vals[:end_pos - offset]
        if len(frag) > 0:
            agg[offset:end_pos] += frag
            n_valid += 1

    print(f"  Aggregated {n_valid} client series into total-demand signal.")

    df = pd.DataFrame({"PJME_MW": agg}, index=full_idx)
    df.index.name = "Datetime"
    df = df[df["PJME_MW"] > 0]          # drop padding zeros at edges

    # Interpolate and trim
    df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear", limit=3)
    df = df.dropna()

    # Cache as CSV for fast reloads
    df.reset_index().to_csv(csv_cache, index=False)
    print(f"  Cached to {csv_cache} ({len(df):,} hourly rows).")
    return df


def load_dataset(dataset_name: str, config: dict = None) -> tuple:
    """
    Unified loader for multiple datasets.
    Returns: (full_df, train_df, test_df, dataset_info)
    """
    if config is None:
        config = load_config()

    if dataset_name.lower() == "pjm":
        df = load_data(config)
        train, test = split_train_test(df, config)
        dataset_info = {"name": "PJM", "frequency": "1h", "location": "Eastern US Power Grid"}
    elif dataset_name.lower() == "ettm1":
        df = load_ettm1_data(config)
        train, test = split_train_test_by_ratio(df, train_ratio=0.6)
        dataset_info = {"name": "ETTm1", "frequency": "15min", "location": "Transformer Temperature"}
    elif dataset_name.lower() in ("uci_electricity", "ucielectricity", "electricity"):
        df = load_uci_electricity_data(config)
        train, test = split_train_test_by_ratio(df, train_ratio=0.6)
        dataset_info = {"name": "UCI_Electricity", "frequency": "1h",
                        "location": "Portuguese Electricity Grid (321 clients aggregated)"}
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Choose 'PJM', 'ETTm1', or 'UCI_Electricity'."
        )

    return df, train, test, dataset_info


if __name__ == "__main__":
    import os
    config = load_config()
    
    # Test PJM
    print("Loading PJM dataset...")
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
    
    # Test ETTm1
    print("\nLoading ETTm1 dataset...")
    df_ettm1, train_ettm1, test_ettm1, info = load_dataset("ETTm1", config)
    print(f"ETTm1 Full dataset: {df_ettm1.shape}, {df_ettm1.index.min()} to {df_ettm1.index.max()}")
    print(f"ETTm1 Train: {train_ettm1.shape}, Test: {test_ettm1.shape}")
    print(f"Dataset info: {info}")
