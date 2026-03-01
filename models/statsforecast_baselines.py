import numpy as np
import pandas as pd
from tqdm import tqdm
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS
import os


class StatsforecastBaselines:
    """
    Classical baselines using statsforecast: SeasonalNaive and AutoETS.
    """

    def __init__(self, horizon: int = 48, season_length: int = 168,
                 dry_run: bool = False, config: dict = None):
        self.horizon = horizon
        self.season_length = season_length
        self.dry_run = dry_run
        self.config = config or {}

    def _build_models(self):
        """Build list of statsforecast models."""
        models = [SeasonalNaive(season_length=self.season_length)]

        if self.dry_run:
            dry_cfg = self.config.get("dry_run", {})
            seasonal_periods = dry_cfg.get("ets_seasonal_periods", [168])
            models.append(AutoETS(
                season_length=seasonal_periods[0],
            ))
        else:
            models.append(AutoETS(season_length=self.season_length))

        return models

    def forecast(self, context: pd.Series) -> dict:
        """
        Run SeasonalNaive and AutoETS on a single context window.
        Returns dict with model names as keys, forecasts as values.
        Also returns ETS model notation.
        """
        # Prepare data in statsforecast format
        df_sf = pd.DataFrame({
            "unique_id": ["series"] * len(context),
            "ds": context.index,
            "y": context.values.astype(np.float64),
        })

        models = self._build_models()

        sf = StatsForecast(
            models=models,
            freq="h",
            n_jobs=1,
        )

        sf.fit(df_sf)
        forecasts_df = sf.predict(h=self.horizon)

        results = {}
        ets_notation = "unknown"

        # Extract SeasonalNaive forecast
        snaive_col = [c for c in forecasts_df.columns if "SeasonalNaive" in c]
        if snaive_col:
            results["SeasonalNaive"] = forecasts_df[snaive_col[0]].values

        # Extract AutoETS forecast and model notation
        ets_col = [c for c in forecasts_df.columns if "AutoETS" in c]
        if ets_col:
            results["AutoETS"] = forecasts_df[ets_col[0]].values

        # Try to extract ETS notation
        for m in sf.fitted_:
            for fitted_model_list in m:
                if hasattr(fitted_model_list, "model_"):
                    ets_notation = str(fitted_model_list.model_)
                    break

        # Alternative: try from models directly
        try:
            for model in models:
                if hasattr(model, "model_"):
                    ets_notation = str(model.model_)
        except Exception:
            pass

        results["ets_notation"] = ets_notation

        return results

    def forecast_all_windows(self, windows: list, log_path: str = "baseline_configs.txt") -> dict:
        """
        Run forecasts on all rolling windows.
        Returns dict of model_name -> list of forecasts.
        Logs ETS notation to file.
        """
        all_results = {"SeasonalNaive": [], "AutoETS": [], "ets_notations": []}

        for i, w in enumerate(tqdm(windows, desc="  Statsforecast baselines", unit="win")):
            res = self.forecast(w["context"])

            if "SeasonalNaive" in res:
                all_results["SeasonalNaive"].append(res["SeasonalNaive"])
            if "AutoETS" in res:
                all_results["AutoETS"].append(res["AutoETS"])

            all_results["ets_notations"].append(res.get("ets_notation", "unknown"))

        # Log ETS notations
        with open(log_path, "a") as f:
            f.write("=== ETS Model Notations ===\n")
            for i, notation in enumerate(all_results["ets_notations"]):
                f.write(f"Window {i}: {notation}\n")

        return all_results


if __name__ == "__main__":
    from data_loader import load_data, split_train_test, create_rolling_windows
    from utils import load_config, set_global_seed

    config = load_config()
    set_global_seed(config["execution"]["random_seed"])
    df = load_data(config)
    train, test = split_train_test(df, config)

    windows = create_rolling_windows(
        df, test,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"],
        dry_run=True,
        max_windows=2,
    )

    baselines = StatsforecastBaselines(
        horizon=config["evaluation"]["horizon"],
        season_length=168,
        dry_run=True,
        config=config,
    )

    if windows:
        res = baselines.forecast(windows[0]["context"])
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: shape={v.shape}, mean={v.mean():.1f}")
            else:
                print(f"{k}: {v}")
