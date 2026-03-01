import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression


class STLDecomposer:
    """
    STL Decomposition Pipeline: STL -> DoW -> Residual.
    Handles context decomposition and forward projection for forecasting.
    """

    def __init__(self, period: int = 168, robust: bool = True):
        self.period = period
        self.robust = robust
        self.dow_adjustments = None  # 7 frozen values from training data

    def fit_dow_adjustments(self, train_df: pd.DataFrame):
        """
        Calculate historical average of STL residuals per Day of Week
        using ONLY the training data. Freezes 7 values.
        """
        # Fit STL on entire training series to get residuals
        series = train_df["PJME_MW"].values
        stl = STL(series, period=self.period, robust=self.robust)
        result = stl.fit()

        residuals = pd.Series(result.resid, index=train_df.index)
        dow = residuals.index.dayofweek  # Monday=0, Sunday=6
        self.dow_adjustments = residuals.groupby(dow).mean().to_dict()

        return self.dow_adjustments

    def decompose_context(self, context: pd.Series):
        """
        Decompose a 336h context window using STL.
        Returns trend, seasonal, residual components as numpy arrays,
        plus the STL result object.
        """
        stl = STL(context.values, period=self.period, robust=self.robust)
        result = stl.fit()
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
            "stl_result": result,
        }

    def project_trend(self, trend: np.ndarray, horizon: int = 48):
        """
        Fit linear regression on the last 48h of the trend component,
        then project forward 48h.
        """
        last_48 = trend[-48:]
        X = np.arange(len(last_48)).reshape(-1, 1)
        y = last_48

        lr = LinearRegression()
        lr.fit(X, y)

        X_future = np.arange(len(last_48), len(last_48) + horizon).reshape(-1, 1)
        trend_forecast = lr.predict(X_future)
        return trend_forecast

    def project_seasonal(self, seasonal: np.ndarray, horizon: int = 48):
        """
        Repeat the last 168h block of the seasonal component for 48h.
        """
        last_cycle = seasonal[-self.period:]
        # Tile enough to cover horizon, then trim
        repeats = (horizon // self.period) + 2
        extended = np.tile(last_cycle, repeats)
        return extended[:horizon]

    def get_dow_for_timestamps(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Look up frozen DoW adjustments for given timestamps.
        """
        if self.dow_adjustments is None:
            raise ValueError("DoW adjustments not fitted. Call fit_dow_adjustments first.")
        return np.array([self.dow_adjustments.get(ts.dayofweek, 0.0) for ts in timestamps])

    def decompose_and_project(self, context: pd.Series, target_index: pd.DatetimeIndex,
                              horizon: int = 48, use_dow: bool = True):
        """
        Full decomposition pipeline for a single window.

        Returns dict with:
          - trend_forecast (horizon,)
          - seasonal_forecast (horizon,)
          - dow_context (context_length,)
          - dow_forecast (horizon,)
          - context_residual (context_length,): Raw - Trend - Seasonal - DoW
          - components for recomposition
        """
        decomp = self.decompose_context(context)
        trend_forecast = self.project_trend(decomp["trend"], horizon)
        seasonal_forecast = self.project_seasonal(decomp["seasonal"], horizon)

        # DoW for context
        dow_context = self.get_dow_for_timestamps(context.index) if use_dow else np.zeros(len(context))
        # DoW for target
        dow_forecast = self.get_dow_for_timestamps(target_index) if use_dow else np.zeros(horizon)

        # Context residual = Raw - Trend - Seasonal - DoW
        context_residual = context.values - decomp["trend"] - decomp["seasonal"] - dow_context

        return {
            "trend_forecast": trend_forecast,
            "seasonal_forecast": seasonal_forecast,
            "dow_context": dow_context,
            "dow_forecast": dow_forecast,
            "context_residual": context_residual,
            "context_trend": decomp["trend"],
            "context_seasonal": decomp["seasonal"],
        }

    def recompose(self, trend_forecast: np.ndarray, seasonal_forecast: np.ndarray,
                  dow_forecast: np.ndarray, residual_forecast: np.ndarray) -> np.ndarray:
        """
        Recompose forecast: Trend + Seasonality + DoW + Chronos_Residual.
        Pure pointwise addition.
        """
        return trend_forecast + seasonal_forecast + dow_forecast + residual_forecast


if __name__ == "__main__":
    from data_loader import load_data, split_train_test, create_rolling_windows
    from utils import load_config

    config = load_config()
    df = load_data(config)
    train, test = split_train_test(df, config)

    decomposer = STLDecomposer(period=168, robust=True)
    dow_adj = decomposer.fit_dow_adjustments(train)
    print(f"DoW adjustments: {dow_adj}")

    windows = create_rolling_windows(
        df, test,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"],
        dry_run=True,
        max_windows=1,
    )

    if windows:
        w = windows[0]
        result = decomposer.decompose_and_project(
            w["context"], w["target"].index, horizon=48, use_dow=True
        )
        print(f"Trend forecast shape: {result['trend_forecast'].shape}")
        print(f"Seasonal forecast shape: {result['seasonal_forecast'].shape}")
        print(f"Context residual shape: {result['context_residual'].shape}")

        # Test recomposition with dummy residual
        dummy_resid = np.zeros(48)
        recomp = decomposer.recompose(
            result["trend_forecast"], result["seasonal_forecast"],
            result["dow_forecast"], dummy_resid
        )
        print(f"Recomposed forecast shape: {recomp.shape}")
