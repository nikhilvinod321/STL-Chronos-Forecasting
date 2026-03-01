import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


class XGBoostBaseline:
    """
    XGBoost baseline with lag features and MultiOutputRegressor
    for direct 48h horizon prediction.
    Lags: [1, 2, ..., 24, 168]
    """

    def __init__(self, horizon: int = 48, lags: list = None, random_seed: int = 42):
        self.horizon = horizon
        self.lags = lags if lags is not None else list(range(1, 25)) + [168]
        self.max_lag = max(self.lags)
        self.random_seed = random_seed
        self.model = None

    def _create_features(self, series: np.ndarray) -> tuple:
        """
        Create lag features from a 1D series.
        Returns (X, y) where each row has lag features and y is the next `horizon` values.
        """
        n = len(series)
        X_list = []
        y_list = []

        for i in range(self.max_lag, n - self.horizon + 1):
            features = [series[i - lag] for lag in self.lags]
            X_list.append(features)
            y_list.append(series[i:i + self.horizon])

        X = np.array(X_list)
        y = np.array(y_list)
        return X, y

    def _create_features_from_context(self, context: np.ndarray) -> np.ndarray:
        """
        Create a single feature vector from the end of the context window.
        """
        features = [context[-lag] for lag in self.lags]
        return np.array(features).reshape(1, -1)

    def fit(self, train_series: np.ndarray):
        """
        Fit XGBoost model on the training series using lag features.
        Uses MultiOutputRegressor for direct multi-step forecasting.
        """
        X_train, y_train = self._create_features(train_series)

        if len(X_train) == 0:
            raise ValueError("Not enough training data for the given lags and horizon.")

        base_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_seed,
            n_jobs=-1,
            verbosity=0,
        )

        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train, y_train)

        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        """
        Predict the next `horizon` values given a context window.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._create_features_from_context(context)
        forecast = self.model.predict(X)
        return forecast.flatten()

    def forecast_all_windows(self, windows: list) -> list:
        """
        Predict for all rolling windows.
        Returns list of forecast arrays.
        """
        forecasts = []
        for i, w in enumerate(tqdm(windows, desc="  XGBoost", unit="win")):
            context = w["context"].values
            pred = self.predict(context)
            forecasts.append(pred)
        return forecasts


if __name__ == "__main__":
    from data_loader import load_data, split_train_test, create_rolling_windows
    from utils import load_config, set_global_seed

    config = load_config()
    set_global_seed(config["execution"]["random_seed"])
    df = load_data(config)
    train, test = split_train_test(df, config)

    xgb_model = XGBoostBaseline(
        horizon=config["evaluation"]["horizon"],
        random_seed=config["execution"]["random_seed"],
    )
    xgb_model.fit(train["PJME_MW"].values)

    windows = create_rolling_windows(
        df, test,
        context_length=config["evaluation"]["context_length"],
        horizon=config["evaluation"]["horizon"],
        stride=config["evaluation"]["stride"],
        dry_run=True,
        max_windows=2,
    )

    if windows:
        pred = xgb_model.predict(windows[0]["context"].values)
        print(f"XGBoost prediction shape: {pred.shape}")
        print(f"XGBoost prediction mean: {pred.mean():.1f}")
