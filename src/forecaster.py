"""Direct multi-horizon XGBoost forecaster for the BTC daily close.

One gradient-boosted model per horizon h (1..7) predicts the log return
log(close[t+h] / close[t]) from features known at day t. Predicting returns
instead of raw prices keeps the target stationary, so the model does not
break when BTC trades at levels it never saw during training.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import FEATURE_COLUMNS, MIN_HISTORY, build_features, target_columns

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    # Single-threaded inference plays nicely with K8s CPU limits.
    "n_jobs": 1,
    "random_state": 42,
}

HISTORY_TAIL = 120
METADATA_FILE = "metadata.json"


class BTCForecaster:
    def __init__(self, horizon: int = 7, params: dict = None):
        self.horizon = horizon
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.models = []
        self.feature_columns = list(FEATURE_COLUMNS)
        self.trained_at = None
        self.train_rows = 0
        self.history_dates = []
        self.history_closes = []

    # ------------------------------------------------------------------ fit
    def fit(self, dataset: pd.DataFrame) -> "BTCForecaster":
        targets = target_columns(self.horizon)
        train = dataset.dropna(subset=self.feature_columns + targets)
        if len(train) < 100:
            raise ValueError(f"not enough training rows: {len(train)}")
        X = train[self.feature_columns].to_numpy()
        self.models = []
        for col in targets:
            model = XGBRegressor(**self.params)
            model.fit(X, train[col].to_numpy())
            self.models.append(model)
        self.train_rows = len(train)
        self.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tail = dataset.tail(HISTORY_TAIL)
        self.history_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in tail["date"]]
        self.history_closes = [float(c) for c in tail["close"]]
        return self

    # -------------------------------------------------------------- predict
    def predict_from_features(self, feature_row, last_close: float, days: int = None) -> np.ndarray:
        """Predicted closes for the next `days` days from one feature row."""
        if not self.models:
            raise RuntimeError("model is not trained")
        days = days or self.horizon
        x = np.asarray(feature_row, dtype="float64").reshape(1, -1)
        log_returns = np.array([m.predict(x)[0] for m in self.models[:days]])
        return float(last_close) * np.exp(log_returns)

    def predict_from_closes(self, closes, days: int = None) -> np.ndarray:
        closes = pd.Series(closes, dtype="float64")
        if len(closes) < MIN_HISTORY:
            raise ValueError(f"need at least {MIN_HISTORY} closing prices, got {len(closes)}")
        feats = build_features(closes)
        last = feats.iloc[-1]
        if last.isna().any():
            raise ValueError("could not compute features from the given history")
        return self.predict_from_features(last.to_numpy(), closes.iloc[-1], days)

    def predict_latest(self, days: int = None):
        """Forecast from the history stored at training time.

        Returns (last_observed_date, last_close, list_of_dates, predictions).
        """
        preds = self.predict_from_closes(self.history_closes, days)
        last_date = pd.Timestamp(self.history_dates[-1])
        dates = [
            (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, len(preds) + 1)
        ]
        return last_date.strftime("%Y-%m-%d"), self.history_closes[-1], dates, preds

    # ---------------------------------------------------------- persistence
    # Native XGBoost JSON + a metadata file instead of pickle: portable across
    # Python/pandas versions (Airflow vs API image) and safe to load.
    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for h, model in enumerate(self.models, start=1):
            model.save_model(directory / f"xgb_h{h}.json")
        meta = {
            "horizon": self.horizon,
            "params": self.params,
            "feature_columns": self.feature_columns,
            "trained_at": self.trained_at,
            "train_rows": self.train_rows,
            "history_dates": self.history_dates,
            "history_closes": self.history_closes,
        }
        tmp = directory / (METADATA_FILE + ".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(directory / METADATA_FILE)  # written last = "commit"
        return directory

    @classmethod
    def load(cls, directory) -> "BTCForecaster":
        directory = Path(directory)
        meta = json.loads((directory / METADATA_FILE).read_text())
        obj = cls(meta["horizon"], meta["params"])
        if meta["feature_columns"] != list(FEATURE_COLUMNS):
            raise ValueError("model was trained with a different feature set")
        obj.models = []
        for h in range(1, obj.horizon + 1):
            model = XGBRegressor(n_jobs=1)
            model.load_model(directory / f"xgb_h{h}.json")
            obj.models.append(model)
        obj.trained_at = meta["trained_at"]
        obj.train_rows = meta["train_rows"]
        obj.history_dates = meta["history_dates"]
        obj.history_closes = meta["history_closes"]
        return obj
