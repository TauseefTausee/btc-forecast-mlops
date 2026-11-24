"""Transform step: clean prices -> technical indicators -> supervised targets.

All features are derived from the close series only. That keeps training and
serving identical and lets POST /predict accept a plain list of closing prices
as custom history.

Usage:
    python -m src.features --input data/btc_raw.csv --output data/btc_features.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.etl_fetch import load_raw

log = logging.getLogger(__name__)

RETURN_LAGS = 7
# 30-day indicators + a few days of lagged returns need this much history.
MIN_HISTORY = 45

FEATURE_COLUMNS = [f"ret_lag_{k}" for k in range(RETURN_LAGS)] + [
    "ret_5d",
    "ret_14d",
    "ret_30d",
    "ma7_ratio",
    "ma30_ratio",
    "ma7_ma30_ratio",
    "volatility_7d",
    "volatility_30d",
    "rsi_14",
]


def target_columns(horizon: int = config.FORECAST_HORIZON) -> list:
    return [f"target_h{h}" for h in range(1, horizon + 1)]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(loss.notna())


def build_features(close: pd.Series) -> pd.DataFrame:
    """Compute indicator features for every row of a close-price series."""
    close = pd.Series(close, dtype="float64").reset_index(drop=True)
    if (close <= 0).any() or close.isna().any():
        raise ValueError("close prices must be positive and non-null")

    log_ret = np.log(close).diff()
    feats = pd.DataFrame(index=close.index)
    for k in range(RETURN_LAGS):
        feats[f"ret_lag_{k}"] = log_ret.shift(k)
    for n in (5, 14, 30):
        feats[f"ret_{n}d"] = np.log(close / close.shift(n))

    ma7 = close.rolling(7).mean()
    ma30 = close.rolling(30).mean()
    feats["ma7_ratio"] = close / ma7 - 1
    feats["ma30_ratio"] = close / ma30 - 1
    feats["ma7_ma30_ratio"] = ma7 / ma30 - 1
    feats["volatility_7d"] = log_ret.rolling(7).std()
    feats["volatility_30d"] = log_ret.rolling(30).std()
    feats["rsi_14"] = _rsi(close) / 100.0
    return feats[FEATURE_COLUMNS]


def build_targets(close: pd.Series, horizon: int = config.FORECAST_HORIZON) -> pd.DataFrame:
    """Direct multi-horizon targets: log(close[t+h] / close[t]) for h = 1..H."""
    close = pd.Series(close, dtype="float64").reset_index(drop=True)
    targets = pd.DataFrame(index=close.index)
    for h in range(1, horizon + 1):
        targets[f"target_h{h}"] = np.log(close.shift(-h) / close)
    return targets


def make_dataset(raw: pd.DataFrame, horizon: int = config.FORECAST_HORIZON) -> pd.DataFrame:
    """Full feature table: date, close, MA/volatility indicators and targets.

    Rows at the end without future values keep NaN targets; they are still
    useful as the latest feature rows for forecasting.
    """
    raw = raw.sort_values("date").reset_index(drop=True)
    feats = build_features(raw["close"])
    targets = build_targets(raw["close"], horizon)
    out = pd.concat([raw[["date", "close"]], feats, targets], axis=1)
    out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Feature engineering for BTC data")
    parser.add_argument("--input", default=str(config.DATA_DIR / "btc_raw.csv"))
    parser.add_argument("--output", default=str(config.DATA_DIR / "btc_features.csv"))
    parser.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON)
    args = parser.parse_args(argv)
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

    dataset = make_dataset(load_raw(Path(args.input)), args.horizon)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(dataset)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
