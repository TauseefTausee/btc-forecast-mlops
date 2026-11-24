"""Train + evaluate the 7-day BTC close forecaster and save model artifacts.

1. Hold out the latest 7 days: fit on everything before them, forecast them,
   and score MAE / RMSE / MAPE against the actual closes (plus a naive
   "tomorrow = today" baseline for context).
2. Refit on the full history so production uses the newest data.
3. Save the candidate to the registry; promote it only if MAPE passes the
   threshold (or always leave that decision to the Airflow branch task).

Usage:
    python -m src.train_forecast --features data/btc_features.csv --register
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, model_registry
from src.features import target_columns
from src.forecaster import BTCForecaster

log = logging.getLogger(__name__)


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    err = predicted - actual
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.abs(actual)) * 100),
    }


def evaluate_holdout(
    dataset: pd.DataFrame, horizon: int = config.FORECAST_HORIZON, params: dict = None
) -> dict:
    """Score the model on the most recent `horizon` days it has never seen."""
    dataset = dataset.reset_index(drop=True)
    n = len(dataset)
    origin = n - horizon - 1  # last day the model is allowed to know
    if origin < 100 + horizon:
        raise ValueError("dataset too small for a holdout evaluation")

    # Only rows whose targets end on/before the origin: no peeking at the test window.
    train = dataset.iloc[: origin - horizon + 1].copy()
    model = BTCForecaster(horizon, params).fit(train)

    origin_row = dataset.iloc[origin]
    preds = model.predict_from_features(
        origin_row[model.feature_columns].to_numpy(), origin_row["close"], horizon
    )
    test = dataset.iloc[origin + 1 : origin + 1 + horizon]
    actual = test["close"].to_numpy()
    naive = np.full(horizon, origin_row["close"])

    metrics = regression_metrics(actual, preds)
    baseline = regression_metrics(actual, naive)
    return {
        **metrics,
        "naive_mae": baseline["mae"],
        "naive_rmse": baseline["rmse"],
        "naive_mape": baseline["mape"],
        "test_window": {
            "origin_date": pd.Timestamp(origin_row["date"]).strftime("%Y-%m-%d"),
            "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test["date"]],
            "actual": [round(float(a), 2) for a in actual],
            "predicted": [round(float(p), 2) for p in preds],
        },
    }


def train_and_evaluate(
    dataset: pd.DataFrame,
    model_dir: Path = config.MODEL_DIR,
    horizon: int = config.FORECAST_HORIZON,
    threshold: float = config.MAPE_THRESHOLD,
) -> dict:
    """Evaluate, refit on all data and save a candidate. Returns a summary."""
    evaluation = evaluate_holdout(dataset, horizon)
    final_model = BTCForecaster(horizon).fit(dataset)
    passed = evaluation["mape"] <= threshold
    metrics = {
        **evaluation,
        "mape_threshold": threshold,
        "passed_threshold": bool(passed),
        "model_type": "XGBoost direct multi-horizon (log-return targets)",
        "horizon_days": horizon,
        "train_rows": final_model.train_rows,
        "data_start": pd.Timestamp(dataset["date"].min()).strftime("%Y-%m-%d"),
        "data_end": pd.Timestamp(dataset["date"].max()).strftime("%Y-%m-%d"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": config.MODEL_STAGE,
    }
    version = model_registry.save_candidate(final_model, metrics, model_dir)
    log.info(
        "Candidate %s: MAPE %.2f%% (threshold %.2f%%, naive %.2f%%) -> %s",
        version,
        metrics["mape"],
        threshold,
        metrics["naive_mape"],
        "PASS" if passed else "FAIL",
    )
    return {"version": version, "passed": bool(passed), "metrics": metrics}


def load_features(path: Path, horizon: int = config.FORECAST_HORIZON) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(target_columns(horizon)) - set(df.columns)
    if missing:
        raise ValueError(f"feature file is missing target columns {sorted(missing)}")
    return df


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the 7-day BTC forecaster")
    parser.add_argument("--features", default=str(config.DATA_DIR / "btc_features.csv"))
    parser.add_argument("--model-dir", default=str(config.MODEL_DIR))
    parser.add_argument("--threshold", type=float, default=config.MAPE_THRESHOLD)
    parser.add_argument(
        "--register", action="store_true", help="promote the candidate if it passes the threshold"
    )
    parser.add_argument(
        "--force-register",
        action="store_true",
        help="promote even if the threshold fails (bootstrap only)",
    )
    parser.add_argument("--summary-out", help="optional path to write the summary JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

    dataset = load_features(Path(args.features))
    summary = train_and_evaluate(dataset, Path(args.model_dir), threshold=args.threshold)
    if args.force_register or (args.register and summary["passed"]):
        model_registry.promote(summary["version"], Path(args.model_dir))
        summary["registered"] = True
    else:
        summary["registered"] = False
    text = json.dumps(summary, indent=2, default=str)
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
