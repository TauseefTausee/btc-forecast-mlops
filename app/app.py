"""Flask REST API serving the 7-day Bitcoin close-price forecast.

Endpoints
    GET  /health   liveness/readiness: status, uptime, loaded model version
    POST /predict  7-day forecast (optional forecast_days, optional custom history)
    GET  /metrics  MAE / RMSE / MAPE of the loaded model and of the latest run
    POST /reload   re-read the model registry (called by Airflow after retraining)

Run in production with Gunicorn:
    gunicorn --config app/gunicorn.conf.py app.app:app      (repo root)
    gunicorn --config gunicorn.conf.py app:app              (inside the container)
"""

import logging
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src import config, model_registry
from src.features import MIN_HISTORY

MAX_HISTORY = 5000


class ModelStore:
    """Holds the production model and hot-swaps it when the registry changes.

    Each Gunicorn worker (and each pod) has its own copy. Instead of relying on
    every worker receiving the /reload call, workers stat registry.json at most
    every MODEL_REFRESH_SECONDS and reload when the pointer moves. All pods
    share the same PVC, so one retraining run updates the whole fleet.
    """

    def __init__(self, model_dir: Path, refresh_seconds: int):
        self.model_dir = Path(model_dir)
        self.refresh_seconds = refresh_seconds
        self.version = None
        self.model = None
        self.metrics = {}
        self.loaded_at = None
        self.last_error = None
        self._last_check = 0.0
        self._lock = threading.Lock()

    def reload(self, force: bool = False) -> bool:
        """Load the registered version. Returns True if a new model was loaded."""
        with self._lock:
            self._last_check = time.monotonic()
            try:
                version = model_registry.current_version(self.model_dir)
                if not version:
                    self.last_error = "no registered model yet"
                    return False
                if version == self.version and not force:
                    return False
                model, metrics = model_registry.load_version(version, self.model_dir)
                self.model, self.metrics, self.version = model, metrics, version
                self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.last_error = None
                logging.getLogger(__name__).info("Loaded model %s", version)
                return True
            except Exception as exc:  # noqa: BLE001 - keep serving the old model
                self.last_error = str(exc)
                logging.getLogger(__name__).exception("Model reload failed")
                return False

    def maybe_refresh(self) -> None:
        if time.monotonic() - self._last_check >= self.refresh_seconds:
            self.reload()


def _error(status: int, message: str, **extra):
    body = {"error": message, "status": status, **extra}
    return jsonify(body), status


def _parse_forecast_days(payload: dict, horizon: int) -> int:
    value = payload.get("forecast_days", horizon)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("forecast_days must be an integer")
    if not 1 <= value <= horizon:
        raise ValueError(f"forecast_days must be between 1 and {horizon}")
    return value


def _parse_history(payload: dict):
    history = payload.get("history")
    if history is None:
        return None
    if not isinstance(history, list):
        raise ValueError("history must be a list of closing prices")
    if not MIN_HISTORY <= len(history) <= MAX_HISTORY:
        raise ValueError(f"history must contain between {MIN_HISTORY} and {MAX_HISTORY} prices")
    clean = []
    for i, v in enumerate(history):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"history[{i}] is not a number")
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"history[{i}] must be a positive finite number")
        clean.append(float(v))
    return clean


def _parse_last_date(payload: dict):
    value = payload.get("last_date")
    if value is None:
        return pd.Timestamp(datetime.now(timezone.utc).date())
    if not isinstance(value, str):
        raise ValueError("last_date must be a YYYY-MM-DD string")
    try:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))
    except ValueError as exc:
        raise ValueError("last_date must be a YYYY-MM-DD string") from exc


def create_app(model_dir=None, api_token=None, refresh_seconds=None) -> Flask:
    logging.basicConfig(
        level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    app = Flask(__name__)
    app.json.sort_keys = False

    store = ModelStore(
        model_dir or config.MODEL_DIR,
        config.MODEL_REFRESH_SECONDS if refresh_seconds is None else refresh_seconds,
    )
    store.reload()
    token = config.API_TOKEN if api_token is None else api_token
    started = time.time()
    counters = {"predict_requests": 0, "predict_errors": 0}
    app.extensions["model_store"] = store

    @app.before_request
    def _refresh_model():
        if request.endpoint in ("health", "predict", "metrics"):
            store.maybe_refresh()

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "uptime_seconds": round(time.time() - started, 1),
                "model_loaded": store.model is not None,
                "model_version": store.version,
                "model_loaded_at": store.loaded_at,
                "model_stage": config.MODEL_STAGE,
                "last_error": store.last_error,
            }
        )

    @app.post("/predict")
    def predict():
        counters["predict_requests"] += 1
        if store.model is None:
            counters["predict_errors"] += 1
            return _error(503, "model not available yet", detail=store.last_error)

        if request.content_length and not request.is_json:
            counters["predict_errors"] += 1
            return _error(415, "Content-Type must be application/json")
        payload = request.get_json(silent=True) if request.content_length else {}
        if payload is None:
            counters["predict_errors"] += 1
            return _error(400, "request body is not valid JSON")
        if not isinstance(payload, dict):
            counters["predict_errors"] += 1
            return _error(400, "request body must be a JSON object")

        model = store.model
        try:
            days = _parse_forecast_days(payload, model.horizon)
            history = _parse_history(payload)
            if history is None:
                last_date, last_close, dates, preds = model.predict_latest(days)
                source = "stored_history"
            else:
                last_ts = _parse_last_date(payload)
                preds = model.predict_from_closes(history, days)
                last_date, last_close = last_ts.strftime("%Y-%m-%d"), history[-1]
                dates = [
                    (last_ts + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, days + 1)
                ]
                source = "custom_history"
        except ValueError as exc:
            counters["predict_errors"] += 1
            return _error(400, str(exc))

        return jsonify(
            {
                "ticker": config.TICKER,
                "model_version": store.version,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input_source": source,
                "last_observed_date": last_date,
                "last_observed_close": round(float(last_close), 2),
                "forecast_days": days,
                "predictions": [
                    {"date": d, "predicted_close": round(float(p), 2)} for d, p in zip(dates, preds)
                ],
            }
        )

    @app.get("/metrics")
    def metrics():
        last_run = model_registry.read_json(Path(store.model_dir) / model_registry.LAST_RUN_FILE)
        m = store.metrics or {}
        keys = (
            "mae",
            "rmse",
            "mape",
            "naive_mae",
            "naive_rmse",
            "naive_mape",
            "mape_threshold",
            "passed_threshold",
            "train_rows",
            "data_start",
            "data_end",
            "trained_at",
            "model_type",
            "test_window",
        )
        return jsonify(
            {
                "model_version": store.version,
                "model_metrics": {k: m.get(k) for k in keys} if m else None,
                "latest_validation_run": (
                    None
                    if not last_run
                    else {
                        "version": last_run.get("version"),
                        "mae": last_run.get("mae"),
                        "rmse": last_run.get("rmse"),
                        "mape": last_run.get("mape"),
                        "passed_threshold": last_run.get("passed_threshold"),
                        "trained_at": last_run.get("trained_at"),
                    }
                ),
                "service": {
                    "uptime_seconds": round(time.time() - started, 1),
                    **counters,
                },
            }
        )

    @app.post("/reload")
    def reload_model():
        if token and request.headers.get("X-API-Token") != token:
            return _error(401, "invalid or missing X-API-Token")
        changed = store.reload(force=bool(request.args.get("force")))
        if store.model is None:
            return _error(503, "no model could be loaded", detail=store.last_error)
        return jsonify(
            {"reloaded": changed, "model_version": store.version, "loaded_at": store.loaded_at}
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        return _error(exc.code, exc.description)

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logging.getLogger(__name__).exception("Unhandled error")
        return _error(500, "internal server error")

    return app


app = create_app()
