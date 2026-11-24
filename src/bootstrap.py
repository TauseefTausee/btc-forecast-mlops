"""Make sure a production model exists on the shared volume.

Runs as the Kubernetes init container (and on first docker-compose start).
If a model is already registered it exits immediately; otherwise it runs
fetch -> features -> train once and registers the result, so a fresh
cluster can serve predictions before Airflow has ever run.
"""

import logging
import sys
from pathlib import Path

from src import config, model_registry
from src.etl_fetch import fetch_btc_data, save_raw
from src.features import make_dataset
from src.train_forecast import train_and_evaluate

log = logging.getLogger("bootstrap")


def ensure_model(model_dir: Path = config.MODEL_DIR, data_dir: Path = config.DATA_DIR) -> str:
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_registry.current_version(model_dir):
        version = model_registry.current_version(model_dir)
        log.info("Model %s already registered, nothing to do", version)
        return version

    # Several replicas may start together; only one should train.
    with model_registry.file_lock(model_dir / ".bootstrap.lock"):
        version = model_registry.current_version(model_dir)
        if version:
            log.info("Model %s was registered by another pod", version)
            return version
        raw = fetch_btc_data()
        save_raw(raw, data_dir / "btc_raw.csv")
        dataset = make_dataset(raw)
        dataset.to_csv(data_dir / "btc_features.csv", index=False, date_format="%Y-%m-%d")
        summary = train_and_evaluate(dataset, model_dir)
        # First model is registered regardless of the threshold so the API can
        # serve; its metrics (including pass/fail) are visible on /metrics.
        model_registry.promote(summary["version"], model_dir)
        log.info(
            "Bootstrapped model %s (MAPE %.2f%%)", summary["version"], summary["metrics"]["mape"]
        )
        return summary["version"]


def main() -> int:
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
    try:
        ensure_model()
    except Exception as exc:  # noqa: BLE001
        log.error("Bootstrap failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
