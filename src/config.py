"""Central configuration, read from environment variables.

Everything that changes between local runs, docker-compose and Kubernetes
lives here so the code never hard-codes paths or thresholds. In Kubernetes
these values come from the ConfigMap / Secret (see k8s/configmap-secret.yaml).
"""

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


TICKER = os.getenv("TICKER", "BTC-USD")

# Model artifacts and historical data both live under MODEL_DIR, which is the
# PVC mount (/app/models) inside Kubernetes.
MODEL_DIR = Path(os.getenv("MODEL_DIR", "shared/models"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(MODEL_DIR / "data")))

FORECAST_HORIZON = _env_int("FORECAST_HORIZON", 7)
HISTORY_DAYS = _env_int("HISTORY_DAYS", 1500)

# A candidate model is only registered if its MAPE on the latest 7-day test
# window is at or below this value (percent).
MAPE_THRESHOLD = _env_float("MAPE_THRESHOLD", 10.0)

MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
PORT = _env_int("PORT", 5000)

# How often (seconds) each API worker checks the registry for a newer model.
MODEL_REFRESH_SECONDS = _env_int("MODEL_REFRESH_SECONDS", 15)

# Shared token protecting POST /reload. Comes from the K8s Secret.
API_TOKEN = os.getenv("API_TOKEN", "")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
