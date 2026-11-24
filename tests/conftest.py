import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synthetic_raw(n_days: int = 600, seed: int = 7) -> pd.DataFrame:
    """Geometric random walk with BTC-like volatility; no network needed."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.03, n_days)
    close = 30_000 * np.exp(np.cumsum(returns))
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n_days, freq="D"
    )
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.005, n_days)),
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.uniform(1e9, 5e9, n_days),
        }
    )


@pytest.fixture
def raw_df():
    return synthetic_raw()


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    """A model directory with one registered model, shared across API tests."""
    from src import model_registry
    from src.features import make_dataset
    from src.train_forecast import train_and_evaluate

    model_dir = tmp_path_factory.mktemp("models")
    dataset = make_dataset(synthetic_raw())
    summary = train_and_evaluate(dataset, model_dir, threshold=100.0)
    model_registry.promote(summary["version"], model_dir)
    return model_dir


@pytest.fixture
def client(trained_model_dir, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token")
    from app.app import create_app

    app = create_app(model_dir=trained_model_dir, api_token="test-token")
    app.config["TESTING"] = True
    return app.test_client()


os.environ.setdefault("LOG_LEVEL", "WARNING")
