"""Data-source parsing and fallback logic, with the HTTP layer mocked."""

import pandas as pd
import pytest

from src import etl_fetch
from tests.conftest import synthetic_raw


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _days_ago_ms(n):
    ts = pd.Timestamp.today().normalize() - pd.Timedelta(days=n)
    return int(ts.timestamp() * 1000)


def test_binance_parsing(monkeypatch):
    klines = [
        [_days_ago_ms(n), "100", "110", "90", str(100 + n), "5.5", 0, "0", 0, "0", "0", "0"]
        for n in range(200, 0, -1)
    ]
    monkeypatch.setattr(etl_fetch.requests, "get", lambda *a, **k: FakeResponse(klines))
    df = etl_fetch.fetch_binance(200)
    assert list(df.columns) == etl_fetch.COLUMNS
    assert len(df) == 200
    assert df["close"].dtype.kind in "fO"
    assert float(df["close"].iloc[-1]) == 101.0


def test_coingecko_parsing(monkeypatch):
    payload = {
        "prices": [[_days_ago_ms(n), 50_000 + n] for n in range(150, 0, -1)],
        "total_volumes": [[_days_ago_ms(n), 1e9] for n in range(150, 0, -1)],
    }
    monkeypatch.setattr(etl_fetch.requests, "get", lambda *a, **k: FakeResponse(payload))
    df = etl_fetch.fetch_coingecko(150)
    assert len(df) == 150
    assert (df["open"] == df["close"]).all()


def test_fallback_to_next_source(monkeypatch):
    def broken(days):
        raise ConnectionError("rate limited")

    monkeypatch.setitem(etl_fetch.SOURCES, "yfinance", broken)
    monkeypatch.setitem(etl_fetch.SOURCES, "binance", lambda days: synthetic_raw(300))
    df = etl_fetch.fetch_btc_data(300)
    assert df.attrs["source"] == "binance"
    assert len(df) >= etl_fetch.MIN_ROWS


def test_all_sources_fail(monkeypatch):
    for name in list(etl_fetch.SOURCES):
        monkeypatch.setitem(etl_fetch.SOURCES, name, lambda days: synthetic_raw(50))
    with pytest.raises(RuntimeError, match="All data sources failed"):
        etl_fetch.fetch_btc_data(50)


def test_bootstrap_trains_once(monkeypatch, tmp_path):
    from src import bootstrap, model_registry

    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return etl_fetch.clean_ohlcv(synthetic_raw(600), drop_partial_day=False)

    monkeypatch.setattr(bootstrap, "fetch_btc_data", fake_fetch)
    v1 = bootstrap.ensure_model(tmp_path, tmp_path / "data")
    v2 = bootstrap.ensure_model(tmp_path, tmp_path / "data")
    assert v1 == v2 == model_registry.current_version(tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "data" / "btc_features.csv").exists()
