import numpy as np

from tests.conftest import synthetic_raw


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("v")
    assert body["uptime_seconds"] >= 0


def test_predict_default_returns_7_days(client):
    resp = client.post("/predict")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["forecast_days"] == 7
    assert len(body["predictions"]) == 7
    dates = [p["date"] for p in body["predictions"]]
    assert dates == sorted(dates) and len(set(dates)) == 7
    assert all(p["predicted_close"] > 0 for p in body["predictions"])


def test_predict_forecast_days(client):
    resp = client.post("/predict", json={"forecast_days": 3})
    assert resp.status_code == 200
    assert len(resp.get_json()["predictions"]) == 3


def test_predict_custom_history(client):
    history = synthetic_raw(90, seed=3)["close"].round(2).tolist()
    resp = client.post("/predict", json={"history": history, "last_date": "2026-01-31"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["input_source"] == "custom_history"
    assert body["predictions"][0]["date"] == "2026-02-01"
    # forecasts should stay in a sane band around the last price
    last = history[-1]
    assert all(0.5 * last < p["predicted_close"] < 1.5 * last for p in body["predictions"])


def test_predict_rejects_bad_forecast_days(client):
    for bad in (0, 8, "7", 2.5, True, None):
        resp = client.post("/predict", json={"forecast_days": bad})
        assert resp.status_code == 400, bad
        assert "forecast_days" in resp.get_json()["error"]


def test_predict_rejects_bad_history(client):
    cases = [
        {"history": "not-a-list"},
        {"history": [100.0] * 10},  # too short
        {"history": [100.0] * 59 + [None]},  # null value
        {"history": [100.0] * 59 + [-5]},  # negative
        {"history": [100.0] * 59 + ["abc"]},  # string
    ]
    for payload in cases:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400, payload


def test_predict_rejects_invalid_json(client):
    resp = client.post("/predict", data="{not json", content_type="application/json")
    assert resp.status_code == 400
    resp = client.post("/predict", data="[1, 2]", content_type="application/json")
    assert resp.status_code == 400
    resp = client.post("/predict", data="forecast_days=3", content_type="text/plain")
    assert resp.status_code == 415


def test_predict_bad_last_date(client):
    history = list(np.linspace(100, 120, 60))
    resp = client.post("/predict", json={"history": history, "last_date": "31-01-2026"})
    assert resp.status_code == 400


def test_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    for key in ("mae", "rmse", "mape"):
        assert body["model_metrics"][key] >= 0
    assert len(body["model_metrics"]["test_window"]["actual"]) == 7
    assert body["latest_validation_run"]["version"]


def test_reload_requires_token(client):
    assert client.post("/reload").status_code == 401
    resp = client.post("/reload", headers={"X-API-Token": "test-token"})
    assert resp.status_code == 200
    assert resp.get_json()["model_version"].startswith("v")


def test_unknown_route_and_method(client):
    assert client.get("/nope").status_code == 404
    resp = client.get("/predict")
    assert resp.status_code == 405
    assert resp.get_json()["status"] == 405


def test_predict_503_without_model(tmp_path):
    from app.app import create_app

    app = create_app(model_dir=tmp_path, api_token="x")
    c = app.test_client()
    assert c.get("/health").get_json()["model_loaded"] is False
    assert c.post("/predict").status_code == 503


def test_hot_reload_picks_up_new_version(trained_model_dir):
    """A newly promoted model is served without restarting the API."""
    from app.app import create_app
    from src import model_registry
    from src.features import make_dataset
    from src.train_forecast import train_and_evaluate

    app = create_app(model_dir=trained_model_dir, api_token="t", refresh_seconds=0)
    c = app.test_client()
    old = c.get("/health").get_json()["model_version"]

    import time

    time.sleep(1.1)  # versions are second-resolution timestamps
    summary = train_and_evaluate(
        make_dataset(synthetic_raw(seed=11)), trained_model_dir, threshold=100.0
    )
    model_registry.promote(summary["version"], trained_model_dir)

    resp = c.post("/predict")
    assert resp.status_code == 200
    assert resp.get_json()["model_version"] == summary["version"] != old
