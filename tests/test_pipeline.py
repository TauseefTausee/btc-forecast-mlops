import json

import numpy as np
import pandas as pd
import pytest

from src import model_registry
from src.etl_fetch import clean_ohlcv, load_raw, save_raw
from src.features import FEATURE_COLUMNS, MIN_HISTORY, build_features, make_dataset
from src.train_forecast import evaluate_holdout, regression_metrics, train_and_evaluate


def test_clean_ohlcv_fills_gaps_and_dedupes(raw_df):
    messy = pd.concat([raw_df, raw_df.iloc[[10]]])  # duplicate date
    messy = messy.drop(index=[20, 21])  # missing days
    messy.loc[5, "close"] = np.nan  # bad row
    cleaned = clean_ohlcv(messy, drop_partial_day=False)
    assert cleaned["date"].is_unique
    assert (cleaned["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert cleaned["close"].notna().all() and (cleaned["close"] > 0).all()


def test_clean_ohlcv_missing_column(raw_df):
    with pytest.raises(ValueError):
        clean_ohlcv(raw_df.drop(columns=["volume"]))


def test_save_and_load_raw_roundtrip(raw_df, tmp_path):
    path = save_raw(clean_ohlcv(raw_df, drop_partial_day=False), tmp_path / "btc_raw.csv")
    again = load_raw(path)
    assert len(again) == len(raw_df)
    assert list((tmp_path / "raw_snapshots").glob("btc_raw_*.csv"))


def test_features_have_no_lookahead(raw_df):
    full = build_features(raw_df["close"])
    partial = build_features(raw_df["close"].iloc[:300])
    # features at day 299 must not change when future data is appended
    pd.testing.assert_series_equal(full.iloc[299], partial.iloc[299], check_names=False)


def test_make_dataset_shapes(raw_df):
    ds = make_dataset(raw_df)
    assert set(FEATURE_COLUMNS) <= set(ds.columns)
    assert ds[FEATURE_COLUMNS].notna().all().all()
    assert ds["target_h7"].isna().sum() == 7  # last week has no future yet
    assert len(ds) == len(raw_df) - (MIN_HISTORY - 15)  # 30-day warm-up dropped


def test_regression_metrics():
    m = regression_metrics([100, 200], [110, 180])
    assert m["mae"] == pytest.approx(15)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["mape"] == pytest.approx(10)


def test_holdout_uses_latest_7_days(raw_df):
    ds = make_dataset(raw_df)
    ev = evaluate_holdout(ds)
    assert ev["test_window"]["dates"][-1] == ds["date"].iloc[-1].strftime("%Y-%m-%d")
    assert len(ev["test_window"]["predicted"]) == 7
    assert ev["mape"] < 50


def test_threshold_gate_and_registry(raw_df, tmp_path):
    ds = make_dataset(raw_df)
    failed = train_and_evaluate(ds, tmp_path, threshold=0.0)
    assert failed["passed"] is False
    assert model_registry.current_version(tmp_path) is None  # not promoted

    ok = train_and_evaluate(ds, tmp_path, threshold=100.0)
    assert ok["passed"] is True
    model_registry.promote(ok["version"], tmp_path)
    version, model, metrics = model_registry.load_current(tmp_path)
    assert version == ok["version"]
    assert metrics["passed_threshold"] is True
    preds = model.predict_latest()[3]
    assert len(preds) == 7

    last_run = json.loads((tmp_path / "last_run.json").read_text())
    assert last_run["version"] == ok["version"]


def test_promote_unknown_version(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_registry.promote("v-does-not-exist", tmp_path)


def test_model_save_load_roundtrip(raw_df, tmp_path):
    from src.forecaster import BTCForecaster

    model = BTCForecaster().fit(make_dataset(raw_df))
    model.save(tmp_path / "model")
    loaded = BTCForecaster.load(tmp_path / "model")
    np.testing.assert_allclose(model.predict_latest()[3], loaded.predict_latest()[3], rtol=1e-6)
    assert not list(tmp_path.rglob("*.pkl")) and not list(tmp_path.rglob("*.joblib"))
