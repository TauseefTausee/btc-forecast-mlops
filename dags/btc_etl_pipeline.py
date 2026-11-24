"""Daily BTC-USD ETL + retraining DAG.

    fetch_data -> preprocess_features -> train_evaluate -> check_threshold
        check_threshold -> register_model -> reload_api -> end
        check_threshold -> skip_registration ----------> end

Runs every day at 00:30 UTC, right after the previous daily candle closes.
Data and model artifacts are written to MODEL_DIR (the shared volume that the
Flask API, and the Kubernetes PVC via `minikube mount`, read from).

Manual runs can override the MAPE gate or force registration:
    airflow dags trigger btc_etl_pipeline --conf '{"force_register": true}'
"""

import logging
import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

log = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/shared/models")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(MODEL_DIR, "data"))
DEFAULT_THRESHOLD = float(os.getenv("MAPE_THRESHOLD", "10"))


# Task callables import project code lazily so the scheduler parses the DAG fast.
def fetch_data(**context):
    from pathlib import Path

    from src.etl_fetch import fetch_btc_data, save_raw

    df = fetch_btc_data()
    path = save_raw(df, Path(DATA_DIR) / "btc_raw.csv")
    info = {
        "raw_path": str(path),
        "rows": len(df),
        "source": df.attrs.get("source"),
        "first_date": df["date"].min().strftime("%Y-%m-%d"),
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
    }
    log.info("Fetched BTC data: %s", info)
    return info


def preprocess_features(ti, **context):
    from pathlib import Path

    from src.etl_fetch import load_raw
    from src.features import FEATURE_COLUMNS, make_dataset

    raw_info = ti.xcom_pull(task_ids="fetch_data")
    dataset = make_dataset(load_raw(Path(raw_info["raw_path"])))
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("NaN values found in engineered features")
    if len(dataset) < 200:
        raise ValueError(f"too few feature rows: {len(dataset)}")
    out = Path(DATA_DIR) / "btc_features.csv"
    dataset.to_csv(out, index=False, date_format="%Y-%m-%d")
    log.info("Wrote %d feature rows (MA7/MA30, volatility, RSI, returns) to %s", len(dataset), out)
    return {"features_path": str(out), "rows": len(dataset)}


def train_evaluate(ti, params, **context):
    from pathlib import Path

    from src.train_forecast import load_features, train_and_evaluate

    feat_info = ti.xcom_pull(task_ids="preprocess_features")
    threshold = float(params.get("mape_threshold", DEFAULT_THRESHOLD))
    summary = train_and_evaluate(
        load_features(Path(feat_info["features_path"])), Path(MODEL_DIR), threshold=threshold
    )
    m = summary["metrics"]
    log.info(
        "Candidate %s | MAE %.2f | RMSE %.2f | MAPE %.2f%% (naive %.2f%%) | threshold %.2f%%",
        summary["version"],
        m["mae"],
        m["rmse"],
        m["mape"],
        m["naive_mape"],
        threshold,
    )
    return {
        "version": summary["version"],
        "passed": summary["passed"],
        "mae": m["mae"],
        "rmse": m["rmse"],
        "mape": m["mape"],
    }


def check_threshold(ti, params, **context):
    result = ti.xcom_pull(task_ids="train_evaluate")
    if result["passed"] or params.get("force_register"):
        return "register_model"
    log.warning("MAPE %.2f%% failed the threshold; keeping current model", result["mape"])
    return "skip_registration"


def register_model(ti, **context):
    from pathlib import Path

    from src.model_registry import promote

    version = ti.xcom_pull(task_ids="train_evaluate")["version"]
    reg = promote(version, Path(MODEL_DIR))
    log.info("Registered %s as the production model", version)
    return {"version": version, "updated_at": reg["updated_at"]}


def reload_api(ti, **context):
    import requests

    version = ti.xcom_pull(task_ids="register_model")["version"]
    urls = [u.strip() for u in os.getenv("RELOAD_URLS", "http://api:5000/reload").split(",")]
    headers = {"X-API-Token": os.getenv("API_TOKEN", "")}
    results = {}
    for url in filter(None, urls):
        try:
            resp = requests.post(url, headers=headers, timeout=15)
            results[url] = resp.status_code
            log.info("Reload %s -> %s %s", url, resp.status_code, resp.text[:200])
        except requests.RequestException as exc:
            results[url] = str(exc)
            log.warning("Reload %s failed: %s", url, exc)
    if not any(code == 200 for code in results.values()):
        raise RuntimeError(f"no API instance acknowledged model {version}: {results}")
    return {"version": version, "results": results}


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ETL, feature engineering, retraining and API reload",
    schedule="30 0 * * *",
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    params={
        "mape_threshold": Param(DEFAULT_THRESHOLD, type="number", minimum=0),
        "force_register": Param(False, type="boolean"),
    },
    tags=["btc", "mlops", "etl", "forecasting"],
    doc_md=__doc__,
) as dag:
    t_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t_features = PythonOperator(task_id="preprocess_features", python_callable=preprocess_features)
    t_train = PythonOperator(task_id="train_evaluate", python_callable=train_evaluate)
    t_branch = BranchPythonOperator(task_id="check_threshold", python_callable=check_threshold)
    t_register = PythonOperator(task_id="register_model", python_callable=register_model)
    t_skip = EmptyOperator(task_id="skip_registration")
    t_reload = PythonOperator(
        task_id="reload_api",
        python_callable=reload_api,
        retries=3,
        retry_delay=timedelta(seconds=30),
    )
    t_end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    t_fetch >> t_features >> t_train >> t_branch
    t_branch >> t_register >> t_reload >> t_end
    t_branch >> t_skip >> t_end
