"""Kubeflow Pipelines (KFP v2) definition of the BTC training workflow.

Three container components run the same project code as Airflow, inside the
trainer image (src/Dockerfile), passing data between steps as KFP artifacts:

    ingest_data -> generate_features -> train_model -> log_metrics

Compile to the YAML spec (what you upload in the KFP UI):
    TRAINER_IMAGE=docker.io/<user>/btc-forecast-trainer:latest \
        python kubeflow/btc_kfp_pipeline.py --output kubeflow/btc_kfp_pipeline.yaml

Optionally submit it as a daily recurring run to a KFP endpoint:
    python kubeflow/btc_kfp_pipeline.py --submit --host http://localhost:8080
"""

import argparse
import os

from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

# Read at import time because container components are resolved when defined.
TRAINER_IMAGE = os.getenv("TRAINER_IMAGE", "docker.io/tauseeftausee/btc-forecast-trainer:latest")


@dsl.container_component
def ingest_data(days: int, raw_data: Output[Dataset]):
    """Pull daily BTC-USD OHLCV data (yfinance -> Binance -> CoinGecko)."""
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=["--days", days, "--output", raw_data.path],
    )


@dsl.container_component
def generate_features(raw_data: Input[Dataset], features: Output[Dataset]):
    """Moving averages, volatility, RSI, lagged returns + 1..7 day targets."""
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "src.features"],
        args=["--input", raw_data.path, "--output", features.path],
    )


@dsl.container_component
def train_model(
    features: Input[Dataset],
    mape_threshold: float,
    model: Output[Model],
    summary: Output[Artifact],
):
    """Holdout-evaluate on the latest 7 days, refit, save versioned artifacts."""
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "src.train_forecast"],
        args=[
            "--features",
            features.path,
            "--model-dir",
            model.path,
            "--threshold",
            mape_threshold,
            "--register",
            "--summary-out",
            summary.path,
        ],
    )


@dsl.component(base_image="python:3.10-slim")
def log_metrics(summary: Input[Artifact], metrics: Output[Metrics]) -> bool:
    """Surface MAE / RMSE / MAPE in the KFP UI and return the gate result."""
    import json

    with open(summary.path) as fh:
        result = json.load(fh)
    m = result["metrics"]
    for key in ("mae", "rmse", "mape", "naive_mape", "mape_threshold"):
        metrics.log_metric(key, float(m[key]))
    metrics.log_metric("registered", float(result.get("registered", False)))
    metrics.metadata["model_version"] = result["version"]
    return bool(result["passed"])


@dsl.pipeline(
    name="btc-7day-forecast-training",
    description="Ingest BTC-USD data, engineer features, train and evaluate the 7-day forecaster",
)
def btc_forecast_pipeline(days: int = 1500, mape_threshold: float = 10.0):
    ingest = ingest_data(days=days)
    ingest.set_caching_options(False)  # fresh market data every run
    ingest.set_display_name("Ingest BTC-USD data")

    feats = generate_features(raw_data=ingest.outputs["raw_data"])
    feats.set_display_name("Feature engineering")

    train = train_model(features=feats.outputs["features"], mape_threshold=mape_threshold)
    train.set_display_name("Train & evaluate model")
    train.set_cpu_request("500m").set_cpu_limit("2").set_memory_limit("2G")

    report = log_metrics(summary=train.outputs["summary"])
    report.set_display_name("Log metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="kubeflow/btc_kfp_pipeline.yaml")
    parser.add_argument("--submit", action="store_true", help="create a daily recurring run")
    parser.add_argument("--host", default="http://localhost:8080", help="KFP API endpoint")
    parser.add_argument("--cron", default="0 30 0 * * *", help="KFP 6-field cron (sec first)")
    args = parser.parse_args()

    compiler.Compiler().compile(btc_forecast_pipeline, package_path=args.output)
    print(f"Compiled pipeline -> {args.output} (image: {TRAINER_IMAGE})")

    if args.submit:
        import kfp

        client = kfp.Client(host=args.host)
        experiment = client.create_experiment("btc-forecasting")
        job = client.create_recurring_run(
            experiment_id=experiment.experiment_id,
            job_name="btc-daily-retrain",
            pipeline_package_path=args.output,
            cron_expression=args.cron,
            max_concurrency=1,
        )
        print(f"Recurring run created: {job.recurring_run_id}")


if __name__ == "__main__":
    main()
