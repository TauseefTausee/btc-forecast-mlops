"""Locust load test for the BTC forecast API (drives the HPA demo).

Local UI:   locust -f loadtest/locustfile.py --host http://btc-forecast.local
Headless:   locust -f loadtest/locustfile.py --host http://btc-forecast.local \
                --headless -u 80 -r 10 -t 5m --csv loadtest/results
In-cluster: scripts/load_test_k8s.sh   (runs this file as a Kubernetes Job)
"""

import random

from locust import HttpUser, between, task

BASE_PRICE = 95_000.0


def random_history(n: int = 90) -> list:
    price, out = BASE_PRICE, []
    for _ in range(n):
        price *= 1 + random.gauss(0, 0.025)
        out.append(round(price, 2))
    return out


class ForecastUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(6)
    def predict_default(self):
        self.client.post("/predict", json={"forecast_days": 7}, name="POST /predict")

    @task(3)
    def predict_custom_history(self):
        self.client.post(
            "/predict",
            json={"history": random_history(), "forecast_days": random.randint(1, 7)},
            name="POST /predict (custom history)",
        )

    @task(1)
    def health_and_metrics(self):
        self.client.get("/health", name="GET /health")
        self.client.get("/metrics", name="GET /metrics")
