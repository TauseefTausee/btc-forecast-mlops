# Runbook: run everything and collect report evidence

Commands assume the repo root. On Windows, use **Git Bash** for the `scripts/*.sh` files
(PowerShell works for the single commands). 📸 = take a screenshot for the report.

## 0. Prerequisites

Docker Desktop, Minikube, kubectl, Helm (optional), Python 3.10+, a GitHub account and a
Docker Hub account.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt -r loadtest/requirements.txt
```

## 1. Part A: API and tests

```bash
pytest -v                                   # 📸 28 passed
flake8 . && black --check .
```

## 2. Part B: Docker, compose, CI/CD

1. Replace `YOUR_DOCKERHUB_USER` everywhere with your Docker Hub username:
   ```bash
   grep -rl YOUR_DOCKERHUB_USER k8s kubeflow | xargs sed -i 's/YOUR_DOCKERHUB_USER/<your-user>/g'
   ```
2. Build the image locally and check its size:
   ```bash
   docker build -f app/Dockerfile -t btc-forecast-api:latest .     # 📸 build output
   docker images btc-forecast-api                                   # 📸 size
   ```
3. Start the local stack (API + Airflow + Postgres):
   ```bash
   cp .env.example .env          # Linux only: set AIRFLOW_UID=$(id -u) in .env
   docker compose up -d --build
   docker compose ps             # 📸 all services healthy
   curl localhost:5000/health
   ```
   The first start trains a bootstrap model (`model-init`) from live BTC data.
4. GitHub: create the repo, then under **Settings → Secrets and variables → Actions** add
   `DOCKER_USERNAME` and `DOCKER_PASSWORD` (a Docker Hub access token is best).
   Push to `main`:
   ```bash
   git init && git add . && git commit -m "BTC forecasting MLOps pipeline"
   git branch -M main && git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   📸 the green Actions run (both jobs) and the Docker Hub repo showing the `latest` and SHA tags.
   Make the Docker Hub repos public, since that link is a deliverable.

## 3. Part C: Airflow and Kubeflow

**Airflow:** open http://localhost:8080 (admin/admin). The `btc_etl_pipeline` DAG is unpaused.
Trigger it with ▶ (or `docker compose exec airflow-scheduler airflow dags trigger btc_etl_pipeline`).
📸 Graph view with all tasks green, and 📸 the `train_evaluate` log line showing MAE/RMSE/MAPE.

**Kubeflow:** compile the spec (CI also does this):
```bash
TRAINER_IMAGE=docker.io/<your-user>/btc-forecast-trainer:latest \
  python kubeflow/btc_kfp_pipeline.py --output kubeflow/btc_kfp_pipeline.yaml
```
📸 the compiled YAML. If you have a KFP install (`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`),
upload the YAML in the UI and run it for a 📸 of the run graph. Installing full KFP on Minikube is
heavy (around 8 GB RAM), so the compiled YAML is usually enough. Confirm with your instructor.

## 4. Part D: Kubernetes

```bash
minikube start --cpus=4 --memory=6g --driver=docker
```

**Share the model store with Airflow** (lets the end-to-end test in step 5 work). Keep this
running in its own terminal, and start it **before** deploying:
```bash
minikube mount "$(pwd)/shared/models:/mnt/btc-models" --uid 10001 --gid 10001
# Windows PowerShell: minikube mount "${PWD}\shared\models:/mnt/btc-models" --uid 10001 --gid 10001
```
(If you skip the mount, the init container trains its own model inside the cluster. Everything
works except Airflow updates reaching Kubernetes.)

Set a real `API_TOKEN` in `.env` (gitignored; `deploy_k8s.sh` loads it into the `btc-api-secret` Secret), then deploy:
```bash
scripts/deploy_k8s.sh docker.io/<your-user>/btc-forecast-api:latest
# or, without Docker Hub: minikube image load btc-forecast-api:latest && scripts/deploy_k8s.sh btc-forecast-api:latest
kubectl -n btc-mlops get pods,svc,hpa,ingress,pvc     # 📸
kubectl -n btc-mlops describe deployment btc-forecast-api   # 📸 probes + limits
```

**Ingress host.** Add `btc-forecast.local` to your hosts file:
- Windows/macOS (docker driver): run `minikube tunnel` in an admin terminal and add
  `127.0.0.1 btc-forecast.local` to `C:\Windows\System32\drivers\etc\hosts`.
- Linux: `echo "$(minikube ip) btc-forecast.local" | sudo tee -a /etc/hosts`

```bash
scripts/smoke_test.sh http://btc-forecast.local <API_TOKEN>      # 📸 predictions JSON
minikube service btc-forecast-api-nodeport -n btc-mlops --url    # NodePort URL (alternative)
```

**HPA load test (D5).** In one terminal: `kubectl -n btc-mlops get hpa --watch` 📸. In another:
```bash
scripts/load_test_k8s.sh
```
Results land in `loadtest/results/` (`hpa_watch.log`, `top_pods.log`, `locust_summary.log`).
📸 replicas going 2 → up to 8 and back down. For a graph, run Locust with its web UI against the
ingress: `locust -f loadtest/locustfile.py --host http://btc-forecast.local` → http://localhost:8089
📸 the Charts tab (RPS, response times, users).

## 5. Part E: verification

```bash
scripts/chaos_test.sh                          # E1 -> docs/evidence/chaos_test.log  📸
scripts/e2e_verify.sh http://btc-forecast.local  # E2 -> docs/evidence/e2e_verify.log 📸
```
`e2e_verify.sh` triggers the Airflow DAG with `force_register`, waits for it to finish, and checks
that the model version served by Kubernetes changed (pods hot-reload within ~15 s through the
shared PVC). It then prints a live 7-day forecast.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Pods stuck in `Init:` | `kubectl -n btc-mlops logs <pod> -c model-bootstrap`. Usually no internet from Minikube, or Yahoo rate limiting (it falls back to Binance/CoinGecko automatically). |
| `ImagePullBackOff` | Image name still says `YOUR_DOCKERHUB_USER`, or the Docker Hub repo is private. |
| HPA shows `<unknown>` | `minikube addons enable metrics-server` and wait about a minute. |
| Pods `Pending` when scaling | Minikube too small: `minikube start --cpus=4 --memory=6g`. |
| Airflow tasks can't write `shared/models` (Linux) | Set `AIRFLOW_UID=$(id -u)` in `.env`, then `docker compose up -d`. |
| `reload_api` task fails | API container not running: `docker compose up -d api`. |
