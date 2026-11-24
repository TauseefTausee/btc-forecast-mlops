#!/usr/bin/env bash
# E2: trigger the Airflow DAG, wait for it, confirm the model version changed
# and query the live Kubernetes endpoint. Run from the repo root with
# docker-compose up and the Minikube deployment running.
#   scripts/e2e_verify.sh [k8s_base_url]
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${1:-http://btc-forecast.local}"
LOG=docs/evidence/e2e_verify.log
mkdir -p "$(dirname "$LOG")"
AF="docker compose exec -T airflow-scheduler airflow"

{
  echo "=== model served by Kubernetes BEFORE"
  curl -s "$BASE/health"; echo
  BEFORE=$(curl -s "$BASE/health" | python -c "import sys,json;print(json.load(sys.stdin)['model_version'])")

  RUN_ID="e2e_$(date +%Y%m%dT%H%M%S)"
  echo "=== triggering DAG run $RUN_ID (force_register so the model always updates)"
  $AF dags trigger btc_etl_pipeline --run-id "$RUN_ID" --conf '{"force_register": true}'

  for i in $(seq 1 60); do
    STATE=$($AF dags state btc_etl_pipeline "$RUN_ID" 2>/dev/null | tail -1 | tr -d '\r')
    echo "$(date +%T) DAG state: $STATE"
    [[ "$STATE" == "success" || "$STATE" == "failed" ]] && break
    sleep 10
  done
  $AF tasks states-for-dag-run btc_etl_pipeline "$RUN_ID" || true

  echo "=== waiting for pods to hot-reload the new model (MODEL_REFRESH_SECONDS=15)"
  sleep 20
  echo "=== model served by Kubernetes AFTER"
  curl -s "$BASE/health"; echo
  AFTER=$(curl -s "$BASE/health" | python -c "import sys,json;print(json.load(sys.stdin)['model_version'])")
  echo "before=$BEFORE after=$AFTER"
  [[ "$BEFORE" != "$AFTER" ]] && echo "MODEL UPDATED" || echo "WARNING: model version unchanged"

  echo "=== live 7-day forecast from Kubernetes"
  curl -s -X POST "$BASE/predict" -H "Content-Type: application/json" -d '{"forecast_days": 7}' \
    | python -m json.tool
  echo "=== metrics"
  curl -s "$BASE/metrics" | python -m json.tool
} 2>&1 | tee "$LOG"
