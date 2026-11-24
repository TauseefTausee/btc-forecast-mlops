#!/usr/bin/env bash
# D5: run the Locust job inside the cluster and record HPA scaling.
# Output: loadtest/results/{hpa_watch.log,pods_watch.log,top_pods.log,locust_summary.log}
set -euo pipefail
cd "$(dirname "$0")/.."
NS=btc-mlops
OUT=loadtest/results
mkdir -p "$OUT"

kubectl -n "$NS" create configmap locustfile --from-file=locustfile.py=loadtest/locustfile.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NS" delete job locust-load-test --ignore-not-found

echo ">> Recording HPA / pods while the load runs (Ctrl+C stops everything)"
kubectl -n "$NS" get hpa btc-forecast-api --watch --output-watch-events > "$OUT/hpa_watch.log" &
HPA_PID=$!
kubectl -n "$NS" get pods -l app=btc-forecast-api --watch --output-watch-events > "$OUT/pods_watch.log" &
PODS_PID=$!
trap 'kill $HPA_PID $PODS_PID 2>/dev/null || true' EXIT

kubectl apply -f loadtest/locust-job.yaml
: > "$OUT/top_pods.log"
for i in $(seq 1 24); do              # ~6 minutes of samples, every 15 s
  { echo "--- $(date +%T)"; kubectl -n "$NS" top pods -l app=btc-forecast-api 2>&1; \
    kubectl -n "$NS" get hpa btc-forecast-api --no-headers; } | tee -a "$OUT/top_pods.log"
  sleep 15
done

kubectl -n "$NS" wait --for=condition=complete job/locust-load-test --timeout=180s || true
kubectl -n "$NS" logs job/locust-load-test > "$OUT/locust_summary.log" || true
echo ">> Load finished. Keep watching scale-down with: kubectl -n $NS get hpa --watch"
tail -25 "$OUT/locust_summary.log"
