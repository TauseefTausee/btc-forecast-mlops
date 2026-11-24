#!/usr/bin/env bash
# E1: delete a running pod and record Kubernetes self-healing.
# Output: docs/evidence/chaos_test.log
set -euo pipefail
cd "$(dirname "$0")/.."
NS=btc-mlops
LOG=docs/evidence/chaos_test.log
mkdir -p "$(dirname "$LOG")"
{
  echo "=== $(date '+%F %T') pods before"
  kubectl -n "$NS" get pods -l app=btc-forecast-api -o wide
  VICTIM=$(kubectl -n "$NS" get pods -l app=btc-forecast-api \
           -o jsonpath='{.items[0].metadata.name}')
  echo "=== $(date '+%T') deleting $VICTIM"
  kubectl -n "$NS" delete pod "$VICTIM" --wait=false
  for i in $(seq 1 12); do
    sleep 5
    echo "=== $(date '+%T') +$((i*5))s"
    kubectl -n "$NS" get pods -l app=btc-forecast-api
  done
  echo "=== replica set status"
  kubectl -n "$NS" get deployment btc-forecast-api
  echo "=== recent events"
  kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -15
} 2>&1 | tee "$LOG"
