#!/usr/bin/env bash
# Hit every endpoint once. Usage: scripts/smoke_test.sh [base_url] [api_token]
#   scripts/smoke_test.sh http://btc-forecast.local
#   scripts/smoke_test.sh "$(minikube service btc-forecast-api-nodeport -n btc-mlops --url)"
set -euo pipefail
BASE="${1:-http://btc-forecast.local}"
TOKEN="${2:-change-me-to-a-long-random-string}"
pp() { python -m json.tool 2>/dev/null || cat; }

echo "== GET /health";  curl -s "$BASE/health" | pp
echo "== POST /predict (7 days)"
curl -s -X POST "$BASE/predict" -H "Content-Type: application/json" -d '{"forecast_days": 7}' | pp
echo "== POST /predict (bad input -> 400)"
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/predict" -H "Content-Type: application/json" \
  -d '{"forecast_days": 30}'
echo "== GET /metrics"; curl -s "$BASE/metrics" | pp
echo "== POST /reload";  curl -s -X POST "$BASE/reload" -H "X-API-Token: $TOKEN" | pp
