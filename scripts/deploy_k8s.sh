#!/usr/bin/env bash
# Deploy the BTC forecast API to Minikube in dependency order.
# Usage: scripts/deploy_k8s.sh [image]     e.g. scripts/deploy_k8s.sh docker.io/me/btc-forecast-api:latest
set -euo pipefail
cd "$(dirname "$0")/.."
NS=btc-mlops
IMAGE="${1:-}"

echo ">> Enabling addons (metrics-server for HPA, ingress for NGINX)"
minikube addons enable metrics-server
minikube addons enable ingress

echo ">> Applying manifests"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap-secret.yaml
# Real API_TOKEN comes from the gitignored .env, so no secret is committed to git.
if [[ -f .env ]] && grep -q '^API_TOKEN=' .env; then
  TOKEN="$(grep '^API_TOKEN=' .env | cut -d= -f2- | tr -d '\r')"
  kubectl -n "$NS" create secret generic btc-api-secret \
    --from-literal=API_TOKEN="$TOKEN" --from-literal=COINGECKO_API_KEY= \
    --dry-run=client -o yaml | kubectl apply -f -
fi
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

if [[ -n "$IMAGE" ]]; then
  echo ">> Using image $IMAGE"
  kubectl -n "$NS" set image deployment/btc-forecast-api api="$IMAGE" model-bootstrap="$IMAGE"
fi

echo ">> Waiting for the ingress controller"
kubectl -n ingress-nginx wait --for=condition=ready pod \
  -l app.kubernetes.io/component=controller --timeout=180s
kubectl apply -f k8s/ingress.yaml

echo ">> Waiting for rollout (first start may train a model in the init container)"
kubectl -n "$NS" rollout status deployment/btc-forecast-api --timeout=600s

kubectl -n "$NS" get pods,svc,hpa,ingress,pvc -o wide
