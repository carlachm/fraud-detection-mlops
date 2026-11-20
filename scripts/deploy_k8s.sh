#!/usr/bin/env bash
# Deploy the serving stack on the current kubectl context (kind / Docker Desktop / minikube).
#   bash scripts/deploy_k8s.sh <api-image:tag> <webapp-image:tag>
#   bash scripts/deploy_k8s.sh ride-api:dev ride-webapp:dev      # images loaded locally (make k8s-up)
# Waits for the rollout, then checks /ready and a real prediction THROUGH the Service
# (kubectl port-forward works the same on kind, Docker Desktop and minikube).
set -euo pipefail

API_IMAGE="${1:?usage: deploy_k8s.sh <api-image:tag> <webapp-image:tag>}"
WEB_IMAGE="${2:?usage: deploy_k8s.sh <api-image:tag> <webapp-image:tag>}"
NS=ride
cd "$(dirname "$0")/.."

kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml
# The image is injected at deploy time so the manifests stay generic
sed "s#image: ride-api:dev#image: ${API_IMAGE}#" k8s/api.yaml | kubectl apply -f -
sed "s#image: ride-webapp:dev#image: ${WEB_IMAGE}#" k8s/webapp.yaml | kubectl apply -f -

for d in ride-api ride-webapp; do
  if ! kubectl -n "$NS" rollout status "deployment/$d" --timeout=300s; then
    kubectl -n "$NS" get pods -o wide
    kubectl -n "$NS" describe "deployment/$d" | tail -n 30
    kubectl -n "$NS" logs "deployment/$d" --tail=50 || true
    echo "Rollout of $d failed: rolling back" >&2
    kubectl -n "$NS" rollout undo "deployment/$d" || true
    exit 1
  fi
done

# Check through the Service (load-balanced over the pods)
kubectl -n "$NS" port-forward svc/ride-api 18080:80 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do curl -fs http://localhost:18080/ready >/dev/null 2>&1 && break; sleep 1; done

curl -fs http://localhost:18080/ready >/dev/null
BODY="$(python3 -c "import json; print(json.dumps({'ride_id': 'deploy-check', 'samples': [dict(ax=0.1*(i%7), ay=-9.1, az=-2.3, rx=0.0, ry=0.1*(i%5), rz=0.0) for i in range(100)]}))")"
curl -fs -X POST http://localhost:18080/v1/predict -H 'content-type: application/json' -d "$BODY" | grep -q '"predicted_class"'

kubectl -n "$NS" get pods,svc
echo "Deployed ${API_IMAGE} and ${WEB_IMAGE}"
