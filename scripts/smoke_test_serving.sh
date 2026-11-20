#!/usr/bin/env bash
# Smoke test of the serving images: start the API with a throw-away MLflow store, register the
# placeholder model INSIDE the container, then check that /health, /v1/predict and /metrics answer.
# Also checks that the web app container starts. If anything fails, the script exits non-zero.
#   bash scripts/smoke_test_serving.sh ride-api:ci ride-webapp:ci
set -euo pipefail

API_IMAGE="${1:?usage: smoke_test_serving.sh <api-image> <webapp-image>}"
WEB_IMAGE="${2:?usage: smoke_test_serving.sh <api-image> <webapp-image>}"
API_NAME="smoke-api-$$"
WEB_NAME="smoke-web-$$"

cleanup() {
  docker logs "$API_NAME" 2>&1 | tail -n 20 || true
  docker rm -f "$API_NAME" "$WEB_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for() {  # wait_for <url> <seconds>
  for _ in $(seq 1 "$2"); do
    curl -fs "$1" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "Timeout waiting for $1" >&2
  return 1
}

# 1) API: the sqlite store lives in the container; register the placeholder, then serve it.
#    /app is read-only for the non-root user and MLflow writes ./mlruns artifacts, hence `cd /tmp` first.
docker run -d --name "$API_NAME" -p 18000:8000 \
  -e MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db \
  "$API_IMAGE" \
  sh -c "cd /tmp && python -m fraud_detection.api.dummy_model && exec uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000" \
  >/dev/null
wait_for http://localhost:18000/health 90

HEALTH="$(curl -fs http://localhost:18000/health)"
echo "health: $HEALTH"
echo "$HEALTH" | grep -q '"status":"ok"'

# 2) A real prediction (100 samples = the minimum), then the contract checks
BODY="$(python3 -c "import json; print(json.dumps({'ride_id': 'smoke', 'samples': [dict(ax=0.1*(i%7), ay=-9.1, az=-2.3, rx=0.0, ry=0.1*(i%5), rz=0.0) for i in range(100)]}))")"
PRED="$(curl -fs -X POST http://localhost:18000/v1/predict -H 'content-type: application/json' -d "$BODY")"
echo "prediction: $PRED"
echo "$PRED" | grep -q '"predicted_class"'

CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:18000/v1/predict -H 'content-type: application/json' -d '{"samples": []}')"
[ "$CODE" = "422" ] || { echo "empty ride should give 422, got $CODE" >&2; exit 1; }

curl -fs http://localhost:18000/metrics/ | grep -q 'api_requests_total{endpoint="/v1/predict",status="200"}'
curl -fs http://localhost:18000/docs >/dev/null

# 3) Web app: the container starts and Streamlit reports healthy
docker run -d --name "$WEB_NAME" -p 18501:8501 -e API_URL=http://host.docker.internal:18000 "$WEB_IMAGE" >/dev/null
wait_for http://localhost:18501/_stcore/health 60

echo "Smoke test passed for $API_IMAGE and $WEB_IMAGE"
