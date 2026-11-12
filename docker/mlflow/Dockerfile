# MLflow tracking server + model registry
# - metadata (runs, params, metrics, registry) -> PostgreSQL
# - artifacts (models, plots)                  -> MinIO (S3), proxied by the server
FROM python:3.12-slim

ARG MLFLOW_VERSION=3.17.2

RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        psycopg2-binary==2.9.11 \
        boto3==1.43.117

EXPOSE 5000

# --serve-artifacts: clients upload/download artifacts THROUGH the MLflow server,
# so training scripts and the API don't need S3 credentials.
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACTS_DESTINATION}" \
    --serve-artifacts \
    --allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-*}"
