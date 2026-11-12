#!/usr/bin/env bash
# Initialise DVC and use the MinIO bucket as remote storage for the data.
# Run once, after `git init` and `docker compose up -d`.
set -euo pipefail

BUCKET="${DVC_BUCKET:-dvc}"
ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"

[ -d .dvc ] || dvc init
dvc remote add -d -f minio "s3://${BUCKET}"
dvc remote modify minio endpointurl "${ENDPOINT}"
# Credentials are NOT written in the repo: DVC reads AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY from the environment (.env).

echo "DVC remote configured -> s3://${BUCKET} (${ENDPOINT})"
