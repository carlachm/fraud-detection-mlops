# Serving image: the ride classification API (FastAPI + MLflow model).
#   docker build -f docker/api/Dockerfile -t ride-api .
#   docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 ride-api
#
# The model is loaded at startup from the MLflow registry (MODEL_URI, default models:/ride-classifier@champion),
# or from a directory mounted into the container (MODEL_URI=/models/current).
FROM python:3.12-slim

# uv pinned (never latest), copied from its official image
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PREDICTION_LOG_PATH=/app/logs/predictions.jsonl

WORKDIR /app

# 1) Dependencies only: this layer stays cached until uv.lock changes
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 2) Then the project code
COPY README.md params.yaml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Do not run as root; the prediction log is the only place the API writes to
RUN useradd --create-home --uid 10001 app \
    && mkdir -p logs \
    && chown -R app:app /app/logs
USER app

EXPOSE 8000

# Kubernetes uses its own probes (/health); this one makes `docker ps` meaningful.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]

# Exec form: uvicorn is PID 1 and receives SIGTERM (clean shutdown on `docker stop` / pod eviction)
CMD ["uvicorn", "fraud_detection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
