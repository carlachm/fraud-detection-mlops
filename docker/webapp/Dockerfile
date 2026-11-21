# Web app image: Streamlit UI that calls the API. No torch / mlflow inside, so the image stays small.
#   docker build -f docker/webapp/Dockerfile -t ride-webapp .
#   docker run --rm -p 8501:8501 -e API_URL=http://host.docker.internal:8000 ride-webapp
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Only the `webapp` dependency group (streamlit, requests, pandas...): --only-group skips the project AND its
# dependencies (torch, mlflow...). `--no-install-project --group webapp` would still pull ~1 GB of them.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group webapp

COPY webapp ./webapp

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "webapp/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
