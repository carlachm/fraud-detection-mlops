# Shortcuts for the most common commands. `make help` lists them.
# Every Python command goes through `uv run` (no venv activation needed).
-include .env
export

RUN = uv run
IMAGE ?= fraud-detection-train

.PHONY: help install lock up down logs dvc-setup data sample prepare train repro push test lint format docker-build docker-smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv exactly as described in uv.lock (project + dev tools)
	uv sync --locked

lock:  ## Update uv.lock after editing pyproject.toml (then commit uv.lock)
	uv lock

up:  ## Start the dev stack (MinIO, PostgreSQL, MLflow)
	docker compose up -d --build

down:  ## Stop the dev stack (data is kept in Docker volumes)
	docker compose down

logs:  ## Follow the dev stack logs
	docker compose logs -f

dvc-setup:  ## Init DVC with the MinIO remote
	bash scripts/setup_dvc.sh

data:  ## Download the Kaggle dataset and version it with DVC
	$(RUN) python -m fraud_detection.data download
	$(RUN) dvc add data/raw/creditcard.csv
	$(RUN) dvc push

sample:  ## Generate a small synthetic dataset (no Kaggle account needed)
	$(RUN) python -m fraud_detection.data sample --rows 20000

prepare:  ## Validate + split the raw data
	$(RUN) python -m fraud_detection.data prepare

train:  ## Train, log to MLflow, register the model (fails if quality gate fails)
	$(RUN) python -m fraud_detection.train

repro:  ## Re-run the DVC pipeline (prepare -> train) when something changed
	$(RUN) dvc repro

push:  ## Push data/pipeline outputs to the DVC remote
	$(RUN) dvc push

test:  ## Run the test suite with coverage
	$(RUN) pytest --cov=fraud_detection --cov-report=term-missing

lint:  ## Lint + check formatting (same as the CI)
	uv lock --check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## Auto-format the code
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

docker-build:  ## Build the training image locally
	docker build -t $(IMAGE):dev .

docker-smoke: docker-build  ## Build + smoke test the image (same as the CI)
	bash scripts/smoke_test.sh $(IMAGE):dev

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov mlruns coverage.xml junit.xml
	find . -name __pycache__ -prune -exec rm -rf {} +
