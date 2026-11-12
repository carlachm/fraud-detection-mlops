# Shortcuts for the most common commands. `make help` lists them.
-include .env
export

PYTHON ?= python
TORCH_CPU = --extra-index-url https://download.pytorch.org/whl/cpu

.PHONY: help install up down logs dvc-setup data prepare train repro push test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package + dev tools (CPU torch)
	$(PYTHON) -m pip install -r requirements-dev.txt $(TORCH_CPU)
	$(PYTHON) -m pip install -e . --no-deps

up:  ## Start the dev stack (MinIO, PostgreSQL, MLflow)
	docker compose up -d --build

down:  ## Stop the dev stack (data is kept in Docker volumes)
	docker compose down

logs:  ## Follow the dev stack logs
	docker compose logs -f

dvc-setup:  ## Init DVC with the MinIO remote
	bash scripts/setup_dvc.sh

data:  ## Download the Kaggle dataset and version it with DVC
	$(PYTHON) -m fraud_detection.data download
	dvc add data/raw/creditcard.csv
	dvc push

prepare:  ## Validate + split the raw data
	$(PYTHON) -m fraud_detection.data prepare

train:  ## Train, log to MLflow, register the model
	$(PYTHON) -m fraud_detection.train

repro:  ## Re-run the DVC pipeline (prepare -> train) when something changed
	dvc repro

push:  ## Push data/pipeline outputs to the DVC remote
	dvc push

test:  ## Run the test suite with coverage
	pytest --cov=fraud_detection --cov-report=term-missing

lint:  ## Lint + check formatting
	ruff check .
	ruff format --check .

format:  ## Auto-format the code
	ruff check --fix .
	ruff format .

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov mlruns
	find . -name __pycache__ -prune -exec rm -rf {} +
