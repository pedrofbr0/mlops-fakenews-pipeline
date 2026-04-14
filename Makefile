.PHONY: help install test lint train serve deploy docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────────────

install: ## Install dependencies
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Run: source .venv/bin/activate"

# ── Development ─────────────────────────────────────────────────────────────

test: ## Run tests
	pytest tests/ -v --tb=short

lint: ## Lint code
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/

# ── Pipeline Steps ──────────────────────────────────────────────────────────

setup-bq: ## Create BigQuery dataset and tables
	bash scripts/setup_bigquery.sh $(GCP_PROJECT_ID)

ingest: ## Load data into BigQuery
	python -m src.data.bigquery_loader --source data/raw/fake_news.csv

features: ## Run feature engineering
	python -m src.features.feature_engineering

train: ## Train all models with MLflow
	python -m src.train.trainer --experiment fakenews-v1

train-single: ## Train a single model (usage: make train-single MODEL=xgboost)
	python -m src.train.trainer --model $(MODEL) --experiment fakenews-v1

monitor: ## Run drift detection
	python -m src.monitor.drift_detector

# ── Serving ─────────────────────────────────────────────────────────────────

serve: ## Start FastAPI server locally
	uvicorn src.serve.app:app --host 0.0.0.0 --port 8080 --reload

mlflow-ui: ## Start MLflow tracking UI
	mlflow server --host 0.0.0.0 --port 5000

# ── Docker ──────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t fakenews-api .

docker-run: ## Run API in Docker
	docker run -p 8080:8080 fakenews-api

docker-up: ## Start all services (API + MLflow)
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-train: ## Run training in Docker
	docker compose --profile training up trainer

# ── Deployment ──────────────────────────────────────────────────────────────

deploy: ## Deploy to Cloud Run
	bash scripts/deploy.sh $(GCP_PROJECT_ID)

# ── Cleanup ─────────────────────────────────────────────────────────────────

clean: ## Remove generated files
	rm -rf __pycache__ .pytest_cache models/ data/features.parquet
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
