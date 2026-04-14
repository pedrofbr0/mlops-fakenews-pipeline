# 🚀 Fake News Detection — MLOps Pipeline on GCP

End-to-end ML pipeline for fake news classification, built with production-grade MLOps practices on Google Cloud Platform.

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Raw Data    │───▶│   BigQuery   │───▶│  Feature     │───▶│   Model      │
│  (CSV/API)   │    │  Data Lake   │    │  Engineering │    │  Training    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘
                                                                 │
                   ┌──────────────┐    ┌─────────────┐           │
                   │  Monitoring  │◀───│  Cloud Run   │◀──── MLflow
                   │  (Drift)     │    │  (FastAPI)   │    Model Registry
                   └──────────────┘    └─────────────┘
                          │
                   Airflow (Orchestration)
```

## Tech Stack

| Layer              | Technology                        |
|--------------------|-----------------------------------|
| Data Warehouse     | BigQuery                          |
| Feature Store      | BigQuery + custom feature views   |
| Training           | Scikit-learn, XGBoost             |
| Experiment Track   | MLflow                            |
| Model Serving      | FastAPI + Cloud Run               |
| Monitoring         | Evidently AI + custom drift checks|
| Orchestration      | Apache Airflow (Cloud Composer)   |
| CI/CD              | Cloud Build + Docker              |
| Infrastructure     | Docker, GCP                       |

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# GCP CLI
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Set Environment Variables

```bash
cp config/.env.example config/.env
# Edit config/.env with your GCP project details
source config/.env
```

### 3. Load Data into BigQuery

```bash
python -m src.data.bigquery_loader --source data/raw/fake_news.csv
```

### 4. Run Feature Engineering

```bash
python -m src.features.feature_engineering
```

### 5. Train Model (with MLflow tracking)

```bash
# Start MLflow server (local)
mlflow server --host 0.0.0.0 --port 5000 &

# Train
python -m src.train.trainer --experiment "fakenews-v1"
```

### 6. Serve Model (FastAPI)

```bash
# Local
uvicorn src.serve.app:app --host 0.0.0.0 --port 8080

# Docker
docker build -t fakenews-api .
docker run -p 8080:8080 fakenews-api
```

### 7. Deploy to Cloud Run

```bash
gcloud builds submit --config cloudbuild.yaml
```

## Project Structure

```
mlops-fakenews-pipeline/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── cloudbuild.yaml
├── config/
│   ├── .env.example
│   └── settings.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── bigquery_loader.py      # Data ingestion → BigQuery
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py   # Feature extraction & storage
│   ├── train/
│   │   ├── __init__.py
│   │   └── trainer.py               # Model training + MLflow
│   ├── serve/
│   │   ├── __init__.py
│   │   └── app.py                   # FastAPI inference server
│   └── monitor/
│       ├── __init__.py
│       └── drift_detector.py        # Data & model drift monitoring
├── dags/
│   └── ml_pipeline_dag.py           # Airflow orchestration DAG
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
└── scripts/
    ├── setup_bigquery.sh
    └── deploy.sh
```

## Key Features

- **Production ML deployment**
- **BigQuery** for data warehouse (partitioned, clustered tables)
- **MLflow** for experiment tracking + model versioning
- **FastAPI** with async inference, health checks, input validation
- **Drift monitoring** with statistical tests (PSI, KS-test)
- **Airflow DAG** for automated retraining pipeline
- **Docker + Cloud Run** for scalable serving
- **CI/CD** with Cloud Build