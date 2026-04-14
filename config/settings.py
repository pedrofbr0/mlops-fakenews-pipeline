"""Centralized configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class Settings:
    # GCP
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")
    BQ_DATASET: str = os.getenv("BQ_DATASET", "fakenews_ml")
    BQ_RAW_TABLE: str = os.getenv("BQ_RAW_TABLE", "raw_tweets")
    BQ_FEATURES_TABLE: str = os.getenv("BQ_FEATURES_TABLE", "features")
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fakenews-detection")

    # Model
    MODEL_NAME: str = os.getenv("MODEL_NAME", "fakenews_classifier")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "latest")
    API_PORT: int = int(os.getenv("API_PORT", "8080"))

    # Monitoring
    DRIFT_THRESHOLD_PSI: float = float(os.getenv("DRIFT_THRESHOLD_PSI", "0.2"))
    DRIFT_CHECK_INTERVAL_HOURS: int = int(os.getenv("DRIFT_CHECK_INTERVAL_HOURS", "6"))

    @property
    def bq_raw_table_id(self) -> str:
        return f"{self.GCP_PROJECT_ID}.{self.BQ_DATASET}.{self.BQ_RAW_TABLE}"

    @property
    def bq_features_table_id(self) -> str:
        return f"{self.GCP_PROJECT_ID}.{self.BQ_DATASET}.{self.BQ_FEATURES_TABLE}"


settings = Settings()
