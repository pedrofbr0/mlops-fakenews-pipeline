"""
Local Pipeline Runner — Run the full ML pipeline without GCP.

This script simulates the full pipeline locally, useful for:
- Testing before deploying to GCP
- Portfolio demos without cloud costs
- CI/CD pipeline validation

Usage:
    python scripts/run_local_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def step_1_generate_data() -> pd.DataFrame:
    """Step 1: Generate synthetic data (replaces BigQuery ingestion)."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating synthetic data...")

    from scripts.generate_sample_data import generate_cascade_features

    rng = np.random.default_rng(42)
    fake = generate_cascade_features(2000, label=1, rng=rng)
    real = generate_cascade_features(3000, label=0, rng=rng)
    df = pd.concat([fake, real], ignore_index=True).sample(frac=1, random_state=42)

    output_path = Path("data/raw/fake_news.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Generated {len(df)} samples → {output_path}")
    return df


def step_2_feature_engineering(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Feature engineering (local mode, no BigQuery)."""
    logger.info("=" * 60)
    logger.info("STEP 2: Running feature engineering...")

    from src.features.feature_engineering import FeatureEngineer, FEATURE_COLUMNS

    df = raw_df.copy()

    # Add derived features that BigQuery SQL would compute
    df["virality_ratio"] = df["retweet_count"] / df["followers_count"].clip(lower=1)
    df["engagement_ratio"] = df["favorite_count"] / df["followers_count"].clip(lower=1)
    df["ff_ratio"] = df["friends_count"] / df["followers_count"].clip(lower=1)
    df["depth_ratio"] = df["cascade_depth"] / df["cascade_size"].clip(lower=1)
    df["breadth_depth_ratio"] = df["cascade_max_breadth"] / df["cascade_depth"].clip(lower=1)

    hour = pd.to_datetime(df["created_at"]).dt.hour
    dow = pd.to_datetime(df["created_at"]).dt.dayofweek + 1
    df["hour_of_day"] = hour
    df["day_of_week"] = dow

    df["cascade_size_pctrank"] = df.groupby("label")["cascade_size"].rank(pct=True)
    df["speed_pctrank"] = df.groupby("label")["spread_speed_mean"].rank(pct=True)

    # Deterministic split
    df["split"] = "train"
    df.loc[df.index % 5 == 0, "split"] = "test"

    # Add local features (log transforms, interactions)
    engineer = FeatureEngineer.__new__(FeatureEngineer)
    df = engineer.add_local_features(df)

    # Save
    features_path = Path("data/features.parquet")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path, index=False)

    logger.info(f"Engineered {len(df.columns)} features → {features_path}")
    logger.info(f"Train: {(df['split']=='train').sum()} | Test: {(df['split']=='test').sum()}")
    return df


def step_3_train_models(features_df: pd.DataFrame) -> dict:
    """Step 3: Train models with MLflow tracking."""
    logger.info("=" * 60)
    logger.info("STEP 3: Training models with MLflow...")

    import mlflow
    from src.train.trainer import ModelTrainer

    # Use local file-based MLflow (no server needed)
    mlflow.set_tracking_uri("file:./mlruns")

    trainer = ModelTrainer(
        experiment_name="fakenews-local-demo",
        tracking_uri="file:./mlruns",
    )

    train_df = features_df[features_df["split"] == "train"]
    test_df = features_df[features_df["split"] == "test"]

    results = trainer.train_all_models(train_df, test_df)

    best = results[0]
    logger.info(f"\nBest model: {best['model_name']}")
    logger.info(f"  F1:  {best['metrics']['f1']:.4f}")
    logger.info(f"  AUC: {best['metrics']['roc_auc']:.4f}")

    return best


def step_4_check_drift(features_df: pd.DataFrame) -> None:
    """Step 4: Run drift detection demo."""
    logger.info("=" * 60)
    logger.info("STEP 4: Running drift detection...")

    from src.monitor.drift_detector import DriftDetector

    # Split data to simulate reference vs production
    mid = len(features_df) // 2
    reference = features_df.iloc[:mid]
    production = features_df.iloc[mid:]

    detector = DriftDetector()
    report = detector.check_all_features(reference, production)

    if detector.should_retrain(report):
        logger.warning("Retraining recommended!")
    else:
        logger.info("No significant drift detected.")


def step_5_test_api() -> None:
    """Step 5: Test the FastAPI endpoints."""
    logger.info("=" * 60)
    logger.info("STEP 5: Testing API endpoints...")

    from fastapi.testclient import TestClient
    from src.serve.app import app

    client = TestClient(app)

    # Health check
    r = client.get("/health")
    logger.info(f"Health: {r.status_code} → {r.json()['status']}")

    # Prediction (will return 503 if no model loaded — that's ok for demo)
    payload = {
        "retweet_count": 500,
        "favorite_count": 120,
        "followers_count": 3000,
        "friends_count": 800,
        "cascade_size": 45,
        "cascade_depth": 5,
        "cascade_max_breadth": 12,
        "spread_speed_mean": 25.5,
        "spread_speed_std": 60.3,
        "hour_of_day": 2,
        "day_of_week": 6,
    }
    r = client.post("/predict", json=payload)
    if r.status_code == 200:
        pred = r.json()
        label = "FAKE" if pred["label"] == 1 else "REAL"
        logger.info(
            f"Prediction: {label} (prob={pred['probability']:.3f}, "
            f"confidence={pred['confidence']})"
        )
    else:
        logger.info(f"Predict returned {r.status_code} (model not loaded in test mode)")

    # Metrics
    r = client.get("/metrics")
    logger.info(f"Metrics: {r.json()}")


def main():
    """Run the full pipeline locally."""
    logger.info("🚀 FAKE NEWS DETECTION — LOCAL PIPELINE RUN")
    logger.info("=" * 60)

    raw_df = step_1_generate_data()
    features_df = step_2_feature_engineering(raw_df)
    best_model = step_3_train_models(features_df)
    step_4_check_drift(features_df)
    step_5_test_api()

    logger.info("=" * 60)
    logger.info("✅ Pipeline complete!")
    logger.info(f"Best model: {best_model['model_name']} (F1={best_model['metrics']['f1']:.4f})")
    logger.info("MLflow UI: mlflow ui --port 5000  (then open http://localhost:5000)")
    logger.info("API server: make serve")
    logger.info("Deploy to GCP: make deploy GCP_PROJECT_ID=your-project")


if __name__ == "__main__":
    main()
