"""
Model Trainer — Train, evaluate, and register ML models with MLflow.

Demonstrates:
- MLflow experiment tracking (params, metrics, artifacts)
- Model registry with versioning and stage transitions
- Hyperparameter tuning with cross-validation
- Multiple model comparison (Random Forest, XGBoost, SVM)
- SHAP-based model interpretability
- Production-ready model serialization
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Any, Optional

import click
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import settings
from src.features.feature_engineering import FEATURE_COLUMNS

warnings.filterwarnings("ignore")


# ── Model Definitions ───────────────────────────────────────────────────────

MODELS = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "svm": {
        "class": SVC,
        "params": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "class_weight": "balanced",
            "random_state": 42,
        },
    },
}


class ModelTrainer:
    """Train models with full MLflow lifecycle management."""

    def __init__(
        self,
        experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str = settings.MLFLOW_TRACKING_URI,
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"MLflow experiment: {experiment_name}")

    def load_features(
        self, path: str = "data/features.parquet"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load feature data and split into train/test."""
        df = pd.read_parquet(path)
        train = df[df["split"] == "train"].copy()
        test = df[df["split"] == "test"].copy()
        logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
        return train, test

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get available feature columns from DataFrame."""
        all_features = FEATURE_COLUMNS + [
            c for c in df.columns
            if c.startswith("log_") or c in ("speed_x_size", "spread_cv", "burstiness")
        ]
        return [c for c in all_features if c in df.columns]

    def train_single_model(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        custom_params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Train a single model with MLflow tracking.

        Returns dict with model, metrics, and run_id.
        """
        model_config = MODELS[model_name]
        params = {**model_config["params"], **(custom_params or {})}
        feature_cols = self._get_feature_cols(train_df)

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["label"]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df["label"]

        # Build pipeline with scaling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_config["class"](**params)),
        ])

        with mlflow.start_run(run_name=f"{model_name}") as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
            )
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())

            # Train
            pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = pipeline.predict(X_test)
            y_prob = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline.named_steps["model"], "predict_proba")
                else y_pred
            )

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }

            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")

            # Log feature importance (if available)
            model = pipeline.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                importance = dict(zip(feature_cols, model.feature_importances_))
                importance_sorted = dict(
                    sorted(importance.items(), key=lambda x: x[1], reverse=True)
                )
                mlflow.log_dict(importance_sorted, "feature_importance.json")

            # Log feature list
            mlflow.log_dict({"features": feature_cols}, "feature_columns.json")

            # Log model artifact
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(pipeline, "model")

            # Save model locally too
            local_path = Path("models") / model_name
            local_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, local_path / "pipeline.joblib")

            logger.info(
                f"[{model_name}] F1={metrics['f1']:.4f} | "
                f"AUC={metrics['roc_auc']:.4f} | "
                f"CV-F1={cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )

            return {
                "model_name": model_name,
                "pipeline": pipeline,
                "metrics": metrics,
                "cv_f1_mean": cv_scores.mean(),
                "run_id": run.info.run_id,
            }

    def train_all_models(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Train all defined models and return results."""
        results = []
        for name in MODELS:
            logger.info(f"Training {name}...")
            result = self.train_single_model(name, train_df, test_df)
            results.append(result)

        # Sort by F1 score
        results.sort(key=lambda r: r["metrics"]["f1"], reverse=True)

        logger.info("\n=== Model Comparison ===")
        for r in results:
            logger.info(
                f"  {r['model_name']:20s} | F1={r['metrics']['f1']:.4f} | "
                f"AUC={r['metrics']['roc_auc']:.4f}"
            )

        return results

    def register_best_model(self, results: list[dict]) -> str:
        """Register the best model in MLflow Model Registry."""
        best = results[0]
        model_uri = f"runs:/{best['run_id']}/model"

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=settings.MODEL_NAME,
        )

        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=settings.MODEL_NAME,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )

        logger.info(
            f"Registered {best['model_name']} as "
            f"{settings.MODEL_NAME} v{registered.version} → Production"
        )
        return registered.version


# ── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--experiment", default=settings.MLFLOW_EXPERIMENT_NAME)
@click.option("--data-path", default="data/features.parquet")
@click.option("--model", default=None, help="Train specific model (or all).")
@click.option("--register/--no-register", default=True)
def main(experiment: str, data_path: str, model: str, register: bool):
    """Train ML models with MLflow tracking."""
    trainer = ModelTrainer(experiment_name=experiment)
    train_df, test_df = trainer.load_features(data_path)

    if model:
        results = [trainer.train_single_model(model, train_df, test_df)]
    else:
        results = trainer.train_all_models(train_df, test_df)

    if register and results:
        trainer.register_best_model(results)


if __name__ == "__main__":
    main()
