"""
Airflow DAG — Orchestrate the full ML pipeline.

Schedule: Daily at 2:00 AM UTC
Pipeline: Ingest → Features → Train → Evaluate → Deploy (conditional)

Demonstrates:
- DAG design with branching logic
- Task dependencies and data passing (XCom)
- Conditional deployment based on model performance
- Alerting on failure
- Idempotent, retryable tasks
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


# ── DAG Configuration ──────────────────────────────────────────────────────

default_args = {
    "owner": "pedro-baccelli",
    "depends_on_past": False,
    "email": ["pedrofbr@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


with DAG(
    dag_id="fakenews_ml_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline: ingest → features → train → deploy",
    schedule="0 2 * * *",  # Daily at 2 AM UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "fakenews", "production"],
) as dag:

    # ── Task 1: Check for new data ─────────────────────────────────────────

    @task(task_id="check_new_data")
    def check_new_data(**context) -> dict:
        """Check if there's new data to process since last run."""
        from google.cloud import bigquery

        client = bigquery.Client()

        # Count new rows since last DAG run
        last_run = context["data_interval_start"].isoformat()
        sql = f"""
            SELECT COUNT(*) as new_rows
            FROM `fakenews_ml.raw_tweets`
            WHERE ingested_at > TIMESTAMP('{last_run}')
        """
        result = client.query(sql).to_dataframe()
        new_rows = int(result["new_rows"].iloc[0])

        return {"new_rows": new_rows, "should_continue": new_rows > 0}

    # ── Task 2: Feature engineering ────────────────────────────────────────

    @task(task_id="run_feature_engineering")
    def run_feature_engineering(data_check: dict) -> str:
        """Extract and store features."""
        if not data_check["should_continue"]:
            return "skipped"

        from src.features.feature_engineering import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.run(save_to_bq=True, save_local=True)
        return f"Processed {len(df)} feature rows"

    # ── Task 3: Train models ──────────────────────────────────────────────

    @task(task_id="train_models")
    def train_models(fe_result: str) -> dict:
        """Train all models and return best metrics."""
        if fe_result == "skipped":
            return {"skipped": True}

        from src.train.trainer import ModelTrainer

        trainer = ModelTrainer()
        train_df, test_df = trainer.load_features()
        results = trainer.train_all_models(train_df, test_df)

        best = results[0]
        return {
            "skipped": False,
            "best_model": best["model_name"],
            "f1": best["metrics"]["f1"],
            "roc_auc": best["metrics"]["roc_auc"],
            "run_id": best["run_id"],
        }

    # ── Task 4: Drift check ──────────────────────────────────────────────

    @task(task_id="check_drift")
    def check_drift() -> dict:
        """Run drift detection on production data."""
        import pandas as pd
        from src.monitor.drift_detector import DriftDetector

        detector = DriftDetector()

        try:
            reference = pd.read_parquet("data/reference_baseline.parquet")
            production = pd.read_parquet("data/features.parquet")
            report = detector.check_all_features(reference, production)

            return {
                "status": report.overall_status,
                "drifted": report.drifted_features,
                "should_retrain": detector.should_retrain(report),
            }
        except FileNotFoundError:
            return {"status": "no_baseline", "drifted": 0, "should_retrain": False}

    # ── Task 5: Decide deployment ─────────────────────────────────────────

    def decide_deployment(**context) -> str:
        """Branch: deploy if model meets quality threshold."""
        ti = context["ti"]
        train_result = ti.xcom_pull(task_ids="train_models")

        if train_result.get("skipped"):
            return "skip_deployment"

        f1 = train_result.get("f1", 0)
        min_f1 = 0.75  # Minimum F1 to deploy

        if f1 >= min_f1:
            return "deploy_model"
        else:
            return "skip_deployment"

    branch_deploy = BranchPythonOperator(
        task_id="decide_deployment",
        python_callable=decide_deployment,
    )

    # ── Task 6a: Deploy model ─────────────────────────────────────────────

    @task(task_id="deploy_model")
    def deploy_model(**context) -> str:
        """Register best model and deploy to Cloud Run."""
        ti = context["ti"]
        train_result = ti.xcom_pull(task_ids="train_models")

        from src.train.trainer import ModelTrainer

        trainer = ModelTrainer()
        version = trainer.register_best_model([{
            "model_name": train_result["best_model"],
            "metrics": {"f1": train_result["f1"]},
            "run_id": train_result["run_id"],
        }])

        # Trigger Cloud Run redeploy (via gcloud or API)
        import subprocess
        subprocess.run([
            "gcloud", "run", "deploy", "fakenews-api",
            "--image", "gcr.io/PROJECT_ID/fakenews-api:latest",
            "--region", "us-central1",
            "--platform", "managed",
            "--set-env-vars", f"MODEL_VERSION={version}",
        ], check=True)

        return f"Deployed model v{version}"

    # ── Task 6b: Skip ─────────────────────────────────────────────────────

    skip_deployment = EmptyOperator(task_id="skip_deployment")

    # ── Task 7: Update baseline ───────────────────────────────────────────

    @task(task_id="update_baseline", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def update_baseline() -> str:
        """Save current features as new drift reference baseline."""
        import shutil
        from pathlib import Path

        src = Path("data/features.parquet")
        dst = Path("data/reference_baseline.parquet")

        if src.exists():
            shutil.copy2(src, dst)
            return "Baseline updated"
        return "No features file found"

    # ── DAG Wiring ─────────────────────────────────────────────────────────

    data_check = check_new_data()
    features = run_feature_engineering(data_check)
    training = train_models(features)
    drift = check_drift()

    [training, drift] >> branch_deploy
    branch_deploy >> [deploy_model(), skip_deployment]
    branch_deploy >> update_baseline()
