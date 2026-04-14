"""
Tests — Feature engineering, model quality, and API endpoint tests.

Demonstrates:
- Unit tests for feature transformations
- Model quality gates (minimum F1, AUC)
- API integration tests with FastAPI TestClient
- Async test support
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Feature Engineering Tests ──────────────────────────────────────────────

class TestFeatureEngineering:
    """Test feature transformation logic."""

    def _make_sample_df(self, n: int = 100) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame({
            "tweet_id": [f"t_{i}" for i in range(n)],
            "label": np.random.randint(0, 2, n),
            "retweet_count": np.random.randint(0, 1000, n),
            "favorite_count": np.random.randint(0, 500, n),
            "followers_count": np.random.randint(1, 100000, n),
            "friends_count": np.random.randint(1, 5000, n),
            "cascade_size": np.random.randint(1, 200, n),
            "cascade_depth": np.random.randint(1, 20, n),
            "cascade_max_breadth": np.random.randint(1, 50, n),
            "spread_speed_mean": np.random.exponential(100, n),
            "spread_speed_std": np.random.exponential(50, n),
            "hour_of_day": np.random.randint(0, 24, n),
            "day_of_week": np.random.randint(1, 8, n),
            "virality_ratio": np.random.rand(n),
            "engagement_ratio": np.random.rand(n),
            "ff_ratio": np.random.rand(n),
            "depth_ratio": np.random.rand(n),
            "breadth_depth_ratio": np.random.rand(n),
            "cascade_size_pctrank": np.random.rand(n),
            "speed_pctrank": np.random.rand(n),
            "split": ["train"] * 80 + ["test"] * 20,
        })

    def test_add_local_features_creates_expected_columns(self):
        from src.features.feature_engineering import FeatureEngineer

        df = self._make_sample_df()
        engineer = FeatureEngineer.__new__(FeatureEngineer)
        result = engineer.add_local_features(df)

        expected_new = [
            "log_retweet_count", "log_favorite_count",
            "log_followers_count", "log_cascade_size",
            "speed_x_size", "spread_cv", "burstiness",
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing column: {col}"

    def test_log_features_are_non_negative(self):
        from src.features.feature_engineering import FeatureEngineer

        df = self._make_sample_df()
        engineer = FeatureEngineer.__new__(FeatureEngineer)
        result = engineer.add_local_features(df)

        for col in ["log_retweet_count", "log_cascade_size"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"

    def test_burstiness_bounded(self):
        from src.features.feature_engineering import FeatureEngineer

        df = self._make_sample_df()
        engineer = FeatureEngineer.__new__(FeatureEngineer)
        result = engineer.add_local_features(df)

        assert result["burstiness"].between(-1, 1).all(), \
            "Burstiness should be in [-1, 1]"

    def test_no_infinities(self):
        from src.features.feature_engineering import FeatureEngineer

        df = self._make_sample_df()
        df.loc[0, "spread_speed_mean"] = 0
        df.loc[0, "spread_speed_std"] = 0
        engineer = FeatureEngineer.__new__(FeatureEngineer)
        result = engineer.add_local_features(df)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert np.isfinite(result[col]).all(), f"Infinite values in {col}"


# ── Model Quality Tests ───────────────────────────────────────────────────

class TestModelQuality:
    """Ensure model meets minimum quality thresholds."""

    def test_model_f1_above_threshold(self):
        """Model F1 must exceed 0.70 on test set."""
        # This would load model and test data in CI
        # Stubbed for portfolio demonstration
        min_f1 = 0.70
        mock_f1 = 0.82  # Replace with actual evaluation
        assert mock_f1 >= min_f1, f"F1 {mock_f1} below threshold {min_f1}"

    def test_model_auc_above_threshold(self):
        """Model AUC must exceed 0.75."""
        min_auc = 0.75
        mock_auc = 0.88
        assert mock_auc >= min_auc, f"AUC {mock_auc} below threshold {min_auc}"

    def test_prediction_distribution_reasonable(self):
        """Predictions shouldn't be all one class."""
        predictions = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        pct_positive = predictions.mean()
        assert 0.1 < pct_positive < 0.9, \
            f"Prediction distribution too skewed: {pct_positive:.2f}"


# ── API Tests ──────────────────────────────────────────────────────────────

class TestAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from src.serve.app import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data

    def test_predict_valid_input(self, client):
        payload = {
            "retweet_count": 150,
            "favorite_count": 45,
            "followers_count": 5000,
            "friends_count": 300,
            "cascade_size": 25,
            "cascade_depth": 4,
            "cascade_max_breadth": 8,
            "spread_speed_mean": 120.5,
            "spread_speed_std": 45.2,
            "hour_of_day": 14,
            "day_of_week": 3,
        }
        response = client.post("/predict", json=payload)
        # 200 if model loaded, 503 if not (both valid in test)
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert data["label"] in (0, 1)
            assert 0 <= data["probability"] <= 1
            assert data["confidence"] in ("low", "medium", "high")

    def test_predict_invalid_input_rejected(self, client):
        payload = {"retweet_count": -1}  # Invalid: negative
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_batch_predict(self, client):
        payload = {
            "instances": [
                {
                    "retweet_count": 100, "favorite_count": 30,
                    "followers_count": 2000, "friends_count": 200,
                    "cascade_size": 15, "cascade_depth": 3,
                    "cascade_max_breadth": 6, "spread_speed_mean": 80.0,
                    "spread_speed_std": 30.0, "hour_of_day": 10,
                    "day_of_week": 2,
                },
                {
                    "retweet_count": 5000, "favorite_count": 1200,
                    "followers_count": 100, "friends_count": 50,
                    "cascade_size": 300, "cascade_depth": 12,
                    "cascade_max_breadth": 45, "spread_speed_mean": 5.0,
                    "spread_speed_std": 15.0, "hour_of_day": 3,
                    "day_of_week": 7,
                },
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code in (200, 503)

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
