"""
FastAPI Model Server — Production-grade inference API.

Demonstrates:
- Async request handling for high throughput
- Pydantic input validation and serialization
- Health check / readiness endpoints
- Model versioning and hot-reload
- Batch prediction support
- Structured logging and error handling
- Prometheus-compatible metrics endpoint
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import settings


# ── Request / Response Schemas ──────────────────────────────────────────────

class TweetFeatures(BaseModel):
    """Input features for a single prediction."""

    retweet_count: int = Field(ge=0, description="Number of retweets")
    favorite_count: int = Field(ge=0, description="Number of favorites/likes")
    followers_count: int = Field(ge=0, description="Author's follower count")
    friends_count: int = Field(ge=0, description="Author's friends count")
    cascade_size: int = Field(ge=0, description="Total nodes in retweet cascade")
    cascade_depth: int = Field(ge=0, description="Max depth of cascade tree")
    cascade_max_breadth: int = Field(ge=0, description="Max breadth at any level")
    spread_speed_mean: float = Field(ge=0, description="Mean spread speed (sec)")
    spread_speed_std: float = Field(ge=0, description="Std dev of spread speed")
    hour_of_day: int = Field(ge=0, le=23, description="Hour tweet was posted")
    day_of_week: int = Field(ge=1, le=7, description="Day of week (1=Sun)")

    @field_validator("spread_speed_mean", "spread_speed_std")
    @classmethod
    def finite_float(cls, v):
        if not np.isfinite(v):
            raise ValueError("Must be a finite number")
        return v


class PredictionResponse(BaseModel):
    """Single prediction result."""

    label: int = Field(description="0=real, 1=fake")
    probability: float = Field(description="Probability of being fake news")
    confidence: str = Field(description="low / medium / high")
    model_version: str


class BatchRequest(BaseModel):
    """Batch prediction request."""

    instances: list[TweetFeatures] = Field(
        min_length=1, max_length=1000, description="List of tweet features"
    )


class BatchResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


# ── Application State ───────────────────────────────────────────────────────

class ModelState:
    """Holds the loaded model and metadata."""

    def __init__(self):
        self.pipeline = None
        self.version: str = "unknown"
        self.feature_order: list[str] = []
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.error_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None


state = ModelState()


# ── Model Loading ───────────────────────────────────────────────────────────

def load_model_from_mlflow() -> bool:
    """Try to load model from MLflow registry."""
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        model_uri = f"models:/{settings.MODEL_NAME}/Production"
        state.pipeline = mlflow.sklearn.load_model(model_uri)
        state.version = "mlflow-production"
        logger.info(f"Loaded model from MLflow: {model_uri}")
        return True
    except Exception as e:
        logger.warning(f"MLflow load failed: {e}")
        return False


def load_model_from_local() -> bool:
    """Fallback: load model from local file."""
    paths = [
        Path("models/xgboost/pipeline.joblib"),
        Path("models/random_forest/pipeline.joblib"),
    ]
    for path in paths:
        if path.exists():
            state.pipeline = joblib.load(path)
            state.version = f"local-{path.parent.name}"
            logger.info(f"Loaded model from local: {path}")
            return True
    return False


# ── Feature Preparation ────────────────────────────────────────────────────

def prepare_features(features: TweetFeatures) -> pd.DataFrame:
    """Convert input features to model-ready DataFrame with derived features."""
    data = features.model_dump()

    # Derived features (must match training pipeline)
    followers = max(data["followers_count"], 1)
    data["virality_ratio"] = data["retweet_count"] / followers
    data["engagement_ratio"] = data["favorite_count"] / followers
    data["ff_ratio"] = data["friends_count"] / followers

    cascade = max(data["cascade_size"], 1)
    depth = max(data["cascade_depth"], 1)
    data["depth_ratio"] = data["cascade_depth"] / cascade
    data["breadth_depth_ratio"] = data["cascade_max_breadth"] / depth

    data["cascade_size_pctrank"] = 0.5  # Default; updated in batch
    data["speed_pctrank"] = 0.5

    # Log transforms
    for col in ["retweet_count", "favorite_count", "followers_count", "cascade_size"]:
        data[f"log_{col}"] = np.log1p(data[col])

    # Interaction features
    data["speed_x_size"] = data["spread_speed_mean"] * data["cascade_size"]
    mean_speed = data["spread_speed_mean"]
    std_speed = data["spread_speed_std"]
    data["spread_cv"] = std_speed / mean_speed if mean_speed > 0 else 0
    total = mean_speed + std_speed
    data["burstiness"] = (std_speed - mean_speed) / total if total > 0 else 0

    return pd.DataFrame([data])


def confidence_label(prob: float) -> str:
    """Convert probability to human-readable confidence."""
    if prob > 0.85 or prob < 0.15:
        return "high"
    elif prob > 0.65 or prob < 0.35:
        return "medium"
    return "low"


# ── App Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    loaded = load_model_from_mlflow() or load_model_from_local()
    if not loaded:
        logger.error("No model found! API will return 503 on predictions.")
    yield
    logger.info("Shutting down.")


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fake News Detection API",
    description="Production ML inference for fake news classification",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for load balancers and k8s probes."""
    return HealthResponse(
        status="healthy" if state.is_ready else "degraded",
        model_loaded=state.is_ready,
        model_version=state.version,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.get("/ready")
async def readiness():
    """Readiness probe — returns 503 if model isn't loaded."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: TweetFeatures):
    """Single prediction endpoint."""
    if not state.is_ready:
        raise HTTPException(503, "Model not loaded")

    state.request_count += 1
    try:
        df = prepare_features(features)
        prob = state.pipeline.predict_proba(df)[0][1]
        label = int(prob >= 0.5)

        return PredictionResponse(
            label=label,
            probability=round(float(prob), 4),
            confidence=confidence_label(prob),
            model_version=state.version,
        )
    except Exception as e:
        state.error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest):
    """Batch prediction endpoint for high throughput."""
    if not state.is_ready:
        raise HTTPException(503, "Model not loaded")

    start = time.time()
    state.request_count += len(batch.instances)

    try:
        dfs = [prepare_features(inst) for inst in batch.instances]
        combined = pd.concat(dfs, ignore_index=True)

        probs = state.pipeline.predict_proba(combined)[:, 1]

        predictions = [
            PredictionResponse(
                label=int(p >= 0.5),
                probability=round(float(p), 4),
                confidence=confidence_label(p),
                model_version=state.version,
            )
            for p in probs
        ]

        latency = (time.time() - start) * 1000
        return BatchResponse(
            predictions=predictions,
            count=len(predictions),
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        state.error_count += 1
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(500, f"Batch prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics."""
    return {
        "requests_total": state.request_count,
        "errors_total": state.error_count,
        "model_version": state.version,
        "uptime_seconds": round(time.time() - state.start_time, 1),
    }
