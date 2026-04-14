"""
Feature Engineering — Extract propagation-based features for fake news detection.

Demonstrates:
- Complex SQL with window functions, CTEs, and aggregations in BigQuery
- Feature extraction pipeline (temporal + structural features)
- BigQuery materialized views for feature serving
- Pandas + NumPy for local feature transformations
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from google.cloud import bigquery
from loguru import logger
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import settings
from src.data.bigquery_loader import BigQueryLoader


# ── SQL Feature Queries ─────────────────────────────────────────────────────

PROPAGATION_FEATURES_SQL = """
WITH cascade_stats AS (
    -- Aggregate cascade-level statistics per tweet
    SELECT
        tweet_id,
        label,
        retweet_count,
        favorite_count,
        followers_count,
        friends_count,
        cascade_size,
        cascade_depth,
        cascade_max_breadth,
        spread_speed_mean,
        spread_speed_std,

        -- Engagement ratio (avoid division by zero)
        SAFE_DIVIDE(retweet_count, GREATEST(followers_count, 1)) AS virality_ratio,
        SAFE_DIVIDE(favorite_count, GREATEST(followers_count, 1)) AS engagement_ratio,
        SAFE_DIVIDE(friends_count, GREATEST(followers_count, 1)) AS ff_ratio,

        -- Cascade shape metrics
        SAFE_DIVIDE(cascade_depth, GREATEST(cascade_size, 1)) AS depth_ratio,
        SAFE_DIVIDE(cascade_max_breadth, GREATEST(cascade_depth, 1)) AS breadth_depth_ratio,

        -- Temporal features
        EXTRACT(HOUR FROM created_at) AS hour_of_day,
        EXTRACT(DAYOFWEEK FROM created_at) AS day_of_week,

        -- Row number for train/test split reproducibility
        ROW_NUMBER() OVER (ORDER BY tweet_id) AS row_num,
        COUNT(*) OVER () AS total_rows

    FROM `{raw_table}`
    WHERE ingested_at IS NOT NULL
),

label_percentiles AS (
    -- Per-class percentile ranks for anomaly detection
    SELECT
        *,
        PERCENT_RANK() OVER (
            PARTITION BY label ORDER BY cascade_size
        ) AS cascade_size_pctrank,
        PERCENT_RANK() OVER (
            PARTITION BY label ORDER BY spread_speed_mean
        ) AS speed_pctrank
    FROM cascade_stats
)

SELECT
    tweet_id,
    label,

    -- Raw propagation features
    retweet_count,
    favorite_count,
    followers_count,
    friends_count,
    cascade_size,
    cascade_depth,
    cascade_max_breadth,
    spread_speed_mean,
    spread_speed_std,

    -- Derived ratios
    virality_ratio,
    engagement_ratio,
    ff_ratio,
    depth_ratio,
    breadth_depth_ratio,

    -- Temporal
    hour_of_day,
    day_of_week,

    -- Statistical position
    cascade_size_pctrank,
    speed_pctrank,

    -- Deterministic split flag (80/20)
    CASE WHEN MOD(row_num, 5) = 0 THEN 'test' ELSE 'train' END AS split

FROM label_percentiles
ORDER BY tweet_id
"""


# ── Feature columns ─────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "retweet_count",
    "favorite_count",
    "followers_count",
    "friends_count",
    "cascade_size",
    "cascade_depth",
    "cascade_max_breadth",
    "spread_speed_mean",
    "spread_speed_std",
    "virality_ratio",
    "engagement_ratio",
    "ff_ratio",
    "depth_ratio",
    "breadth_depth_ratio",
    "hour_of_day",
    "day_of_week",
    "cascade_size_pctrank",
    "speed_pctrank",
]


class FeatureEngineer:
    """Extracts and stores ML features using BigQuery."""

    def __init__(self):
        self.loader = BigQueryLoader()
        self.client = self.loader.client

    def extract_features_from_bq(self) -> pd.DataFrame:
        """Run feature SQL on BigQuery and return DataFrame."""
        sql = PROPAGATION_FEATURES_SQL.format(raw_table=settings.bq_raw_table_id)
        logger.info("Extracting features from BigQuery...")
        df = self.loader.query(sql)
        logger.info(f"Extracted {len(df)} rows with {len(FEATURE_COLUMNS)} features.")
        return df

    def add_local_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that require NumPy/SciPy (not easily done in SQL).

        - Log transforms for skewed distributions
        - Interaction features
        - Statistical moments
        """
        df = df.copy()

        # Log transforms for heavy-tailed features
        for col in ["retweet_count", "favorite_count", "followers_count", "cascade_size"]:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0))

        # Interaction: speed * size indicates viral potential
        df["speed_x_size"] = df["spread_speed_mean"].fillna(0) * df["cascade_size"].fillna(0)

        # Coefficient of variation of spread speed
        df["spread_cv"] = np.where(
            df["spread_speed_mean"] > 0,
            df["spread_speed_std"] / df["spread_speed_mean"],
            0,
        )

        # Cascade "burstiness" approximation
        df["burstiness"] = np.where(
            (df["spread_speed_mean"] + df["spread_speed_std"]) > 0,
            (df["spread_speed_std"] - df["spread_speed_mean"])
            / (df["spread_speed_std"] + df["spread_speed_mean"]),
            0,
        )

        logger.info(f"Added local features. Total columns: {len(df.columns)}")
        return df

    def save_features_to_bq(self, df: pd.DataFrame) -> None:
        """Write computed features back to BigQuery features table."""
        table_id = settings.bq_features_table_id

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            autodetect=True,
        )

        job = self.client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()
        logger.info(f"Saved {len(df)} feature rows to {table_id}.")

    def save_features_local(
        self, df: pd.DataFrame, path: str = "data/features.parquet"
    ) -> None:
        """Save features locally as Parquet for fast reloads."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved features locally to {path}.")

    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics for monitoring baseline."""
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        stats = df[feature_cols].describe().T
        stats["skewness"] = df[feature_cols].apply(skew)
        stats["kurtosis"] = df[feature_cols].apply(kurtosis)
        return stats

    def run(self, save_to_bq: bool = True, save_local: bool = True) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        df = self.extract_features_from_bq()
        df = self.add_local_features(df)

        if save_to_bq:
            self.save_features_to_bq(df)
        if save_local:
            self.save_features_local(df)

        stats = self.get_feature_stats(df)
        logger.info(f"\nFeature statistics:\n{stats}")

        return df


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run()
