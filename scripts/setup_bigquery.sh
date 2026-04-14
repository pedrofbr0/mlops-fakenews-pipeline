#!/bin/bash
# Setup BigQuery dataset and tables for the ML pipeline.
# Usage: bash scripts/setup_bigquery.sh <PROJECT_ID>

set -euo pipefail

PROJECT_ID="${1:?Usage: setup_bigquery.sh <PROJECT_ID>}"
DATASET="fakenews_ml"
REGION="us-central1"

echo "=== Setting up BigQuery for project: $PROJECT_ID ==="

# Create dataset
bq --project_id="$PROJECT_ID" mk \
    --dataset \
    --location="$REGION" \
    --description="Fake News ML Pipeline Data" \
    "$DATASET" 2>/dev/null || echo "Dataset $DATASET already exists."

# Create raw tweets table with partitioning
bq --project_id="$PROJECT_ID" mk \
    --table \
    --time_partitioning_field="ingested_at" \
    --time_partitioning_type="DAY" \
    --clustering_fields="label" \
    --description="Raw tweet data for fake news detection" \
    "$DATASET.raw_tweets" \
    tweet_id:STRING,text:STRING,user_id:STRING,retweet_count:INTEGER,favorite_count:INTEGER,followers_count:INTEGER,friends_count:INTEGER,created_at:TIMESTAMP,cascade_size:INTEGER,cascade_depth:INTEGER,cascade_max_breadth:INTEGER,spread_speed_mean:FLOAT,spread_speed_std:FLOAT,label:INTEGER,ingested_at:TIMESTAMP \
    2>/dev/null || echo "Table raw_tweets already exists."

# Create features table
bq --project_id="$PROJECT_ID" mk \
    --table \
    --description="Engineered features for ML training" \
    "$DATASET.features" \
    2>/dev/null || echo "Table features already exists (schema will be auto-detected)."

# Enable BigQuery API
gcloud services enable bigquery.googleapis.com --project="$PROJECT_ID"

echo "=== BigQuery setup complete ==="
echo "Dataset: $PROJECT_ID.$DATASET"
echo "Tables: raw_tweets (partitioned by ingested_at, clustered by label)"
