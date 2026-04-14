#!/bin/bash
# Deploy the Fake News API to Google Cloud Run.
# Usage: bash scripts/deploy.sh <PROJECT_ID> [REGION]

set -euo pipefail

PROJECT_ID="${1:?Usage: deploy.sh <PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
SERVICE_NAME="fakenews-api"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "=== Deploying $SERVICE_NAME to Cloud Run ==="
echo "Project: $PROJECT_ID | Region: $REGION"

# Enable required APIs
echo "Enabling GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    bigquery.googleapis.com \
    aiplatform.googleapis.com \
    --project="$PROJECT_ID"

# Build and push image
echo "Building Docker image..."
gcloud builds submit \
    --tag "$IMAGE:latest" \
    --project="$PROJECT_ID" \
    --timeout=600s

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE:latest" \
    --region "$REGION" \
    --platform managed \
    --memory 1Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 5 \
    --timeout 60 \
    --concurrency 80 \
    --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,MODEL_NAME=fakenews_classifier" \
    --allow-unauthenticated \
    --project="$PROJECT_ID"

# Get service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --project="$PROJECT_ID" \
    --format="value(status.url)")

echo ""
echo "=== Deployment complete ==="
echo "Service URL: $SERVICE_URL"
echo "Health check: curl $SERVICE_URL/health"
echo "Predict: curl -X POST $SERVICE_URL/predict -H 'Content-Type: application/json' -d '{...}'"
echo "API docs: $SERVICE_URL/docs"
