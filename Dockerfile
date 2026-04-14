# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    fastapi uvicorn[standard] pydantic joblib \
    mlflow scikit-learn xgboost numpy pandas \
    google-cloud-bigquery google-cloud-storage \
    loguru python-dotenv scipy

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY config/ config/
COPY src/ src/

# Copy model artifacts (if building with local model)
COPY models/ models/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Cloud Run expects PORT env var
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:${PORT}/health'); exit(0 if r.status_code == 200 else 1)"

CMD ["sh", "-c", "uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT} --workers 2 --access-log"]
