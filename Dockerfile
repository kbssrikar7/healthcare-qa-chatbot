# Healthcare QA Chatbot — Production Docker image
# Uses Python 3.11 (avoids spacy/pydantic-v1 incompatibility on 3.14)
#
# Build:
#   docker build -t healthcare-qa .
#
# Run (CPU-only):
#   docker run -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -e HF_HUB_OFFLINE=1 \
#     -e TRANSFORMERS_OFFLINE=1 \
#     healthcare-qa
#
# Run with .env file:
#   docker run -p 8000:8000 --env-file .env \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     healthcare-qa

FROM python:3.11-slim

# System deps: gcc (for llama-cpp-python), git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy only requirements first so Docker layer-caches the install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Runtime defaults ──────────────────────────────────────────────────────────
# These can all be overridden via --env or --env-file
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    USE_GPU=false \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    DEFAULT_PIPELINE=standard

# Data + model directories are mounted at runtime (not baked into the image)
# to keep the image small and avoid baking large binaries.
VOLUME ["/app/data", "/app/models"]

EXPOSE 8000

# Health check — waits 90s for the model to load before first probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
