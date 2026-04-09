# Healthcare QA Chatbot - Production Dockerfile
# Multi-stage build: heavy ML deps cached separately from app code

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Layer A: PyTorch CPU-only (largest dep, rarely changes) ──────────────────
# Install CPU wheel first — avoids downloading 2 GB CUDA variant
RUN pip install --no-cache-dir --user \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── Layer B: Heavy ML stack (changes occasionally) ───────────────────────────
RUN pip install --no-cache-dir --user \
    "transformers>=4.36.0,<4.49" \
    "sentence-transformers>=2.2.0" \
    "peft>=0.7.0" \
    "accelerate>=0.25.0" \
    "safetensors>=0.4.0" \
    "optimum<2.0" \
    "sentencepiece>=0.2.0" \
    "chromadb==1.5.4" \
    "faiss-cpu>=1.7.4" \
    "spacy>=3.7.0" \
    "shap>=0.44.0" \
    "bert-score>=0.3.13" \
    "llama-cpp-python>=0.3.0"

# ── Layer C: App dependencies (changes with requirements.txt) ─────────────────
COPY requirements.txt .
# Skip packages already installed above to avoid re-downloading
RUN pip install --no-cache-dir --user \
    --ignore-installed torch \
    -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -r appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code (respects .dockerignore — no data/models/outputs)
COPY --chown=appuser:appuser . .

# Create runtime directories
RUN mkdir -p data/raw data/processed data/knowledge_base models outputs \
    && chown -R appuser:appuser /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV USE_GPU=false
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=production

USER appuser

EXPOSE 8000 8501

# Health check — 120s start-period for BM25 index build
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
