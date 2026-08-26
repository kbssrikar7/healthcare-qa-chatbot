<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="public/mediquery-logo-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="public/mediquery-logo.svg" />
    <img src="public/mediquery-logo.svg" alt="MediQuery" width="260" />
  </picture>
  <br /><br />
  <p>A retrieval-augmented generation system for healthcare question answering, with explainability and safety guardrails built in.</p>

  [![CI](https://github.com/kbssrikar7/healthcare-qa-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/kbssrikar7/healthcare-qa-chatbot/actions/workflows/ci.yml)
  [![Live App](https://img.shields.io/badge/Live%20App-mediquery--healthcare.vercel.app-blue)](https://mediquery-healthcare.vercel.app)
  [![API](https://img.shields.io/badge/API-HF%20Spaces-yellow)](https://kbsss-healthcare-qa-api.hf.space/docs)
  [![Android](https://img.shields.io/badge/Android-Download%20APK-3DDC84?logo=android&logoColor=white)](https://github.com/kbssrikar7/healthcare-qa-chatbot/releases/tag/mobile-v0.1.0)
</div>

---

## Overview

MediQuery answers natural-language healthcare questions by retrieving relevant passages from a curated medical knowledge base and generating grounded responses. Every answer comes with a confidence score, source citations, and an explanation of how the score was derived — so users can see exactly what the system found and how certain it is.

The system is designed for educational use. It includes emergency detection, content filtering, and drug-interaction checking to handle edge cases safely. It does not replace clinical advice.

---

## Architecture

```
Browser
  └── Next.js frontend (Vercel)
        └── /api/* route handlers (proxy)
              └── FastAPI backend (HF Spaces, Docker)
                    ├── Hybrid retriever
                    │     ├── Dense search  ──── Qdrant Cloud (505 k docs)
                    │     └── BM25 keyword search
                    ├── Answer generator
                    ├── XAI module (confidence, attribution, rationale)
                    └── Safety guardrails
```

**Retrieval** combines dense vector search (sentence-transformers) with BM25 keyword search. Results are fused and re-ranked before being passed to the generator.

**Explainability** runs alongside generation. Each response carries a multi-signal confidence breakdown: retrieval score, entity coverage, source agreement, and answer length signal. Passage highlights show which parts of the retrieved documents were used.

**Safety** runs on every query before retrieval and on every response before delivery. The pipeline checks for emergency keywords, dangerous content, and known drug-interaction patterns.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript, assistant-ui |
| Backend | FastAPI, Python 3.11, Uvicorn |
| Retrieval | Qdrant (vector DB), sentence-transformers, rank-bm25 |
| Orchestration | LangChain, LangGraph |
| Containerisation | Docker, GitHub Container Registry |
| Frontend hosting | Vercel |
| Backend hosting | Hugging Face Spaces (Docker, CPU) |
| CI/CD | GitHub Actions |

---

## Repository Structure

```
.
├── api/                  FastAPI application and route handlers
├── src/
│   ├── conversation/     Session and follow-up detection
│   ├── data_pipeline/    Loaders, chunkers, and text cleaners
│   ├── embeddings/       Embedding models and vector store wrapper
│   ├── feedback/         Trajectory logging and reward signal
│   ├── generation/       LLM wrapper and prompt manager
│   ├── langchain/        LangChain pipeline and callbacks
│   ├── langgraph/        Self-correcting RAG graph (LangGraph)
│   ├── pipeline/         Main QA pipeline
│   ├── retrieval/        Hybrid retriever and corrective RAG
│   ├── safety/           Emergency detection, content filtering, drug interaction
│   ├── utils/            Logging, caching, metrics helpers
│   └── xai/              Confidence scorer, passage highlighter, attribution
├── frontend-react/       Next.js application
├── android/              On-device Android app (Kotlin/Compose, LiteRT-LM)
├── evaluation/           Evaluation scripts and benchmark notebooks
├── tests/                pytest test suite
├── config/               App settings and constants
├── Dockerfile            Production image (CPU-only)
└── docker-compose.yml    Local development stack
```

---

## Running Locally

### Prerequisites

- Python 3.11
- Node.js 20
- Docker (optional, for full-stack local run)

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export QDRANT_URL=<your-qdrant-url>
export QDRANT_API_KEY=<your-qdrant-api-key>
export SKIP_BM25=true          # speeds up cold start without BM25 index
export MIN_ANSWER_CONFIDENCE=0.0

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend-react
npm install
# Create .env.local with:
# NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

The frontend runs at `http://localhost:3000`.

### Docker (full stack)

```bash
docker compose up --build
```

---

## API Reference

The backend exposes a REST API documented at `/docs` (Swagger UI) and `/redoc`.

Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/ask` | Submit a question, receive an answer with sources and confidence |
| `GET` | `/health` | Health check including pipeline status |
| `POST` | `/feedback` | Submit thumbs-up/down feedback on a response |
| `DELETE` | `/session/{id}` | Clear conversation history for a session |

**Example request:**

```bash
curl -X POST https://kbsss-healthcare-qa-api.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of Type 2 diabetes?", "session_id": "demo"}'
```

**Example response fields:**

```json
{
  "answer": "...",
  "confidence_score": 0.74,
  "confidence_level": "medium",
  "sources": [...],
  "explanation": "...",
  "highlighted_passages": [...],
  "is_follow_up": false
}
```

---

## Live Deployment

| Component | URL |
|---|---|
| Frontend | https://mediquery-healthcare.vercel.app |
| Backend API | https://kbsss-healthcare-qa-api.hf.space |
| API Docs | https://kbsss-healthcare-qa-api.hf.space/docs |
| HF Space | https://huggingface.co/spaces/kbsss/healthcare-qa-api |

The backend runs on a CPU-only Docker container on Hugging Face Spaces with 16 GB RAM. The knowledge base (505,584 documents) is served from Qdrant Cloud in `europe-west3`. The frontend proxies all `/api/*` calls to the backend, so there are no CORS issues.

---

## Mobile (Android)

A separate, fully **on-device** port of the same idea: no server, no internet permission, everything — retrieval, generation, and explainability — runs locally on the phone.

**[Download the APK](https://github.com/kbssrikar7/healthcare-qa-chatbot/releases/tag/mobile-v0.1.0)** — standalone install, no `adb` or setup required. Requires Android 8.0+ on an arm64-v8a device.

| Layer | Technology |
|---|---|
| Generation | Gemma 3 1B via [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) (GPU-accelerated, CPU fallback) |
| Retrieval | Hand-rolled hybrid BM25 + dense (`all-MiniLM-L6-v2` via ONNX Runtime Mobile) with RRF fusion, over a curated on-device KB subset |
| Explainability | Ported multi-signal confidence scorer, Platt-calibrated against a real 97-question on-device evaluation run |
| UI | Jetpack Compose, Material 3 |

The base model differs from the server deployment (TinyLlama/BioMistral) because it needs to run entirely on-device — Gemma 3 1B is one of the few small models with first-class support in Google's on-device toolchain. The model and knowledge base subset are bundled inside the APK and set themselves up on first launch. See [`android/`](android/) for source and `project_paperwork/scratch/mobile_port_notes.md` for the full build/design log.

Same disclaimer as the web app: educational use only, not a substitute for professional medical advice.

---

## CI/CD

Three GitHub Actions workflows run on every push and pull request:

| Workflow | What it does |
|---|---|
| `ci.yml` | Lint (ruff), backend tests (pytest), frontend type-check and build, secret scan (gitleaks), Trivy security scan. On push to `main`: builds and pushes a Docker image to GHCR, then deploys the backend to HF Spaces. |
| `claude.yml` | Responds to `@claude` mentions in issues and pull request comments. |
| `claude-code-review.yml` | Runs an automated code review on every opened pull request. |

---

## Tests

```bash
# Backend unit and integration tests
pytest tests/ -v --ignore=tests/e2e

# Frontend type checking
cd frontend-react && npx tsc --noEmit
```

Tests run in offline mode (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) to avoid network dependencies in CI. The extractive QA backend is used instead of any hosted model.

---

## Knowledge Base

The knowledge base was built from publicly available medical datasets (MedQuAD, MedQA, clinical guidelines) using a recursive sentence chunker. Chunks are embedded with `all-MiniLM-L6-v2` and stored in Qdrant Cloud. The collection contains 505,584 documents across a range of clinical topics.

---

## License

[MIT](LICENSE) — built as a final-year undergraduate capstone. The source code is available for academic and educational reference.
