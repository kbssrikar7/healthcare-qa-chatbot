# Deployment Progress — Healthcare QA Chatbot (MediQuery)

Last updated: 2026-04-23 (night) — **project complete**

---

## STATUS: FULLY DONE

All systems live. CI passing. Repository clean. README published.

---

### 1. Ollama Backend Integration

- Added `OllamaLLM` class to `src/generation/llm_wrapper.py`
  - Reads `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY` from env
  - Sends `Authorization: Bearer` header when `OLLAMA_API_KEY` is set
  - Falls back to `ExtractiveQA` automatically if Ollama is unreachable
  - Returns `generation_backend_used` field (`"ollama"` or `"extractive_fallback"`)
- Added `generation_backend_used: str` to `GenerationResult` dataclass
- Added `"ollama"` to `AVAILABLE_MODELS` in `config/settings.py`
- Added `OLLAMA = "ollama"` to `ModelChoice` enum
- Wired Ollama routing in `api/main.py` → `get_pipeline()`
- Frontend defaults to `"ollama"` model choice

### 2. Ollama XAI + Quality Gate Fixes

- Bypassed Platt scaling for Ollama (calibration is TinyLlama-specific)
- Enabled entity coverage XAI signal for Ollama responses
- Skipped hallucination detection panel for Ollama (NLI gate targets TinyLlama only)
- Bypassed post-generation quality gates for Ollama backend
- Updated system prompt to handle MedMCQA exam-format context
- Skip irrelevant/exam-dump context; use model knowledge for brand-name drug queries

### 3. Qdrant Credentials Secured

- Removed hardcoded Qdrant URL + JWT from `src/embeddings/vector_store.py`
- Reads `QDRANT_URL` and `QDRANT_API_KEY` from environment only
- Falls back to local ChromaDB if env vars are not set
- Credentials stored only in `.env` (gitignored) and HF Space Secrets

### 4. HF Spaces Backend — LIVE

- **URL**: `https://kbsss-healthcare-qa-api.hf.space`
- **Space**: `kbsss/healthcare-qa-api` (Docker, cpu-basic, 16GB RAM)
- **Knowledge base**: 505,584 docs from Qdrant Cloud (europe-west3)
- **Default model**: Extractive QA (no GPU needed, always available)
- **Secrets set**: `QDRANT_URL`, `QDRANT_API_KEY`, `DEFAULT_MODEL=ollama`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `SKIP_BM25=true`, `MIN_ANSWER_CONFIDENCE=0.0`
- Deployment via `git archive HEAD` export — backend-only files
- Lean `requirements-hf.txt` — no peft/accelerate/shap/datasets, CPU-only torch

### 5. Vercel Frontend — LIVE

- **URL**: `https://mediquery-healthcare.vercel.app`
- **Project**: `kbssrikar7s-projects/frontend-react`
- **Framework**: Next.js 16 (App Router, Turbopack)
- `API_URL=https://kbsss-healthcare-qa-api.hf.space` set in Vercel env
- Frontend API routes proxy to HF Spaces — no CORS issues

### 6. RunPod GPU — CONFIGURED (start on demand)

- **Pod ID**: `whu76ggf45m0ih` — RTX 3090, `ollama/ollama:latest`
- **Endpoint**: `https://whu76ggf45m0ih-11434.proxy.runpod.net`
- **Cost**: ~$0.22/hr — stop when not in use
- Volume mounted at `/root/.ollama` — model persists between restarts
- Tested live 2026-04-23: full pipeline working, latency ~2100ms, confidence 72%

### 7. GitHub Actions CI/CD — ALL GREEN

- **File**: `.github/workflows/ci.yml`
- **Jobs**: secret-scan, lint-python (ruff), test-backend (pytest), test-frontend (tsc + build), docker-backend (GHCR push), deploy-hf-spaces, security (Trivy)
- **All checks passing** as of commit `a8058e10`
- Bugs fixed across three rounds:
  - Removed invalid `--ignore W503` ruff flag
  - Added `frontend-react/` to git (was a nested repo, invisible to CI)
  - Removed `.github/` from `.gitignore` (was silently dropping workflow edits)
  - Fixed `verify_embedding_compatibility()` to check `dimension` param + in-memory cache fallback
  - Fixed `_entity_coverage_score()` no-entities early return (generic regex was matching common English phrases, bypassing the neutral 0.5 return)
  - Removed 13 unused imports (F401), fixed semicolons (E702), lambda (E731), trailing whitespace (W291/W293) across 16 files
  - Added `per-file-ignores` in `pyproject.toml` for intentional E402 ordering in `api/main.py`

### 8. Repository — CLEAN

- Stale branch `fix/healthcare-qa-improvements-862131319265161586` deleted (was 35 commits behind main, superseded)
- Only branch: `main`
- `frontend-react/` properly tracked including `package-lock.json`
- `public/mediquery-logo.svg` and `public/mediquery-logo-dark.svg` committed

### 9. README — PUBLISHED

- Full rewrite: architecture diagram, tech stack, local setup, API reference, deployment URLs, CI/CD overview
- MediQuery logo with `<picture>` dark/light theme switching (dark text on light, white text on dark)
- MIT License file added (`LICENSE`)
- OG image resized from 1024×1024 to 1200×630 (standard Open Graph)

### 10. Knowledge Base v2

- 505,584 docs built with RecursiveSentenceChunker
- Hosted on Qdrant Cloud (europe-west3)
- KB v2 vs v1 A/B eval: kw_v2=0.6765 vs kw_v1=0.3845, delta=+0.2920

---

## HOW TO START (when needed)

### Option A — RunPod (remote, production)
```bash
runpodctl pod start whu76ggf45m0ih
# Wait ~30s, verify:
curl https://whu76ggf45m0ih-11434.proxy.runpod.net/api/tags
# Open: https://mediquery-healthcare.vercel.app
# Stop when done:
runpodctl pod stop whu76ggf45m0ih
```

### Option B — Local Mac (M4 Air, no RunPod cost)
```bash
# One-time setup:
ollama pull qwen3:4b      # 2.5 GB, ~60-80 tok/s on M4

# Every session:
ollama serve
# In .env: OLLAMA_BASE_URL=http://localhost:11434, OLLAMA_MODEL=qwen3:4b
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Frontend: cd frontend-react && npm run dev
```

> `qwen3:4b` is the recommended local model — 2.5 GB on disk, fits easily in 16 GB alongside the RAG stack, 128K context window. Add `/no_think` to the system prompt in `api/main.py` for the Ollama backend to skip chain-of-thought and get direct grounded answers.

---

## Architecture

```
User
 └── Frontend (Vercel)
      └── https://mediquery-healthcare.vercel.app
           │
           └── Backend (HF Spaces Docker, 16GB RAM)
                └── https://kbsss-healthcare-qa-api.hf.space
                     │
                     ├── Qdrant Cloud (505,584 docs) — always on
                     │    └── Dense retrieval (all-MiniLM-L6-v2)
                     │
                     ├── ExtractiveQA — fallback, no external dependency
                     │
                     └── Ollama — RunPod RTX 3090 (remote)
                                  or M4 Air localhost (local dev)
```

---

## Key URLs

| Service | URL |
|---|---|
| Frontend | https://mediquery-healthcare.vercel.app |
| Backend API | https://kbsss-healthcare-qa-api.hf.space |
| API Docs | https://kbsss-healthcare-qa-api.hf.space/docs |
| HF Space | https://huggingface.co/spaces/kbsss/healthcare-qa-api |
| GitHub Repo | https://github.com/kbssrikar7/healthcare-qa-chatbot |
| Vercel Project | https://vercel.com/kbssrikar7s-projects/frontend-react |
| Qdrant Cloud | https://cloud.qdrant.io |
| RunPod Console | https://www.runpod.io/console/pods |
