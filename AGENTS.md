# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

Explainable Healthcare QA Chatbot: a medical question-answering system combining:

- LLM generation (`TinyLlama` default, `BioMistral` optional)
- Hybrid retrieval (dense embeddings + BM25 sparse search + fusion)
- XAI confidence and attribution signals

Core stack:

- Backend: FastAPI (`api/main.py`)
- Primary UI: Streamlit (`frontend/streamlit_app.py`)
- Optional UI: Next.js app in `frontend-react/`

## Preferred Development Workflow

Prefer `make` targets where available.

### Setup

```bash
make install
```

Equivalent manual setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Services

```bash
make run            # FastAPI backend on :8000
make run-frontend   # Streamlit UI on :8501
```

Or:

```bash
python api/main.py
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Tests

```bash
make test       # fast suite
make test-all   # full suite
```

Or direct pytest:

```bash
pytest tests/ -v
pytest tests/test_retrieval.py -v
pytest tests/ -k "test_name" -v
```

### Lint / Format

```bash
make lint
make fmt
```

Current linting/formatting uses `ruff` (not black/isort/flake8).

### Evaluation

```bash
make eval
make eval-quick
python evaluation/run_evaluation.py --test-set evaluation/test_set.json
```

`evaluation/run_paper_eval.py` is the primary paper-quality pipeline used by `make eval`.

### Docker

```bash
make docker-up
make docker-down
make docker-logs
```

## Architecture

### Core Pipeline Flow

1. Retrieval: `src/retrieval/hybrid_retriever.py`
2. Grounding + orchestration: `src/pipeline/qa_pipeline.py`
3. Generation: `src/generation/llm_wrapper.py` + prompt manager
4. Explainability: `src/xai/` + confidence/safety modules

### Pipeline Variants

- Standard pipeline: `src/pipeline/qa_pipeline.py`
- LangChain variant: `src/langchain/`
- LangGraph variant: `src/langgraph/`

API pipeline selection flags on `/ask`:

- `use_langchain=true`
- `use_langgraph=true`

## Configuration

Central config: `config/settings.py` (dataclasses + `Config.from_env()`).

Important environment variables include:

- `ENVIRONMENT`
- `USE_GPU`
- `HUGGINGFACE_TOKEN`
- `KB_PERSIST_DIR`
- `CHROMA_COLLECTION`
- `DEFAULT_PIPELINE`
- `ENABLE_LANGGRAPH_CHECKPOINTING`
- `ENABLE_MCP_SEARCH`

## Data and Model Paths

- Raw datasets (download target): `data/raw/`
- Vector store (default): `data/knowledge_base/`
- Vector store (common v2 output): `data/knowledge_base_v2/`
- Embedding metadata sidecar: `embedding_metadata.json` inside chosen KB directory
- Fine-tuned/adapted models: `models/`
- Evaluation outputs: `evaluation/results/`

## Knowledge Base Build Notes

Build script: `scripts/build_knowledge_base.py`

- Default build:

```bash
python scripts/build_knowledge_base.py
```

- Recursive chunking (KB v2-style):

```bash
python scripts/build_knowledge_base.py --chunker recursive --output-dir data/knowledge_base_v2 --collection medical_knowledge_v2
```

## Testing Conventions

Shared mocks and fixtures are in `tests/conftest.py`, including:

- `MockEmbedder`, `MockRetriever`, `MockLLM`
- `sample_documents`, `sample_qa_pairs`, `edge_case_questions`, `emergency_inputs`

## Agent Guidelines

- Prefer minimal, focused changes over broad refactors.
- Do not commit generated artifacts unless explicitly requested.
- Keep retrieval/generation/XAI behavior changes covered by tests when possible.
- When modifying evaluation scripts, preserve reproducibility of output files in `evaluation/results/`.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
