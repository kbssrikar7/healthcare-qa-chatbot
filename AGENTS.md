# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Explainable Healthcare QA Chatbot - A medical question-answering system combining LLM (BioMistral-7B) + RAG (hybrid dense/sparse retrieval) + XAI (confidence scoring, source attribution).

## Development Commands

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Data Pipeline
```bash
python scripts/download_data.py       # Download medical datasets (MedQuAD, PubMedQA, MedMCQA, etc.)
python scripts/build_knowledge_base.py # Build ChromaDB vector store
```

### Running Services
```bash
python api/main.py                    # FastAPI backend (port 8000)
streamlit run frontend/streamlit_app.py # Streamlit UI (port 8501)
```

### Testing
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest tests/ -v                      # Run all tests
pytest tests/test_retrieval.py -v     # Run specific test file
pytest tests/ -k "test_name" -v       # Run single test by name
pytest tests/ -v --cov=src            # Run with coverage
```

### Linting
```bash
black --check src/ api/ tests/        # Check formatting
black src/ api/ tests/                # Apply formatting
isort --check-only src/ api/ tests/   # Check import sorting
flake8 src/ api/ --max-line-length=100 --ignore=E501,W503
```

### Docker
```bash
cd docker && docker-compose up -d     # Run full stack
```

### Evaluation
```bash
python evaluation/run_evaluation.py --test-set evaluation/test_set.json
```

## Architecture

### Core Pipeline Flow
1. **Retrieval** (`src/retrieval/hybrid_retriever.py`): Hybrid search combining dense embeddings (MedCPT/MiniLM via ChromaDB) + sparse BM25, fused via Reciprocal Rank Fusion (RRF)
2. **Grounding Gate** (`src/pipeline/qa_pipeline.py`): Answerability check - ensures retrieved context is sufficient before generation
3. **Generation** (`src/generation/llm_wrapper.py`): Fine-tuned BioMistral-7B with QLoRA adapter support
4. **XAI** (`src/xai/`): Confidence scoring, source attribution, and rationale generation

### Three Pipeline Variants
- **Standard Pipeline** (`src/pipeline/qa_pipeline.py`): Direct orchestration of components
- **LangChain Pipeline** (`src/langchain/`): LCEL-based composition with LangChain wrappers
- **LangGraph Pipeline** (`src/langgraph/`): Self-correcting RAG with StateGraph, includes query refinement and document grading

### Key Component Interfaces

**Retriever** returns `RetrievedDocument(content, source, score, metadata)`:
```python
documents, context = retriever.retrieve_with_context(query, k=5)
```

**LLM** returns `GenerationResult(response, input_tokens, generated_tokens, probabilities)`:
```python
result = llm.generate(prompt, max_new_tokens=512, return_probabilities=True)
```

**QA Pipeline** returns `QAResponse` with answer, sources, confidence, attributions:
```python
response = pipeline.answer(question, num_documents=5, include_explanation=True)
```

### API Request Flags
The `/ask` endpoint supports pipeline selection:
- `use_langchain=true`: Use LangChain LCEL pipeline
- `use_langgraph=true`: Use LangGraph self-correcting pipeline

### Configuration
All settings in `config/settings.py` with dataclass configs for embedding, LLM, retrieval, safety, and pipeline. Environment-based loading via `Config.from_env()`. Key env vars: `USE_GPU`, `ENVIRONMENT`, `HUGGINGFACE_TOKEN`.

## Testing Conventions

Tests use mock components defined in `tests/conftest.py`:
- `MockEmbedder`, `MockRetriever`, `MockLLM` for testing without GPU/models
- Fixtures: `sample_documents`, `sample_qa_pairs`, `edge_case_questions`, `emergency_inputs`

Tests follow pattern `test_*.py` with pytest-asyncio for async tests.

## Data Locations
- Raw datasets: `data/raw/` (downloaded from HuggingFace)
- Vector store: `data/knowledge_base/` (ChromaDB)
- Fine-tuned models: `models/fine_tuned/medical_adapter/`
- Evaluation results: `evaluation/results/`
