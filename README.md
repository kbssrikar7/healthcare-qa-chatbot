# 🏥 Explainable Healthcare QA Chatbot

![Tests](https://img.shields.io/badge/tests-250%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Model](https://img.shields.io/badge/LLM-TinyLlama%201.1B-orange)

A medical question-answering system that combines hybrid retrieval (MedCPT dense + BM25 sparse) with a 1.1B-parameter LLM and a five-signal XAI layer. Given a clinical question, the system retrieves relevant passages from a 180K-chunk knowledge base built on PubMedQA, MedMCQA, and HealthCareMagic, generates an answer, then produces a calibrated confidence score (Platt-scaled ECE=0.14), source attributions, and a hallucination risk flag. BioMistral-7B (GGUF Q4_K_M) is available as an alternative backend. A QLoRA adapter fine-tuned for 500 steps on medical dialogue is optionally loadable. Evaluated on 97 held-out questions: TinyLlama achieves keyword coverage 0.38 [0.32–0.46] and ROUGE-L 0.21; the full XAI pipeline is intentionally more conservative than a retrieval-only baseline, hedging when sources conflict.

## Quick Start

```bash
make install                  # create venv + install deps
python scripts/download_data.py
python scripts/build_knowledge_base.py
make run                      # API on :8000
make run-frontend             # Streamlit UI on :8501
```

Or via Docker:
```bash
cp .env.example .env          # fill in values
make docker-up                # API + frontend via docker-compose
```

## Development

```bash
make test          # 250 fast tests
make eval          # 97-question paper eval
make eval-table    # print LaTeX/markdown comparison table
```

## Project Structure

```
├── src/              # Core source (retrieval, generation, XAI, pipeline)
├── api/              # FastAPI backend
├── frontend/         # Streamlit UI
├── evaluation/       # Eval scripts, figures, results
├── data/             # Knowledge base + datasets
├── models/           # Fine-tuned adapter + GGUF weights
└── tests/            # 250-test suite
```

## Components

| Component | Technology |
|-----------|------------|
| Primary LLM | TinyLlama 1.1B (transformers) |
| Alt LLM | BioMistral-7B Q4_K_M (GGUF via llama-cpp) |
| Fine-tune | QLoRA adapter, 500 steps |
| Embeddings | MedCPT-Query-Encoder (768-dim) |
| Vector Store | ChromaDB (HNSW) |
| Sparse | BM25 (rank-bm25, pickle-cached) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |

## License

MIT License - For educational purposes only.

## Disclaimer

⚠️ This is an educational project. The information provided by this system is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
