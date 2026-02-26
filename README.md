# 🏥 Explainable Healthcare QA Chatbot

An intelligent medical question-answering system combining LLM + RAG + XAI.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Build knowledge base
python scripts/build_knowledge_base.py

# Run API
python api/main.py

# Run Frontend
streamlit run frontend/streamlit_app.py
```

## Architecture

- **Retrieval**: Hybrid dense (MedCPT) + sparse (BM25) search
- **Generation**: Fine-tuned BioMistral-7B with QLoRA
- **Explainability**: Confidence scoring, source attribution, rationale generation

## Project Structure

```
healthcare_qa_chatbot/
├── src/              # Source code
├── api/              # FastAPI backend
├── frontend/         # Streamlit UI
├── data/             # Datasets and knowledge base
├── models/           # Trained models
└── tests/            # Test suite
```

## Components

| Component | Technology |
|-----------|------------|
| LLM | BioMistral-7B |
| Embeddings | MedCPT / all-MiniLM |
| Vector Store | ChromaDB |
| API | FastAPI |
| Frontend | Streamlit |

## Alternative LLM Backends

The API/UI now support:
- `tinyllama` / `biomistral-7b` (Hugging Face transformers)
- `lmstudio-local` (LM Studio local server)
- `airllm-mistral-7b` (AirLLM sharded inference)

Setup guide: `docs/lmstudio_airllm_integration.md`

## Success Metrics

- Retrieval: Recall@10 > 85%
- Medical Accuracy: > 90%
- Response Time: < 10s
- Safety: < 5% hallucination rate

## License

MIT License - For educational purposes only.

## Disclaimer

⚠️ This is an educational project. The information provided by this system is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
