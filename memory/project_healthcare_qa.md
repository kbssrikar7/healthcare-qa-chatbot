---
name: Healthcare QA Chatbot Project Overview
description: Final year capstone project - Explainable Healthcare QA chatbot (LLM+RAG+XAI) details
type: project
---
Final year CS undergrad capstone: Explainable Healthcare QA Chatbot combining LLM + RAG + XAI.

**Architecture:**
- LLM: TinyLlama 1.1B with fine-tuned medical LoRA adapter (at models/fine_tuned/medical_adapter)
- Retrieval: Hybrid (dense ChromaDB + BM25) with RRF fusion + cross-encoder reranking
- Knowledge base: 367,831 docs at data/knowledge_base/
- XAI: 5-signal confidence scoring, source attribution, hallucination detection, rationale generation
- Safety: Emergency detection, drug interaction checker, content filter, pediatric alerts
- Pipelines: Standard (HealthcareQAPipeline), LangChain LCEL, LangGraph self-correcting RAG
- API: FastAPI on port 8000, Frontend: Streamlit on port 8501

**Start commands (run from /home/kbs/Documents/final_project with venv activated):**
- API: `python api/main.py` (pre-loads pipeline on startup)
- Frontend: `streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true`

**Why:** HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are set in api/main.py to avoid 40s retry delay - all models are cached locally.

**How to apply:** When running or debugging the project, use these start commands and check api.log / frontend.log for issues.
