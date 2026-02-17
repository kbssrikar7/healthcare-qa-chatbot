# MediQuery - Explainable Healthcare QA Chatbot

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An advanced, explainable AI system for medical question answering, powered by RAG (Retrieval-Augmented Generation) and TinyLlama. Designed for accuracy, privacy, and clinical reasoning transparency.



## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [License](#license)

## Features

### Core Capabilities
- **Grounded Answers**: Uses RAG to answer questions *only* from verified medical literature (no hallucination).
- **Explainable AI (XAI)**: Provides rationale, source attribution, and confidence scores for every answer.
- **Search Engine**: Semantic search over 360k+ medical documents (ChromaDB/Qdrant).
- **History & Context**: Remembers previous questions for follow-up.

### User Interface (Streamlit)
- **Professional Dashboard**: Clean, dark-mode UI with sidebar navigation.
- **Real-time Streaming**: Watch the answer generate token-by-token.
- **Suggestion Chips**: One-click example questions (e.g., "Side effects of Dolo 650").
- **Confidence Badges**: Color-coded indicators of AI certainty.

### API (FastAPI)
- **RESTful Endpoints**: `/ask`, `/ask/simple` for integration.
- **Swagger Documentation**: interactive API docs at `/docs`.

## Tech Stack

### AI & NLP
| Component | Technology |
|-----------|------------|
| **LLM** | TinyLlama 1.1B (Chat Finetuned) |
| **Embeddings** | all-MiniLM-L6-v2 (Sentence Transformers) |
| **Vector DB** | ChromaDB (Local) / Qdrant (Production) |
| **Framework** | LangChain, PyTorch |
| **Optimization** | PEFT (Parameter-Efficient Fine-Tuning) |

### Backend & Frontend
| Component | Technology |
|-----------|------------|
| **API** | FastAPI, Uvicorn |
| **UI** | Streamlit |
| **Language** | Python 3.12 |

## Project Structure

```bash
MediQuery/
├── api/                    # FastAPI backend
│   └── main.py             # Server entry point
├── data/                   # Data storage (ignored in git)
│   ├── knowledge_base/     # Vector database used by RAG
│   └── raw/                # Original PDFs/Texts
├── frontend/               # Streamlit UI
│   └── streamlit_app.py    # Main dashboard
├── models/                 # Model weights (ignored in git)
│   └── fine_tuned/         # PEFT adapters
├── scripts/                # Utility scripts
│   ├── ingest_drug_data.py # Data ingestion
│   └── download_data.py    # Setup script
├── src/                    # Core source code
│   ├── conversation/       # History management
│   ├── embeddings/         # Vector store logic
│   ├── generation/         # LLM wrapper & prompts
│   ├── pipeline/           # RAG orchestration
│   └── xai/                # Explainability modules
├── tests/                  # Unit tests
└── requirements.txt        # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Git
- 8GB RAM (minimum) for local LLM inference

### Local Development

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kbssrikar7/healthcare-qa-chatbot.git
   cd healthcare-qa-chatbot
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install peft  # For fine-tuned adapter support
   ```

4. **Initialize Knowledge Base (First Run Only)**
   ```bash
   # Download/Ingest data (script provided in repo)
   python scripts/ingest_drug_data.py
   # python scripts/download_model.py # if needed
   ```

5. **Run the Application**

   **Option A: Startup Script (Recommended)**
   ```bash
   # If start.sh exists
   ./start.sh
   # Or run manually
   ```

   **Option B: Manual Start**
   - **Backend (Terminal 1):**
     ```bash
     uvicorn api.main:app --reload --port 8000
     ```
   - **Frontend (Terminal 2):**
     ```bash
     streamlit run frontend/streamlit_app.py
     ```

6. **Access the App**
   - **UI**: http://localhost:8501
   - **API Docs**: http://localhost:8000/docs

## API Documentation

### `/ask/simple` (GET/POST)
Ask a quick question and get a text answer.
- **Params**: `question` (str)
- **Response**: `{ "answer": "...", "confidence": "high" }`

### `/ask` (POST)
Full RAG pipeline with detailed metadata.
- **Body**: `{ "question": "...", "include_explanation": true }`
- **Response**:
  ```json
  {
    "answer": "...",
    "sources": [ ... ],
    "rationale": "...",
    "confidence": { ... }
  }
  ```

## Architecture

1.  **Ingestion**: Medical PDFs/Texts → Chunks → Embeddings → ChromaDB.
2.  **Retrieval**: User Question → Embedding → Vector Search → Top-K Docs.
3.  **Generation**: Retrieved Docs + Question → TinyLlama (w/ Medical Adapter) → Answer.
4.  **Guardrails**: Answer cleaning to remove hallucinations/artifacts.

## License

MIT License - See LICENSE file for details.

---

**Author**: Kasilanka Bhoopesh Siva Srikar
**Project**: Capstone - Healthcare QA Chatbot
