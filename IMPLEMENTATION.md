# Explainable Healthcare QA Chatbot

## Project Overview

An intelligent medical question-answering system that combines Large Language Models (LLM) with Retrieval-Augmented Generation (RAG) and Explainable AI (XAI) to provide trustworthy, grounded, and interpretable healthcare information to patients.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Implementation Phases](#implementation-phases)
5. [Detailed Implementation Steps](#detailed-implementation-steps)
6. [Data Pipeline](#data-pipeline)
7. [Model Components](#model-components)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Deployment Strategy](#deployment-strategy)
10. [Risk Mitigation](#risk-mitigation)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Web Frontend  │  │   Chat Widget   │  │      REST API Endpoint      │  │
│  │   (Streamlit/   │  │   (Optional)    │  │      (FastAPI/Flask)        │  │
│  │    Gradio)      │  │                 │  │                             │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
└───────────┼─────────────────────┼─────────────────────────┼─────────────────┘
            │                     │                         │
            └─────────────────────┼─────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Query Processing Pipeline                        │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────────┐   │   │
│  │  │  Query   │→ │   Medical    │→ │  Intent  │→ │    Safety       │   │   │
│  │  │ Cleaning │  │  NER/Entity  │  │ Classify │  │    Filter       │   │   │
│  │  └──────────┘  └──────────────┘  └──────────┘  └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                         RAG ENGINE LAYER                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    RETRIEVAL COMPONENT                               │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │    │
│  │  │   Query     │ →  │   Vector    │ →  │    Hybrid Search        │  │    │
│  │  │  Embedding  │    │   Store     │    │  (Dense + Sparse + BM25)│  │    │
│  │  │  (BioBERT/  │    │  (ChromaDB/ │    │                         │  │    │
│  │  │   PubMedBERT│    │   FAISS/    │    └─────────────────────────┘  │    │
│  │  │   /MedCPT)  │    │   Pinecone) │                                 │    │
│  │  └─────────────┘    └─────────────┘                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    GENERATION COMPONENT                              │    │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │    │
│  │  │  Context        │ →  │   Fine-tuned    │ →  │   Response      │  │    │
│  │  │  Aggregation    │    │   LLM (Llama-3/ │    │   Post-Process  │  │    │
│  │  │  & Ranking      │    │   Mistral/      │    │                 │  │    │
│  │  │                 │    │   BioMistral)   │    │                 │  │    │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                         XAI (EXPLAINABILITY) LAYER                           │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │   Confidence   │  │   Source       │  │   Attention    │  │  SHAP/    │  │
│  │   Calibration  │  │   Attribution  │  │   Visualiza-   │  │  LIME     │  │
│  │   & Scoring    │  │   (Citations)  │  │   tion         │  │  Analysis │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  └───────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              Explanation Generation Module                            │   │
│  │  • Rationale extraction from retrieved documents                      │   │
│  │  • Confidence intervals and uncertainty quantification                │   │
│  │  • Evidence highlighting and linking                                  │   │
│  │  • Counterfactual explanations                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                         KNOWLEDGE BASE LAYER                                 │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │   MEDIQA       │  │   PubMed       │  │   Wikipedia    │  │  Drug     │  │
│  │   Dataset      │  │   Abstracts    │  │   Medical      │  │  Database │  │
│  │                │  │                │  │   Articles     │  │  (RxNorm) │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  └───────────┘  │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │   MedQA        │  │   HealthQA     │  │   Clinical Guidelines          │ │
│  │   Dataset      │  │   Corpus       │  │   (CDC, WHO, Mayo Clinic)      │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Components

| Component | Technology Options | Recommendation |
|-----------|-------------------|----------------|
| **LLM Base** | Llama-3-8B, Mistral-7B, BioMistral, Meditron | BioMistral-7B (medical-specific) |
| **Embeddings** | PubMedBERT, BioBERT, MedCPT, BGE-M3 | MedCPT (retrieval-optimized) |
| **Vector Store** | ChromaDB, FAISS, Pinecone, Weaviate | ChromaDB (local) / Pinecone (production) |
| **Framework** | LangChain, LlamaIndex, Haystack | LangChain (flexibility) |
| **Fine-tuning** | LoRA, QLoRA, Full fine-tune | QLoRA (memory efficient) |
| **XAI Tools** | SHAP, LIME, Captum, attention viz | SHAP + Custom attribution |
| **Backend** | FastAPI, Flask | FastAPI |
| **Frontend** | Streamlit, Gradio, React | Gradio (rapid prototyping) |
| **Experiment Tracking** | MLflow, Weights & Biases | Weights & Biases |

### Python Libraries

```
# Core ML/NLP
torch>=2.0.0
transformers>=4.36.0
sentence-transformers>=2.2.0
peft>=0.7.0  # For LoRA/QLoRA
bitsandbytes>=0.41.0  # Quantization
accelerate>=0.25.0

# RAG Components
langchain>=0.1.0
langchain-community>=0.0.10
chromadb>=0.4.0
faiss-cpu>=1.7.4  # or faiss-gpu

# Medical NLP
scispacy>=0.5.3
medspacy>=1.0.0

# XAI
shap>=0.44.0
lime>=0.2.0
captum>=0.6.0

# API & Frontend
fastapi>=0.108.0
uvicorn>=0.25.0
gradio>=4.10.0
streamlit>=1.29.0

# Data Processing
pandas>=2.0.0
datasets>=2.16.0
beautifulsoup4>=4.12.0

# Evaluation
evaluate>=0.4.0
rouge-score>=0.1.2
bert-score>=0.3.13

# Experiment Tracking
wandb>=0.16.0
mlflow>=2.9.0
```

---

## Project Structure

```
healthcare_qa_chatbot/
│
├── README.md
├── IMPLEMENTATION.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Configuration management
│   ├── model_config.yaml        # Model hyperparameters
│   └── prompts.yaml             # Prompt templates
│
├── data/
│   ├── raw/                     # Original datasets
│   │   ├── mediqa/
│   │   ├── pubmed/
│   │   └── medical_wikipedia/
│   ├── processed/               # Cleaned, chunked data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── embeddings/              # Pre-computed embeddings
│   └── knowledge_base/          # Vector store data
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── collectors/          # Data collection scripts
│   │   │   ├── mediqa_collector.py
│   │   │   ├── pubmed_collector.py
│   │   │   └── wikipedia_collector.py
│   │   ├── preprocessors/       # Data cleaning & processing
│   │   │   ├── text_cleaner.py
│   │   │   ├── medical_ner.py
│   │   │   └── chunker.py
│   │   └── loaders/
│   │       └── dataset_loader.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedding_models.py  # Embedding model wrappers
│   │   └── indexer.py           # Vector indexing
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py         # Base retriever
│   │   ├── hybrid_retriever.py  # Dense + Sparse retrieval
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── context_builder.py   # Context aggregation
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_wrapper.py       # LLM interface
│   │   ├── prompt_manager.py    # Prompt engineering
│   │   └── response_parser.py   # Output parsing
│   │
│   ├── fine_tuning/
│   │   ├── __init__.py
│   │   ├── dataset_preparation.py
│   │   ├── trainer.py           # QLoRA training
│   │   ├── evaluation.py        # Model evaluation
│   │   └── merge_adapter.py     # Merge LoRA weights
│   │
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── confidence_scorer.py # Confidence calibration
│   │   ├── source_attribution.py # Citation generation
│   │   ├── attention_explainer.py # Attention visualization
│   │   ├── shap_explainer.py    # SHAP analysis
│   │   └── explanation_generator.py # Combined explanations
│   │
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── content_filter.py    # Harmful content detection
│   │   ├── medical_disclaimer.py # Disclaimer injection
│   │   └── hallucination_detector.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── qa_pipeline.py       # Main orchestration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       ├── metrics.py
│       └── helpers.py
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py              # Chat endpoints
│   │   └── health.py            # Health check
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models
│   └── middleware/
│       └── rate_limiter.py
│
├── frontend/
│   ├── gradio_app.py            # Gradio interface
│   ├── streamlit_app.py         # Streamlit alternative
│   └── components/
│       ├── chat_interface.py
│       └── explanation_display.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_retrieval_experiments.ipynb
│   ├── 04_fine_tuning.ipynb
│   ├── 05_xai_analysis.ipynb
│   └── 06_evaluation.ipynb
│
├── scripts/
│   ├── download_data.py
│   ├── build_knowledge_base.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_server.py
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_xai.py
│   └── test_pipeline.py
│
├── evaluation/
│   ├── benchmarks/
│   │   ├── mediqa_eval.py
│   │   └── custom_eval.py
│   ├── human_eval/
│   │   └── annotation_guidelines.md
│   └── results/
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
└── docs/
    ├── api_documentation.md
    ├── model_card.md
    ├── user_guide.md
    └── research_paper/
        ├── main.tex
        └── figures/
```

---

## Implementation Phases

### Phase 1: Foundation Setup (Week 1-2)
- [ ] Set up project structure and development environment
- [ ] Configure version control and experiment tracking
- [ ] Download and explore datasets
- [ ] Set up base infrastructure (Docker, CI/CD)

### Phase 2: Data Pipeline (Week 2-3)
- [ ] Implement data collectors for MEDIQA, PubMed, Wikipedia
- [ ] Build text preprocessing pipeline
- [ ] Implement medical NER for entity extraction
- [ ] Create document chunking strategies
- [ ] Build data validation framework

### Phase 3: RAG System (Week 3-5)
- [ ] Set up embedding models (MedCPT/PubMedBERT)
- [ ] Build vector store and indexing pipeline
- [ ] Implement hybrid retrieval (dense + BM25)
- [ ] Add cross-encoder reranking
- [ ] Optimize retrieval performance

### Phase 4: LLM Integration & Fine-tuning (Week 5-7)
- [ ] Set up base LLM (BioMistral/Llama-3)
- [ ] Prepare fine-tuning dataset
- [ ] Implement QLoRA fine-tuning pipeline
- [ ] Train and evaluate multiple checkpoints
- [ ] Merge and deploy best model

### Phase 5: Explainability Module (Week 7-9)
- [ ] Implement confidence scoring and calibration
- [ ] Build source attribution system
- [ ] Add attention visualization
- [ ] Integrate SHAP/LIME analysis
- [ ] Create unified explanation generator

### Phase 6: Safety & Guardrails (Week 9-10)
- [ ] Implement content filtering
- [ ] Add hallucination detection
- [ ] Build medical disclaimer system
- [ ] Create emergency detection pipeline

### Phase 7: API & Frontend (Week 10-11)
- [ ] Build FastAPI backend
- [ ] Create Gradio/Streamlit frontend
- [ ] Implement real-time chat interface
- [ ] Add explanation visualization components

### Phase 8: Evaluation & Documentation (Week 11-12)
- [ ] Run comprehensive benchmarks
- [ ] Conduct user studies
- [ ] Write research paper/report
- [ ] Create documentation

---

## Detailed Implementation Steps

### Step 1: Environment Setup

```bash
# Create project directory
mkdir -p healthcare_qa_chatbot
cd healthcare_qa_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pip install pre-commit
pre-commit install

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys and configurations
```

### Step 2: Data Collection

#### 2.1 MEDIQA Dataset
```python
# src/data_pipeline/collectors/mediqa_collector.py

from datasets import load_dataset
import pandas as pd
from pathlib import Path

class MEDIQACollector:
    """Collector for MEDIQA medical QA datasets."""
    
    def __init__(self, output_dir: str = "data/raw/mediqa"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_mediqa_2019(self):
        """Download MEDIQA 2019 dataset."""
        # MEDIQA includes: NLI, RQE, QA tasks
        dataset = load_dataset("bigbio/mediqa_rqe")
        return dataset
    
    def collect_mediqa_ans(self):
        """Download MedQuAD-style datasets."""
        # Alternative: MedQuAD, HealthQA
        dataset = load_dataset("medmcqa")
        return dataset
    
    def collect_pubmedqa(self):
        """Download PubMedQA dataset."""
        dataset = load_dataset("pubmed_qa", "pqa_labeled")
        return dataset
    
    def save_dataset(self, dataset, name: str):
        """Save dataset to disk."""
        output_path = self.output_dir / f"{name}.parquet"
        dataset.to_parquet(output_path)
        print(f"Saved {name} to {output_path}")
```

#### 2.2 Knowledge Base Construction
```python
# src/data_pipeline/collectors/wikipedia_collector.py

import requests
from bs4 import BeautifulSoup
import wikipediaapi
from typing import List, Dict
import time

class MedicalWikipediaCollector:
    """Collect medical articles from Wikipedia."""
    
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia('HealthcareQA/1.0', 'en')
        self.medical_categories = [
            "Category:Diseases_and_disorders",
            "Category:Symptoms",
            "Category:Medical_treatments",
            "Category:Drugs",
            "Category:Medical_diagnosis"
        ]
    
    def get_category_pages(self, category: str, depth: int = 1) -> List[str]:
        """Recursively get pages from a category."""
        pages = []
        cat = self.wiki.page(category)
        
        for title in cat.categorymembers.keys():
            member = cat.categorymembers[title]
            if member.ns == wikipediaapi.Namespace.MAIN:
                pages.append(title)
            elif member.ns == wikipediaapi.Namespace.CATEGORY and depth > 0:
                pages.extend(self.get_category_pages(title, depth - 1))
        
        return pages
    
    def fetch_article(self, title: str) -> Dict:
        """Fetch full article content."""
        page = self.wiki.page(title)
        if page.exists():
            return {
                "title": title,
                "content": page.text,
                "summary": page.summary,
                "url": page.fullurl,
                "categories": list(page.categories.keys())
            }
        return None
    
    def collect_medical_articles(self, max_articles: int = 10000):
        """Collect medical articles from Wikipedia."""
        all_pages = set()
        
        for category in self.medical_categories:
            pages = self.get_category_pages(category, depth=2)
            all_pages.update(pages)
            print(f"Found {len(pages)} pages in {category}")
        
        articles = []
        for i, title in enumerate(list(all_pages)[:max_articles]):
            article = self.fetch_article(title)
            if article:
                articles.append(article)
            
            if i % 100 == 0:
                print(f"Collected {i}/{min(len(all_pages), max_articles)} articles")
            time.sleep(0.1)  # Rate limiting
        
        return articles
```

### Step 3: Text Preprocessing & Chunking

```python
# src/data_pipeline/preprocessors/chunker.py

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class MedicalTextChunker:
    """Chunk medical documents while preserving context."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " "      # Words
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )
    
    def clean_text(self, text: str) -> str:
        """Clean medical text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove references like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        # Normalize medical abbreviations (optional)
        return text.strip()
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """Chunk a single document with metadata."""
        text = self.clean_text(document.get("content", ""))
        chunks = self.splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "content": chunk,
                "source": document.get("title", "unknown"),
                "url": document.get("url", ""),
                "chunk_id": i,
                "total_chunks": len(chunks),
                "metadata": {
                    "original_length": len(text),
                    "chunk_length": len(chunk)
                }
            })
        
        return chunked_docs
    
    def process_corpus(self, documents: List[Dict]) -> List[Dict]:
        """Process entire corpus."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
```

### Step 4: Embedding & Vector Store

```python
# src/embeddings/embedding_models.py

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import torch

class MedicalEmbedder:
    """Medical domain embedding model wrapper."""
    
    SUPPORTED_MODELS = {
        "medcpt": "ncbi/MedCPT-Query-Encoder",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "biobert": "dmis-lab/biobert-v1.1",
        "bge-m3": "BAAI/bge-m3"
    }
    
    def __init__(
        self,
        model_name: str = "medcpt",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name
        
        self.model = SentenceTransformer(model_path, device=self.device)
        self.model_name = model_name
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (may use different prefix for some models)."""
        # Some models like MedCPT have different encoders for queries vs documents
        return self.embed(query, show_progress=False)[0]
    
    @property
    def embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

```python
# src/embeddings/indexer.py

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class VectorStoreManager:
    """Manage vector store operations."""
    
    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        persist_directory: str = "data/knowledge_base"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[Dict],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """Add documents to the vector store."""
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            ids = [str(uuid.uuid4()) for _ in batch_docs]
            contents = [doc["content"] for doc in batch_docs]
            metadatas = [
                {
                    "source": doc.get("source", ""),
                    "url": doc.get("url", ""),
                    "chunk_id": doc.get("chunk_id", 0)
                }
                for doc in batch_docs
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=contents,
                metadatas=metadatas
            )
        
        print(f"Added {len(documents)} documents to collection '{self.collection_name}'")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count()
        }
```

### Step 5: Retrieval System

```python
# src/retrieval/hybrid_retriever.py

from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from src.embeddings.embedding_models import MedicalEmbedder
from src.embeddings.indexer import VectorStoreManager

class HybridRetriever:
    """Combine dense and sparse retrieval for better results."""
    
    def __init__(
        self,
        embedder: MedicalEmbedder,
        vector_store: VectorStoreManager,
        corpus: List[Dict] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize BM25 for sparse retrieval
        if corpus:
            self._init_bm25(corpus)
    
    def _init_bm25(self, corpus: List[Dict]):
        """Initialize BM25 index."""
        self.corpus = corpus
        tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform dense vector search."""
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding.tolist(), n_results=k)
        
        # Return list of (doc_index, score)
        scores = []
        for i, (doc, distance) in enumerate(zip(
            results["documents"][0],
            results["distances"][0]
        )):
            # Convert distance to similarity score
            similarity = 1 - distance
            scores.append((i, similarity))
        
        return scores
    
    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform BM25 sparse search."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        return [(idx, scores[idx]) for idx in top_indices]
    
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}
        
        max_score = max(s for _, s in scores)
        min_score = min(s for _, s in scores)
        
        if max_score == min_score:
            return {idx: 1.0 for idx, _ in scores}
        
        return {
            idx: (score - min_score) / (max_score - min_score)
            for idx, score in scores
        }
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """Retrieve relevant documents using hybrid search."""
        
        if not use_hybrid:
            # Dense only
            dense_results = self._dense_search(query, k)
            return self._format_results(dense_results[:k])
        
        # Get results from both methods
        dense_results = self._dense_search(query, k * 2)
        sparse_results = self._sparse_search(query, k * 2)
        
        # Normalize scores
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)
        
        # Combine scores (Reciprocal Rank Fusion alternative)
        combined_scores = {}
        all_indices = set(dense_normalized.keys()) | set(sparse_normalized.keys())
        
        for idx in all_indices:
            dense_score = dense_normalized.get(idx, 0)
            sparse_score = sparse_normalized.get(idx, 0)
            combined_scores[idx] = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return self._format_results(sorted_results)
    
    def _format_results(self, results: List[Tuple[int, float]]) -> List[Dict]:
        """Format retrieval results."""
        formatted = []
        for idx, score in results:
            if idx < len(self.corpus):
                doc = self.corpus[idx].copy()
                doc["retrieval_score"] = score
                formatted.append(doc)
        return formatted
```

### Step 6: LLM Integration & Fine-tuning

```python
# src/generation/llm_wrapper.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Dict, List, Optional
from peft import PeftModel

class MedicalLLM:
    """Wrapper for medical LLM with optional LoRA adapter."""
    
    def __init__(
        self,
        model_name: str = "BioMistral/BioMistral-7B",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ):
        self.model_name = model_name
        
        # Quantization config for memory efficiency
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if provided
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapter from {adapter_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_attention: bool = False
    ) -> Dict:
        """Generate response from the model."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "output_attentions": return_attention,
            "return_dict_in_generate": True
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        result = {
            "response": generated_text.strip(),
            "prompt_tokens": inputs.input_ids.shape[1],
            "generated_tokens": len(outputs.sequences[0]) - inputs.input_ids.shape[1]
        }
        
        if return_attention and hasattr(outputs, 'attentions'):
            result["attentions"] = outputs.attentions
        
        return result
```

```python
# src/fine_tuning/trainer.py

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch
from typing import Dict, List

class MedicalQATrainer:
    """Fine-tune LLM on medical QA data using QLoRA."""
    
    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "outputs/fine_tuned_model"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
    
    def prepare_model_for_training(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None
    ):
        """Prepare model for QLoRA training."""
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def prepare_dataset(
        self,
        data: List[Dict],
        max_length: int = 2048
    ) -> Dataset:
        """Prepare dataset for training."""
        
        def format_example(example):
            # Format: context + question + answer
            prompt = self._create_training_prompt(example)
            return {"text": prompt}
        
        dataset = Dataset.from_list(data)
        dataset = dataset.map(format_example)
        
        def tokenize(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        dataset = dataset.map(tokenize, batched=True)
        return dataset
    
    def _create_training_prompt(self, example: Dict) -> str:
        """Create training prompt from example."""
        template = """### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""
        
        return template.format(
            context=example.get("context", ""),
            question=example.get("question", ""),
            answer=example.get("answer", "")
        )
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4
    ):
        """Train the model."""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=10,
            save_steps=100,
            eval_steps=100 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            fp16=True,
            report_to="wandb",
            optim="paged_adamw_8bit"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        # Save the adapter
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
```

### Step 7: Explainability Module

```python
# src/xai/confidence_scorer.py

import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.special import softmax

class ConfidenceScorer:
    """Calculate and calibrate confidence scores for model outputs."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_token_probabilities(
        self,
        prompt: str,
        generated_text: str
    ) -> List[Dict]:
        """Get probability for each generated token."""
        
        full_text = prompt + generated_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities for generated tokens
        prompt_length = len(self.tokenizer.encode(prompt))
        generated_tokens = self.tokenizer.encode(generated_text)
        
        token_probs = []
        for i, token_id in enumerate(generated_tokens):
            if prompt_length + i < logits.shape[1]:
                token_logits = logits[0, prompt_length + i - 1]
                probs = torch.softmax(token_logits, dim=-1)
                token_prob = probs[token_id].item()
                
                token_probs.append({
                    "token": self.tokenizer.decode([token_id]),
                    "probability": token_prob,
                    "log_probability": np.log(token_prob + 1e-10)
                })
        
        return token_probs
    
    def calculate_sequence_confidence(
        self,
        token_probs: List[Dict]
    ) -> Dict:
        """Calculate overall confidence metrics."""
        
        probs = [tp["probability"] for tp in token_probs]
        
        return {
            "mean_confidence": np.mean(probs),
            "min_confidence": np.min(probs),
            "geometric_mean": np.exp(np.mean(np.log(probs + 1e-10))),
            "perplexity": np.exp(-np.mean([tp["log_probability"] for tp in token_probs])),
            "low_confidence_tokens": [
                tp for tp in token_probs if tp["probability"] < 0.3
            ]
        }
    
    def calibrate_confidence(
        self,
        raw_confidence: float,
        calibration_params: Dict = None
    ) -> Tuple[float, str]:
        """Apply calibration to raw confidence scores."""
        
        # Simple temperature scaling calibration
        if calibration_params is None:
            calibration_params = {"temperature": 1.5}
        
        temp = calibration_params.get("temperature", 1.5)
        calibrated = raw_confidence ** (1 / temp)
        
        # Map to confidence level
        if calibrated > 0.8:
            level = "high"
        elif calibrated > 0.5:
            level = "medium"
        else:
            level = "low"
        
        return calibrated, level
```

```python
# src/xai/source_attribution.py

from typing import List, Dict, Tuple
import re
from difflib import SequenceMatcher

class SourceAttributor:
    """Attribute generated text to source documents."""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
    
    def find_supporting_evidence(
        self,
        generated_text: str,
        source_documents: List[Dict]
    ) -> List[Dict]:
        """Find source documents that support the generated text."""
        
        # Split generated text into sentences
        sentences = self._split_sentences(generated_text)
        
        attributions = []
        for sentence in sentences:
            best_match = self._find_best_match(sentence, source_documents)
            if best_match:
                attributions.append({
                    "claim": sentence,
                    "source": best_match["source"],
                    "evidence": best_match["evidence"],
                    "similarity": best_match["similarity"],
                    "url": best_match.get("url", "")
                })
        
        return attributions
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _find_best_match(
        self,
        sentence: str,
        documents: List[Dict]
    ) -> Dict:
        """Find the best matching document for a sentence."""
        
        best_match = None
        best_similarity = 0
        
        for doc in documents:
            content = doc.get("content", "")
            
            # Check n-gram overlap
            similarity = self._calculate_similarity(sentence, content)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                
                # Find the most relevant snippet
                evidence = self._extract_evidence(sentence, content)
                
                best_match = {
                    "source": doc.get("source", "Unknown"),
                    "evidence": evidence,
                    "similarity": similarity,
                    "url": doc.get("url", "")
                }
        
        return best_match
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Use longest common subsequence ratio
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _extract_evidence(self, query: str, document: str, window: int = 200) -> str:
        """Extract the most relevant evidence snippet."""
        
        # Find the position with highest word overlap
        query_words = set(query.lower().split())
        doc_words = document.lower().split()
        
        best_start = 0
        best_overlap = 0
        
        for i in range(len(doc_words) - 20):
            window_words = set(doc_words[i:i + 30])
            overlap = len(query_words & window_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_start = i
        
        # Extract snippet
        start_char = len(' '.join(doc_words[:best_start]))
        return document[start_char:start_char + window].strip()
    
    def format_citations(
        self,
        attributions: List[Dict],
        format_type: str = "numbered"
    ) -> str:
        """Format attributions as citations."""
        
        if format_type == "numbered":
            citations = []
            for i, attr in enumerate(attributions, 1):
                citations.append(f"[{i}] {attr['source']}")
                if attr.get("url"):
                    citations.append(f"    URL: {attr['url']}")
            return "\n".join(citations)
        
        elif format_type == "inline":
            return ", ".join([f"({attr['source']})" for attr in attributions])
        
        return ""
```

```python
# src/xai/explanation_generator.py

from typing import Dict, List
from src.xai.confidence_scorer import ConfidenceScorer
from src.xai.source_attribution import SourceAttributor

class ExplanationGenerator:
    """Generate comprehensive explanations for model outputs."""
    
    def __init__(
        self,
        confidence_scorer: ConfidenceScorer,
        source_attributor: SourceAttributor
    ):
        self.confidence_scorer = confidence_scorer
        self.source_attributor = source_attributor
    
    def generate_explanation(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[Dict],
        token_probs: List[Dict] = None
    ) -> Dict:
        """Generate a comprehensive explanation."""
        
        explanation = {
            "question": question,
            "answer": answer,
            "confidence": {},
            "sources": [],
            "rationale": "",
            "limitations": [],
            "disclaimer": ""
        }
        
        # 1. Confidence Analysis
        if token_probs:
            confidence_metrics = self.confidence_scorer.calculate_sequence_confidence(
                token_probs
            )
            calibrated, level = self.confidence_scorer.calibrate_confidence(
                confidence_metrics["mean_confidence"]
            )
            
            explanation["confidence"] = {
                "score": round(calibrated, 3),
                "level": level,
                "low_confidence_parts": [
                    t["token"] for t in confidence_metrics["low_confidence_tokens"]
                ]
            }
        
        # 2. Source Attribution
        attributions = self.source_attributor.find_supporting_evidence(
            answer, retrieved_documents
        )
        explanation["sources"] = attributions
        
        # 3. Generate Rationale
        explanation["rationale"] = self._generate_rationale(
            question, answer, attributions
        )
        
        # 4. Identify Limitations
        explanation["limitations"] = self._identify_limitations(
            answer, attributions, explanation.get("confidence", {})
        )
        
        # 5. Add Medical Disclaimer
        explanation["disclaimer"] = self._get_medical_disclaimer()
        
        return explanation
    
    def _generate_rationale(
        self,
        question: str,
        answer: str,
        attributions: List[Dict]
    ) -> str:
        """Generate a human-readable rationale."""
        
        if not attributions:
            return "This answer is based on the model's general medical knowledge."
        
        sources_mentioned = list(set([a["source"] for a in attributions]))
        
        rationale = f"This answer draws from {len(sources_mentioned)} source(s): "
        rationale += ", ".join(sources_mentioned[:3])
        
        if len(sources_mentioned) > 3:
            rationale += f", and {len(sources_mentioned) - 3} more"
        
        rationale += ". "
        
        # Add key evidence snippets
        if attributions:
            rationale += f"Key supporting evidence: \"{attributions[0]['evidence'][:100]}...\""
        
        return rationale
    
    def _identify_limitations(
        self,
        answer: str,
        attributions: List[Dict],
        confidence: Dict
    ) -> List[str]:
        """Identify limitations of the answer."""
        
        limitations = []
        
        # Check confidence
        if confidence.get("level") == "low":
            limitations.append(
                "The model has low confidence in parts of this answer."
            )
        
        # Check source coverage
        if len(attributions) < 2:
            limitations.append(
                "Limited source coverage - answer may benefit from additional verification."
            )
        
        # Check for hedging language
        hedging_words = ["may", "might", "possibly", "could", "uncertain"]
        if any(word in answer.lower() for word in hedging_words):
            limitations.append(
                "The answer contains uncertain language indicating incomplete information."
            )
        
        return limitations
    
    def _get_medical_disclaimer(self) -> str:
        """Return standard medical disclaimer."""
        return (
            "DISCLAIMER: This information is for educational purposes only and "
            "should not be considered medical advice. Always consult with a "
            "qualified healthcare professional for medical concerns. In case of "
            "emergency, contact your local emergency services immediately."
        )
    
    def format_for_display(self, explanation: Dict) -> str:
        """Format explanation for user display."""
        
        output = []
        
        # Answer
        output.append(f"**Answer:**\n{explanation['answer']}\n")
        
        # Confidence
        conf = explanation.get("confidence", {})
        if conf:
            confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                conf.get("level", ""), "⚪"
            )
            output.append(
                f"**Confidence:** {confidence_emoji} {conf.get('level', 'Unknown').title()} "
                f"({conf.get('score', 0):.1%})\n"
            )
        
        # Sources
        if explanation.get("sources"):
            output.append("**Sources:**")
            for i, src in enumerate(explanation["sources"][:5], 1):
                output.append(f"  {i}. {src['source']}")
                if src.get("url"):
                    output.append(f"     [{src['url']}]")
            output.append("")
        
        # Rationale
        if explanation.get("rationale"):
            output.append(f"**Rationale:**\n{explanation['rationale']}\n")
        
        # Limitations
        if explanation.get("limitations"):
            output.append("**Limitations:**")
            for lim in explanation["limitations"]:
                output.append(f"  ⚠️ {lim}")
            output.append("")
        
        # Disclaimer
        output.append(f"---\n_{explanation.get('disclaimer', '')}_")
        
        return "\n".join(output)
```

### Step 8: Main QA Pipeline

```python
# src/pipeline/qa_pipeline.py

from typing import Dict, Optional, List
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_wrapper import MedicalLLM
from src.xai.explanation_generator import ExplanationGenerator
from src.xai.confidence_scorer import ConfidenceScorer
from src.xai.source_attribution import SourceAttributor

class HealthcareQAPipeline:
    """Main orchestration pipeline for Healthcare QA."""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: MedicalLLM,
        explanation_generator: ExplanationGenerator = None
    ):
        self.retriever = retriever
        self.llm = llm
        
        if explanation_generator is None:
            confidence_scorer = ConfidenceScorer(llm.model, llm.tokenizer)
            source_attributor = SourceAttributor()
            self.explanation_generator = ExplanationGenerator(
                confidence_scorer, source_attributor
            )
        else:
            self.explanation_generator = explanation_generator
    
    def _build_prompt(
        self,
        question: str,
        context: str
    ) -> str:
        """Build the prompt for the LLM."""
        
        prompt_template = """You are a helpful medical assistant. Answer the patient's question based on the provided medical context. Be accurate, clear, and indicate when you're uncertain.

### Medical Context:
{context}

### Patient Question:
{question}

### Instructions:
1. Provide a clear, accurate answer based on the context
2. If the context doesn't contain enough information, say so
3. Use simple language that patients can understand
4. Include relevant warnings or when to seek professional help
5. Never diagnose or prescribe - always recommend consulting a doctor

### Answer:"""
        
        return prompt_template.format(context=context, question=question)
    
    def _format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents as context."""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "Unknown")
            content = doc.get("content", "")
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def answer(
        self,
        question: str,
        num_documents: int = 5,
        include_explanation: bool = True,
        max_tokens: int = 512
    ) -> Dict:
        """Answer a healthcare question with explanations."""
        
        # 1. Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, k=num_documents)
        
        # 2. Build context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        # 3. Generate prompt
        prompt = self._build_prompt(question, context)
        
        # 4. Generate answer
        generation_result = self.llm.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            return_attention=include_explanation
        )
        
        answer = generation_result["response"]
        
        # 5. Generate explanation if requested
        explanation = None
        if include_explanation:
            # Get token probabilities for confidence scoring
            confidence_scorer = ConfidenceScorer(self.llm.model, self.llm.tokenizer)
            token_probs = confidence_scorer.get_token_probabilities(prompt, answer)
            
            explanation = self.explanation_generator.generate_explanation(
                question=question,
                answer=answer,
                retrieved_documents=retrieved_docs,
                token_probs=token_probs
            )
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "explanation": explanation,
            "formatted_explanation": (
                self.explanation_generator.format_for_display(explanation)
                if explanation else None
            )
        }
    
    def answer_batch(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict]:
        """Answer multiple questions."""
        return [self.answer(q, **kwargs) for q in questions]
```

### Step 9: API Implementation

```python
# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

from src.pipeline.qa_pipeline import HealthcareQAPipeline

app = FastAPI(
    title="Healthcare QA Chatbot API",
    description="Explainable Medical Question Answering System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialized on startup)
qa_pipeline: Optional[HealthcareQAPipeline] = None


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    include_explanation: bool = True
    num_sources: int = Field(default=5, ge=1, le=10)


class SourceInfo(BaseModel):
    source: str
    evidence: str
    similarity: float
    url: Optional[str] = None


class ConfidenceInfo(BaseModel):
    score: float
    level: str
    low_confidence_parts: List[str] = []


class ExplanationInfo(BaseModel):
    confidence: Optional[ConfidenceInfo] = None
    sources: List[SourceInfo] = []
    rationale: str = ""
    limitations: List[str] = []
    disclaimer: str = ""


class AnswerResponse(BaseModel):
    question: str
    answer: str
    explanation: Optional[ExplanationInfo] = None
    formatted_response: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the QA pipeline on startup."""
    global qa_pipeline
    # Initialize your pipeline here
    # qa_pipeline = HealthcareQAPipeline(...)
    print("Healthcare QA Pipeline initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": qa_pipeline is not None}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer a healthcare question."""
    
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        result = qa_pipeline.answer(
            question=request.question,
            num_documents=request.num_sources,
            include_explanation=request.include_explanation
        )
        
        # Convert to response format
        explanation = None
        if result.get("explanation"):
            exp = result["explanation"]
            explanation = ExplanationInfo(
                confidence=ConfidenceInfo(**exp["confidence"]) if exp.get("confidence") else None,
                sources=[SourceInfo(**s) for s in exp.get("sources", [])],
                rationale=exp.get("rationale", ""),
                limitations=exp.get("limitations", []),
                disclaimer=exp.get("disclaimer", "")
            )
        
        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            explanation=explanation,
            formatted_response=result.get("formatted_explanation")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_questions(questions: List[str]):
    """Answer multiple questions."""
    
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    results = qa_pipeline.answer_batch(questions)
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 10: Frontend Implementation

```python
# frontend/streamlit_app.py

import streamlit as st
import requests
from typing import Dict, Optional

API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Healthcare QA Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def ask_question(question: str, include_explanation: bool, num_sources: int) -> Optional[Dict]:
    """Send question to API and return response."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "include_explanation": include_explanation,
                "num_sources": num_sources
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Please ensure the backend is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def display_confidence(confidence: Dict):
    """Display confidence information with visual indicators."""
    if not confidence:
        return
    
    level = confidence.get("level", "unknown")
    score = confidence.get("score", 0)
    
    # Confidence emoji and color
    emoji_map = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    color_map = {"high": "confidence-high", "medium": "confidence-medium", "low": "confidence-low"}
    
    emoji = emoji_map.get(level, "⚪")
    color_class = color_map.get(level, "")
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.5rem;">{emoji}</span>
        <span class="{color_class}" style="font-size: 1.2rem; font-weight: bold;">
            {level.title()} Confidence ({score:.1%})
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Show low confidence tokens if any
    low_conf_parts = confidence.get("low_confidence_parts", [])
    if low_conf_parts:
        with st.expander("⚠️ Low confidence sections"):
            st.write("The model was less certain about these parts:")
            st.code(" ".join(low_conf_parts[:10]))


def display_sources(sources: list):
    """Display source attributions."""
    if not sources:
        st.info("No specific sources were attributed to this answer.")
        return
    
    for i, src in enumerate(sources, 1):
        with st.container():
            st.markdown(f"""
            <div class="source-card">
                <strong>📄 Source {i}: {src.get('source', 'Unknown')}</strong><br>
                <em>Similarity: {src.get('similarity', 0):.2%}</em><br>
                <p style="margin-top: 0.5rem;">{src.get('evidence', 'No evidence snippet available.')[:300]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            if src.get("url"):
                st.markdown(f"[🔗 View Source]({src['url']})")


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Explainable Healthcare QA Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666;">
        Ask medical questions and receive trustworthy, source-backed answers with explanations.
    </p>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer:</strong> This is an AI assistant for educational purposes only. 
        The information provided should not be considered medical advice. Always consult with qualified 
        healthcare professionals for medical concerns. In case of emergency, contact your local emergency services.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        include_explanation = st.checkbox(
            "Include Explanation",
            value=True,
            help="Show confidence scores, source attributions, and rationale"
        )
        
        num_sources = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=5,
            help="How many source documents to retrieve"
        )
        
        st.divider()
        
        st.header("💡 Example Questions")
        example_questions = [
            "What are the symptoms of diabetes?",
            "How can I manage high blood pressure?",
            "What causes migraines?",
            "When should I see a doctor for a cough?",
            "What are common symptoms of the flu?",
            "What lifestyle changes can help with anxiety?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
                st.session_state.question = q
        
        st.divider()
        
        st.header("📊 About")
        st.info("""
        This chatbot uses:
        - **RAG** for grounded answers
        - **Fine-tuned LLM** for medical domain
        - **XAI** for explainability
        """)
    
    # Initialize session state
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Main input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_area(
            "🩺 Your Health Question",
            value=st.session_state.question,
            placeholder="Enter your health-related question here...",
            height=100,
            key="question_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        submit_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.question = ""
        st.session_state.chat_history = []
        st.rerun()
    
    # Process question
    if submit_button and question.strip():
        with st.spinner("🔄 Analyzing your question and searching medical knowledge base..."):
            result = ask_question(question, include_explanation, num_sources)
        
        if result:
            # Store in chat history
            st.session_state.chat_history.append({
                "question": question,
                "result": result
            })
            st.session_state.question = ""
    
    # Display chat history
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        result = chat["result"]
        
        st.markdown("---")
        st.markdown(f"**🙋 Question:** {chat['question']}")
        
        # Answer section
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(f"**💬 Answer:**\n\n{result.get('answer', 'No answer available.')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Explanation tabs
        if result.get("explanation"):
            explanation = result["explanation"]
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Confidence", 
                "📚 Sources", 
                "💭 Rationale", 
                "⚠️ Limitations"
            ])
            
            with tab1:
                display_confidence(explanation.get("confidence", {}))
            
            with tab2:
                display_sources(explanation.get("sources", []))
            
            with tab3:
                rationale = explanation.get("rationale", "")
                if rationale:
                    st.markdown(f"**Why this answer?**\n\n{rationale}")
                else:
                    st.info("No rationale available for this answer.")
            
            with tab4:
                limitations = explanation.get("limitations", [])
                if limitations:
                    for lim in limitations:
                        st.warning(lim)
                else:
                    st.success("No significant limitations identified.")
            
            # Disclaimer
            if explanation.get("disclaimer"):
                st.caption(f"_{explanation['disclaimer']}_")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #999; font-size: 0.8rem;">
        Healthcare QA Chatbot | Powered by RAG + LLM + XAI | For Educational Purposes Only
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
```

---

## Evaluation Metrics

### Retrieval Evaluation
| Metric | Description | Target |
|--------|-------------|--------|
| **Recall@K** | % of relevant docs in top-K | > 85% |
| **MRR** | Mean Reciprocal Rank | > 0.7 |
| **NDCG@10** | Normalized DCG at 10 | > 0.75 |

### Generation Evaluation
| Metric | Description | Target |
|--------|-------------|--------|
| **ROUGE-L** | Longest common subsequence | > 0.4 |
| **BERTScore** | Semantic similarity | > 0.85 |
| **Medical F1** | Medical entity accuracy | > 0.7 |
| **Factual Accuracy** | Human-verified correctness | > 90% |

### Explainability Evaluation
| Metric | Description | Target |
|--------|-------------|--------|
| **Attribution Accuracy** | % correct source citations | > 85% |
| **Confidence Calibration** | ECE (Expected Calibration Error) | < 0.1 |
| **User Trust Score** | Survey-based trust rating | > 4/5 |

### Safety Evaluation
| Metric | Description | Target |
|--------|-------------|--------|
| **Hallucination Rate** | % of unsupported claims | < 5% |
| **Harmful Content** | Safety filter effectiveness | 100% |
| **Disclaimer Compliance** | % responses with disclaimer | 100% |

---

## Risk Mitigation

### Medical Domain Risks

| Risk | Mitigation Strategy |
|------|---------------------|
| **Hallucination** | RAG grounding, confidence thresholds, source verification |
| **Harmful advice** | Content filters, emergency detection, mandatory disclaimers |
| **Outdated information** | Regular knowledge base updates, source dating |
| **Misdiagnosis** | Clear non-diagnostic language, professional referrals |
| **Privacy concerns** | No PII storage, anonymized logging |

### Technical Risks

| Risk | Mitigation Strategy |
|------|---------------------|
| **Model bias** | Diverse training data, fairness evaluation |
| **Retrieval failures** | Fallback responses, confidence thresholds |
| **Latency issues** | Caching, model optimization, async processing |
| **Scalability** | Horizontal scaling, load balancing |

---

## Timeline Summary

```
Week 1-2:   Foundation & Setup
Week 2-3:   Data Pipeline
Week 3-5:   RAG System
Week 5-7:   LLM Fine-tuning
Week 7-9:   XAI Module
Week 9-10:  Safety & Guardrails
Week 10-11: API & Frontend
Week 11-12: Evaluation & Documentation
```

---

## Getting Started Checklist

- [ ] Clone repository and set up environment
- [ ] Download required datasets (MEDIQA, PubMedQA, MedMCQA)
- [ ] Set up experiment tracking (Weights & Biases)
- [ ] Run data preprocessing pipeline
- [ ] Build initial vector store
- [ ] Test base LLM inference
- [ ] Implement basic RAG pipeline
- [ ] Add fine-tuning scripts
- [ ] Integrate XAI components
- [ ] Deploy API and frontend
- [ ] Run evaluation benchmarks
- [ ] Write documentation and paper

---

## References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
2. Ben Abacha, A., & Demner-Fushman, D. (2019). MEDIQA 2019: A shared task on textual inference and question entailment
3. Labrak, Y., et al. (2024). BioMistral: A Collection of Open-Source Pretrained LLMs for Medical Domains
4. Jin, Q., et al. (2021). What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
5. Ribeiro, M.T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)
6. Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP)

---

## Contact & Support

For questions about this implementation, please open an issue in the repository or contact the project maintainers.

**License:** MIT (for code) / Check individual dataset licenses for data usage
