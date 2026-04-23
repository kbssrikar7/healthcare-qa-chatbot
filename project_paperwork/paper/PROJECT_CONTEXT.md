# Project Context: ExplainRAG (Healthcare QA Chatbot)

This file provides codebase and architectural context for AI coding tools editing the LaTeX paper in this directory. The purpose of this document is to ensure that agents do not confabulate technical details when rewriting sections of the paper.

## 1. Project Overview
**ExplainRAG** is a locally deployable, privacy-preserving medical question-answering system. It combines a Hybrid RAG pipeline with a real-time Explainable AI (XAI) confidence layer to ensure medical answers are safe, grounded, and structurally trustworthy.

### Core Technologies
*   **Backend:** FastAPI (`api/main.py`)
*   **Frontends:** Streamlit (`frontend/`), Next.js/React (`frontend-react/`)
*   **Generative Models:** TinyLlama (Local/CPU via HuggingFace pipeline), BioMistral (GPU baseline via llama-cpp-python).
*   **Fine-Tuning:** A 500-step QLoRA adapter trained on MedMCQA and PubMedQA to improve strict medical extraction.

## 2. Technical Architecture Details
If expanding on the methodology, align with these factual implementation details:

*   **Ingestion:** The knowledge base uses a custom `RecursiveSentenceChunker` with a 20-token overlap to ensure sentences are never split arbitrarily. Datasets used: PubMedQA, MedMCQA, ChatDoctor.
*   **Retrieval (Hybrid):** Uses BM25 (sparse, rank-bm25) and all-MiniLM-L6-v2 (dense embeddings via ChromaDB). 
*   **Adaptive Fusion:** Results are fused using Reciprocal Rank Fusion (RRF). The fusion weights ($w_D, w_S$) are adaptive based on query type (e.g., "Drug" queries use 55% BM25 / 45% Dense; "Definition" queries use 80% Dense / 20% BM25).
*   **XAI Confidence Layer:** A 5-signal layer that evaluates answer safety before presenting it to the user. Signals include:
    1.  Retrieval Confidence
    2.  Generation Confidence
    3.  Self-Consistency
    4.  Source Agreement
    5.  Medical Entity Coverage
*   **Calibration:** The 5 signals are passed through Platt scaling (resubstitution fitted on n=97 questions) to output a calibrated confidence percentage. Threshold for "Correctness" is set at 0.6.

## 3. Key Findings & Narrative Rules
When editing the conclusion, abstract, or methodology, adhere to these established findings:
*   **The Tradeoff:** The XAI layer intentionally causes the system to "hedge" (abstain) when retrieved chunks conflict. This lowers raw "Keyword Coverage" scores, which we argue is a *good* thing for safety, exposing flaws in standard QA benchmark metrics.
*   **Chunking Mattters:** Switching to sentence-aware chunking provided a massive +0.292 gain in Keyword Coverage, vastly outperforming the gain from QLoRA fine-tuning (+0.079).
*   **Instruction > Pretraining:** When RAG supplies the medical context, instruction-following capabilities (like TinyLlama) can outperform domain-pretrained bulk (like BioMistral on strict extraction).

## 4. LaTeX Editing Guidelines for Agents
*   **Style:** Maintain standard IEEE transaction styling.
*   **Threats to Validity:** Section 7 has deliberately been named "Threats to Validity" (instead of standard "Limitations") to demonstrate methodological honesty. Do not rename it.
*   **Numerical Consistency:** If re-writing tables or text, keep all KW metrics exactly as they are currently written. Do not invent new ablation results or baseline metrics.
*   **Formatting:** Ensure `\resizebox` is used for tables to prevent Overfull `\hbox` margin bleeding.
