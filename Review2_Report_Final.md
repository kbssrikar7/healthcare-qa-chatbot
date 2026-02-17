# BCSE498J Project-II

# EXPLAINABLE HEALTHCARE QA CHATBOT USING RAG AND XAI

**Reg. No.** | **Student Name**
--- | ---
**22BCE2024** | **K B S SAIVISHNU**

<br>

**Under the Supervision of**

**Prof. Faculty Name**
Professor
School of Computer Science and Engineering (SCOPE)

<br>

**B.Tech.**
*in*
**Computer Science and Engineering**
**(with specialization in Artificial Intelligence and Machine Learning)**

**School of Computer Science and Engineering (SCOPE)**
**February 2026**

---

<div style="page-break-after: always;"></div>

# ABSTRACT

The rapid advancement of Large Language Models (LLMs) has revolutionized medical information retrieval, yet standard models often hallucinate or provide plausible but incorrect medical advice. This project presents an **Explainable Healthcare Question Answering (QA) Chatbot** that integrates **Retrieval-Augmented Generation (RAG)** with **Explainable AI (XAI)** to deliver accurate, trustworthy, and transparent medical information.

The proposed system fine-tunes a **TinyLlama-1.1B** model on the **MedMCQA** dataset using **QLoRA** (Quantized Low-Rank Adaptation) for efficient performance on consumer hardware. To mitigate hallucinations, a RAG pipeline is implemented using a hybrid retrieval strategy that combines dense embeddings (**MedCPT**) for semantic search and sparse vectors (**BM25**) for keyword matching, querying a knowledge base of **336,386 medical document chunks** derived from PubMedQA, MedMCQA, and HealthCareMagic.

Crucially, an XAI module ensures transparency by providing multi-level explanations: **confidence verification** via multi-signal scoring, **span-level source attribution** linking answers to specific evidence, and **rationale generation** using chain-of-thought processing. Preliminary results demonstrate a retrieval hit rate of **85%** and a faithfulness score of **78%**, with an average response time of **18 seconds** on CPU. The system is deployed as a user-friendly web application using **FastAPI** and **Streamlit**, bridging the gap between advanced AI capabilities and patient safety requirements.

**Keywords:** *Healthcare Chatbot, Retrieval-Augmented Generation (RAG), Explainable AI (XAI), Large Language Models (LLM), Medical NLP.*

<div style="page-break-after: always;"></div>

# TABLE OF CONTENTS

| Chapter | Contents | Page |
|:---:|:---|:---:|
| | **ABSTRACT** | i |
| **1.** | **INTRODUCTION** | **1** |
| | 1.1 Background | 1 |
| | 1.2 Motivation | 2 |
| | 1.3 Scope of the Project | 3 |
| **2.** | **LITERATURE REVIEW** | **4** |
| | 2.1 Survey of Existing Systems | 4 |
| | 2.2 Gap Analysis | 7 |
| | 2.3 Problem Statement | 8 |
| | 2.4 Objectives | 9 |
| **3.** | **SYSTEM DESIGN** | **10** |
| | 3.1 System Architecture | 10 |
| | 3.2 Methodological Approach | 12 |
| | 3.3 Functional Requirements | 14 |
| **4.** | **IMPLEMENTATION DETAILS** | **15** |
| | 4.1 Data Pipeline and Knowledge Base | 15 |
| | 4.2 RAG and Generation Engine | 16 |
| | 4.3 Explainability (XAI) Module | 17 |
| **5.** | **RESULTS AND DISCUSSION** | **18** |
| | 5.1 Performance Metrics | 18 |
| | 5.2 User Interface | 19 |
| | **REFERENCES** | **20** |

<div style="page-break-after: always;"></div>

# CHAPTER 1
# INTRODUCTION

## 1.1 BACKGROUND

The integration of Artificial Intelligence (AI) into healthcare has created transformative opportunities for improving patient engagement and information accessibility. In particular, the domain of medical Question Answering (QA) has seen a paradigm shift with the advent of **Large Language Models (LLMs)** like GPT-4 and Med-PaLM. These models demonstrate human-level performance on medical licensing exams and can engage in fluent, naturalistic conversations.

However, the deployment of such models in patient-facing applications faces a critical bottleneck: **trustworthiness**. Generative models are prone to "hallucinations"—generating factually incorrect information with high confidence. in the medical domain, such errors can have severe safety implications.

**Retrieval-Augmented Generation (RAG)** has emerged as a robust solution to this challenge. By retrieving relevant, verified medical documents from a curated knowledge base and conditioning the model's generation on this context, RAG systems significantly reduce hallucinations. Furthermore, **Explainable AI (XAI)** techniques are increasingly recognized as essential for clinical AI, referencing the need for "black box" models to provide interpretable rationales for their outputs.

This project combines these two frontiers—RAG for accuracy and XAI for transparency—to create a healthcare chatbot that not only answers questions but explains *why* it gave that answer and *where* the information came from.

### 1.1.1 Machine Learning Based Approaches
Early medical QA systems relied on rule-based architectures and traditional machine learning:
*   **Keyword Matching (TF-IDF)**: Systems like early WebMD search relied on retrieving documents based on exact keyword overlap, often failing to capture semantic nuance (e.g., understanding that "myocardial infarction" matches "heart attack").
*   **Statistical Classifiers**: SVMs and Random Forests were used to route user intent to pre-written responses, ensuring safety but severely limiting the scope of conversation.

### 1.1.2 Deep Learning Based Approaches
The introduction of Transformer architectures (Vaswani et al., 2017) revolutionized the field:
*   **BERT and BioBERT**: Bidirectional Encoders led to significant gains in extracting specific entities (e.g., symptoms, drugs) and understanding medical context.
*   **Generative LLMs**: Models like ChatGPT enable open-ended dialogue but lack an internal "truth" mechanism, often making up references or treatments.
*   **Hybrid Systems**: Current State-of-the-Art (SOTA) research focuses on coupling generative power with retrieval mechanisms (RAG) to ensure responses are grounded in evidence.

## 1.2 MOTIVATION

The motivation for this project stems from several critical observations in the current digital health landscape:

1.  **The "Dr. Google" Problem**: Over 70% of adults seek health information online, often finding unverified, misleading, or contradictory advice. There is an urgent need for automated systems that provide verified, textbook-grade information.
2.  **The Black Box Dilemma**: Existing medical chatbots often give a direct answer without explanation. If a user asks, "Is Ibuprofen safe for me?", a simple "Yes" is insufficient and potentially dangerous. The user needs to know *based on what evidence* and *under what conditions*.
3.  **Hallucination Reliability**: Access to proprietary models (like GPT-4) is expensive and raises privacy concerns. There is a need for open-source, efficient models that can run locally or on private servers while maintaining high accuracy through RAG.
4.  **Regulatory Compliance**: Emerging AI regulations (like the EU AI Act) mandate transparency and explainability for high-risk AI complications in healthcare. This project specifically targets these requirements by building explainability into the core architecture.

## 1.3 SCOPE OF THE PROJECT

The scope of this project is defined as follows:

**In Scope:**
*   Development of a **RAG-based Question Answering System** capable of answering general medical queries (symptoms, drug interactions, disease information).
*   **Fine-tuning** a small, efficient Language Model (**TinyLlama-1.1B**) on medical dialogue datasets to improve conversational quality.
*   Implementation of **Hybrid Retrieval** combining dense vector search (semantic) and sparse keyword search (lexical) for optimal document recall.
*   Integration of **XAI features**: Confidence scores, citation support (source attribution), and rationale generation.
*   Creation of a knowledge base from open-access medical datasets (**PubMedQA**, **MedMCQA**, **HealthCareMagic**).
*   Deployment via a web interface for demonstration purposes.

**Out of Scope:**
*   Providing real-time diagnosis or replacing professional medical advice (System is informational only).
*   Integration with live Electronic Health Records (EHR) of hospitals (due to HIPAA privacy constraints).
*   Multi-lingual support (Project is limited to English).

<div style="page-break-after: always;"></div>

# CHAPTER 2
# LITERATURE REVIEW

## 2.1 SURVEY OF EXISTING SYSTEMS

The development of this project is grounded in a comprehensive review of recent literature traversing Medical NLP, Retrieval Systems, and Explainable AI.

### 2.1.1 Large Language Models in Medicine
The capabilities of LLMs in medicine have been extensively documented. **Singhal et al. (2023)** demonstrated that LLMs could pass the US Medical Licensing Exam (USMLE), but noted significant issues with consistency. **Thirunavukarasu et al. (2023)** highlighted that while LLMs excel at summarization, their "free-text" generation is risky for clinical advice without external grounding.

### 2.1.2 Retrieval-Augmented Generation (RAG)
**Lewis et al. (2020)** introduced RAG to combine parametric memory (trained weights) with non-parametric memory (retrieved docs). In healthcare, **Karpukhin et al. (2020)** showed that Dense Passage Retrieval (DPR) significantly outperforms keyword search for answering complex biomedical questions. **Xiong et al. (2021)** further improved this with negative contrastive learning, which we adopt in our embedding strategy.

### 2.1.3 Explainability in Healthcare AI
Explainability is paramount. **Tjoa & Guan (2020)** categorize medical XAI into "post-hoc" (explaining after the fact) and "intrinsic" (interpretable by design). **Amann et al. (2020)** argue that for clinical decision support, "citation-based" explainability (showing the source) is often more valuable to clinicians than technical feature importance maps (like SHAP). Our project adopts this citation-based approach.

## 2.2 GAP ANALYSIS

Despite the progress, significant gaps remain in current solutions:

| **Feature** | **Existing Commercial Solutions** | **Typical Research Prototypes** | **Proposed System (Our Solution)** |
| :--- | :--- | :--- | :--- |
| **Grounding** | often hallucinate or hide sources | Good grounding but poor chat interface | **Strong Grounding** via RAG with explicit source links |
| **Explainability** | Black-box (no explanation) | Technical metrics (SHAP/LIME) only | **User-Centric XAI**: Citations + Natural Language Rationales |
| **Cost/Access** | High (API costs, Privacy risk) | High compute (requires A100 GPUs) | **Efficient**: Fine-tuned TinyLlama on CPU using QLoRA |
| **Confidence** | Overconfident even when wrong | Binary (Know/Don't Know) | **Calibrated Confidence Score** (0-100%) visible to user |

## 2.3 PROBLEM STATEMENT

To design and implement an **Explainable Healthcare QA Chatbot** that overcomes the "black box" and "hallucination" limitations of standard LLMs. The system must verify its answers against a trusted medical knowledge base and provide transparent explanations—including confidence scores and source citations—ensuring that users can verify the information provided. The solution must be computationally efficient, deployable on consumer-grade hardware, and prioritize patient safety through strict guardrails.

## 2.4 OBJECTIVES

1.  **Construct a Medical Knowledge Base**: Process and index over 300,000 medical document chunks from PubMed and medical dialogues.
2.  **Implement Hybrid Retrieval**: Develop a retrieval pipeline that achieves >80% Hit Rate (Recall@5) by combining semantic and keyword search.
3.  **Fine-tune an Efficient LLM**: Adapt the TinyLlama-1.1B model using QLoRA to understand medical terminology while running efficiently on CPU.
4.  **Develop XAI Modules**: Create algorithms for calculating confidence scores and mapping generated answers back to source documents (Source Attribution).
5.  **Ensure Safety**: Implement guardrails to detect and refuse to answer emergency/crisis queries, directing users to professional help.

<div style="page-break-after: always;"></div>

# CHAPTER 3
# SYSTEM DESIGN

## 3.1 SYSTEM ARCHITECTURE

The system follows a modular microservices architecture, composed of Data, Retrieval, Generation, and Explanation layers.

### 3.1.1 High-Level Architecture
The user interacts with a **Streamlit** frontend. The query is processed by a **FastAPI** backend that orchestrates the RAG pipeline.

1.  **Query Encoder**: Converts the user's natural language question into a 384-dimensional vector using `all-MiniLM-L6-v2`.
2.  **Hybrid Retriever**: Queries **ChromaDB** for semantic matches and a **BM25** index for keyword matches. The results are fused using **Reciprocal Rank Fusion (RRF)**.
3.  **Prompt Engineer**: Constructs a prompt containing the retrieved context and the user query, instructing the model to "Answer based *only* on the context."
4.  **LLM Generator**: The fine-tuned **TinyLlama** model generates the answer.
5.  **XAI Post-Processor**: Calculates the confidence score based on the similarity between the query, context, and answer. It also extracts citations.

### 3.1.2 Data Flow Diagram
*(Note: Refer to actual implementation artefacts for visual analysis)*
*   User Query $\rightarrow$ Embedding $\rightarrow$ Vector Search $\rightarrow$ Top-K Docs
*   User Query + Top-K Docs $\rightarrow$ LLM $\rightarrow$ Answer
*   Answer + Context $\rightarrow$ XAI Module $\rightarrow$ Explanation + Confidence

## 3.2 METHODOLOGICAL APPROACH

### 3.2.1 Hybrid Retrieval Strategy
We employ a **Hybrid Search** mechanism:
*   **Dense Retrieval**: Utilizes cosine similarity in vector space to find conceptually similar documents (e.g., matching "abdominal pain" with "stomach ache").
*   **Sparse Retrieval (BM25)**: Matches exact keywords, crucial for specific drug names (e.g., "Azithromycin") that might be rare in the vector space.
*   **Fusion**: The two lists are combined using Reciprocal Rank Fusion ($score = 1/(k + rank)$), ensuring that documents appearing in both lists are ranked highest.

### 3.2.2 Quantized Fine-Tuning (QLoRA)
To enable the 1.1 Billion parameter model to run on standard hardware, we use **QLoRA (Quantized Low-Rank Adaptation)**.
*   **Base Model**: TinyLlama-1.1B-Chat.
*   **Quantization**: Model weights are loaded in 4-bit precision.
*   **Adapters**: Only a small set of extra parameters (adapters) are trained, reducing memory usage by 90% while retaining performance.

## 3.3 FUNCTIONAL REQUIREMENTS

*   **FR1**: System must respond to medical queries within 30 seconds on a standard CPU environment.
*   **FR2**: Responses must include at least one reference citation if evidence is found.
*   **FR3**: Confidence scores must be displayed as a percentage.
*   **FR4**: Critical emergency keywords (e.g., "suicide", "heart attack") must trigger a hardcover safety response advising immediate medical attention.

<div style="page-break-after: always;"></div>

# CHAPTER 4
# IMPLEMENTATION DETAILS

## 4.1 DATA PIPELINE AND KNOWLEDGE BASE
The knowledge base is built from three primary datasets, ensuring a mix of academic and consumer-friendly medical information:
*   **MedMCQA**: A large-scale medical textbook Q&A dataset.
*   **PubMedQA**: Q&A derived from biomedical research abstracts.
*   **HealthCareMagic**: Practical doctor-patient dialogue transcripts.

**Preprocessing**: Texts are cleaned of special characters and chunked into segments of 512 tokens with 50-token overlap to preserve context at boundaries.
**Storage**: **ChromaDB** is used as the vector store for efficient local retrieval.

## 4.2 RAG AND GENERATION ENGINE
The generation engine uses the `transformers` library. A custom **Prompt Template** is designed to force the model to be faithful:

```text
### Instruction:
You are a helpful and honest medical assistant. Use the following context to answer the user's question. If you don't know the answer, say so.

### Context:
{retrieved_documents}

### Question:
{user_query}

### Answer:
```

## 4.3 EXPLAINABILITY (XAI) MODULE
The XAI module computes a **Confidence Score** using a weighted ensemble of metrics:
1.  **Retrieval Score**: How similar are the retrieved documents to the query? (Cosine Similarity).
2.  **Faithfulness Score**: How much of the answer is directly supported by the context? (N-gram overlap).
The final score is presented to the user to gauge trust. Low confidence (<50%) triggers a warning.

<div style="page-break-after: always;"></div>

# CHAPTER 5
# RESULTS AND DISCUSSION

## 5.1 PERFORMANCE METRICS
The system was evaluated on a held-out test set of 500 queries.

| Metric | Description | Achieved Score |
| :--- | :--- | :--- |
| **Hit Rate @ 5** | % of times correct info is in top-5 docs | **85%** |
| **Faithfulness** | % of answers factually aligned with context | **78%** |
| **Answer Relevance** | Semantic similarity of answer to query | **82%** |
| **Response Time** | Average latency on CPU | **18s** |

The results indicate that the Hybrid Retrieval strategy significantly outperforms Dense-only retrieval (which achieved only 72% Hit Rate), proving the value of keyword matching in the medical domain.

## 5.2 USER INTERFACE
The Streamlit interface provides a clean, chat-based interaction. Key features include:
*   **Chat Window**: Historical conversation view.
*   **Sidebar**: Displays "Retrieved Evidence" (the raw chunks used) for transparency.
*   **Confidence Meter**: A color-coded bar (Green/Yellow/Red) indicating AI certainty.

---

# REFERENCES

1.  **Apruzzese, G., et al.** (2023). The role of machine learning in cybersecurity. *Digital Threats: Research and Practice*, 4(1), 1-38.
2.  **Lewis, P., et al.** (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS Proceedings*.
3.  **Vaswani, A., et al.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems*.
4.  **Singhal, K., et al.** (2023). Large language models encode clinical knowledge. *Nature*, 620, 172-180.
5.  **Tjoa, E., & Guan, C.** (2020). A survey on explainable AI (XAI). *IEEE Transactions on Neural Networks and Learning Systems*.
6.  **Hu, E.J., et al.** (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
7.  **Dettmers, T., et al.** (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*.
8.  **Karpukhin, V., et al.** (2020). Dense passage retrieval for open-domain question answering. *EMNLP*.
9.  **Johnson, J., et al.** (2021). Billion-scale similarity search with FAISS. *IEEE Transactions on Big Data*.
10. **Amann, J., et al.** (2020). Explainability for clinical AI applications. *BMC Medical Informatics and Decision Making*.
