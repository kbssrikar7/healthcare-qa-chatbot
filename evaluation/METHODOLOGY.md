# Evaluation Methodology

## Test Set

- **97 Q&A pairs** drawn from three medical domains:
  - **MedQA-USMLE**: Medical board exam questions (clinical reasoning)
  - **PubMedQA**: Research-based questions (evidence synthesis)
  - **ChatDoctor**: Patient-style clinical queries (practical medicine)
- Each entry contains: `query`, `reference_answer`, `expected_keywords`, `category`, `difficulty`
- Keywords are domain-specific medical terms (stop words removed)

## Automated Metrics

| Metric | What it Measures | Range |
|--------|-----------------|-------|
| **ROUGE-L** | Longest common subsequence overlap with reference | 0–1 |
| **BERTScore F1** | Semantic similarity via contextual embeddings | 0–1 |
| **Keyword Coverage** | Fraction of expected medical terms in answer | 0–1 |
| **ECE** | Expected Calibration Error (confidence reliability) | 0–1 (lower = better) |

## Correctness Definition

An answer is "correct" when:
- `keyword_coverage >= 0.4` **AND** `ROUGE-L >= 0.2`
- When ROUGE-L is unavailable (e.g., calibration from trajectories): `keyword_coverage >= 0.4`

## Statistical Method

- **Bootstrap resampling**: 1,000 iterations, 95% confidence intervals
- Applied to all per-question metrics (keyword coverage, ROUGE-L, BERTScore)
- Implemented in `evaluation/eval_utils.py::bootstrap_ci()`

## Hardware

- **CPU**: [Fill in: e.g., Intel i7-12700H]
- **RAM**: [Fill in: e.g., 16 GB DDR5]
- **GPU** (Colab): NVIDIA Tesla T4, 15 GB VRAM
- **OS**: Ubuntu 22.04 LTS

## Models Evaluated

| Model | Size | Quantization | Purpose |
|-------|------|-------------|---------|
| TinyLlama 1.1B | 1.1B params | FP16/BF16 | Primary lightweight model |
| BioMistral-7B | 7B params | Q4_K_M GGUF | Biomedical domain model |
| QLoRA Fine-tuned TinyLlama | 1.1B + 49MB adapter | 4-bit QLoRA | Domain-adapted model |

## Pipeline Variants

1. **Standard**: Direct orchestration (retrieval → grounding → generation → XAI)
2. **LangChain**: LCEL-based composition with LangChain wrappers
3. **LangGraph**: Self-correcting RAG with StateGraph, query refinement, document grading

## Baselines

1. **No-RAG**: Question fed directly to LLM without retrieval context
2. **Dense-only**: Vector search only (no BM25, no RRF fusion)
3. **No-XAI**: Full retrieval, no confidence/hallucination/attribution scoring

## Calibration

- **Method**: Platt scaling (logistic calibration)
- **Labels**: Keyword coverage ≥ 0.4 as correctness proxy
- **Metric**: Expected Calibration Error (ECE) with 10 bins
- **Visualization**: Reliability diagrams (before/after calibration)

## Human Evaluation (Optional)

- 25 stratified responses (5 high-confidence, 10 medium, 10 low)
- 1–5 Likert scale across 4 dimensions:
  - Factual Correctness, Relevance, Completeness, Safety
- Inter-annotator agreement: Cohen's kappa
- Template: `evaluation/human_eval_template.csv`
