# Healthcare QA Chatbot - Improvement Plan

**Created:** 2026-04-06
**Status:** Ready for execution
**Baseline Metrics (TinyLlama, n=97):** Keyword Coverage 0.484 | ROUGE-L 0.234 | BERTScore F1 0.186 | Warm Latency 106s

---

## Current State Summary

### What's Working
- Full RAG pipeline (Standard + LangChain + LangGraph variants)
- Knowledge base: 182K+ chunks in ChromaDB (2.9GB), built from 401K raw QA pairs
- QLoRA fine-tuned TinyLlama adapter (r=16, alpha=32, 100 steps)
- BioMistral-7B GGUF as secondary model
- 5-signal confidence scoring with Platt calibration (ECE 0.49 raw -> 0.07 calibrated)
- Evaluation pipeline: 97-case test suite, ablation study, latency profiling
- Paper-quality figures (6 plots)
- Safety guardrails (emergency detection, drug interaction, content filter)
- Frontend with Streamlit (chat, feedback, source display)
- Colab MCP connected for GPU-accelerated tasks

### What's Broken / Semi-Finished
- Unstaged `api/main.py` fix for `/ask` endpoint parameter naming (`http_request` -> `request`)
- Unstaged `.env.example` cleanup (not committed)
- Audio/Whisper code loaded in frontend but no UI to trigger it
- Only 1 user feedback record (RL pipeline starved)
- BM25 lazy-loads 182K docs on first hybrid query (20-60s hang)
- Raw ECE = 0.49 (severe miscalibration before Platt scaling)

### Evaluation Data Available
| Variant | Keyword Cov | ROUGE-L | BERTScore | Latency |
|---------|------------|---------|-----------|---------|
| Full TinyLlama | 0.484 | 0.234 | 0.186 | 27.6s |
| QLoRA TinyLlama | 0.463 | 0.237 | 0.179 | 35.3s |
| Dense-only | 0.419 | 0.234 | 0.184 | 35.6s |
| No RAG | 0.236 | 0.151 | 0.114 | 12.3s |
| No XAI | 0.434 | 0.226 | 0.176 | 34.0s |
| BioMistral | 0.211 | 0.172 | 0.107 | 143.1s |

---

## Phase 1: Bug Fixes & Commit Hygiene (Day 1)

### 1.1 Commit Unstaged Fixes
- [ ] `api/main.py` — `/ask` endpoint parameter rename fix (`http_request` -> `request`, `request` -> `body`)
- [ ] `.env.example` — Cleaned up env template with offline mode flags
- [ ] `.mcp.json` — New Colab MCP config (untracked)

### 1.2 Fix Silent `except: pass` Blocks
**Files:**
- `frontend/streamlit_app.py` lines 930, 1353, 1409, 1510, 1516 — Add `logger.warning()` at minimum
- `scripts/build_knowledge_base.py` line 123 — Bare `except:` should catch specific exceptions
- `src/data_pipeline/loaders/dataset_loader.py` line 317 — Bare `except:` should be `except Exception`

### 1.3 Frontend API Timeout
- [ ] Add `timeout=30` to all `requests.post()` / `requests.get()` calls in `frontend/streamlit_app.py`
- [ ] Add connection error handling with user-friendly message

---

## Phase 2: Performance Critical Fixes (Day 1-2)

### 2.1 Pre-Initialize BM25 at Startup
**File:** `src/retrieval/hybrid_retriever.py`
**Problem:** `_lazy_init_bm25_from_store()` loads 182K docs from ChromaDB on first hybrid query, blocking 20-60s.
**Fix:**
- Move BM25 init to `__init__()` or add explicit `warm_up()` method
- Call it during API startup in `api/main.py` (at the `@app.on_event("startup")` handler)
- Add progress logging so startup shows "Initializing BM25 index..."
**Impact:** Eliminates first-query latency spike

### 2.2 Pre-Load NLI Model at Startup
**File:** `src/xai/hallucination_detector.py`
**Problem:** DeBERTa NLI model (~500MB) loads on first query with `use_nli=True`
**Fix:**
- Add `warm_up()` method to `HallucinationDetector`
- Call during API startup alongside other model loading
- Log model loading progress
**Impact:** Eliminates 30s+ first-query NLI latency spike

### 2.3 Enable Response Cache by Default
**File:** `config/settings.py`, `src/utils/cache_manager.py`
**Current:** 143 cached responses exist in `data/cache/` but caching is opt-in
**Fix:**
- Enable cache by default in production config
- Add cache hit/miss stats to `/health` endpoint
- Add cache TTL configuration (suggest 24h for medical QA)
**Impact:** Instant responses for repeated common medical questions

---

## Phase 3: RAG Quality Improvements (Day 2-4)

### 3.1 Fix Confidence Score Normalization
**File:** `src/xai/multi_signal_confidence.py`
**Problem:** 5 signals on different scales combined without normalization:
- RRF retrieval scores: ~0.01-0.04
- NLI consistency: 0-1
- Source agreement: 0-1
- Entity coverage: 0-1
- Generation confidence: variable

**Evidence from ablation:** Removing entity_coverage *improves* ECE (0.2417 -> 0.2079), suggesting it's adding noise. Removing retrieval signal hurts most (ECE 0.2417 -> 0.2593).

**Fix:**
- Normalize each signal to [0, 1] independently before combining
- Re-weight signals based on ablation data:
  - Retrieval: high weight (strongest ECE contributor)
  - Source agreement: high weight (2nd strongest)
  - Generation: medium weight
  - Consistency: low weight (removing it helps slightly)
  - Entity coverage: low weight or remove (removing it helps ECE)
- Apply Platt scaling parameters from `calibration.json` (a=14.44, b=-11.25)
**Impact:** Better calibrated confidence, fewer misleading high-confidence wrong answers

### 3.2 Fix RRF Score Range for Downstream Consumers
**File:** `src/retrieval/hybrid_retriever.py`, `src/pipeline/qa_pipeline.py`
**Problem:** RRF produces scores in 0.01-0.04 range, then normalized to 0-1 by dividing by max. This destroys relative score information.
**Fix:**
- Track score type metadata (rrf vs cosine) on each document
- Use score-type-aware thresholds in grounding gate
- Keep raw RRF scores for confidence scoring; only normalize for display
**Impact:** More reliable retrieval quality assessment

### 3.3 Improve Source Attribution
**File:** `src/xai/source_attribution.py`
**Problem:** Uses substring + fuzzy matching. Misses medical synonyms and paraphrases.
**Fix:**
- Add embedding-based claim-to-source matching (reuse the already-loaded embedding model)
- Compute cosine similarity between each claim sentence and each source chunk
- Set attribution threshold at 0.7 cosine similarity
- Fall back to current fuzzy matching for edge cases
**Impact:** More accurate "supported by source X" attributions in XAI output

### 3.4 Add Negation-Aware Emergency Detection
**File:** `src/safety/guardrails.py`
**Problem:** "I don't have chest pain" triggers emergency alert due to naive keyword matching.
**Fix:**
- Add negation window check: scan 3 tokens before each keyword for negation words ("not", "don't", "no", "never", "without", "deny", "denies")
- Add context requirement: require first-person indicator ("I", "my", "I'm") within sentence
- Add exception list: educational queries ("what is a heart attack", "symptoms of stroke")
**Impact:** Fewer false positive emergency alerts

---

## Phase 4: Architecture Cleanup (Day 4-5)

### 4.1 Pipeline Consolidation
**Files:** `api/main.py`, `src/langchain/`, `src/langgraph/`
**Problem:** Three pipelines loaded simultaneously, all consuming memory
**Fix:**
- Make pipeline selection a startup config (not per-request)
- Default to Standard pipeline (best keyword coverage: 0.484 vs others)
- Load only the selected pipeline at startup
- Keep LangChain/LangGraph code but don't import unless configured
**Impact:** ~30% memory reduction, simpler debugging

### 4.2 Component Health Check Endpoint
**File:** `api/main.py`
**Problem:** Optional components (reranker, query enhancer, corrective RAG, factual consistency) fail silently
**Fix:** Add `/health/components` endpoint returning:
```json
{
  "pipeline": "standard",
  "model": "tinyllama-1.1b",
  "components": {
    "vector_store": {"status": "ok", "doc_count": 182000},
    "bm25_index": {"status": "ok", "doc_count": 182000},
    "reranker": {"status": "disabled"},
    "query_enhancer": {"status": "ok"},
    "corrective_rag": {"status": "ok"},
    "hallucination_detector": {"status": "ok", "nli_loaded": true},
    "cache": {"status": "ok", "entries": 143, "hit_rate": 0.0}
  }
}
```
**Impact:** Instant visibility into which components are active

### 4.3 Clean Up Dead Frontend Code
**File:** `frontend/streamlit_app.py`
- Remove Whisper/audio transcription imports and setup (no UI button exists)
- OR: Add a microphone button to the chat input area and wire it to Whisper
- **Recommendation:** Remove it. Audio adds complexity for a text-focused QA system. Can add later if needed.
**Impact:** Faster frontend load, less confusing code

---

## Phase 5: Evaluation & Data Quality (Day 5-7)

### 5.1 Expand Ablation Study (Colab MCP Task)
**Current:** Only 15 questions in ablation study — too small for reliable conclusions
**Fix:** Re-run ablation on full 97-question test set using Colab MCP:
- Use `evaluation/colab_runner.ipynb` on Colab GPU
- Run all 6 ablation variants on test_set_v2.json
- Update `evaluation/results/ablation.json` with larger sample
- Regenerate `fig4_ablation.png`
**Impact:** Statistically significant ablation results for paper

### 5.2 Re-Calibrate Confidence Scorer
**Current:** Platt parameters (a=14.44, b=-11.25) fitted on only 39 samples
**Fix:** After fixing signal normalization (Phase 3.1):
- Re-run calibration on full test set (97+ cases) via Colab
- Update Platt scaling parameters
- Regenerate reliability diagrams
- Store new parameters in config
**Impact:** Better calibrated confidence scores in production

### 5.3 Add End-to-End Integration Tests
**File:** `tests/test_integration.py` (new)
**Current:** Only unit tests for retrieval, generation, XAI individually
**Fix:** Add tests for:
- `/ask` endpoint full round-trip (question -> retrieval -> generation -> XAI -> response)
- `/chat` endpoint with session context
- `/health` endpoint with component status
- Cache hit/miss behavior
- Error handling (invalid input, timeout simulation)
**Impact:** Catch integration regressions before they reach users

### 5.4 Investigate BioMistral Poor Performance
**Data:** BioMistral scores *worse* than TinyLlama on all metrics (keyword 0.21 vs 0.48, ROUGE-L 0.17 vs 0.23) despite being 7x larger.
**Possible causes:**
- GGUF quantization too aggressive (Q4_K_M)
- Prompt template mismatch (BioMistral may expect different format)
- CPU inference causing truncation or timeout issues (143s latency)
**Fix:** Investigate prompt template compatibility. If prompt is wrong, fix it. If model is genuinely worse for this task, document it and remove from production options.
**Impact:** Either unlock a better model or simplify by removing a bad option

---

## Phase 6: Advanced RAG Improvements (Day 7-10)

### 6.1 Semantic Chunking
**File:** `src/data_pipeline/preprocessors/chunker.py`
**Current:** Fixed 512-token chunks, 50-token overlap
**Fix:**
- Implement recursive character text splitter that respects sentence boundaries
- Use document-type-aware chunk sizes:
  - PubMedQA abstracts: 256 tokens (already concise)
  - MedMCQA questions: 128 tokens (short Q&A pairs)
  - HealthCareMagic consultations: 512 tokens (longer dialogues)
  - Clinical guidelines: 768 tokens (need more context)
- Ensure chunks end at sentence boundaries (split on `. ` not mid-word)
**Impact:** Better retrieval relevance, fewer truncated medical concepts
**Note:** Requires knowledge base rebuild (use Colab MCP for GPU-accelerated embedding)

### 6.2 Document-Level Safety Filtering at Retrieval
**File:** `src/retrieval/hybrid_retriever.py`
**Problem:** Safety guardrails filter LLM output but not retrieved source documents
**Fix:**
- Add lightweight content filter on retrieved chunks before passing to generation
- Flag chunks containing harmful/misleading content
- Add metadata tag `safety_reviewed: true/false` during indexing
**Impact:** Prevents harmful knowledge base content from appearing in source citations

### 6.3 Embedding Model Consistency Check
**File:** `src/embeddings/vector_store.py`, `api/main.py`
**Problem:** No verification that query embedding model matches index embedding model
**Fix:**
- Store embedding model name + dimension in ChromaDB metadata during indexing
- At startup, read metadata and verify query model matches
- Fail fast with clear error if mismatch detected
**Impact:** Prevents silent retrieval failures from dimension mismatches

---

## Phase 7: Colab MCP Specific Tasks

These tasks benefit from GPU and should be run via the connected Colab MCP:

| Task | Notebook/Script | What It Produces | When to Run |
|------|----------------|------------------|-------------|
| Rebuild KB with semantic chunks | `scripts/build_knowledge_base_colab.py` (modified) | New ChromaDB index | After Phase 6.1 |
| Full ablation study (97 cases) | `evaluation/colab_runner.ipynb` | Updated ablation.json + plots | After Phase 3.1 |
| Re-calibrate confidence | `evaluation/compute_calibration.py` | Updated calibration.json + reliability diagrams | After Phase 3.1 |
| Fine-tune with more steps | `notebooks/medical_qa_finetuning_colab.ipynb` | Improved LoRA adapter | After Phase 5.4 |
| BioMistral prompt investigation | `notebooks/BioMistral_Evaluation_Colab.ipynb` | Diagnosis of poor performance | Phase 5.4 |

---

## Execution Checklist

### Phase 1 - Bug Fixes (Est. 1-2 hours)
- [ ] 1.1 Commit unstaged fixes
- [ ] 1.2 Fix silent except:pass blocks
- [ ] 1.3 Add frontend API timeout

### Phase 2 - Performance (Est. 2-3 hours)
- [ ] 2.1 Pre-init BM25 at startup
- [ ] 2.2 Pre-load NLI model at startup
- [ ] 2.3 Enable response cache by default

### Phase 3 - RAG Quality (Est. 4-6 hours)
- [ ] 3.1 Fix confidence score normalization
- [ ] 3.2 Fix RRF score range
- [ ] 3.3 Improve source attribution
- [ ] 3.4 Negation-aware emergency detection

### Phase 4 - Architecture (Est. 3-4 hours)
- [ ] 4.1 Pipeline consolidation (startup-time selection)
- [ ] 4.2 Component health check endpoint
- [ ] 4.3 Clean up dead frontend code

### Phase 5 - Evaluation (Est. 4-6 hours)
- [ ] 5.1 Expand ablation study (Colab)
- [ ] 5.2 Re-calibrate confidence scorer (Colab)
- [ ] 5.3 Add integration tests
- [ ] 5.4 Investigate BioMistral performance

### Phase 6 - Advanced RAG (Est. 6-8 hours)
- [ ] 6.1 Semantic chunking + KB rebuild (Colab)
- [ ] 6.2 Document-level safety filtering
- [ ] 6.3 Embedding model consistency check

---

## Success Criteria

After all phases, target metrics (on 97-case test set):

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| Keyword Coverage | 0.484 | 0.55+ | Better chunking + retrieval |
| ROUGE-L | 0.234 | 0.28+ | Better context + generation |
| BERTScore F1 | 0.186 | 0.22+ | Better semantic matching |
| ECE (calibrated) | 0.071 | 0.05 | Signal normalization + re-calibration |
| Warm Latency | 106s | <60s | BM25 pre-init + caching |
| Cold First Query | 277s | <30s | Pre-load all models at startup |
| False Positive Emergency | Unknown | <5% | Negation-aware detection |
