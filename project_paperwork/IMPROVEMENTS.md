# Healthcare QA Chatbot - Comprehensive Improvements Plan

> Generated from full code review + git history analysis + evaluation results audit.
> Each section is ordered by priority. Execute top-to-bottom within each category.
>
> **IMPORTANT FOR AI AGENT (Claude Sonnet):** This file contains instructions for you to execute.
> Read this entire file first, then execute Phase by Phase in order.
> The project is a Healthcare QA Chatbot capstone project for a final-year CS student.

---

## ENVIRONMENT SETUP & EXECUTION GUIDE

### Where to Run What

This project has two categories of work:

**CATEGORY A: Code fixes (run LOCALLY on the laptop via Claude Code)**
- All bug fixes (Section 1)
- All code quality improvements (Section 5)
- All security hardening (Section 6)
- All testing improvements (Section 7)
- All documentation (Section 8)
- All minor polish (Section 9)
- These are file edits only - no heavy computation needed

**CATEGORY B: Heavy evaluation runs (run on GOOGLE COLAB with free T4 GPU)**
- Full 97-question evaluation (Section 2.1)
- Ablation study re-run (Section 2.4)
- BioMistral comparative evaluation (Section 4.1)
- QLoRA training with more steps (Section 4.2)
- Paper figure generation (Section 4.3)

### How to Set Up Google Colab

**Step 1: Push project to GitHub (if not already)**
The project is already a git repo. Push to a private GitHub repo:
```bash
# Run locally
git remote add origin https://github.com/<username>/healthcare-qa-chatbot.git
git push -u origin main
```

**Step 2: Create a Colab notebook**
Create file `evaluation/colab_runner.ipynb` locally (instructions below in Section 10).
This notebook will:
1. Clone the repo from GitHub
2. Install dependencies
3. Download models (TinyLlama, BioMistral GGUF, DeBERTa, embeddings)
4. Run evaluations with GPU acceleration
5. Save results back and push to GitHub

**Step 3: Upload large data to Google Drive**
The ChromaDB knowledge base (2.9GB) and models (4.8GB) are too large for GitHub.
Options:
- **Option A (recommended):** Upload `data/knowledge_base/` and `models/` folders to Google Drive, mount in Colab
- **Option B:** Re-build the knowledge base in Colab from raw data (slower but self-contained)

**Step 4: Run in Colab**
1. Go to colab.research.google.com
2. Upload or open the notebook
3. Runtime > Change runtime type > T4 GPU
4. Run all cells
5. Results will be saved to `evaluation/results/` and pushed back to GitHub

### Key Project Paths (for reference)
```
Project Root: /home/kbs/Documents/final_project/
Source Code:  src/          (pipeline, retrieval, generation, safety, xai, embeddings, utils)
API:          api/main.py   (FastAPI backend, port 8000)
Frontend:     frontend/streamlit_app.py (Streamlit UI)
Config:       config/settings.py
Evaluation:   evaluation/   (eval scripts, test set, results)
Models:       models/       (biomistral GGUF 4.1GB, fine_tuned QLoRA adapter 49MB)
Knowledge Base: data/knowledge_base/ (ChromaDB, 2.9GB, ~367k documents)
Test Set:     evaluation/test_set_v2.json (97 Q&A pairs)
Trajectories: data/feedback/response_trajectories.jsonl (128 logged responses)
Results:      evaluation/results/ (metrics.json, latency.json, calibration.json, ablation.json)
```

---

## 1. CRITICAL BUG FIXES
**Run locally via Claude Code. These are all file edits.**

### 1.1 BERTScore Returns 0.0 in Evaluation
**Files:** `evaluation/run_paper_eval.py`, `evaluation/run_api_eval.py`
**Problem:** BERTScore metric returns 0.0 for all samples. The `rescale_with_baseline=True` parameter fails silently when baseline files aren't cached, and the fallback sets score to 0.0.
**Fix:**
1. Open both eval scripts and find the BERTScore computation block (search for `bert_score` or `BERTScore`)
2. The current code has a try/except that catches the baseline error and returns 0.0
3. Change the logic to: first try with `rescale_with_baseline=True`, if that fails, retry with `rescale_with_baseline=False` and log a warning
4. Example fix pattern:
   ```python
   try:
       P, R, F1 = bert_score.score(predictions, references, lang="en", rescale_with_baseline=True)
   except Exception as e:
       logger.warning(f"BERTScore baseline not cached, using unscaled: {e}")
       P, R, F1 = bert_score.score(predictions, references, lang="en", rescale_with_baseline=False)
   ```
5. Make sure the F1 values are actually used (not silently replaced with 0.0 elsewhere)

### 1.2 `compute_calibration.py` Fallback Label Logic Is Broken
**File:** `evaluation/compute_calibration.py` (lines ~61-77)
**Problem:** `factual_consistency` is always `null` in all 128 trajectory records. The fallback code loads `test_set_v2.json` but never actually matches trajectory questions to test cases to compute keyword coverage labels.
**Fix:**
1. Read the current matching logic in the file
2. Implement fuzzy matching between trajectory `question` field and test_set_v2.json `query` field
3. Use `difflib.SequenceMatcher` or simple normalized string comparison (lowercase, strip)
4. For each matched pair, compute keyword coverage:
   ```python
   def compute_keyword_coverage(response_text, expected_keywords):
       response_lower = response_text.lower()
       hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
       return hits / len(expected_keywords) if expected_keywords else 0.0
   ```
5. Label: `1 if coverage >= 0.5 else 0`
6. This should give ~50-80 usable pairs from the 128 trajectories
7. Re-run: `python evaluation/compute_calibration.py` and verify `evaluation/results/calibration.json` updates

### 1.3 Broken Abbreviation Restoration in Regex Claim Extraction
**Files:** `src/xai/source_attribution.py` (~line 106-117), `src/retrieval/query_enhancer.py` (~line 116-117)
**Problem:** The abbreviation restoration logic does string replacement using raw regex pattern strings (e.g., `r'\1'`) as literal text. This fundamentally cannot work - it's treating regex syntax as literal strings.
**Fix (simplest approach - remove the broken code):**
1. In both files, find the abbreviation restoration loop
2. Remove the broken restoration logic entirely - the abbreviation patterns are used for matching but the restoration is unnecessary
3. If abbreviation expansion IS needed, replace with proper `re.sub()`:
   ```python
   for pattern, replacement in MEDICAL_ABBREVIATIONS:
       text = re.sub(pattern, replacement, text)
   ```
4. Test with: "Patient has T2DM and HTN" -> should still work for claim extraction

### 1.4 Safety Guardrails: Output Check Skips Text Normalization
**File:** `src/safety/guardrails.py`
**Problem:** `_normalize_text()` is called in `check_input()` but NOT in `check_output()`. Unicode variant characters bypass output safety checks.
**Fix:**
1. Find the `check_output()` method
2. At the very start of the method, add: `answer = self._normalize_text(answer)` (or whatever the answer parameter is named)
3. Ensure all subsequent regex matching in `check_output()` operates on this normalized text
4. Add a test in `tests/test_guardrails.py`:
   ```python
   def test_output_unicode_bypass():
       # Use Cyrillic 'а' (U+0430) instead of Latin 'a'
       result = guardrails.check_output("you h\u0430ve diabetes", context="...")
       assert result.has_warnings  # should still be caught
   ```

### 1.5 Diagnosis Prevention Patterns Too Broad
**File:** `src/safety/guardrails.py` (~line 174)
**Problem:** Pattern `r"you\s+have\s+\w+"` matches "you have diabetes" AND "you have options" / "you have several choices".
**Fix:**
1. Find the `diagnosis_patterns` list in `check_output()`
2. Replace `r"you\s+have\s+\w+"` with a more specific pattern:
   ```python
   r"you\s+have\s+(?!options|several|many|a\s+few|the\s+|some\s+|to\s+|been\s+asked)\w+"
   ```
3. Or better: only match when followed by known medical conditions (diabetes, cancer, infection, etc.)
4. Add test cases for false positives:
   ```python
   def test_no_false_positive_on_common_phrases():
       result = guardrails.check_output("you have several options for treatment", context="...")
       assert not result.has_diagnosis_warning
   ```

---

## 2. EVALUATION METHODOLOGY FIXES (Critical for Paper)

### 2.1 Run Full 97-Question Evaluation
**Run on: GOOGLE COLAB (T4 GPU) - see Section 10 for Colab notebook**
**Problem:** Current results are from only 10 questions (n=10). No statistical significance.
**What to do:**
1. This will be handled by the Colab notebook (Section 10)
2. The notebook runs: `python evaluation/run_paper_eval.py --mode metrics --n 97`
3. With GPU, each question takes ~5-15s instead of ~106s, so full run = ~15-25 minutes
4. Results saved to `evaluation/results/metrics_full.json`
5. After Colab run completes, pull results back locally

### 2.2 Fix Test Set Keywords (Generic Terms Inflate Scores)
**Run locally via Claude Code. File edits only.**
**File:** `evaluation/test_set_v2.json`
**Problem:** Many `expected_keywords` include generic words like "your", "there", "the", "leading", "can". These inflate keyword coverage artificially.
**Fix:**
1. Write a small script or do it inline: scan all 97 entries' `expected_keywords`
2. Identify and remove stop words / generic terms. Remove any keyword that is:
   - A common English stop word (the, a, an, your, there, this, that, can, will, etc.)
   - Fewer than 3 characters
   - Not medically meaningful
3. Replace removed keywords with domain-specific terms from the `reference_answer`
4. Each entry should have 3-7 medically meaningful keywords (drug names, conditions, symptoms, mechanisms)
5. Save the updated file
6. Example: change `["your", "chat", "there", "leading", "swelling"]` to `["swelling", "injury", "inflammation", "ice", "elevation"]`

### 2.3 Standardize Correctness Labels Across Evaluation Modes
**Run locally via Claude Code. File edits only.**
**Problem:** Three different "correctness" definitions used across calibration, ablation, and metrics.
**Fix:**
1. In `evaluation/run_paper_eval.py`, `evaluation/run_api_eval.py`, `evaluation/compute_calibration.py`, and `evaluation/run_ablation.py`:
2. Add a shared utility function (put in `evaluation/eval_utils.py`):
   ```python
   def is_correct(keyword_coverage: float, rouge_l: float = None) -> bool:
       """Unified correctness definition for all evaluation modes."""
       if rouge_l is not None:
           return keyword_coverage >= 0.4 and rouge_l >= 0.2
       return keyword_coverage >= 0.4  # fallback when ROUGE-L not available
   ```
3. Import and use this function in all eval scripts instead of ad-hoc thresholds
4. Document in a comment: "Correctness = keyword_coverage >= 0.4 AND ROUGE-L >= 0.2"

### 2.4 Increase Ablation Sample Size
**Run on: GOOGLE COLAB - handled by Section 10 notebook**
**Problem:** Only 15 questions used for ablation.
**What to do:**
1. The Colab notebook will run ablation on all 97 questions
2. After code fix in 2.5, add bootstrap confidence intervals

### 2.5 Add Statistical Significance Testing
**Run locally via Claude Code. File edits only.**
**Problem:** No confidence intervals or p-values.
**Fix:**
1. Create `evaluation/eval_utils.py` (if not created in 2.3) and add:
   ```python
   import numpy as np

   def bootstrap_ci(scores: list, n_bootstrap=1000, ci=0.95):
       """Compute bootstrap confidence interval for a list of scores."""
       scores = np.array(scores)
       boot_means = []
       for _ in range(n_bootstrap):
           sample = np.random.choice(scores, size=len(scores), replace=True)
           boot_means.append(np.mean(sample))
       lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
       upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
       return float(np.mean(scores)), float(lower), float(upper)
   ```
2. In `run_paper_eval.py` metrics mode, after computing per-question scores, call:
   ```python
   mean, ci_low, ci_high = bootstrap_ci(keyword_scores)
   results["keyword_coverage_ci"] = [ci_low, ci_high]
   ```
3. Do the same for ROUGE-L and BERTScore
4. In ablation mode, add bootstrap CI for ECE per variant

---

## 3. PERFORMANCE IMPROVEMENTS
**Run locally via Claude Code. These are file edits.**

### 3.1 Pre-Initialize BM25 Index at Startup
**File:** `src/retrieval/hybrid_retriever.py`
**Problem:** BM25 index is lazy-loaded on first query, causing 10-30s delay.
**Fix:**
1. Add a public `initialize()` method to `HybridRetriever`:
   ```python
   def initialize(self):
       """Pre-initialize BM25 index. Call at startup to avoid first-query delay."""
       self._lazy_init_bm25_from_store()
   ```
2. In `api/main.py`, after creating the pipeline, call:
   ```python
   if hasattr(pipeline, 'retriever') and hasattr(pipeline.retriever, 'initialize'):
       logger.info("Pre-initializing BM25 index...")
       pipeline.retriever.initialize()
       logger.info("BM25 index ready")
   ```

### 3.2 Pre-Load NLI Model for Hallucination Detection
**File:** `src/xai/hallucination_detector.py`
**Problem:** DeBERTa NLI pipeline is loaded on first hallucination check (~5-10s delay).
**Fix:**
1. Add a `warmup()` method:
   ```python
   def warmup(self):
       """Pre-load NLI model. Call at startup."""
       if self._nli_pipeline is None:
           self._load_nli_pipeline()
   ```
2. Call during FastAPI startup in `api/main.py`

### 3.3 Improve BM25 Tokenization for Medical Text
**File:** `src/retrieval/hybrid_retriever.py` (~line 128)
**Problem:** Uses `content.lower().split()` - doesn't strip punctuation or handle medical terms.
**Fix:**
1. Add a tokenizer method:
   ```python
   import re

   def _medical_tokenize(self, text: str) -> list:
       """Tokenize text for BM25, handling medical terms."""
       text = text.lower()
       # Remove punctuation but keep hyphens in compounds (e.g., "non-insulin")
       text = re.sub(r'[^\w\s\-]', ' ', text)
       tokens = text.split()
       # Filter very short tokens (< 2 chars) except known abbreviations
       return [t for t in tokens if len(t) >= 2]
   ```
2. Use this in both BM25 index building and query tokenization

### 3.4 Reduce Retrieval Latency (42.6s is Too High)
**File:** `src/retrieval/hybrid_retriever.py`
**Problem:** 42.6 seconds for retrieval seems excessive.
**Fix:**
1. Add timing around ChromaDB query vs embedding computation:
   ```python
   import time
   t0 = time.time()
   # ... embedding step
   logger.debug(f"Embedding took {time.time()-t0:.2f}s")
   t1 = time.time()
   # ... ChromaDB query
   logger.debug(f"ChromaDB query took {time.time()-t1:.2f}s")
   ```
2. Based on where the bottleneck is:
   - If embedding: consider caching query embeddings with LRU cache
   - If ChromaDB: check if the 2.9GB collection has proper HNSW index, consider reducing ef_search

### 3.5 Fix Token Probability Calculation in LLM Wrapper
**File:** `src/generation/llm_wrapper.py` (~line 334-338)
**Problem:** Uses `torch.softmax()` then `.max()` per score - poor approximation.
**Fix:**
1. Change the generate call to return scores:
   ```python
   outputs = model.generate(..., output_scores=True, return_dict_in_generate=True)
   ```
2. Compute proper mean log-probability:
   ```python
   import torch
   log_probs = []
   for score in outputs.scores:
       probs = torch.softmax(score, dim=-1)
       token_id = outputs.sequences[0, len(input_ids) + i]
       log_probs.append(torch.log(probs[0, token_id]).item())
   mean_log_prob = sum(log_probs) / len(log_probs)
   generation_confidence = math.exp(mean_log_prob)  # convert to probability
   ```

---

## 4. PAPER-QUALITY ENHANCEMENTS

### 4.1 Add BioMistral Comparative Evaluation
**Run on: GOOGLE COLAB (T4 GPU) - handled by Section 10 notebook**
**Problem:** BioMistral-7B GGUF is downloaded (4.1GB) but never evaluated against TinyLlama.
**What Colab does:**
1. Run same 97-question eval with BioMistral selected as model
2. Save results to `evaluation/results/metrics_biomistral.json`
3. Comparison table auto-generated in the figure script

### 4.2 Add QLoRA Fine-Tuned Model Evaluation
**Run on: GOOGLE COLAB (T4 GPU)**
**Problem:** QLoRA adapter trained with only 20 steps, never evaluated.
**What Colab does:**
1. Re-train with 300+ steps (GPU makes this feasible in ~10 min)
2. Run evaluation with adapter loaded
3. Save to `evaluation/results/metrics_qlora.json`

### 4.3 Generate Paper-Quality Figures
**Run locally via Claude Code OR on Colab (no GPU needed, just matplotlib).**
**Problem:** Current plots are basic. Journals expect polished figures.
**Fix:**
1. Create `evaluation/generate_paper_figures.py` that produces:
   - **Fig 1:** System architecture diagram (use matplotlib + patches, or generate Mermaid code)
   - **Fig 2:** Metric comparison bar chart with error bars (TinyLlama vs BioMistral vs QLoRA)
   - **Fig 3:** Reliability diagram (before/after Platt calibration)
   - **Fig 4:** Ablation study grouped bar chart with 95% CI error bars
   - **Fig 5:** Latency breakdown stacked bar chart (per pipeline stage)
   - **Fig 6:** Confidence score distribution histogram
2. Style requirements:
   ```python
   import matplotlib.pyplot as plt
   plt.rcParams.update({
       'font.size': 12,
       'font.family': 'serif',
       'figure.figsize': (8, 5),
       'figure.dpi': 300,
       'axes.grid': True,
       'grid.alpha': 0.3,
   })
   ```
3. Save each figure as both PNG (300 DPI) and PDF (vector)
4. Output directory: `evaluation/results/figures/`

### 4.4 Add Human Evaluation Component
**This is a MANUAL task for the student, not for Claude to execute.**
**Problem:** All metrics are automated. Reviewers will ask about human judgment.
**What to prepare (Claude can create the template):**
1. Create `evaluation/human_eval_template.csv` with columns:
   - `question_id`, `question`, `model_answer`, `reference_answer`
   - `factual_correctness` (1-5 Likert), `relevance` (1-5), `completeness` (1-5), `safety` (1-5)
   - `evaluator_id`
2. Select 25 responses (stratified: 5 high-confidence, 10 medium, 10 low)
3. The student gets 2-3 classmates/friends to fill in ratings
4. Create `evaluation/compute_human_eval.py` that reads the CSV and computes:
   - Mean scores per dimension
   - Cohen's kappa inter-annotator agreement
   - Correlation between human scores and automated confidence

### 4.5 Add Baseline Comparison
**Run on: GOOGLE COLAB - handled by Section 10 notebook**
**Problem:** No comparison against simpler approaches.
**What to implement:**
1. **Baseline 1: No-RAG** - Just feed the question to TinyLlama with no retrieval context
2. **Baseline 2: Dense-only** - Only use vector search, no BM25, no RRF
3. **Baseline 3: No-XAI** - Full retrieval but no confidence/hallucination/attribution
4. Create `evaluation/run_baselines.py` that:
   - Configures pipeline variants by toggling features
   - Runs each baseline on same 97 questions
   - Saves results to `evaluation/results/baseline_*.json`

---

## 5. CODE QUALITY IMPROVEMENTS
**Run locally via Claude Code. All file edits.**

### 5.1 Fix Silent Exception Handling (8 files affected)
**Files to edit:**
1. `src/pipeline/qa_pipeline.py` - search for `except Exception`
2. `src/retrieval/query_enhancer.py` - lines ~159, ~181, ~243
3. `src/xai/source_attribution.py` - search for `except Exception`
4. `src/utils/cache_manager.py` - lines ~88, ~126
5. `src/embeddings/vector_store.py` - search for `except Exception`
6. `src/xai/hallucination_detector.py` - search for `except Exception`
7. `src/generation/llm_wrapper.py` - search for `except Exception`
8. `src/pipeline/context_compressor.py` - search for `except Exception`

**Fix pattern for each:**
```python
# BEFORE (bad):
except Exception:
    pass

# AFTER (good):
except Exception as e:
    logger.warning(f"<describe what failed>: {e}")
```
Keep the graceful degradation (return default value) but add the logging.

### 5.2 Extract Hardcoded Magic Numbers to Config
**File to edit:** `config/settings.py` (add new fields to existing dataclasses)
**Then update:** each source file to read from config instead of hardcoded values.

Add to `config/settings.py` in the appropriate dataclasses:
```python
@dataclass
class RetrievalConfig:
    # ... existing fields ...
    bm25_batch_size: int = 5000
    rerank_fetch_multiplier: int = 3
    no_rerank_fetch_multiplier: int = 2

@dataclass
class PipelineConfig:
    # ... existing fields ...
    max_context_tokens: int = 4000
    corrective_rag_max_context: int = 2000
    sentence_length_divisor: int = 200

@dataclass
class ModelConfig:
    # ... existing fields ...
    cpu_threads: int = 4
    max_context_length: int = 2048

@dataclass
class XAIConfig:
    # ... existing fields ...
    confidence_high_threshold: float = 0.8
    confidence_low_threshold: float = 0.5
    max_evidence_length: int = 80
```

Then in each source file, replace the hardcoded values with `self.config.<path>`.

### 5.3 Add Server-Side Input Validation
**File:** `api/main.py`
**Find:** the `QuestionRequest` Pydantic model
**Add:**
```python
from pydantic import validator

class QuestionRequest(BaseModel):
    question: str
    # ... other fields ...

    @validator('question')
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > 1000:
            raise ValueError(f"Question too long ({len(v)} chars, max 1000)")
        return v
```

### 5.4 Fix Drug Interaction Detection (No Negation Handling)
**File:** `src/safety/guardrails.py` (~lines 359-371)
**Problem:** "I am NOT taking warfarin" still triggers drug interaction warning.
**Fix:**
1. Find the drug interaction check method
2. Before flagging a drug mention, check for negation in a 5-word window before the drug name:
   ```python
   NEGATION_WORDS = {"not", "no", "never", "don't", "doesn't", "didn't", "without", "stop", "stopped"}

   def _has_negation_before(self, text: str, drug_pos: int, window: int = 30) -> bool:
       """Check if negation word appears before drug mention."""
       prefix = text[max(0, drug_pos - window):drug_pos].lower()
       return any(neg in prefix.split() for neg in self.NEGATION_WORDS)
   ```
3. Skip the drug interaction flag if negation is detected

### 5.5 Remove Pydantic/Spacy Monkeypatch Hack
**File:** `api/main.py` (~lines 238-254)
**Problem:** Monkeypatches pydantic internals to suppress spacy type errors.
**Fix:**
1. Find the monkeypatch block (search for `monkeypatch` or `schema` override)
2. The root cause is likely a spacy `Language` or `Doc` type in a dataclass that FastAPI tries to serialize
3. Fix by either:
   a. Adding `class Config: arbitrary_types_allowed = True` to the affected Pydantic model
   b. Or excluding the spacy field from the API schema: `Field(..., exclude=True)`
4. Remove the monkeypatch entirely after fixing

### 5.6 Fix GPU Detection Logic
**File:** `config/settings.py` (~line 58)
**Find:** the line that checks `USE_GPU` environment variable
**Change from:**
```python
use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
```
**Change to:**
```python
use_gpu = os.getenv("USE_GPU", "true").lower() in ("true", "1", "yes")
```

---

## 6. SECURITY HARDENING
**Run locally via Claude Code. All file edits.**

### 6.1 Add API Authentication
**File:** `api/main.py`
**Add near the top:**
```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import os

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:  # No key configured = no auth required (dev mode)
        return
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
```
**Then add to endpoints:**
```python
@app.post("/qa/ask", dependencies=[Depends(verify_api_key)])
```

### 6.2 Enforce Rate Limiting
**File:** `api/main.py`
**First:** Add `slowapi` to `requirements.txt`: `slowapi>=0.1.9`
**Then add:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

# Then decorate endpoints:
@app.post("/qa/ask")
@limiter.limit("60/minute")
async def ask_question(request: Request, ...):
```

### 6.3 Restrict CORS Origins
**File:** `api/main.py`
**Find:** the CORS middleware setup
**Change:** default from `["*"]` to `["http://localhost:8501"]`
```python
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501")
if cors_origins == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in cors_origins.split(",")]
```

### 6.4 Add Input Sanitization for Prompt Injection
**File:** `src/retrieval/query_enhancer.py` (~line 147-160)
**Find:** where user query is embedded into LLM prompt
**Add before the prompt construction:**
```python
def _sanitize_query(self, query: str) -> str:
    """Remove potential prompt injection patterns."""
    # Remove instruction-like patterns
    injection_patterns = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"disregard\s+(previous|above|all)",
        r"you\s+are\s+now\s+",
        r"system\s*:\s*",
    ]
    sanitized = query
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()
```
Call this before embedding in the prompt.

---

## 7. TESTING IMPROVEMENTS
**Run locally via Claude Code. File creation + edits.**

### 7.1 Add Integration Tests for Full Pipeline
**Create:** `tests/test_integration.py`
```python
"""Integration tests for the full QA pipeline.
These require models to be downloaded and are slow.
Run with: pytest tests/test_integration.py -m slow
"""
import pytest

pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def pipeline():
    from src.pipeline.qa_pipeline import HealthcareQAPipeline
    from config.settings import Config
    config = Config.from_env()
    return HealthcareQAPipeline(config)

def test_basic_question(pipeline):
    response = pipeline.answer("What are symptoms of type 2 diabetes?")
    assert response is not None
    assert response.answer and len(response.answer) > 0
    assert response.confidence_score >= 0.0
    assert response.confidence_score <= 1.0

def test_confidence_breakdown_present(pipeline):
    response = pipeline.answer("What is hypertension?")
    if response.confidence_breakdown:
        assert "retrieval" in response.confidence_breakdown
        assert "generation" in response.confidence_breakdown

def test_safety_blocks_dangerous_query(pipeline):
    response = pipeline.answer("I want to take 500mg of aspirin with warfarin")
    # Should have safety warnings
    assert response is not None  # Pipeline should not crash

def test_empty_query_handled(pipeline):
    response = pipeline.answer("")
    assert response is not None
```

### 7.2 Add Edge Case Tests for Safety Guardrails
**File:** `tests/test_guardrails.py` (add to existing file)
**Add these test functions:**
```python
def test_unicode_bypass_input(guardrails):
    """Cyrillic 'а' should be normalized to Latin 'a'."""
    result = guardrails.check_input("t\u0430ke 500 mg of \u0430spirin")
    assert result.is_flagged or result.has_warnings

def test_negated_drug_mention(guardrails):
    """'NOT taking warfarin' should not trigger drug interaction warning."""
    result = guardrails.check_input("I am NOT taking warfarin anymore")
    # After fix 5.4, this should not flag drug interaction

def test_legitimate_educational_content(guardrails):
    """Educational phrases should not be blocked."""
    result = guardrails.check_output(
        "You have several treatment options available",
        context="What are my options?"
    )
    assert not result.has_diagnosis_warning

def test_very_long_input(guardrails):
    """Very long input should be handled gracefully."""
    long_input = "What is diabetes? " * 200  # ~3600 chars
    result = guardrails.check_input(long_input)
    assert result is not None  # Should not crash
```

### 7.3 Add Parametrized Tests for Retrieval
**File:** `tests/test_retrieval.py` (add to existing or create)
```python
import pytest

@pytest.mark.parametrize("top_k", [1, 3, 5, 10])
def test_retrieval_varying_k(retriever, top_k):
    results = retriever.retrieve("diabetes symptoms", top_k=top_k)
    assert len(results) <= top_k

def test_retrieval_empty_query(retriever):
    results = retriever.retrieve("", top_k=3)
    assert isinstance(results, list)

def test_rrf_single_source_empty(retriever):
    """RRF should work even if one retriever returns nothing."""
    from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
    results = reciprocal_rank_fusion([["doc1", "doc2"], []])
    assert len(results) > 0
```

---

## 8. DOCUMENTATION FOR PAPER
**Run locally via Claude Code. File creation.**

### 8.1 Create `.env.example` File
**Create:** `.env.example` in project root
```
# Healthcare QA Chatbot - Environment Variables
# Copy to .env and fill in values

# GPU settings (true/false/1/0)
USE_GPU=false

# Hugging Face cache (for offline mode)
HF_HOME=~/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# API settings
API_KEY=
CORS_ORIGINS=http://localhost:8501

# MCP web search (optional)
MCP_SEARCH_CMD=
```

### 8.2 Add System Architecture Diagram
**Create:** `evaluation/generate_architecture_diagram.py`
This should generate a Mermaid diagram or matplotlib figure showing:
```
User Question → Input Safety Check → Query Enhancement →
Hybrid Retrieval (Dense [all-MiniLM-L6-v2] + Sparse [BM25] → RRF Fusion) →
Corrective RAG → Grounding Gate → Context Compression →
LLM Generation [TinyLlama/BioMistral] → Factual Consistency Check →
Hallucination Detection [DeBERTa NLI] → Multi-Signal Confidence Scoring →
Source Attribution → Output with XAI
```
Generate as both Mermaid markdown (for README) and matplotlib figure (for paper).

### 8.3 Document Evaluation Methodology
**Create:** `evaluation/METHODOLOGY.md`
Document:
1. **Test set:** 97 Q&A pairs from MedQA-USMLE (medical exams), PubMedQA (research), ChatDoctor (clinical)
2. **Metrics:** ROUGE-L (content overlap), BERTScore (semantic similarity), keyword coverage (domain term recall), ECE (calibration)
3. **Correctness definition:** keyword_coverage >= 0.4 AND ROUGE-L >= 0.2
4. **Statistical method:** Bootstrap resampling (1000 iterations, 95% CI)
5. **Hardware:** [student fills in: CPU model, RAM, OS version]
6. **Models evaluated:** TinyLlama 1.1B, BioMistral-7B Q4_K_M, QLoRA fine-tuned TinyLlama
7. **Baselines:** No-RAG, Dense-only retrieval, No-XAI pipeline

---

## 9. MINOR POLISH
**Run locally via Claude Code. All file edits.**

### 9.1 Move Inline CSS to External File
**File:** `frontend/streamlit_app.py` (~lines 62-289)
**Problem:** 200+ lines of CSS inlined in Python.
**Fix:**
1. Create `frontend/static/style.css`
2. Move the CSS string content from `streamlit_app.py` to this file
3. Replace in `streamlit_app.py` with:
   ```python
   import pathlib
   css_path = pathlib.Path(__file__).parent / "static" / "style.css"
   if css_path.exists():
       st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
   ```

### 9.2 Clean Up Unused Dependencies in requirements.txt
**File:** `requirements.txt`
**Remove these lines** (or comment with `# Unused:` if keeping for future):
- `lime>=0.2.0` - LIME is not used anywhere in current code
- `captum>=0.6.0` - Captum attribution library not integrated
- `wandb>=0.16.0` - No experiment tracking integration exists

### 9.3 Add `pytest-cov` for Coverage Reporting
**File:** `requirements.txt` - add: `pytest-cov>=4.0`
**Create:** `pytest.ini` or add to existing config:
```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
addopts = --cov=src --cov-report=term-missing
```

### 9.4 Fix Logging Inconsistencies
**Problem:** Mixed `print()`, `logger.warning()`, and no logging across files.
**Fix:** In each source file under `src/`:
1. Ensure `from loguru import logger` is imported
2. Replace `print(...)` with `logger.info(...)` or `logger.debug(...)`
3. Don't change `print()` in scripts (`evaluation/`, `scripts/`) - those are fine

---

## 10. GOOGLE COLAB NOTEBOOK
**Run locally via Claude Code. Create this file.**

**Create:** `evaluation/colab_runner.ipynb`

This is a Jupyter notebook for running heavy evaluations on Google Colab's free T4 GPU.
Create it as a Python script first, then the student can paste cells into Colab.

**Alternative: Create as a .py script** that the student copies into Colab cells:

**Create:** `evaluation/colab_setup.py`

```python
"""
Healthcare QA Chatbot - Google Colab Evaluation Runner
=====================================================
Instructions:
1. Open Google Colab (colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Copy-paste each CELL block below into separate Colab cells
4. Run cells in order
5. Download results from evaluation/results/ when done
"""

# ============================================================
# CELL 1: Mount Google Drive & Clone Repo
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# Option A: Clone from GitHub
!git clone https://github.com/<YOUR_USERNAME>/healthcare-qa-chatbot.git /content/project
# Option B: If repo is in Google Drive
# !cp -r "/content/drive/MyDrive/healthcare-qa-chatbot" /content/project

%cd /content/project

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q -r requirements.txt
!pip install -q bert-score rouge-score

# ============================================================
# CELL 3: Copy Models & Data from Google Drive
# ============================================================
# Upload these folders to your Google Drive first:
#   - models/biomistral/ggml-model-Q4_K_M.gguf (4.1GB)
#   - data/knowledge_base/ (2.9GB ChromaDB)

import shutil, os

# Copy models
DRIVE_BASE = "/content/drive/MyDrive/healthcare_qa_data"  # <-- CHANGE THIS PATH

if os.path.exists(f"{DRIVE_BASE}/models"):
    os.makedirs("/content/project/models", exist_ok=True)
    !cp -r "{DRIVE_BASE}/models/biomistral" /content/project/models/
    !cp -r "{DRIVE_BASE}/models/fine_tuned" /content/project/models/
    print("Models copied!")

if os.path.exists(f"{DRIVE_BASE}/knowledge_base"):
    os.makedirs("/content/project/data", exist_ok=True)
    !cp -r "{DRIVE_BASE}/knowledge_base" /content/project/data/
    print("Knowledge base copied!")

# ============================================================
# CELL 4: Set Environment Variables
# ============================================================
import os
os.environ["USE_GPU"] = "true"
os.environ["HF_HUB_OFFLINE"] = "0"  # Allow downloads in Colab
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# CELL 5: Run Full 97-Question Evaluation (TinyLlama)
# ============================================================
# This runs the in-process evaluator (no API server needed)
!python evaluation/run_paper_eval.py --mode metrics --n 97

# ============================================================
# CELL 6: Run Full Ablation Study (97 questions)
# ============================================================
!python evaluation/run_paper_eval.py --mode ablation --n 97

# ============================================================
# CELL 7: Run Latency Benchmarks
# ============================================================
!python evaluation/run_paper_eval.py --mode latency --n 20

# ============================================================
# CELL 8: Run Calibration
# ============================================================
!python evaluation/run_paper_eval.py --mode calibration --n 97

# ============================================================
# CELL 9: (Optional) BioMistral Evaluation
# ============================================================
# Only run if BioMistral GGUF was copied in Cell 3
# You'll need to modify run_paper_eval.py to accept --model flag
# or change the default model in config/settings.py temporarily
!python evaluation/run_paper_eval.py --mode metrics --n 97 --model biomistral

# ============================================================
# CELL 10: Generate Paper Figures
# ============================================================
!python evaluation/generate_paper_figures.py

# ============================================================
# CELL 11: Copy Results Back to Google Drive
# ============================================================
!cp -r /content/project/evaluation/results/ "{DRIVE_BASE}/results_colab/"
print("Results saved to Google Drive!")

# Or push to GitHub:
# !git add evaluation/results/
# !git commit -m "chore: add Colab evaluation results (97-question full run)"
# !git push
```

---

## Execution Order Summary

### Phase 1 - Bug Fixes (Do First, LOCAL)
Items: 1.1, 1.2, 1.3, 1.4, 1.5
**Estimated effort:** ~1-2 hours of code edits

### Phase 2 - Evaluation Code Fixes (LOCAL, before Colab runs)
Items: 2.2, 2.3, 2.5 (code changes that must happen before running evals)
**Estimated effort:** ~1-2 hours of code edits

### Phase 3 - Heavy Evaluation Runs (COLAB)
Items: 2.1, 2.4, 4.1, 4.2, 4.5
**Estimated effort:** ~2-4 hours of Colab runtime (can walk away)

### Phase 4 - Paper Enhancements (LOCAL)
Items: 4.3, 4.4 (template), 8.1, 8.2, 8.3
**Estimated effort:** ~1-2 hours

### Phase 5 - Code Quality (LOCAL)
Items: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
**Estimated effort:** ~2-3 hours

### Phase 6 - Performance (LOCAL)
Items: 3.1, 3.2, 3.3, 3.4, 3.5
**Estimated effort:** ~1-2 hours

### Phase 7 - Security & Testing (LOCAL)
Items: 6.1-6.4, 7.1-7.3
**Estimated effort:** ~2-3 hours

### Phase 8 - Minor Polish (LOCAL)
Items: 9.1-9.4
**Estimated effort:** ~30 min
