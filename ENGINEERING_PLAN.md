# Healthcare QA Chatbot — Senior Engineering Plan

**Author role:** Senior RAG Engineer / Architect
**Created:** 2026-04-07
**Project state:** ~85% of IMPROVEMENTS.md + IMPROVEMENT_PLAN.md complete. 235 tests green. 97-question eval baseline captured.
**Primary goal:** Ship a defensible capstone paper + reproducible system within 2 weeks.
**Secondary goal:** Close the retrieval-quality gap (0.484 → 0.55 keyword coverage) and fix BioMistral.

---

## Guiding Principles

1. **Measure before optimizing.** Every change must be validated against `evaluation/run_paper_eval.py` on the same 97-case test set.
2. **Cheap wins first.** Local fixes before Colab runs. Colab time is the bottleneck — batch work to minimize cold starts.
3. **Paper-first sequencing.** Whatever blocks the paper goes first. Nice-to-haves ship after submission.
4. **No regressions.** Run the full 235-test suite after every phase. A phase is not "done" until tests are green.
5. **Atomic commits.** One phase = one commit (or a tight series). Every commit should build and pass tests.
6. **Snapshot metrics before risky changes.** Before any retrieval/generation change, copy `evaluation/results/*.json` to `evaluation/results/_snapshots/<date>/` so regressions are visible.

---

## Phase 0 — Pre-Flight (1 hour)

**Goal:** Ensure clean baseline before we touch anything. No regressions hiding.

| # | Task | File(s) | Acceptance |
|---|---|---|---|
| 0.1 | Create `evaluation/results/_snapshots/2026-04-07/` and copy all current JSONs | `evaluation/results/` | Snapshot directory exists with all metrics files |
| 0.2 | Delete stale `evaluation/results/evaluation_summary.json` (dated 2026-01-29, placeholder values) | `evaluation/results/evaluation_summary.json` | File removed, git committed |
| 0.3 | Run full test suite, capture pass count | `tests/` | `235 passed` recorded as baseline |
| 0.4 | Run a single warm `/ask` query against running API, record end-to-end latency | `api.log` | Latency number written to `ENGINEERING_PLAN.md` under Phase 0 log |
| 0.5 | Commit pre-flight snapshot | — | One commit: `chore: pre-flight baseline snapshot 2026-04-07` |

**Exit criteria:** Tests green, baselines snapshotted, git clean.

**Rollback plan:** None needed — read-only phase.

---

## Phase 1 — Cheap Local Wins (3–4 hours)

**Goal:** Measurable latency + correctness improvements with zero GPU and zero risk. These ship immediately.

### 1.1 — Query embedding LRU cache
**Why:** Repeated queries in evaluation loops and common user queries re-embed from scratch. Embedding a query with `all-MiniLM-L6-v2` on CPU takes 200–800ms.
**Where:** `src/embeddings/embedder.py` (find `embed_query`)
**How:**
```python
from functools import lru_cache

class Embedder:
    @lru_cache(maxsize=512)
    def _embed_query_cached(self, query: str) -> tuple:
        # must return a tuple (hashable) — convert numpy at call site
        return tuple(self._embed_query_uncached(query))

    def embed_query(self, query: str):
        import numpy as np
        return np.array(self._embed_query_cached(query))
```
**Acceptance:** Re-querying the same question twice back-to-back shows <10ms on second call (log `embedding_ms` in retriever timings).
**Test:** Add a parametrized test in `tests/test_retrieval_parametrized.py` that calls `embed_query("diabetes")` twice and asserts the second call is >10x faster.
**Risk:** Low. LRU cache key is the raw string — no staleness possible.
**Rollback:** Remove the decorator.

---

### 1.2 — Startup warmup query
**Why:** Cold first query is 277s — the paper's worst-case number looks terrible. Running one throw-away query at startup amortizes the cost.
**Where:** `api/main.py` `on_startup` handler (after all models loaded)
**How:**
```python
# At the end of startup_event(), after all components are ready:
try:
    logger.info("Startup: running warmup query to prime caches...")
    t0 = time.time()
    _ = pipelines["standard"].answer("What is diabetes?", num_sources=2)
    logger.info(f"Startup: warmup complete in {time.time()-t0:.1f}s")
except Exception as e:
    logger.warning(f"Startup warmup failed (non-fatal): {e}")
```
**Acceptance:** First real user query after startup is within 2× of warm latency (~200s, not 277s). Log shows warmup timing.
**Test:** Manual — restart API, measure first `/ask` call.
**Risk:** Adds 100s to cold startup but saves it from the first user.
**Rollback:** Remove the block.

---

### 1.3 — Fix the 42s retrieval bottleneck
**Why:** `latency.json` shows `retrieval_ms_mean=42624` which is absurd for HNSW. Something is wrong.
**Where:** `src/retrieval/hybrid_retriever.py`
**How:**
1. Add fine-grained timing inside `retrieve()`:
   - `embedding_ms`, `chroma_query_ms`, `bm25_query_ms`, `rrf_ms`, `rerank_ms`, `safety_filter_ms`
2. Run a single query, read the log, identify the dominant stage.
3. **Most likely culprit**: `retrieve_with_context` is iterating through ALL returned docs to build a >10KB context string character-by-character. Check the loop.
4. **Second most likely**: ChromaDB's `ef_search` too high. Default is 100; try 40.
5. **Third**: Cross-encoder reranker fetches `3 * k` then re-embeds everything.
6. **Fourth**: `_safety_filter_documents` (hybrid_retriever.py:396-420) runs 3 `re.search()` calls over every document's full content. For `fetch_k=30` that's 90 regex scans over potentially multi-KB docs per query. Pre-compile patterns once at module level (`_HARMFUL_PATTERNS_RE = [re.compile(p) for p in _HARMFUL_PATTERNS]`) and add `safety_filter_ms` to the per-stage log.

**Acceptance:** Per-stage log appears in DEBUG. Total `retrieval_ms_mean` drops below 10s (target: <5s).
**Test:** `python evaluation/run_paper_eval.py --mode latency --n 5` and verify new breakdown.
**Risk:** Medium — changing HNSW params could hurt recall. Mitigation: run retrieval quality check on 20 test cases before/after.
**Rollback:** Git revert the commit.

---

### 1.4 — Remove dead code (`langgraph_nodes.py:402`, stale stubs)
**Why:** Small but signals discipline. Senior engineers don't leave `pass` after a logged exception.
**Where:**
- `src/langgraph/langgraph_nodes.py:402` — remove the `pass`
- `src/generation/llm_wrapper.py:18,22,26,30` — audit these `pass` lines (likely except-block stubs for optional imports; keep only if intentional, otherwise remove)
- `src/retrieval/hybrid_retriever.py:114` — `self.corpus_map: Dict[str, int] = {}` is built in `_init_bm25` but never read. Either wire it into `_sparse_retrieve` for O(1) dedup or delete it.
**Acceptance:** `grep -n "^                pass$" src/` returns no stray orphans.
**Risk:** None.

---

### 1.5 — Replace `__embedding_meta__` sentinel doc with sidecar JSON
**Why:** I added a sentinel document hack because ChromaDB can't update collection metadata. A sidecar `data/knowledge_base/embedding_meta.json` is cleaner and won't leak into retrieval results.
**Where:** `src/embeddings/vector_store.py`
**How:**
```python
import json as _json
_META_FILENAME = "embedding_meta.json"

def record_embedding_model(self, model_name: str, dimension: int) -> None:
    meta_path = self.persist_directory / _META_FILENAME
    meta_path.write_text(_json.dumps({
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "recorded_at": datetime.utcnow().isoformat(),
    }, indent=2))

def verify_embedding_compatibility(self, model_name: str, dimension: int) -> bool:
    meta_path = self.persist_directory / _META_FILENAME
    if not meta_path.exists():
        # Fall back to checking collection metadata sentinel doc (legacy)
        return self._verify_via_sentinel(model_name, dimension)
    meta = _json.loads(meta_path.read_text())
    # ... same checks as before
```
**Acceptance:** New KB builds write `embedding_meta.json`. Existing KB with sentinel doc still works. Sentinel document can be left in place for backward compat.
**Test:** Add unit test that creates a temp VectorStore, records, then verifies.
**Risk:** Low — purely additive.

---

### 1.6 — BM25 index pickle cache across API restarts
**Why:** Even after Phase 1.2 warmup, every API restart re-loads 182K docs from ChromaDB and re-tokenizes for BM25 (~minutes). The tokenized corpus is deterministic given the KB — persist it.
**Where:** `src/retrieval/hybrid_retriever.py` + `api/main.py`
**How:**
```python
# data/knowledge_base/bm25_index.pkl
# Key: (doc_count, embedding_meta.json mtime)
# On startup: load pickle if key matches current KB state, else rebuild and rewrite
```
**Acceptance:** Cold API startup drops from ~2min to <15s on second and subsequent runs. First run after KB rebuild regenerates pickle automatically.
**Test:** Restart API twice, confirm log shows "BM25 loaded from cache" on second boot.
**Risk:** Low — mismatched key falls back to full rebuild.

---

**Phase 1 exit criteria:**
- All 235 tests still pass
- Cold query: <200s; warm query: <100s (measured on 5 test questions)
- Retrieval latency: <10s warm
- Commits: one per sub-task or one bundled `perf: local latency & cleanup improvements`

---

## Phase 2 — Fix BioMistral (2–3 hours)

**Goal:** Either fix BioMistral so it's genuinely competitive, or document why it's not and remove it from production options. A 7B model scoring 0.21 kw vs TinyLlama 1.1B at 0.48 is a paper-quality embarrassment.

### 2.1 — Diagnose: is it prompt format or decoding?
**Where:** `src/generation/llm_wrapper.py` (the `gguf` branch)
**How:**
1. Pick 3 test questions where TinyLlama scores well (kw >0.6).
2. Run both models on them, print the raw LLM output (before any cleaning).
3. Check: is BioMistral's output truncated? repeated? off-topic? blank?
4. BioMistral-7B was trained on Mistral-instruct format: `<s>[INST] {prompt} [/INST]`. Our current template probably uses TinyLlama chat format (`<|system|>...<|user|>...`).

**Acceptance:** Root cause documented in `ENGINEERING_PLAN.md` under Phase 2 log. One of:
- (a) Prompt format mismatch → proceed to 2.2
- (b) Decoding params wrong (temp/top_p/max_tokens) → proceed to 2.3
- (c) Model genuinely worse for RAG → proceed to 2.4

---

### 2.2 — Fix prompt template (if diagnosis is format mismatch)
**Where:** `src/generation/prompts.py` or wherever prompt templates live
**How:**
```python
BIOMISTRAL_TEMPLATE = """<s>[INST] You are a medical assistant. Use the following context to answer the question.

Context:
{context}

Question: {question} [/INST]"""
```
Add model-name dispatch in the LLM wrapper so each model gets its native template.
**Acceptance:** Re-run on same 3 diagnosis questions, outputs are coherent. Then run on 20 random test cases (quick sanity check).

---

### 2.3 — Fix decoding params (if diagnosis is params)
- BioMistral GGUF Q4_K_M often needs `temperature=0.1`, `top_p=0.9`, `repeat_penalty=1.1`
- Check current llama.cpp params in `llm_wrapper.py`

---

### 2.4 — If genuinely worse: document and remove from defaults
- Add a section to `evaluation/METHODOLOGY.md`: "BioMistral evaluation"
- Keep it selectable but remove from production defaults in `config/settings.py`
- **Paper angle:** "Domain-specific LMs do not automatically dominate in RAG settings; small instruct-tuned models may outperform larger base models when retrieval provides domain context."

**Phase 2 exit criteria:**
- BioMistral score is either >0.40 kw OR fully documented as not fit for production
- Decision recorded in paper draft notes
- **Runs on Colab** if we re-evaluate (only if 2.2/2.3 succeeds)

**Risk:** Medium. BioMistral might be a dead end. Time-box to 3 hours — if no progress, execute 2.4.

---

## Phase 3 — Colab Pro Batch Run (~2h wall clock, parallel sessions)

**Goal:** Run every GPU-dependent task including the KB rebuild (formerly Phase 4.2). User has **Colab Pro**: A100/L4 on demand, high-RAM runtime, background execution up to ~24h, priority access. Optimize for GPU choice and parallelism, not for minimizing cold starts.

**Pre-Colab (local, must land before Session A starts):**
1. Phase 4.1 semantic chunker merged and unit-tested (local code change only, no rebuild)
2. Push current `main` to GitHub
3. `data/knowledge_base/` (2.9GB v1), `models/biomistral/*.gguf`, `models/fine_tuned/` uploaded to Drive
4. **Verify** `evaluation/run_paper_eval.py` supports `--adapter` and `--model` flags (`grep -n "adapter\|--model" evaluation/run_paper_eval.py`). Add them if missing.
5. Verify `eval_utils.py` bootstrap CI is wired into metrics-mode output.

### Session A — A100 High-RAM (~2h wall clock)
Runtime hint: `a100-highram`. Needs VRAM/throughput for retrain + embedding.

| Cell | Task | Est. time |
|---|---|---|
| A1 | Env setup, mount Drive, clone repo, `pip install` | 5 min |
| A2 | Sync data + models from Drive | 10 min |
| A3 | **QLoRA retrain** (IMPROVEMENTS 4.2) — 500 steps, bs=8, lr=2e-4 → `models/fine_tuned/medical_adapter_v2/` | 15 min |
| A4 | **KB rebuild v2** with semantic chunking (Phase 4.1 output) — ~400K chunks | ~60 min |
| A5 | Write `embedding_meta.json` sidecar, push `data/knowledge_base_v2/` to Drive | 10 min |
| A6 | Commit adapter + KB index artifact manifests | 5 min |

### Session B — L4 standard (~1.5h wall clock, runs in parallel with A)
Doesn't need A100; evaluation is I/O + CPU-heavy once models load.

| Cell | Task | Est. time |
|---|---|---|
| B1 | Env setup, mount Drive, clone repo | 5 min |
| B2 | Sync **v1** KB + models from Drive | 10 min |
| B3 | **Ablation n=97** (IMPROVEMENTS 2.4) — `--mode ablation --n 97` → `ablation.json` | 25 min |
| B4 | **Recalibration n=97** (IMPROVEMENT_PLAN 5.2) → new Platt params + reliability diagrams | 15 min |
| B5 | **BioMistral re-eval n=97** (after Phase 2 fix) — `--mode metrics --model biomistral` | 25 min |
| B6 | **Baseline re-eval with bootstrap CI** (no-RAG / dense-only / no-XAI) | 20 min |
| B7 | Regenerate figures — `generate_paper_figures.py` | 5 min |
| B8 | Sync results to Drive + push to GitHub | 5 min |

### Session C — Follow-up after A+B finish (~45 min, any GPU)
Requires Session A's v2 KB artifact.

| Cell | Task | Est. time |
|---|---|---|
| C1 | Pull v2 KB from Drive | 5 min |
| C2 | **v1 vs v2 A/B eval** on 97 questions, same model — two full `run_paper_eval` passes | 30 min |
| C3 | Write decision record to `evaluation/results/kb_v1_vs_v2.json` + update `metrics_full_qlora_v2.json` | 5 min |

**Merged scope note:** Phase 3 and Phase 4.2 (old "Rebuild KB on Colab") are now one Colab visit. Phase 4.1 (semantic chunker, local) remains a prerequisite. Phase 4.3 (interpret A/B results) stays local, post-Colab.

**While Colab runs (see Phase 5.4):** Distribute the human eval template to 2-3 classmates during the ~2h wall-clock window. Wall-clock cost is near zero if parallelized.

**Acceptance for Phase 3:**
- All cells in Sessions A, B, C complete without error
- Updated JSONs pulled back to local `evaluation/results/`
- Updated figures in `evaluation/figures/`
- New `calibration.json` shows `n=97` and lower `ece_calibrated`
- New `ablation.json` shows `n=97` per variant
- `kb_v1_vs_v2.json` records A/B decision
- `models/fine_tuned/medical_adapter_v2/` exists locally

**Risk:** Pro background execution removes the free-tier timeout panic; still checkpoint after each cell to make cheap recovery possible. A100 availability can dip — L4 fallback is acceptable for Session A at ~2x the rebuild time.
**Rollback:** The Phase 0 snapshot is authoritative if new Colab results are worse. Keep `data/knowledge_base_v2/` on disk alongside v1 until A/B proves v2 wins by ≥0.03 kw.

---

## Phase 4 — Retrieval Quality (6–8 hours, highest ROI)

**Goal:** Close the 0.484 → 0.55 keyword coverage gap. This is the biggest remaining lever for paper metrics.

### 4.1 — Semantic chunking (IMPROVEMENT_PLAN 6.1)
**Why:** Fixed 512-token chunks split medical concepts mid-sentence. Domain-specific concepts (drug interactions, disease causality) need sentence-boundary-aware chunks.
**Where:** `src/data_pipeline/preprocessors/chunker.py`
**How:**
1. Implement `RecursiveSentenceChunker` using the hierarchy: `\n\n` → `\n` → `. ` → `; ` → ` `
2. Document-type-aware sizes:
   - PubMedQA abstracts → 256 tokens (already dense)
   - MedMCQA → 128 tokens (Q&A pairs are short)
   - HealthCareMagic → 512 tokens (long dialogues)
   - Clinical guidelines → 768 tokens
3. Pass `source` metadata to the chunker to dispatch on type.
4. **Validation:** write a unit test that chunks a known paragraph and asserts no chunk ends mid-sentence.

**Acceptance:** Chunk boundary test passes. Chunking speed unchanged (~ 100 docs/sec).
**Risk:** Medium — changes chunk distribution, requires KB rebuild to actually affect retrieval.

### 4.2 — Rebuild KB on Colab *(merged into Phase 3 Session A)*
The KB rebuild now lives in Phase 3 Session A cell A4 (A100 High-RAM, ~60 min). See Phase 3 for details. This section is retained as a pointer only. Acceptance criteria are the same:
- `data/knowledge_base_v2/` exists with `embedding_meta.json`
- Chunk count close to v1 (~180K, ±20% acceptable due to different chunking)
- Sample queries return reasonable results

### 4.3 — A/B evaluate v1 vs v2 KB (local interpretation of Session C results)
**How:**
1. Run `run_paper_eval.py` once against each KB
2. Compare all metrics
3. **Keep whichever wins.** Don't switch KB without evidence.

**Acceptance:** Decision recorded. If v2 wins by ≥0.03 kw, make it default.
**Rollback:** Point `persist_directory` back to `data/knowledge_base`.

### 4.4 — (Stretch) Query rewriting via LLM
**Why:** User queries are often underspecified. LangGraph variant has query refinement but it's slow. A cheaper alternative: use a small synonym expander.
**Defer:** only if 4.1–4.3 doesn't hit 0.55 target. Not in original plan — new idea.

**Phase 4 exit criteria:**
- Keyword coverage ≥0.52 (better) or 0.50 (acceptable with documented tradeoff)
- Tests green
- KB switch is reversible via config

---

## Phase 5 — Paper Writing (3–5 days)

**Goal:** Draft publishable capstone paper.

### 5.1 — Paper skeleton
**Sections:**
1. **Abstract** (150 words)
2. **Introduction** — RAG + healthcare + XAI gap
3. **Related Work** — RAG, medical QA, XAI for LLMs, confidence calibration
4. **System Architecture** — use `fig1_architecture.png`, walk through each stage
5. **XAI Methodology** — 5-signal confidence, Platt calibration, hallucination detection
6. **Evaluation**
   - 5.1 Setup (97-question test set, 3 sources, bootstrap CI)
   - 5.2 Model comparison (TinyLlama vs BioMistral vs QLoRA) → use `fig2`
   - 5.3 Ablation study → use `fig4`
   - 5.4 Calibration analysis → use `fig3`
   - 5.5 Latency profile → use `fig5`
   - 5.6 Baseline comparisons (no-RAG, dense-only, no-XAI)
7. **Discussion** — why small model beat big model, limitations of CPU deployment
8. **Limitations** — no human eval (if you don't collect it), single-language, CPU bottleneck
9. **Conclusion**

### 5.2 — Tables and figures ready?
Verify from current state:
- ✅ Fig 1: architecture (`evaluation/generate_architecture_diagram.py`)
- ✅ Fig 2: model comparison (`evaluation/figures/fig1_model_comparison.png`)
- ✅ Fig 3: latency (`fig3_latency.png`)
- ✅ Fig 4: ablation (`fig4_ablation.png` — but needs n=97 rerun)
- ✅ Fig 5: radar (`fig5_radar.png`)
- ✅ Fig 6: summary (`fig6_full_summary.png`)
- ❓ Calibration reliability diagram — have raw + calibrated PNGs already
- ❌ Baseline comparison table — data is in `metrics_baseline_*.json`, needs a render script

### 5.3 — Add missing baseline table
**New script:** `evaluation/generate_baseline_table.py`
Reads all 6 metrics JSONs, produces LaTeX table.

### 5.4 — Human evaluation collection (kick off during Phase 3 Colab wait)
**Why:** Reviewers will ask. The Phase 3 Colab sessions have ~2h of wall-clock idle time — that's the ideal window to distribute the eval template, so wall-clock cost is near zero.
**How:**
1. Pick 25 responses, stratified by confidence level (5 high / 10 medium / 10 low)
2. Fill `evaluation/human_eval_template.csv` with the selected responses
3. Send to 2-3 classmates while Colab runs, give them 15 minutes each
4. Once responses are back, run `evaluation/compute_human_eval.py` to get Cohen's kappa + correlations
**Acceptance:** `evaluation/results/human_eval.json` exists with aggregate scores and inter-annotator agreement. If responses don't come back in time, cite as a limitation — do not block the paper on it.

**Phase 5 exit criteria:** 8-page draft PDF with all figures inlined, ready for advisor review.

---

## Phase 6 — Reproducibility Polish (2 hours, post-paper)

### 6.1 — Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Pin Python 3.11 to avoid the spacy/pydantic v1 mess on 3.14.

### 6.2 — `requirements.lock`
`pip freeze > requirements.lock` from a working venv.

### 6.3 — README badge row
Test status, license, Python version, model size.

### 6.4 — `Makefile`
```makefile
.PHONY: test run eval
test:
	pytest tests/ -m "not slow"
run:
	python3 api/main.py
eval:
	python3 evaluation/run_paper_eval.py --mode metrics --n 97
```

### 6.5 — Verify `.env.example` is complete
Every environment variable used in `config/settings.py` should be documented.

---

## Phase 7 — Deferred / Post-Paper (time permitting)

These are good ideas but not blockers for the capstone submission. Address after the paper.

| # | Task | Why deferred |
|---|---|---|
| 7.1 | Pipeline consolidation (startup-time selection) | Risk of breaking API; ~30% memory win is nice-to-have, not blocking |
| 7.2 | Document-level safety filter with metadata flag at indexing time | Current runtime filter is sufficient for the paper |
| 7.3 | `/ask/batch` endpoint for eval speedup | Evaluation already works; nice optimization |
| 7.4 | Streaming responses in frontend UI | Works without streaming; cosmetic |
| 7.5 | RL pipeline with real feedback | Needs >100 user ratings; not feasible pre-submission |
| 7.6 | Multi-language support | Out of scope for capstone |
| 7.7 | Switch to Python 3.11 to unblock spacy | Known workaround in place |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Colab Pro A100 unavailable, forced to L4 for Session A | Low | Low | Accept ~2x rebuild time on L4; Session B (L4) still runs fine in parallel |
| Colab Pro session hits 24h ceiling | Very Low | Medium | Session A+B each <3h; background execution makes idle cost-free |
| BioMistral fix is a dead end | Medium | Medium | Time-box Phase 2 to 3 hours, then document and remove |
| Semantic chunker unit tests pass but KB v2 retrieval degrades vs v1 | Medium | High | Mandatory A/B eval in Session C before switching default; keep v1 on disk until v2 wins by ≥0.03 kw |
| KB rebuild (Phase 3 Session A) degrades metrics | Low | High | A/B test (Session C) against v1, don't switch without evidence |
| Retrieval latency fix breaks recall | Low | Medium | Snapshot before/after, compare on 20 held-out queries |
| Python 3.14 spacy crash blocks tests | Already mitigated | — | Try/except wrappers in place |
| Paper deadline slippage | Medium | Critical | Phase 5 can start in parallel with Phase 4 — use the plateaus |

---

## Test Gates (run before each commit)

```bash
source venv/bin/activate
pytest tests/ -q -m "not slow"        # must show 235 passed (or higher)
```

Before Phase 3 and Phase 4 commits, also run:
```bash
python3 evaluation/run_paper_eval.py --mode metrics --n 20  # sanity check
```

---

## Execution Order (critical path)

```
Phase 0 (1h)
   │
   ▼
Phase 1 (4h, local)
   │
   ▼
Phase 2 (3h, local)
   │
   ▼
Phase 4.1 (local — semantic chunker + tests, no rebuild yet)
   │
   ▼
Phase 3 Colab Pro (~2h wall, parallel Sessions A+B, then Session C)
   │   ├─ Session A (A100): QLoRA retrain + KB v2 rebuild
   │   ├─ Session B (L4):   ablation + recalibration + BioMistral + baselines + figures
   │   └─ Session C:        v1 vs v2 A/B eval
   │   (during wait: Phase 5.4 human eval distribution)
   ▼
Phase 4.3 (local — interpret A/B, switch KB if v2 wins)
   │
   ▼
Phase 5 (3-5d paper draft)
   │
   ▼
Phase 6 (2h reproducibility polish)
   │
   ▼
Phase 7 (deferred)
```

**Total active engineering time:** ~20 hours
**Total wall clock (including Colab + writing):** ~2 weeks

---

## Daily Stand-up Checklist

Every morning:
1. What did I finish yesterday?
2. What am I blocked on?
3. Are tests still green?
4. Did any metric move? (check `evaluation/results/_snapshots/` diff)
5. What's the next smallest testable chunk?

---

## Definition of Done (for the whole project)

- [ ] Paper draft submitted to advisor
- [ ] All 235+ tests passing
- [ ] Metrics meet or exceed targets (or gap is documented with reason)
- [ ] Reproducibility: `make test && make run && make eval` works from clean checkout
- [ ] `.env.example` is complete and accurate
- [ ] No stale or placeholder data in `evaluation/results/`
- [ ] BioMistral either works or is documented as not fit
- [ ] README explains the system in <200 words
- [ ] Paper references every figure/table/metric that exists
