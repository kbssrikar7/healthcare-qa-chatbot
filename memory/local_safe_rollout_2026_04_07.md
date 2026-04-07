---
name: Local-safe rollout handoff (2026-04-07)
description: Summary of laptop-safe implementation changes, validation, and remaining Colab-owned work
type: project
---

Local-safe implementation pass completed on 2026-04-07. This pass explicitly avoided changing Colab execution assets and GPU-heavy workflows.

**What changed:**
- Fixed the `MedicalTextChunker` regression in `src/data_pipeline/preprocessors/chunker.py` by restoring a stable `self.chunk_size` default and making `chunk_text()` safe for direct calls.
- Replaced eager package exports with lazy imports in `src/langchain/__init__.py` and `src/langgraph/__init__.py` so optional integration packages do not fail prematurely.
- Removed the Pydantic/spaCy schema monkeypatch from `api/main.py`.
- Added adapter-aware pipeline loading in `api/main.py` so standard/LangChain/LangGraph loaders can be keyed by model plus adapter path.
- Added model-aware prompt selection:
  - TinyLlama keeps chat-template prompts.
  - BioMistral gets native Mistral `[INST]` formatting via `src/generation/prompt_manager.py`.
  - GGUF generation in `src/generation/llm_wrapper.py` no longer double-wraps prompts that already contain `[INST]`.
- Formalized retrieval telemetry in `src/retrieval/hybrid_retriever.py` with these timing keys:
  - `embedding_ms`
  - `chroma_query_ms`
  - `bm25_query_ms`
  - `rrf_ms`
  - `rerank_ms`
  - `safety_filter_ms`
  - `total_ms`
- Preserved and hardened embedding metadata sidecar + legacy sentinel compatibility in `src/embeddings/vector_store.py`.
- Standardized local evaluation logic across:
  - `evaluation/run_paper_eval.py`
  - `evaluation/run_api_eval.py`
  - `evaluation/run_ablation.py`
  - `evaluation/compute_calibration.py`
  - `evaluation/generate_paper_figures.py`
  - `evaluation/compute_human_eval.py`
- Added focused rollout regression tests in `tests/test_local_rollout.py`.

**Validation completed:**
- Ran full suite with repo venv:
  - `./venv/bin/pytest tests/ -q`
  - Result: `240 passed`
- Ran targeted local-safe regression pack:
  - `./venv/bin/pytest tests/test_local_rollout.py tests/test_retrieval.py tests/test_langchain_pipeline.py tests/test_langgraph_pipeline.py tests/test_integration.py -q`
  - Result: passed
- Ran syntax sanity check:
  - `python3 -m compileall evaluation api src tests`

**Important files intentionally left untouched:**
- `evaluation/results/calibration.json`
- `COLAB_EXECUTION_GUIDE.md`

These were treated as unrelated workspace changes and should not be reverted or silently folded into other commits.

**What remains for future agents:**
- Colab/GPU execution is still owned by the user and was deliberately excluded here.
- If future work needs full evaluation runs or retraining, prefer Colab T4/A100 instead of local CPU.
- If BioMistral still underperforms in Colab evaluation, inspect prompt-format effects first before changing decoding.
- Optional cleanup later:
  - migrate FastAPI startup handlers from `@app.on_event("startup")` to lifespan
  - reduce Python 3.14 deprecation noise from third-party dependencies

**How to reason about the current state:**
- The laptop-safe codebase is now in a green-test state.
- Retrieval/evaluation plumbing is more explicit and easier to instrument.
- Colab assets were not edited in this pass, so future GPU-side work can proceed without local-path churn.
