# Healthcare QA Project Completion Plan (Localhost End-to-End)

This document is strictly for finishing and validating the product end-to-end on localhost.

---

## 1) Verified Completed Items

The checked items below were re-verified in code:

- Retrieval source diversity + fallback + context scaling
  - `src/retrieval/hybrid_retriever.py` has `_apply_source_diversity`, `_apply_post_retrieval_filter(... target_k=k)`, and dynamic `max_context_length`.
- Source transparency in API
  - `api/main.py` includes `num_sources_requested`, `num_sources_effective`, `num_sources_returned` in response model and response payload.
- Causation hardening
  - `src/pipeline/qa_pipeline.py` has cache schema `qa-v6`, `_apply_plan_postprocessors`, `_ground_causation_answer`.
- Startup controls
  - `api/main.py` honors `SKIP_STARTUP_WARMUP`, `SKIP_NLI_WARMUP`.
- API hardening
  - production API key guard, `/clear-cache` auth dependency, safer CORS behavior.
- Streamlit transparency + retry
  - `frontend/streamlit_app.py` shows source count details, diagnostics (`stage_latencies`), and retry action.
- Evaluation runners and manifest
  - `evaluation/run_retrieval_benchmark.py`, `evaluation/run_e2e_benchmark.py`, `evaluation/manifest.py`, `evaluation/data/gold_retrieval.jsonl`, `scripts/run_eval_bundle.sh`.
- Log redaction + safety tests
  - `src/utils/log_redaction.py`, redaction wired into feedback/trajectory logging, `tests/test_safety_adversarial.py`.
- CI + release checks
  - `Makefile` (`test`, `eval-retrieval`, `release-check`) and `.github/workflows/ci.yml`.

Validation snapshot:

- `pytest tests/ -k "not langsmith"` passes locally (with selenium e2e auto-skipped when dependency missing).

---

## 2) Remaining Work Plan (Step-by-Step)

## Phase R1 - True Streaming (Highest Priority)

Goal: replace simulated chunk streaming with real token streaming path where supported.

### Tasks

- Add streaming method in generation layer:
  - update `src/generation/llm_wrapper.py` with `generate_stream(...)` iterator.
- Update API stream endpoint:
  - in `api/main.py` `/ask/stream`, prefer model token stream and fallback to chunk stream only if unavailable.
- Keep response contract stable:
  - `meta`, `token`, `done`, `error` events remain backward-compatible.
- Add tests:
  - new API contract tests for `/ask/stream` event order and completion.

### Implementation steps

1. Add interface:
  - `MedicalLLM.generate_stream(prompt, max_new_tokens, **kwargs)` yields strings/tokens.
2. In `_prepare_and_execute_pipeline`, return both final answer and optional stream generator.
3. In `/ask/stream`, emit `meta` first, then streamed tokens directly.
4. Preserve current fallback for models that cannot stream.

### Verify

- `curl -N -X POST http://127.0.0.1:8000/ask/stream ...` shows incremental tokens before completion.
- API contract tests pass.

---

## Phase R2 - Cache Policy Review + Behavior Tests

Goal: make cache behavior deterministic and explicit.

### Tasks

- Define cache policy by endpoint/query type:
  - TTL for standard ask
  - bypass for sensitive/emergency flags
  - context-key versioning policy
- Add tests for stale/refresh behavior.
- Add API diagnostics for cache hit/miss reason.

### Implementation steps

1. Update `src/utils/cache_manager.py` and `src/pipeline/qa_pipeline.py`.
2. Add structured cache metadata in trajectory logs.
3. Add tests in `tests/test_integration.py` for:
  - cache hit on repeated prompt
  - miss when `num_sources` differs
  - miss when schema/version changes

### Verify

- `pytest tests/test_integration.py -k cache -q`
- Manual `/ask` repeated requests show expected `from_cache` behavior.

---

## Phase R3 - Docker Parity with Local Runtime

Goal: local shell run and docker run produce same API behavior and flags.

### Tasks

- Confirm Docker image uses same env defaults as local.
- Ensure mounted KB path and collection name match local config.
- Add compose profile for fast dev startup (`SKIP_STARTUP_WARMUP=true`).
- Add smoke check script for dockerized API endpoints.

### Implementation steps

1. Update `docker/Dockerfile` and `docker/docker-compose.yml`.
2. Add a script `scripts/smoke_docker_api.sh`:
  - check `/health`
  - one `/ask`
  - verify source count fields exist.
3. Document command flow in `docs/REPRODUCTION.md`.

### Verify

- `docker compose up -d`
- smoke script exits 0.
- `/ask` schema matches local run.

---

## Phase R4 - Remove Remaining `sys.path.insert` Bootstrapping

Goal: clean import hygiene for maintainability and predictable runtime.

### Tasks

- Remove `sys.path.insert(...)` usage in:
  - `api/main.py`
  - `src/pipeline/qa_pipeline.py`
  - scripts using path injection.
- Keep imports working via package install and `PYTHONPATH`.
- Run full test suite after each removal batch.

### Implementation steps

1. Ensure package install works:
  - `pip install -e .`
2. Replace path hacks with absolute package imports.
3. Fix any import cycles discovered during cleanup.

### Verify

- `python -m api.main` starts without path hacks.
- `pytest tests/ -k "not langsmith"` passes.

---

## Phase R5 - Scoped API Security + Basic Ops

Goal: tighten production controls beyond single API key.

### Tasks

- Add optional key scopes:
  - read (`/ask`, `/health`)
  - admin (`/clear-cache`, future admin APIs)
- Add request audit metadata (route, key id hash).
- Add basic rate-limit docs for operational tuning.

### Implementation steps

1. Extend `verify_api_key` logic in `api/main.py`.
2. Add env format for scoped keys (simple JSON or `KEY:scope` pairs).
3. Add tests for forbidden admin access.

### Verify

- unauthorized admin request returns 401/403.
- authorized read still works for ask flow.

---

## Phase R6 - Dashboard + Profiling

Goal: expose operational quality per query type.

### Tasks

- Add per-query-type metrics report from evaluation results.
- Add simple Streamlit metrics panel for:
  - latency distribution
  - source counts requested vs returned
  - risk-class retrieval hit.
- Add a profiling script for hot-path latency contributors.

### Implementation steps

1. Extend `evaluation/run_retrieval_benchmark.py` output to include per-type detail.
2. Create `scripts/profile_pipeline.py` for timing summaries.
3. Add a frontend diagnostics panel toggle.

### Verify

- metrics JSON contains per-type breakdown.
- diagnostics panel renders with no API errors.

---

## 3) Execution Order for You

Follow this strict order:

1. R1 (streaming)
2. R2 (cache)
3. R3 (docker parity)
4. R4 (path cleanup)
5. R5 (scoped security)
6. R6 (dashboard/profiling)

Do not start the next phase until current phase verify checks pass.

---

## 4) Commands to Run After Each Phase

```bash
export PYTHONPATH=$(pwd)
pytest tests/ -q -k "not langsmith"
make eval-retrieval
```

For API behavior:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"question":"What causes acute migraine?","num_sources":8}'
```

For streaming once R1 is done:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream -H 'Content-Type: application/json' -d '{"question":"What is hypertension?","num_sources":3}'
```

---

## 5) Current Open Checklist

- True token streaming path (R1)
- Cache policy + tests (R2)
- Docker parity + smoke script (R3)
- Remove remaining `sys.path.insert` (R4)
- Scoped API key roles (R5)
- Query-type dashboard + profiling (R6)

