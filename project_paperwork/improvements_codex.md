# Improvements and Bug Review

Date: 2026-03-10

## Scope

Reviewed first-party project code in `src/`, `api/`, `config/`, `frontend/`, `evaluation/`, `scripts/`, and `tests/`.

Excluded from detailed review:
- `node_modules/`
- binary assets and archives (`*.zip`, `*.pdf`, images, generated slide assets)
- generated content files that are not part of the runtime path

## Method

- Static review of the main runtime path, retrieval stack, XAI modules, API, frontend, evaluation scripts, and tests.
- Test run: `venv312/bin/python -m pytest tests -q`
- Result: `155 passed`
- Targeted runtime checks for component initialization and feature-flagged code paths.

## Executive View

The repository has a solid structure and good ambition: hybrid retrieval, three orchestration variants, safety modules, evaluation utilities, and user feedback logging are all present. The main issue is not lack of ideas. It is that the live runtime path, the alternative pipelines, and the surrounding tooling have drifted away from each other.

The project currently looks stronger in mocked tests than it does in real execution. Several important defects only appear when real components are instantiated or when rarely used features are enabled.

## Confirmed Bugs

### 1. Core runtime initialization is broken by missing `logger` imports

- `src/embeddings/embedding_models.py:42`
- `src/embeddings/embedding_models.py:50`
- `src/embeddings/embedding_models.py:52`
- `src/embeddings/vector_store.py:87`
- `src/embeddings/vector_store.py:88`

Both modules call `logger.*`, but `from loguru import logger` is inside the module docstring instead of being imported as code. I directly confirmed:

- `MedicalEmbedder(model_name="all-minilm")` raises `NameError: name 'logger' is not defined`

Impact:
- Shared component loading in `api/main.py` can fail before the API becomes usable.
- This defect is not caught by the current test suite because tests mostly use mocks.

### 2. MCP web fallback cannot work from the FastAPI request path

- `src/pipeline/qa_pipeline.py:253`
- `api/main.py:857`

The standard pipeline uses `asyncio.run(...)` inside `HealthcareQAPipeline.answer()`. That method is called from async FastAPI endpoints, so when the MCP fallback path is hit, it fails with:

- `RuntimeError: asyncio.run() cannot be called from a running event loop`

I reproduced this behavior directly.

Impact:
- The advertised MCP/web fallback is effectively unavailable in the main API path.
- Under failure it also produces coroutine-not-awaited warnings.

### 3. Successful MCP fallback still crashes downstream

- `src/pipeline/qa_pipeline.py:264`
- `src/pipeline/qa_pipeline.py:335`
- `src/pipeline/qa_pipeline.py:356`
- `src/pipeline/qa_pipeline.py:387`

After MCP succeeds, the code fabricates a dummy `Doc` object with only `source` and `content`. Later code assumes every document also has `score` and `metadata`.

I reproduced the resulting failure:

- `AttributeError: 'Doc' object has no attribute 'score'`

Impact:
- Even if MCP search succeeds outside the async-loop issue, response building still breaks.

### 4. Context compression is wired with the wrong interface

- `src/pipeline/qa_pipeline.py:293`
- `src/pipeline/context_compressor.py:47`

The pipeline calls:

- `compress(context=context, query=question)`

But `ContextCompressor.compress()` expects:

- `compress(documents, query)`

And it returns `CompressedContext`, not a plain string.

I reproduced:

- `TypeError: ContextCompressor.compress() got an unexpected keyword argument 'context'`

Impact:
- Turning on `enable_context_compression` will break the standard pipeline immediately.

### 5. Multi-turn chat is effectively disabled in the Streamlit UI

- `frontend/streamlit_app.py:1097`
- `frontend/streamlit_app.py:1110`
- `api/main.py:612`
- `api/main.py:725`
- `api/main.py:805`

The frontend never calls `/sessions` before asking a question. The API also does not auto-create a session when `session_id` is missing. The response only echoes back an existing `session_id`, so the UI never gets one during normal use.

Impact:
- The chat UI looks multi-turn, but follow-up retrieval context is usually never active.
- This is a functional product bug, not just a missing enhancement.

### 6. Request controls are silently ignored for LangChain and LangGraph modes

- `api/main.py:734`
- `api/main.py:737`
- `api/main.py:1020`
- `api/main.py:1051`

When `use_langchain=true` or `use_langgraph=true`, the API ignores:

- `num_sources`
- `include_explanation`
- effective model selection behavior

Those pipelines are created with their own fixed defaults and the request only sends `answer(effective_question)`.

Impact:
- Same request payload behaves differently depending on pipeline mode.
- Benchmark and UX comparisons across pipeline variants are not apples-to-apples.

### 7. `api/demo_server.py` is a dangerous operational footgun

- `api/demo_server.py:112`
- `api/demo_server.py:153`

The demo server bypasses retrieval entirely and answers from the LLM alone while still presenting itself as a healthcare QA API.

Impact:
- Easy to accidentally run the wrong server and deploy an ungrounded medical assistant.
- This directly conflicts with the project's stated RAG-and-explainability reliability goal.

## High-Priority Improvements

### 1. Unify all three pipeline variants around one policy layer

- `config/settings.py:91`
- `src/pipeline/qa_pipeline.py:124`
- `src/langchain/langchain_pipeline.py:105`
- `src/langgraph/langgraph_nodes.py:18`

Right now Standard, LangChain, and LangGraph use different thresholds, different answerability logic, different prompt behavior, and different feature support.

Needed improvement:
- one shared config source for thresholds and feature flags
- one shared answerability policy
- one shared source formatting/confidence policy
- one shared request contract across all variants

Why it matters:
- medical QA systems need consistent refusal behavior, not three separate definitions of "answerable"

### 2. Stop injecting conversation context into the literal question string

- `api/main.py:727`
- `api/main.py:730`
- `src/pipeline/qa_pipeline.py:156`
- `src/pipeline/qa_pipeline.py:196`

The API currently rewrites the user question into:

- `Previous context: ... Current question: ...`

and passes that as the actual question text. The standard pipeline already has a `conversation_context` parameter but the API does not use it.

Needed improvement:
- keep the user question clean
- pass follow-up context separately for retrieval and prompt construction
- log original and augmented forms separately

Why it matters:
- cleaner prompts
- more accurate evaluation
- easier attribution and better query rewriting behavior

### 3. Fix the retrieval stack drift between docs, config, and runtime

- `README.md`
- `config/settings.py:43`
- `api/main.py:241`
- `src/embeddings/embedding_models.py:29`

The project narrative says MedCPT/BioMistral and specialized medical retrieval. The live runtime currently defaults to `all-minilm` and TinyLlama-only behavior.

Needed improvement:
- decide what the real supported production path is
- align README/config/runtime
- if MedCPT is the target, actually use dual query/article encoders in live retrieval

Why it matters:
- this is a project credibility issue for reviews, demos, and evaluation claims

### 4. Rework ingestion and chunking for scale and retrieval quality

- `scripts/build_knowledge_base.py:49`
- `scripts/build_knowledge_base.py:64`
- `src/data_pipeline/loaders/dataset_loader.py:279`
- `src/data_pipeline/preprocessors/chunker.py:40`

Current problems:
- "streaming" build still accumulates `all_chunks` in memory
- documents are indexed as concatenated `Question + Answer`, which can bias retrieval toward question-word echoing instead of evidence quality
- chunking is character/regex based with naive overlap handling

Needed improvement:
- true batch ingestion straight into the vector store
- document schema with passage-level chunks and richer metadata
- dataset provenance fields, chunk hashes, and deterministic re-ingestion strategy

Why it matters:
- this repo will struggle on larger corpora
- retrieval quality and explainability both depend on better chunk boundaries

### 5. Replace score heuristics with calibrated evidence handling

- `src/retrieval/hybrid_retriever.py:276`
- `src/xai/confidence_scorer.py:57`
- `frontend/streamlit_app.py:870`

RRF scores are treated like calibrated probabilities in some places, then shown as "Match: XX%". That is misleading.

Needed improvement:
- keep rank-fusion scores internal
- expose either normalized retrieval ranks or calibrated support probabilities
- calibrate confidence against held-out evaluation data

Why it matters:
- in a medical assistant, score semantics must be defensible

### 6. Wire the unused safety/XAI modules into the actual product or remove them from the claim surface

- `src/xai/hallucination_detector.py`
- `src/xai/multi_signal_confidence.py`
- `src/xai/passage_highlighter.py`
- `src/safety/guardrails.py:465`
- `api/main.py`

Several advanced modules exist but are not part of the live answer path. The implementation story is ahead of the product story.

Needed improvement:
- either integrate these modules into the API/frontend
- or explicitly mark them as experimental and keep them out of user-facing claims

Why it matters:
- it reduces architectural confusion
- it improves trust during demos and reviews

## Medium-Priority Improvements

### 1. Improve sparse retrieval quality and scalability

- `src/retrieval/hybrid_retriever.py:133`
- `src/retrieval/hybrid_retriever.py:218`

Issues:
- BM25 is built by loading the entire vector corpus into RAM
- tokenization is naive whitespace splitting
- no stemming, no normalization, no domain-aware sparse preprocessing

Recommended direction:
- persisted sparse index
- stronger normalization/tokenization
- field-aware indexing if source schemas differ

### 2. Make generation deterministic by default for medical QA

- `src/generation/llm_wrapper.py:204`

The live generation path samples by default (`do_sample=True`) even for strict medical QA prompts.

Recommended direction:
- greedy or near-deterministic decoding by default
- sampling only in experiments

### 3. Reduce broad exception swallowing

- `api/main.py`
- `frontend/streamlit_app.py`
- `src/utils/cache_manager.py`
- `scripts/build_knowledge_base.py`

There are many `except Exception: pass` or warning-only fallbacks in critical paths.

Recommended direction:
- fail closed for safety-critical features
- log stack traces with structured context
- distinguish degraded mode from healthy mode in API responses

### 4. Remove `sys.path.insert(...)` import hacks

- `api/main.py:20`
- `src/pipeline/qa_pipeline.py:16`
- many files under `src/`, `scripts/`, and `tests/`

Recommended direction:
- package the project properly
- use module execution (`python -m ...`) and clean imports

Why it matters:
- easier deployment
- fewer environment-specific bugs

### 5. Fix CORS/security posture

- `api/main.py:52`
- `api/demo_server.py:24`

`allow_origins=["*"]` with `allow_credentials=True` is a bad default for a medical app.

Recommended direction:
- explicit allowed origins by environment
- no wildcard in anything user-facing

## Evaluation and Testing Gaps

### 1. Current tests are strong on logic, weak on real startup behavior

The suite passing `155/155` is useful, but it did not catch:

- broken embedder initialization
- broken context compressor wiring
- broken MCP fallback
- dead session flow in the frontend/API pairing

Needed additions:
- smoke test that instantiates the real embedder
- smoke test that initializes the vector store with a temporary local path
- API tests for `/ask`, `/sessions`, `/feedback`, `/health`
- feature-flag tests for MCP fallback and context compression
- end-to-end UI/session flow tests or at least backend contract tests for session creation

### 2. Evaluation tooling is too coupled to heavy runtime initialization

- `evaluation/run_evaluation.py:519`

Even retrieval-only mode initializes the LLM and full pipeline.

Needed improvement:
- retrieval evaluation should run without generation model startup
- end-to-end evaluation should support mock judge vs real judge explicitly

### 3. There is no evidence-gated regression suite for safety-critical behavior

Needed regression set:
- emergency redirection
- unanswerable refusal
- grounded answer with citations
- alternative pipeline parity
- follow-up question handling
- cache invalidation after KB updates

## Product/Architecture Drift

### 1. Docs and runtime disagree on supported backends

- `README.md`
- `config/settings.py:20`
- `src/generation/llm_wrapper.py:47`

The repo mentions BioMistral, LM Studio, AirLLM, and MedCPT, but the live code path currently exposes TinyLlama-only behavior in the main API.

### 2. Explainability is broader in code than in the exposed product

The repo contains source attribution, rationale generation, passage highlighting, hallucination detection, factual consistency, and multi-signal confidence. The live UI/API only expose part of that story.

### 3. The demo mode undermines the thesis if not clearly fenced off

If the project claim is "explainable healthcare QA chatbot with RAG and XAI", then any non-RAG demo mode must be unmistakably labeled as unsafe-for-evaluation/demo-only.

## Recommended Order of Work

1. Fix runtime blockers first: missing `logger` imports, MCP fallback path, context compressor contract.
2. Make sessions real: auto-create or explicitly create `session_id`, then use `conversation_context` instead of rewriting the question text.
3. Unify Standard/LangChain/LangGraph request handling and thresholds.
4. Decide the real supported model/retrieval stack and align docs, config, and runtime.
5. Rework ingestion/indexing so the KB builder is truly batch-based and chunk quality improves.
6. Replace heuristic confidence presentation with calibrated support semantics.
7. Add real-component smoke tests and API contract tests so these regressions are caught automatically.

## Bottom Line

This is a good project foundation with a strong architecture outline, but it is not yet internally consistent enough for a clinical-quality RAG demo. The biggest risk is false confidence: the repository looks feature-rich, the tests are green, but several important runtime and safety behaviors are currently either broken, disconnected, or inconsistent across pipeline variants.
