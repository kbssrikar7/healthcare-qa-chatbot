"""
FastAPI application for Healthcare QA Chatbot.

Enhanced with:
- Single model: TinyLlama 1.1B
- Safety guardrails integration (emergency detection, content filtering)
- Conversation history with follow-up detection
- Streaming responses
- Enhanced health-check with system metrics
- Passage highlighting for XAI
"""
import asyncio
import os
from loguru import logger
import sys
import time
import json
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use cached models without network round-trips (avoids 40s retry delays when offline)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import AVAILABLE_MODELS

# ── Rate limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── API key auth ─────────────────────────────────────────────────────────────

_API_KEY = os.getenv("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    """Verify API key. If API_KEY env var is unset, auth is disabled (dev mode)."""
    if not _API_KEY:
        return  # No key configured — open access (dev)
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── App setup ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup warmup and shutdown cleanup."""
    logger.info("Startup: pre-loading shared components...")
    shared = _get_shared_components()

    # Verify embedding model compatibility between index and query model
    if shared and "vector_store" in shared and "embedder" in shared:
        vs = shared["vector_store"]
        embedder = shared["embedder"]
        if hasattr(vs, "verify_embedding_compatibility") and hasattr(embedder, "model_name"):
            dim = getattr(embedder, "dimension", 384)
            compatible = vs.verify_embedding_compatibility(embedder.model_name, dim)
            if compatible:
                logger.info(f"Startup: embedding compatibility verified ({embedder.model_name})")
            else:
                logger.warning("Startup: embedding mismatch detected — retrieval may be unreliable")

    # Pre-load NLI model for hallucination detection (avoids 30s delay on first NLI check)
    try:
        from src.xai.hallucination_detector import HallucinationDetector
        _detector = HallucinationDetector(use_nli=True)
        _detector.warm_up()
        # Store for reuse by pipelines
        app.state.hallucination_detector = _detector
        logger.info("Startup: NLI hallucination detector ready")
    except Exception as e:
        logger.warning(f"Startup: NLI model pre-load skipped: {e}")

    # Pre-load default pipeline
    _init_guardrails()
    _init_conversation_manager()
    _init_feedback_system()

    # ── Pre-initialize BM25 index ─────────────────────────────────────────────
    # BM25 lazy-loads 182K docs from ChromaDB on first hybrid query (20-60s hang).
    # Pre-initialize it here during startup so the first user query is fast.
    try:
        pipeline = get_pipeline(model_choice="tinyllama")
        if pipeline and hasattr(pipeline, "retriever") and hasattr(pipeline.retriever, "warm_up"):
            logger.info("Startup: pre-initializing BM25 index...")
            _t_bm25 = __import__("time").perf_counter()
            pipeline.retriever.warm_up()
            _bm25_ms = (__import__("time").perf_counter() - _t_bm25) * 1000
            logger.info(f"Startup: BM25 initialization complete ({_bm25_ms:.0f} ms)")
    except Exception as e:
        logger.warning(f"Startup: BM25 warmup failed (non-fatal): {e}")

    # ── Warmup query: run a dummy question through the full pipeline ──────────
    # This ensures the tokenizer, generation model, NLI model, and embedding
    # model are all loaded and warmed up before the first real user request.
    # Cold-start latency is typically 30–120s; this pays that cost at startup.
    try:
        pipeline = get_pipeline(model_choice="tinyllama")
        if pipeline:
            logger.info("Startup: running warmup query through default pipeline...")
            _t_warmup = __import__("time").perf_counter()
            _ = pipeline.answer(
                "What is hypertension?",
                num_documents=3,
                include_explanation=False,  # skip XAI to keep warmup fast
            )
            _warmup_ms = (__import__("time").perf_counter() - _t_warmup) * 1000
            logger.info(f"Startup: warmup query complete ({_warmup_ms:.0f} ms)")
    except Exception as e:
        logger.warning(f"Startup: warmup query failed (non-fatal): {e}")

    logger.info("Startup: all components ready")

    yield  # Application runs here

    # Shutdown: save conversation sessions
    try:
        from src.pipeline.conversation_manager import ConversationManager
        cm = getattr(app.state, "conversation_manager", None)
        if cm:
            cm.save_sessions()
            logger.info("Shutdown: conversation sessions saved")
    except Exception:
        pass


app = FastAPI(
    title="Healthcare QA Chatbot API",
    description="An explainable medical question-answering system combining LLM + RAG + XAI",
    version="2.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Service health and readiness checks"},
        {"name": "QA", "description": "Ask medical questions and get explainable answers"},
        {"name": "Models", "description": "Available LLM model information"},
        {"name": "Sessions", "description": "Conversation session management"},
        {"name": "Feedback", "description": "User feedback and rating collection"},
    ],
)

# Record startup time
app.state.start_time = time.time()
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded — please slow down."})

# CORS
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allow_origins = ["*"] if cors_origins_env == "*" else [o.strip() for o in cors_origins_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error response model ─────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Structured error response returned by all error handlers."""
    detail: str
    error_code: str = "INTERNAL_ERROR"
    request_id: Optional[str] = None


# ── Correlation ID middleware ────────────────────────────────────────────────

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Attach a unique request_id to every request for traceability."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return structured JSON."""
    request_id = getattr(request.state, "request_id", None)
    logger.error("Unhandled exception [{}]: {}", request_id, exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="An internal error occurred. Please try again.",
            error_code="INTERNAL_ERROR",
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return structured JSON for HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}",
            request_id=request_id,
        ).model_dump(),
    )

# ── Global state ─────────────────────────────────────────────────────────────

# Pipeline instances keyed by model name
pipelines: Dict[str, Any] = {}

# Shared components (loaded once)
_shared_components: Dict[str, Any] = {}

# Safety guardrails
guardrails = None
emergency_detector = None
drug_checker = None

# Conversation manager
conversation_manager = None

# Feedback and RL trajectory logging
feedback_collector = None
trajectory_logger = None


# ── Request / Response models ────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    include_explanation: bool = True
    num_sources: int = Field(default=3, ge=1, le=10)
    model_choice: Optional[str] = Field(
        default=None,
        description="Model key from /models (e.g. tinyllama)",
    )
    use_langchain: bool = Field(default=False, description="Use LangChain LCEL pipeline")
    use_langgraph: bool = Field(default=False, description="Use LangGraph self-correcting RAG")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID for follow-ups")


class SourceInfo(BaseModel):
    source: str
    content: str
    score: float
    url: Optional[str] = ""
    highlight_spans: Optional[List[Dict]] = None


class ConfidenceInfo(BaseModel):
    score: float
    level: str
    explanation: str


class AttributionInfo(BaseModel):
    claim: str
    source: str
    evidence: str
    similarity: float


class SafetyInfo(BaseModel):
    level: str = "safe"
    flags: List[str] = Field(default_factory=list)
    is_emergency: bool = False
    emergency_message: Optional[str] = None
    drug_warnings: Optional[List[str]] = None


class AnswerResponse(BaseModel):
    response_id: str
    question: str
    answer: str
    sources: List[SourceInfo]
    confidence: ConfidenceInfo
    attributions: List[AttributionInfo]
    disclaimer: str
    rationale: Optional[str] = None
    model_used: Optional[str] = None
    pipeline_used: Optional[str] = None
    session_id: Optional[str] = None
    safety: Optional[SafetyInfo] = None
    latency_ms: Optional[float] = None
    confidence_breakdown: Optional[Dict[str, Any]] = None
    hallucination: Optional[Dict[str, Any]] = None
    stage_latencies: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    message: str
    uptime_seconds: Optional[float] = None
    models_loaded: Optional[List[str]] = None
    knowledge_base_docs: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cache_stats: Optional[Dict] = None


class FeedbackRequest(BaseModel):
    response_id: str = Field(..., min_length=1)
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    was_helpful: Optional[bool] = None
    was_accurate: Optional[bool] = None
    was_safe: Optional[bool] = None
    feedback_text: Optional[str] = Field(default=None, max_length=2000)
    session_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    response_id: str
    reward_signal: float
    trajectory_found: bool


# ── Component loaders ────────────────────────────────────────────────────────

def _get_shared_components():
    """Load shared components (embedder, vector store, retriever) once."""
    global _shared_components
    if _shared_components:
        return _shared_components

    try:
        from src.embeddings.embedding_models import MedicalEmbedder
        from src.embeddings.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.generation.prompt_manager import MedicalPromptManager
        from src.xai.confidence_scorer import ConfidenceScorer
        from src.xai.source_attribution import SourceAttributor
        from config.settings import config
        pipeline_config = getattr(config, 'pipeline', None) if config else None

        logger.info(" Loading shared pipeline components...")
        embedder = MedicalEmbedder(model_name="all-minilm")
        vector_store = VectorStore(
            collection_name=config.retrieval.collection_name,
            persist_directory=config.retrieval.persist_directory,
        )
        
        # Reranker
        reranker = None
        if pipeline_config and getattr(pipeline_config, 'enable_reranker', False):
            try:
                from src.retrieval.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
                logger.info(" Reranker initialized")
            except Exception as e:
                logger.warning(f" Failed to init reranker: {e}")

        retrieval_config = getattr(config, "retrieval", None)
        retriever = HybridRetriever(
            embedder,
            vector_store,
            reranker=reranker,
            min_score=float(getattr(retrieval_config, "min_retrieval_score", 0.0)),
            context_window_sentences=int(
                getattr(retrieval_config, "context_window_sentences", 0)
            ),
        )
        prompt_manager = MedicalPromptManager()
        confidence_scorer = ConfidenceScorer()
        source_attributor = SourceAttributor(embedder=embedder, similarity_threshold=0.3)
        
        # Other quality hooks
        query_enhancer = None
        context_compressor = None
        corrective_rag = None
        factual_checker = None
        
        if pipeline_config:
            if getattr(pipeline_config, 'enable_query_enhancement', False):
                try:
                    from src.retrieval.query_enhancer import QueryEnhancer
                    query_enhancer = QueryEnhancer()
                    logger.info(" Query enhancer initialized")
                except Exception as e:
                    logger.warning(f" Failed to init query enhancer: {e}")
                    
            if getattr(pipeline_config, 'enable_context_compression', False):
                try:
                    from src.pipeline.context_compressor import ContextCompressor
                    context_compressor = ContextCompressor()
                    logger.info(" Context compressor initialized")
                except Exception as e:
                    logger.warning(f" Failed to init context compressor: {e}")
                    
            if getattr(pipeline_config, 'enable_corrective_rag', False):
                try:
                    from src.retrieval.corrective_rag import CorrectiveRAG
                    corrective_rag = CorrectiveRAG(retriever=retriever)
                    logger.info(" Corrective RAG initialized")
                except Exception as e:
                    logger.warning(f" Failed to init corrective RAG: {e}")
                    
            if getattr(pipeline_config, 'enable_factual_consistency', False):
                try:
                    from src.xai.factual_consistency import FactualConsistencyChecker
                    factual_checker = FactualConsistencyChecker()
                    logger.info(" Factual consistency checker initialized")
                except Exception as e:
                    logger.warning(f" Failed to init factual checker: {e}")

        _shared_components = {
            "embedder": embedder,
            "vector_store": vector_store,
            "retriever": retriever,
            "prompt_manager": prompt_manager,
            "confidence_scorer": confidence_scorer,
            "source_attributor": source_attributor,
            "query_enhancer": query_enhancer,
            "context_compressor": context_compressor,
            "corrective_rag": corrective_rag,
            "factual_consistency_checker": factual_checker,
        }
        logger.info(" Shared components loaded")
    except Exception as e:
        import traceback
        logger.error(f" Failed to load shared components:\n{traceback.format_exc()}")

    return _shared_components


def _resolve_adapter_path(adapter_path: Optional[str] = None) -> Optional[str]:
    """Resolve adapter path with a sensible local default when present."""
    if adapter_path:
        candidate = Path(adapter_path)
    else:
        candidate = Path("models/fine_tuned/medical_adapter")
    return str(candidate) if candidate.exists() else None


def get_pipeline(model_choice: str = "tinyllama", adapter_path: Optional[str] = None):
    """Get or create a pipeline for the given model choice."""
    global pipelines

    resolved_adapter_path = _resolve_adapter_path(adapter_path)
    cache_key = f"{model_choice}::{resolved_adapter_path or 'base'}"

    if cache_key in pipelines:
        return pipelines[cache_key]

    import torch
    import gc

    # FREE PREVIOUS PIPELINES TO PREVENT MEMORY LEAKS ON LAPTOPS
    for k in list(pipelines.keys()):
        logger.info(f" Unloading previous model: {k} to free memory...")
        try:
            if hasattr(pipelines[k], "llm") and hasattr(pipelines[k].llm, "cleanup"):
                pipelines[k].llm.cleanup()
        except Exception as e:
            logger.warning(f" Error during cleanup of {k}: {e}")
        del pipelines[k]
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── GPU safety guard ─────────────────────────────────────────────────
    model_info = AVAILABLE_MODELS.get(model_choice)
    if not model_info:
        logger.error(f" Unknown model: {model_choice}")
        return None

    if model_info.get("requires_gpu", False):
        import torch
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model_info['display_name']}' requires a GPU but none is available. "
                    f"Please use 'tinyllama' instead, or run on Google Colab with a T4 GPU."
                ),
            )

    shared = _get_shared_components()
    if not shared:
        return None

    try:
        from src.generation.llm_wrapper import MedicalLLM
        from src.pipeline.qa_pipeline import HealthcareQAPipeline

        logger.info(f" Loading LLM: {model_info['display_name']}...")

        # Ensure local writable HF cache path (avoids ~/.cache permission issues on some systems)
        local_hf_cache = str((Path(__file__).parent.parent / ".hf_cache").resolve())
        os.environ.setdefault("HF_HOME", local_hf_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", local_hf_cache)
        Path(local_hf_cache).mkdir(parents=True, exist_ok=True)

        backend = model_info.get("backend", "transformers")
        resolved_model_name = model_info.get("model_name") or model_choice

        # For GGUF models pass the shorthand key — MedicalLLM._resolve_model_path handles it
        llm_model_arg = model_choice if backend == "gguf" else resolved_model_name

        # Check for fine-tuned adapter (transformers only)
        llm_kwargs = {
            "model_name": llm_model_arg,
            "load_in_4bit": model_info.get("load_in_4bit", False),
            "hf_token": os.getenv("HUGGINGFACE_TOKEN"),
        }

        if backend == "transformers" and resolved_adapter_path:
            logger.info(f" Found fine-tuned adapter at {resolved_adapter_path}")
            llm_kwargs["adapter_path"] = resolved_adapter_path
            llm = MedicalLLM(**llm_kwargs)
        else:
            if backend == "transformers":
                logger.warning(" No adapter found, using base model")
            llm = MedicalLLM(**llm_kwargs)

        # KB fingerprint for cache
        try:
            kb_count = shared["vector_store"].collection.count()
        except Exception:
            kb_count = 0
        adapter_tag = Path(resolved_adapter_path).name if resolved_adapter_path else "base"
        cache_context_key = f"{resolved_model_name}_{backend}_{adapter_tag}_kb{kb_count}"

        pipeline = HealthcareQAPipeline(
            retriever=shared["retriever"],
            llm=llm,
            prompt_manager=shared["prompt_manager"],
            confidence_scorer=shared["confidence_scorer"],
            source_attributor=shared["source_attributor"],
            query_enhancer=shared.get("query_enhancer"),
            context_compressor=shared.get("context_compressor"),
            corrective_rag=shared.get("corrective_rag"),
            factual_consistency_checker=shared.get("factual_consistency_checker"),
            cache_context_key=cache_context_key,
        )
        pipelines[cache_key] = pipeline
        logger.info(f" Pipeline loaded for {model_info['display_name']}")
        return pipeline

    except Exception as e:
        logger.error(f" Failed to load pipeline for {model_choice}: {e}")
        return None



def _init_guardrails():
    """Initialize safety guardrails."""
    global guardrails, emergency_detector, drug_checker
    if guardrails is not None:
        return
    try:
        from src.safety.guardrails import MedicalGuardrails, EmergencyDetector, DrugInteractionChecker
        guardrails = MedicalGuardrails()
        emergency_detector = EmergencyDetector()
        drug_checker = DrugInteractionChecker()
        logger.info(" Safety guardrails initialized")
    except Exception as e:
        logger.warning(f" Failed to load guardrails: {e}")


def _init_conversation_manager():
    """Initialize conversation manager."""
    global conversation_manager
    if conversation_manager is not None:
        return
    try:
        from config.settings import config as _cfg
        from src.conversation.history import ConversationManager
        api_cfg = getattr(_cfg, "api", None)
        conversation_manager = ConversationManager(
            storage_path=getattr(
                api_cfg, "conversation_storage_path", "data/conversations/sessions.json"
            ),
            max_session_age_hours=getattr(api_cfg, "conversation_max_age_hours", 24),
            cleanup_interval_minutes=getattr(
                api_cfg, "conversation_cleanup_interval_minutes", 15
            ),
        )
        logger.info(" Conversation manager initialized")
    except Exception as e:
        logger.warning(f" Failed to load conversation manager: {e}")


def _init_feedback_system():
    """Initialize feedback collector and trajectory logger."""
    global feedback_collector, trajectory_logger
    if feedback_collector is not None and trajectory_logger is not None:
        return
    try:
        from src.feedback.collector import FeedbackCollector
        from src.feedback.trajectory_logger import TrajectoryLogger

        if feedback_collector is None:
            feedback_collector = FeedbackCollector(
                storage_path="data/feedback/user_feedback.jsonl"
            )
        if trajectory_logger is None:
            trajectory_logger = TrajectoryLogger(
                storage_path="data/feedback/response_trajectories.jsonl"
            )
        logger.info(" Feedback system initialized")
    except Exception as e:
        logger.warning(f" Failed to initialize feedback system: {e}")


def _extract_source_scores(sources: List[Any]) -> List[float]:
    """Extract numeric source scores from QA response sources."""
    scores: List[float] = []
    for source in sources or []:
        score = source.get("score") if isinstance(source, dict) else getattr(source, "score", None)
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    return scores


def _build_score_stats(scores: List[float]) -> Dict[str, float]:
    """Build simple retrieval score statistics."""
    if not scores:
        return {
            "count": 0.0,
            "top_score": 0.0,
            "mean_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
        }

    return {
        "count": float(len(scores)),
        "top_score": float(max(scores)),
        "mean_score": float(sum(scores) / len(scores)),
        "min_score": float(min(scores)),
        "max_score": float(max(scores)),
    }


def _compute_feedback_reward(
    rating: int,
    was_helpful: bool,
    was_accurate: bool,
    was_safe: bool,
) -> float:
    """
    Compute a normalized reward signal (0-1) from feedback.

    Weighted for RL experiments:
    - Rating: 40%
    - Helpful: 20%
    - Accurate: 20%
    - Safe: 20%
    """
    rating_component = (rating - 1) / 4.0
    reward = (
        0.4 * rating_component
        + 0.2 * float(was_helpful)
        + 0.2 * float(was_accurate)
        + 0.2 * float(was_safe)
    )
    return round(max(0.0, min(1.0, reward)), 4)


def _log_response_trajectory(trajectory_payload: Dict[str, Any]) -> None:
    """Persist a response trajectory for RL data collection."""
    _init_feedback_system()
    if trajectory_logger is None:
        return
    try:
        trajectory_logger.log(trajectory_payload)
    except Exception as e:
        logger.warning(f" Failed to persist response trajectory: {e}")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="ok",
        pipeline_ready=len(pipelines) > 0,
        message="Healthcare QA Chatbot API is running",
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Enhanced health check with system metrics."""
    import psutil

    # Gather metrics
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    uptime = time.time() - app.state.start_time

    # Knowledge base doc count
    doc_count = None
    shared = _shared_components
    if shared and "vector_store" in shared:
        try:
            doc_count = shared["vector_store"].collection.count()
        except Exception:
            pass

    # Cache stats
    cache_stats = None
    for p in pipelines.values():
        if hasattr(p, "cache_manager") and p.cache_manager:
            cache_stats = p.cache_manager.get_cache_stats()
            break

    return HealthResponse(
        status="healthy",
        pipeline_ready=len(pipelines) > 0,
        message="Service is healthy",
        uptime_seconds=round(uptime, 1),
        models_loaded=list(pipelines.keys()),
        knowledge_base_docs=doc_count,
        memory_usage_mb=round(memory_mb, 1),
        cache_stats=cache_stats,
    )


@app.get("/health/components", tags=["Health"])
async def component_health():
    """Detailed component health status for diagnostics."""
    from config.settings import config as _cfg
    pipeline_cfg = getattr(_cfg, 'pipeline', None)
    shared = _shared_components

    components = {}

    # Vector store
    try:
        if shared and "vector_store" in shared:
            doc_count = shared["vector_store"].collection.count()
            components["vector_store"] = {"status": "ok", "doc_count": doc_count}
        else:
            components["vector_store"] = {"status": "not_loaded"}
    except Exception as e:
        components["vector_store"] = {"status": "error", "error": str(e)}

    # BM25 index
    if shared and "retriever" in shared:
        retriever = shared["retriever"]
        if retriever.bm25 is not None:
            corpus_size = len(retriever.corpus) if retriever.corpus else 0
            components["bm25_index"] = {"status": "ok", "doc_count": corpus_size}
        else:
            components["bm25_index"] = {"status": "not_initialized"}
    else:
        components["bm25_index"] = {"status": "not_loaded"}

    # Optional components
    for name, key in [
        ("query_enhancer", "query_enhancer"),
        ("corrective_rag", "corrective_rag"),
        ("factual_consistency", "factual_consistency_checker"),
    ]:
        if shared and shared.get(key):
            components[name] = {"status": "ok"}
        else:
            enabled = getattr(pipeline_cfg, f"enable_{name}", False) if pipeline_cfg else False
            components[name] = {"status": "disabled" if not enabled else "failed_to_load"}

    # Reranker lives inside the retriever, not as a top-level shared component
    if shared and shared.get("retriever"):
        retriever_obj = shared["retriever"]
        reranker_obj = getattr(retriever_obj, "reranker", None)
        if reranker_obj is not None:
            model_id = getattr(reranker_obj, "model_name", "cross-encoder")
            components["reranker"] = {"status": "ok", "model": model_id}
        else:
            enabled = getattr(pipeline_cfg, "enable_reranker", False) if pipeline_cfg else False
            components["reranker"] = {"status": "disabled" if not enabled else "failed_to_load"}
    else:
        components["reranker"] = {"status": "not_loaded"}

    # NLI hallucination detector
    if hasattr(app.state, "hallucination_detector") and app.state.hallucination_detector:
        components["nli_detector"] = {
            "status": "ok",
            "model_loaded": app.state.hallucination_detector._nli_pipeline is not None,
        }
    else:
        components["nli_detector"] = {"status": "not_loaded"}

    # Cache
    cache_stats = None
    for p in pipelines.values():
        if hasattr(p, "cache_manager") and p.cache_manager:
            cache_stats = p.cache_manager.get_cache_stats()
            break
    components["cache"] = (
        {"status": "ok", **cache_stats} if cache_stats
        else {"status": "enabled" if getattr(pipeline_cfg, 'enable_response_cache', False) else "disabled"}
    )

    # Pipeline info
    default_pipe = getattr(pipeline_cfg, 'default_pipeline', 'standard') if pipeline_cfg else 'standard'

    return {
        "default_pipeline": default_pipe,
        "models_loaded": list(pipelines.keys()),
        "components": components,
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """Return available models and their descriptions."""
    result = {}
    for key, info in AVAILABLE_MODELS.items():
        result[key] = {
            "display_name": info["display_name"],
            "description": info["description"],
            "parameters": info["parameters"],
            "requires_gpu": info["requires_gpu"],
            "backend": info.get("backend", "transformers"),
            "loaded": any(k.startswith(key) for k in pipelines),
        }
    return result


@app.post("/sessions", tags=["Sessions"])
async def create_session():
    """Create a new conversation session."""
    _init_conversation_manager()
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Conversation manager not available")
    session = conversation_manager.create_session()
    return {"session_id": session.session_id}


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Get conversation history for a session."""
    _init_conversation_manager()
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Conversation manager not available")
    session = conversation_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


async def _prepare_and_execute_pipeline(request: QuestionRequest, start_ts: float) -> AnswerResponse:
    """Shared orchestration logic for standard and streaming requests."""
    response_id = str(uuid.uuid4())
    question_id = str(uuid.uuid4())
    request_timestamp = datetime.now(timezone.utc).isoformat()

    # ── 1. Safety check on input ─────────────────────────────────────────
    _init_guardrails()
    safety_info = SafetyInfo()

    if guardrails:
        check = guardrails.check_input(request.question)
        safety_info.level = check.level.value if hasattr(check.level, "value") else str(check.level)
        safety_info.flags = check.flags
        if check.redirect_message:
            safety_info.is_emergency = True
            safety_info.emergency_message = check.redirect_message

        # If emergency, return immediately with redirect
        if safety_info.is_emergency:
            latency = round((time.time() - start_ts) * 1000, 1)
            emergency_response = AnswerResponse(
                response_id=response_id,
                question=request.question,
                answer=check.redirect_message or "Please seek immediate medical attention.",
                sources=[],
                confidence=ConfidenceInfo(score=1.0, level="high", explanation="Emergency detected"),
                attributions=[],
                disclaimer="If this is a medical emergency, call emergency services (911) immediately.",
                model_used=request.model_choice or "tinyllama",
                pipeline_used="EmergencyRedirect",
                safety=safety_info,
                latency_ms=latency,
            )
            _log_response_trajectory(
                {
                    "response_id": response_id,
                    "question_id": question_id,
                    "timestamp": request_timestamp,
                    "question": request.question,
                    "effective_question": request.question,
                    "session_id": request.session_id,
                    "state_features": {
                        "question_length": len(request.question),
                        "num_sources_requested": float(request.num_sources),
                        "include_explanation": request.include_explanation,
                        "has_session_context": False,
                    },
                    "action": {
                        "model_choice": request.model_choice or "tinyllama",
                        "pipeline": "EmergencyRedirect",
                        "use_langchain": request.use_langchain,
                        "use_langgraph": request.use_langgraph,
                    },
                    "retrieval": _build_score_stats([]),
                    "outcome": {
                        "answer_length": len(emergency_response.answer),
                        "confidence_score": 1.0,
                        "confidence_level": "high",
                        "is_answerable": False,
                        "from_cache": False,
                        "factual_consistency": None,
                        "latency_ms": latency,
                    },
                    "safety": emergency_response.safety.model_dump() if emergency_response.safety else {},
                }
            )
            return emergency_response

    # ── 2. Resolve model and pipeline ────────────────────────────────────
    model_choice = request.model_choice or "tinyllama"
    pipeline_name = "Standard"

    # Determine pipeline: per-request flags override config default
    use_langgraph = request.use_langgraph
    use_langchain = request.use_langchain
    if not use_langgraph and not use_langchain:
        # Apply configured default pipeline
        from config.settings import config as _cfg
        default_pipe = getattr(getattr(_cfg, 'pipeline', None), 'default_pipeline', 'standard')
        if default_pipe == "langgraph":
            use_langgraph = True
        elif default_pipe == "langchain":
            use_langchain = True

    if use_langgraph:
        pipeline_name = "LangGraph"
        qa_pipeline = _get_langgraph_pipeline(model_choice)
    elif use_langchain:
        pipeline_name = "LangChain"
        qa_pipeline = _get_langchain_pipeline(model_choice)
    else:
        qa_pipeline = get_pipeline(model_choice=model_choice)

    if qa_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not initialized for {model_choice}. Check model loading.",
        )

    # ── 3. Conversation context ──────────────────────────────────────────
    _init_conversation_manager()
    effective_question = request.question
    session_id = request.session_id
    conversation_context = None

    # Auto-create session if none provided (enables multi-turn without explicit /sessions call)
    if conversation_manager and not session_id:
        try:
            session = conversation_manager.create_session()
            session_id = session.session_id
        except Exception:
            pass  # Proceed without session — single-turn still works

    if conversation_manager and session_id:
        ctx = conversation_manager.get_context_for_query(session_id, request.question)
        if ctx:
            # Pass conversation context separately — do NOT rewrite the question.
            # The pipeline uses it for better retrieval and prompt construction.
            conversation_context = ctx

    # ── 4. Get answer ────────────────────────────────────────────────────
    # Run synchronous LLM inference in a thread-pool executor so the asyncio
    # event loop stays free to serve /health, /models, and other fast endpoints
    # concurrently while TinyLlama/BioMistral generates (60-120s on CPU).
    try:
        kwargs = dict(
            question=effective_question,
            num_documents=request.num_sources,
            include_explanation=request.include_explanation,
        )
        if conversation_context and not (request.use_langgraph or request.use_langchain):
            kwargs["conversation_context"] = conversation_context

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            partial(qa_pipeline.answer, **kwargs),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question with {pipeline_name} pipeline: {str(e)}",
        )

    # ── 5. Safety check on output ────────────────────────────────────────
    if guardrails:
        output_check = guardrails.check_output(response.answer)
        if output_check.flags:
            safety_info.flags.extend(output_check.flags)
        
        # If output is blocked, return redirect message instead of raw answer
        if not output_check.passed:
            response.answer = output_check.redirect_message or "I cannot provide that information. Please consult a healthcare professional."
            response.confidence = {"score": 0.0, "level": "blocked", "explanation": output_check.message}
            safety_info.level = output_check.level

    if drug_checker:
        warnings = drug_checker.check_interaction_risk(response.answer)
        if warnings:
            safety_info.drug_warnings = [drug_checker.get_interaction_warning(warnings)]

    # ── 6. Record conversation turn ──────────────────────────────────────
    if conversation_manager and session_id:
        try:
            conversation_manager.add_turn(
                session_id=session_id,
                question=request.question,
                answer=response.answer,
                confidence=response.confidence.get("score", 0) if isinstance(response.confidence, dict) else getattr(response.confidence, "score", 0),
                sources=[s.get("source", "") if isinstance(s, dict) else getattr(s, "source", "") for s in response.sources],
            )
        except Exception:
            pass  # Don't fail the request if history save fails

    # ── 7. Build response ────────────────────────────────────────────────
    latency = round((time.time() - start_ts) * 1000, 1)
    confidence_payload = (
        response.confidence
        if isinstance(response.confidence, dict)
        else {
            "score": getattr(response.confidence, "score", 0.0),
            "level": getattr(response.confidence, "level", "low"),
            "explanation": getattr(response.confidence, "explanation", ""),
        }
    )

    confidence_score = float(confidence_payload.get("score", 0.0) or 0.0)
    confidence_level = str(confidence_payload.get("level", "low"))
    source_scores = _extract_source_scores(response.sources)
    score_stats = _build_score_stats(source_scores)
    source_names = [
        s.get("source", "") if isinstance(s, dict) else getattr(s, "source", "")
        for s in response.sources
    ]
    includes_followup_context = effective_question != request.question

    answer_response = AnswerResponse(
        response_id=response_id,
        question=request.question,
        answer=response.answer,
        sources=[SourceInfo(**s) if isinstance(s, dict) else SourceInfo(source=s.source, content=s.content, score=s.score, url=getattr(s, "url", ""), highlight_spans=getattr(s, "highlight_spans", None)) for s in response.sources],
        confidence=ConfidenceInfo(**response.confidence) if isinstance(response.confidence, dict) else ConfidenceInfo(score=response.confidence.score, level=response.confidence.level, explanation=response.confidence.explanation),
        attributions=[AttributionInfo(**a) if isinstance(a, dict) else AttributionInfo(claim=a.claim, source=a.source, evidence=a.evidence, similarity=a.similarity) for a in response.attributions],
        disclaimer=response.disclaimer,
        rationale=getattr(response, "rationale", None),
        model_used=model_choice,
        pipeline_used=pipeline_name,
        session_id=session_id,
        safety=safety_info,
        latency_ms=latency,
        confidence_breakdown=getattr(response, "confidence_breakdown", None),
        hallucination=getattr(response, "hallucination", None),
        stage_latencies=getattr(response, "stage_latencies", None),
    )

    _log_response_trajectory(
        {
            "response_id": response_id,
            "question_id": question_id,
            "timestamp": request_timestamp,
            "question": request.question,
            "effective_question": effective_question,
            "session_id": session_id,
            "state_features": {
                "question_length": len(request.question),
                "effective_question_length": len(effective_question),
                "num_sources_requested": float(request.num_sources),
                "include_explanation": request.include_explanation,
                "has_session_context": includes_followup_context,
            },
            "action": {
                "model_choice": model_choice,
                "pipeline": pipeline_name,
                "use_langchain": request.use_langchain,
                "use_langgraph": request.use_langgraph,
            },
            "retrieval": {
                **score_stats,
                "scores": source_scores[:10],
                "source_names": source_names[:10],
            },
            "outcome": {
                "answer_length": len(answer_response.answer),
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "is_answerable": bool(getattr(response, "is_answerable", True)),
                "from_cache": bool(getattr(response, "from_cache", False)),
                "factual_consistency": getattr(response, "factual_consistency", None),
                "latency_ms": latency,
                "num_sources_returned": len(answer_response.sources),
            },
            "safety": answer_response.safety.model_dump() if answer_response.safety else {},
            "metadata": {
                "model_used": answer_response.model_used,
                "pipeline_used": answer_response.pipeline_used,
            },
        }
    )

    return answer_response


@app.post("/ask", response_model=AnswerResponse, tags=["QA"],
          dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def ask_question(request: Request, body: QuestionRequest):
    """
    Ask a medical question and get an explainable answer.

    - Set model_choice to select any model exposed by /models.
    - Set session_id for multi-turn conversations.
    - Set use_langchain/use_langgraph for alternative pipelines.
    """
    start_ts = time.time()
    return await _prepare_and_execute_pipeline(body, start_ts)


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback tied to a prior response_id.

    Stores rating/labels in FeedbackCollector and links trajectory metadata
    for downstream RL training datasets.
    """
    _init_feedback_system()
    if feedback_collector is None:
        raise HTTPException(status_code=503, detail="Feedback system not available")

    trajectory = trajectory_logger.get(request.response_id) if trajectory_logger else None

    rating = request.rating
    if rating is None:
        observed_signals: List[float] = []
        if request.was_helpful is not None:
            observed_signals.append(float(request.was_helpful))
        if request.was_accurate is not None:
            observed_signals.append(float(request.was_accurate))
        if request.was_safe is not None:
            observed_signals.append(float(request.was_safe))
        if not observed_signals:
            raise HTTPException(
                status_code=400,
                detail="Provide either rating or at least one of was_helpful/was_accurate/was_safe.",
            )
        rating = int(round(1 + 4 * (sum(observed_signals) / len(observed_signals))))
        rating = max(1, min(5, rating))

    was_helpful = request.was_helpful if request.was_helpful is not None else rating >= 4
    was_accurate = request.was_accurate if request.was_accurate is not None else rating >= 4
    was_safe = request.was_safe if request.was_safe is not None else True
    reward_signal = _compute_feedback_reward(
        rating=rating,
        was_helpful=was_helpful,
        was_accurate=was_accurate,
        was_safe=was_safe,
    )

    metadata = {
        "response_id": request.response_id,
        "reward_signal": reward_signal,
        "trajectory_found": bool(trajectory),
    }

    if trajectory:
        metadata.update(
            {
                "original_question_id": trajectory.get("question_id"),
                "model_choice": trajectory.get("action", {}).get("model_choice"),
                "pipeline": trajectory.get("action", {}).get("pipeline"),
                "latency_ms": trajectory.get("outcome", {}).get("latency_ms"),
                "confidence_score": trajectory.get("outcome", {}).get("confidence_score"),
            }
        )

    feedback = feedback_collector.submit_feedback(
        question_id=request.response_id,
        rating=rating,
        was_helpful=was_helpful,
        was_accurate=was_accurate,
        was_safe=was_safe,
        feedback_text=request.feedback_text,
        session_id=request.session_id,
        **metadata,
    )

    return FeedbackResponse(
        status="ok",
        feedback_id=feedback.feedback_id,
        response_id=request.response_id,
        reward_signal=reward_signal,
        trajectory_found=bool(trajectory),
    )


@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats():
    """Return aggregate user feedback statistics."""
    _init_feedback_system()
    if feedback_collector is None:
        raise HTTPException(status_code=503, detail="Feedback system not available")
    stats = feedback_collector.get_statistics()
    return {
        "total_feedback": stats.total_feedback,
        "avg_rating": stats.avg_rating,
        "helpful_rate": stats.helpful_rate,
        "accuracy_rate": stats.accuracy_rate,
        "safety_rate": stats.safety_rate,
        "low_rated_count": stats.low_rated_count,
    }


@app.post("/ask/stream", tags=["QA"])
async def ask_stream(request: QuestionRequest):
    """Stream a medical answer token by token (NDJSON)."""
    start_ts = time.time()
    
    async def generate():
        try:
            response = await _prepare_and_execute_pipeline(request, start_ts)
            
            # Send metadata first (omit full answer to stream it separately)
            meta = response.model_dump()
            meta["answer"] = ""
            yield json.dumps({"type": "meta", "data": meta}) + "\n"
            
            # Simulated token stream (useful for LLMs that don't natively stream, maintaining UX)
            words = response.answer.split()
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield json.dumps({"type": "token", "data": chunk}) + "\n"
                
            yield json.dumps({"type": "done", "data": ""}) + "\n"
            
        except HTTPException as e:
            yield json.dumps({"type": "error", "data": e.detail}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/clear-cache", tags=["Health"])
async def clear_cache():
    """Clear cached Q&A responses so fresh answers are generated."""
    cleared = False
    for p in pipelines.values():
        if hasattr(p, "cache_manager") and p.cache_manager:
            p.cache_manager.invalidate_cache()
            cleared = True
    # Also clear disk cache
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        for f in cache_dir.glob("qa_*.json"):
            try:
                f.unlink()
                cleared = True
            except Exception:
                pass
    return {"status": "ok", "cleared": cleared}


# ── Alternative pipeline loaders ─────────────────────────────────────────────

def _get_langchain_pipeline(model_choice: str = "tinyllama", adapter_path: Optional[str] = None):
    """Lazy load LangChain pipeline with specified model.
    
    Args:
        model_choice: Model to use (tinyllama or biomistral)
    """
    resolved_adapter_path = _resolve_adapter_path(adapter_path)
    cache_key = f"langchain_{model_choice}::{resolved_adapter_path or 'base'}"
    if cache_key in pipelines:
        return pipelines[cache_key]
    try:
        shared = _get_shared_components()
        if not shared:
            return None
        from src.generation.llm_wrapper import MedicalLLM
        from src.langchain import create_langchain_pipeline

        import torch as _torch
        llm = MedicalLLM(
            model_name=model_choice,  # Use requested model instead of hardcoded tinyllama
            adapter_path=resolved_adapter_path,
            load_in_4bit=bool(resolved_adapter_path) and _torch.cuda.is_available(),
        )
        pipeline = create_langchain_pipeline(
            retriever=shared["retriever"],
            llm=llm,
            confidence_scorer=shared["confidence_scorer"],
            source_attributor=shared["source_attributor"],
        )
        pipelines[cache_key] = pipeline
        logger.info(f" LangChain pipeline loaded with {model_choice}")
        return pipeline
    except Exception as e:
        logger.error(f" Failed to load LangChain pipeline: {e}")
        return None


def _get_langgraph_pipeline(model_choice: str = "tinyllama", adapter_path: Optional[str] = None):
    """Lazy load LangGraph pipeline with specified model.
    
    Args:
        model_choice: Model to use (tinyllama or biomistral)
    """
    resolved_adapter_path = _resolve_adapter_path(adapter_path)
    cache_key = f"langgraph_{model_choice}::{resolved_adapter_path or 'base'}"
    if cache_key in pipelines:
        return pipelines[cache_key]
    try:
        shared = _get_shared_components()
        if not shared:
            return None
        from config.settings import config as _cfg
        from src.generation.llm_wrapper import MedicalLLM
        from src.langgraph import create_langgraph_pipeline

        import torch as _torch
        pipeline_cfg = getattr(_cfg, "pipeline", None)
        llm = MedicalLLM(
            model_name=model_choice,  # Use requested model instead of hardcoded tinyllama
            adapter_path=resolved_adapter_path,
            load_in_4bit=bool(resolved_adapter_path) and _torch.cuda.is_available(),
        )
        pipeline = create_langgraph_pipeline(
            retriever=shared["retriever"],
            llm=llm,
            confidence_scorer=shared["confidence_scorer"],
            source_attributor=shared["source_attributor"],
            enable_checkpointing=bool(
                getattr(pipeline_cfg, "enable_langgraph_checkpointing", False)
            ),
        )
        pipelines[cache_key] = pipeline
        logger.info(f" LangGraph pipeline loaded with {model_choice}")
        return pipeline
    except Exception as e:
        logger.error(f" Failed to load LangGraph pipeline: {e}")
        return None


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load shared components and default pipeline
    shared = _get_shared_components()

    # Pre-initialize BM25 index to avoid first-query delay
    if shared and "retriever" in shared:
        retriever = shared["retriever"]
        if hasattr(retriever, "initialize"):
            logger.info("Pre-initializing BM25 index...")
            retriever.initialize()
            logger.info("BM25 index ready")

    get_pipeline("tinyllama")
    _init_guardrails()
    _init_conversation_manager()
    _init_feedback_system()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
