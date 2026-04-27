"""
Central configuration management for Healthcare QA Chatbot.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class ModelChoice(str, Enum):
    """Available LLM model choices."""
    TINYLLAMA = "tinyllama"
    BIOMISTRAL = "biomistral"
    EXTRACTIVE = "extractive"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


# Model registry with metadata
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "extractive": {
        "model_name": "extractive",
        "display_name": "Extractive QA",
        "description": "Extracts answers directly from retrieved sources — fast, faithful, no hallucination",
        "parameters": "0",
        "max_new_tokens": 0,
        "requires_gpu": False,
        "load_in_4bit": False,
        "backend": "extractive",
    },
    "tinyllama": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "display_name": "TinyLlama 1.1B",
        "description": "Lightweight, fast responses — good for quick answers",
        "parameters": "1.1B",
        "max_new_tokens": 512,
        "requires_gpu": False,
        "load_in_4bit": False,
        "backend": "transformers",
    },
    "biomistral": {
        "model_name": str(MODELS_DIR / "biomistral" / "ggml-model-Q4_K_M.gguf"),
        "display_name": "BioMistral 7B (Q4_K_M)",
        "description": "Medical-domain Mistral 7B, quantised for CPU inference",
        "parameters": "7B",
        "max_new_tokens": 512,
        "requires_gpu": False,
        "load_in_4bit": False,
        "backend": "gguf",
    },
    "ollama": {
        "model_name": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        "display_name": "Llama 3.2 3B (Local)",
        "description": "Meta Llama 3.2 3B running locally via Ollama — no API key needed",
        "parameters": "3B",
        "max_new_tokens": 512,
        "requires_gpu": False,
        "load_in_4bit": False,
        "backend": "ollama",
    },
    "openrouter": {
        "model_name": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct"),
        "display_name": "Llama 3.2 3B (OpenRouter)",
        "description": "Meta Llama 3.2 3B served via OpenRouter API — requires OPENROUTER_API_KEY",
        "parameters": "3B",
        "max_new_tokens": 512,
        "requires_gpu": False,
        "load_in_4bit": False,
        "backend": "openrouter",
    },
}


@dataclass
class EmbeddingConfig:
    """Embedding model configuration (defaults match KB build + API query encoder)."""
    model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_name_article: str = "ncbi/MedCPT-Article-Encoder"
    dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    batch_size: int = 32
    device: str = "cuda" if os.getenv("USE_GPU", "true").lower() in ("true", "1", "yes") else "cpu"
    cache_dir: str = str(DATA_DIR / "embeddings")

@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    default_model: str = "tinyllama"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = False
    device_map: str = "auto"
    hf_home: str = os.getenv("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
    
    # Fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""
    vector_store_type: str = "chromadb"  # chromadb or pinecone
    collection_name: str = os.getenv("CHROMA_COLLECTION", "medical_knowledge_v2")
    persist_directory: str = os.getenv("KB_PERSIST_DIR", str(DATA_DIR / "knowledge_base_v2"))
    top_k: int = 7
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rerank_top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50
    bm25_batch_size: int = 5000
    rerank_fetch_multiplier: int = 3
    no_rerank_fetch_multiplier: int = 2
    # Score threshold: drop retrieved docs below this relevance score.
    # 0.0 = no filtering. After score normalization, typical cutoffs are 0.15–0.35.
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.0"))
    # Context window expansion: sentences before/after each chunk (0 = disabled).
    context_window_sentences: int = int(os.getenv("CONTEXT_WINDOW_SENTENCES", "0"))
    # BM25Okapi parameters (medical passages often benefit from slightly lower b).
    bm25_k1: float = float(os.getenv("BM25_K1", "1.2"))
    bm25_b: float = float(os.getenv("BM25_B", "0.5"))
    # When False, use dense_weight/sparse_weight for RRF instead of query-type table.
    use_adaptive_fusion: bool = os.getenv("USE_ADAPTIVE_FUSION", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    # Optional MMR re-query after embedding (adds latency; diversity vs pure RRF).
    enable_mmr_diversity: bool = os.getenv("ENABLE_MMR_DIVERSITY", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.5"))
    # Chroma HNSW metadata (applies when collection is created / rebuilt).
    hnsw_construction_ef: int = int(os.getenv("HNSW_CONSTRUCTION_EF", "200"))
    hnsw_M: int = int(os.getenv("HNSW_M", "16"))
    hnsw_search_ef: int = int(os.getenv("HNSW_SEARCH_EF", "100"))

@dataclass
class SafetyConfig:
    """Safety and guardrails configuration."""
    confidence_threshold: float = 0.5
    hallucination_threshold: float = 0.6
    enable_content_filter: bool = True
    enable_emergency_detection: bool = True
    always_include_disclaimer: bool = True

@dataclass
class PipelineConfig:
    """QA Pipeline configuration."""
    # Grounding gate thresholds
    min_retrieval_score: float = 0.1
    min_relevant_docs: int = 1
    enable_grounding_gate: bool = True
    # Adaptive threshold: doc is relevant if score >= max(absolute_floor, ratio * top_score)
    adaptive_threshold_ratio: float = 0.5
    absolute_score_floor: float = 0.01  # Used for normalized scores in [0, 1]

    # Fuse multiple query variants from QueryEnhancer (RRF over per-query rankings).
    enable_multi_query_fusion: bool = True
    # Grounding gate: require at least one doc with this token-overlap ratio vs query.
    grounding_min_term_overlap: float = 0.3

    # Enhanced pipeline feature flags
    enable_reranker: bool = True
    enable_query_enhancement: bool = True
    enable_context_compression: bool = False
    enable_corrective_rag: bool = False   # second retrieval pass adds 2s+ on CPU
    enable_factual_consistency: bool = False  # DeBERTa reloads per-request; hallucination_detector covers this
    
    # Pipeline selection: "standard" (default), "langchain", or "langgraph"
    default_pipeline: str = os.getenv("DEFAULT_PIPELINE", "standard")
    # In-memory LangGraph checkpoints are useful for debugging but can grow
    # without bound in a long-lived API process, so keep them opt-in.
    enable_langgraph_checkpointing: bool = os.getenv(
        "ENABLE_LANGGRAPH_CHECKPOINTING", "false"
    ).lower() == "true"

    # MCP Integration
    enable_mcp_search: bool = os.getenv("ENABLE_MCP_SEARCH", "false").lower() == "true"
    mcp_search_cmd: str = os.getenv("MCP_SEARCH_CMD", "npx")
    mcp_search_args: str = os.getenv("MCP_SEARCH_ARGS", "-y @modelcontextprotocol/server-brave-search")
    
    # Post-generation quality gate: answers with calibrated confidence below this
    # Unused threshold kept for backward compat. Quality gate now uses
    # medication-entity grounding check and caps-spam detection instead.
    min_answer_confidence: float = float(os.getenv("MIN_ANSWER_CONFIDENCE", "0.0"))

    # Caching
    enable_response_cache: bool = False
    cache_ttl_seconds: int = 3600
    max_cache_items: int = 1000
    cache_dir: str = str(DATA_DIR / "cache")
    max_context_tokens: int = 4000
    corrective_rag_max_context: int = 2000
    sentence_length_divisor: int = 200


@dataclass
class XAIConfig:
    """XAI (Explainability) configuration."""
    confidence_high_threshold: float = 0.8
    confidence_low_threshold: float = 0.5
    max_evidence_length: int = 80
    nli_entailment_threshold: float = 0.5


@dataclass
class ModelConfig:
    """Model runtime configuration."""
    cpu_threads: int = 4
    max_context_length: int = 2048


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    rate_limit_per_minute: int = 60
    max_question_length: int = 1000
    min_question_length: int = 5
    conversation_storage_path: str = os.getenv(
        "CONVERSATION_STORAGE_PATH",
        str(DATA_DIR / "conversations" / "sessions.json"),
    )
    conversation_max_age_hours: int = int(
        os.getenv("CONVERSATION_MAX_AGE_HOURS", "24")
    )
    conversation_cleanup_interval_minutes: int = int(
        os.getenv("CONVERSATION_CLEANUP_INTERVAL_MINUTES", "15")
    )

@dataclass
class Config:
    """Main configuration class."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    api: APIConfig = field(default_factory=APIConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    
    @classmethod
    def from_env(cls, env: str = None) -> "Config":
        """Create config based on environment."""
        env = env or os.getenv("ENVIRONMENT", "development")
        
        if env == "production":
            return cls(
                llm=LLMConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", load_in_4bit=False),
                safety=SafetyConfig(confidence_threshold=0.6),
                pipeline=PipelineConfig(
                    min_retrieval_score=0.1,
                    cache_ttl_seconds=7200,
                    enable_response_cache=False,
                    min_answer_confidence=0.0,
                ),
            )
        elif env == "testing":
            return cls(
                llm=LLMConfig(model_name="tinyllama", load_in_4bit=False),
                pipeline=PipelineConfig(enable_response_cache=False)
            )
        else:  # development
            return cls(
                llm=LLMConfig(model_name="tinyllama", load_in_4bit=False),
                # Cache enabled in dev too — same question twice should be instant.
                # TTL is shorter (30 min) so stale answers don't linger during iteration.
                pipeline=PipelineConfig(
                    enable_response_cache=False,
                    cache_ttl_seconds=1800,
                ),
            )

# Global config instance — honor ENVIRONMENT for production/testing overrides.
config = Config.from_env()
