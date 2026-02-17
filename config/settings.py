"""
Central configuration management for Healthcare QA Chatbot.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "ncbi/MedCPT-Query-Encoder"
    model_name_article: str = "ncbi/MedCPT-Article-Encoder"
    dimension: int = 768
    batch_size: int = 32
    device: str = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"
    cache_dir: str = str(DATA_DIR / "embeddings")

@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "BioMistral/BioMistral-7B"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = True
    device_map: str = "auto"
    
    # Fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""
    vector_store_type: str = "chromadb"  # chromadb or pinecone
    collection_name: str = "medical_knowledge"
    persist_directory: str = str(DATA_DIR / "knowledge_base")
    top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rerank_top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50

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
    min_retrieval_score: float = 0.3
    min_relevant_docs: int = 1
    enable_grounding_gate: bool = True
    
    # Caching
    enable_response_cache: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_items: int = 1000
    cache_dir: str = str(DATA_DIR / "cache")


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    rate_limit_per_minute: int = 60
    max_question_length: int = 1000
    min_question_length: int = 5

@dataclass
class Config:
    """Main configuration class."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    api: APIConfig = field(default_factory=APIConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
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
                llm=LLMConfig(model_name="BioMistral/BioMistral-7B", load_in_4bit=True),
                safety=SafetyConfig(confidence_threshold=0.6),
                pipeline=PipelineConfig(min_retrieval_score=0.4, cache_ttl_seconds=7200)
            )
        elif env == "testing":
            return cls(
                llm=LLMConfig(model_name="tinyllama", load_in_4bit=False),
                pipeline=PipelineConfig(enable_response_cache=False)
            )
        else:  # development
            return cls(
                llm=LLMConfig(model_name="tinyllama", load_in_4bit=False)
            )

# Global config instance
config = Config()

