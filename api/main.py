"""
FastAPI application for Healthcare QA Chatbot.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare QA Chatbot API",
    description="An explainable medical question-answering system combining LLM + RAG + XAI",
    version="1.0.0"
)

# CORS configuration - Environment-based for security
# In production, set CORS_ORIGINS to comma-separated allowed origins
# e.g., CORS_ORIGINS="https://example.com,https://app.example.com"
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    allow_origins = ["*"]  # Development mode - allow all
else:
    allow_origins = [origin.strip() for origin in cors_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances (lazy loaded)
pipeline = None
langchain_pipeline = None
langgraph_pipeline = None

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    include_explanation: bool = True
    num_sources: int = Field(default=3, ge=1, le=10)
    use_langchain: bool = Field(default=False, description="Use LangChain LCEL-based pipeline")
    use_langgraph: bool = Field(default=False, description="Use LangGraph self-correcting RAG pipeline")

class SourceInfo(BaseModel):
    source: str
    content: str
    score: float
    url: Optional[str] = ""

class ConfidenceInfo(BaseModel):
    score: float
    level: str
    explanation: str

class AttributionInfo(BaseModel):
    claim: str
    source: str
    evidence: str
    similarity: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    confidence: ConfidenceInfo
    attributions: List[AttributionInfo]
    disclaimer: str
    rationale: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    message: str

def get_pipeline():
    """Lazy load the pipeline."""
    global pipeline
    if pipeline is None:
        try:
            from src.embeddings.embedding_models import MedicalEmbedder
            from src.embeddings.vector_store import VectorStore
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.generation.llm_wrapper import MedicalLLM
            from src.generation.prompt_manager import MedicalPromptManager
            from src.xai.confidence_scorer import ConfidenceScorer
            from src.xai.source_attribution import SourceAttributor
            from src.pipeline.qa_pipeline import HealthcareQAPipeline
            
            print("🔄 Loading pipeline components...")
            embedder = MedicalEmbedder(model_name="all-minilm")
            vector_store = VectorStore(
                collection_name="medical_knowledge",
                persist_directory="data/knowledge_base"
            )
            retriever = HybridRetriever(embedder, vector_store)
            
            # Load LLM with fine-tuned adapter if available
            from pathlib import Path
            adapter_path = Path("models/fine_tuned/medical_adapter")
            if adapter_path.exists():
                print(f"✅ Found fine-tuned adapter at {adapter_path}")
                llm = MedicalLLM(
                    model_name="tinyllama",
                    adapter_path=str(adapter_path),
                    load_in_4bit=True
                )
            else:
                print("⚠️ No adapter found, using base model")
                llm = MedicalLLM(model_name="tinyllama", load_in_4bit=False)
            prompt_manager = MedicalPromptManager()
            confidence_scorer = ConfidenceScorer()
            source_attributor = SourceAttributor()
            
            pipeline = HealthcareQAPipeline(
                retriever=retriever,
                llm=llm,
                prompt_manager=prompt_manager,
                confidence_scorer=confidence_scorer,
                source_attributor=source_attributor
            )
            print("✅ Pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            pipeline = None
    return pipeline

def get_langchain_pipeline():
    """Lazy load the LangChain-based pipeline."""
    global langchain_pipeline
    if langchain_pipeline is None:
        try:
            from src.embeddings.embedding_models import MedicalEmbedder
            from src.embeddings.vector_store import VectorStore
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.generation.llm_wrapper import MedicalLLM
            from src.xai.confidence_scorer import ConfidenceScorer
            from src.xai.source_attribution import SourceAttributor
            from src.langchain import create_langchain_pipeline
            
            print("🔄 Loading LangChain pipeline components...")
            embedder = MedicalEmbedder(model_name="all-minilm")
            vector_store = VectorStore(
                collection_name="medical_knowledge",
                persist_directory="data/knowledge_base"
            )
            retriever = HybridRetriever(embedder, vector_store)
            
            # Load LLM with fine-tuned adapter if available
            from pathlib import Path
            adapter_path = Path("models/fine_tuned/medical_adapter")
            if adapter_path.exists():
                print(f"✅ Found fine-tuned adapter at {adapter_path}")
                llm = MedicalLLM(
                    model_name="tinyllama",
                    adapter_path=str(adapter_path),
                    load_in_4bit=True
                )
            else:
                print("⚠️ No adapter found, using base model")
                llm = MedicalLLM(model_name="tinyllama", load_in_4bit=False)
            
            confidence_scorer = ConfidenceScorer()
            source_attributor = SourceAttributor()
            
            langchain_pipeline = create_langchain_pipeline(
                retriever=retriever,
                llm=llm,
                confidence_scorer=confidence_scorer,
                source_attributor=source_attributor
            )
            print("✅ LangChain pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LangChain pipeline: {e}")
            langchain_pipeline = None
    return langchain_pipeline

def get_langgraph_pipeline():
    """Lazy load the LangGraph-based pipeline."""
    global langgraph_pipeline
    if langgraph_pipeline is None:
        try:
            from src.embeddings.embedding_models import MedicalEmbedder
            from src.embeddings.vector_store import VectorStore
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.generation.llm_wrapper import MedicalLLM
            from src.xai.confidence_scorer import ConfidenceScorer
            from src.xai.source_attribution import SourceAttributor
            from src.langgraph import create_langgraph_pipeline
            
            print("🔄 Loading LangGraph pipeline components...")
            embedder = MedicalEmbedder(model_name="all-minilm")
            vector_store = VectorStore(
                collection_name="medical_knowledge",
                persist_directory="data/knowledge_base"
            )
            retriever = HybridRetriever(embedder, vector_store)
            
            # Load LLM with fine-tuned adapter if available
            from pathlib import Path
            adapter_path = Path("models/fine_tuned/medical_adapter")
            if adapter_path.exists():
                print(f"✅ Found fine-tuned adapter at {adapter_path}")
                llm = MedicalLLM(
                    model_name="tinyllama",
                    adapter_path=str(adapter_path),
                    load_in_4bit=True
                )
            else:
                print("⚠️ No adapter found, using base model")
                llm = MedicalLLM(model_name="tinyllama", load_in_4bit=False)
            
            confidence_scorer = ConfidenceScorer()
            source_attributor = SourceAttributor()
            
            langgraph_pipeline = create_langgraph_pipeline(
                retriever=retriever,
                llm=llm,
                confidence_scorer=confidence_scorer,
                source_attributor=source_attributor
            )
            print("✅ LangGraph pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LangGraph pipeline: {e}")
            langgraph_pipeline = None
    return langgraph_pipeline

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="ok",
        pipeline_ready=pipeline is not None,
        message="Healthcare QA Chatbot API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        pipeline_ready=pipeline is not None,
        message="Service is healthy"
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a medical question and get an explainable answer.
    
    Set use_langchain=true for the LangChain LCEL-based pipeline.
    Set use_langgraph=true for the LangGraph self-correcting RAG pipeline.
    """
    # Choose pipeline based on request
    if request.use_langgraph:
        qa_pipeline = get_langgraph_pipeline()
        pipeline_name = "LangGraph"
    elif request.use_langchain:
        qa_pipeline = get_langchain_pipeline()
        pipeline_name = "LangChain"
    else:
        qa_pipeline = get_pipeline()
        pipeline_name = "Standard"
    
    if qa_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"{pipeline_name} pipeline not initialized. Please check that the knowledge base is built."
        )
    
    try:
        # LangGraph and LangChain pipelines use simple .answer(question) interface
        if request.use_langgraph or request.use_langchain:
            response = qa_pipeline.answer(request.question)
        else:
            response = qa_pipeline.answer(
                question=request.question,
                num_documents=request.num_sources,
                include_explanation=request.include_explanation
            )
        
        return AnswerResponse(
            question=response.question,
            answer=response.answer,
            sources=[SourceInfo(**s) for s in response.sources],
            confidence=ConfidenceInfo(**response.confidence),
            attributions=[AttributionInfo(**a) for a in response.attributions],
            disclaimer=response.disclaimer,
            rationale=getattr(response, 'rationale', None)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question with {pipeline_name} pipeline: {str(e)}"
        )

@app.post("/ask/simple")
async def ask_simple(question: str):
    """
    Simple question endpoint (minimal response).
    """
    qa_pipeline = get_pipeline()
    
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        response = qa_pipeline.answer(
            question=question,
            num_documents=3,
            include_explanation=False
        )
        return {
            "answer": response.answer,
            "confidence": response.confidence["level"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-cache")
async def clear_cache():
    """Clear cached Q&A responses so fresh answers are generated."""
    cleared = False
    # Clear pipeline cache if available
    if pipeline and hasattr(pipeline, 'cache_manager') and pipeline.cache_manager:
        pipeline.cache_manager.invalidate_cache()
        cleared = True
    # Also clear cache files directly as fallback
    import glob
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        for f in cache_dir.glob("qa_*.json"):
            try:
                f.unlink()
                cleared = True
            except Exception:
                pass
    return {"status": "ok", "cleared": cleared}

if __name__ == "__main__":
    # Pre-load pipeline
    get_pipeline()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
