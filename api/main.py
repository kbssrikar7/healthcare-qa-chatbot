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

# Global pipeline instance (lazy loaded)
pipeline = None

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    include_explanation: bool = True
    num_sources: int = Field(default=3, ge=1, le=10)

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
    """
    # Get or initialize pipeline
    qa_pipeline = get_pipeline()
    
    if qa_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please check that the knowledge base is built."
        )
    
    try:
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
            detail=f"Error processing question: {str(e)}"
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
