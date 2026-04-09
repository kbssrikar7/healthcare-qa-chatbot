"""
Simplified API server for Healthcare QA Chatbot - Demo Mode.

⚠️ SAFETY WARNING ⚠️
This server bypasses the RAG retrieval pipeline entirely and generates answers
from the LLM alone (no grounding, no source attribution, no factual verification).

Do NOT use this server for:
- Any evaluation or benchmarking (results are not comparable to the RAG pipeline)
- Any user-facing deployment (answers are ungrounded and potentially hallucinatory)
- Any academic demonstration claiming RAG+XAI capabilities

This server exists ONLY for quick LLM-only smoke tests.
For the full pipeline, use: python api/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Healthcare QA Chatbot API (⚠️ DEMO MODE — UNGROUNDED)",
    description=(
        "⚠️ DEMO ONLY — This server bypasses RAG retrieval entirely. "
        "Answers are generated from the LLM alone without source grounding. "
        "For the full pipeline, use: python api/main.py"
    ),
    version="1.0.0-demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    include_explanation: bool = True
    num_sources: int = Field(default=3, ge=1, le=10)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    confidence: Dict
    attributions: List[Dict]
    disclaimer: str
    rationale: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    message: str


# Global LLM instance
llm = None
rationale_gen = None


def get_llm():
    """Load the fine-tuned LLM."""
    global llm, rationale_gen
    if llm is None:
        try:
            from src.generation.llm_wrapper import MedicalLLM
            from src.xai.rationale_generator import RationaleGenerator

            print("🔄 Loading LLM...")

            # Check if we have a fine-tuned adapter
            project_root = Path(__file__).parent.parent
            adapter_path = project_root / "models/fine_tuned/medical_adapter"
            if adapter_path.exists():
                print(f"✅ Found adapter at {adapter_path}")
                llm = MedicalLLM(
                    model_name="tinyllama", adapter_path=str(adapter_path), load_in_4bit=True
                )
            else:
                print("⚠️ No adapter found, using base model")
                llm = MedicalLLM(model_name="tinyllama", load_in_4bit=True)

            rationale_gen = RationaleGenerator(llm)
            print("✅ LLM loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            llm = None
    return llm, rationale_gen


# Medical prompts
MEDICAL_PROMPT = """You are a knowledgeable medical assistant. Answer the following medical question accurately and helpfully.

Question: {question}

Provide a clear, informative answer. Include relevant medical information but always recommend consulting healthcare professionals for medical decisions.

Answer:"""


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="ok",
        pipeline_ready=llm is not None,
        message="⚠️ DEMO MODE — Answers are NOT grounded in retrieved sources. Use api/main.py for full RAG pipeline.",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        pipeline_ready=llm is not None,
        message="⚠️ DEMO MODE (ungrounded) — For production, use api/main.py",
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a medical question."""
    model, rationale_generator = get_llm()

    if model is None:
        raise HTTPException(status_code=503, detail="LLM not initialized. Check model loading.")

    try:
        # Generate answer
        prompt = MEDICAL_PROMPT.format(question=request.question)
        result = model.generate(prompt, max_new_tokens=300, temperature=0.7)
        answer = result.response.strip()

        # Generate rationale
        rationale = None
        if request.include_explanation and rationale_generator:
            try:
                rationale = rationale_generator.generate_rationale(
                    question=request.question,
                    answer=answer,
                    context="Based on medical knowledge and training data.",
                )
            except Exception as e:
                print(f"Rationale generation failed: {e}")

        # Calculate confidence (simplified)
        confidence = {
            "score": 0.75,
            "level": "medium",
            "explanation": "Answer generated from fine-tuned medical knowledge model.",
        }

        disclaimer = (
            "⚠️ DEMO MODE: This answer was generated WITHOUT source retrieval. "
            "It is NOT grounded in peer-reviewed medical literature. "
            "This information is for educational purposes only. "
            "Always consult a healthcare professional for medical advice."
        )

        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=[],  # No retrieval in demo mode
            confidence=confidence,
            attributions=[],
            disclaimer=disclaimer,
            rationale=rationale,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


if __name__ == "__main__":
    import sys as _sys

    print("\n" + "=" * 70, file=_sys.stderr)
    print("⚠️  WARNING: RUNNING IN DEMO MODE (UNGROUNDED)", file=_sys.stderr)
    print("   Answers are generated from the LLM alone.", file=_sys.stderr)
    print("   There is NO retrieval, NO source attribution, NO XAI.", file=_sys.stderr)
    print("   For the full RAG pipeline, use: python api/main.py", file=_sys.stderr)
    print("=" * 70 + "\n", file=_sys.stderr)

    # Pre-load LLM
    get_llm()

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
