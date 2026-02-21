"""
LangGraph integration package for Healthcare QA Chatbot.

Provides a self-correcting RAG pipeline using LangGraph StateGraph
with support for:
- Automatic query refinement
- Document relevance grading
- Grounding verification
- XAI enrichment
- Checkpointing for debugging

Usage:
    from src.langgraph import (
        create_langgraph_pipeline,
        LangGraphHealthcareQAPipeline
    )
    
    # Create pipeline with existing components
    pipeline = create_langgraph_pipeline(
        retriever=my_hybrid_retriever,
        llm=my_medical_llm,
        confidence_scorer=my_scorer
    )
    
    # Answer a question
    result = pipeline.answer("What is diabetes?")
    
    # Stream execution for debugging
    for event in pipeline.stream("What is hypertension?"):
        print(event)
"""

from src.langgraph.langgraph_state import (
    HealthcareRAGState,
    create_initial_state
)

from src.langgraph.langgraph_nodes import (
    HealthcareRAGNodes,
    MEDICAL_DISCLAIMER,
    UNANSWERABLE_RESPONSE
)

from src.langgraph.langgraph_routing import (
    route_after_grading,
    route_after_verify,
    route_after_xai,
    should_continue_retrieval
)

from src.langgraph.langgraph_pipeline import (
    LangGraphHealthcareQAPipeline,
    LangGraphQAResult,
    create_langgraph_pipeline
)


__all__ = [
    # State
    "HealthcareRAGState",
    "create_initial_state",
    
    # Nodes
    "HealthcareRAGNodes",
    "MEDICAL_DISCLAIMER",
    "UNANSWERABLE_RESPONSE",
    
    # Routing
    "route_after_grading",
    "route_after_verify",
    "route_after_xai",
    "should_continue_retrieval",
    
    # Pipeline
    "LangGraphHealthcareQAPipeline",
    "LangGraphQAResult",
    "create_langgraph_pipeline",
]
