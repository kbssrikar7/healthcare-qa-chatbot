"""
LangGraph Routing Logic for Healthcare RAG Pipeline.

Defines conditional edge functions that determine the next node
based on current state.
"""
from typing import Literal  # noqa: F401 — kept for re-export compatibility
from langgraph.graph import END

from src.langgraph.langgraph_state import HealthcareRAGState
from src.langgraph.langgraph_nodes import MAX_RETRY_COUNT, MIN_RELEVANT_DOCS


def route_after_grading(state: HealthcareRAGState) -> Literal["generate", "refine", "unanswerable"]:
    """
    Route after document grading.
    
    Decides whether to:
    - "generate": Proceed with answer generation (enough relevant docs)
    - "refine": Retry with refined query (not enough relevant docs, can retry)
    - "unanswerable": Give up and return safe response (max retries reached)
    """
    is_answerable = state.get("is_answerable", False)
    retry_count = state.get("retry_count", 0)
    documents = state.get("documents", [])
    
    # No documents at all
    if not documents:
        if retry_count < MAX_RETRY_COUNT:
            return "refine"
        return "unanswerable"
    
    # Enough relevant documents
    if is_answerable:
        return "generate"
    
    # Not enough relevant docs, can still retry
    if retry_count < MAX_RETRY_COUNT:
        return "refine"
    
    # Exhausted retries and question not answerable
    # For medical QA, respect the grounding gate - don't generate if not answerable
    return "unanswerable"


def route_after_verify(state: HealthcareRAGState) -> Literal["enrich_xai", "refine"]:
    """
    Route after grounding verification.
    
    Decides whether to:
    - "enrich_xai": Answer is grounded, proceed to XAI enrichment
    - "refine": Answer not grounded, retry retrieval/refinement loop
    """
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)
    
    if is_grounded:
        return "enrich_xai"

    if retry_count < MAX_RETRY_COUNT:
        return "refine"

    # If retries are exhausted, continue with current answer path.
    # This avoids infinite loops in degraded scenarios.
    return "enrich_xai"


