"""
LangGraph State Schema for Healthcare RAG Pipeline.

Defines the state structure that flows through all nodes in the graph.
Uses TypedDict with Annotated reducers for proper state accumulation.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class HealthcareRAGState(TypedDict, total=False):
    """
    State schema for the Healthcare RAG graph.

    This state flows through all nodes, each node reading what it needs
    and returning partial updates.

    Attributes:
        question: The user's medical question
        documents: Retrieved documents from the knowledge base
        doc_grades: Relevance grades for each document (accumulates)
        query_history: History of query refinements (accumulates)
        retry_count: Number of retrieval retry attempts
        context: Formatted context string for LLM
        answer: Generated answer text
        confidence: Confidence scoring results
        attributions: Source attribution results
        rationale: Generated rationale/explanation
        is_answerable: Whether the question can be answered
        is_grounded: Whether the answer is grounded in context
        needs_review: Whether human review is required
        error: Any error message
    """

    # Input
    question: str

    # Retrieval phase
    documents: List[Document]
    doc_grades: Annotated[List[Dict], operator.add]  # Accumulates grades
    query_history: Annotated[List[str], operator.add]  # Accumulates refined queries
    retry_count: int

    # Generation phase
    context: str
    answer: str

    # XAI enrichment
    confidence: Dict[str, Any]
    attributions: List[Dict]
    rationale: Optional[str]

    # Control flow flags
    is_answerable: bool
    is_grounded: bool
    needs_review: bool

    # Error handling
    error: Optional[str]


# Default state factory
def create_initial_state(question: str) -> HealthcareRAGState:
    """
    Create initial state for a new query.

    Args:
        question: The user's medical question

    Returns:
        Initial state with defaults set
    """
    return HealthcareRAGState(
        question=question,
        documents=[],
        doc_grades=[],
        query_history=[question],
        retry_count=0,
        context="",
        answer="",
        confidence={},
        attributions=[],
        rationale=None,
        is_answerable=True,
        is_grounded=True,
        needs_review=False,
        error=None,
    )


# Routing decision types
RouteAfterGrading = Literal["generate", "refine", "unanswerable"]
RouteAfterVerify = Literal["enrich_xai", "refine"]
RouteAfterXAI = Literal["end", "review"]
