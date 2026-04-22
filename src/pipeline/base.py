"""
Structural Protocol defining the shared interface for all QA pipeline variants.

All three pipeline implementations (Standard, LangChain, LangGraph) must
satisfy this protocol so the API layer can dispatch to any of them without
knowing the concrete type.
"""

from typing import Protocol, runtime_checkable

from src.pipeline.qa_pipeline import QAResponse


@runtime_checkable
class PipelineProtocol(Protocol):
    """Minimal interface that every pipeline backend must expose."""

    def answer(
        self,
        question: str,
        *,
        num_documents: int = 5,
        include_explanation: bool = True,
        **kwargs,
    ) -> QAResponse:
        """Answer a medical question and return a structured QAResponse."""
        ...
