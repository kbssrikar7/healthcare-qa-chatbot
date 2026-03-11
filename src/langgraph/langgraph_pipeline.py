"""
LangGraph Pipeline for Healthcare RAG.

Builds a self-correcting RAG graph using StateGraph with:
- Retrieval with automatic retry/refinement
- Document grading and quality gates
- Answer generation with grounding verification
- XAI enrichment (confidence, attribution, rationale)
"""
from typing import Optional, Any
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.langgraph.langgraph_state import HealthcareRAGState, create_initial_state
from src.langgraph.langgraph_nodes import HealthcareRAGNodes, MEDICAL_DISCLAIMER
from src.langgraph.langgraph_routing import (
    route_after_grading,
    route_after_verify,
    route_after_xai
)
from src.pipeline.qa_pipeline import QAResponse


@dataclass
class LangGraphQAResult:
    """Result from LangGraph QA pipeline."""
    question: str
    answer: str
    documents: list
    context: str
    is_answerable: bool
    is_grounded: bool
    confidence: dict
    attributions: list
    rationale: Optional[str]
    needs_review: bool
    disclaimer: str = MEDICAL_DISCLAIMER


class LangGraphHealthcareQAPipeline:
    """
    LangGraph-based Healthcare QA Pipeline.
    
    Uses a self-correcting RAG graph that:
    1. Retrieves documents
    2. Grades their relevance
    3. Refines query if needed (loop)
    4. Generates answer
    5. Verifies grounding
    6. Enriches with XAI components
    
    Example:
        pipeline = LangGraphHealthcareQAPipeline(
            retriever=hybrid_retriever,
            llm=medical_llm,
            confidence_scorer=scorer
        )
        result = pipeline.invoke("What is diabetes?")
    """
    
    def __init__(
        self,
        retriever,
        llm,
        confidence_scorer=None,
        source_attributor=None,
        rationale_generator=None,
        k: int = 5,
        enable_checkpointing: bool = True
    ):
        """
        Initialize LangGraph Healthcare QA Pipeline.
        
        Args:
            retriever: HybridRetriever instance
            llm: MedicalLLM instance
            confidence_scorer: Optional ConfidenceScorer
            source_attributor: Optional SourceAttributor  
            rationale_generator: Optional RationaleGenerator
            k: Number of documents to retrieve
            enable_checkpointing: Enable state checkpointing for debugging
        """
        self.k = k
        self.enable_checkpointing = enable_checkpointing
        
        # Create nodes with existing components
        self.nodes = HealthcareRAGNodes(
            retriever=retriever,
            llm=llm,
            confidence_scorer=confidence_scorer,
            source_attributor=source_attributor,
            rationale_generator=rationale_generator,
            k=k
        )
        
        # Build and compile the graph
        self._graph = self._build_graph()
    
    def _build_graph(self):
        """
        Build the self-correcting RAG StateGraph.
        
        Graph structure:
            START → retrieve → grade → [conditional]
                                    ├→ generate → verify → enrich_xai → END
                                    ├→ refine → retrieve (loop)
                                    └→ unanswerable → END
        """
        builder = StateGraph(HealthcareRAGState)
        
        # Add nodes
        builder.add_node("retrieve", self.nodes.retrieve_documents)
        builder.add_node("grade", self.nodes.grade_relevance)
        builder.add_node("refine", self.nodes.refine_query)
        builder.add_node("generate", self.nodes.generate_answer)
        builder.add_node("verify", self.nodes.verify_grounding)
        builder.add_node("enrich_xai", self.nodes.enrich_xai)
        builder.add_node("unanswerable", self.nodes.unanswerable_response)
        
        # Static edges
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_edge("refine", "retrieve")  # LOOP back
        builder.add_edge("generate", "verify")
        builder.add_edge("unanswerable", END)
        
        # Conditional edges
        builder.add_conditional_edges(
            "grade",
            route_after_grading,
            {
                "generate": "generate",
                "refine": "refine",
                "unanswerable": "unanswerable"
            }
        )
        
        # After verification: re-route to refine if not grounded, else proceed
        builder.add_conditional_edges(
            "verify",
            route_after_verify,
            {
                "enrich_xai": "enrich_xai",
                "refine": "refine",
            }
        )
        
        # After XAI: always end (review flag is in state for caller to inspect)
        builder.add_edge("enrich_xai", END)
        
        # Compile with optional checkpointing
        if self.enable_checkpointing:
            checkpointer = MemorySaver()
            return builder.compile(checkpointer=checkpointer)
        else:
            return builder.compile()
    
    def invoke(self, question: str, config: Optional[dict] = None) -> LangGraphQAResult:
        """
        Answer a medical question using the LangGraph pipeline.
        
        Args:
            question: User's medical question
            config: Optional config with thread_id for checkpointing
            
        Returns:
            LangGraphQAResult with answer and metadata
        """
        # Create initial state
        initial_state = create_initial_state(question)
        
        # Use default config if not provided
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # Execute the graph
        final_state = self._graph.invoke(initial_state, config)
        
        # Extract results
        return LangGraphQAResult(
            question=question,
            answer=final_state.get("answer", ""),
            documents=final_state.get("documents", []),
            context=final_state.get("context", ""),
            is_answerable=final_state.get("is_answerable", False),
            is_grounded=final_state.get("is_grounded", False),
            confidence=final_state.get("confidence", {}),
            attributions=final_state.get("attributions", []),
            rationale=final_state.get("rationale"),
            needs_review=final_state.get("needs_review", False),
            disclaimer=MEDICAL_DISCLAIMER
        )
    
    async def ainvoke(self, question: str, config: Optional[dict] = None) -> LangGraphQAResult:
        """Async version of invoke."""
        initial_state = create_initial_state(question)
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        final_state = await self._graph.ainvoke(initial_state, config)
        
        return LangGraphQAResult(
            question=question,
            answer=final_state.get("answer", ""),
            documents=final_state.get("documents", []),
            context=final_state.get("context", ""),
            is_answerable=final_state.get("is_answerable", False),
            is_grounded=final_state.get("is_grounded", False),
            confidence=final_state.get("confidence", {}),
            attributions=final_state.get("attributions", []),
            rationale=final_state.get("rationale"),
            needs_review=final_state.get("needs_review", False),
            disclaimer=MEDICAL_DISCLAIMER
        )
    
    def stream(self, question: str, config: Optional[dict] = None):
        """
        Stream the graph execution for debugging/observability.
        
        Yields state updates after each node.
        """
        initial_state = create_initial_state(question)
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        for event in self._graph.stream(initial_state, config, stream_mode="updates"):
            yield event
    
    def to_qa_response(self, result: LangGraphQAResult) -> QAResponse:
        """
        Convert LangGraphQAResult to QAResponse for API compatibility.
        """
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": doc.metadata.get("score", 0.0),
                "url": doc.metadata.get("url", "")
            }
            for doc in result.documents
        ]
        
        return QAResponse(
            question=result.question,
            answer=result.answer,
            sources=sources,
            confidence=result.confidence if result.confidence else {"score": 0.0, "level": "low", "explanation": ""},
            attributions=result.attributions if result.attributions else [],
            disclaimer=result.disclaimer,
            rationale=result.rationale,
            is_answerable=result.is_answerable,
            from_cache=False
        )
    
    def answer(self, question: str, num_documents: int = None, include_explanation: bool = True) -> QAResponse:
        """
        Answer a question and return QAResponse (compatible with existing pipeline).
        
        Args:
            question: User's medical question
            num_documents: Number of documents to retrieve (overrides self.k if provided)
            include_explanation: Whether to include XAI explanations
            
        Returns:
            QAResponse compatible with existing API
        """
        original_k = self.k
        if num_documents is not None:
            self.k = num_documents
        try:
            result = self.invoke(question)
            return self.to_qa_response(result)
        finally:
            self.k = original_k
    
    def get_graph_visualization(self) -> str:
        """
        Get a Mermaid diagram of the graph for visualization.
        """
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return "Graph visualization not available"


def create_langgraph_pipeline(
    retriever,
    llm,
    confidence_scorer=None,
    source_attributor=None,
    rationale_generator=None,
    **kwargs
) -> LangGraphHealthcareQAPipeline:
    """
    Factory function to create a LangGraph Healthcare QA Pipeline.
    
    Args:
        retriever: HybridRetriever instance
        llm: MedicalLLM instance
        confidence_scorer: Optional ConfidenceScorer
        source_attributor: Optional SourceAttributor
        rationale_generator: Optional RationaleGenerator
        **kwargs: Additional pipeline configuration
        
    Returns:
        Configured LangGraphHealthcareQAPipeline
    """
    return LangGraphHealthcareQAPipeline(
        retriever=retriever,
        llm=llm,
        confidence_scorer=confidence_scorer,
        source_attributor=source_attributor,
        rationale_generator=rationale_generator,
        **kwargs
    )
