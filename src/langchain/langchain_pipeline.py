"""
LangChain LCEL-based RAG Pipeline for Healthcare QA.

This module provides a LangChain-native RAG pipeline that wraps
the existing components (HybridRetriever, MedicalLLM, XAI modules)
using LangChain Expression Language (LCEL) for declarative composition.
"""
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
    Runnable
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import LangChain wrappers
from src.langchain.langchain_llm import LangChainMedicalLLM, LangChainMedicalLLMFromExisting
from src.langchain.langchain_retriever import (
    LangChainHybridRetriever,
    format_docs_as_context,
    docs_to_retrieved_documents
)
from src.langchain.langchain_prompts import (
    MEDICAL_QA_CHAT_TEMPLATE,
    EXPLAINABLE_QA_CHAT_TEMPLATE,
    SIMPLE_QA_TEMPLATE,
    MEDICAL_DISCLAIMER,
    format_context_for_prompt
)

# Import existing components
from src.pipeline.qa_pipeline import QAResponse


@dataclass
class LangChainQAResult:
    """Result from LangChain QA pipeline."""
    question: str
    answer: str
    documents: List[Document]
    context: str
    is_answerable: bool = True
    confidence: Optional[Dict] = None
    attributions: Optional[List[Dict]] = None
    rationale: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER


class LangChainHealthcareQAPipeline:
    """
    LangChain LCEL-based Healthcare QA Pipeline.
    
    Uses LangChain Expression Language for declarative pipeline composition
    while leveraging existing components (HybridRetriever, MedicalLLM, XAI).
    
    Example:
        pipeline = LangChainHealthcareQAPipeline(
            retriever=hybrid_retriever,
            llm=medical_llm
        )
        result = pipeline.invoke("What are the symptoms of diabetes?")
    """
    
    UNANSWERABLE_RESPONSE = (
        "I don't have enough information in my knowledge base to answer this question accurately. "
        "Please consult a healthcare professional for specific medical advice."
    )
    
    def __init__(
        self,
        retriever,
        llm,
        confidence_scorer=None,
        source_attributor=None,
        rationale_generator=None,
        enable_grounding_gate: bool = True,
        min_retrieval_score: float = 0.3,
        min_relevant_docs: int = 1,
        k: int = 5,
        template_name: str = "medical_qa"
    ):
        """
        Initialize LangChain Healthcare QA Pipeline.
        
        Args:
            retriever: HybridRetriever instance
            llm: MedicalLLM instance
            confidence_scorer: Optional ConfidenceScorer
            source_attributor: Optional SourceAttributor
            rationale_generator: Optional RationaleGenerator
            enable_grounding_gate: Whether to check answerability
            min_retrieval_score: Minimum score for relevant documents
            min_relevant_docs: Minimum number of relevant documents needed
            k: Number of documents to retrieve
            template_name: Prompt template name (medical_qa, explainable, simple)
        """
        self.k = k
        self.enable_grounding_gate = enable_grounding_gate
        self.min_retrieval_score = min_retrieval_score
        self.min_relevant_docs = min_relevant_docs
        self.confidence_scorer = confidence_scorer
        self.source_attributor = source_attributor
        self.rationale_generator = rationale_generator
        
        # Create LangChain wrappers
        self.lc_retriever = LangChainHybridRetriever(
            hybrid_retriever=retriever,
            k=k
        )
        self.lc_llm = LangChainMedicalLLMFromExisting(llm=llm)
        
        # Select prompt template
        if template_name == "explainable":
            self.prompt = EXPLAINABLE_QA_CHAT_TEMPLATE
        elif template_name == "simple":
            self.prompt = SIMPLE_QA_TEMPLATE
        else:
            self.prompt = MEDICAL_QA_CHAT_TEMPLATE
        
        # Build the LCEL chain
        self._chain = self._build_chain()
    
    def _build_chain(self) -> Runnable:
        """
        Build the LCEL pipeline chain.
        
        Pipeline structure:
        1. Retrieve documents
        2. Check answerability (grounding gate)
        3. Format context
        4. Generate answer
        5. Enrich with XAI components
        """
        # Step 1: Retrieve documents
        retrieve_step = RunnableLambda(self._retrieve_documents)
        
        # Step 2: Grounding gate (check if answerable)
        grounding_gate = RunnableLambda(self._check_answerability)
        
        # Step 3: Format context + generate
        format_and_generate = RunnableLambda(self._format_and_generate)
        
        # Step 4: Enrich with XAI
        xai_enrichment = RunnableLambda(self._enrich_with_xai)
        
        # Full chain
        chain = (
            retrieve_step
            | grounding_gate
            | format_and_generate
            | xai_enrichment
        )
        
        return chain
    
    def _retrieve_documents(self, question: str) -> Dict[str, Any]:
        """Retrieve relevant documents."""
        docs = self.lc_retriever.invoke(question)
        return {
            "question": question,
            "documents": docs
        }
    
    def _check_answerability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the question can be answered from retrieved documents.
        
        Implements grounding gate based on document relevance scores.
        """
        if not self.enable_grounding_gate:
            state["is_answerable"] = True
            return state
        
        docs = state.get("documents", [])
        
        if not docs:
            state["is_answerable"] = False
            return state
        
        # Count relevant documents
        relevant_docs = [
            doc for doc in docs
            if doc.metadata.get("score", 0) >= self.min_retrieval_score
        ]
        
        is_answerable = len(relevant_docs) >= self.min_relevant_docs
        
        # Also check top document score
        if docs:
            top_score = max(doc.metadata.get("score", 0) for doc in docs)
            if top_score < self.min_retrieval_score:
                is_answerable = False
        
        state["is_answerable"] = is_answerable
        return state
    
    def _format_and_generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format context and generate answer."""
        if not state.get("is_answerable", True):
            # Return unanswerable response
            state["answer"] = self.UNANSWERABLE_RESPONSE
            state["context"] = ""
            return state
        
        # Format documents as context
        docs = state.get("documents", [])
        context = format_docs_as_context(docs, max_length=2000)
        state["context"] = context
        
        # Build prompt and generate
        question = state["question"]
        
        # For ChatPromptTemplate
        if hasattr(self.prompt, 'invoke'):
            formatted_prompt = self.prompt.invoke({
                "context": context,
                "question": question
            })
            # Convert to string for LLM
            if hasattr(formatted_prompt, 'to_string'):
                prompt_str = formatted_prompt.to_string()
            else:
                # Handle ChatPromptValue
                messages = formatted_prompt.to_messages()
                prompt_str = "\n".join(msg.content for msg in messages)
        else:
            # For simple PromptTemplate
            prompt_str = self.prompt.format(context=context, question=question)
        
        # Generate answer
        answer = self.lc_llm.invoke(prompt_str)
        state["answer"] = answer
        
        return state
    
    def _enrich_with_xai(self, state: Dict[str, Any]) -> LangChainQAResult:
        """Enrich response with XAI components."""
        question = state["question"]
        answer = state.get("answer", "")
        docs = state.get("documents", [])
        context = state.get("context", "")
        is_answerable = state.get("is_answerable", True)
        
        # Default values
        confidence = None
        attributions = None
        rationale = None
        
        if is_answerable and docs:
            # Calculate confidence
            if self.confidence_scorer:
                try:
                    retrieved_docs = docs_to_retrieved_documents(docs)
                    retrieval_scores = [doc.score for doc in retrieved_docs]
                    confidence_result = self.confidence_scorer.calculate_confidence(
                        generation_probs=None,  # Not available in this flow
                        retrieval_scores=retrieval_scores,
                        num_sources=len(docs)
                    )
                    confidence = {
                        "score": confidence_result.calibrated_score,
                        "level": confidence_result.level,
                        "explanation": confidence_result.explanation
                    }
                except Exception:
                    confidence = {"score": 0.7, "level": "medium", "explanation": "Confidence scoring unavailable"}
            
            # Attribute sources
            if self.source_attributor:
                try:
                    doc_dicts = [
                        {"content": doc.page_content, "source": doc.metadata.get("source", ""), "url": doc.metadata.get("url", "")}
                        for doc in docs
                    ]
                    attributions_list = self.source_attributor.attribute_answer(answer, doc_dicts)
                    attributions = [
                        {
                            "claim": a.claim,
                            "source": a.source,
                            "evidence": a.evidence,
                            "similarity": a.similarity_score
                        }
                        for a in attributions_list
                    ]
                except Exception:
                    attributions = []
            
            # Generate rationale
            if self.rationale_generator:
                try:
                    rationale = self.rationale_generator.generate_rationale(
                        question=question,
                        answer=answer,
                        context=context
                    )
                except Exception:
                    rationale = None
        
        if not is_answerable:
            confidence = {
                "score": 0.0,
                "level": "low",
                "explanation": "Insufficient relevant context found in knowledge base"
            }
        
        return LangChainQAResult(
            question=question,
            answer=answer,
            documents=docs,
            context=context,
            is_answerable=is_answerable,
            confidence=confidence,
            attributions=attributions,
            rationale=rationale,
            disclaimer=MEDICAL_DISCLAIMER
        )
    
    def invoke(self, question: str) -> LangChainQAResult:
        """
        Answer a medical question.
        
        Args:
            question: User's medical question
            
        Returns:
            LangChainQAResult with answer and metadata
        """
        return self._chain.invoke(question)
    
    async def ainvoke(self, question: str) -> LangChainQAResult:
        """Async version of invoke."""
        return await self._chain.ainvoke(question)
    
    def to_qa_response(self, result: LangChainQAResult) -> QAResponse:
        """
        Convert LangChainQAResult to QAResponse for API compatibility.
        
        Args:
            result: LangChainQAResult from invoke()
            
        Returns:
            QAResponse compatible with existing API
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
            confidence=result.confidence or {"score": 0.7, "level": "medium", "explanation": ""},
            attributions=result.attributions or [],
            disclaimer=result.disclaimer,
            rationale=result.rationale,
            is_answerable=result.is_answerable,
            from_cache=False
        )
    
    def answer(self, question: str) -> QAResponse:
        """
        Answer a question and return QAResponse (compatible with existing pipeline).
        
        Args:
            question: User's medical question
            
        Returns:
            QAResponse compatible with existing API
        """
        result = self.invoke(question)
        return self.to_qa_response(result)


def create_langchain_pipeline(
    retriever,
    llm,
    confidence_scorer=None,
    source_attributor=None,
    rationale_generator=None,
    **kwargs
) -> LangChainHealthcareQAPipeline:
    """
    Factory function to create a LangChain Healthcare QA Pipeline.
    
    Args:
        retriever: HybridRetriever instance
        llm: MedicalLLM instance
        confidence_scorer: Optional ConfidenceScorer
        source_attributor: Optional SourceAttributor
        rationale_generator: Optional RationaleGenerator
        **kwargs: Additional pipeline configuration
        
    Returns:
        Configured LangChainHealthcareQAPipeline
    """
    return LangChainHealthcareQAPipeline(
        retriever=retriever,
        llm=llm,
        confidence_scorer=confidence_scorer,
        source_attributor=source_attributor,
        rationale_generator=rationale_generator,
        **kwargs
    )
