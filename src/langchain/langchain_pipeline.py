"""
LangChain LCEL-based RAG Pipeline for Healthcare QA.

This module provides a LangChain-native RAG pipeline that wraps
the existing components (HybridRetriever, MedicalLLM, XAI modules)
using LangChain Expression Language (LCEL) for declarative composition.

Fixes applied (v2):
- _build_safe_context_lc() strips "Question: …" / "[MedQuAD]:" prefixes
  that appear at the start of raw MedQuAD chunks.  These caused TinyLlama
  to treat the source line as a Q&A example and continue that pattern,
  producing answers that were verbatim copies of the wrong document.
- _format_and_generate now builds the prompt using TinyLlama's native
  chat-template tokens (<|system|> / <|user|> / <|assistant|>) instead of
  converting ChatPromptTemplate messages to plain joined text, which had
  stripped all role markers and left the model with no instruction structure.
- clean_llm_response is called on every generated answer.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)
from loguru import logger


# Import LangChain wrappers
from src.langchain.langchain_llm import LangChainMedicalLLMFromExisting
from src.langchain.langchain_prompts import (
    EXPLAINABLE_QA_CHAT_TEMPLATE,
    MEDICAL_DISCLAIMER,
    MEDICAL_QA_CHAT_TEMPLATE,
    SIMPLE_QA_TEMPLATE,
)
from src.langchain.langchain_retriever import (
    LangChainHybridRetriever,
    docs_to_retrieved_documents,
)

# Import existing components
from src.pipeline.qa_pipeline import QAResponse

# ---------------------------------------------------------------------------
# Helper: build a safe context string for LangChain pipeline
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Shared context builder (single source of truth — eliminates duplication
# between the LangChain and LangGraph pipelines).
# ---------------------------------------------------------------------------
from src.utils.context_builder import build_safe_context_lc as _build_safe_context_lc  # noqa: E402

# ---------------------------------------------------------------------------


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
        template_name: str = "medical_qa",
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
        # Adaptive threshold parameters (match standard pipeline)
        self.adaptive_threshold_ratio = 0.5
        self.absolute_score_floor = 0.01  # handles RRF-scale scores ~0.016
        self.confidence_scorer = confidence_scorer
        self.source_attributor = source_attributor
        self.rationale_generator = rationale_generator

        # Cache GPU availability at init (avoids per-call import torch overhead)
        import torch

        self._skip_rationale = not torch.cuda.is_available()

        # Create LangChain wrappers
        self.lc_retriever = LangChainHybridRetriever(hybrid_retriever=retriever, k=k)
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
        chain = retrieve_step | grounding_gate | format_and_generate | xai_enrichment

        return chain

    def _retrieve_documents(self, question: str) -> Dict[str, Any]:
        """Retrieve relevant documents using current k setting."""
        # Update the retriever's k to match pipeline's current k
        self.lc_retriever.k = self.k
        docs = self.lc_retriever.invoke(question)
        return {"question": question, "documents": docs}

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

        # Count relevant documents using adaptive threshold
        # This handles both cosine-similarity scores (~0.3-0.9) AND RRF scores (~0.016)
        top_score = max((doc.metadata.get("score", 0) for doc in docs), default=0)
        adaptive_threshold = max(
            self.absolute_score_floor, self.adaptive_threshold_ratio * top_score
        )

        relevant_docs = [doc for doc in docs if doc.metadata.get("score", 0) >= adaptive_threshold]

        is_answerable = len(relevant_docs) >= self.min_relevant_docs

        state["is_answerable"] = is_answerable
        return state

    def _format_and_generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format context and generate answer."""
        if not state.get("is_answerable", True):
            # Return unanswerable response
            state["answer"] = self.UNANSWERABLE_RESPONSE
            state["context"] = ""
            return state

        # Format documents as context.
        # Use _build_safe_context to strip "Question: …" / "[MedQuAD]:" prefixes
        # that appear at the start of raw MedQuAD chunks — these caused TinyLlama
        # to treat the source line as a Q&A example and continue that pattern.
        docs = state.get("documents", [])
        context = _build_safe_context_lc(docs, max_chars=2000)
        state["context"] = context

        # Build prompt using TinyLlama's native chat-template tokens.
        # Previous code converted ChatPromptTemplate messages to plain text by
        # joining msg.content with "\n", which stripped all role markers
        # (<|system|>, <|user|>, <|assistant|>).  Without those tokens
        # TinyLlama has no instruction structure and simply continues the
        # context verbatim — reproducing MedQuAD headers, wrong Q&A pairs, etc.
        question = state["question"]

        _SYS = "<|system|>"
        _USR = "<|user|>"
        _AST = "<|assistant|>"
        _END = "</s>"

        _SYSTEM_INSTRUCTION = (
            "You are a medical fact extractor. "
            "Read the REFERENCE TEXT and answer the QUESTION.\n"
            "RULES:\n"
            "1. ONLY use facts that appear in the REFERENCE TEXT below.\n"
            "2. Do NOT add information from your own knowledge.\n"
            "3. Do NOT copy document headers, source labels, or question lines from the text.\n"
            "4. Do NOT guess or make up information.\n"
            "5. If the reference text does not contain a clear answer, respond with:\n"
            "   'I do not have enough information in my references to answer this.'\n"
            "6. Keep your answer concise, factual, and patient-friendly."
        )

        prompt_str = (
            f"{_SYS}\n"
            f"{_SYSTEM_INSTRUCTION}\n"
            f"{_END}\n"
            f"{_USR}\n"
            f"REFERENCE TEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            f"{_END}\n"
            f"{_AST}\n"
        )

        # Generate answer via the wrapped MedicalLLM
        raw_answer = self.lc_llm.invoke(prompt_str)

        # Clean: strip leaked training data, MedQuAD artefacts, sign-offs, etc.
        from src.utils.text_cleaning import clean_llm_response

        answer = clean_llm_response(raw_answer).strip()

        # If cleaning left nothing meaningful, return the safe fallback
        if not answer:
            answer = self.UNANSWERABLE_RESPONSE

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
                        num_sources=len(docs),
                    )
                    confidence = {
                        "score": confidence_result.calibrated_score,
                        "level": confidence_result.level,
                        "explanation": confidence_result.explanation,
                    }
                except Exception as e:
                    logger.warning(f"Confidence scoring failed, using default: {e}")
                    confidence = {
                        "score": 0.5,
                        "level": "medium",
                        "explanation": "Confidence scoring unavailable",
                    }

            # Attribute sources
            if self.source_attributor:
                try:
                    doc_dicts = [
                        {
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", ""),
                            "url": doc.metadata.get("url", ""),
                        }
                        for doc in docs
                    ]
                    attributions_list = self.source_attributor.attribute_answer(answer, doc_dicts)
                    attributions = [
                        {
                            "claim": a.claim,
                            "source": a.source,
                            "evidence": a.evidence,
                            "similarity": a.similarity_score,
                        }
                        for a in attributions_list
                    ]
                except Exception as e:
                    logger.warning(f"Source attribution failed, returning empty list: {e}")
                    attributions = []

            # Generate rationale
            # Rationale requires a full extra LLM call (~2-4 min on CPU).
            # Only run it when GPU is available to keep latency reasonable.
            if self.rationale_generator and not self._skip_rationale:
                try:
                    rationale = self.rationale_generator.generate_rationale(
                        question=question, answer=answer, context=context
                    )
                except Exception as e:
                    logger.warning(f"Rationale generation failed, skipping: {e}")
                    rationale = None

        if not is_answerable:
            confidence = {
                "score": 0.0,
                "level": "low",
                "explanation": "Insufficient relevant context found in knowledge base",
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
            disclaimer=MEDICAL_DISCLAIMER,
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
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "score": doc.metadata.get("score", 0.0),
                "url": doc.metadata.get("url", ""),
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
            from_cache=False,
        )

    def answer(
        self,
        question: str,
        num_documents: Optional[int] = None,
        include_explanation: bool = True,
    ) -> QAResponse:
        """
        Answer a question and return QAResponse (compatible with existing pipeline).

        Args:
            question: User's medical question
            num_documents: Number of documents to retrieve (overrides self.k if provided)
            include_explanation: Whether to include XAI explanations

        Returns:
            QAResponse compatible with existing API
        """
        # Temporarily override k if caller specifies num_documents
        original_k = self.k
        if num_documents is not None:
            self.k = num_documents
        try:
            result = self.invoke(question)
            return self.to_qa_response(result)
        finally:
            self.k = original_k


def create_langchain_pipeline(
    retriever,
    llm,
    confidence_scorer=None,
    source_attributor=None,
    rationale_generator=None,
    **kwargs,
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
        **kwargs,
    )
