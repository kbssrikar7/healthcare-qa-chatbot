"""
Main QA pipeline orchestrating all components.

Enhanced with:
- Adaptive grounding gate (hybrid absolute + relative threshold)
- Optional query enhancement (pre-retrieval)
- Optional corrective RAG (post-retrieval quality check)
- Optional context compression (before generation)
- Optional factual consistency check (post-generation)
- Cache context key including model/pipeline/KB fingerprint
"""
from loguru import logger
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import asyncio
import time

# Import configuration
try:
    from config.settings import config
except ImportError:
    config = None

# Import cache manager
try:
    from src.utils.cache_manager import CacheManager
except ImportError:
    CacheManager = None

try:
    from src.xai.rationale_generator import RationaleGenerator
except ImportError:
    RationaleGenerator = None

try:
    from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer
except ImportError:
    MultiSignalConfidenceScorer = None

try:
    from src.xai.hallucination_detector import HallucinationDetector
except ImportError:
    HallucinationDetector = None


@dataclass
class QAResponse:
    """Complete response from the QA pipeline."""
    question: str
    answer: str
    sources: List[Dict]
    confidence: Dict
    attributions: List[Dict]
    disclaimer: str
    rationale: Optional[str] = None
    is_answerable: bool = True
    from_cache: bool = False
    factual_consistency: Optional[Dict] = None
    confidence_breakdown: Optional[Dict] = None  # 5-signal XAI breakdown
    hallucination: Optional[Dict] = None          # NLI-based hallucination result
    stage_latencies: Optional[Dict[str, float]] = None  # per-stage wall-clock ms


class HealthcareQAPipeline:
    """
    Main pipeline orchestrating retrieval, generation, and XAI.
    
    Stages (each skippable when component is None):
      1. Query enhancement (pre-retrieval)
      2. Retrieval (+ reranking inside HybridRetriever)
      3. Corrective RAG (post-retrieval quality check)
      4. Grounding gate (adaptive threshold)
      5. Context compression (before generation)
      6. LLM generation
      7. Factual consistency check (post-generation)
      8. Source attribution & confidence scoring
      9. Cache response
    """
    
    UNANSWERABLE_RESPONSE = (
        "I don't have enough information in my knowledge base to answer this question accurately. "
        "Please consult a healthcare professional for specific medical advice."
    )
    
    def __init__(
        self,
        retriever,
        llm,
        prompt_manager,
        confidence_scorer=None,
        source_attributor=None,
        rationale_generator=None,
        enable_grounding_gate: bool = None,
        cache_manager: Optional["CacheManager"] = None,
        # Enhanced pipeline components (all optional)
        query_enhancer=None,
        context_compressor=None,
        corrective_rag=None,
        factual_consistency_checker=None,
        # Configurable thresholds (override config if provided)
        min_retrieval_score: float = None,
        min_relevant_docs: int = None,
        # Cache context key for model/pipeline-aware caching
        cache_context_key: str = "",
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.confidence_scorer = confidence_scorer
        self.source_attributor = source_attributor
        self.rationale_generator = rationale_generator
        
        # MCP Configuration
        self.enable_mcp_search = False
        self.mcp_search_cmd = "npx"
        self.mcp_search_args = "-y @modelcontextprotocol/server-brave-search"
        if config and hasattr(config, "pipeline"):
            self.enable_mcp_search = getattr(config.pipeline, "enable_mcp_search", False)
            self.mcp_search_cmd = getattr(config.pipeline, "mcp_search_cmd", "npx")
            self.mcp_search_args = getattr(config.pipeline, "mcp_search_args", "-y @modelcontextprotocol/server-brave-search")
        
        # Enhanced pipeline components
        self.query_enhancer = query_enhancer
        self.context_compressor = context_compressor
        self.corrective_rag = corrective_rag
        self.factual_consistency_checker = factual_consistency_checker
        self.cache_context_key = cache_context_key
        
        # Initialize rationale generator if not provided but class is available
        if self.rationale_generator is None and RationaleGenerator is not None and self.llm:
            self.rationale_generator = RationaleGenerator(self.llm)

        # Multi-signal confidence scorer (5-signal XAI breakdown)
        self.multi_signal_scorer = MultiSignalConfidenceScorer() if MultiSignalConfidenceScorer else None

        # Hallucination detector with DeBERTa NLI enabled (model cached locally)
        self.hallucination_detector = HallucinationDetector(use_nli=True) if HallucinationDetector else None

        # Load from config with fallbacks
        pipeline_config = getattr(config, 'pipeline', None) if config else None
        
        self.enable_grounding_gate = enable_grounding_gate if enable_grounding_gate is not None else (
            getattr(pipeline_config, 'enable_grounding_gate', True) if pipeline_config else True
        )
        self.min_retrieval_score = min_retrieval_score if min_retrieval_score is not None else (
            getattr(pipeline_config, 'min_retrieval_score', 0.3) if pipeline_config else 0.3
        )
        self.min_relevant_docs = min_relevant_docs if min_relevant_docs is not None else (
            getattr(pipeline_config, 'min_relevant_docs', 1) if pipeline_config else 1
        )
        # Adaptive thresholds
        self.adaptive_threshold_ratio = getattr(pipeline_config, 'adaptive_threshold_ratio', 0.5) if pipeline_config else 0.5
        self.absolute_score_floor = getattr(pipeline_config, 'absolute_score_floor', 0.01) if pipeline_config else 0.01
        
        # Initialize cache manager
        self.cache_manager = cache_manager
        if self.cache_manager is None and CacheManager is not None:
            if pipeline_config and getattr(pipeline_config, 'enable_response_cache', False):
                self.cache_manager = CacheManager(
                    cache_dir=getattr(pipeline_config, 'cache_dir', 'data/cache'),
                    ttl_seconds=getattr(pipeline_config, 'cache_ttl_seconds', 3600),
                    max_memory_items=getattr(pipeline_config, 'max_cache_items', 1000)
                )
    
    def answer(
        self,
        question: str,
        num_documents: int = 5,
        include_explanation: bool = True,
        template_name: str = "medical_qa",
        conversation_context: str = None
    ) -> QAResponse:
        """
        Answer a medical question with explanations.
        
        Full pipeline: enhance → retrieve → correct → gate → compress → generate → verify.
        
        Args:
            question: The user's question.
            num_documents: Number of documents to retrieve.
            include_explanation: Whether to include XAI explanations.
            template_name: Prompt template name.
            conversation_context: Optional previous conversation context for follow-ups.
        """
        # Build dynamic context key for cache (combining static pipeline info + per-request flags + conversation context)
        # Include conversation context to prevent stale answers across sessions
        context_hash = ""
        if conversation_context:
            import hashlib
            context_hash = "_ctx" + hashlib.md5(conversation_context.encode()).hexdigest()[:8]
        dynamic_context_key = f"{self.cache_context_key}_srcs{num_documents}_expl{include_explanation}_{template_name}{context_hash}"

        # Latency tracking dict (wall-clock milliseconds per stage)
        _t = {}
        _t0 = time.perf_counter()

        # 0. Check cache first (with context key for model/pipeline awareness)
        if self.cache_manager:
            cached = self.cache_manager.get_cached_response(
                question, context_key=dynamic_context_key
            )
            if cached:
                return QAResponse(
                    question=cached.get('question', question),
                    answer=cached.get('answer', ''),
                    sources=cached.get('sources', []),
                    confidence=cached.get('confidence', {}),
                    attributions=cached.get('attributions', []),
                    disclaimer=cached.get('disclaimer', ''),
                    rationale=cached.get('rationale', None),
                    is_answerable=cached.get('is_answerable', True),
                    from_cache=True,
                    factual_consistency=cached.get('factual_consistency', None),
                )
        
        # 1. QUERY ENHANCEMENT (pre-retrieval)
        _s = time.perf_counter()
        retrieval_query = question

        # Augment with conversation context for follow-ups
        if conversation_context:
            retrieval_query = f"{conversation_context}\n\nCurrent question: {question}"

        if self.query_enhancer:
            try:
                enhanced = self.query_enhancer.enhance(question)
                # enhance() returns an EnhancedQuery dataclass with .all_queries
                if hasattr(enhanced, 'all_queries') and enhanced.all_queries:
                    retrieval_query = enhanced.all_queries[0]
                elif isinstance(enhanced, str) and enhanced:
                    retrieval_query = enhanced
            except Exception as e:
                logger.warning(f"Query enhancement failed, using original: {e}")
        _t["query_enhancement_ms"] = (time.perf_counter() - _s) * 1000

        # 2. RETRIEVE relevant documents
        _s = time.perf_counter()
        documents, context = self.retriever.retrieve_with_context(
            retrieval_query,
            k=num_documents
        )
        _t["retrieval_ms"] = (time.perf_counter() - _s) * 1000
        
        # 3. CORRECTIVE RAG (post-retrieval quality check)
        _s = time.perf_counter()
        if self.corrective_rag and documents:
            try:
                corrected_result = self.corrective_rag.retrieve_with_correction(
                    query=retrieval_query,
                    k=num_documents,
                )
                # retrieve_with_correction returns Tuple[List, bool]
                corrected_docs, was_corrected = corrected_result
                if corrected_docs and len(corrected_docs) > 0:
                    documents = corrected_docs
                    # Rebuild context from corrected docs
                    context_parts = []
                    total_length = 0
                    for i, doc in enumerate(documents, 1):
                        if total_length + len(doc.content) > 2000:
                            break
                        context_parts.append(f"[{i}] Source: {doc.source}\n{doc.content}")
                        total_length += len(doc.content)
                    context = "\n\n".join(context_parts)
            except Exception as e:
                logger.warning(f"Corrective RAG failed, using original retrieval: {e}")
        _t["corrective_rag_ms"] = (time.perf_counter() - _s) * 1000

        # 4. GROUNDING GATE: adaptive check
        is_answerable = True
        if self.enable_grounding_gate:
            is_answerable = self._check_answerability(documents)
        
        if not is_answerable:
            # Fallback to MCP Search if enabled
            if self.enable_mcp_search:
                logger.info("Local knowledge insufficient. Falling back to MCP Web Search...")
                try:
                    from src.mcp_client.agent import execute_mcp_tool_oneshot
                    
                    mcp_args = self.mcp_search_args.split(" ")
                    coro = execute_mcp_tool_oneshot(
                        server_cmd=self.mcp_search_cmd,
                        server_args=mcp_args,
                        tool_name="brave_web_search",
                        tool_args={"query": question, "count": 3}
                    )
                    
                    # Safe async execution: handle both sync and async calling contexts.
                    # asyncio.run() crashes inside FastAPI because the event loop is already running.
                    try:
                        loop = asyncio.get_running_loop()
                        # We're inside an async context (e.g. FastAPI) — use a new thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            mcp_result = pool.submit(asyncio.run, coro).result(timeout=30)
                    except RuntimeError:
                        # No event loop running — safe to use asyncio.run()
                        mcp_result = asyncio.run(coro)
                    
                    if mcp_result and not mcp_result.startswith("[MCP Error"):
                        context = f"CONTEXT FROM LIVE WEB SEARCH:\n{mcp_result}"
                        is_answerable = True
                        # Build a complete document object that matches the interface
                        # downstream code expects: .source, .content, .score, .metadata
                        mcp_doc = type('MCPDocument', (object,), {
                            'source': 'MCP Web Search',
                            'content': context[:500],
                            'score': 0.5,  # lower than KB docs — web results are less curated
                            'metadata': {'url': '', 'source': 'MCP Web Search', 'source_type': 'mcp_web_search'}
                        })()
                        documents = [mcp_doc]
                        logger.info("MCP Web Search succeeded")
                    else:
                        logger.warning(f"MCP Web Search failed or returned nothing: {mcp_result}")
                except Exception as e:
                    logger.warning(f"MCP Web Search exception: {e}")
                    
            # If still not answerable after MCP fallback attempt
            if not is_answerable:
                disclaimer = self.prompt_manager.get_medical_disclaimer()
                return QAResponse(
                    question=question,
                    answer=self.UNANSWERABLE_RESPONSE,
                    sources=[],
                    confidence={
                        "score": 0.0,
                        "level": "low",
                        "explanation": "Insufficient relevant context found in knowledge base and web search"
                    },
                    attributions=[],
                    disclaimer=disclaimer,
                    rationale=None,
                    is_answerable=False
                )
        
        # 5. CONTEXT COMPRESSION (before generation)
        generation_context = context
        if self.context_compressor:
            try:
                # ContextCompressor.compress() takes (documents, query) positionally
                # and returns a CompressedContext dataclass with .text
                compressed = self.context_compressor.compress(documents, question)
                if compressed and compressed.text:
                    generation_context = compressed.text
                    logger.info(
                        f"Context compressed: {compressed.original_length} → {compressed.compressed_length} chars "
                        f"(ratio: {compressed.compression_ratio:.2f})"
                    )
            except Exception as e:
                logger.warning(f"Context compression failed, using full context: {e}")
        
        # 6. BUILD PROMPT & GENERATE ANSWER
        _s = time.perf_counter()
        prompt = self.prompt_manager.build_prompt(
            question=question,
            context=generation_context,
            template_name=template_name
        )
        
        generation_result = self.llm.generate(
            prompt,
            max_new_tokens=256,
            return_probabilities=include_explanation
        )
        
        answer = self._clean_answer(generation_result.response)
        _t["generation_ms"] = (time.perf_counter() - _s) * 1000

        # 7. FACTUAL CONSISTENCY CHECK (post-generation)
        factual_result = None
        if self.factual_consistency_checker and answer:
            try:
                fc = self.factual_consistency_checker.check_consistency(
                    answer=answer,
                    context=context
                )
                factual_result = {
                    "is_consistent": fc.is_consistent if hasattr(fc, 'is_consistent') else True,
                    "score": fc.consistency_score if hasattr(fc, 'consistency_score') else 1.0,
                    "details": [vars(c) if hasattr(c, '__dict__') else c for c in fc.claim_results] if hasattr(fc, 'claim_results') else [],
                }
            except Exception as e:
                logger.warning(f" Factual consistency check failed: {e}")
        
        # 7b. HALLUCINATION DETECTION (DeBERTa NLI + rule-based)
        _s = time.perf_counter()
        hallucination_result = None
        if self.hallucination_detector and include_explanation and answer:
            try:
                doc_dicts_for_hal = [
                    {"content": doc.content} for doc in documents
                ]
                hal = self.hallucination_detector.detect(
                    answer=answer,
                    retrieved_documents=doc_dicts_for_hal,
                    query=question,
                )
                hallucination_result = {
                    "has_hallucination": hal.has_hallucination,
                    "type": hal.hallucination_type.value,
                    "score": hal.hallucination_score,
                    "flagged_claims": hal.flagged_claims,
                    "medical_accuracy_flags": hal.medical_accuracy_flags,
                    "explanation": hal.explanation,
                }
            except Exception as e:
                logger.warning(f"Hallucination detection failed: {e}")
        _t["hallucination_ms"] = (time.perf_counter() - _s) * 1000

        # 8. CONFIDENCE SCORING — multi-signal (5-signal XAI breakdown)
        _s = time.perf_counter()
        confidence_breakdown = None
        if self.multi_signal_scorer and include_explanation:
            try:
                # RRF scores live in 0.01–0.04 range but MultiSignalConfidenceScorer
                # expects cosine-similarity-like 0–1 values.  Normalize by max score
                # so the top doc always maps to 1.0 and others scale proportionally.
                raw_scores = [doc.score for doc in documents]
                max_score = max(raw_scores) if raw_scores else 1.0
                if max_score > 0:
                    for doc in documents:
                        doc.score = doc.score / max_score

                bd = self.multi_signal_scorer.compute_confidence(
                    query=question,
                    answer=answer,
                    retrieved_documents=documents,
                    generation_probabilities=generation_result.probabilities,
                )

                # Restore original scores so source list stays accurate
                for doc, orig in zip(documents, raw_scores):
                    doc.score = orig
                confidence = {
                    "score": bd.calibrated_confidence,
                    "level": bd.confidence_level,
                    "explanation": bd.explanation,
                }
                confidence_breakdown = {
                    "retrieval_confidence": bd.retrieval_confidence,
                    "generation_confidence": bd.generation_confidence,
                    "consistency_score": bd.consistency_score,
                    "source_agreement": bd.source_agreement,
                    "medical_entity_coverage": bd.medical_entity_coverage,
                    "signal_weights": bd.signal_weights,
                }
            except Exception as e:
                logger.warning(f"Multi-signal confidence scoring failed: {e}")
                confidence_breakdown = None

        # Fallback to basic confidence scorer if multi-signal failed or unavailable
        if confidence_breakdown is None:
            if self.confidence_scorer and include_explanation:
                retrieval_scores = [doc.score for doc in documents]
                confidence_result = self.confidence_scorer.calculate_confidence(
                    generation_probs=generation_result.probabilities,
                    retrieval_scores=retrieval_scores,
                    num_sources=len(documents)
                )
                confidence = {
                    "score": confidence_result.calibrated_score,
                    "level": confidence_result.level,
                    "explanation": confidence_result.explanation,
                }
            else:
                confidence = {
                    "score": 0.7,
                    "level": "medium",
                    "explanation": "Confidence scoring not available",
                }
        
        _t["confidence_ms"] = (time.perf_counter() - _s) * 1000

        # 9. SOURCE ATTRIBUTION
        if self.source_attributor and include_explanation:
            doc_dicts = [
                {"content": doc.content, "source": doc.source, "url": doc.metadata.get("url", "")}
                for doc in documents
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
        else:
            attributions = []
        
        # 10. RATIONALE
        rationale = None
        if self.rationale_generator and include_explanation:
            combined_context = "\n".join([d.content for d in documents])
            rationale = self.rationale_generator.generate_rationale(
                question=question,
                answer=answer,
                context=combined_context
            )

        # 11. BUILD SOURCE LIST
        sources = [
            {
                "source": doc.source,
                "content": doc.content,
                "score": doc.score,
                "url": doc.metadata.get("url", "")
            }
            for doc in documents
        ]
        
        disclaimer = self.prompt_manager.get_medical_disclaimer()

        _t["total_ms"] = (time.perf_counter() - _t0) * 1000

        response = QAResponse(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            attributions=attributions,
            disclaimer=disclaimer,
            rationale=rationale,
            is_answerable=True,
            from_cache=False,
            factual_consistency=factual_result,
            confidence_breakdown=confidence_breakdown,
            hallucination=hallucination_result,
            stage_latencies=_t,
        )
        
        # 12. CACHE THE RESPONSE (with context key)
        if self.cache_manager:
            self.cache_manager.cache_response(
                question,
                {
                    'question': question,
                    'answer': answer,
                    'sources': sources,
                    'confidence': confidence,
                    'attributions': attributions,
                    'disclaimer': disclaimer,
                    'rationale': rationale,
                    'is_answerable': True,
                    'factual_consistency': factual_result,
                    'confidence_breakdown': confidence_breakdown,
                    'hallucination': hallucination_result,
                },
                context_key=dynamic_context_key,
            )
        
        return response
    
    def _check_answerability(self, documents: List) -> bool:
        """
        Adaptive grounding gate: doc is relevant if
            score >= max(absolute_score_floor, adaptive_ratio * top_score)
        
        This handles both cosine-similarity scores (0–1) and RRF scores (~0.016)
        without the static threshold problem.
        """
        if not documents:
            return False
        
        top_score = max(doc.score for doc in documents)
        
        # Hybrid threshold: whichever is larger wins
        threshold = max(
            self.absolute_score_floor,
            self.adaptive_threshold_ratio * top_score,
        )
        
        relevant_docs = [doc for doc in documents if doc.score >= threshold]
        
        if len(relevant_docs) < self.min_relevant_docs:
            return False
        
        return True
    
    def _clean_answer(self, answer: str) -> str:
        """
        Pipeline-level answer cleaning to strip training data artifacts.
        Delegates to the shared text cleaning utility.
        """
        from src.utils.text_cleaning import clean_llm_response
        return clean_llm_response(answer)
    
    def batch_answer(
        self,
        questions: List[str],
        **kwargs
    ) -> List[QAResponse]:
        """Answer multiple questions."""
        return [self.answer(q, **kwargs) for q in questions]
