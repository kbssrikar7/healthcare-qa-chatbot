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

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.stopwords import ENGLISH_STOPWORDS as _PIPELINE_STOPWORDS

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

try:
    from src.utils.context_builder import build_safe_context
except ImportError:
    build_safe_context = None  # type: ignore[assignment]


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
    hallucination: Optional[Dict] = None  # NLI-based hallucination result
    stage_latencies: Optional[Dict[str, float]] = None  # per-stage wall-clock ms
    generation_backend_used: Optional[str] = None  # "ollama", "extractive", "extractive_fallback"


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
    _MED_BRAND_TO_GENERIC = {
        # Analgesics / antipyretics
        "dolo": ["paracetamol", "acetaminophen"],
        "dolo 650": ["paracetamol", "acetaminophen"],
        "crocin": ["paracetamol", "acetaminophen"],
        "tylenol": ["paracetamol", "acetaminophen"],
        "brufen": ["ibuprofen"],
        "advil": ["ibuprofen"],
        "nurofen": ["ibuprofen"],
        "voveran": ["diclofenac"],
        # Antibiotics
        "augmentin": ["amoxicillin", "clavulanate"],
        "zithromax": ["azithromycin"],
        "azee": ["azithromycin"],
        "cipro": ["ciprofloxacin"],
        "cifran": ["ciprofloxacin"],
        # Cardiovascular
        "ecosprin": ["aspirin", "acetylsalicylic acid"],
        "disprin": ["aspirin"],
        "lipitor": ["atorvastatin"],
        "crestor": ["rosuvastatin"],
        "norvasc": ["amlodipine"],
        "amlokind": ["amlodipine"],
        # Antidiabetics
        "glucophage": ["metformin"],
        "glycomet": ["metformin"],
        "januvia": ["sitagliptin"],
        # GI / acid
        "nexium": ["esomeprazole"],
        "omez": ["omeprazole"],
        "pan": ["pantoprazole"],
        "pantop": ["pantoprazole"],
        # Respiratory
        "asthalin": ["salbutamol", "albuterol"],
        "ventolin": ["salbutamol", "albuterol"],
        # Psychiatric
        "xanax": ["alprazolam"],
        "restyl": ["alprazolam"],
    }

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
            self.mcp_search_args = getattr(
                config.pipeline,
                "mcp_search_args",
                "-y @modelcontextprotocol/server-brave-search",
            )

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
        # Load fitted Platt calibration params from evaluation/results/calibration.json
        _platt_params = None
        try:
            import json as _json

            _cal_path = (
                Path(__file__).parent.parent.parent / "evaluation" / "results" / "calibration.json"
            )
            if _cal_path.exists():
                _cal = _json.loads(_cal_path.read_text())
                if "platt_a" in _cal and "platt_b" in _cal:
                    _platt_params = {
                        "a": float(_cal["platt_a"]),
                        "b": float(_cal["platt_b"]),
                    }
                    logger.info(
                        f"Loaded Platt calibration: a={_platt_params['a']:.3f}, b={_platt_params['b']:.3f}"
                    )
        except Exception as e:
            logger.warning(f"Could not load calibration params, using defaults: {e}")
        self.multi_signal_scorer = (
            MultiSignalConfidenceScorer(calibration_params=_platt_params)
            if MultiSignalConfidenceScorer
            else None
        )

        # Hallucination detector with DeBERTa NLI enabled (model cached locally)
        self.hallucination_detector = (
            HallucinationDetector(use_nli=True) if HallucinationDetector else None
        )

        # Load from config with fallbacks
        pipeline_config = getattr(config, "pipeline", None) if config else None

        self.enable_grounding_gate = (
            enable_grounding_gate
            if enable_grounding_gate is not None
            else (
                getattr(pipeline_config, "enable_grounding_gate", True) if pipeline_config else True
            )
        )
        self.min_retrieval_score = (
            min_retrieval_score
            if min_retrieval_score is not None
            else (getattr(pipeline_config, "min_retrieval_score", 0.3) if pipeline_config else 0.3)
        )
        self.min_relevant_docs = (
            min_relevant_docs
            if min_relevant_docs is not None
            else (getattr(pipeline_config, "min_relevant_docs", 1) if pipeline_config else 1)
        )
        # Adaptive thresholds
        self.adaptive_threshold_ratio = (
            getattr(pipeline_config, "adaptive_threshold_ratio", 0.5) if pipeline_config else 0.5
        )
        self.absolute_score_floor = (
            getattr(pipeline_config, "absolute_score_floor", 0.01) if pipeline_config else 0.01
        )
        self.enable_multi_query_fusion = (
            getattr(pipeline_config, "enable_multi_query_fusion", True)
            if pipeline_config
            else True
        )
        self.grounding_min_term_overlap = (
            getattr(pipeline_config, "grounding_min_term_overlap", 0.3)
            if pipeline_config
            else 0.3
        )

        # Post-generation quality gate threshold
        self.min_answer_confidence = float(
            getattr(pipeline_config, "min_answer_confidence", 0.525) if pipeline_config else 0.525
        )

        # Initialize cache manager
        self.cache_manager = cache_manager
        if self.cache_manager is None and CacheManager is not None:
            if pipeline_config and getattr(pipeline_config, "enable_response_cache", False):
                self.cache_manager = CacheManager(
                    cache_dir=getattr(pipeline_config, "cache_dir", "data/cache"),
                    ttl_seconds=getattr(pipeline_config, "cache_ttl_seconds", 3600),
                    max_memory_items=getattr(pipeline_config, "max_cache_items", 1000),
                )

    def answer(
        self,
        question: str,
        num_documents: int = 5,
        include_explanation: bool = True,
        template_name: str = "medical_qa",
        conversation_context: str = None,
        generation_max_tokens: Optional[int] = None,
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
                    question=cached.get("question", question),
                    answer=cached.get("answer", ""),
                    sources=cached.get("sources", []),
                    confidence=cached.get("confidence", {}),
                    attributions=cached.get("attributions", []),
                    disclaimer=cached.get("disclaimer", ""),
                    rationale=cached.get("rationale", None),
                    is_answerable=cached.get("is_answerable", True),
                    from_cache=True,
                    factual_consistency=cached.get("factual_consistency", None),
                    confidence_breakdown=cached.get("confidence_breakdown", None),
                    hallucination=cached.get("hallucination", None),
                    stage_latencies=cached.get("stage_latencies", None),
                )

        # 1. QUERY ENHANCEMENT (pre-retrieval)
        # Work with raw (context-free) queries here; _prepare_retrieval_queries
        # applies conversation_context to every query at the end in one place,
        # so context is never silently dropped when query enhancement runs.
        _s = time.perf_counter()
        retrieval_query = question
        all_retrieval_queries: List[str] = [retrieval_query]

        if self.query_enhancer:
            try:
                enhanced = self.query_enhancer.enhance(question)
                # enhance() returns an EnhancedQuery dataclass with .all_queries
                if hasattr(enhanced, "all_queries") and enhanced.all_queries:
                    all_retrieval_queries = list(enhanced.all_queries)
                    retrieval_query = enhanced.all_queries[0]
                elif isinstance(enhanced, str) and enhanced:
                    retrieval_query = enhanced
                    all_retrieval_queries = [retrieval_query]
            except Exception as e:
                logger.warning(f"Query enhancement failed, using original: {e}")

        # Merge: add generic-name variants for brand-name drug queries AND apply
        # conversation_context once to all queries (fixes context-drop after enhancement).
        all_retrieval_queries = self._prepare_retrieval_queries(
            question, retrieval_query, all_retrieval_queries[1:], conversation_context
        )
        retrieval_query = all_retrieval_queries[0]
        _t["query_enhancement_ms"] = (time.perf_counter() - _s) * 1000

        # 2. RETRIEVE relevant documents
        # When include_explanation is False (fast path): single query, no cross-encoder rerank.
        _s = time.perf_counter()
        use_rerank = bool(include_explanation)
        fusion_queries = (
            all_retrieval_queries[:4]
            if include_explanation
            else [retrieval_query]
        )
        if (
            self.enable_multi_query_fusion
            and len(fusion_queries) > 1
            and hasattr(self.retriever, "retrieve")
        ):
            documents, context = self._retrieve_multi_query_fused(
                fusion_queries, k=num_documents, use_reranking=use_rerank
            )
        else:
            documents, context = self.retriever.retrieve_with_context(
                retrieval_query, k=num_documents, use_reranking=use_rerank
            )
        _t["retrieval_ms"] = (time.perf_counter() - _s) * 1000
        # Merge per-stage retriever timings (dense, sparse, rrf, rerank sub-ms)
        if hasattr(self.retriever, "last_timings") and self.retriever.last_timings:
            for sub_key, sub_val in self.retriever.last_timings.items():
                _t[f"retrieval_{sub_key}"] = sub_val

        # Apply lost-in-the-middle reordering, then build context string.
        documents = self._reorder_lost_in_middle(documents)
        # build_safe_context is imported at module level; the None branch is a
        # defensive fallback only (ImportError at startup would be very loud).
        if build_safe_context is None:
            from src.utils.context_builder import build_safe_context as _bsc
            context = _bsc(documents, max_chars=2000)
        else:
            context = build_safe_context(documents, max_chars=2000)

        # 3. CORRECTIVE RAG (post-retrieval quality check)
        _s = time.perf_counter()
        if self.corrective_rag and documents and include_explanation:
            try:
                corrected_result = self.corrective_rag.retrieve_with_correction(
                    query=retrieval_query,
                    k=num_documents,
                )
                # retrieve_with_correction returns Tuple[List, bool]
                corrected_docs, was_corrected = corrected_result
                if corrected_docs and len(corrected_docs) > 0:
                    documents = self._reorder_lost_in_middle(corrected_docs)
                    context = build_safe_context(documents, max_chars=2000)
            except Exception as e:
                logger.warning(f"Corrective RAG failed, using original retrieval: {e}")
        _t["corrective_rag_ms"] = (time.perf_counter() - _s) * 1000

        # 4. GROUNDING GATE: adaptive check
        is_answerable = True
        if self.enable_grounding_gate:
            is_answerable = self._check_answerability(documents, question)

        if not is_answerable:
                disclaimer = self.prompt_manager.get_medical_disclaimer()
                return QAResponse(
                    question=question,
                    answer=self.UNANSWERABLE_RESPONSE,
                    sources=[],
                    confidence={
                        "score": 0.0,
                        "level": "low",
                        "explanation": "Insufficient relevant context found in knowledge base.",
                    },
                    attributions=[],
                    disclaimer=disclaimer,
                    rationale=None,
                    is_answerable=False,
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
        _max_tokens = int(generation_max_tokens) if generation_max_tokens else 512
        _max_tokens = max(32, min(_max_tokens, 512))

        # Extractive and Ollama backends: skip prompt_manager — they own their own prompt format
        _backend_type = getattr(self.llm, "backend", None)
        if _backend_type in ("extractive", "ollama"):
            generation_result = self.llm.generate_with_context(
                question=question,
                context=generation_context,
                max_new_tokens=_max_tokens,
            )
        else:
            # Auto-route factoid questions to a tighter 1-2 sentence prompt so TinyLlama
            # doesn't pad with unsupported treatment discussion.
            _effective_template = template_name
            if template_name == "medical_qa" and self._is_factoid_question(question):
                _effective_template = "concise_factoid"
            prompt = self.prompt_manager.build_prompt(
                question=question,
                context=generation_context,
                template_name=_effective_template,
                model_name=getattr(self.llm, "model_name", None),
            )
            generation_result = self.llm.generate(
                prompt,
                max_new_tokens=_max_tokens,
                do_sample=False,
                return_probabilities=include_explanation,
            )

        answer = self._clean_answer(generation_result.response)
        _generation_backend_used = getattr(generation_result, "generation_backend_used", None) or _backend_type or "unknown"
        _t["generation_ms"] = (time.perf_counter() - _s) * 1000

        # Evidence seen by the LLM (may differ after compression)
        verify_context = generation_context if generation_context else context

        # 7. FACTUAL CONSISTENCY CHECK (post-generation)
        factual_result = None
        if self.factual_consistency_checker and answer and include_explanation:
            try:
                fc = self.factual_consistency_checker.check_consistency(
                    answer=answer, context=verify_context
                )
                factual_result = {
                    "is_consistent": fc.is_consistent if hasattr(fc, "is_consistent") else True,
                    "score": fc.consistency_score if hasattr(fc, "consistency_score") else 1.0,
                    "details": (
                        [vars(c) if hasattr(c, "__dict__") else c for c in fc.claim_results]
                        if hasattr(fc, "claim_results")
                        else []
                    ),
                }
            except Exception as e:
                logger.warning(f" Factual consistency check failed: {e}")

        # 7b. HALLUCINATION DETECTION (DeBERTa NLI + rule-based)
        # Skip for Ollama — DeBERTa NLI produces systematic false positives on MCQ-format
        # context combined with Ollama's paraphrased answers (not word-for-word copies).
        _s = time.perf_counter()
        hallucination_result = None
        _hal_gate_result = None  # used by quality gate even on fast path
        _is_ollama_backend = (_generation_backend_used == "ollama")
        if self.hallucination_detector and answer and not _is_ollama_backend:
            try:
                if verify_context != context:
                    doc_dicts_for_hal = [
                        {"content": verify_context, "source": "generation_context"}
                    ]
                else:
                    doc_dicts_for_hal = [{"content": doc.content} for doc in documents]
                hal = self.hallucination_detector.detect(
                    answer=answer,
                    retrieved_documents=doc_dicts_for_hal,
                    query=question,
                )
                _hal_gate_result = {
                    "has_hallucination": hal.has_hallucination,
                    "score": hal.hallucination_score,
                }
                if include_explanation:
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
                # MultiSignalConfidenceScorer handles score normalization
                # internally — no need to mutate document scores here.
                bd = self.multi_signal_scorer.compute_confidence(
                    query=question,
                    answer=answer,
                    retrieved_documents=documents,
                    generation_probabilities=generation_result.probabilities,
                    backend=_generation_backend_used or "extractive",
                )
                # Build explanation that leads with the calibrated level (not sub-signals).
                _lvl = bd.confidence_level
                _level_prefix = {
                    "high": "High confidence",
                    "medium": "Medium confidence",
                    "low": "Low confidence",
                }.get(_lvl, "Confidence")
                # Source agreement (grounding) takes priority over retrieval quality.
                if bd.source_agreement < 0.2:
                    _signal_note = "answer may not be well grounded in the retrieved sources."
                elif bd.source_agreement <= 0.5:
                    _signal_note = "partial source support for this answer."
                elif bd.source_agreement > 0.7 and _lvl == "low":
                    # Good grounding but Platt calibration still gives low score —
                    # surface the grounding quality so the explanation isn't contradictory.
                    _signal_note = "answer is well grounded in sources; verify with a doctor."
                elif bd.retrieval_confidence > 0.7:
                    _signal_note = "retrieved documents are relevant."
                elif bd.retrieval_confidence < 0.4:
                    _signal_note = "retrieved documents have limited relevance."
                else:
                    _signal_note = "moderate retrieval relevance."
                _explanation = f"{_level_prefix} — {_signal_note}"
                logger.debug(f"[confidence] level={_lvl} expl={_explanation!r}")
                confidence = {
                    "score": bd.calibrated_confidence,
                    "level": _lvl,
                    "explanation": _explanation,
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
                    num_sources=len(documents),
                )
                confidence = {
                    "score": confidence_result.calibrated_score,
                    "level": confidence_result.level,
                    "explanation": confidence_result.explanation,
                }
            else:
                # Low-latency path: compute a lightweight retrieval-based score
                # so the UI always has a meaningful confidence signal.
                retrieval_scores = [doc.score for doc in documents]
                if retrieval_scores:
                    avg_score = sum(retrieval_scores) / len(retrieval_scores)
                    top_score = max(retrieval_scores)
                    raw = 0.4 * avg_score + 0.6 * top_score
                    raw = max(0.0, min(1.0, raw))
                else:
                    raw = 0.5
                level = "high" if raw >= 0.65 else ("low" if raw < 0.35 else "medium")
                level_desc = {
                    "high": "Retrieved documents are highly relevant to this query.",
                    "medium": "Retrieved documents are moderately relevant to this query.",
                    "low": "Retrieved documents have limited relevance — verify with a healthcare provider.",
                }[level]
                confidence = {
                    "score": round(raw, 3),
                    "level": level,
                    "explanation": f"{level_desc} ({len(retrieval_scores)} source(s) retrieved)",
                }

        _t["confidence_ms"] = (time.perf_counter() - _s) * 1000

        # 8b. POST-GENERATION QUALITY GATE
        # Ollama (Qwen2.5-7B) is a strong model — skip TinyLlama-targeted hallucination
        # gates which incorrectly reject good Ollama answers based on source_agreement/NLI.
        _skip_quality_gates = (_generation_backend_used == "ollama")

        if answer and not _skip_quality_gates:
            # (a) Medication entity grounding
            med_verdict = self._medication_entity_verdict(question, answer, documents)
            if not med_verdict.get("supported", True):
                logger.warning(
                    f"Quality gate rejected answer — medication entity not in KB "
                    f"({med_verdict.get('reason')}) — falling back to UNANSWERABLE_RESPONSE"
                )
                answer = self.UNANSWERABLE_RESPONSE

            # (b) NLI hallucination gate — uses _hal_gate_result which is populated in
            #     both fast and full-XAI paths.
            # TinyLlama answers against MCQ-format docs score systematically high on NLI
            # (0.80–0.87) even when the answer is reasonable, because DeBERTa-MNLI rarely
            # finds entailment in exam-bank snippets.
            #
            # Gate conditions (raised from 0.80/0.90 to 0.85/0.95 to reduce false-negatives):
            #   • (combined) hallucination_score > 0.85 AND source_agreement < 0.20
            #   • (extreme)  hallucination_score > 0.95 alone — truly pathological
            # Bypass: if the answer shares ≥3 key medical terms with the query AND
            #   retrieval_confidence > 0.6, NLI is unreliable — skip the gate.
            if answer != self.UNANSWERABLE_RESPONSE and _hal_gate_result:
                _hal_score = float(_hal_gate_result.get("score", 0.0))
                _has_hal = bool(_hal_gate_result.get("has_hallucination", False))
                _sa_for_gate = (
                    float(confidence_breakdown.get("source_agreement", 1.0))
                    if confidence_breakdown else 1.0
                )
                _ret_conf = (
                    float(confidence_breakdown.get("retrieval_confidence", 0.0))
                    if confidence_breakdown else 0.0
                )

                # Bypass: on-topic + well-retrieved answers should not be blocked by NLI
                # (DeBERTa-MNLI is unreliable for exam-format medical knowledge base chunks)
                import re as _re
                _q_terms = {
                    t for t in _re.findall(r"\w+", question.lower())
                    if len(t) > 3
                }
                _a_terms = {
                    t for t in _re.findall(r"\w+", answer.lower())
                    if len(t) > 3
                }
                _medical_overlap = len(_q_terms & _a_terms)
                _nli_gate_bypass = _medical_overlap >= 3 and _ret_conf > 0.6

                _combined_bad = _has_hal and _hal_score > 0.85 and _sa_for_gate < 0.20
                _extreme_hal = _has_hal and _hal_score > 0.95
                if not _nli_gate_bypass and (_combined_bad or _extreme_hal):
                    logger.warning(
                        f"Quality gate rejected answer "
                        f"(hallucination={_hal_score:.2f}, source_agreement={_sa_for_gate:.3f}, "
                        f"medical_overlap={_medical_overlap}) — falling back to UNANSWERABLE_RESPONSE"
                    )
                    answer = self.UNANSWERABLE_RESPONSE
                elif _nli_gate_bypass and (_combined_bad or _extreme_hal):
                    logger.debug(
                        f"NLI gate bypassed: answer is medically on-topic "
                        f"(overlap={_medical_overlap}, retrieval_conf={_ret_conf:.2f})"
                    )

            # (c) Source agreement gate — when multi-signal XAI ran and source_agreement
            #     is near zero, the answer is not grounded in the KB at all.
            # Threshold lowered from 0.10 to 0.05: bag-of-words overlap underestimates
            # agreement when TinyLlama paraphrases (e.g. "hypertension" vs "high blood
            # pressure"). The semantic source agreement fix (D1) will improve this signal;
            # until then, 0.05 avoids blocking valid paraphrased answers.
            if answer != self.UNANSWERABLE_RESPONSE and confidence_breakdown:
                _sa = float(confidence_breakdown.get("source_agreement", 1.0))
                if _sa < 0.05:
                    logger.warning(
                        f"Quality gate rejected answer (source_agreement={_sa:.3f} < 0.05) — "
                        "falling back to UNANSWERABLE_RESPONSE"
                    )
                    answer = self.UNANSWERABLE_RESPONSE
                else:
                    logger.debug(f"Source agreement gate passed: sa={_sa:.3f}")

            # (d) Caps spam check (only if nothing above already blocked)
            if answer != self.UNANSWERABLE_RESPONSE:
                try:
                    from src.utils.text_cleaning import is_hallucinated_caps_spam
                    if is_hallucinated_caps_spam(answer):
                        logger.warning(
                            "Quality gate rejected answer (ALL-CAPS acronym spam) — "
                            "falling back to UNANSWERABLE_RESPONSE"
                        )
                        answer = self.UNANSWERABLE_RESPONSE
                except Exception:
                    pass

            # (e) Extractive fallback — for factoid questions with low confidence,
            # pull the answer directly from the best retrieved sentence instead of
            # relying on TinyLlama's paraphrase (which can misclassify drug classes, etc.)
            # Skip when already using the extractive or ollama backend.
            # OllamaLLM handles its own fallback to ExtractiveQA internally.
            _already_extractive = _backend_type in ("extractive", "ollama")
            if not _already_extractive and answer != self.UNANSWERABLE_RESPONSE and self._is_factoid_question(question):
                _conf_score = confidence.get("score", 1.0) if isinstance(confidence, dict) else 1.0
                if _conf_score < 0.55:
                    _extractive = self._build_extractive_answer(question, documents)
                    if _extractive:
                        logger.info(
                            f"Extractive fallback triggered for factoid question "
                            f"(confidence={_conf_score:.2f}) — replacing LLM answer"
                        )
                        answer = _extractive

            # When the gate fires, the previously-computed XAI fields describe the
            # rejected answer, not the fallback. Reset them so the response is
            # internally consistent (no confidence/hallucination data for discarded text).
            if answer == self.UNANSWERABLE_RESPONSE:
                confidence = {
                    "score": 0.0,
                    "level": "low",
                    "explanation": "Insufficient information in knowledge base.",
                }
                factual_result = None
                hallucination_result = None
                confidence_breakdown = None

        # 9. SOURCE ATTRIBUTION
        if self.source_attributor and include_explanation:
            doc_dicts = [
                {
                    "content": doc.content,
                    "source": doc.source,
                    "url": doc.metadata.get("url", ""),
                }
                for doc in documents
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
        else:
            attributions = []

        # 10. RATIONALE
        rationale = None
        if self.rationale_generator and include_explanation:
            combined_context = "\n".join([d.content for d in documents])
            rationale = self.rationale_generator.generate_rationale(
                question=question, answer=answer, context=combined_context
            )

        # 11. BUILD SOURCE LIST — sorted strongest-to-weakest for the user-facing
        # reference list. (documents itself stays in prompt order for generation.)
        sources = sorted(
            [
                {
                    "source": doc.source,
                    "content": doc.content,
                    "score": doc.score,
                    "url": doc.metadata.get("url", ""),
                }
                for doc in documents
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        disclaimer = self.prompt_manager.get_medical_disclaimer()

        _t["total_ms"] = (time.perf_counter() - _t0) * 1000

        # SLO budget checks — warn when any stage exceeds its target
        _SLO = {"retrieval_ms": 500, "generation_ms": 90_000, "total_ms": 120_000}
        for _stage, _budget in _SLO.items():
            _actual = _t.get(_stage, 0)
            if _actual > _budget:
                logger.warning(
                    f"SLO breach: {_stage}={_actual:.0f}ms > budget={_budget}ms"
                )

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
            generation_backend_used=_generation_backend_used,
        )

        # 12. CACHE THE RESPONSE (with context key)
        # Never cache UNANSWERABLE answers — a stale "no info" entry would keep
        # returning the fallback even after the KB is updated or config changes.
        if self.cache_manager and answer != self.UNANSWERABLE_RESPONSE:
            self.cache_manager.cache_response(
                question,
                {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "attributions": attributions,
                    "disclaimer": disclaimer,
                    "rationale": rationale,
                    "is_answerable": True,
                    "factual_consistency": factual_result,
                    "confidence_breakdown": confidence_breakdown,
                    "hallucination": hallucination_result,
                    "stage_latencies": _t,
                    "generation_backend_used": _generation_backend_used,
                },
                context_key=dynamic_context_key,
            )

        return response

    def _retrieve_multi_query_fused(
        self, queries: List[str], k: int, use_reranking: bool = True
    ):
        """RRF-fuse rankings from multiple query strings (QueryEnhancer variants)."""
        from src.retrieval.hybrid_retriever import (
            RetrievedDocument,
            normalize_retrieval_scores,
            reciprocal_rank_fusion,
        )

        if not queries:
            return [], ""

        per_query_docs: List = []
        ranked_lists = []
        for q in queries:
            docs = self.retriever.retrieve(
                q, k=max(k * 2, 10), use_reranking=use_reranking
            )
            per_query_docs.append(docs)
            ranked_lists.append([(d.doc_id, float(d.score)) for d in docs])

        if len(ranked_lists) == 1:
            documents = per_query_docs[0]
        else:
            fused = reciprocal_rank_fusion(
                ranked_lists,
                k=self.retriever.rrf_k,
                weights=[1.0] * len(ranked_lists),
            )
            id_to_doc: Dict[str, Any] = {}
            for docs in per_query_docs:
                for d in docs:
                    id_to_doc.setdefault(d.doc_id, d)
            documents = []
            for did, rrf_s in sorted(fused.items(), key=lambda x: x[1], reverse=True):
                if did in id_to_doc:
                    d = id_to_doc[did]
                    documents.append(
                        RetrievedDocument(
                            content=d.content,
                            source=d.source,
                            score=rrf_s,
                            metadata=dict(d.metadata),
                            doc_id=d.doc_id,
                            score_type="rrf",
                        )
                    )
            normalize_retrieval_scores(documents, "rrf")

        documents = documents[:k]
        return documents, ""

    @staticmethod
    def _reorder_lost_in_middle(documents: List) -> List:
        """Place strongest evidence first and second-strongest last (Liu et al., 2023)."""
        if not documents or len(documents) < 3:
            return documents

        def _score(i: int) -> float:
            d = documents[i]
            return float(d.score) if hasattr(d, "score") else 0.0

        order_idx = sorted(range(len(documents)), key=_score, reverse=True)
        perm = [order_idx[0]] + order_idx[2:] + [order_idx[1]]
        return [documents[i] for i in perm]

    def _check_answerability(self, documents: List, question: str = "") -> bool:
        """
        Adaptive grounding gate: doc is relevant if
            score >= max(absolute_score_floor, adaptive_ratio * top_score)

        When ``question`` is set, at least one relevant doc must meet a minimum
        token-overlap ratio with the question (reduces topic-adjacent non-answers).
        """
        if not documents:
            return False

        top_score = max(doc.score for doc in documents)

        score_type = getattr(documents[0], "score_type", "cosine")
        if score_type == "normalized":
            abs_floor = self.absolute_score_floor
        elif score_type == "rrf":
            abs_floor = 0.005
        else:
            abs_floor = self.absolute_score_floor

        threshold = max(
            abs_floor,
            self.adaptive_threshold_ratio * top_score,
        )

        relevant_docs = [doc for doc in documents if doc.score >= threshold]

        if len(relevant_docs) < self.min_relevant_docs:
            return False

        if self.grounding_min_term_overlap > 0 and question.strip():
            import re as _re
            q_terms = {
                t for t in _re.findall(r"\w+", question.lower())
                if len(t) > 3 and t not in _PIPELINE_STOPWORDS
            }
            if q_terms:
                ok = False
                for doc in relevant_docs:
                    content = doc.content if hasattr(doc, "content") else str(doc)
                    c_terms = set(_re.findall(r"\w+", content.lower()))
                    if len(q_terms & c_terms) / len(q_terms) >= self.grounding_min_term_overlap:
                        ok = True
                        break
                if not ok:
                    return False

        # Causation questions require at least one doc with causal language.
        # MCQ treatment snippets ("drug of choice") pass the term-overlap check
        # because the topic word appears, but they don't answer "what causes X?".
        if question.strip() and self._is_causation_question(question):
            _CAUSE_MARKERS = (
                "caused by", "cause", "causes", "etiology",
                "due to", "trigger", "triggers", "because",
                "pathophysiology", "mechanism",
            )
            has_causal_doc = False
            for doc in relevant_docs:
                content = (doc.content if hasattr(doc, "content") else str(doc)).lower()
                if any(m in content for m in _CAUSE_MARKERS):
                    has_causal_doc = True
                    break
            if not has_causal_doc:
                return False

        return True

    def _clean_answer(self, answer: str) -> str:
        """
        Pipeline-level answer cleaning to strip training data artifacts.
        Delegates to the shared text cleaning utility.
        """
        from src.utils.text_cleaning import clean_llm_response

        return clean_llm_response(answer)

    @classmethod
    def _is_medication_side_effect_query(cls, question: str) -> bool:
        q = (question or "").lower()
        has_effect = any(k in q for k in ("side effect", "side-effects", "adverse effect"))
        med_tokens = list(cls._MED_BRAND_TO_GENERIC.keys()) + ["tablet", "capsule", "mg"]
        return has_effect and any(t in q for t in med_tokens)

    @staticmethod
    def _contains_medication_term(text: str, term: str) -> bool:
        import re

        t = re.escape(term.lower().strip())
        # Allow common pluralization (e.g. paracetamols), require word boundaries.
        return bool(re.search(rf"\b{t}(?:s)?\b", (text or "").lower()))

    @classmethod
    def _filter_docs_by_medication_terms(cls, docs: List, med_terms: List[str]) -> List:
        out = []
        for d in docs:
            content = d.content if hasattr(d, "content") else str(d)
            if any(cls._contains_medication_term(content, term) for term in med_terms):
                out.append(d)
        return out

    @classmethod
    def _prepare_retrieval_queries(
        cls,
        question: str,
        base_query: str,
        expanded_queries: List[str],
        conversation_context: Optional[str],
    ) -> List[str]:
        queries: List[str] = []
        if base_query:
            queries.append(base_query)
        for q in expanded_queries or []:
            if q and q not in queries:
                queries.append(q)

        q_lower = (question or "").lower()
        generic_terms: List[str] = []
        for brand, generics in cls._MED_BRAND_TO_GENERIC.items():
            if brand in q_lower:
                generic_terms.extend(generics)
        if generic_terms:
            generic_query = f"{question} {' '.join(sorted(set(generic_terms)))}"
            if generic_query not in queries:
                queries.append(generic_query)

        if conversation_context and queries:
            queries = [f"{conversation_context}\n\nCurrent question: {q}" for q in queries]
        return queries

    @classmethod
    def _medication_entity_verdict(cls, question: str, answer: str, docs: List) -> Dict[str, Any]:
        q_lower = (question or "").lower()
        asked_terms: List[str] = []
        for brand, generics in cls._MED_BRAND_TO_GENERIC.items():
            if brand in q_lower:
                asked_terms.extend([brand] + generics)
        asked_terms = sorted(set(asked_terms))
        if not asked_terms:
            return {"supported": True, "reason": "not_medication_query"}

        answer_has_med = any(cls._contains_medication_term(answer, t) for t in asked_terms)
        if not answer_has_med:
            return {"supported": False, "reason": "answer_not_anchored_to_asked_medication"}

        filtered_docs = cls._filter_docs_by_medication_terms(docs, asked_terms)
        if not filtered_docs:
            return {"supported": False, "reason": "retrieved_context_not_about_asked_medication"}

        return {"supported": True, "reason": "supported"}

    @classmethod
    def _format_medication_extractive_answer(cls, text: str, med_terms: List[str]) -> Optional[str]:
        if not text:
            return None
        t = text.strip()
        has_med = any(cls._contains_medication_term(t, m) for m in med_terms)
        if not has_med:
            return None
        lowered = t.lower()
        rare_note = "rare" if "rare" in lowered else ""
        prefix = "Based on retrieved evidence: "
        if "side effects include" in lowered:
            body = t
        else:
            body = f"Reported side effects include: {t}"
        if rare_note and "rare" not in body.lower():
            body = f"Rarely, {body[0].lower()}{body[1:]}"
        return prefix + body

    @staticmethod
    def _is_factoid_question(question: str) -> bool:
        """Detect short factoid queries: drug of choice, mechanism, dosage, classification."""
        q = (question or "").lower()
        markers = (
            "drug of choice",
            "doc for",
            "mechanism of action",
            "moa of",
            "what is the drug",
            "which drug",
            "first line treatment",
            "first-line",
            "dose of",
            "dosage of",
            "half life",
            "half-life",
            "classified as",
            "belongs to",
        )
        return any(m in q for m in markers)

    @staticmethod
    def _is_causation_question(question: str) -> bool:
        """Heuristic detection for causation-style queries."""
        q = (question or "").lower()
        markers = (
            "what causes",
            "what cause",
            "caused by",
            "why does",
            "why do",
            "etiology",
            "reason for",
        )
        return any(m in q for m in markers)

    def _build_extractive_answer(self, question: str, documents: List) -> Optional[str]:
        """
        Build a lightweight extractive answer from retrieved snippets.

        Extracts the single most query-relevant sentence from each of the
        top-2 documents (scored by query term overlap + causal/factual markers)
        rather than dumping entire chunks. Produces concise, readable answers.
        """
        if not documents:
            return None

        import re as _re

        # Markers that indicate a sentence directly answers a factoid question.
        # Weighted by relevance: treatment/first-line markers score highest.
        _cause_markers = (
            "first-line", "first line", "drug of choice", "recommended",
            "guideline", "jnc", "aha", "who recommends",
            "caused by", "cause", "causes", "because",
            "trigger", "triggers", "due to", "etiology",
            "treat", "treatment", "therapy", "drug", "medication",
            "inhibitor", "blocker", "diuretic", "agonist",
        )
        _stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "have", "has",
            "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "of", "in", "on", "at", "to", "for", "with", "this",
            "that", "these", "those", "and", "but", "or", "as", "by", "from",
        }
        q_terms = {
            t for t in _re.findall(r"\w+", question.lower())
            if len(t) > 3 and t not in _stopwords
        }

        def _best_sentence(content: str) -> tuple:
            """Return (score, best_sentence) from a document chunk."""
            # For MedMCQA/MedQA docs, prefer the Answer: section over the question text.
            _ans_match = _re.search(r"Answer\s*:\s*(.+?)(?:\n|$)", content, _re.IGNORECASE)
            if _ans_match:
                ans_text = _ans_match.group(1).strip()
                if len(ans_text) > 10:
                    return 1.0, ans_text

            sentences = [
                s.strip() for s in _re.split(r"(?<=[.!?])\s+", content)
                if len(s.strip()) > 20 and not s.strip().endswith("?")  # skip question sentences
            ]
            if not sentences:
                return 0.0, content[:300].strip()
            best_s, best_score = "", 0.0
            for sent in sentences:
                s_lower = sent.lower()
                s_terms = set(_re.findall(r"\w+", s_lower))
                overlap = len(q_terms & s_terms) / max(len(q_terms), 1) if q_terms else 0.0
                causal_bonus = sum(0.15 for m in _cause_markers if m in s_lower)
                s_score = overlap + causal_bonus
                if s_score > best_score:
                    best_score, best_s = s_score, sent
            return best_score, (best_s if best_s else sentences[0])

        # Score each doc; extract its best sentence
        scored_docs: List = []
        for d in documents:
            content = d.content if hasattr(d, "content") else str(d)
            doc_score = float(getattr(d, "score", 0.0))
            sent_score, best_sent = _best_sentence(content)
            scored_docs.append((doc_score + sent_score, best_sent))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        if not scored_docs:
            return None

        top_score, top_sent = scored_docs[0]
        if not top_sent:
            return None

        # If top doc had a clean Answer: extraction (score==1.0), use it alone —
        # don't pollute it with a second doc that may have bad content.
        if top_score >= 1.0:
            return top_sent

        # Otherwise take up to 2 sentences, deduplicating near-identical ones
        top_sentences = [txt for _, txt in scored_docs[:2] if txt]
        if len(top_sentences) == 2:
            t1 = set(top_sentences[0].lower().split())
            t2 = set(top_sentences[1].lower().split())
            if t1 and len(t1 & t2) / max(len(t1), 1) > 0.6:
                top_sentences = top_sentences[:1]

        return " ".join(top_sentences)

    def batch_answer(self, questions: List[str], **kwargs) -> List[QAResponse]:
        """Answer multiple questions."""
        return [self.answer(q, **kwargs) for q in questions]
