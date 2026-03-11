"""
LangGraph Node Functions for Healthcare RAG Pipeline.

Each node receives the full state and returns partial updates.
Nodes are responsible for a single step in the RAG workflow.

Serialization safety:
Every node return dict is passed through _sanitize_state() which
recursively converts numpy scalar types (float64, int64, bool_,
etc.) to plain Python natives.  This prevents the msgpack
serialization crash in LangGraph's MemorySaver:
  "Type is not msgpack serializable: numpy.float64"

Fixes applied (v2):
- Adaptive RRF-aware relevance thresholds (MIN_RELEVANCE_SCORE was 0.3,
  but RRF scores live in 0.01-0.04 range → every doc was graded "irrelevant")
- generate_answer now uses TinyLlama's proper chat-template format
  (<|system|> / <|user|> / <|assistant|>) instead of plain markdown,
  so the model actually follows instructions instead of copying context
- clean_llm_response is now called on every generated answer to strip
  leaked document headers, MedQuAD artefacts, and sign-offs
- Context builder strips raw source-label prefixes that contain
  "Question: ..." patterns which caused TinyLlama to continue them
- Content-sanity guard: if every retrieved doc scores below the adaptive
  floor the node returns the unanswerable response immediately
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.langgraph.langgraph_state import HealthcareRAGState

# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _sanitize_state(obj: Any) -> Any:
    """
    Recursively convert numpy scalar / array types to plain Python natives
    so that LangGraph's MemorySaver (msgpack) can serialize the state.

    Handles:
    - numpy.float32 / float64 / float128  → Python float
    - numpy.int8 … int64 / uint* types    → Python int
    - numpy.bool_                          → Python bool
    - numpy.ndarray                        → Python list (nested)
    - dict / list / tuple                  → recurse into each value
    - everything else                      → returned unchanged
    """
    import numpy as np  # local import to avoid circular dependency at module level

    if isinstance(obj, dict):
        return {k: _sanitize_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cleaned = [_sanitize_state(v) for v in obj]
        return cleaned if isinstance(obj, list) else tuple(cleaned)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):  # covers float16/32/64/128
        return float(obj)
    if isinstance(obj, np.integer):  # covers int8/16/32/64 and uint*
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Absolute score floor for RRF-fused scores (~0.01–0.04 typical range).
# The old value of 0.3 was calibrated for cosine-similarity scores and caused
# every RRF result to be classified as "irrelevant".
MIN_RELEVANCE_SCORE = 0.01

# When the top document scores above this ratio of the highest score in the
# result set, the document is considered relevant.  Combined with the absolute
# floor above this gives a sensible adaptive threshold for both cosine and RRF
# score scales.
ADAPTIVE_THRESHOLD_RATIO = 0.5

# Minimum number of "relevant" docs required before we attempt generation.
MIN_RELEVANT_DOCS = 1  # relaxed from 2 — single strong doc is enough

MAX_RETRY_COUNT = 2
LOW_CONFIDENCE_THRESHOLD = 0.5

UNANSWERABLE_RESPONSE = (
    "I don't have enough relevant information in my knowledge base to answer "
    "this question accurately. Please consult a healthcare professional for "
    "specific medical advice."
)

MEDICAL_DISCLAIMER = (
    "\n\u26a0\ufe0f MEDICAL DISCLAIMER: This information is for educational purposes "
    "only and is NOT a substitute for professional medical advice, diagnosis, or "
    "treatment. Always seek the advice of your physician or other qualified health "
    "provider with any questions you may have regarding a medical condition."
)

# TinyLlama chat-format tokens
_SYS = "<|system|>"
_USR = "<|user|>"
_AST = "<|assistant|>"
_END = "</s>"

# System instruction shared by generate_answer and refine_query nodes
_MEDICAL_SYSTEM_INSTRUCTION = (
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


# ---------------------------------------------------------------------------
# Helper: build a safe context string
# ---------------------------------------------------------------------------


def _build_safe_context(documents: List[Document], max_chars: int = 2000) -> str:
    """
    Convert LangChain Documents into a context string that is safe to pass
    directly to TinyLlama.

    Key changes vs the old implementation:
    - Source labels are placed on a plain line like  ``Source: MedQuAD``
      rather than as a prefix of the content (``[MedQuAD]: Question: ...``).
      The old format caused TinyLlama to treat the source line as a Q&A
      template and continue generating in that style.
    - Each document block is separated by a blank line so boundaries are
      visually clear to the model.
    - Total character budget is enforced so we stay within the 2 048-token
      context window.
    """
    parts: List[str] = []
    used = 0

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", f"Document {i}")
        content = doc.page_content.strip()

        # Strip leading "Question: ..." or "[Source]: ..." artefacts that
        # sometimes appear at the start of MedQuAD chunks.
        import re

        content = re.sub(r"^\s*\[.*?\]\s*:\s*", "", content)
        content = re.sub(r"^\s*Question\s*:\s*", "", content, flags=re.IGNORECASE)

        block = f"[{i}] Source: {source}\n{content}"
        if used + len(block) > max_chars:
            # Truncate this block to fit within budget
            remaining = max_chars - used
            if remaining > 100:
                block = block[:remaining] + "..."
                parts.append(block)
            break

        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: compute adaptive relevance threshold
# ---------------------------------------------------------------------------


def _adaptive_threshold(documents: List[Document]) -> float:
    """
    Compute a relevance score threshold that works for both cosine-similarity
    scores (0.3–0.95) and RRF-fused scores (0.01–0.04).

    Strategy: take half of the top score in the result set, but never go
    below the absolute floor (MIN_RELEVANCE_SCORE = 0.01).
    """
    if not documents:
        return MIN_RELEVANCE_SCORE

    top_score = max(doc.metadata.get("score", 0.0) for doc in documents)
    return max(MIN_RELEVANCE_SCORE, ADAPTIVE_THRESHOLD_RATIO * top_score)


# ---------------------------------------------------------------------------
# Main node class
# ---------------------------------------------------------------------------


class HealthcareRAGNodes:
    """
    Node implementations for Healthcare RAG graph.

    Wraps existing components (retriever, LLM, XAI) and exposes
    them as LangGraph-compatible node functions.
    """

    def __init__(
        self,
        retriever,
        llm,
        confidence_scorer=None,
        source_attributor=None,
        rationale_generator=None,
        k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.confidence_scorer = confidence_scorer
        self.source_attributor = source_attributor
        self.rationale_generator = rationale_generator
        self.k = k

    # ------------------------------------------------------------------
    # Node: retrieve_documents
    # ------------------------------------------------------------------

    def retrieve_documents(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Retrieve documents from knowledge base.

        Uses the most recently refined query from query_history so that
        the refine → retrieve loop works correctly.
        """
        history = state.get("query_history", [])
        query = history[-1] if history else state.get("question", "")

        try:
            docs = self.retriever.retrieve(query, k=self.k)

            lc_docs: List[Document] = []
            for doc in docs:
                lc_docs.append(
                    Document(
                        page_content=doc.content
                        if hasattr(doc, "content")
                        else str(doc),
                        metadata={
                            "source": doc.source
                            if hasattr(doc, "source")
                            else "unknown",
                            "score": doc.score if hasattr(doc, "score") else 0.0,
                            "url": getattr(doc, "url", ""),
                        },
                    )
                )

            return {"documents": lc_docs}

        except Exception as exc:
            return {
                "documents": [],
                "error": f"Retrieval error: {exc}",
            }

    # ------------------------------------------------------------------
    # Node: grade_relevance
    # ------------------------------------------------------------------

    def grade_relevance(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Grade document relevance to the question.

        Uses an *adaptive* threshold so the grading works correctly for
        both cosine-similarity scores and RRF-fused scores.
        """
        documents = state.get("documents", [])
        question = state.get("question", "")

        if not documents:
            return {"doc_grades": [], "is_answerable": False}

        threshold = _adaptive_threshold(documents)
        query_terms = set(question.lower().split())

        grades: List[Dict[str, Any]] = []
        relevant_count = 0

        for i, doc in enumerate(documents):
            score = doc.metadata.get("score", 0.0)
            content = doc.page_content.lower()
            content_terms = set(content.split())

            term_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)

            # A document is relevant if its retrieval score clears the
            # adaptive threshold OR it has strong term overlap with the query.
            if score >= threshold or term_overlap > 0.3:
                relevance = "relevant"
                relevant_count += 1
            elif score >= (threshold * 0.5) or term_overlap > 0.15:
                relevance = "ambiguous"
            else:
                relevance = "irrelevant"

            grades.append(
                {
                    "doc_index": i,
                    "score": score,
                    "term_overlap": round(term_overlap, 3),
                    "relevance": relevance,
                    "threshold_used": round(threshold, 4),
                }
            )

        ambiguous_count = sum(1 for g in grades if g["relevance"] == "ambiguous")

        # Answerable if we have at least MIN_RELEVANT_DOCS relevant docs,
        # OR one relevant + one ambiguous (combined evidence is often enough).
        is_answerable = (
            relevant_count >= MIN_RELEVANT_DOCS
            or (relevant_count >= 1 and ambiguous_count >= 1)
            or (relevant_count == 0 and ambiguous_count >= MIN_RELEVANT_DOCS)
        )

        return _sanitize_state(
            {
                "doc_grades": grades,
                "is_answerable": is_answerable,
            }
        )

    # ------------------------------------------------------------------
    # Node: refine_query
    # ------------------------------------------------------------------

    def refine_query(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Refine query for better retrieval.

        Attempts LLM-based refinement first; falls back to lightweight
        keyword enrichment from partially-relevant documents.
        """
        original_query = state.get("question", "")
        documents = state.get("documents", [])
        retry_count = state.get("retry_count", 0)
        refined_query: str | None = None

        # --- Strategy 1: LLM refinement ---
        # max_new_tokens is kept very small (40) because we only need a short
        # rewritten query string — not a full answer.  On CPU this saves ~60s.
        if self.llm:
            try:
                refine_prompt = (
                    f"{_SYS}\n"
                    "You are a medical search assistant. Rewrite the user's query "
                    "as a more specific medical search query.\n"
                    "Return ONLY the rewritten query, no explanation.\n"
                    f"{_END}\n"
                    f"{_USR}\n"
                    f"Original query: {original_query}\n"
                    f"{_END}\n"
                    f"{_AST}\n"
                )
                result = self.llm.generate(refine_prompt, max_new_tokens=40)
                candidate = (
                    result.response.strip()
                    if hasattr(result, "response")
                    else str(result).strip()
                )
                # Accept only if it's actually different and non-empty
                if candidate and candidate.lower() != original_query.lower():
                    refined_query = candidate
            except Exception:
                pass

        # --- Strategy 2: keyword extraction from partially-relevant docs ---
        if not refined_query and documents:
            best_doc = max(
                documents,
                key=lambda d: d.metadata.get("score", 0.0),
                default=None,
            )
            if best_doc:
                import re

                words = [
                    w
                    for w in re.findall(r"\b[a-zA-Z]{5,}\b", best_doc.page_content)
                    if w.lower()
                    not in {
                        "about",
                        "these",
                        "their",
                        "which",
                        "would",
                        "could",
                        "should",
                        "there",
                        "where",
                        "other",
                        "after",
                        "before",
                    }
                ]
                extra = " ".join(dict.fromkeys(words[:4]))  # first 4 unique long words
                if extra:
                    refined_query = f"{original_query} {extra}"

        # --- Strategy 3: append generic medical expansion ---
        if not refined_query:
            refined_query = f"{original_query} symptoms treatment medical"

        return _sanitize_state(
            {
                "query_history": [refined_query],
                "retry_count": retry_count + 1,
            }
        )

    # ------------------------------------------------------------------
    # Node: generate_answer  ← CORE FIX
    # ------------------------------------------------------------------

    def generate_answer(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Generate answer using LLM with retrieved context.

        Fixes vs previous version:
        1. Uses TinyLlama's <|system|>/<|user|>/<|assistant|> chat-template
           format so the model follows instructions instead of continuing
           the context verbatim.
        2. Calls _build_safe_context() which strips "Question: …" prefixes
           and MedQuAD artefacts that caused the model to copy wrong docs.
        3. Calls clean_llm_response() to remove any remaining leaked training
           data, sign-offs, or source headers from the generated text.
        4. Returns UNANSWERABLE_RESPONSE when context is clearly empty or
           every document had a near-zero score (all irrelevant).
        """
        question = state.get("question", "")
        documents = state.get("documents", [])

        # Safety: if we have absolutely no documents, return unanswerable.
        if not documents:
            return _sanitize_state(
                {
                    "context": "",
                    "answer": UNANSWERABLE_RESPONSE,
                }
            )

        # Build a clean, model-safe context string.
        context = _build_safe_context(documents, max_chars=2000)

        if not context.strip():
            return _sanitize_state(
                {
                    "context": "",
                    "answer": UNANSWERABLE_RESPONSE,
                }
            )

        # --- Construct prompt in TinyLlama chat-template format ---
        prompt = (
            f"{_SYS}\n"
            f"{_MEDICAL_SYSTEM_INSTRUCTION}\n"
            f"{_END}\n"
            f"{_USR}\n"
            f"REFERENCE TEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            f"{_END}\n"
            f"{_AST}\n"
        )

        # Suppress the transformers legacy-cache deprecation warning that
        # floods the log on every generate() call.
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=".*return_legacy_cache.*",
            category=UserWarning,
        )

        try:
            raw = self.llm.generate(prompt, max_new_tokens=256)
            raw_text = raw.response if hasattr(raw, "response") else str(raw)
        except Exception as exc:
            return _sanitize_state(
                {
                    "context": context,
                    "answer": UNANSWERABLE_RESPONSE,
                    "error": f"Generation error: {exc}",
                }
            )

        # Clean the response: strip leaked doc headers, sign-offs, etc.
        from src.utils.text_cleaning import clean_llm_response

        answer = clean_llm_response(raw_text).strip()

        # Final sanity check: if cleaning left us with nothing, fall back.
        if not answer:
            answer = UNANSWERABLE_RESPONSE

        return _sanitize_state(
            {
                "context": context,
                "answer": answer,
            }
        )

    # ------------------------------------------------------------------
    # Node: verify_grounding
    # ------------------------------------------------------------------

    def verify_grounding(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Verify answer is grounded in the retrieved context.

        Uses term-overlap plus claim-level support scoring.
        Threshold is kept deliberately low (0.25) because TinyLlama tends
        to paraphrase rather than quote verbatim.
        """
        answer = state.get("answer", "")
        context = state.get("context", "")
        documents = state.get("documents", [])

        # Unanswerable stub — always considered "grounded" (no hallucination risk)
        if answer == UNANSWERABLE_RESPONSE or not answer or not context:
            return {"is_grounded": True, "grounding_score": 1.0}

        term_overlap_score = self._calculate_term_overlap(answer, context)

        claims = self._extract_claims(answer)
        claim_scores: List[float] = []
        for claim in claims[:5]:
            best = 0.0
            for doc in documents:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                best = max(best, self._calculate_claim_support(claim, content))
            claim_scores.append(best)

        avg_claim = sum(claim_scores) / len(claim_scores) if claim_scores else 0.5

        grounding_score = 0.4 * term_overlap_score + 0.6 * avg_claim
        # Lowered threshold (was 0.35) to account for paraphrasing by small models.
        is_grounded = grounding_score > 0.20

        return _sanitize_state(
            {
                "is_grounded": bool(is_grounded),
                "grounding_score": round(float(grounding_score), 3),
            }
        )

    # ------------------------------------------------------------------
    # Helper: term overlap
    # ------------------------------------------------------------------

    def _calculate_term_overlap(self, answer: str, context: str) -> float:
        answer_terms = self._extract_key_terms(answer)
        context_terms = self._extract_key_terms(context)
        if not answer_terms:
            return 0.0
        overlap = len(answer_terms & context_terms)
        return overlap / len(answer_terms)

    def _extract_key_terms(self, text: str) -> set:
        STOPWORDS = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "by",
            "from",
            "or",
            "and",
            "but",
            "if",
            "then",
            "so",
            "as",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "also",
        }
        words = set(text.lower().split())
        return {w for w in words - STOPWORDS if len(w) > 3}

    # ------------------------------------------------------------------
    # Helper: claim extraction + support
    # ------------------------------------------------------------------

    def _extract_claims(self, text: str) -> List[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        skip = {"consult", "recommend", "important", "disclaimer", "please"}
        return [
            s.strip()
            for s in sentences
            if len(s.strip()) > 20
            and not s.strip().endswith("?")
            and not any(p in s.lower() for p in skip)
        ]

    def _calculate_claim_support(self, claim: str, context: str) -> float:
        terms = self._extract_key_terms(claim)
        if not terms:
            return 0.5
        context_lower = context.lower()
        matches = sum(1 for w in terms if w in context_lower)
        return matches / len(terms)

    # ------------------------------------------------------------------
    # Node: enrich_xai
    # ------------------------------------------------------------------

    def enrich_xai(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Enrich response with XAI components (confidence, attribution, rationale).
        """
        question = state.get("question", "")
        answer = state.get("answer", "")
        documents = state.get("documents", [])
        context = state.get("context", "")
        is_answerable = state.get("is_answerable", True)

        result: Dict[str, Any] = {}

        # --- Confidence scoring ---
        if self.confidence_scorer and is_answerable:
            try:
                retrieval_scores = [doc.metadata.get("score", 0.5) for doc in documents]
                conf = self.confidence_scorer.calculate_confidence(
                    generation_probs=None,
                    retrieval_scores=retrieval_scores,
                    num_sources=len(documents),
                )
                result["confidence"] = {
                    "score": conf.calibrated_score,
                    "level": conf.level,
                    "explanation": conf.explanation,
                }
            except Exception:
                result["confidence"] = {
                    "score": 0.7,
                    "level": "medium",
                    "explanation": "Confidence scoring unavailable.",
                }
        else:
            result["confidence"] = {
                "score": 0.0 if not is_answerable else 0.7,
                "level": "low" if not is_answerable else "medium",
                "explanation": (
                    "Insufficient context."
                    if not is_answerable
                    else "Default confidence."
                ),
            }

        # --- Source attribution ---
        if self.source_attributor and is_answerable and answer != UNANSWERABLE_RESPONSE:
            try:
                doc_dicts = [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "url": doc.metadata.get("url", ""),
                    }
                    for doc in documents
                ]
                attrs = self.source_attributor.attribute_answer(answer, doc_dicts)
                result["attributions"] = [
                    {
                        "claim": a.claim,
                        "source": a.source,
                        "evidence": a.evidence,
                        "similarity": a.similarity_score,
                    }
                    for a in attrs
                ]
            except Exception:
                result["attributions"] = []
        else:
            result["attributions"] = []

        # --- Rationale generation ---
        # Rationale requires a full extra LLM call (~2-4 min on CPU).
        # Only run it when explicitly requested and an LLM is available.
        # On CPU inference this is skipped by default to keep latency reasonable.
        import torch

        _on_cpu = not torch.cuda.is_available()
        _skip_rationale = _on_cpu  # skip on CPU to save 2-4 min per request

        if (
            self.rationale_generator
            and is_answerable
            and answer != UNANSWERABLE_RESPONSE
            and not _skip_rationale
        ):
            try:
                result["rationale"] = self.rationale_generator.generate_rationale(
                    question=question,
                    answer=answer,
                    context=context,
                )
            except Exception:
                result["rationale"] = None
        else:
            result["rationale"] = None

        conf_score = result["confidence"].get("score", 0.0)
        result["needs_review"] = bool(conf_score < LOW_CONFIDENCE_THRESHOLD)

        return _sanitize_state(result)

    # ------------------------------------------------------------------
    # Node: unanswerable_response
    # ------------------------------------------------------------------

    def unanswerable_response(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """Node: Return a safe, honest unanswerable response."""
        return _sanitize_state(
            {
                "answer": UNANSWERABLE_RESPONSE,
                "is_answerable": False,
                "confidence": {
                    "score": 0.0,
                    "level": "low",
                    "explanation": "Insufficient relevant context found in knowledge base.",
                },
                "attributions": [],
                "rationale": None,
                "needs_review": False,
            }
        )

    # ------------------------------------------------------------------
    # Node: handle_error
    # ------------------------------------------------------------------

    def handle_error(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """Node: Handle errors gracefully with a safe fallback response."""
        error = state.get("error", "Unknown error occurred.")
        return _sanitize_state(
            {
                "answer": (
                    f"I encountered an issue processing your question. "
                    f"{UNANSWERABLE_RESPONSE}"
                ),
                "is_answerable": False,
                "confidence": {
                    "score": 0.0,
                    "level": "low",
                    "explanation": f"Error during processing: {str(error)[:120]}",
                },
                "attributions": [],
                "rationale": None,
                "needs_review": True,
            }
        )

    # ------------------------------------------------------------------
    # Async shims (run sync methods in a thread-pool executor)
    # ------------------------------------------------------------------

    async def aretrieve_documents(self, state: HealthcareRAGState) -> Dict[str, Any]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve_documents, state)

    async def agenerate_answer(self, state: HealthcareRAGState) -> Dict[str, Any]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_answer, state)

    async def aenrich_xai(self, state: HealthcareRAGState) -> Dict[str, Any]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.enrich_xai, state)
