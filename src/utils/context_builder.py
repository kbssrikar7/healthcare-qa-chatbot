"""
Shared context builder utility for safe prompt construction.

Eliminates duplication between the LangChain and LangGraph pipelines.
Both ``_build_safe_context_lc`` (langchain_pipeline.py) and
``_build_safe_context`` (langgraph_nodes.py) were functionally identical —
this module is the single source of truth.

Key design decisions
--------------------
* Source labels are placed on a numbered line (``[1] Source: MedQuAD``)
  rather than as a content prefix.  The old ``[Source: MedQuAD (relevance: 0.02)]``
  format confused TinyLlama, which tried to continue the pattern as a
  Q&A template.
* Leading "Question: …" and "[SomeSource]: " artefacts that appear at the
  top of many MedQuAD / MedMCQA chunks are stripped before the content
  is placed into the prompt.
* A strict character budget is enforced so the assembled context stays
  within the model's context window.

Usage
-----
>>> from src.utils.context_builder import build_safe_context
>>> context = build_safe_context(documents, max_chars=2000)
"""

from __future__ import annotations

import re
from typing import Any, List

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_safe_context(
    documents: List[Any],
    max_chars: int = 2000,
) -> str:
    """
    Convert a list of documents into a context string that is safe to pass
    directly to a small-parameter LLM such as TinyLlama.

    Accepted document types
    -----------------------
    * LangChain ``Document`` objects  — uses ``.page_content`` and ``.metadata``
    * ``RetrievedDocument`` dataclass — uses ``.content`` and ``.source``
    * Plain ``dict``                  — uses keys ``"content"`` / ``"source"``
    * Any other object                — coerced to ``str()``

    Parameters
    ----------
    documents : list
        Retrieved documents in any of the formats listed above.
    max_chars : int
        Maximum total character budget for the assembled context string.
        Blocks that would exceed this limit are truncated (with an ellipsis)
        or skipped if the remaining budget is too small to be useful.

    Returns
    -------
    str
        A numbered, newline-separated context string ready for prompt
        insertion.
    """
    parts: List[str] = []
    used = 0

    for i, doc in enumerate(documents, 1):
        content, source = _extract_content_and_source(doc, i)

        # Strip common MedQuAD / MedMCQA artefacts from the start of content
        content = _strip_leading_artefacts(content)

        block = f"[{i}] Source: {source}\n{content}"

        if used + len(block) > max_chars:
            remaining = max_chars - used
            # Only include a truncated block if there is enough budget to be
            # meaningful; otherwise stop adding blocks.
            if remaining > 100:
                block = block[:remaining] + "..."
                parts.append(block)
            break

        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content_and_source(doc: Any, fallback_index: int) -> tuple[str, str]:
    """
    Extract (content, source) from any supported document type.

    Parameters
    ----------
    doc            : Document object, dict, or any value.
    fallback_index : Used to build a generic source label when no source
                     metadata is available.

    Returns
    -------
    tuple[str, str]
        (content text, source label)
    """
    fallback_source = f"Document {fallback_index}"

    # LangChain Document
    if hasattr(doc, "page_content"):
        content = str(doc.page_content).strip()
        source = (
            doc.metadata.get("source", fallback_source)
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict)
            else fallback_source
        )
        return content, source

    # RetrievedDocument dataclass (src/retrieval/hybrid_retriever.py)
    if hasattr(doc, "content") and hasattr(doc, "source"):
        content = str(doc.content).strip()
        source = str(doc.source) if doc.source else fallback_source
        return content, source

    # Plain dict
    if isinstance(doc, dict):
        content = str(doc.get("content", "")).strip()
        source = str(doc.get("source", fallback_source))
        return content, source

    # Fallback: stringify the whole object
    return str(doc).strip(), fallback_source


def _strip_leading_artefacts(content: str) -> str:
    """
    Remove common MedQuAD / MedMCQA / ChatDoctor artefacts so that TinyLlama
    receives clean factual text — not raw Q&A pairs it will parrot back.

    Patterns handled
    ----------------
    * ``[SomeLabel]: …``          — source-label prefix from ingestion
    * ``Question: … Answer: …``  — MedQA-USMLE Q&A pair format.  Extracts ONLY
                                    the Answer portion (the actual medical fact)
    * ``Answer: nan``             — ChatDoctor garbage (literal "nan" string)
    * Multi-choice options (A. / B. / C. / D.) — MedMCQA exam-format leakage
    * Leading ``Question:`` prefix alone

    The key insight: The knowledge base stores chunks as
    ``Question: <clinical vignette> Answer: <medical fact>``.
    Passing the entire chunk to TinyLlama causes two problems:
      1. TinyLlama echoes the exam-question vignette as its answer
      2. Post-processing ``AGGRESSIVE_STOP_PATTERNS`` then truncates the
         actual answer because it contains exam-format language

    Solution: extract only the Answer portion and discard the Question vignette.
    """
    # Remove "[SomeLabel]: " prefix (non-greedy match inside brackets)
    content = re.sub(r"^\s*\[.*?\]\s*:\s*", "", content)

    # ─── Handle "Question: ... Answer: ..." format ─────────────────────
    # This is the critical fix: MedQA-USMLE chunks are stored as:
    #   "Question: A 56-year-old... Answer: ACE inhibitor, ARB, CCB, or thiazide"
    # We extract ONLY the Answer portion.
    answer_match = re.search(r"\bAnswer\s*:\s*(.+)", content, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Guard: "Answer: nan" is garbage from ChatDoctor; discard
        if answer_text.lower() in ("nan", "none", "n/a", "na", "null", ""):
            # Fall through — strip the Question: prefix below and keep the question
            # text (it may contain useful context)
            pass
        else:
            # Strip multi-choice options from the extracted answer
            # e.g. "A. Lisinopril B. Amlodipine C. HCTZ D. All of the above"
            answer_text = re.sub(
                r"\b[A-E]\.\s+[A-Z][a-zA-Z,\- ]{2,40}\s*(?=\b[A-E]\.|\Z)",
                "", answer_text,
            ).strip()
            if answer_text:
                content = answer_text

    # Remove leading "Question:" prefix if still present (no Answer: found, or
    # Answer was garbage and we fell through)
    content = re.sub(r"^\s*Question\s*:\s*", "", content, flags=re.IGNORECASE)
    return content.strip()


# ---------------------------------------------------------------------------
# Convenience wrapper that mirrors the old function signatures exactly,
# so existing call sites can be updated with a one-line import change.
# ---------------------------------------------------------------------------


def build_safe_context_lc(
    docs: List[Any],
    max_chars: int = 2000,
) -> str:
    """
    Alias kept for backward compatibility with the LangChain pipeline.

    Previously defined as ``_build_safe_context_lc`` in
    ``src/langchain/langchain_pipeline.py``.  New code should use
    :func:`build_safe_context` directly.
    """
    return build_safe_context(docs, max_chars=max_chars)


def build_safe_context_lg(
    documents: List[Any],
    max_chars: int = 2000,
) -> str:
    """
    Alias kept for backward compatibility with the LangGraph node helpers.

    Previously defined as ``_build_safe_context`` in
    ``src/langgraph/langgraph_nodes.py``.  New code should use
    :func:`build_safe_context` directly.
    """
    return build_safe_context(documents, max_chars=max_chars)
