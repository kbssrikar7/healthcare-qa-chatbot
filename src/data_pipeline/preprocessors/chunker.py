"""
Document chunking for medical texts with context preservation.

Classes
-------
MedicalTextChunker
    Original fixed-size chunker (kept for backward compatibility).
RecursiveSentenceChunker
    Improved recursive chunker that respects sentence boundaries using a
    separator hierarchy.  This is the chunker to use for KB v2 builds.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Represents a document chunk."""

    content: str
    source: str
    chunk_id: int
    total_chunks: int
    metadata: Dict[str, Any]


class MedicalTextChunker:
    """
    Chunk medical documents while preserving context.

    Features (Phase 2 improvements):
    - Domain-adaptive chunk sizes based on document type
    - Sentence-boundary awareness (never splits mid-sentence)
    - Recursive chunking with overlaps
    """

    # Domain-specific chunk sizes in characters
    # (roughly 4 chars/token: 256 chars ≈ 64 tokens).
    DOMAIN_CHUNK_SIZES = {
        "pubmedqa": 256,  # ~64 tokens, dense abstracts
        "medmcqa": 128,  # ~32 tokens, short Q&A pairs
        "healthcaremagic": 512,  # ~128 tokens, long consultations
        "clinical_guideline": 768,  # ~192 tokens, context-heavy guidelines
        "default": 512,  # ~128 tokens baseline
    }

    def __init__(
        self,
        chunk_size: int = None,  # Now optional - auto-determined by source
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        use_domain_adaptive: bool = True,
    ):
        self.default_chunk_size = chunk_size or 512
        # Keep a stable base chunk size on the instance. Domain-adaptive chunking
        # may override this temporarily for a given document, but direct calls to
        # chunk_text() must still work without going through chunk_document().
        self.chunk_size = self.default_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_domain_adaptive = use_domain_adaptive

        # Separators in order of priority
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            "; ",  # Clauses
            ", ",  # Phrases
            " ",  # Words
        ]

    def _get_chunk_size_for_source(self, source: str) -> int:
        """Determine optimal chunk size based on document source."""
        if not self.use_domain_adaptive:
            return self.default_chunk_size

        source_lower = source.lower()
        for key, size in self.DOMAIN_CHUNK_SIZES.items():
            if key in source_lower:
                return size
        return self.DOMAIN_CHUNK_SIZES["default"]

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []

        chunk_size = getattr(self, "chunk_size", self.default_chunk_size)

        if len(text) <= chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []

        chunks = []
        current_chunk = ""

        # Split by sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Handle very long sentences
                if len(sentence) > chunk_size:
                    # Split by words
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= chunk_size:
                            current_chunk += (" " if current_chunk else "") + word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word
                else:
                    current_chunk = sentence

        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        # Add overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        overlapped = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get last N characters from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = (
                    prev_chunk[-self.chunk_overlap :]
                    if len(prev_chunk) > self.chunk_overlap
                    else prev_chunk
                )

                # Find word boundary
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1 :]

                overlapped.append(overlap_text + " " + chunk)

        return overlapped

    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a document with metadata.

        Uses domain-adaptive chunk sizing: determines optimal chunk size
        based on document source type (e.g., PubMedQA vs HealthCareMagic).
        """
        content = document.get("content", "")
        source = document.get("source", "unknown")
        metadata = document.get("metadata", {})

        # Domain-adaptive chunk size
        original_chunk_size = (
            self.chunk_size if hasattr(self, "chunk_size") else self.default_chunk_size
        )
        if self.use_domain_adaptive:
            self.chunk_size = self._get_chunk_size_for_source(source)
        else:
            self.chunk_size = self.default_chunk_size

        text_chunks = self.chunk_text(content)

        # Restore original if needed (for reusability across sources)
        if self.use_domain_adaptive:
            self.chunk_size = self.default_chunk_size

        return [
            Chunk(
                content=chunk,
                source=source,
                chunk_id=i,
                total_chunks=len(text_chunks),
                metadata={
                    **metadata,
                    "url": document.get("url", ""),
                    "chunk_length": len(chunk),
                    "adaptive_chunk_size": (
                        self._get_chunk_size_for_source(source)
                        if self.use_domain_adaptive
                        else original_chunk_size
                    ),
                },
            )
            for i, chunk in enumerate(text_chunks)
        ]

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


# ---------------------------------------------------------------------------
# RecursiveSentenceChunker — improved chunker for KB v2
# ---------------------------------------------------------------------------


class RecursiveSentenceChunker:
    """
    Recursive text splitter that respects sentence boundaries.

    Algorithm
    ---------
    1. Try to split on the highest-priority separator (paragraph breaks).
    2. If any resulting piece is still too large, recurse with the next
       separator in the hierarchy.
    3. Once all pieces fit within `chunk_size`, merge adjacent pieces while
       keeping the total below the limit, then add overlap.

    Domain-adaptive sizes are applied per document source so the caller
    does not need to know about chunk sizes at all.

    Usage
    -----
    >>> chunker = RecursiveSentenceChunker()
    >>> chunks = chunker.chunk_document({"content": "...", "source": "PubMedQA"})
    """

    # Separator hierarchy: try paragraph first, fall back to finer splits
    _SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    # Domain-specific chunk sizes (characters, ~4 chars/token)
    _DOMAIN_SIZES: Dict[str, int] = {
        "pubmedqa": 256 * 4,  # ~1 024 chars
        "medmcqa": 128 * 4,  # ~  512 chars
        "healthcaremagic": 512 * 4,  # ~2 048 chars
        "clinical_guideline": 768 * 4,  # ~3 072 chars
        "default": 512 * 4,  # ~2 048 chars
    }

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 100,
        min_chunk_size: int = 80,
        use_domain_adaptive: bool = True,
        keep_separator: bool = True,
    ):
        self._default_chunk_size = chunk_size or self._DOMAIN_SIZES["default"]
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._use_domain_adaptive = use_domain_adaptive
        self._keep_separator = keep_separator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single document dict, returning a list of Chunk objects."""
        content = document.get("content", "")
        source = document.get("source", "unknown")
        metadata = document.get("metadata", {})
        if not content:
            return []

        chunk_size = self._size_for_source(source)
        raw_chunks = self._split(content, self._SEPARATORS, chunk_size)

        return [
            Chunk(
                content=c,
                source=source,
                chunk_id=i,
                total_chunks=len(raw_chunks),
                metadata={
                    **metadata,
                    "chunk_length": len(c),
                    "chunker": "recursive",
                    "chunk_size_chars": chunk_size,
                },
            )
            for i, c in enumerate(raw_chunks)
        ]

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _size_for_source(self, source: str) -> int:
        if not self._use_domain_adaptive:
            return self._default_chunk_size
        src = source.lower()
        for key, size in self._DOMAIN_SIZES.items():
            if key != "default" and key in src:
                return size
        return self._DOMAIN_SIZES["default"]

    def _split(self, text: str, separators: List[str], chunk_size: int) -> List[str]:
        """Recursively split *text* using the separator hierarchy."""
        if not text:
            return []

        # Find the first separator that is present in text
        sep = ""
        remaining_seps = separators[:]
        for candidate in separators:
            if candidate == "" or candidate in text:
                sep = candidate
                remaining_seps = separators[separators.index(candidate) + 1 :]
                break

        # Split on chosen separator
        if sep:
            pieces = re.split(re.escape(sep), text)
            if self._keep_separator and sep.strip():
                # Re-attach the separator to the end of each piece
                pieces = [p + sep if i < len(pieces) - 1 else p for i, p in enumerate(pieces)]
        else:
            # No separator left → character-level fallback
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Recurse on pieces that are still too large
        good: List[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= chunk_size:
                good.append(piece)
            else:
                good.extend(self._split(piece, remaining_seps, chunk_size))

        # Merge adjacent small pieces to avoid tiny fragments, then add overlap
        merged = self._merge(good, chunk_size)
        return self._add_overlap(merged) if self._chunk_overlap > 0 else merged

    def _merge(self, pieces: List[str], chunk_size: int) -> List[str]:
        """Greedily merge adjacent pieces to fill chunks up to chunk_size."""
        merged: List[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            candidate = (current + " " + piece).strip() if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if len(current) >= self._min_chunk_size:
                    merged.append(current)
                current = piece
        if len(current) >= self._min_chunk_size:
            merged.append(current)
        return merged

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Prepend the last `_chunk_overlap` chars of chunk[i-1] to chunk[i]."""
        if len(chunks) <= 1:
            return chunks
        result: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self._chunk_overlap :]
            # Trim to word boundary to avoid splitting mid-word
            space = tail.find(" ")
            if space > 0:
                tail = tail[space + 1 :]
            result.append((tail + " " + chunks[i]).strip())
        return result
