"""
Passage Highlighting for XAI.

Identifies which spans in retrieved passages directly support
claims in the generated answer, enabling visual highlighting
in the frontend.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional


@dataclass
class HighlightSpan:
    """A highlighted span within a passage."""

    start: int
    end: int
    matched_claim: str
    match_ratio: float


@dataclass
class HighlightedPassage:
    """A passage with highlighted supporting spans."""

    text: str
    source: str
    relevance_score: float
    highlight_spans: List[HighlightSpan] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "highlight_spans": [
                {
                    "start": s.start,
                    "end": s.end,
                    "matched_claim": s.matched_claim,
                    "match_ratio": round(s.match_ratio, 3),
                }
                for s in self.highlight_spans
            ],
        }


class PassageHighlighter:
    """Identify which parts of retrieved passages directly support the answer.

    Uses sequence matching to find the best-matching spans in source
    passages for each claim in the generated answer.
    """

    def __init__(self, min_match_ratio: float = 0.35, min_span_length: int = 15):
        """
        Args:
            min_match_ratio: Minimum ratio to consider a span a match
            min_span_length: Minimum character length for a matched span
        """
        self.min_match_ratio = min_match_ratio
        self.min_span_length = min_span_length

    def highlight(
        self,
        answer: str,
        passages: List[Dict],
    ) -> List[HighlightedPassage]:
        """
        Identify supporting spans in passages for the given answer.

        Args:
            answer: Generated answer text
            passages: List of passage dicts with 'content', 'source', 'score'

        Returns:
            List of HighlightedPassage with spans marked
        """
        answer_sentences = self._split_sentences(answer)
        results = []

        for passage in passages:
            passage_text = passage.get("content", "")
            spans: List[HighlightSpan] = []

            for sentence in answer_sentences:
                if len(sentence) < 10:
                    continue
                match = self._find_best_match(sentence, passage_text)
                if match:
                    spans.append(match)

            # Merge overlapping spans
            merged_spans = self._merge_spans(spans)

            results.append(
                HighlightedPassage(
                    text=passage_text,
                    source=passage.get("source", "unknown"),
                    relevance_score=passage.get("score", 0.0),
                    highlight_spans=merged_spans,
                )
            )

        return results

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling medical abbreviations."""
        # Protect common abbreviations from splitting
        protected = text
        abbreviations = ["Dr.", "Mr.", "Mrs.", "Ms.", "mg.", "ml.", "vs.", "i.e.", "e.g.", "etc."]
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

        sentences = re.split(r"[.!?]+", protected)
        return [
            s.strip().replace("<DOT>", ".") for s in sentences if s.strip() and len(s.strip()) > 10
        ]

    def _find_best_match(self, needle: str, haystack: str) -> Optional[HighlightSpan]:
        """Find the best matching span in haystack for the needle text."""
        if not needle or not haystack:
            return None

        needle_lower = needle.lower()
        haystack_lower = haystack.lower()

        matcher = SequenceMatcher(None, needle_lower, haystack_lower)
        match = matcher.find_longest_match(0, len(needle_lower), 0, len(haystack_lower))

        if match.size >= self.min_span_length:
            ratio = match.size / len(needle_lower)
            if ratio >= self.min_match_ratio:
                # Expand to word boundaries
                start = match.b
                end = match.b + match.size

                # Snap to word boundaries
                while start > 0 and haystack[start - 1] not in " \n\t":
                    start -= 1
                while end < len(haystack) and haystack[end] not in " \n\t.!?,;:":
                    end += 1

                return HighlightSpan(
                    start=start,
                    end=end,
                    matched_claim=needle[:80],
                    match_ratio=ratio,
                )

        return None

    def _merge_spans(self, spans: List[HighlightSpan]) -> List[HighlightSpan]:
        """Merge overlapping highlight spans."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda s: s.start)
        merged = [sorted_spans[0]]

        for span in sorted_spans[1:]:
            prev = merged[-1]
            if span.start <= prev.end:
                # Merge: extend end, keep best ratio
                merged[-1] = HighlightSpan(
                    start=prev.start,
                    end=max(prev.end, span.end),
                    matched_claim=(
                        prev.matched_claim
                        if prev.match_ratio >= span.match_ratio
                        else span.matched_claim
                    ),
                    match_ratio=max(prev.match_ratio, span.match_ratio),
                )
            else:
                merged.append(span)

        return merged
