"""Backward-compatible wrapper around QueryEnhancer."""

from typing import List, Set

from src.retrieval.query_enhancer import QueryEnhancer


class MedicalQueryExpander:
    """Compatibility shim; prefer ``QueryEnhancer`` directly."""

    def __init__(self, max_expansions: int = 3):
        self._enhancer = QueryEnhancer(max_expansions=max_expansions)

    def expand(self, query: str) -> List[str]:
        enhanced = self._enhancer.enhance(query)
        return enhanced.all_queries

    def get_expanded_terms(self, query: str) -> Set[str]:
        query_lower = query.lower()
        terms: Set[str] = set()

        for term, synonyms in self._enhancer.MEDICAL_SYNONYMS.items():
            if term in query_lower:
                terms.update(synonyms)

        for token in query_lower.split():
            clean = "".join(ch for ch in token if ch.isalnum())
            expanded = self._enhancer.MEDICAL_EXPANSIONS.get(clean)
            if expanded:
                terms.add(expanded)

        return terms
