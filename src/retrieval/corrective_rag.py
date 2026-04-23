"""
Corrective RAG (CRAG) - Validates and improves retrieval quality.

Based on rag-agent-builder and rag-architecture skill patterns.
Implements iterative retrieval refinement when initial results are poor.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger


@dataclass
class RelevanceScore:
    """Document relevance assessment."""

    document_id: str
    relevance: str  # "relevant", "ambiguous", "irrelevant"
    score: float
    reason: str


class CorrectiveRAG:
    """
    Corrective RAG implementation for improved retrieval quality.

    Features:
    - Evaluates document relevance before generation
    - Re-retrieves with modified queries if needed
    - Supports multiple retrieval strategies
    """

    # Thresholds assume HybridRetriever ``normalize_retrieval_scores`` output in [0, 1].
    HIGH_RELEVANCE_THRESHOLD = 0.7
    LOW_RELEVANCE_THRESHOLD = 0.4

    def __init__(self, retriever, llm=None, max_iterations: int = 2, query_enhancer=None):
        """
        Initialize Corrective RAG.

        Args:
            retriever: Base retriever for document search
            llm: Optional LLM for relevance grading
            max_iterations: Maximum retrieval refinement iterations
            query_enhancer: Optional QueryEnhancer instance. When provided,
                            retry queries use abbreviation expansion and
                            synonym-based refinement instead of naive keyword
                            extraction from the top document chunk.
        """
        self.retriever = retriever
        self.llm = llm
        self.max_iterations = max_iterations
        self.query_enhancer = query_enhancer

    def retrieve_with_correction(
        self, query: str, k: int = 5, min_relevant_docs: int = 2
    ) -> Tuple[List, bool]:
        """
        Retrieve documents with automatic correction.

        Args:
            query: User query
            k: Number of documents to retrieve
            min_relevant_docs: Minimum relevant docs required

        Returns:
            Tuple of (documents, is_corrected)
        """
        documents = self.retriever.retrieve(query, k=k)

        if not documents:
            return [], False

        # Grade documents for relevance
        grades = self._grade_documents(query, documents)

        # Count relevant documents
        relevant_count = sum(1 for g in grades if g.relevance == "relevant")

        # If enough relevant docs, return
        if relevant_count >= min_relevant_docs:
            return documents, False

        # Try to correct with refined query
        for iteration in range(self.max_iterations):
            refined_query = self._refine_query(query, documents, grades)

            if refined_query and refined_query != query:
                new_documents = self.retriever.retrieve(refined_query, k=k)
                new_grades = self._grade_documents(refined_query, new_documents)

                new_relevant_count = sum(1 for g in new_grades if g.relevance == "relevant")

                # If improvement, use new results
                if new_relevant_count > relevant_count:
                    documents = new_documents
                    grades = new_grades
                    relevant_count = new_relevant_count

                    if relevant_count >= min_relevant_docs:
                        return documents, True

        # Return best effort
        return documents, True

    def _grade_documents(self, query: str, documents: List) -> List[RelevanceScore]:
        """
        Grade documents for relevance to query.

        Uses retrieval score + content analysis.
        """
        grades = []

        for i, doc in enumerate(documents):
            score = doc.score if hasattr(doc, "score") else 0.5

            # Determine relevance level
            if score >= self.HIGH_RELEVANCE_THRESHOLD:
                relevance = "relevant"
                reason = "High similarity score"
            elif score >= self.LOW_RELEVANCE_THRESHOLD:
                relevance = "ambiguous"
                reason = "Moderate similarity score"
            else:
                relevance = "irrelevant"
                reason = "Low similarity score"

            # Additional keyword check for medical queries
            content = doc.content if hasattr(doc, "content") else str(doc)
            query_terms = set(query.lower().split())
            content_terms = set(content.lower().split())

            term_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            if term_overlap > 0.5 and relevance == "ambiguous":
                relevance = "relevant"
                reason = "High term overlap"

            grades.append(
                RelevanceScore(document_id=str(i), relevance=relevance, score=score, reason=reason)
            )

        return grades

    def _refine_query(
        self, original_query: str, documents: List, grades: List[RelevanceScore]
    ) -> Optional[str]:
        """
        Refine query based on feedback from document grades.
        """
        # Strategy 1: QueryEnhancer abbreviation + synonym expansion (preferred)
        if self.query_enhancer is not None:
            try:
                enhanced = self.query_enhancer.enhance(original_query)
                # Take just the expanded query text (not decomposed sub-questions)
                refined = enhanced.enhanced_query if hasattr(enhanced, "enhanced_query") else str(enhanced)
                if refined and refined.lower().strip() != original_query.lower().strip():
                    logger.debug(f"CorrectiveRAG: QueryEnhancer refined '{original_query}' -> '{refined[:80]}'")
                    return refined
            except Exception as e:
                logger.debug(f"CorrectiveRAG: QueryEnhancer failed, trying fallback: {e}")

        # Strategy 2: LLM-based reformulation
        if self.llm:
            prompt = f"""The search query "{original_query}" returned documents that weren't relevant enough.

Suggest a more specific query that might find better results.
Return only the refined query, nothing else."""

            try:
                response = self.llm.generate(prompt, max_new_tokens=50)
                return response.response.strip()
            except Exception as e:
                logger.warning(f"LLM-based query reformulation failed: {e}")

        # Strategy 3: Abbreviation-based expansion (lightweight fallback without QueryEnhancer)
        # Prefer expanding known abbreviations over extracting random document words.
        _COMMON_EXPANSIONS = {
            "COPD": "chronic obstructive pulmonary disease",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "MI": "myocardial infarction",
            "CVA": "cerebrovascular accident stroke",
            "UTI": "urinary tract infection",
            "CAD": "coronary artery disease",
            "CHF": "congestive heart failure",
            "CKD": "chronic kidney disease",
            "GERD": "gastroesophageal reflux disease",
        }
        for abbr, expansion in _COMMON_EXPANSIONS.items():
            if abbr in original_query and expansion not in original_query:
                return f"{original_query} {expansion}"

        # Strategy 4: Extract key terms from top partially-relevant doc
        relevant_docs = [
            doc for i, doc in enumerate(documents) if grades[i].relevance != "irrelevant"
        ]
        if relevant_docs:
            import re as _re
            content = relevant_docs[0].content if hasattr(relevant_docs[0], "content") else str(relevant_docs[0])
            # Use first few meaningful long words (>5 chars) not already in query
            query_words = set(original_query.lower().split())
            keywords = [
                w for w in _re.findall(r"\b[a-zA-Z]{5,}\b", content)
                if w.lower() not in query_words
            ][:3]
            if keywords:
                return original_query + " " + " ".join(keywords)

        return None

    def get_action_decision(self, grades: List[RelevanceScore]) -> str:
        """
        Determine action based on document relevance.

        Returns:
            "proceed" - Generate answer with current docs
            "refine" - Try different retrieval strategy
            "fallback" - Use web search or other fallback
        """
        relevant_count = sum(1 for g in grades if g.relevance == "relevant")
        ambiguous_count = sum(1 for g in grades if g.relevance == "ambiguous")

        if relevant_count >= 2:
            return "proceed"
        elif relevant_count + ambiguous_count >= 2:
            return "proceed"  # Can try with ambiguous docs
        elif relevant_count + ambiguous_count >= 1:
            return "refine"  # Need to try harder
        else:
            return "fallback"  # Need external sources
