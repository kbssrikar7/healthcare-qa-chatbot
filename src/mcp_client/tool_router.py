"""
MCP Tool Router — intelligently selects the right MCP tool based on question type.

Prevents unnecessary API calls by routing:
1. Drug questions → FDA server (authoritative, real-time)
2. Clinical/research questions → PubMed server (peer-reviewed)
3. General medical questions → Brave web search (broad, less curated)
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MCPToolSelection:
    """Result of MCP tool routing decision."""

    tool_server: str  # "pubmed", "fda", "brave"
    tool_name: str  # exact MCP tool name
    tool_args: dict  # arguments to pass
    confidence: float  # routing confidence (0-1)
    reason: str  # human-readable explanation


class MCPToolRouter:
    """
    Routes medical questions to the most appropriate MCP tool.

    Uses keyword-based classification with regex patterns to detect
    question type. This is intentionally simple — an LLM-based router
    would add latency and complexity for minimal gain in this domain.

    Priority order:
    1. Drug questions → FDA server (authoritative, real-time)
    2. Clinical/research questions → PubMed server (peer-reviewed)
    3. General medical questions → Brave search (broad coverage)
    """

    # Drug-related patterns
    DRUG_PATTERNS = [
        r"\b(drug|medication|medicine|pill|tablet|capsule|dose|dosage|prescription)\b",
        r"\b(side effect|adverse effect|interaction|contraindication|overdose)\b",
        r"\b(antibiotic|antidepressant|antihypertensive|statin|insulin|vaccine)\b",
        r"\b(mg|mcg|ml|IV|oral|topical|injection)\b",
        r"\b(FDA approved|recall|warning|black box)\b",
    ]

    # Research/clinical patterns
    RESEARCH_PATTERNS = [
        r"\b(study|trial|research|evidence|guideline|protocol|meta-analysis)\b",
        r"\b(clinical|randomized|controlled|systematic review|cohort)\b",
        r"\b(efficacy|effectiveness|outcome|prognosis|survival rate)\b",
        r"\b(pathophysiology|mechanism|etiology|epidemiology)\b",
    ]

    # General medical condition patterns (route to PubMed, not Brave)
    MEDICAL_PATTERNS = [
        r"\b(symptom|disease|condition|treatment|diagnosis|therapy)\b",
        r"\b(chronic|acute|congenital|autoimmune|infection)\b",
        r"\b(surgery|procedure|screening|prevention)\b",
    ]

    def route(self, question: str) -> MCPToolSelection:
        """
        Select the best MCP tool for a given question.

        Args:
            question: User's medical question

        Returns:
            MCPToolSelection with the chosen tool and arguments
        """
        question_lower = question.lower()

        # Score each category
        drug_score = self._score_patterns(question_lower, self.DRUG_PATTERNS)
        research_score = self._score_patterns(question_lower, self.RESEARCH_PATTERNS)
        medical_score = self._score_patterns(question_lower, self.MEDICAL_PATTERNS)

        # Drug question → FDA
        if drug_score >= 2:
            drug_name = self._extract_drug_name(question)
            return MCPToolSelection(
                tool_server="fda",
                tool_name="fda_drug_search",
                tool_args={"drug_name": drug_name or question[:50]},
                confidence=min(0.9, 0.5 + drug_score * 0.1),
                reason=f"Drug question (matched {drug_score} drug patterns)",
            )

        # Research/clinical question → PubMed
        if research_score >= 1:
            return MCPToolSelection(
                tool_server="pubmed",
                tool_name="pubmed_search",
                tool_args={"query": question, "max_results": 3},
                confidence=min(0.85, 0.5 + research_score * 0.1),
                reason=f"Research question (matched {research_score} research patterns)",
            )

        # General medical question → PubMed (still better than Brave for medical)
        if medical_score >= 1:
            return MCPToolSelection(
                tool_server="pubmed",
                tool_name="pubmed_search",
                tool_args={"query": question, "max_results": 3},
                confidence=0.6,
                reason=f"Medical question (matched {medical_score} medical patterns) → PubMed",
            )

        # Fallback → Brave web search
        return MCPToolSelection(
            tool_server="brave",
            tool_name="brave_web_search",
            tool_args={"query": question + " medical health", "count": 3},
            confidence=0.4,
            reason="No strong domain match — fallback to web search",
        )

    def _score_patterns(self, text: str, patterns: list) -> int:
        """Count how many patterns match in the text."""
        return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

    def _extract_drug_name(self, question: str) -> Optional[str]:
        """Try to extract a drug name from the question."""
        # Look for words before dosage units (e.g., "metformin 500mg")
        mg_match = re.search(r"(\w+)\s+\d+\s*mg", question, re.IGNORECASE)
        if mg_match:
            return mg_match.group(1)

        # Look for words after prescribing verbs
        for trigger in ["taking", "prescribed", "on", "using", "about"]:
            match = re.search(rf"{trigger}\s+(\w+)", question, re.IGNORECASE)
            if match:
                candidate = match.group(1)
                # Filter common non-drug words
                if candidate.lower() not in {"a", "the", "my", "this", "some", "it"}:
                    return candidate

        return None
