"""
Shared evaluation utilities for Healthcare QA Chatbot.

Provides:
  - Unified correctness definition across all eval scripts
  - Bootstrap confidence intervals for statistical significance
  - Keyword coverage computation
"""
import re
import numpy as np
from typing import List, Tuple, Optional


# ── Unified correctness definition ───────────────────────────────────────────
# Correctness = keyword_coverage >= 0.4 AND ROUGE-L >= 0.2 (when available)

def is_correct(keyword_coverage: float, rouge_l: float = None) -> bool:
    """Unified correctness definition for all evaluation modes.

    Args:
        keyword_coverage: Fraction of expected keywords found in the answer.
        rouge_l: Optional ROUGE-L F1 score.

    Returns:
        True if the answer meets the correctness threshold.
    """
    if rouge_l is not None:
        return keyword_coverage >= 0.4 and rouge_l >= 0.2
    return keyword_coverage >= 0.4  # fallback when ROUGE-L not available


def keyword_coverage(answer: str, keywords: list) -> float:
    """Compute fraction of expected keywords present in the answer.

    Args:
        answer: Model-generated answer text.
        keywords: List of expected domain keywords.

    Returns:
        Coverage ratio in [0, 1]. Returns 1.0 if keywords list is empty.
    """
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(1 for k in keywords if re.search(r'\b' + re.escape(k.lower()) + r'\b', a)) / len(keywords)


# ── Bootstrap confidence intervals ──────────────────────────────────────────

def bootstrap_ci(
    scores: list,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a list of scores.

    Args:
        scores: List of numeric scores.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    scores = np.array(scores, dtype=float)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    lower = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return float(np.mean(scores)), lower, upper


# ── Stop words for keyword cleaning ─────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must",
    "and", "but", "or", "nor", "not", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "his", "her", "their", "them", "us",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "some", "any",
    "no", "than", "too", "very", "just", "also", "about", "up", "out",
    "if", "then", "there", "here", "such", "only", "own", "same",
    "other", "now", "well", "also", "like", "even", "back", "still",
    "way", "much", "many", "get", "go", "make", "see", "know",
    "take", "come", "think", "look", "want", "give", "use", "find",
    "tell", "ask", "work", "call", "try", "need", "feel", "become",
    "leave", "put", "mean", "keep", "let", "begin", "seem", "help",
    "show", "hear", "play", "run", "move", "live", "believe", "bring",
    "happen", "write", "provide", "sit", "stand", "lose", "pay",
    "meet", "include", "continue", "set", "learn", "change", "lead",
    "understand", "watch", "follow", "stop", "create", "speak", "read",
    "allow", "add", "grow", "open", "walk", "win", "offer", "remember",
    "consider", "appear", "buy", "wait", "serve", "die", "send",
    "build", "stay", "fall", "cut", "reach", "kill", "remain",
    "chat", "leading", "cause", "causes", "causing", "caused",
    "common", "often", "usually", "including", "however", "although",
    "because", "since", "while", "whether", "though",
}


def is_medical_keyword(word: str) -> bool:
    """Check if a word is medically meaningful (not a stop word, >= 3 chars)."""
    w = word.lower().strip()
    if len(w) < 3:
        return False
    if w in STOP_WORDS:
        return False
    return True
