"""
Shared text cleaning utilities for LLM response post-processing.

Consolidates duplicate cleaning logic from MedicalLLM._clean_response()
and HealthcareQAPipeline._clean_answer(). Both modules now delegate here.
"""

import re
from typing import List

# Prefixes that the model sometimes prepends to answers
ANSWER_PREFIXES = [
    "Answer:",
    "Factual Answer:",
    "Evidence-Based Answer:",
    "Based on the reference text above, the answer is:",
    "Based on the reference text above, the evidence-based answer is:",
    "Based on the reference text,",
]

# Patterns indicating leaked training data, new Q&A, or sign-offs.
# If found beyond min_match_pos, the response is truncated there.
STOP_PATTERNS = [
    r"\nQuestion:",
    r"\nQ:",
    r"\nAnswer:",
    r"Best regards",
    r"Kind regards",
    r"Sincerely",
    r"Yours truly",
    r"Warm regards",
    r"With best wishes",
    r"\[Your Name\]",
    r"[Doctor'?s? Name]",
    r"Chat Doctor",
    r"ChatDoctor",
    r"HealthCareMagic",
    r"Thank you for choosing",
    r"Thank you for using",
    r"Thank you for reaching out",
    r"Thank you for contacting",
    r"If you have any further questions",
    r"please do not hesitate",
    r"don't hesitate to ask",
    r"I hope this (?:helps|information|answers)",
    r"Wishing you (?:good|the best)",
    r"Take care",
    r"\nHi,?\s",
    r"\nHello,?\s",
    r"\nDear ",
    r"\nHi doctor",
    r"\nHello doctor",
    r"\nHi,\s*\n",
    r"\[\d+\]\s*Source:",
    r"\n---",
    r"<\|",
    r"\[/INST\]",
    r"</s>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
]


def clean_llm_response(
    response: str,
    stop_patterns: List[str] | None = None,
    min_match_pos: int = 50,
) -> str:
    """
    Clean an LLM response by removing training-data leakage.

    Steps:
    1. Strip known answer-prefix headers
    2. Truncate at earliest stop-pattern match (after *min_match_pos* chars)
    3. Remove citation markers like ``[1]``
    4. Trim to last complete sentence if response ends mid-word
    5. Collapse excess whitespace

    Parameters
    ----------
    response : Raw LLM text.
    stop_patterns : Override default stop patterns (list of regex strings).
    min_match_pos : Don't truncate if match is within the first N chars.

    Returns
    -------
    Cleaned response string.
    """
    if not response:
        return response

    response = response.strip()

    # 1. Strip prefix headers
    for prefix in ANSWER_PREFIXES:
        if response.startswith(prefix):
            response = response[len(prefix) :].strip()

    # 2. Truncate at earliest stop-pattern
    patterns = stop_patterns if stop_patterns is not None else STOP_PATTERNS
    safe_min = min(min_match_pos, len(response))
    earliest_pos = len(response)

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.start() >= safe_min and match.start() < earliest_pos:
            earliest_pos = match.start()

    if earliest_pos < len(response):
        response = response[:earliest_pos]

    # 3. Remove citation markers [1], [2], etc.
    response = re.sub(r"\[\d+\]", "", response)

    # 4. Trim to last sentence boundary if cut mid-word
    response = response.rstrip()
    if response and response[-1] not in '.!?:)"':
        last_period = max(
            response.rfind("."),
            response.rfind("!"),
            response.rfind("?"),
        )
        if last_period > len(response) * 0.3:
            response = response[: last_period + 1]

    # 5. Collapse whitespace
    response = re.sub(r"  +", " ", response)
    response = re.sub(r"\n\n\n+", "\n\n", response)

    return response.strip()
