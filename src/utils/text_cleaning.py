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

# Patterns that are ALWAYS truncated wherever they appear (min_pos=0).
# These indicate exam-format leakage from MedMCQA / board-exam context chunks.
# Always truncated at any position — definitive exam/dataset leakage markers.
AGGRESSIVE_STOP_PATTERNS = [
    # MedMCQA board-exam format
    r"Per the (?:Joint National|American Heart|WHO|CDC|European)",
    r"of the following (?:combinations|options|choices|drugs|statements)",
    r"[Ww]hich (?:of the following|can be considered|is the most effective|is the best)",
    r"[Aa] \d{2}-year-old (?:man|woman|patient|child|boy|girl) presents",
    r"[Hh]is (?:temperature|blood pressure|pulse|heart rate) is \d",
    r"[Hh]er (?:temperature|blood pressure|pulse|heart rate) is \d",
    r"(?:^|\n)###\s*(?:Question|Answer|Q|A)\s*:",
    # HealthCareMagic / ChatDoctor sign-off loops — cut wherever they appear
    r"[Tt]hank(?:s| you) for (?:your|the) (?:question|query|message|email|time|concern)",
    r"[Tt]hank(?:s| you) for (?:choosing|using|reaching out|contacting|writing|asking)",
    r"I hope this ",
    r"I hope (?:I have|I've|that) (?:helps|helped|answered|answers|clarified)",
    r"[Ll]et me know if (?:you have|I can|there)",
    r"[Ff]eel free to (?:ask|reach out|contact)",
    r"If you have any further questions",
    r"please do not hesitate",
    r"don't hesitate to ask",
    r"Chat\s*Doctor",
    r"ChatDoctor",
    r"HealthCareMagic",
    r"### End of Chat",
    r"Take care[,\.]",
    r"\bRegards[,\.]",
    r"Best regards",
    r"Kind regards",
    r"Sincerely",
    r"Yours truly",
    r"Warm regards",
    r"With best wishes",
    r"\[Your Name\]",
]

# Truncated only after min_match_pos characters of real content.
STOP_PATTERNS = [
    # Board-exam Q&A continuation
    r"(?:\n|\.\s+|\?\s+)(?:Question|Q)\s*:",
    r"(?:\n|\.\s+|\?\s+)Answer\s*:",
    # Repetitive summary phrases
    r"\bIn conclusion[,\s]",
    r"\bTo summarize[,\s]",
    r"\bIn summary[,\s]",
    r"[Tt]he (?:patient|physician|doctor) should\b",
    r"[Tt]his patient should\b",
    r"[Tt]herefore, (?:the|this)",
    r"Wishing you (?:good|the best)",
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
    min_match_pos: int = 200,
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

    # 2a. Aggressive stop — exam/board-format leakage, truncate anywhere
    earliest_pos = len(response)
    for pattern in AGGRESSIVE_STOP_PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    if earliest_pos < len(response):
        response = response[:earliest_pos]

    # 2b. Normal stop patterns — only after min_match_pos characters
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
