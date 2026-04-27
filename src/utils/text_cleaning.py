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

# Preamble lines the model sometimes generates before the actual answer.
# Stripped from the START of the response (not truncated mid-text).
PREAMBLE_STRIP_PATTERNS = [
    # Opening SENTENCE (up to first ".!?") that contains a ChatDoctor/HealthCareMagic marker.
    # Uses [^.!?]* to stop at the sentence boundary so only that sentence is stripped.
    r"^[^.!?]*(?:Chat\s*Doctor|ChatDoctor|HealthCareMagic)[^.!?]*[.!?]+\s*",
    # "Welcome to [anything]" opening line (newline-terminated)
    r"^[Ww]elcome to[^\n]*\n+",
    # "Welcome To ..." opening sentence (no trailing newline, e.g. "Welcome To Chat Doctor . ")
    r"^[Ww]elcome [Tt]o[^.!?]*[.!?]*\s+",
    # "Thank you for your question/query/..." as an opener
    r"^[Tt]hank(?:s| you) for (?:your|the) (?:question|query|message|email|time|concern)[^.!?\n]*[.!?\n]+\s*",
    # "Thanks for posting/writing/..." as an opener
    r"^[Tt]hank(?:s| you) for (?:posting|choosing|using|reaching out|contacting|writing|asking)[^.!?\n]*[.!?\n]+\s*",
    # Doctor self-introduction sentences
    r"^[Mm]y name is (?:Dr|Doctor)[^.!?]*[.!?]+\s*",
    r"^[Ii] am (?:a |an )?(?:Dr|Doctor|physician|surgeon|specialist)[^.!?]*[.!?]+\s*",
    r"^[Hh]ello,?\s+[Aa]s per (?:your|the)[^.!?]*[.!?,]\s*",
    r"^[Hh]ello,?\s+[Ii]'?m (?:Dr|Doctor)[^\n]*\n+",
    r"^[Hh]i,?\s+[Ii]'?m (?:Dr|Doctor)[^\n]*\n+",
    r"^[Hh]i\s*[,!][^\n]*\n+",
    r"^[Hh]ello\s*[,!][^\n]*\n+",
    r"^[Gg]ood (?:morning|afternoon|evening|day)[,!\s][^\n]*\n+",
    r"^[Dd]ear [^\n]+\n+",
    # Model narrating its own output instead of answering directly
    r"^[Bb]ased on the (?:given material|reference text|provided (?:text|information|context))[^:\n]*:\s*",
    r"^[Hh]ere (?:is|are) (?:the |some |a )?(?:possible )?(?:answer|answers|information|details|summary)[^:\n]*:\s*",
    r"^[Aa]ccording to the (?:given material|reference text|provided (?:text|information|context))[^:\n]*:\s*",
    r"^[Aa]nswers?\s*(?:\(in order\)|:)\s*",
    r"^[Tt]he (?:following|answer|information|details)[^:\n]*:\s*",
    # MedMCQA exam-format leakage at the start of the answer
    # e.g. "Correct answer: B\nExplanation: ..." or "Correct answer: Explanation: ..."
    r"^[Cc]orrect [Aa]nswer\s*:\s*(?:[A-E]\b\.?\s*)?",
    r"^[Ee]xplanation\s*:\s*",
]

# Patterns that are ALWAYS truncated wherever they appear (min_pos=0).
# ONLY patterns that are *never* legitimate in a medical answer belong here.
# Guideline references like "Per the JNC" and clinical descriptions CAN be
# part of a valid answer — those were moved to STOP_PATTERNS (deferred, min_pos=200)
# to avoid truncating correct answers about hypertension treatment, etc.
AGGRESSIVE_STOP_PATTERNS = [
    # Exam-format Q&A continuation — definitive leakage markers
    r"of the following (?:combinations|options|choices|drugs|statements)",
    r"(?:^|\n)###\s*(?:Question|Answer|Q|A)\s*:",
    # Hard identity markers — never legitimate in a medical answer
    r"Chat\s*Doctor",
    r"ChatDoctor",
    r"HealthCareMagic",
    r"### End of Chat",
    r"\[Your Name\]",
]

# Patterns that COULD appear in legitimate medical answers (guideline references,
# clinical vignettes) but indicate exam-format leakage when they appear deep in
# the response (after min_match_pos chars of real content).
# These were previously in AGGRESSIVE_STOP_PATTERNS and caused false truncation
# of valid answers about hypertension treatment, drug guidelines, etc.
_DEFERRED_EXAM_PATTERNS = [
    r"Per the (?:Joint National|American Heart|WHO|CDC|European)",
    r"[Ww]hich (?:of the following|can be considered|is the most effective|is the best)",
    r"[Aa] \d{2}-year-old (?:man|woman|patient|child|boy|girl) presents",
    r"[Hh]is (?:temperature|blood pressure|pulse|heart rate) is \d",
    r"[Hh]er (?:temperature|blood pressure|pulse|heart rate) is \d",
    r"[Tt]he patient'?s? blood pressure at (?:his|her|the) last",
    r"[Tt]he patient'?s? (?:dietary|nutritional) intake is (?:low|high)",
]

# Sign-off / closing patterns — only truncate after min_sign_off_pos chars of real content.
# These can appear as opening preambles when the model imitates ChatDoctor format, so
# we must NOT cut them at position 0 (the preamble-strip step handles that case).
_SIGN_OFF_MIN_POS = 80
SIGN_OFF_STOP_PATTERNS = [
    r"[Tt]hank(?:s| you) for (?:your|the) (?:question|query|message|email|time|concern)",
    r"[Tt]hank(?:s| you) for (?:choosing|using|reaching out|contacting|writing|asking)",
    r"I hope this ",
    r"[Hh]ope (?:I have|I've|that|this) (?:helps|helped|answered|answers|clarified)",
    r"[Hh]ope you (?:found|find|enjoyed|liked) (?:what|this|that)",
    r"I hope (?:I have|I've|that) (?:helps|helped|answered|answers|clarified)",
    r"[Ll]et me know if (?:you have|I can|there)",
    r"[Ff]eel free to (?:ask|reach out|contact)",
    r"If you have any further questions",
    r"please do not hesitate",
    r"don't hesitate to ask",
    r"Take care[,\.]?",
    r"\bRegards[,\.]",
    r"Best regards",
    r"Kind regards",
    r"Sincerely",
    r"Yours truly",
    r"Warm regards",
    r"With best wishes",
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
] + _DEFERRED_EXAM_PATTERNS  # exam patterns deferred here (not aggressive at pos 0)


# Matches 3+ consecutive "MULTI WORD PHRASE (ABBR)" blocks separated by commas/semicolons.
# TinyLlama produces this pattern when hallucinating fake medical acronyms, e.g.:
#   "POLIO EXTRACTED FROM ORAL CULTURES (POX), THYROID GLAND LABORATORY TESTING (THLDG), ..."
_CAPS_ACRONYM_SPAM_RE = re.compile(
    r"(?:[A-Z][A-Z /\-]{3,}\([A-Z]{2,8}\))"      # WORD WORD (ABBR)
    r"(?:\s*[,;]\s*(?:[A-Z][A-Z /\-]{3,}\([A-Z]{2,8}\))){2,}",  # 2+ more occurrences
    re.MULTILINE,
)


def is_hallucinated_caps_spam(text: str) -> bool:
    """
    Return True if the text looks like TinyLlama ALL-CAPS acronym hallucination.

    Checks for 3+ consecutive ``WORD WORD (ABBR)`` blocks — a signature pattern
    where the model invents fake medical acronyms not present in any real source.
    """
    if not text:
        return False
    return bool(_CAPS_ACRONYM_SPAM_RE.search(text))


def _cut_repetition_loop(text: str, min_phrase_len: int = 80, max_repeats: int = 2) -> str:
    """
    Detect and cut repetition loops in LLM output.

    Scans for any clause/sentence of length >= min_phrase_len that appears
    max_repeats or more times.  Truncates before the second occurrence.

    This catches patterns like:
      "I understand your concern. I can understand your concern. I have gone
       through your query. I can understand your concern. I have gone..."
    """
    if not text or len(text) < min_phrase_len * 2:
        return text

    # Split on sentence boundaries / newlines to find candidate phrases
    # Use a sliding window of ~15-60 chars
    for phrase_len in range(min_phrase_len, min(60, len(text) // 2)):
        seen: dict = {}
        for start in range(0, len(text) - phrase_len):
            chunk = text[start : start + phrase_len]
            if chunk in seen:
                if seen[chunk] >= max_repeats - 1:
                    # Truncate just before the first repeated occurrence
                    first_pos = text.find(chunk)
                    second_pos = text.find(chunk, first_pos + 1)
                    if second_pos > 0:
                        return text[:second_pos].rstrip()
                else:
                    seen[chunk] = seen[chunk] + 1
            else:
                seen[chunk] = 1
    return text


def clean_llm_response(
    response: str,
    stop_patterns: List[str] | None = None,
    min_match_pos: int = 200,
) -> str:
    """
    Clean an LLM response by removing training-data leakage.

    Steps:
    0. Strip ChatDoctor-style preamble lines from the start
    1. Strip known answer-prefix headers
    2. Truncate at earliest aggressive stop-pattern match (exam-format leakage)
    3. Truncate at earliest sign-off pattern (only after _SIGN_OFF_MIN_POS chars)
    4. Truncate at earliest normal stop pattern (only after min_match_pos chars)
    5. Remove citation markers like ``[1]``
    6. Trim to last complete sentence if response ends mid-word
    7. Collapse excess whitespace

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

    # 0. Strip preamble lines from the START (ChatDoctor greeting/opening gambits).
    #    Apply up to 4 times to handle stacked preambles.
    for _ in range(4):
        stripped = False
        for pat in PREAMBLE_STRIP_PATTERNS:
            m = re.match(pat, response, re.IGNORECASE)
            if m:
                response = response[m.end():].strip()
                stripped = True
                break
        if not stripped:
            break

    if not response:
        return response

    # 1. Strip prefix headers
    for prefix in ANSWER_PREFIXES:
        if response.startswith(prefix):
            response = response[len(prefix) :].strip()

    # 2. Aggressive stop — exam/board-format leakage, truncate at any position
    earliest_pos = len(response)
    for pattern in AGGRESSIVE_STOP_PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    if earliest_pos < len(response):
        response = response[:earliest_pos]

    # 3. Sign-off stop — only truncate after _SIGN_OFF_MIN_POS chars
    sign_off_min = min(_SIGN_OFF_MIN_POS, len(response))
    earliest_pos = len(response)
    for pattern in SIGN_OFF_STOP_PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.start() >= sign_off_min and match.start() < earliest_pos:
            earliest_pos = match.start()
    if earliest_pos < len(response):
        response = response[:earliest_pos]

    # 4. Normal stop patterns — only after min_match_pos characters
    patterns = stop_patterns if stop_patterns is not None else STOP_PATTERNS
    safe_min = min(min_match_pos, int(len(response) * 0.4))
    earliest_pos = len(response)

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.start() >= safe_min and match.start() < earliest_pos:
            earliest_pos = match.start()

    if earliest_pos < len(response):
        response = response[:earliest_pos]

    # 4b. Repetition-loop detection — if a sentence/clause repeats 2+ times,
    #     truncate before the second occurrence. Catches ChatDoctor spin loops
    #     that survive all other filters (e.g. "I understand your concern." × N).
    response = _cut_repetition_loop(response)

    # 5. Remove citation markers [1], [2], etc.
    response = re.sub(r"\[\d+\]", "", response)
    # Remove square brackets around drug/medical terms (source doc formatting artifacts)
    # e.g. [codeine phosphate] → codeine phosphate
    response = re.sub(r"\[([a-z][a-zA-Z0-9 \-]{1,60})\]", r"\1", response)

    # 6. Trim to last sentence boundary if cut mid-word
    response = response.rstrip()
    if response and response[-1] not in '.!?:)"':
        last_period = max(
            response.rfind("."),
            response.rfind("!"),
            response.rfind("?"),
        )
        if last_period > len(response) * 0.3:
            response = response[: last_period + 1]

    # 7. Collapse whitespace
    response = re.sub(r"  +", " ", response)
    response = re.sub(r"\n\n\n+", "\n\n", response)

    return response.strip()
