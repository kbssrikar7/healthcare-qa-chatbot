"""
Redact likely PHI / sensitive patterns before writing logs or feedback files.

Used by trajectory logging and feedback storage (plan Phase C3).
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, List, Union

# Email, phone (simple), US SSN-like, long digit runs (MRN-like)
_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b\d{10,16}\b"), "[REDACTED_ID]"),
]


def redact_text(text: str) -> str:
    if not text:
        return text
    out = str(text)
    for rx, repl in _PATTERNS:
        out = rx.sub(repl, out)
    return out


def redact_payload(obj: Any) -> Any:
    """
    Recursively redact strings in dict/list structures.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_payload(v) for v in obj]
    return obj


def should_redact_logs() -> bool:
    return os.getenv("REDACT_LOGS", "true").lower() in ("1", "true", "yes")


def maybe_redact(obj: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    if not should_redact_logs():
        return obj
    return redact_payload(copy.deepcopy(obj))
