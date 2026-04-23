"""Smoke tests for plan.md implementation assets."""

import json
from pathlib import Path

import pytest

from src.utils.log_redaction import redact_text, should_redact_logs


def test_gold_retrieval_jsonl_parse():
    path = Path(__file__).resolve().parent.parent / "evaluation" / "data" / "gold_retrieval.jsonl"
    assert path.exists()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert "question" in row and row["question"]
        assert "risk_class" in row
        n += 1
    assert n >= 30


def test_email_redaction():
    t = "Contact me at user@example.com please"
    assert "@" not in redact_text(t) or "[REDACTED_EMAIL]" in redact_text(t)


def test_should_redact_default():
    assert should_redact_logs() is True
