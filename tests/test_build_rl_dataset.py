"""
Tests for offline RL dataset builder script.
"""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "build_rl_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_rl_dataset", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_response_id_prefers_metadata():
    feedback = {
        "question_id": "old_value",
        "metadata": {"response_id": "resp_abc"},
    }
    assert MODULE._resolve_response_id(feedback) == "resp_abc"


def test_resolve_response_id_falls_back_to_question_id():
    feedback = {
        "question_id": "resp_fallback",
        "metadata": {},
    }
    assert MODULE._resolve_response_id(feedback) == "resp_fallback"


def test_compute_reward_uses_metadata_when_present():
    feedback = {
        "rating": 1,
        "was_helpful": False,
        "metadata": {"reward_signal": 0.87},
    }
    assert MODULE._compute_reward_from_feedback(feedback) == 0.87


def test_compute_reward_formula_without_metadata():
    feedback = {
        "rating": 5,
        "was_helpful": True,
        "was_accurate": True,
        "was_safe": True,
    }
    assert MODULE._compute_reward_from_feedback(feedback) == 1.0


def test_deterministic_split_is_stable():
    first = MODULE._deterministic_split("sample_key", 0.2)
    second = MODULE._deterministic_split("sample_key", 0.2)
    assert first == second
    assert first in {"train", "val"}
