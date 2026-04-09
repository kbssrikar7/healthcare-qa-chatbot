"""
Tests for RL trajectory logging utilities.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.trajectory_logger import TrajectoryLogger


def test_log_and_get_trajectory(tmp_path):
    storage = tmp_path / "response_trajectories.jsonl"
    logger = TrajectoryLogger(storage_path=str(storage))

    payload = {
        "response_id": "resp_123",
        "question_id": "q_1",
        "action": {"pipeline": "Standard"},
        "outcome": {"confidence_score": 0.8},
    }
    logger.log(payload)

    fetched = logger.get("resp_123")
    assert fetched is not None
    assert fetched["response_id"] == "resp_123"
    assert fetched["action"]["pipeline"] == "Standard"
    assert storage.exists()


def test_get_missing_trajectory_returns_none(tmp_path):
    storage = tmp_path / "response_trajectories.jsonl"
    logger = TrajectoryLogger(storage_path=str(storage))
    assert logger.get("does_not_exist") is None


def test_log_requires_response_id(tmp_path):
    storage = tmp_path / "response_trajectories.jsonl"
    logger = TrajectoryLogger(storage_path=str(storage))

    with pytest.raises(ValueError):
        logger.log({"question_id": "q_1"})
