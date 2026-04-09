"""
Trajectory logging utilities for RL policy optimization.

Stores response trajectories as JSONL so feedback can be linked
to concrete state/action/outcome tuples.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class TrajectoryLogger:
    """
    Persist RL-ready response trajectories.

    Each line is one JSON object keyed by `response_id`.
    A small in-memory index is kept for fast feedback lookups.
    """

    def __init__(
        self,
        storage_path: str = "data/feedback/response_trajectories.jsonl",
        max_index_size: int = 5000,
    ):
        self.storage_path = Path(storage_path)
        self.max_index_size = max_index_size
        self._recent_index: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure parent directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, trajectory: Dict[str, Any]) -> None:
        """
        Append a trajectory record to JSONL storage.

        Args:
            trajectory: JSON-serializable trajectory payload.
                        Must include `response_id`.
        """
        response_id = trajectory.get("response_id")
        if not response_id:
            raise ValueError("Trajectory must include response_id")

        with open(self.storage_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory, default=str) + "\n")

        self._recent_index[str(response_id)] = trajectory
        self._recent_index.move_to_end(str(response_id))
        if len(self._recent_index) > self.max_index_size:
            self._recent_index.popitem(last=False)

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trajectory by response_id.

        Checks in-memory index first, then falls back to file scan.
        """
        if response_id in self._recent_index:
            return self._recent_index[response_id]

        if not self.storage_path.exists():
            return None

        found: Optional[Dict[str, Any]] = None
        with open(self.storage_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed trajectory line")
                    continue
                if payload.get("response_id") == response_id:
                    found = payload

        if found:
            self._recent_index[response_id] = found
            self._recent_index.move_to_end(response_id)
            if len(self._recent_index) > self.max_index_size:
                self._recent_index.popitem(last=False)
        return found
