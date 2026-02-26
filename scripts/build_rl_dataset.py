#!/usr/bin/env python3
"""
Build offline RL training data from trajectories + user feedback.

Inputs:
- data/feedback/response_trajectories.jsonl
- data/feedback/user_feedback.jsonl

Output:
- Joined dataset (jsonl/json/csv) with reward labels and train/val split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def _parse_datetime(value: Optional[str]) -> datetime:
    """Parse ISO timestamp with safe fallback."""
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value to [low, high]."""
    return max(low, min(high, value))


def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert value to bool using sensible defaults."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_rating(value: Any) -> int:
    """Normalize rating into inclusive 1..5 range."""
    try:
        rating = int(value)
    except (TypeError, ValueError):
        rating = 3
    return int(_clamp(rating, 1, 5))


def _compute_reward_from_feedback(feedback: Dict[str, Any]) -> float:
    """
    Compute normalized reward from feedback fields.

    Priority:
    1) feedback.metadata.reward_signal (if present)
    2) weighted formula from rating/helpful/accurate/safe
    """
    metadata = feedback.get("metadata") or {}
    if isinstance(metadata, dict) and "reward_signal" in metadata:
        return round(_clamp(_as_float(metadata.get("reward_signal"), 0.5), 0.0, 1.0), 4)

    rating = _normalize_rating(feedback.get("rating", 3))
    was_helpful = _as_bool(feedback.get("was_helpful"), default=rating >= 4)
    was_accurate = _as_bool(feedback.get("was_accurate"), default=rating >= 4)
    was_safe = _as_bool(feedback.get("was_safe"), default=True)

    rating_component = (rating - 1) / 4.0
    reward = (
        0.4 * rating_component
        + 0.2 * float(was_helpful)
        + 0.2 * float(was_accurate)
        + 0.2 * float(was_safe)
    )
    return round(_clamp(reward, 0.0, 1.0), 4)


def _deterministic_split(key: str, val_ratio: float) -> str:
    """Assign train/val split deterministically by key hash."""
    if val_ratio <= 0.0:
        return "train"
    if val_ratio >= 1.0:
        return "val"

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "val" if bucket < val_ratio else "train"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping malformed JSONL line {line_no} in {path}")
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_response_id(feedback: Dict[str, Any]) -> Optional[str]:
    """Resolve response_id from feedback payload."""
    metadata = feedback.get("metadata") or {}
    candidates = [
        metadata.get("response_id") if isinstance(metadata, dict) else None,
        feedback.get("response_id"),
        feedback.get("question_id"),  # current API stores response_id in question_id
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return None


def _flatten_trajectory(trajectory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten trajectory fields for tabular training output."""
    trajectory = trajectory or {}
    state = trajectory.get("state_features") or {}
    action = trajectory.get("action") or {}
    retrieval = trajectory.get("retrieval") or {}
    outcome = trajectory.get("outcome") or {}
    safety = trajectory.get("safety") or {}
    metadata = trajectory.get("metadata") or {}

    flags = safety.get("flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]

    return {
        "question_id": trajectory.get("question_id"),
        "question": trajectory.get("question"),
        "effective_question": trajectory.get("effective_question"),
        "trajectory_timestamp": trajectory.get("timestamp"),
        "state_question_length": _as_float(state.get("question_length"), 0.0),
        "state_effective_question_length": _as_float(state.get("effective_question_length"), 0.0),
        "state_num_sources_requested": _as_float(state.get("num_sources_requested"), 0.0),
        "state_include_explanation": _as_bool(state.get("include_explanation"), default=True),
        "state_has_session_context": _as_bool(state.get("has_session_context"), default=False),
        "action_model_choice": action.get("model_choice"),
        "action_pipeline": action.get("pipeline"),
        "action_use_langchain": _as_bool(action.get("use_langchain"), default=False),
        "action_use_langgraph": _as_bool(action.get("use_langgraph"), default=False),
        "retrieval_count": _as_float(retrieval.get("count"), 0.0),
        "retrieval_top_score": _as_float(retrieval.get("top_score"), 0.0),
        "retrieval_mean_score": _as_float(retrieval.get("mean_score"), 0.0),
        "retrieval_min_score": _as_float(retrieval.get("min_score"), 0.0),
        "retrieval_max_score": _as_float(retrieval.get("max_score"), 0.0),
        "retrieval_scores": retrieval.get("scores", []),
        "retrieval_source_names": retrieval.get("source_names", []),
        "outcome_answer_length": _as_float(outcome.get("answer_length"), 0.0),
        "outcome_confidence_score": _as_float(outcome.get("confidence_score"), 0.0),
        "outcome_confidence_level": outcome.get("confidence_level"),
        "outcome_is_answerable": _as_bool(outcome.get("is_answerable"), default=True),
        "outcome_from_cache": _as_bool(outcome.get("from_cache"), default=False),
        "outcome_latency_ms": _as_float(outcome.get("latency_ms"), 0.0),
        "outcome_num_sources_returned": _as_float(outcome.get("num_sources_returned"), 0.0),
        "safety_level": safety.get("level"),
        "safety_flag_count": len(flags),
        "safety_flags": flags,
        "metadata_model_used": metadata.get("model_used"),
        "metadata_pipeline_used": metadata.get("pipeline_used"),
    }


def _aggregate_feedback_rows(
    response_id: str,
    feedback_rows: List[Dict[str, Any]],
    trajectory: Optional[Dict[str, Any]],
    val_ratio: float,
) -> Dict[str, Any]:
    """Aggregate multiple feedback events for one response."""
    rewards = [_compute_reward_from_feedback(row) for row in feedback_rows]
    ratings = [_normalize_rating(row.get("rating", 3)) for row in feedback_rows]
    helpful = [_as_bool(row.get("was_helpful"), default=False) for row in feedback_rows]
    accurate = [_as_bool(row.get("was_accurate"), default=False) for row in feedback_rows]
    safe = [_as_bool(row.get("was_safe"), default=True) for row in feedback_rows]

    latest = max(
        feedback_rows,
        key=lambda row: _parse_datetime(str(row.get("timestamp", ""))),
    )

    feedback_ids = [str(row.get("feedback_id", "")) for row in feedback_rows]
    feedback_texts = [str(row.get("feedback_text", "")) for row in feedback_rows if row.get("feedback_text")]

    row: Dict[str, Any] = {
        "sample_id": response_id,
        "response_id": response_id,
        "split": _deterministic_split(response_id, val_ratio),
        "has_trajectory": trajectory is not None,
        "num_feedback": len(feedback_rows),
        "reward": round(mean(rewards), 4) if rewards else 0.0,
        "reward_min": round(min(rewards), 4) if rewards else 0.0,
        "reward_max": round(max(rewards), 4) if rewards else 0.0,
        "rating_mean": round(mean(ratings), 4) if ratings else 0.0,
        "rating_min": min(ratings) if ratings else 0,
        "rating_max": max(ratings) if ratings else 0,
        "helpful_rate": round(mean(float(v) for v in helpful), 4) if helpful else 0.0,
        "accurate_rate": round(mean(float(v) for v in accurate), 4) if accurate else 0.0,
        "safe_rate": round(mean(float(v) for v in safe), 4) if safe else 0.0,
        "feedback_timestamp_latest": latest.get("timestamp"),
        "feedback_ids": feedback_ids,
        "feedback_texts": feedback_texts[:5],
    }

    row.update(_flatten_trajectory(trajectory))
    return row


def _per_feedback_rows(
    response_id: str,
    feedback_row: Dict[str, Any],
    trajectory: Optional[Dict[str, Any]],
    val_ratio: float,
) -> Dict[str, Any]:
    """Build one training row per feedback event."""
    feedback_id = str(feedback_row.get("feedback_id", "") or response_id)
    rating = _normalize_rating(feedback_row.get("rating", 3))
    was_helpful = _as_bool(feedback_row.get("was_helpful"), default=rating >= 4)
    was_accurate = _as_bool(feedback_row.get("was_accurate"), default=rating >= 4)
    was_safe = _as_bool(feedback_row.get("was_safe"), default=True)

    row: Dict[str, Any] = {
        "sample_id": feedback_id,
        "response_id": response_id,
        "feedback_id": feedback_id,
        "split": _deterministic_split(feedback_id, val_ratio),
        "has_trajectory": trajectory is not None,
        "num_feedback": 1,
        "reward": _compute_reward_from_feedback(feedback_row),
        "rating_mean": float(rating),
        "rating_min": rating,
        "rating_max": rating,
        "helpful_rate": float(was_helpful),
        "accurate_rate": float(was_accurate),
        "safe_rate": float(was_safe),
        "feedback_timestamp_latest": feedback_row.get("timestamp"),
        "feedback_ids": [feedback_id],
        "feedback_texts": [feedback_row.get("feedback_text")] if feedback_row.get("feedback_text") else [],
    }

    row.update(_flatten_trajectory(trajectory))
    return row


def _write_output(rows: List[Dict[str, Any]], output_path: Path, output_format: str) -> None:
    """Write dataset in chosen format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
        return

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, default=str)
        return

    # csv
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    serialized[key] = json.dumps(value, default=str)
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def _build_summary(
    trajectories_count: int,
    feedback_count: int,
    rows: List[Dict[str, Any]],
    unmatched_feedback: int,
    aggregate_by_response: bool,
) -> Dict[str, Any]:
    """Build summary stats for dataset generation."""
    train_count = sum(1 for row in rows if row.get("split") == "train")
    val_count = sum(1 for row in rows if row.get("split") == "val")
    mean_reward = round(mean([_as_float(row.get("reward"), 0.0) for row in rows]), 4) if rows else 0.0
    mean_feedback = round(mean([_as_float(row.get("num_feedback"), 1.0) for row in rows]), 4) if rows else 0.0

    return {
        "trajectories_loaded": trajectories_count,
        "feedback_loaded": feedback_count,
        "samples_built": len(rows),
        "aggregation_mode": "response" if aggregate_by_response else "per_feedback",
        "unmatched_feedback_skipped": unmatched_feedback,
        "train_samples": train_count,
        "val_samples": val_count,
        "mean_reward": mean_reward,
        "mean_feedback_per_sample": mean_feedback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline RL dataset from trajectories + feedback.")
    parser.add_argument(
        "--trajectories",
        default="data/feedback/response_trajectories.jsonl",
        help="Path to response trajectories JSONL",
    )
    parser.add_argument(
        "--feedback",
        default="data/feedback/user_feedback.jsonl",
        help="Path to user feedback JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/feedback/rl_training_dataset.jsonl",
        help="Output dataset path",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "json", "csv"],
        default="jsonl",
        help="Output serialization format",
    )
    parser.add_argument(
        "--per-feedback",
        dest="aggregate_by_response",
        action="store_false",
        help="Keep one row per feedback event (default is aggregated per response_id)",
    )
    parser.add_argument(
        "--include-unmatched-feedback",
        action="store_true",
        help="Include feedback rows even when no trajectory is found",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for deterministic train/val assignment",
    )
    parser.add_argument(
        "--summary-output",
        default="data/feedback/rl_training_dataset_summary.json",
        help="Path to write dataset build summary JSON",
    )
    parser.set_defaults(aggregate_by_response=True)
    args = parser.parse_args()

    trajectories_path = Path(args.trajectories)
    feedback_path = Path(args.feedback)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)

    trajectories = _load_jsonl(trajectories_path)
    feedback_rows = _load_jsonl(feedback_path)

    trajectory_by_response: Dict[str, Dict[str, Any]] = {}
    for row in trajectories:
        response_id = row.get("response_id")
        if not response_id:
            continue
        trajectory_by_response[str(response_id)] = row

    grouped_feedback: Dict[str, List[Dict[str, Any]]] = {}
    missing_response_id = 0
    for row in feedback_rows:
        response_id = _resolve_response_id(row)
        if not response_id:
            missing_response_id += 1
            continue
        grouped_feedback.setdefault(response_id, []).append(row)

    samples: List[Dict[str, Any]] = []
    unmatched_feedback = 0

    for response_id, group in grouped_feedback.items():
        trajectory = trajectory_by_response.get(response_id)
        if trajectory is None and not args.include_unmatched_feedback:
            unmatched_feedback += len(group)
            continue

        if args.aggregate_by_response:
            samples.append(
                _aggregate_feedback_rows(
                    response_id=response_id,
                    feedback_rows=group,
                    trajectory=trajectory,
                    val_ratio=float(_clamp(args.val_ratio, 0.0, 1.0)),
                )
            )
        else:
            for feedback in group:
                samples.append(
                    _per_feedback_rows(
                        response_id=response_id,
                        feedback_row=feedback,
                        trajectory=trajectory,
                        val_ratio=float(_clamp(args.val_ratio, 0.0, 1.0)),
                    )
                )

    _write_output(samples, output_path=output_path, output_format=args.output_format)

    summary = _build_summary(
        trajectories_count=len(trajectories),
        feedback_count=len(feedback_rows),
        rows=samples,
        unmatched_feedback=unmatched_feedback + missing_response_id,
        aggregate_by_response=args.aggregate_by_response,
    )

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 60)
    print("RL DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"Trajectories loaded: {summary['trajectories_loaded']}")
    print(f"Feedback loaded: {summary['feedback_loaded']}")
    print(f"Samples built: {summary['samples_built']}")
    print(f"Aggregation mode: {summary['aggregation_mode']}")
    print(f"Train samples: {summary['train_samples']}")
    print(f"Val samples: {summary['val_samples']}")
    print(f"Mean reward: {summary['mean_reward']:.4f}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_output_path}")


if __name__ == "__main__":
    main()
