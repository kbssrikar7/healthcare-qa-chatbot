#!/usr/bin/env python3
"""
Fit Platt scaling for the Android on-device (Gemma 3 1B) confidence layer.

Reads calibration_results.jsonl produced by the app's CalibrationRunner
(android/app/.../CalibrationRunner.kt) — one row per question from
evaluation/test_set_v2.json, containing the on-device raw_score
(retrieval_confidence + source_agreement only, renormalized — LiteRT-LM has no
token-probability API for a generation_confidence signal) and the generated
answer. Correctness labels are computed here, offline, using the same
keyword_coverage/is_correct definition as every other evaluation script in
this repo (evaluation/eval_utils.py) — not reimplemented.

This intentionally does NOT reuse desktop's platt_a=14.44/platt_b=-11.25
(src/xai/multi_signal_confidence.py) — those were fitted on TinyLlama's raw-
score distribution, which a different base model (Gemma 3 1B) has no reason
to share.

Usage:
    python evaluation/fit_mobile_calibration.py \
        --input evaluation/results/mobile_calibration_results.jsonl
"""

import sys
import json
import argparse
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_utils import keyword_coverage, is_correct
from evaluation.compute_calibration import (
    compute_ece,
    fit_platt,
    apply_platt,
    save_reliability_diagram,
)


def load_results(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pairs(rows: list) -> tuple:
    """Return (raw_scores, labels, skipped_errors) for rows with a raw_score."""
    raw_scores, labels = [], []
    skipped_errors = 0
    for row in rows:
        if "error" in row:
            skipped_errors += 1
            continue
        cov = keyword_coverage(row["answer"], row.get("expected_keywords", []))
        label = 1 if is_correct(cov) else 0
        raw_scores.append(float(row["raw_score"]))
        labels.append(label)
    return raw_scores, labels, skipped_errors


def main():
    parser = argparse.ArgumentParser(description="Fit Platt scaling for mobile (Gemma) confidence layer")
    parser.add_argument("--input", default="evaluation/results/mobile_calibration_results.jsonl")
    parser.add_argument("--out-dir", default="evaluation/results")
    args = parser.parse_args()

    in_path = PROJECT_ROOT / args.input
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        print("Pull it from the device first:")
        print("  adb pull /sdcard/Android/data/com.mediquery.mobile/files/calibration_results.jsonl "
              f"{in_path}")
        sys.exit(1)

    rows = load_results(in_path)
    print(f"Loaded {len(rows)} rows from {in_path}")

    raw_scores, labels, skipped = build_pairs(rows)
    print(f"Usable pairs: {len(raw_scores)}  (skipped {skipped} error rows)")

    if len(raw_scores) < 10:
        print("Too few pairs for a meaningful Platt fit.")
        sys.exit(1)

    print(
        f"Raw score   — mean={statistics.mean(raw_scores):.3f}  "
        f"min={min(raw_scores):.3f}  max={max(raw_scores):.3f}"
    )
    print(f"Label rate  — {sum(labels) / len(labels):.2%} positive (keyword_coverage >= 0.4)")

    ece_raw, bc_raw, ba_raw, bn_raw = compute_ece(raw_scores, labels)
    print(f"\nRaw ECE (uncalibrated raw_score as confidence) = {ece_raw:.4f}")

    a, b = fit_platt(raw_scores, labels)
    print(f"Platt params — a={a:.4f}  b={b:.4f}")

    cal_scores = apply_platt(raw_scores, a, b)
    ece_cal, bc_cal, ba_cal, bn_cal = compute_ece(cal_scores, labels)
    print(f"Calibrated ECE = {ece_cal:.4f}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_reliability_diagram(
        bc_raw, ba_raw, bn_raw, ece_raw,
        "Mobile (Gemma 3 1B) Reliability — Raw",
        out_dir / "mobile_reliability_diagram_raw.png",
    )
    save_reliability_diagram(
        bc_cal, ba_cal, bn_cal, ece_cal,
        "Mobile (Gemma 3 1B) Reliability — Platt-calibrated",
        out_dir / "mobile_reliability_diagram_calibrated.png",
    )

    result = {
        "model": "gemma3-1b-it (LiteRT-LM, GPU backend)",
        "n": len(raw_scores),
        "skipped_error_rows": skipped,
        "label_positive_rate": round(sum(labels) / len(labels), 4),
        "mean_raw_score": round(statistics.mean(raw_scores), 4),
        "ece_raw": round(ece_raw, 4),
        "platt_a": round(a, 6),
        "platt_b": round(b, 6),
        "mean_calibrated_score": round(statistics.mean(cal_scores), 4),
        "ece_calibrated": round(ece_cal, 4),
        "bin_confidences_raw": bc_raw,
        "bin_accuracies_raw": ba_raw,
        "bin_counts_raw": bn_raw,
        "bin_confidences_calibrated": bc_cal,
        "bin_accuracies_calibrated": ba_cal,
        "bin_counts_calibrated": bn_cal,
    }
    out_path = out_dir / "mobile_calibration.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCalibration JSON → {out_path}")

    print("\nDone.")
    print(f"  Raw ECE        : {ece_raw:.4f}")
    print(f"  Calibrated ECE : {ece_cal:.4f}")
    print(f"  Platt (a, b)   : ({a:.4f}, {b:.4f})")


if __name__ == "__main__":
    main()
