#!/usr/bin/env python3
"""
Compute human evaluation metrics from annotator CSV files.

Reads evaluation/human_eval_*.csv files and computes:
  - Mean scores per dimension (correctness, relevance, completeness, safety)
  - Cohen's kappa inter-annotator agreement
  - Correlation between human scores and automated confidence

Usage:
    python evaluation/compute_human_eval.py
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"


def load_annotations(pattern="human_eval_*.csv"):
    """Load all annotator CSV files."""
    files = list(EVAL_DIR.glob(pattern))
    if not files:
        # Try the template as fallback
        template = EVAL_DIR / "human_eval_template.csv"
        if template.exists():
            files = [template]

    all_annotations = []
    for f in files:
        with open(f) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Skip empty rows
                if not row.get("factual_correctness"):
                    continue
                try:
                    automated_confidence = row.get("automated_confidence", "")
                    all_annotations.append({
                        "question_id": row["question_id"],
                        "evaluator_id": row.get("evaluator_id", f.stem),
                        "factual_correctness": int(row["factual_correctness"]),
                        "relevance": int(row["relevance"]),
                        "completeness": int(row["completeness"]),
                        "safety": int(row["safety"]),
                        "automated_confidence": (
                            float(automated_confidence) if automated_confidence not in ("", None) else None
                        ),
                    })
                except (ValueError, KeyError):
                    continue
    return all_annotations


def compute_mean_scores(annotations):
    """Compute mean score per dimension."""
    dims = ["factual_correctness", "relevance", "completeness", "safety"]
    scores = {d: [] for d in dims}
    for a in annotations:
        for d in dims:
            scores[d].append(a[d])
    return {d: round(np.mean(v), 2) if v else 0 for d, v in scores.items()}


def cohens_kappa(ratings1, ratings2):
    """Compute Cohen's kappa for two lists of ordinal ratings."""
    if len(ratings1) != len(ratings2) or not ratings1:
        return None
    n = len(ratings1)
    categories = sorted(set(ratings1) | set(ratings2))
    # Build confusion matrix
    matrix = defaultdict(int)
    for r1, r2 in zip(ratings1, ratings2):
        matrix[(r1, r2)] += 1
    # Observed agreement
    po = sum(matrix[(c, c)] for c in categories) / n
    # Expected agreement
    pe = sum(
        (sum(1 for r in ratings1 if r == c) / n) *
        (sum(1 for r in ratings2 if r == c) / n)
        for c in categories
    )
    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def compute_inter_annotator(annotations):
    """Compute pairwise Cohen's kappa across annotators."""
    # Group by evaluator
    by_eval = defaultdict(dict)
    for a in annotations:
        by_eval[a["evaluator_id"]][a["question_id"]] = a

    evaluators = list(by_eval.keys())
    if len(evaluators) < 2:
        return {"note": "Need >= 2 annotators for agreement"}

    results = {}
    dims = ["factual_correctness", "relevance", "completeness", "safety"]

    for i, e1 in enumerate(evaluators):
        for e2 in evaluators[i + 1:]:
            common = set(by_eval[e1].keys()) & set(by_eval[e2].keys())
            if not common:
                continue
            pair_key = f"{e1}_vs_{e2}"
            results[pair_key] = {}
            for dim in dims:
                r1 = [by_eval[e1][q][dim] for q in common]
                r2 = [by_eval[e2][q][dim] for q in common]
                results[pair_key][dim] = cohens_kappa(r1, r2)

    return results


def compute_confidence_correlation(annotations):
    """Correlate automated confidence with mean human factual correctness."""
    grouped = defaultdict(lambda: {"scores": [], "confidence": None})
    for ann in annotations:
        grouped[ann["question_id"]]["scores"].append(ann["factual_correctness"])
        if ann.get("automated_confidence") is not None:
            grouped[ann["question_id"]]["confidence"] = ann["automated_confidence"]

    human_scores = []
    automated_scores = []
    for item in grouped.values():
        if item["confidence"] is None or not item["scores"]:
            continue
        human_scores.append(float(np.mean(item["scores"])))
        automated_scores.append(float(item["confidence"]))

    if len(human_scores) < 2:
        return {"note": "Need >= 2 rows with automated_confidence for correlation"}

    return {
        "pearson_r": round(float(np.corrcoef(human_scores, automated_scores)[0, 1]), 4)
    }


def main():
    annotations = load_annotations()
    if not annotations:
        print("No annotations found. Fill in evaluation/human_eval_template.csv first.")
        print("Template columns: question_id, question, model_answer, reference_answer,")
        print("  factual_correctness (1-5), relevance (1-5), completeness (1-5),")
        print("  safety (1-5), evaluator_id")
        sys.exit(0)

    print(f"Loaded {len(annotations)} annotations")

    means = compute_mean_scores(annotations)
    print("\nMean scores per dimension:")
    for dim, score in means.items():
        print(f"  {dim}: {score}")

    agreement = compute_inter_annotator(annotations)
    print(f"\nInter-annotator agreement: {json.dumps(agreement, indent=2)}")
    confidence_correlation = compute_confidence_correlation(annotations)
    print(f"\nAutomated confidence correlation: {json.dumps(confidence_correlation, indent=2)}")

    # Save results
    out = {
        "n_annotations": len(annotations),
        "mean_scores": means,
        "inter_annotator_agreement": agreement,
        "automated_confidence_correlation": confidence_correlation,
    }
    out_path = EVAL_DIR / "results" / "human_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
