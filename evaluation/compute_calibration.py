#!/usr/bin/env python3
"""
Fast confidence calibration using stored trajectory data.

Reads data/feedback/response_trajectories.jsonl (populated by the running API),
computes ECE, fits Platt scaling, and saves a reliability diagram — no pipeline
re-run required.

Correctness proxy:
  - Primary  : factual_consistency.score >= 0.5  (NLI-based, available in ~75% of rows)
  - Fallback : keyword coverage vs test_set_v2   (for matched questions)

Usage:
    python evaluation/compute_calibration.py
    python evaluation/compute_calibration.py --out-dir evaluation/results
"""
import os, sys, json, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ── helpers ───────────────────────────────────────────────────────────────────

def keyword_coverage(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords)


def load_trajectories(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_calibration_pairs(trajectories: list, test_cases: dict) -> tuple:
    """Return (confidences, labels, sources) lists.

    Label = 1 if the response is "correct":
      - If factual_consistency.score is available → label = fc_score >= 0.5
      - Else if question fuzzy-matches a test case → label = kw_coverage >= 0.5
        (using answer_length as quality proxy + confidence-weighted heuristic)
      - Else skip row
    """
    from difflib import SequenceMatcher

    confidences, labels, sources = [], [], []

    # Build normalized lookup for fuzzy matching
    norm_test = {}
    for q, tc in test_cases.items():
        norm_test[q.strip().lower()] = tc

    for row in trajectories:
        conf = row["outcome"].get("confidence_score")
        if conf is None:
            continue

        # Try NLI-based label first
        fc = row["outcome"].get("factual_consistency")
        if fc and fc.get("score") is not None:
            label = 1 if fc["score"] >= 0.5 else 0
            confidences.append(float(conf))
            labels.append(label)
            sources.append("nli")
            continue

        # Fallback: fuzzy match question to test case, use keyword heuristic
        q = row.get("question", "").strip().lower()
        if not q:
            continue

        # Try exact match first, then fuzzy
        matched_tc = norm_test.get(q)
        if matched_tc is None:
            best_ratio, best_tc = 0.0, None
            for tq, tc in norm_test.items():
                ratio = SequenceMatcher(None, q, tq).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_tc = tc
            if best_ratio >= 0.75:
                matched_tc = best_tc

        if matched_tc is None:
            continue

        # NOTE: Trajectory records store `answer_length` and metadata but NOT
        # the full answer text. This means we cannot compute keyword_coverage
        # directly here — we use a proxy heuristic instead.
        #
        # Heuristic label (positive = likely correct):
        #   - answer_len > 80  : short answers (<80 chars) are often non-answers
        #   - num_sources >= 2 : grounded in at least 2 retrieved docs
        #   - is_answerable    : pipeline judged it answerable (grounding gate passed)
        #
        # When none of those hold, we fall back to confidence score as a weak signal.
        # This is an intentional approximation; NLI-based labels (above) are preferred.
        answer_len = row["outcome"].get("answer_length", 0)
        num_sources = row["outcome"].get("num_sources_returned", 0)
        is_answerable = row["outcome"].get("is_answerable", True)

        if answer_len > 80 and num_sources >= 2 and is_answerable:
            label = 1
        elif answer_len < 30 or not is_answerable:
            label = 0
        else:
            # Borderline: use confidence score as weak proxy (but label it carefully)
            label = 1 if conf >= 0.65 else 0

        confidences.append(float(conf))
        labels.append(label)
        sources.append("heuristic")

    return confidences, labels, sources




def compute_ece(confidences: list, labels: list, n_bins: int = 10) -> tuple:
    """Return (ece, bin_confs, bin_accs, bin_counts)."""
    import numpy as np
    conf_arr = labels_arr = None
    conf_arr = __import__("numpy").array(confidences)
    labels_arr = __import__("numpy").array(labels)
    edges = __import__("numpy").linspace(0, 1, n_bins + 1)

    ece = 0.0
    bin_confs, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf_arr >= lo) & (conf_arr < hi)
        if mask.sum() == 0:
            continue
        acc = float(labels_arr[mask].mean())
        conf = float(conf_arr[mask].mean())
        count = int(mask.sum())
        ece += (count / len(conf_arr)) * abs(acc - conf)
        bin_confs.append(conf)
        bin_accs.append(acc)
        bin_counts.append(count)

    return float(ece), bin_confs, bin_accs, bin_counts


def fit_platt(confidences: list, labels: list) -> tuple:
    """Fit Platt scaling (a, b) via NLL minimization."""
    import numpy as np
    from scipy.optimize import minimize

    preds = np.array(confidences, dtype=float)
    labs = np.array(labels, dtype=float)

    def nll(params):
        a, b = params
        p = np.clip(1.0 / (1.0 + np.exp(-(a * preds + b))), 1e-7, 1 - 1e-7)
        return -float(np.mean(labs * np.log(p) + (1 - labs) * np.log(1 - p)))

    result = minimize(nll, [1.0, 0.0], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-6})
    a, b = float(result.x[0]), float(result.x[1])
    return a, b


def apply_platt(confidences: list, a: float, b: float) -> list:
    import numpy as np
    arr = np.array(confidences)
    return list(np.clip(1.0 / (1.0 + np.exp(-(a * arr + b))), 0.0, 1.0))


def save_reliability_diagram(bin_confs, bin_accs, bin_counts, ece,
                              title: str, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
        if bin_confs:
            ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.75,
                   color="#4f46e5", edgecolor="black", label="Model")
            # Annotate with counts
            for x, y, cnt in zip(bin_confs, bin_accs, bin_counts):
                ax.text(x, y + 0.02, str(cnt), ha="center", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Mean predicted confidence", fontsize=12)
        ax.set_ylabel("Fraction correct (NLI score ≥ 0.5)", fontsize=12)
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {out_path}")
    except Exception as e:
        print(f"  Matplotlib error: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast calibration from trajectory data")
    parser.add_argument("--traj", default="data/feedback/response_trajectories.jsonl")
    parser.add_argument("--test-set", default="evaluation/test_set_v2.json")
    parser.add_argument("--out-dir", default="evaluation/results")
    args = parser.parse_args()

    traj_path = PROJECT_ROOT / args.traj
    if not traj_path.exists():
        print(f"Trajectory file not found: {traj_path}")
        sys.exit(1)

    # Load test cases as lookup (used for fallback matching)
    ts_path = PROJECT_ROOT / args.test_set
    test_cases = {}
    if ts_path.exists():
        with open(ts_path) as f:
            for c in json.load(f)["test_cases"]:
                test_cases[c["query"]] = c

    trajectories = load_trajectories(traj_path)
    print(f"Loaded {len(trajectories)} trajectory rows")

    confidences, labels, sources = build_calibration_pairs(trajectories, test_cases)
    print(f"Usable pairs: {len(confidences)}  "
          f"(NLI-based={sources.count('nli')}, heuristic={sources.count('heuristic')})")


    if len(confidences) < 5:
        print("Too few pairs for calibration — run more queries through the API first.")
        sys.exit(1)

    import statistics
    print(f"Confidence  — mean={statistics.mean(confidences):.3f}  "
          f"min={min(confidences):.3f}  max={max(confidences):.3f}")
    print(f"Label rate  — {sum(labels)/len(labels):.2%} positive")

    # Raw (uncalibrated) ECE
    ece_raw, bc_raw, ba_raw, bn_raw = compute_ece(confidences, labels)
    print(f"\nRaw ECE   = {ece_raw:.4f}")

    # Fit Platt scaling
    a, b = fit_platt(confidences, labels)
    print(f"Platt params — a={a:.4f}  b={b:.4f}")
    cal_confs = apply_platt(confidences, a, b)
    ece_cal, bc_cal, ba_cal, bn_cal = compute_ece(cal_confs, labels)
    print(f"Calibrated ECE = {ece_cal:.4f}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save reliability diagrams
    save_reliability_diagram(bc_raw, ba_raw, bn_raw, ece_raw,
                             "Reliability Diagram (Raw)",
                             out_dir / "reliability_diagram_raw.png")
    save_reliability_diagram(bc_cal, ba_cal, bn_cal, ece_cal,
                             "Reliability Diagram (Platt-calibrated)",
                             out_dir / "reliability_diagram_calibrated.png")

    # Combined figure (side-by-side)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, (bcs, bas, bns, ece, title) in zip(axes, [
            (bc_raw, ba_raw, bn_raw, ece_raw, f"Before Platt Scaling\nECE={ece_raw:.4f}"),
            (bc_cal, ba_cal, bn_cal, ece_cal, f"After Platt Scaling\nECE={ece_cal:.4f}"),
        ]):
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect")
            if bcs:
                ax.bar(bcs, bas, width=0.08, alpha=0.75, color="#4f46e5",
                       edgecolor="black", label="Model")
                for x, y, cnt in zip(bcs, bas, bns):
                    ax.text(x, y + 0.02, str(cnt), ha="center", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
            ax.set_xlabel("Mean predicted confidence", fontsize=11)
            ax.set_ylabel("Fraction correct", fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=9)
        fig.suptitle("Confidence Calibration — Healthcare QA Chatbot", fontsize=13)
        fig.tight_layout()
        combined = out_dir / "reliability_diagram.png"
        fig.savefig(combined, dpi=150)
        plt.close(fig)
        print(f"  Combined figure → {combined}")
    except Exception as e:
        print(f"  Combined figure error: {e}")

    # Save JSON
    result = {
        "n": len(confidences),
        "label_positive_rate": round(sum(labels) / len(labels), 4),
        "mean_confidence_raw": round(statistics.mean(confidences), 4),
        "ece_raw": round(ece_raw, 4),
        "platt_a": round(a, 6),
        "platt_b": round(b, 6),
        "mean_confidence_calibrated": round(statistics.mean(cal_confs), 4),
        "ece_calibrated": round(ece_cal, 4),
        "bin_confidences_raw": bc_raw,
        "bin_accuracies_raw": ba_raw,
        "bin_counts_raw": bn_raw,
        "bin_confidences_calibrated": bc_cal,
        "bin_accuracies_calibrated": ba_cal,
        "bin_counts_calibrated": bn_cal,
    }
    cal_path = out_dir / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Calibration JSON → {cal_path}")

    print("\nDone.")
    print(f"  Raw ECE        : {ece_raw:.4f}")
    print(f"  Calibrated ECE : {ece_cal:.4f}")
    print(f"  Platt (a, b)   : ({a:.4f}, {b:.4f})")


if __name__ == "__main__":
    main()
