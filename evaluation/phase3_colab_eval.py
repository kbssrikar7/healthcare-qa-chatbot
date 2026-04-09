#!/usr/bin/env python3
"""
Phase 3 Evaluation Script for Colab T4 GPU
Runs: Full ablation study (n=97), Bootstrap CIs, BioMistral testing
"""

import os
import sys
import json
import time
from pathlib import Path

# Set project root
PROJECT_ROOT = Path("/content/project") if Path("/content").exists() else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    print("=" * 80)
    print("PHASE 3 EVALUATION - Colab T4 GPU")
    print("=" * 80)

    # Task 1: Full Ablation Study (n=97)
    print("\n[1/3] Running Full Ablation Study (n=97)...")
    start = time.time()
    os.system(
        f"cd {PROJECT_ROOT} && python evaluation/run_ablation.py --n 97 --out-dir evaluation/results"
    )
    print(f"✓ Ablation complete ({time.time() - start:.1f}s)")

    # Task 2: Run full evaluation with bootstrap CIs
    print("\n[2/3] Running Full Evaluation with Bootstrap CIs...")
    start = time.time()
    os.system(
        f"cd {PROJECT_ROOT} && python evaluation/run_paper_eval.py --mode metrics --n 97"
    )
    print(f"✓ Metrics complete ({time.time() - start:.1f}s)")

    # Task 3: BioMistral evaluation
    print("\n[3/3] Running BioMistral Evaluation...")
    start = time.time()
    os.system(
        f"cd {PROJECT_ROOT} && python evaluation/run_paper_eval.py --mode metrics --model biomistral --n 20"
    )
    print(f"✓ BioMistral test complete ({time.time() - start:.1f}s)")

    # Results summary
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE - Results Summary")
    print("=" * 80)

    results_dir = PROJECT_ROOT / "evaluation/results"
    for fname in [
        "ablation.json",
        "metrics_full_tinyllama.json",
        "metrics_full_biomistral.json",
    ]:
        fpath = results_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
                print(f"\n{fname}:")
                print(json.dumps(data, indent=2)[:500])

    print("\n✅ All Phase 3 tasks completed!")
    print("   Results saved to: evaluation/results/")


if __name__ == "__main__":
    main()
