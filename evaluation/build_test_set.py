#!/usr/bin/env python3
"""
Build expanded test set (100+ questions with reference answers) from
downloaded datasets: MedQA USMLE, PubMedQA, ChatDoctor HealthCareMagic.

Usage:
    python evaluation/build_test_set.py
    python evaluation/build_test_set.py --output evaluation/test_set_v2.json --n 120
"""

import json
import re
import sys
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Categories to distribute over ───────────────────────────────────────────
MEDQA_CATEGORIES = {
    "step1": ["physiology", "pharmacology", "biochemistry"],
    "step2": ["diagnosis", "treatment", "clinical"],
    "step3": ["management", "emergency", "prevention"],
}

SYMPTOM_KEYWORDS = [
    "symptom",
    "present",
    "complain",
    "feel",
    "suffer",
    "pain",
    "ache",
    "nausea",
    "fever",
    "cough",
    "fatigue",
]
DIAGNOSIS_KEYWORDS = ["diagnos", "most likely", "cause", "etiology", "what is"]
TREATMENT_KEYWORDS = [
    "treat",
    "manag",
    "therap",
    "medication",
    "drug",
    "prescrib",
    "administer",
    "dosage",
]
PREVENTION_KEYWORDS = [
    "prevent",
    "avoid",
    "reduce risk",
    "protective",
    "vaccine",
    "screen",
]


def _classify(text: str) -> str:
    t = text.lower()
    if any(k in t for k in TREATMENT_KEYWORDS):
        return "treatment"
    if any(k in t for k in DIAGNOSIS_KEYWORDS):
        return "diagnosis"
    if any(k in t for k in SYMPTOM_KEYWORDS):
        return "symptoms"
    if any(k in t for k in PREVENTION_KEYWORDS):
        return "prevention"
    return "clinical"


def _extract_keywords(text: str, n: int = 6) -> list:
    """Pull the most informative nouns/medical terms from answer text."""
    # Remove short words, stopwords, keep medical-ish terms
    STOP = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "this",
        "that",
        "it",
        "its",
        "which",
        "who",
        "what",
        "when",
        "where",
        "and",
        "or",
        "but",
        "not",
        "no",
        "can",
        "also",
        "used",
        "most",
        "more",
        "than",
        "than",
        "from",
        "as",
        "if",
        "such",
        "one",
        "two",
        "three",
        "patient",
        "patients",
        "doctor",
        "medical",
        "health",
    }
    words = re.findall(r"\b[a-z][a-z\-]{3,}\b", text.lower())
    seen: dict = {}
    for w in words:
        if w not in STOP:
            seen[w] = seen.get(w, 0) + 1
    ranked = sorted(seen, key=lambda w: -seen[w])
    return ranked[:n]


def load_medqa(path: Path, n: int = 45, seed: int = 42) -> list:
    import pandas as pd

    df = pd.read_parquet(path)
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), min(n * 3, len(df)))  # oversample then trim
    cases = []
    for idx in indices:
        if len(cases) >= n:
            break
        row = df.iloc[idx]
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        # Skip very short questions/answers
        if len(q.split()) < 10 or len(a.split()) < 5:
            continue
        # Shorten question if it has a long clinical vignette (keep last sentence)
        sentences = re.split(r"(?<=[.?])\s+", q)
        if len(sentences) > 4:
            short_q = " ".join(sentences[-2:]).strip()
        else:
            short_q = q
        meta = str(row.get("meta_info", "step1"))
        case_id = f"medqa_{len(cases) + 1:03d}"
        cases.append(
            {
                "id": case_id,
                "query": short_q,
                "reference_answer": a,
                "expected_keywords": _extract_keywords(a, 6),
                "category": _classify(q),
                "source": f"MedQA-USMLE-{meta}",
                "relevant_ids": [case_id + "_src"],
            }
        )
    return cases


def load_pubmedqa(path: Path, n: int = 40, seed: int = 42) -> list:
    import pandas as pd

    df = pd.read_parquet(path)
    # Filter to yes/no final_decision so we have clear answers
    df_yn = df[df["final_decision"].isin(["yes", "no"])].reset_index(drop=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(df_yn)), min(n, len(df_yn)))
    cases = []
    for i, idx in enumerate(indices):
        row = df_yn.iloc[idx]
        q = str(row.get("question", "")).strip()
        long_ans = str(row.get("long_answer", "")).strip()
        decision = str(row.get("final_decision", "")).strip()
        if len(q.split()) < 5 or len(long_ans.split()) < 10:
            continue
        # Combine decision + long answer as reference
        ref = f"{decision.capitalize()}. {long_ans}" if long_ans else decision
        case_id = f"pubmed_{i + 1:03d}"
        cases.append(
            {
                "id": case_id,
                "query": q,
                "reference_answer": ref[:600],  # keep concise
                "expected_keywords": _extract_keywords(long_ans, 6),
                "category": "research",
                "source": "PubMedQA",
                "relevant_ids": [case_id + "_src"],
            }
        )
    return cases


def load_chatdoctor(path: Path, n: int = 25, seed: int = 42) -> list:
    import pandas as pd

    df = pd.read_parquet(path)
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), min(n * 5, len(df)))
    cases = []
    for idx in indices:
        if len(cases) >= n:
            break
        row = df.iloc[idx]
        q = str(row.get("input", "")).strip()
        a = str(row.get("output", "")).strip()
        if len(q.split()) < 8 or len(a.split()) < 20:
            continue
        # Trim greetings from answer
        clean_a = re.sub(
            r"^(Hi[,.]?|Hello[,.]?|Thank you[^.]*\.|Dear[^.]*\.)\s*", "", a
        )
        case_id = f"chatdoc_{len(cases) + 1:03d}"
        cases.append(
            {
                "id": case_id,
                "query": q[:300],
                "reference_answer": clean_a[:500],
                "expected_keywords": _extract_keywords(clean_a, 5),
                "category": _classify(q),
                "source": "ChatDoctor-HealthCareMagic",
                "relevant_ids": [case_id + "_src"],
            }
        )
    return cases


def build_test_set(n: int = 110, seed: int = 42) -> dict:
    raw = PROJECT_ROOT / "data" / "raw"
    cases = []

    medqa_path = raw / "medqa" / "medqa_usmle_test.parquet"
    pubmed_path = raw / "pubmed" / "pubmedqa_labeled.parquet"
    chatdoc_path = raw / "chatdoctor" / "chatdoctor_healthcaremagic.parquet"

    if medqa_path.exists():
        print(f"  Loading MedQA... ({medqa_path.name})")
        cases += load_medqa(medqa_path, n=45, seed=seed)
        print(f"    → {sum(1 for c in cases if 'medqa' in c['id'])} cases loaded")

    if pubmed_path.exists():
        print(f"  Loading PubMedQA... ({pubmed_path.name})")
        cases += load_pubmedqa(pubmed_path, n=40, seed=seed)
        print(f"    → {sum(1 for c in cases if 'pubmed' in c['id'])} cases loaded")

    if chatdoc_path.exists():
        print(f"  Loading ChatDoctor... ({chatdoc_path.name})")
        cases += load_chatdoctor(chatdoc_path, n=25, seed=seed)
        print(f"    → {sum(1 for c in cases if 'chatdoc' in c['id'])} cases loaded")

    # Shuffle and cap
    rng = random.Random(seed)
    rng.shuffle(cases)
    cases = cases[:n]

    return {
        "version": "2.0",
        "description": "Expanded evaluation test set (100+ Q&A with reference answers)",
        "created": "2026-03-29",
        "sources": ["MedQA-USMLE", "PubMedQA", "ChatDoctor-HealthCareMagic"],
        "total": len(cases),
        "test_cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description="Build expanded test set")
    parser.add_argument(
        "--output", default="evaluation/test_set_v2.json", help="Output JSON path"
    )
    parser.add_argument(
        "--n", type=int, default=110, help="Number of test cases to generate"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Building test set ({args.n} cases)...")
    data = build_test_set(n=args.n, seed=args.seed)

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    # Print category breakdown
    from collections import Counter

    cats = Counter(c["category"] for c in data["test_cases"])
    srcs = Counter(c["source"] for c in data["test_cases"])
    print(f"\nTest set built: {len(data['test_cases'])} cases → {out_path}")
    print("Category breakdown:", dict(cats))
    print("Source breakdown:", dict(srcs))


if __name__ == "__main__":
    main()
