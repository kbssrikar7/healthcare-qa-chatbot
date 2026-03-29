#!/usr/bin/env python3
"""
QLoRA fine-tuning runner using downloaded medical datasets.

Trains TinyLlama-1.1B with LoRA adapters on ChatDoctor + MedQA data.
Saves the adapter to models/fine_tuned/medical_adapter.

CPU-safe: skips 4-bit quantization (bitsandbytes requires CUDA).
Uses bfloat16 when available, else float32.
Target: ~500 training examples, max_steps=100 to keep runtime tractable.

Usage:
    python scripts/run_training.py                # default: 100 steps
    python scripts/run_training.py --max-steps 50  # quick test
    python scripts/run_training.py --max-steps 500 # longer run
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

# Point to the shared HuggingFace cache where TinyLlama is already stored
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
# Avoid 40s network retries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_training_data(n: int = 500, seed: int = 42) -> list:
    """Load QA pairs from downloaded datasets."""
    raw = PROJECT_ROOT / "data" / "raw"
    pairs = []
    rng = random.Random(seed)

    # ── ChatDoctor HealthCareMagic ──────────────────────────────────────────
    chatdoc_path = raw / "chatdoctor" / "chatdoctor_healthcaremagic.parquet"
    if chatdoc_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(chatdoc_path)
            indices = rng.sample(range(len(df)), min(n // 2, len(df)))
            for idx in indices:
                row = df.iloc[idx]
                q = str(row.get("input", "")).strip()
                a = str(row.get("output", "")).strip()
                if len(q.split()) < 6 or len(a.split()) < 15:
                    continue
                # Strip email-like greetings from answers
                import re
                a = re.sub(r'^(Hi[,.]?|Hello[,.]?|Dear[^.]*\.|Thank you[^.]*\.)\s*', '', a)
                pairs.append({"question": q[:300], "answer": a[:500]})
            print(f"  ChatDoctor: {sum(1 for p in pairs if p)}/{len(indices)} pairs loaded")
        except Exception as e:
            print(f"  ChatDoctor load error: {e}")

    # ── MedQA USMLE ─────────────────────────────────────────────────────────
    medqa_path = raw / "medqa" / "medqa_usmle_train.parquet"
    if medqa_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(medqa_path)
            before = len(pairs)
            indices = rng.sample(range(len(df)), min(n // 3, len(df)))
            for idx in indices:
                row = df.iloc[idx]
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                if len(q.split()) < 10 or len(a.split()) < 5:
                    continue
                # Shorten long clinical vignettes to last 2 sentences
                import re
                sents = re.split(r'(?<=[.?])\s+', q)
                short_q = " ".join(sents[-2:]).strip() if len(sents) > 3 else q
                pairs.append({"question": short_q, "answer": a})
            print(f"  MedQA: {len(pairs) - before} pairs loaded")
        except Exception as e:
            print(f"  MedQA load error: {e}")

    # ── PubMedQA ────────────────────────────────────────────────────────────
    pubmed_path = raw / "pubmed" / "pubmedqa_labeled.parquet"
    if pubmed_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pubmed_path)
            df_yn = df[df["final_decision"].isin(["yes", "no"])].reset_index(drop=True)
            before = len(pairs)
            indices = rng.sample(range(len(df_yn)), min(n // 6, len(df_yn)))
            for idx in indices:
                row = df_yn.iloc[idx]
                q = str(row.get("question", "")).strip()
                a = str(row.get("long_answer", "")).strip()
                decision = str(row.get("final_decision", "")).strip()
                if len(q.split()) < 5 or len(a.split()) < 10:
                    continue
                full_a = f"{decision.capitalize()}. {a}"[:400]
                pairs.append({"question": q, "answer": full_a})
            print(f"  PubMedQA: {len(pairs) - before} pairs loaded")
        except Exception as e:
            print(f"  PubMedQA load error: {e}")

    rng.shuffle(pairs)
    result = pairs[:n]
    print(f"  Total training pairs: {len(result)}")
    return result


def run_training(max_steps: int = 100, output_dir: str = None):
    """Run QLoRA fine-tuning on medical QA data."""
    import torch

    output_path = output_dir or str(PROJECT_ROOT / "models" / "fine_tuned" / "medical_adapter")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  QLoRA Fine-Tuning: TinyLlama-1.1B on Medical QA Data")
    print(f"{'=' * 60}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Max steps: {max_steps}")
    print(f"  Output: {output_path}")
    print()

    # ── Load training data ───────────────────────────────────────────────────
    print("Loading training data...")
    qa_pairs = load_training_data(n=600)
    if not qa_pairs:
        print("ERROR: No training data found. Run scripts/download_data.py first.")
        return

    # ── Trainer setup ────────────────────────────────────────────────────────
    from src.fine_tuning.trainer import QLoRAConfig, QLoRATrainer, MedicalQADatasetBuilder

    # CPU-safe config: disable 4-bit (bitsandbytes requires CUDA)
    config = QLoRAConfig(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir=output_path,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        num_epochs=3,
        max_steps=max_steps,
        batch_size=1,                   # keep RAM usage low
        gradient_accumulation_steps=8,  # effective batch size = 8
        learning_rate=2e-4,
        max_seq_length=256,             # shorter for CPU speed
        load_in_4bit=False,             # 4-bit requires CUDA
    )

    trainer = QLoRATrainer(config)

    # Patch TrainingArguments to CPU-safe settings
    import transformers
    _orig_ta = transformers.TrainingArguments.__init__

    def _patched_ta(self, *a, **kw):
        kw["fp16"] = False       # fp16 doesn't work on CPU
        kw["optim"] = "adamw_torch"   # paged_adamw_8bit requires CUDA
        kw["no_cuda"] = not torch.cuda.is_available()
        _orig_ta(self, *a, **kw)

    transformers.TrainingArguments.__init__ = _patched_ta

    print("\nSetting up model with LoRA adapters...")
    trainer.setup_model()

    print("\nPreparing dataset...")
    builder = MedicalQADatasetBuilder(trainer.tokenizer, max_length=config.max_seq_length)
    train_ds = builder.prepare_dataset(qa_pairs[:int(0.9 * len(qa_pairs))])
    eval_ds = builder.prepare_dataset(qa_pairs[int(0.9 * len(qa_pairs)):]) if len(qa_pairs) > 10 else None

    print(f"  Train: {len(train_ds)} examples")
    if eval_ds:
        print(f"  Eval:  {len(eval_ds)} examples")

    print("\nStarting training...")
    trainer.train(train_ds, eval_ds)

    print(f"\nFine-tuning complete! Adapter saved to: {output_path}")

    # Save training metadata
    meta = {
        "base_model": config.base_model,
        "max_steps": max_steps,
        "train_examples": len(train_ds),
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "sources": ["ChatDoctor-HealthCareMagic", "MedQA-USMLE", "PubMedQA"],
    }
    with open(Path(output_path) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved to {output_path}/training_meta.json")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning runner")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max training steps (default 100, ~30 min on CPU)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()
    run_training(max_steps=args.max_steps, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
