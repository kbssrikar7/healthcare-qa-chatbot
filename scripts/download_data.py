#!/usr/bin/env python3
"""
Dataset download script for Healthcare QA Chatbot.
Downloads: MEDIQA, PubMedQA, MedMCQA, and medical Wikipedia articles.
"""
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

DATA_DIR = Path("/home/kbs/final_project/data/raw")

def download_mediqa():
    """Download MEDIQA-related datasets."""
    print("📥 Downloading MEDIQA datasets...")
    output_dir = DATA_DIR / "mediqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MedQuAD - Medical Question Answering Dataset
    try:
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        dataset["train"].to_parquet(output_dir / "medquad.parquet")
        print(f"✅ MedQuAD: {len(dataset['train'])} samples")
    except Exception as e:
        print(f"⚠️ MedQuAD download failed: {e}")
    
    return output_dir

def download_pubmedqa():
    """Download PubMedQA dataset."""
    print("📥 Downloading PubMedQA...")
    output_dir = DATA_DIR / "pubmed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        dataset["train"].to_parquet(output_dir / "pubmedqa_labeled.parquet")
        print(f"✅ PubMedQA labeled: {len(dataset['train'])} samples")
    except Exception as e:
        print(f"⚠️ PubMedQA download failed: {e}")
    
    return output_dir

def download_medmcqa():
    """Download MedMCQA dataset."""
    print("📥 Downloading MedMCQA...")
    output_dir = DATA_DIR / "mediqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset("openlifescienceai/medmcqa")
        dataset["train"].to_parquet(output_dir / "medmcqa_train.parquet")
        dataset["validation"].to_parquet(output_dir / "medmcqa_val.parquet")
        print(f"✅ MedMCQA: {len(dataset['train'])} train, {len(dataset['validation'])} val")
    except Exception as e:
        print(f"⚠️ MedMCQA download failed: {e}")
    
    return output_dir

def download_medical_qa():
    """Download additional medical QA datasets."""
    print("📥 Downloading additional medical QA...")
    output_dir = DATA_DIR / "mediqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # HealthCareMagic dataset
        dataset = load_dataset("wangrongsheng/HealthCareMagic-100k-en")
        # Take a subset for manageable size
        subset = dataset["train"].select(range(min(50000, len(dataset["train"]))))
        subset.to_parquet(output_dir / "healthcare_magic.parquet")
        print(f"✅ HealthCareMagic: {len(subset)} samples")
    except Exception as e:
        print(f"⚠️ HealthCareMagic download failed: {e}")
    
    return output_dir

def create_data_summary():
    """Create summary of downloaded data."""
    summary = {"datasets": []}
    
    for parquet_file in DATA_DIR.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            summary["datasets"].append({
                "file": str(parquet_file.relative_to(DATA_DIR)),
                "rows": len(df),
                "columns": list(df.columns)
            })
        except Exception as e:
            print(f"⚠️ Could not read {parquet_file}: {e}")
    
    summary_path = DATA_DIR / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Data Summary saved to {summary_path}")
    return summary

def main():
    """Main download function."""
    print("🏥 Healthcare QA Chatbot - Data Download\n")
    print("=" * 50)
    
    download_mediqa()
    download_pubmedqa()
    download_medmcqa()
    download_medical_qa()
    
    print("\n" + "=" * 50)
    summary = create_data_summary()
    
    print("\n✅ Data download complete!")
    print(f"📁 Data location: {DATA_DIR}")
    print(f"📊 Total datasets: {len(summary['datasets'])}")

if __name__ == "__main__":
    main()
