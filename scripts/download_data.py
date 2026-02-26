#!/usr/bin/env python3
"""
Dataset download script for Healthcare QA Chatbot.
Downloads: MedQuAD, PubMedQA, MedMCQA, HealthCareMagic, MedQA USMLE, 
ChatDoctor, and Medical Meadow datasets.
"""
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_mediqa():
    """Download MEDIQA-related datasets."""
    print("\n" + "=" * 60)
    print("Downloading MedQuAD Dataset")
    print("=" * 60)
    output_dir = DATA_DIR / "mediqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MedQuAD - Medical Question Answering Dataset
    try:
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        dataset["train"].to_parquet(output_dir / "medquad.parquet")
        print(f"  [OK] MedQuAD: {len(dataset['train']):,} samples")
    except Exception as e:
        print(f"  [FAIL] MedQuAD download failed: {e}")
    
    return output_dir


def download_pubmedqa():
    """Download PubMedQA dataset."""
    print("\n" + "=" * 60)
    print("Downloading PubMedQA Dataset")
    print("=" * 60)
    output_dir = DATA_DIR / "pubmed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        dataset["train"].to_parquet(output_dir / "pubmedqa_labeled.parquet")
        print(f"  [OK] PubMedQA labeled: {len(dataset['train']):,} samples")
    except Exception as e:
        print(f"  [FAIL] PubMedQA download failed: {e}")
    
    return output_dir


def download_medmcqa():
    """Download MedMCQA dataset."""
    print("\n" + "=" * 60)
    print("Downloading MedMCQA Dataset")
    print("=" * 60)
    output_dir = DATA_DIR / "mediqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset("openlifescienceai/medmcqa")
        dataset["train"].to_parquet(output_dir / "medmcqa_train.parquet")
        dataset["validation"].to_parquet(output_dir / "medmcqa_val.parquet")
        print(f"  [OK] MedMCQA: {len(dataset['train']):,} train, {len(dataset['validation']):,} val")
    except Exception as e:
        print(f"  [FAIL] MedMCQA download failed: {e}")
    
    return output_dir


def download_healthcare_magic():
    """Download HealthCareMagic dataset."""
    print("\n" + "=" * 60)
    print("Downloading HealthCareMagic Dataset")
    print("=" * 60)
    output_dir = DATA_DIR / "mediqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # HealthCareMagic dataset
        dataset = load_dataset("wangrongsheng/HealthCareMagic-100k-en")
        # Take a subset for manageable size
        subset = dataset["train"].select(range(min(50000, len(dataset["train"]))))
        subset.to_parquet(output_dir / "healthcare_magic.parquet")
        print(f"  [OK] HealthCareMagic: {len(subset):,} samples")
    except Exception as e:
        print(f"  [FAIL] HealthCareMagic download failed: {e}")
    
    return output_dir


def download_medqa_usmle():
    """Download MedQA USMLE dataset - US medical licensing exam questions."""
    print("\n" + "=" * 60)
    print("Downloading MedQA USMLE Dataset")
    print("=" * 60)
    output_dir = DATA_DIR / "medqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # MedQA USMLE with 4 options
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
        
        # Save train and test splits
        if "train" in dataset:
            dataset["train"].to_parquet(output_dir / "medqa_usmle_train.parquet")
            print(f"  [OK] MedQA USMLE Train: {len(dataset['train']):,} samples")
        
        if "test" in dataset:
            dataset["test"].to_parquet(output_dir / "medqa_usmle_test.parquet")
            print(f"  [OK] MedQA USMLE Test: {len(dataset['test']):,} samples")
            
    except Exception as e:
        print(f"  [FAIL] MedQA USMLE download failed: {e}")
    
    return output_dir


def download_chatdoctor():
    """Download ChatDoctor iCliniq dataset - doctor-patient conversations."""
    print("\n" + "=" * 60)
    print("Downloading ChatDoctor iCliniq Dataset")
    print("=" * 60)
    output_dir = DATA_DIR / "chatdoctor"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ChatDoctor iCliniq - real doctor-patient conversations
        dataset = load_dataset("lavita/ChatDoctor-iCliniq")
        
        if "train" in dataset:
            # Take up to 100k samples
            data = dataset["train"]
            if len(data) > 100000:
                data = data.select(range(100000))
            data.to_parquet(output_dir / "chatdoctor_icliniq.parquet")
            print(f"  [OK] ChatDoctor iCliniq: {len(data):,} samples")
            
    except Exception as e:
        print(f"  [FAIL] ChatDoctor iCliniq download failed: {e}")
    
    # Also try the HealthCareMagic version
    try:
        dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
        if "train" in dataset:
            data = dataset["train"]
            if len(data) > 100000:
                data = data.select(range(100000))
            data.to_parquet(output_dir / "chatdoctor_healthcaremagic.parquet")
            print(f"  [OK] ChatDoctor HealthCareMagic: {len(data):,} samples")
    except Exception as e:
        print(f"  [INFO] ChatDoctor HealthCareMagic not available: {e}")
    
    return output_dir


def download_medical_meadow():
    """Download Medical Meadow datasets - curated medical instruction data."""
    print("\n" + "=" * 60)
    print("Downloading Medical Meadow Datasets")
    print("=" * 60)
    output_dir = DATA_DIR / "medical_meadow"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meadow_datasets = [
        ("medalpaca/medical_meadow_wikidoc", "wikidoc"),
        ("medalpaca/medical_meadow_wikidoc_patient_information", "wikidoc_patient"),
        ("medalpaca/medical_meadow_mediqa", "mediqa"),
        ("medalpaca/medical_meadow_medqa", "medqa"),
    ]
    
    for dataset_name, short_name in meadow_datasets:
        try:
            dataset = load_dataset(dataset_name)
            split_name = "train" if "train" in dataset else list(dataset.keys())[0]
            data = dataset[split_name]
            data.to_parquet(output_dir / f"meadow_{short_name}.parquet")
            print(f"  [OK] Medical Meadow {short_name}: {len(data):,} samples")
        except Exception as e:
            print(f"  [FAIL] Medical Meadow {short_name}: {e}")
    
    return output_dir


def download_additional_qa():
    """Download additional high-quality medical QA datasets."""
    print("\n" + "=" * 60)
    print("Downloading Additional Medical QA Datasets")
    print("=" * 60)
    output_dir = DATA_DIR / "additional"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download medical-qa-datasets (large consolidated dataset)
    try:
        dataset = load_dataset("lavita/medical-qa-datasets", split="train", streaming=True)
        # Take first 50k samples to keep manageable
        samples = []
        for i, sample in enumerate(tqdm(dataset, desc="Loading samples", total=50000)):
            samples.append(sample)
            if i >= 49999:
                break
        
        if samples:
            df = pd.DataFrame(samples)
            df.to_parquet(output_dir / "lavita_medical_qa.parquet")
            print(f"  [OK] Lavita Medical QA: {len(samples):,} samples")
    except Exception as e:
        print(f"  [INFO] Lavita Medical QA not available: {e}")
    
    return output_dir


def create_data_summary():
    """Create summary of downloaded data."""
    print("\n" + "=" * 60)
    print("Creating Data Summary")
    print("=" * 60)
    
    summary = {"datasets": [], "total_rows": 0}
    
    for parquet_file in sorted(DATA_DIR.rglob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_file)
            dataset_info = {
                "file": str(parquet_file.relative_to(DATA_DIR)),
                "rows": len(df),
                "columns": list(df.columns),
                "size_mb": round(parquet_file.stat().st_size / (1024 * 1024), 2)
            }
            summary["datasets"].append(dataset_info)
            summary["total_rows"] += len(df)
            print(f"  {dataset_info['file']}: {len(df):,} rows ({dataset_info['size_mb']} MB)")
        except Exception as e:
            print(f"  [FAIL] Could not read {parquet_file}: {e}")
    
    summary_path = DATA_DIR / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nData Summary saved to {summary_path}")
    return summary


def main():
    """Main download function."""
    print("\n" + "=" * 60)
    print("  HEALTHCARE QA CHATBOT - DATA DOWNLOAD")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    
    # Core datasets
    download_mediqa()
    download_pubmedqa()
    download_medmcqa()
    download_healthcare_magic()
    
    # New standard datasets
    download_medqa_usmle()
    download_chatdoctor()
    download_medical_meadow()
    download_additional_qa()
    
    # Summary
    print("\n")
    summary = create_data_summary()
    
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Total datasets: {len(summary['datasets'])}")
    print(f"  Total rows: {summary['total_rows']:,}")
    print(f"  Data location: {DATA_DIR}")
    print("\nNext step: Run 'python scripts/build_knowledge_base.py' to index the data")


if __name__ == "__main__":
    main()
