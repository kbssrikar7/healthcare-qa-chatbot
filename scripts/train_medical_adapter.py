#!/usr/bin/env python3
"""
Script to run QLoRA fine-tuning on medical datasets.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.loaders.dataset_loader import MedicalDatasetLoader
from src.fine_tuning.trainer import QLoRAConfig, QLoRATrainer, MedicalQADatasetBuilder

def main():
    parser = argparse.ArgumentParser(description='Train Medical Adapter')
    parser.add_argument('--output-dir', type=str, default='models/fine_tuned', help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples to use for quick training')
    
    args = parser.parse_args()
    
    print("🏥 Medical QA Fine-tuning Setup")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    
    # 1. Load Data
    print("\n📦 Loading datasets...")
    loader = MedicalDatasetLoader()
    qa_pairs = loader.load_all_qa_pairs()
    print(f"Total QA pairs found: {len(qa_pairs)}")
    
    # Filter/Sample data
    if args.max_samples and len(qa_pairs) > args.max_samples:
        import random
        random.seed(42)
        qa_pairs = random.sample(qa_pairs, args.max_samples)
        print(f"Subsampled to {len(qa_pairs)} examples for training.")
    
    # 2. Config
    config = QLoRAConfig(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Using small model for feasibility
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        max_steps=5, # Force short training for demo
        batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4
    )
    
    # 3. Initialize Trainer
    trainer_wrapper = QLoRATrainer(config)
    trainer_wrapper.setup_model()
    
    # 4. Prepare Dataset
    print("\n🔄 processing dataset...")
    dataset_builder = MedicalQADatasetBuilder(
        trainer_wrapper.tokenizer,
        max_length=config.max_seq_length
    )
    train_dataset = dataset_builder.prepare_dataset(qa_pairs)
    
    # 5. Train
    print("\n🚀 Starting Training...")
    trainer_wrapper.train(train_dataset)
    
    print("\n✅ Training Complete!")

if __name__ == "__main__":
    main()
