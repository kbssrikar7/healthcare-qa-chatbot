#!/usr/bin/env python3
"""
Medical QA Fine-tuning Script for Google Colab

HOW TO USE:
1. Open Google Colab (colab.research.google.com)
2. Create new notebook
3. Go to Runtime > Change runtime type > T4 GPU
4. Upload this file or copy cells into notebook
5. Run all cells in order
6. Download the adapter from ./medical_qa_adapter/
"""

# ============================================================
# CELL 1: Install Dependencies (run this first!)
# Uncomment the line below in Colab:
# !pip install -q transformers>=4.36.0 peft>=0.7.0 bitsandbytes>=0.41.0 accelerate>=0.25.0 datasets trl
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# ============================================================
# CELL 2: GPU Check
# ============================================================
print("=" * 50)
print("GPU CHECK")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory: {gpu_mem:.1f} GB")
else:
    print("WARNING: No GPU detected!")
    print("Go to: Runtime > Change runtime type > T4 GPU")

# ============================================================
# CELL 3: Configuration
# ============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./medical_qa_adapter"
TRAIN_SAMPLES = 5000  # Adjust: more samples = better but slower
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================
# CELL 4: Load Dataset
# ============================================================
print("\n" + "=" * 50)
print("LOADING DATASET")
print("=" * 50)

dataset = load_dataset("openlifescienceai/medmcqa", split="train")
print(f"Total dataset: {len(dataset)} examples")

dataset = dataset.shuffle(seed=42).select(range(TRAIN_SAMPLES))
print(f"Using: {TRAIN_SAMPLES} samples for training")

# ============================================================
# CELL 5: Format Data
# ============================================================
def format_example(example):
    q = example['question']
    opts = {0: example.get('opa',''), 1: example.get('opb',''), 
            2: example.get('opc',''), 3: example.get('opd','')}
    answer = opts.get(example.get('cop', 0), '')
    exp = example.get('exp', '') or 'Based on medical knowledge.'
    
    return {"text": f"
