"""
QLoRA fine-tuning for medical QA models.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""
    # Model
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "models/fine_tuned"
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training parameters
    num_epochs: int = 3
    max_steps: int = -1  # Default to -1 (use epochs)
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

class MedicalQADatasetBuilder:
    """Build training dataset for medical QA fine-tuning."""
    
    INSTRUCTION_TEMPLATE = """Below is a medical question. Provide an accurate, helpful answer based on medical knowledge.

### Question:
{question}

### Answer:
{answer}"""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def format_example(self, question: str, answer: str) -> str:
        """Format a single QA pair for training."""
        return self.INSTRUCTION_TEMPLATE.format(
            question=question,
            answer=answer
        )
    
    def prepare_dataset(self, qa_pairs: List[Dict]) -> Dataset:
        """Prepare training dataset from QA pairs."""
        formatted_texts = []
        
        for qa in qa_pairs:
            text = self.format_example(
                question=qa.get("question", ""),
                answer=qa.get("answer", "")
            )
            formatted_texts.append({"text": text})
        
        dataset = Dataset.from_list(formatted_texts)
        
        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset

class QLoRATrainer:
    """Fine-tune medical LLM using QLoRA."""
    
    def __init__(self, config: QLoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def setup_model(self):
        """Load and prepare model for QLoRA training."""
        print(f"🔄 Loading base model: {self.config.base_model}")
        
        # Quantization config
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_double_quant
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        if self.config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        print(f"✅ Model loaded. Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return self.peft_model
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Run QLoRA fine-tuning."""
        if self.peft_model is None:
            self.setup_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",  # or "wandb"
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        print("🚀 Starting QLoRA fine-tuning...")
        trainer.train()
        
        # Save model
        self.save_model()
        
        return trainer
    
    def save_model(self, path: Optional[str] = None):
        """Save fine-tuned model."""
        save_path = path or self.config.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_path = Path(save_path) / "qlora_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "base_model": self.config.base_model,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha
            }, f, indent=2)
        
        print(f"✅ Model saved to {save_path}")
    
    def load_fine_tuned(self, path: str):
        """Load a fine-tuned model."""
        from peft import PeftModel
        
        # Load config
        config_path = Path(path) / "qlora_config.json"
        with open(config_path) as f:
            saved_config = json.load(f)
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        base_model = AutoModelForCausalLM.from_pretrained(
            saved_config["base_model"],
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(base_model, path)
        
        print(f"✅ Loaded fine-tuned model from {path}")
        return self.peft_model


def main():
    """Example fine-tuning script."""
    print("🏥 Medical QA Fine-tuning with QLoRA\n")
    print("=" * 50)
    
    # Sample training data
    sample_qa = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include increased thirst (polydipsia), frequent urination (polyuria), unexplained weight loss, fatigue, blurred vision, slow-healing wounds, and frequent infections. Type 1 diabetes symptoms often appear suddenly, while Type 2 diabetes symptoms may develop gradually."
        },
        {
            "question": "How is high blood pressure treated?",
            "answer": "High blood pressure is typically treated through lifestyle modifications and medications. Lifestyle changes include reducing sodium intake, regular exercise, maintaining a healthy weight, limiting alcohol, and managing stress. Medications may include ACE inhibitors, ARBs, calcium channel blockers, diuretics, or beta-blockers."
        }
    ]
    
    # Initialize trainer
    config = QLoRAConfig(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir="models/medical_qa_lora",
        num_epochs=1,
        max_steps=5, # Force short training
        batch_size=2
    )
    
    trainer = QLoRATrainer(config)
    trainer.setup_model()
    
    # Prepare dataset
    dataset_builder = MedicalQADatasetBuilder(
        trainer.tokenizer,
        max_length=config.max_seq_length
    )
    train_dataset = dataset_builder.prepare_dataset(sample_qa)
    
    print(f"\n📊 Training dataset: {len(train_dataset)} examples")
    
    # Train (uncomment to actually train)
    # trainer.train(train_dataset)
    
    print("\n✅ Fine-tuning setup complete!")
    print("To train, uncomment trainer.train(train_dataset)")


if __name__ == "__main__":
    main()
