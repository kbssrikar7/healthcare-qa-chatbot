# Medical QA Fine-tuning - Google Colab Guide

## Quick Start

1. Open https://colab.research.google.com
2. Create new notebook
3. Runtime > Change runtime type > T4 GPU
4. Copy cells below and run in order

---

## Cell 1: Install

```python
!pip install -q transformers peft bitsandbytes accelerate datasets trl
```

## Cell 2: GPU Check

```python
import torch
assert torch.cuda.is_available(), "Enable GPU: Runtime > Change runtime type > T4"
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Cell 3: Load Data

```python
from datasets import load_dataset
ds = load_dataset("openlifescienceai/medmcqa", split="train").shuffle(42).select(range(5000))
print(f"Loaded {len(ds)} samples")
```

## Cell 4: Format

```python
def fmt(ex):
    opts = {0: ex.get('opa',''), 1: ex.get('opb',''), 2: ex.get('opc',''), 3: ex.get('opd','')}
    ans = opts.get(ex.get('cop', 0), '')
    exp = ex.get('exp', '') or ''
    return {"text": f"Question: {ex['question']}\nAnswer: {ans}\nExplanation: {exp}"}

ds = ds.map(fmt)
```

## Cell 5: Load Model with QLoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()
```

## Cell 6: Train

```python
args = TrainingArguments(
    output_dir="./medical_adapter",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(model=model, train_dataset=ds, args=args, tokenizer=tokenizer, dataset_text_field="text", max_seq_length=512)
trainer.train()
```

## Cell 7: Save and Download

```python
model.save_pretrained("./medical_adapter")
tokenizer.save_pretrained("./medical_adapter")
print("Saved to ./medical_adapter")

# Download
from google.colab import files
!zip -r medical_adapter.zip ./medical_adapter
files.download("medical_adapter.zip")
```

---

## After Download

1. Unzip `medical_adapter.zip`
2. Copy contents to your project: `models/fine_tuned/`
3. The adapter can be loaded with PEFT
