"""
Enhanced Fine-tuning Script for TinyLlama Healthcare SLM
---------------------------------------------------------
Improvements over original:
- Validation loss monitoring
- More LoRA target modules
- Longer context (512 tokens)
- Checkpoint saving
- Better training configuration
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

print("=" * 60)
print("Healthcare SLM Fine-tuning (TinyLlama 1.1B + LoRA)")
print("=" * 60)

model_path = "base_model"
output_path = "healthcare_slm"
checkpoint_path = "./checkpoint"

# -------------------------
# Load Tokenizer
# -------------------------
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load Model in 4-bit
# -------------------------
print("[2/5] Loading model in 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# -------------------------
# Configure LoRA (Enhanced)
# -------------------------
print("[3/5] Configuring LoRA adapters...")
lora_config = LoraConfig(
    r=16,                    # Increased from 8 for better capacity
    lora_alpha=32,           # Increased proportionally
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"        # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# -------------------------
# Load Dataset
# -------------------------
print("[4/5] Loading and tokenizing dataset...")
dataset = load_dataset("json", data_files={
    "train": "data/train.json",
    "validation": "data/val.json"
})

def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,        # Increased from 256
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names,
    num_proc=1
)

print(f"   Train: {len(dataset['train'])} examples")
print(f"   Validation: {len(dataset['validation'])} examples")

# -------------------------
# Training Configuration
# -------------------------
print("[5/5] Starting training...")

training_args = TrainingArguments(
    output_dir=checkpoint_path,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,              # Increased from 3
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    save_steps=50,
    save_total_limit=3,
    fp16=True,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="paged_adamw_32bit",        # Memory-efficient optimizer
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

# Train
trainer.train()

# -------------------------
# Save Model
# -------------------------
print("\nSaving fine-tuned model...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"   Model saved to: {output_path}/")
print(f"   Checkpoints in: {checkpoint_path}/")
print(f"{'='*60}")
