"""
Prepare training dataset for fine-tuning TinyLlama SLM
Formats healthcare_dataset.json into instruction-following format
with train/validation split.
"""

import json
import random

random.seed(42)

print("Loading healthcare dataset...")
with open("data/healthcare_dataset.json") as f:
    data = json.load(f)

print(f"Loaded {len(data)} instruction-response pairs")

# Format into instruction-following template
formatted = []
for item in data:
    text = f"""### Instruction:
{item['instruction']}

### Input:
{item['context']}

### Response:
{item['response']}"""
    formatted.append({"text": text})

# Shuffle and split
random.shuffle(formatted)
split_idx = int(len(formatted) * 0.9)
train_data = formatted[:split_idx]
val_data = formatted[split_idx:]

# Save train set
with open("data/train.json", "w") as f:
    json.dump(train_data, f, indent=2)

# Save validation set
with open("data/val.json", "w") as f:
    json.dump(val_data, f, indent=2)

print(f"Training set: {len(train_data)} examples -> data/train.json")
print(f"Validation set: {len(val_data)} examples -> data/val.json")
print("Dataset preparation complete!")
