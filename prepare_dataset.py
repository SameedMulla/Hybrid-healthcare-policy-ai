import json

with open("data/healthcare_dataset.json") as f:
    data = json.load(f)

formatted = []

for item in data:
    text = f"""
### Instruction:
{item['instruction']}

### Input:
{item['context']}

### Response:
{item['response']}
"""
    formatted.append({"text": text})

with open("data/train.json", "w") as f:
    json.dump(formatted, f, indent=2)

print("Dataset formatted.")
