import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_path = "base_model"
lora_model_path = "healthcare_slm"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, lora_model_path)

prompt = """
### Instruction: Generate healthcare budget allocation recommendation for Uttar Pradesh 2028-29.
### Input: Disease Index 0.52, Infrastructure Gap High, HR Gap High.
### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(
    **inputs,
    max_length=200,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
