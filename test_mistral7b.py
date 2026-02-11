import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading 7B model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model.eval()

prompt = """
How should Maharashtra deploy vaccines in rural districts?
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

start = time.time()

output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.5,
    top_p=0.9,
    do_sample=True
)
end = time.time()
print(tokenizer.decode(output[0], skip_special_tokens=True))


print("Response Time:", round(end - start, 2), "seconds")