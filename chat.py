import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import joblib
import numpy as np

# -------------------------
# Load Vector Store
# -------------------------
print("Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# -------------------------
# Load Budget ML Model
# -------------------------
budget_model = joblib.load("budget_predictor.pkl")

def predict_budget(disease, infra, population):
    features = np.array([[disease, infra, population]])
    return budget_model.predict(features)[0]


# -------------------------
# Load Model
# -------------------------
base_model_path = "base_model"
lora_model_path = "healthcare_slm"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

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

model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()

print("\nHealthcare Policy AI Assistant with RAG (type 'exit' to quit)\n")

conversation_history = ""

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # -------------------------
    # Retrieve Relevant Documents (RAG)
    # -------------------------
    docs = db.similarity_search(user_input, k=2)
    retrieved_context = "\n".join([doc.page_content for doc in docs])

    # -------------------------
    # Budget Prediction (ML Model)
    # -------------------------
    disease_index = 0.5
    infra_gap = 2
    population = 10

    predicted_budget = predict_budget(disease_index, infra_gap, population)

    # -------------------------
    # Build Prompt (Improved Structured Prompt)
    # -------------------------
    prompt = f"""
Question: {user_input}

Relevant Data:
{retrieved_context}

Predicted Budget: ₹{predicted_budget:.2f} crore

Answer:
"""


    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.4,
    top_p=0.8,
    repetition_penalty=1.4,
    no_repeat_ngram_size=4,
    do_sample=True,
    use_cache=True
)


    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    response = decoded.split("Answer:")[-1].strip()

    print("\nAI:\n", response)
    print("-" * 60)
