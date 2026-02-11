import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import joblib
import numpy as np

st.set_page_config(page_title="Healthcare AI", layout="wide")

# -------------------------
# CACHE MODEL LOADING
# -------------------------
@st.cache_resource
def load_models():
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

    return tokenizer, model

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

@st.cache_resource
def load_budget_model():
    return joblib.load("budget_predictor.pkl")

tokenizer, model = load_models()
db = load_vectorstore()
budget_model = load_budget_model()

def predict_budget(disease, infra, population):
    features = np.array([[disease, infra, population]])
    return budget_model.predict(features)[0]

st.title("Government Healthcare AI Assistant")

user_input = st.text_input("Ask your healthcare question:")

if user_input:

    # RAG Retrieval
    docs = db.similarity_search(user_input, k=2)
    retrieved_context = "\n".join([doc.page_content for doc in docs])

    # Budget prediction (demo values)
    predicted_budget = predict_budget(0.5, 2, 10)

    # CLEAN MINIMAL PROMPT
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

    st.markdown("### AI Response")
    st.write(response)
