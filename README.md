# Hybrid Healthcare AI System

A comprehensive, AI-powered Government Medical Supply Chain and Fund Management decision support system. This hybrid architecture combines Machine Learning predictive algorithms, Retrieval-Augmented Generation (RAG), and local Mistral 7B execution through Ollama.

---

## Key Features

* **Offline AI Chat:** Streamlit chat UI with real-time token streaming powered by Mistral 7B through Ollama.
* **Predictive Analytics Engine:** Generates 2028-2029 resource requirement predictions per Indian state using Gradient Boosting Regressors and Random Forests.
* **Knowledge Store (RAG):** Integrates text chunks and statistics from `knowledge.txt`, government PDFs when available, PowerPoint files, and CSV data using `langchain-huggingface` and FAISS.
* **Dashboard Visualizations:** Streamlit dashboard with interactive healthcare infrastructure, budget, and vaccine coverage comparisons.

---

## Windows Setup

### Prerequisites

* Python 3.10 or newer installed and added to `PATH`.
* Ollama installed from https://ollama.com/download.

### Step 1: Open the Project

```powershell
cd C:\Users\mohit\Documents\GitHub\Healthcare-Policy-AI-Dashboard\Hybrid-healthcare-policy-ai
```

### Step 2: Create and Activate a Virtual Environment

```powershell
python -m venv .venv314
.\.venv314\Scripts\activate
```

### Step 3: Install Python Dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Pull the 7B Chat Model

```powershell
ollama pull mistral:7b
```

Keep Ollama running in the background while using the dashboard. To use a different Ollama model, set `OLLAMA_MODEL` before launching Streamlit.

### Step 5: Train the Predictive Models

```powershell
python train_predictors.py
```

### Step 6: Build the RAG Vector Store

```powershell
python rag_pdf_setup.py
```

If a `documents/` folder exists, PDFs inside it are included. If it does not exist, the setup uses the available knowledge base, PowerPoint, and CSV data.

### Step 7: Launch the Application

```powershell
streamlit run app.py
```

Then open http://localhost:8501.

---

## Technology Stack

* **LLM Backend:** Ollama, Mistral 7B
* **RAG Engine:** FAISS, `sentence-transformers/all-MiniLM-L6-v2`, `langchain-huggingface`, `pypdf`
* **Machine Learning:** Scikit-Learn
* **Frontend UI:** Streamlit, Plotly
