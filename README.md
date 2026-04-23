# Hybrid Healthcare AI System

A comprehensive, AI-powered Government Medical Supply Chain and Fund Management decision support system. This hybrid architecture combines Machine Learning predictive algorithms, complex Retrieval-Augmented Generation (RAG) using official government healthcare PDFs, and state-of-the-art native execution of **Llama 3.1 8B Instruct** locally using the GGUF framework. 

---

## 🚀 Key Features

*   **100% Offline AI Chat:** Native chat UI featuring real-time token streaming powered by **Llama 3.1 8B Instruct (GGUF)** running securely and independently on local hardware (via `llama-cpp-python`).
*   **Predictive Analytics Engine:** Generates realistic 2028-2029 resource requirement predictions per Indian state using Gradient Boosting Regressors (GBR) and Random Forests.
*   **Massive Knowledge Store (RAG):** Integrates text chunks and statistics parsed directly from native government PDFs and PowerPoints using `langchain-huggingface` and FAISS for zero-hallucination analysis.
*   **Dashboard Visualizations:** Modern Streamlit dashboard delivering real-time interactive mapping and comparisons of healthcare infrastructure gaps, budgets, and vaccine coverages.

---

## 💻 Exact Step-by-Step Installation Guide (Windows)

Follow these exact steps to run this project on a fresh Windows PC.

### Prerequisites
* **Python 3.10 or 3.11** installed and added to your system `PATH`.
* A dedicated NVIDIA GPU with at least **8GB VRAM** (e.g., RTX 4060) is highly recommended for running the LLM. 

### Step 1: Clone or Download the Project
Copy the entire `healthcare_slm` folder onto your new machine (or `git clone` if hosted on GitHub).
Navigate into the directory using your terminal (PowerShell or Command Prompt):
```powershell
cd path\to\healthcare_slm
```

### Step 2: Create a Virtual Environment
It is heavily recommended to use an isolated environment to prevent dependency conflicts.
```powershell
python -m venv venv
```
Activate the environment:
```powershell
.\venv\Scripts\activate
```

### Step 3: Install Core Dependencies
With your virtual environment active, run the following command to install all the required Python libraries.
```powershell
pip install -r requirements.txt
```

### Step 4: Special Note on `llama-cpp-python` (Hardware Acceleration)
The standard `requirements.txt` installs a CPU version of `llama-cpp-python` by default. To get the **fastest streaming generation**, you must install the CUDA-accelerated version for your NVIDIA GPU. 

Run this in your active environment to force GPU acceleration (Make sure you have Visual Studio C++ build tools and CUDA Toolkit installed):
```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
$env:FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
*(If you do not have an NVIDIA GPU, you can skip this step and the bot will run slowly on your CPU).*

### Step 5: Train the Predictive Machine Learning Models
Before the dashboard can show future predictions, you must train the Scikit-Learn models using the historical CSV data.
```powershell
python train_predictors.py
```
This will take a few seconds and will generate several `.pkl` files and the final `predictions_2028_29.csv`.

### Step 6: Parse the PDF Documents (Build the RAG Vector Database)
The AI needs to read the massive government PDFs in the `documents/` folder. Build the FAISS search engine by running:
```powershell
python rag_pdf_setup.py
```
*Note: This will split thousands of pages of text into embedded chunks. It might take 2 to 5 minutes depending on your CPU.*

### Step 7: Launch the Application
You are completely ready. Start the local server:
```powershell
streamlit run app.py
```

### First Launch Note: 
The very first time you go to the `💬 Policy Chat` tab and ask a question, the backend will automatically download the **Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf** model file from HuggingFace (approx 4.9 GB). This is a one-time download and will be cached locally on your machine for all future sessions.

---

## 🛠️ Technology Stack
*   **LLM Backend:** `llama-cpp-python`, Llama 3.1 8B Instruct (4-bit Q4_K_M GGUF)
*   **RAG Engine:** FAISS, `sentence-transformers/all-MiniLM-L6-v2`, `langchain-huggingface`, `pypdf`
*   **Machine Learning:** Scikit-Learn (GBR, RandomForest)
*   **Frontend UI:** Streamlit, Plotly
