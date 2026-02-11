import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading PDF documents...")

documents = []

for file in os.listdir("documents"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"documents/{file}")
        documents.extend(loader.load())

print("Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)

db.save_local("vector_store")

print("PDF-based vector database created successfully.")
