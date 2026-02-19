# test_indexing.py - create this file in rag_realestate/

from rag_system import RealEstateRAG
import os

# Check documents exist
print("Checking for documents...")
if os.path.exists("./documents"):
    docs = os.listdir("./documents")
    print(f"Found documents: {docs}")
else:
    print("ERROR: ./documents directory does not exist!")
    exit(1)

# Delete old index if it exists
import shutil
if os.path.exists("./chroma_db"):
    print("Removing old index...")
    shutil.rmtree("./chroma_db")

# Create new RAG system
print("\nInitializing RAG system...")
rag = RealEstateRAG(docs_path="./documents")

# Force indexing
print("\nIndexing documents (this may take 30-60 seconds)...")
num_chunks = rag.load_and_index_documents()
print(f"Successfully indexed {num_chunks} chunks")

# Build QA chain
print("\nBuilding QA chain...")
rag.build_qa_chain()

# Test query
print("\nTesting query...")
result = rag.query("What is the price of the Oak Street property?")

print(f"\nQuestion: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
