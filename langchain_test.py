# test_setup.py
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

llm = OllamaLLM(model="mistral")
response = llm.invoke("Say setup successful and nothing else")
print(response)
print("Setup complete!")
