# rag_realestate/rag_system.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os


class RealEstateRAG:
    """
    RAG system for real estate document Q&A.
    Documents are chunked, embedded, stored in ChromaDB,
    and retrieved to answer questions using Mistral LLM.
    """

    def __init__(self, docs_path: str, db_path: str = "./chroma_db"):
        self.docs_path = docs_path
        self.db_path = db_path
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = OllamaEmbeddings(model="mistral")
        self.llm = OllamaLLM(model="mistral", temperature=0.1)

    def load_and_index_documents(self):
        """Load documents, split into chunks, store in vector DB."""
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        # Key decision: chunk size affects retrieval quality
        # Too large: retrieval less precise
        # Too small: lose context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print(f"Indexed {len(chunks)} chunks into ChromaDB")
        return len(chunks)

    def load_existing_index(self):
        """Load previously created vector store."""
        if os.path.exists(self.db_path):
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            print("Loaded existing index")
            return True
        return False

    def build_qa_chain(self):
        """Build the question-answering chain."""
        prompt_template = """You are a helpful real estate assistant.
Use the following context to answer the question.
If the answer is not in the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("QA chain ready")

    def query(self, question: str) -> dict:
        """Query the RAG system and return answer with sources."""
        result = self.qa_chain.invoke({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "sources": [
                doc.metadata.get("source", "unknown")
                for doc in result["source_documents"]
            ]
        }


def main():
    rag = RealEstateRAG(docs_path="./documents")

    if not rag.load_existing_index():
        rag.load_and_index_documents()

    rag.build_qa_chain()

    questions = [
        "What is the price of the Oak Street property?",
        "What was the median home price in Charlottesville in Q4 2024?",
        "What is the 1% rule in real estate investing?",
    ]

    for q in questions:
        result = rag.query(q)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
