# rag_realestate/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_system import RealEstateRAG
import time

app = FastAPI(title="Real Estate RAG API", version="1.0.0")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list
    latency_ms: float


rag = RealEstateRAG(docs_path="./documents")


@app.on_event("startup")
async def startup():
    if not rag.load_existing_index():
        rag.load_and_index_documents()
    rag.build_qa_chain()


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "mistral"}


@app.post("/query", response_model=AnswerResponse)
async def query(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    result = rag.query(request.question)
    latency_ms = (time.time() - start) * 1000

    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=round(latency_ms, 2)
    )
