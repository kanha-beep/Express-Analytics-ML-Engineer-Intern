from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag import ask, ingest_seed_documents, ingest_text, list_documents

load_dotenv()
app = FastAPI(title="Technical Documentation Assistant", version="1.0.0")
feedback_store: list[dict] = []


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class FeedbackRequest(BaseModel):
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)
    question: str | None = None


@app.on_event("startup")
def seed_corpus() -> None:
    if not list_documents():
        ingest_seed_documents()


@app.get("/")
def root():
    return {
        "message": "Technical Documentation Assistant is running.",
        "docs": "/docs",
        "endpoints": ["POST /query", "POST /ingest", "GET /documents", "POST /feedback"],
    }


@app.post("/query")
def query(request: QueryRequest):
    try:
        return ask(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if Path(file.filename or "document.txt").suffix.lower() not in {".txt", ".md"}:
        raise HTTPException(status_code=415, detail="Only .txt and .md files are supported.")
    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(status_code=400, detail="The uploaded document is empty.")
    try:
        chunks = ingest_text(file.filename or "document.txt", content)
        return {"message": "Document indexed", "filename": file.filename, "chunks": chunks}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/documents")
def documents():
    try:
        return {"documents": list_documents()}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    feedback_store.append(request.model_dump())
    return {"message": "Feedback recorded"}
