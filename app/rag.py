"""Small, local RAG pipeline used by the FastAPI application."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "technical_docs"
TOP_K = 4
MAX_RETRIES = 1


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    query_type: str
    documents: list[dict]
    relevant_documents: list[dict]
    retries: int
    answer: str
    sources: list[dict]


def _collection():
    """Return a persistent collection backed by OpenAI embeddings."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Add it to a .env file.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding = OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding)


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split on paragraphs when possible, with a small overlap for context."""
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def ingest_text(name: str, text: str, source: str | None = None) -> int:
    """Index one text document and replace an older version with the same name."""
    chunks = split_text(text)
    if not chunks:
        return 0
    collection = _collection()
    safe_name = hashlib.sha1(name.encode()).hexdigest()[:12]
    ids = [f"{safe_name}-{index}" for index in range(len(chunks))]
    collection.delete(where={"document_name": name})
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=[{"document_name": name, "source": source or name, "chunk": index + 1} for index in range(len(chunks))],
    )
    return len(chunks)


def ingest_seed_documents() -> int:
    """Index the included corpus. Safe to call repeatedly."""
    total = 0
    for path in DOCUMENTS_DIR.glob("*.md"):
        total += ingest_text(path.name, path.read_text(encoding="utf-8"), str(path))
    return total


def list_documents() -> list[dict]:
    collection = _collection()
    result = collection.get(include=["metadatas"])
    grouped: dict[str, dict] = {}
    for metadata in result.get("metadatas", []):
        if metadata is None:
            continue
        name = metadata["document_name"]
        grouped.setdefault(name, {"name": name, "source": metadata["source"], "chunks": 0})
        grouped[name]["chunks"] += 1
    return list(grouped.values())


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def analyze_query(state: RAGState) -> RAGState:
    question = state["question"]
    prompt = f"""Classify this technical documentation question and improve it only if useful.
Return exactly two lines:
TYPE: conceptual | how-to | troubleshooting | api-reference
QUERY: a concise search query

Question: {question}"""
    response = _llm().invoke(prompt).content
    lines = str(response).splitlines()
    query_type = next((line.split(":", 1)[1].strip() for line in lines if line.upper().startswith("TYPE:")), "conceptual")
    query = next((line.split(":", 1)[1].strip() for line in lines if line.upper().startswith("QUERY:")), question)
    return {"search_query": query or question, "query_type": query_type}


def retrieve(state: RAGState) -> RAGState:
    result = _collection().query(query_texts=[state["search_query"]], n_results=TOP_K, include=["documents", "metadatas", "distances"])
    documents = [
        {"content": content, "metadata": metadata, "distance": distance}
        for content, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
    return {"documents": documents}


def grade_documents(state: RAGState) -> RAGState:
    relevant = []
    for document in state.get("documents", []):
        prompt = f"""Is this documentation passage relevant for answering the question?
Reply with only YES or NO.

Question: {state['question']}
Passage: {document['content']}"""
        verdict = str(_llm().invoke(prompt).content).strip().upper()
        if verdict.startswith("YES"):
            relevant.append(document)
    return {"relevant_documents": relevant}


def route_after_grading(state: RAGState) -> Literal["generate", "retry", "fallback"]:
    if state.get("relevant_documents"):
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "retry"
    return "fallback"


def rewrite_query(state: RAGState) -> RAGState:
    prompt = f"""Rewrite this search query using different technical terms. Keep it short.
Original question: {state['question']}
Previous query: {state['search_query']}"""
    return {"search_query": str(_llm().invoke(prompt).content).strip(), "retries": state.get("retries", 0) + 1}


def generate(state: RAGState) -> RAGState:
    docs = state["relevant_documents"]
    context = "\n\n".join(f"[{i + 1}] {doc['content']}" for i, doc in enumerate(docs))
    prompt = f"""Answer the question using only the context below. Be concise. If context does not support a detail, say so.
Add citations like [1] after relevant claims.

Question: {state['question']}
Context:\n{context}"""
    answer = str(_llm().invoke(prompt).content)
    sources = [{"index": i + 1, **doc["metadata"]} for i, doc in enumerate(docs)]
    return {"answer": answer, "sources": sources}


def fallback(_: RAGState) -> RAGState:
    return {"answer": "I don't know based on the indexed documentation. Try ingesting a document that covers this topic.", "sources": []}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading, {"generate": "generate", "retry": "rewrite_query", "fallback": "fallback"})
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def ask(question: str) -> dict:
    result = build_graph().invoke({"question": question, "retries": 0})
    return {"answer": result["answer"], "sources": result["sources"], "query_type": result.get("query_type"), "retries": result.get("retries", 0)}
