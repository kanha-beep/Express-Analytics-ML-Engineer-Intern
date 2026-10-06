# RAG-Based Technical Documentation Assistant

A compact take-home implementation of a self-corrective RAG assistant using **Python, LangGraph, FastAPI, ChromaDB, and OpenAI**.

## What it does

- Indexes local Markdown/text technical documents in ChromaDB.
- Uses a LangGraph workflow to analyze the question, retrieve chunks, grade relevance, retry once when nothing is relevant, and generate a cited answer.
- Provides the required API endpoints: `POST /query`, `POST /ingest`, `GET /documents`, and `POST /feedback`.

## Workflow

`Query analysis → Retrieval → Document grading → Generation`

If grading finds no useful chunks, the graph rewrites the search query and returns to retrieval once. If that also fails, it returns a transparent “I don't know” response.

State carries the original question, search query, query type, retrieved chunks, filtered chunks, retry count, answer, and sources. This keeps the conditional routing explicit and easy to inspect.

## Project structure

```text
app/            FastAPI app and LangGraph workflow
data/documents/ Small FastAPI documentation corpus
requirements.txt
README.md
```

## Setup and run

uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI. The included documents are indexed automatically on the first run. ChromaDB data is stored locally in `chroma_db/`.

## Example API calls

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"question\": \"How do I handle a missing item in FastAPI?\"}"
curl http://127.0.0.1:8000/documents
curl -X POST http://127.0.0.1:8000/ingest -F "file=@my-notes.md"
curl -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" -d "{\"helpful\": true, \"comment\": \"Clear answer\"}"
```

## Design decisions and tradeoffs

The corpus is intentionally small and local so the project runs without a crawler or external database. Paragraph-first chunking uses 900 characters with 150-character overlap; that preserves documentation headings and nearby code guidance while avoiding overly large embedding inputs. ChromaDB is persistent and simple for a prototype. GPT-4o-mini handles query analysis, grading, query rewriting, and final generation for a consistent, low-cost implementation.

The feedback endpoint stores feedback in memory to keep the scope focused on the RAG flow. With more time, I would persist feedback, add user/session history, expose URL ingestion, and add a groundedness check after generation.
"# Express-Analytics-ML-Engineer-Intern" 
