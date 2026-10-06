# Running a FastAPI Application

Uvicorn is an ASGI server that can run a FastAPI application. Start an application defined as `app` in `app/main.py` with `uvicorn app.main:app --reload`.

The `--reload` option is useful during development because Uvicorn restarts the server after source code changes. Open `/docs` in the browser to use FastAPI's generated interactive API documentation.
