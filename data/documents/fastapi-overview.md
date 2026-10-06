# FastAPI Overview

FastAPI is a Python web framework for building APIs. It uses standard Python type hints to validate request data and create interactive API documentation.

## Path operations

A path operation combines an HTTP method and a URL path. `@app.get("/items/{item_id}")` creates an endpoint for GET requests. Function parameters declared in the path are converted and validated from the request URL.

## Request bodies

Use a Pydantic model for JSON request bodies. FastAPI reads the request body, validates it, and returns a clear validation error when required fields are missing or invalid.
