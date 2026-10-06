# FastAPI Error Handling

Use `HTTPException` to return an error response from a path operation. Supply a status code and a detail message, for example `raise HTTPException(status_code=404, detail="Item not found")`.

Common status codes include 400 for a bad request, 404 for a missing resource, 415 for an unsupported media type, and 500 for an unexpected server failure.

## File uploads

FastAPI accepts uploaded files with `UploadFile` and `File`. The application needs the `python-multipart` package to process form data uploads.
