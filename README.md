# FastAPI RAG V2

This project is a step-by-step FastAPI + RAG learning app.

The current milestone demonstrates the full loop:

1. Submit text to the API.
2. Split text into chunks.
3. Embed chunks.
4. Store chunks in a SQLite vector store.
5. Ask a question.
6. Retrieve relevant chunks.
7. Generate an answer from retrieved context.

By default the app uses mock providers and stores chunks in local SQLite, so it runs without external services and keeps indexed chunks after a restart.

If `OPENAI_API_KEY` is set, the app uses OpenAI embeddings and chat generation.

## Setup

Create and activate a virtual environment:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest -v
```

## Start The API

Use port `8001` so it does not conflict with the first project:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

The API will be available at:

```text
http://127.0.0.1:8001
```

Interactive API docs:

```text
http://127.0.0.1:8001/docs
```

## SQLite Storage

The default database path is:

```text
data/rag.db
```

The app creates this file automatically the first time you add a document.
The `data/` directory is ignored by git because it is local runtime state.

## Try It With Curl

Health check:

```bash
curl http://127.0.0.1:8001/health
```

Add a document:

```bash
curl -X POST http://127.0.0.1:8001/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a Python framework for building APIs. RAG retrieves relevant context before answering.",
    "metadata": {"title": "Learning notes"}
  }'
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What builds Python APIs?",
    "top_k": 2
  }'
```

To verify persistence, stop the server, start it again, and run the same `/ask` request.
The source should still be available because the chunk was stored in SQLite.

## Current Limits

- The default embedding provider is a mock implementation.
- The default answer generator is a mock implementation.
- File uploads are not included yet.
- Streaming responses are not included yet.
- User accounts are not included yet.

## Project Structure

```text
app/
  main.py
  api/
    routes.py
  rag/
    chunking.py
    embeddings.py
    llm.py
    service.py
    vector_store.py
  schemas.py
  settings.py
tests/
  conftest.py
  test_health.py
  test_rag_flow.py
  test_rag_units.py
  test_settings.py
```

## Learning Notes

The important flow is:

```text
POST /documents
-> validate request with Pydantic
-> split text into chunks
-> embed each chunk
-> store chunks in SQLite

POST /ask
-> validate request with Pydantic
-> embed the question
-> search similar chunks from SQLite
-> generate a mock answer from retrieved context
-> return answer and sources
```
