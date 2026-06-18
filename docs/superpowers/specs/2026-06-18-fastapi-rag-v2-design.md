# FastAPI RAG V2 Teaching Design

Date: 2026-06-18

## Goal

Rebuild the FastAPI + RAG learning project from a blank directory, step by step,
with the learner typing and running each stage.

The project teaches the complete first RAG loop:

1. Submit text to an API.
2. Validate the request with Pydantic.
3. Split text into chunks.
4. Embed chunks.
5. Store chunks in an in-memory vector store.
6. Ask a question.
7. Retrieve relevant chunks.
8. Generate a mock answer from retrieved context.

## Teaching Mode

Use pure guided mode:

- Explain the purpose of each task before code.
- Give exact commands and code for the learner to run.
- Wait for the learner's output before moving to the next step.
- Keep implementation minimal and readable.

## Project Location

```text
/Users/xianqiu/Documents/Codex/2026-06-12/fastapi-rag-v2
```

## Final Structure

```text
app/
  main.py
  api/routes.py
  schemas.py
  settings.py
  rag/
    chunking.py
    embeddings.py
    vector_store.py
    llm.py
    service.py
tests/
  conftest.py
  test_health.py
  test_rag_flow.py
  test_rag_units.py
  test_settings.py
```

## Task Sequence

1. Project skeleton and `GET /health`.
2. Schemas and request validation.
3. Chunking, mock embeddings, and in-memory vector store.
4. RAG service and mock answer generator.
5. Settings and provider selection.
6. Connect API routes to RAG service.
7. README, `.env.example`, and manual verification.

## First Milestone Scope

The first rebuild uses mock providers by default and does not require OpenAI.
OpenAI integration remains present as a later learning point, but the first pass
focuses on understanding FastAPI, request validation, the RAG data flow, and
tests.

