# FastAPI RAG V2 Teaching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for interactive implementation with the learner. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the FastAPI + RAG learning project from a blank directory while teaching each layer.

**Architecture:** Build a small layered FastAPI app. Routes handle HTTP, schemas validate input, RAG modules handle chunking, embeddings, vector search, answer generation, and orchestration. Mock providers are used first so the project runs without OpenAI.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, pydantic-settings, Uvicorn, pytest, httpx, optional OpenAI SDK.

## Global Constraints

- Project path: `/Users/xianqiu/Documents/Codex/2026-06-12/fastapi-rag-v2`
- Teaching mode: explain first, then give commands/code, then wait for learner output.
- First pass uses mock providers and must not require `OPENAI_API_KEY`.
- Each task must end with a runnable command or test.
- Keep code minimal and readable.

---

## File Map

- `pyproject.toml`: package metadata, dependencies, pytest config, setuptools package discovery.
- `.gitignore`: local files, virtualenvs, cache files, `.env`, and editable install artifacts.
- `.env.example`: optional OpenAI settings.
- `README.md`: setup and curl walkthrough.
- `app/main.py`: FastAPI app factory and provider selection.
- `app/api/routes.py`: HTTP routes.
- `app/schemas.py`: Pydantic request and response models.
- `app/settings.py`: environment settings.
- `app/rag/chunking.py`: fixed-size text chunking.
- `app/rag/embeddings.py`: mock and OpenAI embedding providers.
- `app/rag/vector_store.py`: in-memory vector store and cosine similarity.
- `app/rag/llm.py`: mock and OpenAI answer generators.
- `app/rag/service.py`: RAG workflow orchestration.
- `tests/`: pytest coverage for each learning step.

---

### Task 1: Project Skeleton and Health Endpoint

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/api/__init__.py`
- Create: `app/api/routes.py`
- Create: `tests/conftest.py`
- Create: `tests/test_health.py`

**Interfaces:**
- Produces: `app.main.app`, a FastAPI application.
- Produces: `GET /health`, returning `{"status": "ok"}`.

- [ ] Create virtualenv and install dependencies.
- [ ] Write first test for `GET /health`.
- [ ] Create minimal FastAPI app and route.
- [ ] Run `python -m pytest tests/test_health.py -v`.
- [ ] Start server and visit `/health`.
- [ ] Commit with `feat: add FastAPI health endpoint`.

### Task 2: Schemas and Request Validation

**Files:**
- Create: `app/schemas.py`
- Modify: `app/api/routes.py`
- Create/Modify: `tests/test_rag_flow.py`

**Interfaces:**
- Consumes: `app.main.app`.
- Produces: `DocumentRequest`, `DocumentResponse`, `AskRequest`, `AskResponse`, `SourceChunk`.
- Produces placeholder `POST /documents` and `POST /ask`.

- [ ] Add tests for empty and whitespace-only document/question.
- [ ] Add tests for invalid `top_k`.
- [ ] Create Pydantic schemas.
- [ ] Add placeholder routes.
- [ ] Run validation tests.
- [ ] Commit with `feat: add API schemas and validation`.

### Task 3: Chunking, Mock Embeddings, and Vector Store

**Files:**
- Create: `app/rag/__init__.py`
- Create: `app/rag/chunking.py`
- Create: `app/rag/embeddings.py`
- Create: `app/rag/vector_store.py`
- Create/Modify: `tests/test_rag_units.py`

**Interfaces:**
- Produces: `split_text(text, chunk_size=500, overlap=50) -> list[str]`.
- Produces: `MockEmbeddingProvider.embed(text) -> list[float]`.
- Produces: `InMemoryVectorStore.add_many(records)` and `search(query_embedding, top_k)`.

- [ ] Test chunking output and invalid parameters.
- [ ] Test deterministic mock embeddings.
- [ ] Test cosine similarity.
- [ ] Test vector search ordering and invalid `top_k`.
- [ ] Implement minimal RAG primitives.
- [ ] Run RAG unit tests.
- [ ] Commit with `feat: add RAG core units`.

### Task 4: RAG Service and Mock Answer Generator

**Files:**
- Create: `app/rag/llm.py`
- Create: `app/rag/service.py`
- Modify: `tests/test_rag_units.py`

**Interfaces:**
- Consumes: `split_text`, `MockEmbeddingProvider`, `InMemoryVectorStore`.
- Produces: `RagService.ingest_document(text, metadata)`.
- Produces: `RagService.answer_question(question, top_k)`.

- [ ] Test ingest-then-answer flow.
- [ ] Test ask-before-ingest flow.
- [ ] Test metadata isolation.
- [ ] Implement mock answer generator.
- [ ] Implement RAG service orchestration.
- [ ] Run RAG unit tests and full suite.
- [ ] Commit with `feat: add RAG service orchestration`.

### Task 5: Settings and Provider Selection

**Files:**
- Create: `app/settings.py`
- Modify: `app/main.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_settings.py`

**Interfaces:**
- Produces: `Settings`.
- Produces: `create_app(settings=None)`.
- Produces: `build_rag_service(settings)`.

- [ ] Test default mock mode.
- [ ] Test OpenAI mode when key exists.
- [ ] Test injected settings do not require process env.
- [ ] Add app factory.
- [ ] Store `rag_service` on `app.state`.
- [ ] Run settings tests and full suite.
- [ ] Commit with `feat: add settings and provider selection`.

### Task 6: Connect API Routes to RAG Service

**Files:**
- Modify: `app/api/routes.py`
- Modify: `tests/test_rag_flow.py`

**Interfaces:**
- Consumes: `request.app.state.rag_service`.
- Produces real `POST /documents` ingest behavior.
- Produces real `POST /ask` retrieval and mock answer behavior.

- [ ] Add API flow test: ask before ingest.
- [ ] Add API flow test: ingest then ask.
- [ ] Replace placeholder routes with service calls.
- [ ] Run API flow tests and full suite.
- [ ] Manually test with curl.
- [ ] Commit with `feat: connect RAG service to API`.

### Task 7: Documentation and Manual Verification

**Files:**
- Create: `.env.example`
- Create: `README.md`

**Interfaces:**
- Produces documented setup commands.
- Produces documented curl examples for `/health`, `/documents`, and `/ask`.

- [ ] Add `.env.example`.
- [ ] Add README setup instructions.
- [ ] Add curl walkthrough.
- [ ] Run full test suite.
- [ ] Start Uvicorn and manually verify API.
- [ ] Commit with `docs: add setup and usage instructions`.

