# OpenAI Provider Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the real OpenAI embedding and answer generation mode testable, documented, and manually verifiable while keeping mock mode as the default.

**Architecture:** Keep `RagService` provider-based: embeddings and answer generation are injected dependencies. Update OpenAI providers so tests can inject fake SDK clients, while runtime continues to create real `OpenAI` clients from `OPENAI_API_KEY`.

**Tech Stack:** FastAPI, OpenAI Python SDK, Pydantic Settings, pytest, SQLite vector store.

## Global Constraints

- Mock mode remains the default when `OPENAI_API_KEY` is not set.
- OpenAI mode is opt-in through environment configuration.
- Automated tests must not call the real OpenAI API.
- Manual live verification may call the real OpenAI API and can spend API credits.
- SQLite remains the vector store in both mock and OpenAI modes.
- Do not add streaming answers, Responses API migration, async OpenAI client, token counting, cost tracking, per-request provider selection, or production-grade secret management.
- API keys must only live in local `.env` or shell environment, never in committed files.

---

## File Structure

- `app/rag/embeddings.py`: add optional `client` injection to `OpenAIEmbeddingProvider`.
- `app/rag/llm.py`: add optional `client` injection to `OpenAIAnswerGenerator`.
- `tests/test_openai_providers.py`: new unit tests using fake OpenAI SDK clients.
- `.env.example`: keep safe placeholders and clarify optional OpenAI mode.
- `README.md`: document OpenAI mode setup and manual live verification.

---

### Task 1: Test OpenAI Provider Calls With Fake Clients

**Files:**
- Create: `tests/test_openai_providers.py`

**Interfaces:**
- Consumes: `OpenAIEmbeddingProvider(model: str, api_key: str | None = None, client: object | None = None)` and `OpenAIAnswerGenerator(model: str, api_key: str | None = None, client: object | None = None)`.
- Produces: tests that prove OpenAI providers call the SDK shape correctly without network access.

- [ ] **Step 1: Create failing tests**

Create `tests/test_openai_providers.py`:

```python
from types import SimpleNamespace

from app.rag.embeddings import OpenAIEmbeddingProvider
from app.rag.llm import OpenAIAnswerGenerator
from app.rag.vector_store import ChunkRecord, SearchResult


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: str) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsResource()


class FakeChatCompletionsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:
        self.calls.append({"model": model, "messages": messages})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OpenAI answer"))]
        )


class FakeChatResource:
    def __init__(self) -> None:
        self.completions = FakeChatCompletionsResource()


class FakeChatClient:
    def __init__(self) -> None:
        self.chat = FakeChatResource()


def test_openai_embedding_provider_sends_model_and_input_to_client() -> None:
    client = FakeEmbeddingClient()
    provider = OpenAIEmbeddingProvider(model="embedding-model", client=client)

    embedding = provider.embed("FastAPI and RAG")

    assert embedding == [0.1, 0.2, 0.3]
    assert client.embeddings.calls == [
        {"model": "embedding-model", "input": "FastAPI and RAG"}
    ]


def test_openai_answer_generator_returns_no_documents_message_without_api_call() -> None:
    client = FakeChatClient()
    generator = OpenAIAnswerGenerator(model="chat-model", client=client)

    answer = generator.generate("What is FastAPI?", [])

    assert answer == "No documents have been indexed yet."
    assert client.chat.completions.calls == []


def test_openai_answer_generator_sends_context_and_question_to_client() -> None:
    client = FakeChatClient()
    generator = OpenAIAnswerGenerator(model="chat-model", client=client)
    source = SearchResult(
        record=ChunkRecord(
            document_id="doc-1",
            chunk_id="doc-1-chunk-0",
            text="FastAPI builds Python APIs.",
            embedding=[1.0, 0.0],
            metadata={"title": "FastAPI"},
        ),
        score=1.0,
    )

    answer = generator.generate("What builds Python APIs?", [source])

    assert answer == "OpenAI answer"
    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    assert call["model"] == "chat-model"
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert "provided context" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "FastAPI builds Python APIs." in messages[1]["content"]
    assert "What builds Python APIs?" in messages[1]["content"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_openai_providers.py -v
```

Expected: fails because `OpenAIEmbeddingProvider` and `OpenAIAnswerGenerator` do not accept `client=...` yet.

- [ ] **Step 3: Add client injection to OpenAIEmbeddingProvider**

In `app/rag/embeddings.py`, change the constructor to:

```python
class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=resolved_api_key)
```

Keep `embed()` as:

```python
    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)
```

- [ ] **Step 4: Add client injection to OpenAIAnswerGenerator**

In `app/rag/llm.py`, change the constructor to:

```python
class OpenAIAnswerGenerator:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=resolved_api_key)
```

Keep `generate()` behavior the same.

- [ ] **Step 5: Run provider tests**

Run:

```bash
python -m pytest tests/test_openai_providers.py -v
```

Expected: all provider tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/rag/embeddings.py app/rag/llm.py tests/test_openai_providers.py
git commit -m "test: cover openai providers with fake clients"
```

---

### Task 2: Document OpenAI Mode And Local API Key Setup

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `Settings` fields `openai_api_key`, `openai_embedding_model`, `openai_chat_model`, `sqlite_path`.
- Produces: clear user-facing instructions for mock mode, OpenAI mode, and manual live verification.

- [ ] **Step 1: Update `.env.example`**

Replace `.env.example` with:

```text
# Leave empty for mock mode.
# Set this locally to enable real OpenAI embeddings and chat generation.
# Never commit your real API key.
OPENAI_API_KEY=

# Optional model overrides.
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini

# Optional local SQLite database path.
SQLITE_PATH=data/rag.db
```

- [ ] **Step 2: Update README OpenAI setup**

In `README.md`, add this section after "SQLite Storage":

````markdown
## OpenAI Mode

Mock mode is the default. It does not call external services and does not spend API credits.

To enable real OpenAI embeddings and answer generation, create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```text
OPENAI_API_KEY=your-api-key-here
```

Do not commit `.env`. It is ignored by git.

With `OPENAI_API_KEY` set:

- `/documents` uses `OpenAIEmbeddingProvider`.
- `/ask` uses `OpenAIAnswerGenerator`.
- chunks are still stored in local SQLite.
- API calls can spend OpenAI credits.

If you change `OPENAI_EMBEDDING_MODEL` after indexing documents, clear `data/rag.db`.
Different embedding models can produce vectors with different dimensions, and mixed dimensions cannot be compared.
````

- [ ] **Step 3: Add live verification commands to README**

Add this subsection after the existing curl examples:

````markdown
### Live OpenAI Check

Only run this after setting `OPENAI_API_KEY` in `.env`.

Start the API:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

Add a document:

```bash
curl -X POST http://127.0.0.1:8001/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a Python framework for building APIs. RAG retrieves context before answering.",
    "metadata": {"title": "OpenAI live test"}
  }'
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What framework builds Python APIs?",
    "top_k": 2
  }'
```

In OpenAI mode, the answer should not start with `Mock answer based on retrieved context`.
````

- [ ] **Step 4: Run full tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md
git commit -m "docs: explain openai mode"
```

---

### Task 3: Manual Live Verification With User API Key

**Files:**
- Local only: `.env`

**Interfaces:**
- Consumes: user-provided `OPENAI_API_KEY`.
- Produces: manual confirmation that real OpenAI mode works through the API.

- [ ] **Step 1: Create local `.env`**

Run:

```bash
cp .env.example .env
```

- [ ] **Step 2: User enters API key**

Open `.env` and replace:

```text
OPENAI_API_KEY=
```

with:

```text
OPENAI_API_KEY=<your real API key>
```

Do not paste the key into chat. Do not commit `.env`.

- [ ] **Step 3: Clear old mock database**

Run:

```bash
rm -f data/rag.db data/rag.db-shm data/rag.db-wal
```

This avoids mixing old mock 16-dimensional embeddings with OpenAI embedding vectors.

- [ ] **Step 4: Start the API**

Run:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

- [ ] **Step 5: Ingest a document**

In another terminal, run:

```bash
curl -X POST http://127.0.0.1:8001/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a Python framework for building APIs. RAG retrieves relevant context before answering.",
    "metadata": {"title": "OpenAI live test"}
  }'
```

Expected: response includes `"chunks_added": 1` or greater.

- [ ] **Step 6: Ask a question**

Run:

```bash
curl -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What framework builds Python APIs?",
    "top_k": 2
  }'
```

Expected: answer does not start with `"Mock answer based on retrieved context"` and sources include `"title": "OpenAI live test"`.

- [ ] **Step 7: Check git does not include secrets**

Run:

```bash
git status --short
```

Expected: `.env` is not listed.

---

## Final Verification

- [ ] Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] Confirm latest commits include:

```bash
git log --oneline -5
```

Expected: includes provider tests and OpenAI mode documentation commits.
