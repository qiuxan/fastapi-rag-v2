# SQLite Vector Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SQLite-backed vector store and make it the default runtime storage so indexed RAG chunks survive app restarts.

**Architecture:** Keep `RagService` as the orchestrator and give it a vector store dependency with the same `add_many/search/clear` shape used today. Add `SQLiteVectorStore` beside `InMemoryVectorStore`, storing embeddings and metadata as JSON text in a `chunks` table, then calculating cosine similarity in Python.

**Tech Stack:** FastAPI, Pydantic Settings, Python stdlib `sqlite3`, Python stdlib `json`, pytest, FastAPI `TestClient`.

## Global Constraints

- Keep `InMemoryVectorStore` for simple unit tests and algorithm learning.
- SQLite becomes the default runtime storage.
- Do not add `sqlite-vec`, `sqlite-vss`, approximate nearest-neighbor indexing, migrations, deletion, multi-user tenancy, or async database access.
- Store one row per chunk in a `chunks` table with `chunk_id`, `document_id`, `text`, `embedding`, and `metadata`.
- Store `embedding` and `metadata` as JSON text.
- Tests must use `tmp_path` database files, not the real development database.
- Ignore local SQLite database files and the `data/` directory in git.

---

## File Structure

- `app/rag/vector_store.py`: keep shared dataclasses, cosine similarity, `InMemoryVectorStore`; add `SQLiteVectorStore`.
- `app/rag/service.py`: widen the `vector_store` constructor type so `RagService` accepts either store.
- `app/settings.py`: add `sqlite_path: str = "data/rag.db"`.
- `app/main.py`: instantiate `SQLiteVectorStore(settings.sqlite_path)` in `build_rag_service()`.
- `tests/test_rag_units.py`: add SQLite store unit tests.
- `tests/test_settings.py`: assert default service uses SQLite store and settings expose the path.
- `tests/test_rag_flow.py`: add API persistence test across two app instances.
- `tests/conftest.py`: use a temporary SQLite database for the shared API test client.
- `.gitignore`: ignore `data/` and `*.db`.
- `README.md`: update the current behavior and add persistence notes.

---

### Task 1: Add SQLiteVectorStore

**Files:**
- Modify: `app/rag/vector_store.py`
- Test: `tests/test_rag_units.py`

**Interfaces:**
- Consumes: `ChunkRecord`, `SearchResult`, `cosine_similarity`.
- Produces: `SQLiteVectorStore(db_path: str | Path)` with `add_many(records)`, `search(query_embedding, top_k)`, and `clear()`.

- [ ] **Step 1: Add failing SQLite unit tests**

Add these imports near the top of `tests/test_rag_units.py`:

```python
from app.rag.vector_store import ChunkRecord, InMemoryVectorStore, SQLiteVectorStore, cosine_similarity
```

Replace the existing vector store import line with the line above, then append:

```python
def test_sqlite_vector_store_returns_most_similar_chunks(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "rag.db")
    fastapi = ChunkRecord(
        document_id="doc-1",
        chunk_id="doc-1-chunk-0",
        text="FastAPI builds Python APIs.",
        embedding=[1.0, 0.0],
        metadata={"title": "FastAPI"},
    )
    rag = ChunkRecord(
        document_id="doc-2",
        chunk_id="doc-2-chunk-0",
        text="RAG retrieves context.",
        embedding=[0.0, 1.0],
        metadata={"title": "RAG"},
    )

    store.add_many([fastapi, rag])
    results = store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].record.text == "FastAPI builds Python APIs."
    assert results[0].record.metadata == {"title": "FastAPI"}
    assert results[0].score == 1.0


def test_sqlite_vector_store_orders_results_by_descending_similarity(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "rag.db")
    low = ChunkRecord(
        document_id="doc-1",
        chunk_id="doc-1-chunk-0",
        text="Less relevant.",
        embedding=[0.0, 1.0],
        metadata={},
    )
    high = ChunkRecord(
        document_id="doc-2",
        chunk_id="doc-2-chunk-0",
        text="More relevant.",
        embedding=[1.0, 0.0],
        metadata={},
    )

    store.add_many([low, high])
    results = store.search([1.0, 0.0], top_k=2)

    assert [result.record.text for result in results] == ["More relevant.", "Less relevant."]
    assert [result.score for result in results] == [1.0, 0.0]


@pytest.mark.parametrize("top_k", [0, -1])
def test_sqlite_vector_store_rejects_non_positive_top_k(tmp_path, top_k: int) -> None:
    store = SQLiteVectorStore(tmp_path / "rag.db")

    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=top_k)


def test_sqlite_vector_store_persists_across_instances(tmp_path) -> None:
    db_path = tmp_path / "rag.db"
    first_store = SQLiteVectorStore(db_path)
    first_store.add_many(
        [
            ChunkRecord(
                document_id="doc-1",
                chunk_id="doc-1-chunk-0",
                text="FastAPI survives restarts with SQLite.",
                embedding=[1.0, 0.0],
                metadata={"title": "Persistence"},
            )
        ]
    )

    second_store = SQLiteVectorStore(db_path)
    results = second_store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].record.text == "FastAPI survives restarts with SQLite."
    assert results[0].record.metadata["title"] == "Persistence"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_rag_units.py -v
```

Expected: collection fails or tests fail because `SQLiteVectorStore` does not exist yet.

- [ ] **Step 3: Implement SQLiteVectorStore**

In `app/rag/vector_store.py`, add imports:

```python
import json
import sqlite3
from pathlib import Path
```

Then add this class after `InMemoryVectorStore`:

```python
class SQLiteVectorStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )

    def add_many(self, records: list[ChunkRecord]) -> None:
        rows = [
            (
                record.chunk_id,
                record.document_id,
                record.text,
                json.dumps(record.embedding),
                json.dumps(record.metadata),
            )
            for record in records
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id,
                    document_id,
                    text,
                    embedding,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, chunk_id, text, embedding, metadata
                FROM chunks
                """
            ).fetchall()

        scored = [
            SearchResult(
                record=ChunkRecord(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=text,
                    embedding=json.loads(embedding),
                    metadata=json.loads(metadata),
                ),
                score=cosine_similarity(query_embedding, json.loads(embedding)),
            )
            for document_id, chunk_id, text, embedding, metadata in rows
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
```

- [ ] **Step 4: Run SQLite unit tests**

Run:

```bash
python -m pytest tests/test_rag_units.py -v
```

Expected: all `test_rag_units.py` tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/rag/vector_store.py tests/test_rag_units.py
git commit -m "feat: add sqlite vector store"
```

---

### Task 2: Make SQLite The Default Store

**Files:**
- Modify: `app/rag/service.py`
- Modify: `app/settings.py`
- Modify: `app/main.py`
- Modify: `tests/test_settings.py`
- Modify: `tests/conftest.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `SQLiteVectorStore(db_path)`.
- Produces: `Settings.sqlite_path: str`; `build_rag_service(settings)` returns a `RagService` with `service.vector_store` set to `SQLiteVectorStore`.

- [ ] **Step 1: Add failing settings tests**

In `tests/test_settings.py`, import `SQLiteVectorStore`:

```python
from app.rag.vector_store import SQLiteVectorStore
```

Add these tests:

```python
def test_settings_default_sqlite_path() -> None:
    settings = Settings(openai_api_key=None)

    assert settings.sqlite_path == "data/rag.db"


def test_build_rag_service_uses_sqlite_store_by_default(tmp_path) -> None:
    service = build_rag_service(
        Settings(openai_api_key=None, sqlite_path=str(tmp_path / "rag.db"))
    )

    assert isinstance(service.vector_store, SQLiteVectorStore)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_settings.py -v
```

Expected: fails because `Settings.sqlite_path` or `SQLiteVectorStore` default wiring is missing.

- [ ] **Step 3: Add sqlite_path setting**

In `app/settings.py`, add:

```python
sqlite_path: str = "data/rag.db"
```

The class should contain:

```python
class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    sqlite_path: str = "data/rag.db"
```

- [ ] **Step 4: Wire SQLite into build_rag_service**

In `app/main.py`, import `SQLiteVectorStore`:

```python
from app.rag.vector_store import SQLiteVectorStore
```

Then pass it in both branches:

```python
vector_store=SQLiteVectorStore(settings.sqlite_path),
```

The mock branch should become:

```python
return RagService(
    embedding_provider=MockEmbeddingProvider(),
    answer_generator=MockAnswerGenerator(),
    vector_store=SQLiteVectorStore(settings.sqlite_path),
)
```

- [ ] **Step 5: Widen RagService vector_store type**

In `app/rag/service.py`, update imports:

```python
from app.rag.vector_store import ChunkRecord, InMemoryVectorStore, SQLiteVectorStore
```

Update the constructor parameter:

```python
vector_store: InMemoryVectorStore | SQLiteVectorStore | None = None,
```

- [ ] **Step 6: Keep API tests isolated with tmp_path**

In `tests/conftest.py`, change the fixture signature and app creation:

```python
from pathlib import Path
```

```python
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    app = create_app(Settings(openai_api_key=None, sqlite_path=str(tmp_path / "rag.db")))
    with TestClient(app) as test_client:
        yield test_client
```

- [ ] **Step 7: Ignore local SQLite files**

Add to `.gitignore`:

```gitignore
data/
*.db
*.db-shm
*.db-wal
```

- [ ] **Step 8: Run settings and API tests**

Run:

```bash
python -m pytest tests/test_settings.py tests/test_rag_flow.py -v
```

Expected: tests pass.

- [ ] **Step 9: Commit**

```bash
git add app/main.py app/rag/service.py app/settings.py tests/conftest.py tests/test_settings.py .gitignore
git commit -m "feat: use sqlite vector store by default"
```

---

### Task 3: Verify API Persistence And Update README

**Files:**
- Modify: `tests/test_rag_flow.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `create_app(Settings(sqlite_path=...))`.
- Produces: documented SQLite persistence behavior and a test that proves data survives app recreation.

- [ ] **Step 1: Add failing API persistence test**

Add imports to `tests/test_rag_flow.py`:

```python
from pathlib import Path

from app.main import create_app
from app.settings import Settings
```

Append this test:

```python
def test_ingested_document_persists_across_app_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    first_app = create_app(Settings(openai_api_key=None, sqlite_path=str(db_path)))

    with TestClient(first_app) as client:
        document_response = client.post(
            "/documents",
            json={
                "text": "SQLite keeps RAG chunks after the FastAPI app restarts.",
                "metadata": {"title": "SQLite notes"},
            },
        )

    assert document_response.status_code == 200

    second_app = create_app(Settings(openai_api_key=None, sqlite_path=str(db_path)))
    with TestClient(second_app) as client:
        ask_response = client.post(
            "/ask",
            json={"question": "What keeps RAG chunks after restart?", "top_k": 1},
        )

    assert ask_response.status_code == 200
    body = ask_response.json()
    assert "SQLite" in body["answer"]
    assert body["sources"][0]["metadata"]["title"] == "SQLite notes"
```

- [ ] **Step 2: Run persistence test**

Run:

```bash
python -m pytest tests/test_rag_flow.py::test_ingested_document_persists_across_app_instances -v
```

Expected: passes if Task 2 was completed correctly.

- [ ] **Step 3: Update README behavior**

In `README.md`, change the milestone list item:

```markdown
4. Store chunks in a SQLite vector store.
```

Replace the default storage paragraph with:

```markdown
By default the app uses mock providers and stores chunks in local SQLite, so it runs without external services and keeps indexed chunks after a restart.
```

Add this section after "Start The API":

````markdown
## SQLite Storage

The default database path is:

```text
data/rag.db
```

The app creates this file automatically the first time you add a document.
The `data/` directory is ignored by git because it is local runtime state.
````

Update current limits by removing:

```markdown
- The vector store is in memory.
- Data disappears when the server restarts.
- Persistent storage is not included yet.
```

Keep the mock provider and future feature limits.

- [ ] **Step 4: Run full test suite**

Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_rag_flow.py
git commit -m "docs: document sqlite persistence"
```

---

## Final Verification

- [ ] Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] Manual check:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8001/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "SQLite stores RAG chunks on disk. FastAPI serves the API.",
    "metadata": {"title": "Manual SQLite test"}
  }'
```

Restart the server, then run:

```bash
curl -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What stores RAG chunks on disk?",
    "top_k": 1
  }'
```

Expected: response includes a source with `"title": "Manual SQLite test"`.
