# SQLite Vector Store Design

## Goal

Replace the default in-memory vector store with a SQLite-backed vector store so indexed
chunks survive application restarts. This is the next learning step after the basic
FastAPI + RAG flow.

The project will still keep the in-memory vector store for simple unit tests and for
understanding the core algorithm. SQLite becomes the default runtime storage.

## Why SQLite First

The current project stores chunks in a Python list. That makes the flow easy to
understand, but every server restart loses all indexed documents.

SQLite is a good next step because it teaches persistence without introducing a
separate database server or vector database extension. We can see the full RAG data
model directly:

- document chunk text
- embedding vector
- metadata
- similarity search behavior

This design intentionally does not use `sqlite-vec` or `sqlite-vss` yet. Those are
good later steps after the storage boundary is clear.

## Architecture

Add a new `SQLiteVectorStore` class beside the current `InMemoryVectorStore`.

Both stores will expose the same small interface:

- `add_many(records: list[ChunkRecord]) -> None`
- `search(query_embedding: list[float], top_k: int) -> list[SearchResult]`
- `clear() -> None`

`RagService` will continue to depend on this interface shape. It should not know
whether chunks are stored in memory or SQLite.

## Database Schema

SQLite will store one row per chunk:

```sql
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    metadata TEXT NOT NULL
);
```

`embedding` will be JSON text, for example:

```json
[0.1, 0.0, 0.3]
```

`metadata` will also be JSON text, for example:

```json
{"title": "Learning notes"}
```

This keeps the implementation transparent for learning. It is not the fastest
possible vector search design, but it is simple and easy to inspect.

## Data Flow

### Ingest

1. API receives `POST /documents`.
2. `RagService.ingest_document()` splits text into chunks.
3. Each chunk gets embedded.
4. `RagService` creates `ChunkRecord` objects.
5. `SQLiteVectorStore.add_many()` writes those records to the `chunks` table.

### Ask

1. API receives `POST /ask`.
2. `RagService.answer_question()` embeds the question.
3. `SQLiteVectorStore.search()` loads stored chunks from SQLite.
4. It decodes JSON embeddings.
5. It calculates cosine similarity in Python.
6. It sorts by score descending and returns the top results.
7. The mock or OpenAI answer generator writes an answer from the retrieved sources.

## Settings

Add a setting:

```python
sqlite_path: str = "data/rag.db"
```

`build_rag_service(settings)` will create:

```python
SQLiteVectorStore(settings.sqlite_path)
```

The `data/` directory and local `.db` files should be ignored by git.

Tests will pass temporary database paths with `tmp_path` so they do not write to the
real development database.

## Error Handling

`SQLiteVectorStore.search()` will keep the existing behavior:

- `top_k <= 0` raises `ValueError`
- no stored chunks returns an empty list
- mismatched vector lengths raise `ValueError` through `cosine_similarity`

The store will create the parent database directory if needed.

## Testing

Add tests for:

- SQLite store can add chunks and retrieve the most similar chunk.
- SQLite store orders results by descending similarity.
- SQLite store rejects non-positive `top_k`.
- SQLite store persists data across two store instances using the same database file.
- `build_rag_service()` uses `SQLiteVectorStore` by default.
- API flow can ingest a document, recreate the app with the same SQLite path, and ask a
  question using the previously stored chunk.

Existing in-memory tests will remain. They still explain the algorithm without file
I/O.

## Out Of Scope

This task will not add:

- `sqlite-vec`
- `sqlite-vss`
- approximate nearest-neighbor indexing
- migrations
- document deletion
- multi-user tenancy
- async database access

Those are useful future topics, but they would distract from the current learning goal:
understanding how RAG storage becomes persistent.
