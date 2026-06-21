# Ingest Folder Design

## Goal

Add a command-line ingestion tool that reads local text knowledge-base files from a
folder and indexes them into the existing RAG pipeline.

The user should be able to run:

```bash
python -m app.scripts.ingest_folder knowledge_base
```

and have supported files ingested into SQLite through `RagService`.

## First Version Scope

Support these file types:

- `.txt`
- `.md`
- `.markdown`

Unsupported files will be skipped.

This task will not parse PDF, Word, HTML, websites, images, or binary files. Those are
separate learning tasks.

## Design Decision

Use a command-line script instead of an API endpoint.

Reasons:

- The files are on the user's local machine.
- A local script can safely read local folders.
- An API endpoint that reads arbitrary server-side folders creates confusing security
  boundaries.
- This keeps the learning focus on ingestion, metadata, and reuse of the existing RAG
  service.

## Data Flow

1. User runs `python -m app.scripts.ingest_folder knowledge_base`.
2. Script loads `Settings()`.
3. Script builds `RagService` using `build_rag_service(settings)`.
4. Script recursively finds supported files under the folder.
5. For each supported file:
   - read text as UTF-8
   - skip empty or whitespace-only files
   - create metadata
   - call `service.ingest_document(text=..., metadata=...)`
6. Script prints a concise summary.

## Metadata

Each ingested file will include metadata:

```python
{
    "source": "knowledge_base/rag-notes.md",
    "filename": "rag-notes.md",
    "extension": ".md",
}
```

This metadata will later show up in `/ask` sources.

## CLI Behavior

Command:

```bash
python -m app.scripts.ingest_folder <folder>
```

Behavior:

- If the folder does not exist, print an error and exit with status code `1`.
- If the path is not a directory, print an error and exit with status code `1`.
- If no supported files are found, print a message and exit with status code `0`.
- For supported files, ingest each valid non-empty file.
- Print one line per ingested file.
- Print a final summary with counts.

Example output:

```text
Ingested knowledge_base/fastapi-notes.md: 3 chunks
Ingested knowledge_base/rag-notes.txt: 2 chunks
Done. Files ingested: 2, files skipped: 0, chunks added: 5
```

## OpenAI Mode

The script uses the same settings as the API.

- If `.env` has `OPENAI_API_KEY`, ingestion uses OpenAI embeddings.
- If `.env` does not have `OPENAI_API_KEY`, ingestion uses mock embeddings.
- Both modes store chunks in SQLite.

Users should clear `data/rag.db` before switching between mock embeddings and OpenAI
embeddings, because vector dimensions differ.

## Testing

Automated tests will not call OpenAI.

Tests will use mock mode with a temporary SQLite database path.

Test coverage:

- supported file detection
- folder ingestion indexes `.txt`, `.md`, and `.markdown`
- unsupported files are skipped
- empty files are skipped
- missing folder returns a non-zero exit code
- metadata includes `source`, `filename`, and `extension`

## Out Of Scope

This task will not add:

- file upload API
- web UI
- PDF parsing
- Word document parsing
- recursive delete or update behavior
- deduplication
- background jobs
- progress bars
- production logging
