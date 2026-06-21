# Ingest Folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a command-line tool that ingests local `.txt`, `.md`, and `.markdown` knowledge-base files into the existing RAG SQLite store.

**Architecture:** Put reusable folder-ingestion logic in `app/rag/ingest_folder.py`, then keep the command-line wrapper in `app/scripts/ingest_folder.py`. The reusable logic accepts a `RagService`, so tests can use mock providers and a temporary SQLite store while runtime reuses `build_rag_service(Settings())`.

**Tech Stack:** Python stdlib `pathlib`, Python stdlib `argparse`, pytest, existing `RagService`, existing SQLite vector store.

## Global Constraints

- Support `.txt`, `.md`, and `.markdown`.
- Skip unsupported files.
- Skip empty or whitespace-only supported files.
- Recursively read files under the target folder.
- Use UTF-8 text reading.
- Reuse `RagService.ingest_document(text=..., metadata=...)`.
- Metadata must include `source`, `filename`, and `extension`.
- The CLI exits with status code `1` for missing paths and non-directory paths.
- The CLI exits with status code `0` when no supported files are found.
- Do not add PDF parsing, Word parsing, web UI, file upload API, deduplication, background jobs, or progress bars.

---

## File Structure

- Create `app/rag/ingest_folder.py`: reusable folder discovery and ingestion functions.
- Create `app/scripts/__init__.py`: package marker so `python -m app.scripts.ingest_folder` works.
- Create `app/scripts/ingest_folder.py`: CLI wrapper.
- Create `tests/test_ingest_folder.py`: unit tests for folder ingestion.
- Modify `README.md`: document the new command and workflow.

---

### Task 1: Reusable Folder Ingestion Logic

**Files:**
- Create: `app/rag/ingest_folder.py`
- Test: `tests/test_ingest_folder.py`

**Interfaces:**
- Consumes: `RagService.ingest_document(text: str, metadata: dict[str, Any]) -> IngestResult`.
- Produces: `SUPPORTED_EXTENSIONS`, `is_supported_file(path: Path) -> bool`, `ingest_folder(folder: Path, service: RagService) -> FolderIngestResult`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest_folder.py`:

```python
from pathlib import Path

from app.rag.ingest_folder import ingest_folder, is_supported_file
from app.rag.service import RagService


def test_is_supported_file_accepts_text_and_markdown() -> None:
    assert is_supported_file(Path("notes.txt")) is True
    assert is_supported_file(Path("notes.md")) is True
    assert is_supported_file(Path("notes.markdown")) is True


def test_is_supported_file_rejects_other_extensions() -> None:
    assert is_supported_file(Path("notes.pdf")) is False
    assert is_supported_file(Path("image.png")) is False
    assert is_supported_file(Path("README")) is False


def test_ingest_folder_indexes_supported_files_with_metadata(tmp_path) -> None:
    folder = tmp_path / "knowledge_base"
    folder.mkdir()
    nested = folder / "nested"
    nested.mkdir()
    (folder / "fastapi.md").write_text("FastAPI builds Python APIs.", encoding="utf-8")
    (nested / "rag.txt").write_text("RAG retrieves context.", encoding="utf-8")
    (folder / "ignored.pdf").write_text("not really a pdf", encoding="utf-8")

    service = RagService()
    result = ingest_folder(folder, service)
    answer = service.answer_question("What retrieves context?", top_k=2)

    assert result.files_ingested == 2
    assert result.files_skipped == 1
    assert result.chunks_added >= 2
    assert answer.sources
    assert {source.metadata["extension"] for source in answer.sources} <= {".md", ".txt"}
    assert all("source" in source.metadata for source in answer.sources)
    assert all("filename" in source.metadata for source in answer.sources)


def test_ingest_folder_skips_empty_supported_files(tmp_path) -> None:
    folder = tmp_path / "knowledge_base"
    folder.mkdir()
    (folder / "empty.md").write_text("   ", encoding="utf-8")

    service = RagService()
    result = ingest_folder(folder, service)

    assert result.files_ingested == 0
    assert result.files_skipped == 1
    assert result.chunks_added == 0


def test_ingest_folder_reports_no_supported_files(tmp_path) -> None:
    folder = tmp_path / "knowledge_base"
    folder.mkdir()
    (folder / "image.png").write_text("not text", encoding="utf-8")

    service = RagService()
    result = ingest_folder(folder, service)

    assert result.files_ingested == 0
    assert result.files_skipped == 1
    assert result.chunks_added == 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_ingest_folder.py -v
```

Expected: collection fails because `app.rag.ingest_folder` does not exist yet.

- [ ] **Step 3: Implement reusable ingestion logic**

Create `app/rag/ingest_folder.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.service import RagService

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class FileIngestResult:
    path: Path
    chunks_added: int


@dataclass(frozen=True)
class FolderIngestResult:
    files_ingested: int
    files_skipped: int
    chunks_added: int
    ingested_files: list[FileIngestResult] = field(default_factory=list)


def is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def build_file_metadata(path: Path, folder: Path) -> dict[str, Any]:
    source = path.relative_to(folder.parent).as_posix()
    return {
        "source": source,
        "filename": path.name,
        "extension": path.suffix.lower(),
    }


def iter_supported_and_skipped_files(folder: Path) -> tuple[list[Path], int]:
    supported: list[Path] = []
    skipped = 0
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if is_supported_file(path):
            supported.append(path)
        else:
            skipped += 1
    return supported, skipped


def ingest_folder(folder: Path, service: RagService) -> FolderIngestResult:
    supported_files, skipped = iter_supported_and_skipped_files(folder)
    ingested_files: list[FileIngestResult] = []
    chunks_added = 0

    for path in supported_files:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            skipped += 1
            continue

        result = service.ingest_document(
            text=text,
            metadata=build_file_metadata(path, folder),
        )
        ingested_files.append(FileIngestResult(path=path, chunks_added=result.chunks_added))
        chunks_added += result.chunks_added

    return FolderIngestResult(
        files_ingested=len(ingested_files),
        files_skipped=skipped,
        chunks_added=chunks_added,
        ingested_files=ingested_files,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_ingest_folder.py -v
```

Expected: all `test_ingest_folder.py` tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/rag/ingest_folder.py tests/test_ingest_folder.py
git commit -m "feat: add reusable folder ingestion"
```

---

### Task 2: Command-Line Entry Point

**Files:**
- Create: `app/scripts/__init__.py`
- Create: `app/scripts/ingest_folder.py`
- Test: `tests/test_ingest_folder.py`

**Interfaces:**
- Consumes: `ingest_folder(folder: Path, service: RagService) -> FolderIngestResult`.
- Produces: `main(argv: list[str] | None = None) -> int` and `python -m app.scripts.ingest_folder <folder>`.

- [ ] **Step 1: Add failing CLI tests**

Append to `tests/test_ingest_folder.py`:

```python
from app.scripts.ingest_folder import main


def test_ingest_folder_cli_returns_error_for_missing_folder(tmp_path, capsys) -> None:
    missing = tmp_path / "missing"

    exit_code = main([str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "does not exist" in captured.err


def test_ingest_folder_cli_returns_error_for_file_path(tmp_path, capsys) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("FastAPI notes", encoding="utf-8")

    exit_code = main([str(file_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not a directory" in captured.err
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_ingest_folder.py -v
```

Expected: collection fails because `app.scripts.ingest_folder` does not exist yet.

- [ ] **Step 3: Create scripts package**

Create `app/scripts/__init__.py`:

```python
"""Command-line helpers for the FastAPI RAG learning app."""
```

- [ ] **Step 4: Implement CLI module**

Create `app/scripts/ingest_folder.py`:

```python
import argparse
from pathlib import Path

from app.main import build_rag_service
from app.rag.ingest_folder import ingest_folder
from app.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a folder into the RAG index.")
    parser.add_argument("folder", help="Folder containing .txt, .md, or .markdown files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    folder = Path(args.folder)

    if not folder.exists():
        parser.exit(status=1, message=f"Error: {folder} does not exist\n")

    if not folder.is_dir():
        parser.exit(status=1, message=f"Error: {folder} is not a directory\n")

    service = build_rag_service(Settings())
    result = ingest_folder(folder, service)

    if result.files_ingested == 0:
        print(
            f"No supported non-empty files found. Files skipped: {result.files_skipped}"
        )
        return 0

    for file_result in result.ingested_files:
        print(f"Ingested {file_result.path}: {file_result.chunks_added} chunks")

    print(
        "Done. "
        f"Files ingested: {result.files_ingested}, "
        f"files skipped: {result.files_skipped}, "
        f"chunks added: {result.chunks_added}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
python -m pytest tests/test_ingest_folder.py -v
```

Expected: all `test_ingest_folder.py` tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/scripts/__init__.py app/scripts/ingest_folder.py tests/test_ingest_folder.py
git commit -m "feat: add ingest folder cli"
```

---

### Task 3: Documentation And Manual Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `python -m app.scripts.ingest_folder <folder>`.
- Produces: documented local knowledge-base workflow.

- [ ] **Step 1: Update README**

Add this section after "OpenAI Mode":

````markdown
## Ingest A Knowledge Base Folder

Create a local folder with text or Markdown files:

```text
knowledge_base/
  fastapi-notes.md
  rag-notes.txt
  project-faq.markdown
```

Supported file types:

- `.txt`
- `.md`
- `.markdown`

Ingest the folder:

```bash
python -m app.scripts.ingest_folder knowledge_base
```

The script reads supported files recursively and stores chunks in SQLite.
Each source includes metadata:

- `source`
- `filename`
- `extension`

If `.env` has `OPENAI_API_KEY`, ingestion uses real OpenAI embeddings.
If not, ingestion uses mock embeddings.

When switching between mock and OpenAI embeddings, clear the local SQLite database first:

```bash
rm -f data/rag.db data/rag.db-shm data/rag.db-wal
```
````

- [ ] **Step 2: Run full tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "docs: explain folder ingestion"
```

- [ ] **Step 4: Manual verification with a sample folder**

Create sample files:

```bash
mkdir -p knowledge_base
printf "FastAPI builds Python APIs. RAG retrieves context before answering.\n" > knowledge_base/fastapi-notes.md
printf "SQLite stores RAG chunks on disk.\n" > knowledge_base/sqlite-notes.txt
```

If `.env` has `OPENAI_API_KEY`, this will call OpenAI embeddings and can spend credits.

Run:

```bash
python -m app.scripts.ingest_folder knowledge_base
```

Expected: output lists the two ingested files and a final summary.

- [ ] **Step 5: Ask the API about ingested files**

Start the API:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

Ask:

```bash
curl -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Where are RAG chunks stored?",
    "top_k": 2
  }'
```

Expected: response sources include metadata from `knowledge_base/sqlite-notes.txt`.

---

## Final Verification

- [ ] Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] Run:

```bash
git status --short
```

Expected: no tracked source changes remain. Local runtime files such as `.env`, `data/`, or `knowledge_base/` must not be committed.
