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
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


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