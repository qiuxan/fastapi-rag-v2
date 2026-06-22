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