import argparse
from pathlib import Path
import sys

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
        print(f"Error: {folder} does not exist", file=sys.stderr)
        return 1

    if not folder.is_dir():
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        return 1

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