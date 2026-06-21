import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    document_id: str
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    record: ChunkRecord
    score: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length")

    left_magnitude = sqrt(sum(value * value for value in left))
    right_magnitude = sqrt(sum(value * value for value in right))
    if left_magnitude == 0 or right_magnitude == 0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_magnitude * right_magnitude)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[ChunkRecord] = []

    def add_many(self, records: list[ChunkRecord]) -> None:
        self._records.extend(records)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scored = [
            SearchResult(record=record, score=cosine_similarity(query_embedding, record.embedding))
            for record in self._records
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._records.clear()

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

        scored: list[SearchResult] = []
        for document_id, chunk_id, text, embedding_json, metadata_json in rows:
            embedding = json.loads(embedding_json)
            metadata = json.loads(metadata_json)
            record = ChunkRecord(
                document_id=document_id,
                chunk_id=chunk_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
            )
            scored.append(
                SearchResult(
                    record=record,
                    score=cosine_similarity(query_embedding, embedding),
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")