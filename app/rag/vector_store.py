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
