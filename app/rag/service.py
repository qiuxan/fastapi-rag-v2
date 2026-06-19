from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.rag.chunking import split_text
from app.rag.embeddings import EmbeddingProvider, MockEmbeddingProvider
from app.rag.llm import AnswerGenerator, MockAnswerGenerator
from app.rag.vector_store import ChunkRecord, InMemoryVectorStore


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    chunks_added: int


@dataclass(frozen=True)
class AnswerSource:
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[AnswerSource]


class RagService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        answer_generator: AnswerGenerator | None = None,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        self.embedding_provider = (
            embedding_provider if embedding_provider is not None else MockEmbeddingProvider()
        )
        self.answer_generator = answer_generator if answer_generator is not None else MockAnswerGenerator()
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()

    def ingest_document(self, text: str, metadata: dict[str, Any]) -> IngestResult:
        document_id = str(uuid4())
        chunks = split_text(text)
        records = [
            ChunkRecord(
                document_id=document_id,
                chunk_id=f"{document_id}-chunk-{index}",
                text=chunk,
                embedding=self.embedding_provider.embed(chunk),
                metadata=dict(metadata),
            )
            for index, chunk in enumerate(chunks)
        ]
        self.vector_store.add_many(records)
        return IngestResult(document_id=document_id, chunks_added=len(records))

    def answer_question(self, question: str, top_k: int) -> AnswerResult:
        query_embedding = self.embedding_provider.embed(question)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        answer = self.answer_generator.generate(question, results)
        sources = [
            AnswerSource(
                text=result.record.text,
                score=result.score,
                metadata=dict(result.record.metadata),
            )
            for result in results
        ]
        return AnswerResult(answer=answer, sources=sources)
