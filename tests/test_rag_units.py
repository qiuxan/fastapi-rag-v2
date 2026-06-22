import pytest

from app.rag.chunking import split_text
from app.rag.embeddings import MockEmbeddingProvider
from app.rag.vector_store import ChunkRecord, InMemoryVectorStore, SQLiteVectorStore, cosine_similarity

def test_split_text_creates_trimmed_chunks() -> None:
    text = "FastAPI helps build APIs. RAG retrieves context before answering."

    chunks = split_text(text, chunk_size=28, overlap=5)

    assert chunks == [
        "FastAPI helps build APIs. RA",
        "s. RAG retrieves context bef",
        "t before answering.",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"overlap": -1},
        {"chunk_size": 10, "overlap": 10},
    ],
)
def test_split_text_validates_invalid_parameters_before_blank_text(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        split_text("", **kwargs)


def test_mock_embeddings_are_deterministic() -> None:
    provider = MockEmbeddingProvider()

    first = provider.embed("FastAPI RAG")
    second = provider.embed("FastAPI RAG")

    assert first == second
    assert len(first) == 16
    assert any(value > 0 for value in first)


def test_cosine_similarity_scores_identical_vectors_highest() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


def test_vector_store_returns_most_similar_chunks() -> None:
    store = InMemoryVectorStore()
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
    assert results[0].score == 1.0


def test_vector_store_orders_results_by_descending_similarity() -> None:
    store = InMemoryVectorStore()
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
def test_vector_store_rejects_non_positive_top_k(top_k: int) -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=top_k)


from app.rag.llm import MockAnswerGenerator
from app.rag.service import RagService


def test_rag_service_ingests_document_and_answers_from_sources() -> None:
    service = RagService()

    result = service.ingest_document(
        text="FastAPI is a Python framework for building APIs. RAG retrieves relevant context.",
        metadata={"title": "Learning notes"},
    )
    answer = service.answer_question("What builds Python APIs?", top_k=2)

    assert result.chunks_added > 0
    assert answer.sources
    assert "FastAPI" in answer.answer
    assert answer.sources[0].metadata["title"] == "Learning notes"


def test_rag_service_handles_ask_before_ingest() -> None:
    service = RagService(answer_generator=MockAnswerGenerator())

    answer = service.answer_question("What is FastAPI?", top_k=3)

    assert answer.answer == "No documents have been indexed yet."
    assert answer.sources == []

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


def test_rag_service_filters_sources_below_min_score() -> None:
    service = RagService(min_score=0.2)
    service.vector_store.add_many(
        [
            ChunkRecord(
                document_id="doc-1",
                chunk_id="doc-1-chunk-0",
                text="Very low relevance.",
                embedding=[0.0] * 16,
                metadata={"title": "Low"},
            )
        ]
    )

    answer = service.answer_question("FastAPI", top_k=1)

    assert answer.sources == []


def test_rag_service_returns_i_do_not_know_when_matches_are_below_min_score() -> None:
    service = RagService(min_score=0.2)
    service.vector_store.add_many(
        [
            ChunkRecord(
                document_id="doc-1",
                chunk_id="doc-1-chunk-0",
                text="Unrelated low score content.",
                embedding=[0.0] * 16,
                metadata={"title": "Low"},
            )
        ]
    )

    answer = service.answer_question("FastAPI", top_k=1)

    assert answer.answer == "I don't know."
    assert answer.sources == []
