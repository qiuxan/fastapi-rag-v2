from fastapi.testclient import TestClient

from pathlib import Path

from app.main import create_app
from app.settings import Settings


def test_document_text_must_not_be_empty(client: TestClient) -> None:
    response = client.post("/documents", json={"text": "", "metadata": {}})

    assert response.status_code == 422


def test_document_text_must_not_be_whitespace(client: TestClient) -> None:
    response = client.post("/documents", json={"text": "   ", "metadata": {}})

    assert response.status_code == 422


def test_question_must_not_be_empty(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "", "top_k": 3})

    assert response.status_code == 422


def test_question_must_not_be_whitespace(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "   ", "top_k": 3})

    assert response.status_code == 422


def test_top_k_must_be_positive(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is FastAPI?", "top_k": 0})

    assert response.status_code == 422


def test_ask_before_ingest_returns_empty_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is FastAPI?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "No documents have been indexed yet."
    assert body["sources"] == []


def test_ingest_then_ask_returns_answer_and_sources(client: TestClient) -> None:
    document_response = client.post(
        "/documents",
        json={
            "text": "FastAPI is a Python framework for building APIs. RAG retrieves relevant context before answering.",
            "metadata": {"title": "Learning notes"},
        },
    )

    assert document_response.status_code == 200
    document_body = document_response.json()
    assert document_body["document_id"]
    assert document_body["chunks_added"] >= 1

    ask_response = client.post("/ask", json={"question": "What builds Python APIs?", "top_k": 2})

    assert ask_response.status_code == 200
    ask_body = ask_response.json()
    assert "FastAPI" in ask_body["answer"]
    assert len(ask_body["sources"]) >= 1
    assert ask_body["sources"][0]["metadata"]["title"] == "Learning notes"
    
def test_ingested_document_persists_across_app_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    first_app = create_app(Settings(openai_api_key=None, sqlite_path=str(db_path)))

    with TestClient(first_app) as client:
        document_response = client.post(
            "/documents",
            json={
                "text": "SQLite keeps RAG chunks after the FastAPI app restarts.",
                "metadata": {"title": "SQLite notes"},
            },
        )

    assert document_response.status_code == 200

    second_app = create_app(Settings(openai_api_key=None, sqlite_path=str(db_path)))
    with TestClient(second_app) as client:
        ask_response = client.post(
            "/ask",
            json={"question": "What keeps RAG chunks after restart?", "top_k": 1},
        )

    assert ask_response.status_code == 200
    body = ask_response.json()
    assert "SQLite" in body["answer"]
    assert body["sources"][0]["metadata"]["title"] == "SQLite notes"