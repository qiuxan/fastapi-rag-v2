from fastapi.testclient import TestClient


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
