from types import SimpleNamespace

from app.rag.embeddings import OpenAIEmbeddingProvider
from app.rag.llm import OpenAIAnswerGenerator
from app.rag.vector_store import ChunkRecord, SearchResult


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: str) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsResource()


class FakeChatCompletionsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:
        self.calls.append({"model": model, "messages": messages})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OpenAI answer"))]
        )


class FakeChatResource:
    def __init__(self) -> None:
        self.completions = FakeChatCompletionsResource()


class FakeChatClient:
    def __init__(self) -> None:
        self.chat = FakeChatResource()


def test_openai_embedding_provider_sends_model_and_input_to_client() -> None:
    client = FakeEmbeddingClient()
    provider = OpenAIEmbeddingProvider(model="embedding-model", client=client)

    embedding = provider.embed("FastAPI and RAG")

    assert embedding == [0.1, 0.2, 0.3]
    assert client.embeddings.calls == [
        {"model": "embedding-model", "input": "FastAPI and RAG"}
    ]


def test_openai_answer_generator_returns_no_documents_message_without_api_call() -> None:
    client = FakeChatClient()
    generator = OpenAIAnswerGenerator(model="chat-model", client=client)

    answer = generator.generate("What is FastAPI?", [])

    assert answer == "No documents have been indexed yet."
    assert client.chat.completions.calls == []


def test_openai_answer_generator_sends_context_and_question_to_client() -> None:
    client = FakeChatClient()
    generator = OpenAIAnswerGenerator(model="chat-model", client=client)
    source = SearchResult(
        record=ChunkRecord(
            document_id="doc-1",
            chunk_id="doc-1-chunk-0",
            text="FastAPI builds Python APIs.",
            embedding=[1.0, 0.0],
            metadata={"title": "FastAPI"},
        ),
        score=1.0,
    )

    answer = generator.generate("What builds Python APIs?", [source])

    assert answer == "OpenAI answer"
    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    assert call["model"] == "chat-model"
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert "provided context" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "FastAPI builds Python APIs." in messages[1]["content"]
    assert "What builds Python APIs?" in messages[1]["content"]