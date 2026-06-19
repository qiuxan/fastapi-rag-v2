from app.main import build_rag_service, create_app
from app.rag.embeddings import MockEmbeddingProvider, OpenAIEmbeddingProvider
from app.rag.llm import MockAnswerGenerator, OpenAIAnswerGenerator
from app.settings import Settings


def test_settings_default_to_mock_mode() -> None:
    settings = Settings(openai_api_key=None)

    assert settings.use_openai is False


def test_settings_use_openai_when_key_exists() -> None:
    settings = Settings(openai_api_key="test-key")

    assert settings.use_openai is True
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.openai_chat_model == "gpt-4.1-mini"


def test_build_rag_service_uses_mock_providers_without_key() -> None:
    service = build_rag_service(Settings(openai_api_key=None))

    assert isinstance(service.embedding_provider, MockEmbeddingProvider)
    assert isinstance(service.answer_generator, MockAnswerGenerator)


def test_build_rag_service_uses_openai_providers_with_key() -> None:
    service = build_rag_service(Settings(openai_api_key="test-key"))

    assert isinstance(service.embedding_provider, OpenAIEmbeddingProvider)
    assert isinstance(service.answer_generator, OpenAIAnswerGenerator)


def test_create_app_stores_rag_service() -> None:
    app = create_app(Settings(openai_api_key=None))

    assert hasattr(app.state, "rag_service")
