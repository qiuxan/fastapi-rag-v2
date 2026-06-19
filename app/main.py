from fastapi import FastAPI

from app.api.routes import router
from app.rag.embeddings import MockEmbeddingProvider, OpenAIEmbeddingProvider
from app.rag.llm import MockAnswerGenerator, OpenAIAnswerGenerator
from app.rag.service import RagService
from app.settings import Settings


def build_rag_service(settings: Settings) -> RagService:
    if settings.use_openai:
        return RagService(
            embedding_provider=OpenAIEmbeddingProvider(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            ),
            answer_generator=OpenAIAnswerGenerator(
                model=settings.openai_chat_model,
                api_key=settings.openai_api_key,
            ),
        )

    return RagService(
        embedding_provider=MockEmbeddingProvider(),
        answer_generator=MockAnswerGenerator(),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings if settings is not None else Settings()
    app = FastAPI(title="FastAPI RAG V2")
    app.state.rag_service = build_rag_service(resolved_settings)
    app.include_router(router)
    return app


app = create_app()
