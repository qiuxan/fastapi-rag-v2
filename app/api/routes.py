from fastapi import APIRouter, Request

from app.rag.service import RagService
from app.schemas import AskRequest, AskResponse, DocumentRequest, DocumentResponse, SourceChunk

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


@router.post("/documents", response_model=DocumentResponse)
def add_document(request_body: DocumentRequest, request: Request) -> DocumentResponse:
    result = get_rag_service(request).ingest_document(
        text=request_body.text,
        metadata=request_body.metadata,
    )
    return DocumentResponse(document_id=result.document_id, chunks_added=result.chunks_added)


@router.post("/ask", response_model=AskResponse)
def ask(request_body: AskRequest, request: Request) -> AskResponse:
    result = get_rag_service(request).answer_question(
        question=request_body.question,
        top_k=request_body.top_k,
    )
    return AskResponse(
        answer=result.answer,
        sources=[
            SourceChunk(text=source.text, score=source.score, metadata=source.metadata)
            for source in result.sources
        ],
    )
