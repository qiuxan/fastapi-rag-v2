from fastapi import APIRouter

from app.schemas import AskRequest, AskResponse, DocumentRequest, DocumentResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents", response_model=DocumentResponse)
def add_document(request: DocumentRequest) -> DocumentResponse:
    return DocumentResponse(document_id="placeholder", chunks_added=0)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return AskResponse(answer="No documents have been indexed yet.", sources=[])
