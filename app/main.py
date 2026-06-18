from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="FastAPI RAG V2")
app.include_router(router)
