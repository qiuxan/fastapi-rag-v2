import os
from typing import Protocol

from openai import OpenAI

from app.rag.vector_store import SearchResult


class AnswerGenerator(Protocol):
    def generate(self, question: str, sources: list[SearchResult]) -> str:
        ...


class MockAnswerGenerator:
    def generate(self, question: str, sources: list[SearchResult]) -> str:
        if not sources:
            return "No documents have been indexed yet."

        context = " ".join(source.record.text for source in sources)
        return f"Mock answer based on retrieved context: {context}"


class OpenAIAnswerGenerator:
    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        self.model = model
        resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=resolved_api_key)

    def generate(self, question: str, sources: list[SearchResult]) -> str:
        context = "\n\n".join(source.record.text for source in sources)
        if not context:
            return "No documents have been indexed yet."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer using only the provided context. If the context is insufficient, say so.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ],
        )
        return response.choices[0].message.content or ""
