import hashlib
import math
import os
from typing import Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class MockEmbeddingProvider:
    dimensions = 16

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = digest[0] % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


class OpenAIEmbeddingProvider:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model = model
        resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=resolved_api_key)

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)
