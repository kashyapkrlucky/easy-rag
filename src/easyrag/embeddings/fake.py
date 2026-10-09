"""Deterministic, dependency-free embedding provider for tests and offline development.

Not a semantic model: it hashes lowercase words into signed buckets ("feature hashing")
and normalises the result. Texts sharing words are therefore similar, which is enough
to test retrieval, indexing and RAG orchestration end-to-end without Ollama.
"""

import hashlib
import re
from collections.abc import Sequence

from easyrag.embeddings.base import Vector, normalize

_WORD = re.compile(r"\w+")


class FakeEmbeddingProvider:
    def __init__(self, dimension: int = 64, model_id: str = "fake/lexical-hashing") -> None:
        if dimension < 2:
            raise ValueError("dimension must be at least 2")
        self._dimension = dimension
        self._model_id = model_id
        # Call log, so tests can assert how often the "backend" was used.
        self.document_batches: list[int] = []
        self.query_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def embedded_document_count(self) -> int:
        return sum(self.document_batches)

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        self.document_batches.append(len(texts))
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> Vector:
        self.query_count += 1
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            bucket, sign = self._hash(word)
            vector[bucket] += sign
        if not any(vector):
            # No words (or all cancelled out): a deterministic unit vector per text.
            bucket, sign = self._hash("\x00" + text)
            vector[bucket] = sign
        return normalize(vector)

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self._dimension, (1.0 if value >> 63 else -1.0)
