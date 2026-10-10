"""End-to-end tests against a real Ollama.

Start one with the embedding and chat models:
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d ollama ollama-pull
and run:
    EASYRAG_TEST_OLLAMA_URL=http://127.0.0.1:11434 uv run pytest -m e2e

Without EASYRAG_TEST_OLLAMA_URL every test in this package is skipped.
"""

import os
from collections.abc import AsyncIterator

import httpx2
import pytest

from easyrag.embeddings.ollama import OllamaEmbeddingProvider
from easyrag.llm.ollama import OllamaLLMProvider

TEST_OLLAMA_URL_ENV = "EASYRAG_TEST_OLLAMA_URL"
MODEL = os.environ.get("EASYRAG_TEST_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
DIMENSION = int(os.environ.get("EASYRAG_TEST_EMBEDDING_DIMENSION", "768"))
LLM_MODEL = os.environ.get("EASYRAG_TEST_LLM_MODEL", "qwen2.5:1.5b-instruct-q4_K_M")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
            # Only tests that talk to Ollama need it (fixturenames include dependencies).
            uses_ollama = "ollama_client" in getattr(item, "fixturenames", ())
            if uses_ollama and not os.environ.get(TEST_OLLAMA_URL_ENV):
                item.add_marker(pytest.mark.skip(reason=f"{TEST_OLLAMA_URL_ENV} is not set"))


@pytest.fixture
async def ollama_client() -> AsyncIterator[httpx2.AsyncClient]:
    async with httpx2.AsyncClient(
        base_url=os.environ[TEST_OLLAMA_URL_ENV], timeout=httpx2.Timeout(300.0, connect=5.0)
    ) as client:
        yield client


@pytest.fixture
def ollama_embeddings(ollama_client: httpx2.AsyncClient) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        ollama_client,
        model=MODEL,
        dimension=DIMENSION,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )


@pytest.fixture
def ollama_llm(ollama_client: httpx2.AsyncClient) -> OllamaLLMProvider:
    return OllamaLLMProvider(ollama_client, model=LLM_MODEL, context_window=8192)
