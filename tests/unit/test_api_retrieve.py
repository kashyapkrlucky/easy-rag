from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from easyrag.config import Settings
from easyrag.container import AppContainer, build_container
from easyrag.domain.chunks import RetrievedChunk
from easyrag.domain.errors import IndexIncompatibleError
from easyrag.embeddings.base import EmbeddingProviderUnavailableError, EmbeddingResponseError
from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.main import create_app
from easyrag.retrieval.vector import VectorRetriever
from easyrag.storage.vector_store import PostgresVectorStore
from tests.support import SettingsFactory

CHUNK = RetrievedChunk(
    chunk_id="c" * 32,
    source_path="guides/deployment.md",
    document_title="Deployment Guide",
    heading_path=("Deployment", "Docker Deployment"),
    section_anchor="docker-deployment",
    content="Run docker compose up.",
    start_line=12,
    end_line=14,
    score=0.8312345678,
)


class StubRetriever:
    def __init__(self, result: list[RetrievedChunk] | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "top_k": top_k, "min_score": min_score})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def client_with(settings: Settings, retriever: StubRetriever) -> TestClient:
    def factory(settings: Settings) -> AppContainer:
        return replace(build_container(settings), retriever=retriever)

    # raise_server_exceptions=False: assert on the 500 response instead of re-raising.
    return TestClient(
        create_app(settings, container_factory=factory), raise_server_exceptions=False
    )


def test_returns_results_in_api_shape(make_settings: SettingsFactory) -> None:
    retriever = StubRetriever([CHUNK])

    with client_with(make_settings(), retriever) as client:
        response = client.post("/api/v1/retrieve", json={"query": "  How do I deploy?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "How do I deploy?"
    assert body["embedding_model"].startswith("ollama/nomic-embed-text:v1.5")
    assert body["results"] == [
        {
            "chunk_id": "c" * 32,
            "document": "guides/deployment.md",
            "title": "Deployment Guide",
            "section": "Docker Deployment",
            "heading_path": ["Deployment", "Docker Deployment"],
            "anchor": "docker-deployment",
            "lines": [12, 14],
            "score": 0.831235,
            "content": "Run docker compose up.",
        }
    ]
    assert retriever.calls == [{"query": "How do I deploy?", "top_k": None, "min_score": None}]


def test_options_are_passed_through(make_settings: SettingsFactory) -> None:
    retriever = StubRetriever([])

    with client_with(make_settings(), retriever) as client:
        response = client.post(
            "/api/v1/retrieve", json={"query": "q", "top_k": 3, "min_score": 0.5}
        )

    assert response.json()["results"] == []
    assert retriever.calls == [{"query": "q", "top_k": 3, "min_score": 0.5}]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "q", "top_k": 0},
        {"query": "q", "top_k": 51},
        {"query": "q", "min_score": 2},
    ],
)
def test_invalid_requests_get_a_uniform_validation_error(
    make_settings: SettingsFactory, payload: dict[str, Any]
) -> None:
    retriever = StubRetriever([])

    with client_with(make_settings(), retriever) as client:
        response = client.post("/api/v1/retrieve", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["details"]
    assert retriever.calls == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (EmbeddingProviderUnavailableError("cannot reach Ollama"), 503, "embedding_unavailable"),
        (EmbeddingResponseError("bad output"), 502, "embedding_failed"),
        (
            IndexIncompatibleError(
                stored_model="a", stored_dimension=8, configured_model="b", configured_dimension=8
            ),
            503,
            "index_incompatible",
        ),
    ],
)
def test_domain_errors_map_to_http_errors(
    make_settings: SettingsFactory, error: Exception, status: int, code: str
) -> None:
    with client_with(make_settings(), StubRetriever(error)) as client:
        response = client.post("/api/v1/retrieve", json={"query": "q"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["message"]


def test_index_incompatible_message_explains_the_fix(make_settings: SettingsFactory) -> None:
    error = IndexIncompatibleError(
        stored_model="a", stored_dimension=8, configured_model="b", configured_dimension=8
    )

    with client_with(make_settings(), StubRetriever(error)) as client:
        message = client.post("/api/v1/retrieve", json={"query": "q"}).json()["error"]["message"]

    assert "easyrag index --rebuild" in message


def test_database_errors_do_not_leak_connection_details(make_settings: SettingsFactory) -> None:
    # Fake embeddings succeed, so the request reaches the real Postgres store, whose
    # database (with password "secret" in its URL) is unreachable.
    def factory(settings: Settings) -> AppContainer:
        container = build_container(settings)
        retriever = VectorRetriever(
            embeddings=FakeEmbeddingProvider(), search=PostgresVectorStore(container.engine)
        )
        return replace(container, retriever=retriever)

    app = create_app(make_settings(), container_factory=factory)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/retrieve", json={"query": "q"})

    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "database_unavailable", "message": "The database is unavailable"}
    }
    assert "secret" not in response.text


def test_unexpected_errors_return_a_generic_500(make_settings: SettingsFactory) -> None:
    with client_with(make_settings(), StubRetriever(RuntimeError("internal detail"))) as client:
        response = client.post("/api/v1/retrieve", json={"query": "q"})

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "An unexpected error occurred"}
    }
