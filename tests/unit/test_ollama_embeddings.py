import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from easyrag.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderUnavailableError,
    EmbeddingResponseError,
)
from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.embeddings.ollama import OllamaEmbeddingProvider
from tests.contracts.embedding_provider import EmbeddingProviderContract

DIMENSION = 8

Scripted = httpx2.Response | Exception


class FakeOllama:
    """In-process stand-in for the Ollama HTTP API, served through httpx2.MockTransport.

    By default it embeds inputs deterministically; tests can queue scripted responses
    or exceptions that are consumed before falling back to the default behaviour.
    """

    def __init__(self, dimension: int = DIMENSION, embedding_length: int | None = None) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.script: list[Scripted] = []
        self._embedder = FakeEmbeddingProvider(dimension)
        self._embedding_length = dimension if embedding_length is None else embedding_length

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if request.url.path == "/api/embed":
            embeddings = [self._embedder._embed(text) for text in body["input"]]
            return httpx2.Response(200, json={"model": body["model"], "embeddings": embeddings})
        if request.url.path == "/api/show":
            info: dict[str, object] = {
                "general.architecture": "nomic-bert",
                "nomic-bert.embedding_length": self._embedding_length,
            }
            return httpx2.Response(200, json={"model_info": info})
        return httpx2.Response(404, json={"error": "not found"})

    def embed_requests(self) -> list[dict[str, Any]]:
        return [body for path, body in self.requests if path == "/api/embed"]


def make_provider(backend: FakeOllama, **overrides: Any) -> OllamaEmbeddingProvider:
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(backend.handler), base_url="http://ollama.test"
    )
    options: dict[str, Any] = {
        "model": "nomic-embed-text:v1.5",
        "dimension": DIMENSION,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "retry_backoff_seconds": 0,
        **overrides,
    }
    return OllamaEmbeddingProvider(client, **options)


def error(status: int, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": message})


class TestOllamaEmbeddingProviderContract(EmbeddingProviderContract):
    @pytest.fixture
    def provider(self) -> EmbeddingProvider:
        return make_provider(FakeOllama())


# -- requests --------------------------------------------------------------------------


async def test_documents_are_sent_with_document_prefix_and_without_truncation() -> None:
    backend = FakeOllama()

    await make_provider(backend).embed_documents(["alpha", "beta"])

    assert backend.embed_requests() == [
        {
            "model": "nomic-embed-text:v1.5",
            "input": ["search_document: alpha", "search_document: beta"],
            "truncate": False,
        }
    ]


async def test_query_is_sent_with_query_prefix() -> None:
    backend = FakeOllama()

    await make_provider(backend).embed_query("how to deploy?")

    assert backend.embed_requests()[0]["input"] == ["search_query: how to deploy?"]


async def test_large_inputs_are_batched_in_order() -> None:
    backend = FakeOllama()
    texts = [f"text {i}" for i in range(70)]

    vectors = await make_provider(backend, batch_size=32).embed_documents(texts)

    assert [len(r["input"]) for r in backend.embed_requests()] == [32, 32, 6]
    expected = await make_provider(FakeOllama(), batch_size=70).embed_documents(texts)
    assert vectors == expected


async def test_empty_input_makes_no_request() -> None:
    backend = FakeOllama()

    assert await make_provider(backend).embed_documents([]) == []
    assert backend.requests == []


def test_model_id_identifies_backend_model_and_prefixes() -> None:
    with_prefixes = make_provider(FakeOllama()).model_id
    other_prefixes = make_provider(FakeOllama(), query_prefix="query: ").model_id
    no_prefixes = make_provider(FakeOllama(), query_prefix="", document_prefix="").model_id

    assert with_prefixes.startswith("ollama/nomic-embed-text:v1.5+p:")
    assert with_prefixes != other_prefixes  # different prefixes, different vector space
    assert no_prefixes == "ollama/nomic-embed-text:v1.5"


# -- response validation ---------------------------------------------------------------


async def test_output_is_normalised() -> None:
    backend = FakeOllama(dimension=2)
    backend.script = [httpx2.Response(200, json={"embeddings": [[3, 4]]})]

    assert await make_provider(backend, dimension=2).embed_documents(["x"]) == [[0.6, 0.8]]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx2.Response(200, json={"embeddings": [[1.0] * DIMENSION]}), "expected 2 embeddings"),
        (httpx2.Response(200, json={"embeddings": [[1.0], [1.0]]}), "has dimension 1"),
        (httpx2.Response(200, json={"embeddings": [["a"] * DIMENSION] * 2}), "non-numeric"),
        (httpx2.Response(200, json={"oops": True}), "no 'embeddings' list"),
        (httpx2.Response(200, content=b"<html>"), "invalid JSON"),
        (httpx2.Response(200, json=[1, 2]), "unexpected JSON"),
    ],
)
async def test_invalid_responses_are_rejected(response: httpx2.Response, message: str) -> None:
    backend = FakeOllama()
    backend.script = [response]

    with pytest.raises(EmbeddingResponseError, match=message):
        await make_provider(backend).embed_documents(["a", "b"])


# -- errors and retries ----------------------------------------------------------------


async def test_missing_model_explains_how_to_pull_and_is_not_retried() -> None:
    backend = FakeOllama()
    backend.script = [error(404, 'model "nomic-embed-text:v1.5" not found, try pulling it first')]

    with pytest.raises(EmbeddingProviderUnavailableError, match="ollama pull nomic-embed-text"):
        await make_provider(backend).embed_query("q")
    assert len(backend.requests) == 1


async def test_client_errors_are_not_retried() -> None:
    backend = FakeOllama()
    backend.script = [error(400, "input length exceeds the context length")]

    with pytest.raises(EmbeddingResponseError, match="exceeds the context length"):
        await make_provider(backend).embed_documents(["very long"])
    assert len(backend.requests) == 1


@pytest.mark.parametrize(
    "transient",
    [
        lambda: error(503, "server busy, please try again"),
        lambda: error(500, "llama runner process has terminated"),
        lambda: httpx2.ConnectError("connection refused"),
        lambda: httpx2.ReadTimeout("timed out"),
    ],
)
async def test_transient_failures_are_retried(transient: Callable[[], Scripted]) -> None:
    backend = FakeOllama()
    backend.script = [transient(), transient()]

    vectors = await make_provider(backend, max_retries=2).embed_documents(["a"])

    assert len(vectors) == 1
    assert len(backend.requests) == 3


async def test_persistent_failures_raise_unavailable_after_retries() -> None:
    backend = FakeOllama()
    backend.script = [error(503, "server busy")] * 3

    with pytest.raises(EmbeddingProviderUnavailableError, match="HTTP 503: server busy"):
        await make_provider(backend, max_retries=2).embed_documents(["a"])
    assert len(backend.requests) == 3


async def test_unreachable_server_is_unavailable() -> None:
    backend = FakeOllama()
    backend.script = [httpx2.ConnectError("refused")] * 2

    with pytest.raises(EmbeddingProviderUnavailableError, match="cannot reach Ollama"):
        await make_provider(backend, max_retries=1).embed_query("q")


async def test_timeout_is_unavailable() -> None:
    backend = FakeOllama()
    backend.script = [httpx2.ReadTimeout("slow")]

    with pytest.raises(EmbeddingProviderUnavailableError, match="timed out"):
        await make_provider(backend, max_retries=0).embed_query("q")


# -- readiness -------------------------------------------------------------------------


async def test_readiness_ok_when_model_present_with_matching_dimension() -> None:
    backend = FakeOllama()

    assert await make_provider(backend).readiness() == "ok"
    assert backend.requests == [("/api/show", {"model": "nomic-embed-text:v1.5"})]


async def test_readiness_reports_missing_model() -> None:
    backend = FakeOllama()
    backend.script = [error(404, "model not found")]

    assert await make_provider(backend).readiness() == "missing"


async def test_readiness_reports_dimension_mismatch() -> None:
    backend = FakeOllama(embedding_length=1024)

    assert await make_provider(backend).readiness() == "misconfigured"


async def test_readiness_ok_when_model_does_not_report_its_dimension() -> None:
    backend = FakeOllama()
    backend.script = [httpx2.Response(200, json={"model_info": {}})]

    assert await make_provider(backend).readiness() == "ok"


async def test_readiness_raises_on_server_error() -> None:
    # run_checks turns any exception into "unavailable".
    backend = FakeOllama()
    backend.script = [error(500, "boom")]

    with pytest.raises(httpx2.HTTPStatusError):
        await make_provider(backend).readiness()
