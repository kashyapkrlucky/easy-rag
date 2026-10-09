from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from easyrag.config import Settings
from easyrag.container import AppContainer, build_container
from easyrag.domain.answers import Answer, RagTrace
from easyrag.domain.chunks import RetrievedChunk
from easyrag.embeddings.base import EmbeddingProviderUnavailableError
from easyrag.llm.base import LLMProviderUnavailableError, LLMResponseError
from easyrag.main import create_app
from tests.support import SettingsFactory

SOURCE = RetrievedChunk(
    chunk_id="c" * 32,
    source_path="deployment.md",
    document_title="Deployment Guide",
    heading_path=("Deployment", "Docker Deployment"),
    section_anchor="docker-deployment",
    content="Run docker compose up.",
    start_line=10,
    end_line=12,
    score=0.7421234,
)
TRACE = RagTrace(
    prompt_version="1",
    llm_model="fake",
    retrieved=(("c" * 32, 0.74),),
    included_chunk_ids=("c" * 32,),
    excluded_chunk_ids=(),
    prompt_tokens_estimate=100,
    prompt_tokens=98,
    completion_tokens=12,
    finish_reason="stop",
    retrieval_seconds=0.1,
    generation_seconds=1.0,
    total_seconds=1.1,
)


class StubRag:
    def __init__(self, result: Answer | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def answer(self, question: str, *, top_k: int | None = None) -> Answer:
        self.calls.append({"question": question, "top_k": top_k})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def client_with(settings: Settings, rag: StubRag) -> TestClient:
    def factory(settings: Settings) -> AppContainer:
        return replace(build_container(settings), rag=rag)  # type: ignore[arg-type]

    return TestClient(
        create_app(settings, container_factory=factory), raise_server_exceptions=False
    )


def test_chat_returns_answer_and_sources(make_settings: SettingsFactory) -> None:
    answer = Answer("How do I deploy?", "Use docker compose up.", (SOURCE,), False, TRACE)
    rag = StubRag(answer)

    with client_with(make_settings(), rag) as client:
        response = client.post("/api/v1/chat", json={"question": " How do I deploy? ", "top_k": 4})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Use docker compose up.",
        "truncated": False,
        "sources": [
            {
                "document": "deployment.md",
                "title": "Deployment Guide",
                "section": "Docker Deployment",
                "heading_path": ["Deployment", "Docker Deployment"],
                "anchor": "docker-deployment",
                "lines": [10, 12],
                "score": 0.742123,
                "chunk_id": "c" * 32,
            }
        ],
    }
    assert rag.calls == [{"question": "How do I deploy?", "top_k": 4}]


def test_truncated_answers_are_flagged(make_settings: SettingsFactory) -> None:
    rag = StubRag(Answer("q", "partial", (), True, TRACE))

    with client_with(make_settings(), rag) as client:
        assert client.post("/api/v1/chat", json={"question": "q"}).json()["truncated"] is True


@pytest.mark.parametrize(
    "payload", [{}, {"question": ""}, {"question": "x" * 2001}, {"question": "q", "top_k": 0}]
)
def test_invalid_requests_are_rejected(
    make_settings: SettingsFactory, payload: dict[str, Any]
) -> None:
    rag = StubRag(Answer("q", "a", (), False, TRACE))

    with client_with(make_settings(), rag) as client:
        response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert rag.calls == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMProviderUnavailableError("cannot reach Ollama"), 503, "llm_unavailable"),
        (LLMResponseError("bad output"), 502, "llm_failed"),
        (EmbeddingProviderUnavailableError("down"), 503, "embedding_unavailable"),
    ],
)
def test_errors_map_to_http_errors(
    make_settings: SettingsFactory, error: Exception, status: int, code: str
) -> None:
    with client_with(make_settings(), StubRag(error)) as client:
        response = client.post("/api/v1/chat", json={"question": "q"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
