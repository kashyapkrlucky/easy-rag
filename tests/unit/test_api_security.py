import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send

from easyrag.api.middleware import BodySizeLimitMiddleware
from tests.support import SettingsFactory
from tests.unit.test_api_chat import StubRag, client_with, make_answer

KEY = "k" * 24
CHAT = {"question": "How do I deploy?"}


@pytest.fixture
def secured(make_settings: SettingsFactory) -> Iterator[TestClient]:
    with client_with(make_settings(api_key=KEY), StubRag(make_answer())) as client:
        yield client


def error_code(response: Any) -> str:
    return str(response.json()["error"]["code"])


# -- API key ---------------------------------------------------------------------------


def test_without_a_configured_key_the_api_is_open(make_settings: SettingsFactory) -> None:
    with client_with(make_settings(), StubRag(make_answer())) as client:
        assert client.post("/api/v1/chat", json=CHAT).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-key-wrong-key-wrong"},
        {"Authorization": f"Basic {KEY}"},
        {"Authorization": "Bearer "},
        {"X-API-Key": KEY[:-1]},
        {"X-API-Key": KEY + "x"},
    ],
)
def test_requests_without_the_key_are_rejected(
    secured: TestClient, headers: dict[str, str]
) -> None:
    response = secured.post("/api/v1/chat", json=CHAT, headers=headers)

    assert response.status_code == 401
    assert error_code(response) == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["x-request-id"]  # still correlated and logged
    assert KEY not in response.text


@pytest.mark.parametrize(
    "headers",
    [{"Authorization": f"Bearer {KEY}"}, {"authorization": f"bearer {KEY}"}, {"X-API-Key": KEY}],
)
def test_requests_with_the_key_are_served(secured: TestClient, headers: dict[str, str]) -> None:
    assert secured.post("/api/v1/chat", json=CHAT, headers=headers).status_code == 200


def test_rejected_before_the_body_is_validated(secured: TestClient) -> None:
    # An unauthenticated client learns nothing about the endpoint, not even 422 vs 200.
    response = secured.post("/api/v1/chat", content=b"{not json")

    assert response.status_code == 401


@pytest.mark.parametrize(
    "path", ["/api/v1/documents", "/api/v1/index/status", "/metrics", "/no/such/route"]
)
def test_every_other_path_is_protected(secured: TestClient, path: str) -> None:
    assert secured.get(path).status_code == 401


@pytest.mark.parametrize("path", ["/health/live", "/openapi.json", "/docs", "/redoc"])
def test_probes_and_api_schema_are_public(secured: TestClient, path: str) -> None:
    assert secured.get(path).status_code == 200


def test_openapi_documents_the_security_schemes(secured: TestClient) -> None:
    schema = secured.get("/openapi.json").json()

    assert set(schema["components"]["securitySchemes"]) == {"HTTPBearer", "APIKeyHeader"}
    assert schema["paths"]["/api/v1/chat"]["post"]["security"]
    assert "security" not in schema["paths"]["/health/live"]["get"]


def test_rejections_are_counted(secured: TestClient) -> None:
    secured.post("/api/v1/chat", json=CHAT)

    metrics = secured.get("/metrics", headers={"X-API-Key": KEY}).text
    assert 'easyrag_api_errors_total{code="unauthorized"} 1.0' in metrics


# -- Body size limit -------------------------------------------------------------------


def test_oversized_body_is_rejected(make_settings: SettingsFactory) -> None:
    with client_with(make_settings(max_request_bytes=1024), StubRag(make_answer())) as client:
        response = client.post("/api/v1/chat", json={"question": "x" * 2000})

    assert response.status_code == 413
    assert error_code(response) == "payload_too_large"


def test_body_within_the_limit_reaches_the_route_intact(make_settings: SettingsFactory) -> None:
    rag = StubRag(make_answer())
    with client_with(make_settings(max_request_bytes=1024), rag) as client:
        response = client.post("/api/v1/chat", json={"question": "é" * 400})

    assert response.status_code == 200
    assert rag.calls == [{"question": "é" * 400, "top_k": None}]


class Recorder:
    def __init__(self) -> None:
        self.app_body: bytes | None = None
        self.sent: list[Message] = []

    async def app(self, scope: Scope, receive: Receive, send: Send) -> None:
        message = await receive()
        self.app_body = message["body"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(self, message: Message) -> None:
        self.sent.append(message)


def run(chunks: list[bytes], headers: list[tuple[bytes, bytes]] | None = None) -> Recorder:
    """Drive the middleware with a chunked body (no Content-Length unless given)."""
    recorder = Recorder()
    middleware = BodySizeLimitMiddleware(recorder.app, max_bytes=10, metrics=lambda: None)
    messages: list[Message] = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]

    async def receive() -> Message:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    scope: Scope = {"type": "http", "method": "POST", "path": "/", "headers": headers or []}
    asyncio.run(middleware(scope, receive, recorder.send))
    return recorder


def test_chunked_body_over_the_limit_is_rejected_without_reaching_the_app() -> None:
    recorder = run([b"123456", b"789012"])

    assert recorder.app_body is None
    assert recorder.sent[0]["status"] == 413


def test_chunked_body_within_the_limit_is_reassembled() -> None:
    recorder = run([b"12345", b"67890"])

    assert recorder.app_body == b"1234567890"
    assert recorder.sent[0]["status"] == 200


def test_declared_oversized_length_is_rejected_before_reading() -> None:
    recorder = run([b"1"], headers=[(b"content-length", b"11")])

    assert recorder.app_body is None
    assert recorder.sent[0]["status"] == 413


# -- CORS ------------------------------------------------------------------------------


def preflight(client: TestClient, origin: str) -> Any:
    return client.options(
        "/api/v1/chat",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


def test_cors_is_disabled_by_default(make_settings: SettingsFactory) -> None:
    with client_with(make_settings(), StubRag(make_answer())) as client:
        response = client.post("/api/v1/chat", json=CHAT, headers={"Origin": "https://a.test"})

    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_configured_origins(make_settings: SettingsFactory) -> None:
    settings = make_settings(api_key=KEY, cors_allow_origins=["https://wiki.example.com"])
    with client_with(settings, StubRag(make_answer())) as client:
        allowed = preflight(client, "https://wiki.example.com")
        denied = preflight(client, "https://evil.example.com")
        # A rejected request still carries CORS headers, so the browser can show the 401.
        unauthorized = client.post(
            "/api/v1/chat", json=CHAT, headers={"Origin": "https://wiki.example.com"}
        )

    assert allowed.status_code == 200  # answered by CORS, without the key
    assert allowed.headers["access-control-allow-origin"] == "https://wiki.example.com"
    assert "access-control-allow-credentials" not in allowed.headers
    assert "access-control-allow-origin" not in denied.headers
    assert unauthorized.status_code == 401
    assert unauthorized.headers["access-control-allow-origin"] == "https://wiki.example.com"


# -- Settings --------------------------------------------------------------------------


def test_blank_api_key_means_unset(make_settings: SettingsFactory) -> None:
    assert make_settings(api_key="  ").api_key is None


def test_short_api_key_is_rejected(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValidationError, match="api_key"):
        make_settings(api_key="short")


def test_api_key_is_never_shown(make_settings: SettingsFactory) -> None:
    settings = make_settings(api_key=KEY)

    assert KEY not in repr(settings) and KEY not in str(settings.model_dump())


def test_cors_origins_are_read_comma_separated(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYRAG_CORS_ALLOW_ORIGINS", "https://a.test, http://b.test:8080")

    assert make_settings().cors_allow_origins == ["https://a.test", "http://b.test:8080"]


@pytest.mark.parametrize("origin", ["https://a.test/path", "a.test", "ftp://a.test", "https://"])
def test_invalid_cors_origins_are_rejected(make_settings: SettingsFactory, origin: str) -> None:
    with pytest.raises(ValidationError, match="invalid CORS origin"):
        make_settings(cors_allow_origins=[origin])
