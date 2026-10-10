import json
import logging
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyrag.config import Settings
from easyrag.container import AppContainer, build_container
from easyrag.domain.chunks import RetrievedChunk
from easyrag.main import create_app
from tests.support import SettingsFactory


class LoggingRetriever:
    """Logs like the real pipeline does, so tests can check request correlation."""

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        logging.getLogger("easyrag.retrieval").info("retrieved nothing")
        return []


def app(settings: Settings) -> FastAPI:
    def factory(settings: Settings) -> AppContainer:
        retriever = LoggingRetriever()
        modes = {"vector": retriever, "keyword": retriever, "hybrid": retriever}
        return replace(build_container(settings), retriever=retriever, retrievers=modes)

    return create_app(settings, container_factory=factory)


@pytest.fixture
def logs(make_settings: SettingsFactory) -> Iterator[tuple[TestClient, list[dict[str, Any]]]]:
    """A client plus the JSON log entries it produced."""
    entries: list[dict[str, Any]] = []
    application = app(make_settings(log_level="DEBUG", log_format="json"))
    root = logging.getLogger()
    configured = root.handlers[0].formatter
    assert configured is not None
    formatter: logging.Formatter = configured

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            entries.append(json.loads(formatter.format(record)))

    capture = Capture()
    capture.addFilter(root.handlers[0].filters[0])  # the ContextFilter
    root.addHandler(capture)
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client, entries
    finally:
        root.removeHandler(capture)


def request_lines(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if e["logger"] == "easyrag.http"]


def test_a_request_id_is_generated_returned_and_logged(
    logs: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, entries = logs

    response = client.post("/api/v1/retrieve", json={"query": "q"})

    rid = response.headers["x-request-id"]
    assert len(rid) == 16
    (line,) = request_lines(entries)
    assert line["request_id"] == rid
    assert (line["method"], line["route"], line["status"]) == ("POST", "/api/v1/retrieve", 200)
    assert line["level"] == "INFO" and line["duration_ms"] >= 0
    # Every line logged while handling the request carries the same ID.
    (pipeline,) = [e for e in entries if e["logger"] == "easyrag.retrieval"]
    assert pipeline["request_id"] == rid


def test_a_safe_incoming_request_id_is_honoured(
    logs: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, _ = logs

    response = client.get("/health/live", headers={"X-Request-ID": "trace-abc.123"})

    assert response.headers["x-request-id"] == "trace-abc.123"


@pytest.mark.parametrize("bad", ["has spaces", "x" * 65, 'quote"d'])
def test_unsafe_incoming_request_ids_are_replaced(
    logs: tuple[TestClient, list[dict[str, Any]]], bad: str
) -> None:
    client, _ = logs

    response = client.get("/health/live", headers={"X-Request-ID": bad})

    assert response.headers["x-request-id"] != bad
    assert len(response.headers["x-request-id"]) == 16


def test_probes_are_logged_at_debug(logs: tuple[TestClient, list[dict[str, Any]]]) -> None:
    client, entries = logs

    client.get("/health/live")

    (line,) = request_lines(entries)
    assert line["level"] == "DEBUG"


def test_metrics_expose_http_requests_by_route_template(
    logs: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, _ = logs
    client.post("/api/v1/retrieve", json={"query": "q"})
    client.get("/no/such/path/123")

    body = client.get("/metrics").text

    assert (
        'easyrag_http_requests_total{method="POST",route="/api/v1/retrieve",status="200"} 1.0'
        in body
    )
    assert 'route="unmatched",status="404"' in body
    assert "/no/such/path/123" not in body  # raw paths never become labels
    assert 'easyrag_http_request_duration_seconds_bucket{le="0.005",method="POST"' in body
    assert "easyrag_build_info{" in body


def test_error_responses_are_counted_by_code(
    logs: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, _ = logs

    client.post("/api/v1/retrieve", json={})
    client.post("/api/v1/retrieve", json={})

    body = client.get("/metrics").text
    assert 'easyrag_api_errors_total{code="validation_error"} 2.0' in body
