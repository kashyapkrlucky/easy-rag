"""HTTP transport that times every request to the model backend (Ollama).

Wrapping the shared client's transport measures each attempt, retries included, by
endpoint and outcome, without metrics code in the Ollama adapters. Ollama replies to
non-streaming requests only when it has finished, so time-to-headers is the full call.
"""

import time

import httpx2

from easyrag.observability.metrics import Metrics

# Endpoints EasyRAG calls; anything else is grouped so the label set stays bounded.
_OPERATIONS = frozenset({"/api/embed", "/api/chat", "/api/show", "/api/tags", "/api/version"})


def operation_of(request: httpx2.Request) -> str:
    path = request.url.path
    return path if path in _OPERATIONS else "other"


class InstrumentedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, inner: httpx2.AsyncBaseTransport, metrics: Metrics) -> None:
        self._inner = inner
        self._metrics = metrics

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        operation = operation_of(request)
        started = time.perf_counter()
        outcome = "error"
        try:
            response = await self._inner.handle_async_request(request)
            outcome = str(response.status_code)
            return response
        except httpx2.TimeoutException:
            outcome = "timeout"
            raise
        except httpx2.ConnectError:
            outcome = "connect_error"
            raise
        finally:
            self._metrics.record_provider_request(operation, outcome, time.perf_counter() - started)

    async def aclose(self) -> None:
        await self._inner.aclose()
