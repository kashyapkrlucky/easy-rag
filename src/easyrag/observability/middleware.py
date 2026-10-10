"""Request context: a correlation ID per request, one structured log line, HTTP metrics.

Pure ASGI (not BaseHTTPMiddleware): cheap, streaming-safe, and it observes the final
status code even when an error handler produced the response.
"""

import logging
import time
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from easyrag.observability.context import accept_request_id, bound, request_id
from easyrag.observability.metrics import Metrics

logger = logging.getLogger("easyrag.http")

HEADER = b"x-request-id"
# Probes and scrapes: counted, but logged only at DEBUG so they don't flood the logs.
_QUIET_PREFIXES = ("/health/", "/metrics")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, metrics: Callable[[], Metrics | None]) -> None:
        self._app = app
        # Resolved per request: the metrics object lives in the container, which the
        # lifespan creates after the middleware stack is built.
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(HEADER)
        rid = accept_request_id(incoming.decode("latin-1") if incoming else None)
        status = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [*message.get("headers", []), (HEADER, rid.encode())]
            await send(message)

        with bound(request_id, rid):
            try:
                await self._app(scope, receive, send_with_id)
            finally:
                self._observe(scope, status, time.perf_counter() - started)

    def _observe(self, scope: Scope, status: int, seconds: float) -> None:
        method, path = scope["method"], scope["path"]
        route = getattr(scope.get("route"), "path", None) or "unmatched"
        metrics = self._metrics()
        if metrics is not None:
            metrics.http_requests.labels(method=method, route=route, status=str(status)).inc()
            metrics.http_duration.labels(method=method, route=route).observe(seconds)
        level = logging.DEBUG if path.startswith(_QUIET_PREFIXES) else logging.INFO
        logger.log(
            level,
            "%s %s %d %.0fms",
            method,
            path,
            status,
            seconds * 1000,
            extra={
                "method": method,
                "path": path,
                "route": route,
                "status": status,
                "duration_ms": round(seconds * 1000, 1),
            },
        )
