"""HTTP-level protection: API key authentication and a request body size limit.

Pure ASGI, so a request is rejected before routing, body parsing or validation, and every
route is covered, including ones added later (deny by default). Rejections use the API's
error shape and are counted in easyrag_api_errors_total.
"""

import hashlib
import hmac
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from easyrag.api.errors import error_response
from easyrag.observability.metrics import Metrics

MetricsGetter = Callable[[], Metrics | None]


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key == name:
            return bytes(value).decode("latin-1")
    return None


def _count(metrics: MetricsGetter, code: str) -> None:
    current = metrics()
    if current is not None:
        current.api_errors.labels(code=code).inc()


class ApiKeyMiddleware:
    """Require the API key on every path except ``public_paths``."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str,
        public_paths: frozenset[str],
        metrics: MetricsGetter,
    ) -> None:
        self._app = app
        # Comparing fixed-length digests in constant time leaks neither content nor length.
        self._digest = _digest(api_key)
        self._public_paths = public_paths
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self._public_paths:
            await self._app(scope, receive, send)
            return
        provided = self._provided_key(scope)
        if provided is not None and hmac.compare_digest(_digest(provided), self._digest):
            await self._app(scope, receive, send)
            return
        _count(self._metrics, "unauthorized")
        response = error_response(
            401,
            "unauthorized",
            "A valid API key is required (Authorization: Bearer <key> or X-API-Key)",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        await response(scope, receive, send)

    @staticmethod
    def _provided_key(scope: Scope) -> str | None:
        authorization = _header(scope, b"authorization")
        if authorization is not None:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()
        return _header(scope, b"x-api-key")


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` with 413.

    The body is buffered (up to the limit) before the app runs, so the limit also holds for
    chunked bodies without Content-Length. API requests are small, so this costs nothing.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int, metrics: MetricsGetter) -> None:
        self._app = app
        self._max_bytes = max_bytes
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max_bytes:
            await self._reject(scope, receive, send)
            return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # the client went away; nobody to answer
            body = message.get("body", b"")
            size += len(body)
            if size > self._max_bytes:
                await self._reject(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        buffered: Message | None = {"type": "http.request", "body": b"".join(chunks)}

        async def replay() -> Message:
            nonlocal buffered
            if buffered is not None:
                message, buffered = buffered, None
                return message
            return await receive()  # afterwards: wait for disconnect, as usual

        await self._app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        _count(self._metrics, "payload_too_large")
        response = error_response(
            413, "payload_too_large", f"The request body exceeds {self._max_bytes} bytes"
        )
        await response(scope, receive, send)
