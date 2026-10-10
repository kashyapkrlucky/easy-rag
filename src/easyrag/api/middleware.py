"""HTTP-level protection: API key, body size limit, and browser security headers.

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
    """Require the API key on every path except ``public_paths`` and ``public_prefixes``."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str,
        public_paths: frozenset[str],
        public_prefixes: tuple[str, ...] = (),
        metrics: MetricsGetter,
    ) -> None:
        self._app = app
        # Comparing fixed-length digests in constant time leaks neither content nor length.
        self._digest = _digest(api_key)
        self._public_paths = public_paths
        self._public_prefixes = public_prefixes
        self._metrics = metrics

    def _is_public(self, path: str) -> bool:
        return path in self._public_paths or path.startswith(self._public_prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._is_public(scope["path"]):
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


# The web UI loads only its own files: no inline script, no third parties, no framing.
UI_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
_COMMON_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
]


class SecurityHeadersMiddleware:
    """Browser hardening on every response, plus a strict CSP on the web UI's paths.

    The CSP is limited to the UI because /docs (Swagger UI) loads its assets from a CDN.
    UI responses are also revalidated on every load, so an upgrade is picked up at once.
    """

    def __init__(self, app: ASGIApp, *, ui_paths: frozenset[str], ui_prefixes: tuple[str, ...]):
        self._app = app
        self._ui_paths = ui_paths
        self._ui_prefixes = ui_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        path = scope["path"]
        extra = list(_COMMON_HEADERS)
        if path in self._ui_paths or path.startswith(self._ui_prefixes):
            extra += [
                (b"content-security-policy", UI_CONTENT_SECURITY_POLICY.encode()),
                (b"cache-control", b"no-cache"),
            ]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {bytes(key).lower() for key, _ in message.get("headers", [])}
                message["headers"] = [
                    *message.get("headers", []),
                    *((key, value) for key, value in extra if key not in present),
                ]
            await send(message)

        await self._app(scope, receive, send_with_headers)
