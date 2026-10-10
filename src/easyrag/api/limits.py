"""A non-blocking concurrency limit for expensive endpoints.

CPU generation takes seconds to minutes and Ollama answers one request at a time by
default, so an unbounded backlog only produces timeouts. Requests over the limit fail fast
with 503 ``busy`` and Retry-After instead of queueing (ARCHITECTURE D39).
"""

from collections.abc import Iterator
from contextlib import contextmanager


class ServiceBusyError(Exception):
    """Too many requests of this kind are in progress."""


class ConcurrencyLimit:
    def __init__(self, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("maximum must be at least 1")
        self.maximum = maximum
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold one slot for the duration of the block, or raise ``ServiceBusyError``.

        Single event loop: the check and the increment can't interleave with another task.
        """
        if self._active >= self.maximum:
            raise ServiceBusyError(f"{self._active} requests are already being answered")
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
