"""Correlation identifiers carried through async code and attached to every log line."""

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
index_run_id: ContextVar[str | None] = ContextVar("index_run_id", default=None)

_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def accept_request_id(candidate: str | None) -> str:
    """Use a client-supplied ID only if it is short and safe; otherwise generate one."""
    if candidate and _SAFE_ID.fullmatch(candidate):
        return candidate
    return new_id()


@contextmanager
def bound(variable: ContextVar[str | None], value: str) -> Iterator[str]:
    token = variable.set(value)
    try:
        yield value
    finally:
        variable.reset(token)
