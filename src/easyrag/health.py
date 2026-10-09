"""Readiness checks.

A readiness check is a named, argument-free coroutine reporting one dependency's status.
The composition root registers them; ``run_checks`` executes them concurrently, each
bounded by a timeout, and treats any exception as "unavailable".
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal

logger = logging.getLogger(__name__)

CheckStatus = Literal["ok", "unavailable", "outdated", "missing", "misconfigured"]
ReadinessCheck = Callable[[], Awaitable[CheckStatus]]


async def run_checks(
    checks: Mapping[str, ReadinessCheck], timeout_seconds: float
) -> dict[str, CheckStatus]:
    names = list(checks)
    results = await asyncio.gather(*(_run(name, checks[name], timeout_seconds) for name in names))
    return dict(zip(names, results, strict=True))


async def _run(name: str, check: ReadinessCheck, timeout_seconds: float) -> CheckStatus:
    try:
        async with asyncio.timeout(timeout_seconds):
            status = await check()
    except Exception as exc:
        # Log the exception type only: messages can include connection details.
        logger.warning("Readiness check %r failed (%s)", name, type(exc).__name__)
        return "unavailable"
    if status != "ok":
        logger.warning("Readiness check %r: %s", name, status)
    return status
