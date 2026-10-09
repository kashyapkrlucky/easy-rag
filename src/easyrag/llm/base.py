"""LLM provider port.

Every implementation must honour this contract (enforced by the shared test suite in
``tests/contracts/llm_provider.py``):

- ``generate`` accepts a non-empty sequence of chat messages (system/user/assistant);
- it returns a ``Generation`` whose ``finish_reason`` is "length" when the output was cut
  off by ``max_output_tokens``, and "stop" when the model finished on its own;
- with ``temperature=0`` and a fixed ``seed`` the same input yields the same text;
- ``context_window`` is the number of tokens the model actually processes, and it must be
  applied explicitly: some backends silently truncate prompts beyond their default.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "other"]

_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})


class LLMError(Exception):
    """Base class for generation failures."""


class LLMProviderUnavailableError(LLMError):
    """The backend cannot be reached, timed out, or the model is not available. Retryable."""


class LLMResponseError(LLMError):
    """The backend rejected the request or answered with output that violates the contract."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    # Deterministic by default (ARCHITECTURE D14).
    temperature: float = 0.0
    seed: int | None = 42
    # Upper bound on generated tokens; None leaves it to the backend.
    max_output_tokens: int | None = None
    stop: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and 0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if any(not stop for stop in self.stop):
            raise ValueError("stop sequences must not be empty")


# Immutable, so safe to share as a default argument.
DEFAULT_OPTIONS = GenerationOptions()


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    # Model that actually produced the text, as reported by the backend.
    model: str
    finish_reason: FinishReason
    # Reported by the backend when available (used for observability, never for logic).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_seconds: float = 0.0


class LLMProvider(Protocol):
    @property
    def model_id(self) -> str:
        """Identifier of the configured model, e.g. ``"ollama/qwen3:4b"``."""
        ...

    @property
    def context_window(self) -> int:
        """Tokens the model processes per request (prompt + output)."""
        ...

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerationOptions = DEFAULT_OPTIONS
    ) -> Generation: ...


def validate_messages(messages: Sequence[ChatMessage]) -> None:
    """Shared input checks for adapters. Raises ``ValueError``."""
    if not messages:
        raise ValueError("at least one message is required")
    for position, message in enumerate(messages):
        if message.role not in _ROLES:
            raise ValueError(f"message {position} has an invalid role: {message.role!r}")
        if not isinstance(message.content, str):
            raise ValueError(f"message {position} content must be a string")
    if messages[-1].role != "user":
        raise ValueError("the last message must come from the user")
