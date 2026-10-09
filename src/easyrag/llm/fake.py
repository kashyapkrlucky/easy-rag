"""Deterministic, scriptable LLM provider for tests and offline development.

Replies come from, in order of precedence:
1. a queue of scripted items (a reply string, or an exception to raise);
2. a ``responder`` function of the messages;
3. a fixed default reply.

Every call is recorded, so tests can assert what was sent — or that nothing was.
"""

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from easyrag.llm.base import (
    DEFAULT_OPTIONS,
    ChatMessage,
    FinishReason,
    Generation,
    GenerationOptions,
    validate_messages,
)
from easyrag.tokens import estimate_tokens

Responder = Callable[[Sequence[ChatMessage]], str]

DEFAULT_REPLY = "This is a fake answer."


@dataclass(frozen=True, slots=True)
class RecordedCall:
    messages: tuple[ChatMessage, ...]
    options: GenerationOptions


class FakeLLMProvider:
    def __init__(
        self,
        script: Iterable[str | Exception] = (),
        *,
        responder: Responder | None = None,
        model_id: str = "fake/scripted",
        context_window: int = 8192,
    ) -> None:
        if context_window < 1:
            raise ValueError("context_window must be positive")
        self.script: list[str | Exception] = list(script)
        self._responder = responder
        self._model_id = model_id
        self._context_window = context_window
        self.calls: list[RecordedCall] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def context_window(self) -> int:
        return self._context_window

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerationOptions = DEFAULT_OPTIONS
    ) -> Generation:
        validate_messages(messages)
        started = time.perf_counter()
        self.calls.append(RecordedCall(tuple(messages), options))

        text = self._next_reply(messages)
        for stop in options.stop:
            if stop in text:
                text = text[: text.index(stop)]
        finish_reason: FinishReason = "stop"
        limit = options.max_output_tokens
        if limit is not None and estimate_tokens(text) > limit:
            text = _truncate(text, limit)
            finish_reason = "length"

        return Generation(
            text=text,
            model=self._model_id,
            finish_reason=finish_reason,
            prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
            completion_tokens=estimate_tokens(text),
            duration_seconds=time.perf_counter() - started,
        )

    def _next_reply(self, messages: Sequence[ChatMessage]) -> str:
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self._responder is not None:
            return self._responder(messages)
        return DEFAULT_REPLY


def _truncate(text: str, max_tokens: int) -> str:
    """Longest prefix of ``text`` within ``max_tokens`` estimated tokens."""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]
