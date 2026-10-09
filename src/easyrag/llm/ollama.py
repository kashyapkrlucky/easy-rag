"""Ollama chat adapter (``POST /api/chat``, non-streaming).

Ollama-specific details stay in this module: request options (``num_ctx``,
``num_predict``), response fields (``done_reason``, ``prompt_eval_count``…) and model
metadata. The rest of the application sees only ``LLMProvider`` and the LLM errors.
"""

import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx2

from easyrag.health import CheckStatus
from easyrag.llm.base import (
    DEFAULT_OPTIONS,
    ChatMessage,
    FinishReason,
    Generation,
    GenerationOptions,
    LLMProviderUnavailableError,
    LLMResponseError,
    validate_messages,
)
from easyrag.ollama import (
    OllamaModelNotFoundError,
    OllamaRequestError,
    OllamaUnavailableError,
    model_info_value,
    post_json,
    show_model,
)
from easyrag.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_FINISH_REASONS: dict[str, FinishReason] = {"stop": "stop", "length": "length"}


class OllamaLLMProvider:
    def __init__(
        self,
        client: httpx2.AsyncClient,
        *,
        model: str,
        context_window: int,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if context_window < 1:
            raise ValueError("context_window must be positive")
        self._client = client
        self._model = model
        self._context_window = context_window
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    @property
    def model_id(self) -> str:
        return f"ollama/{self._model}"

    @property
    def context_window(self) -> int:
        return self._context_window

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerationOptions = DEFAULT_OPTIONS
    ) -> Generation:
        validate_messages(messages)
        prompt_estimate = sum(estimate_tokens(message.content) for message in messages)
        if prompt_estimate > self._context_window:
            # Ollama would silently drop the beginning of the prompt (the instructions).
            raise LLMResponseError(
                f"prompt of ~{prompt_estimate} tokens exceeds the context window of "
                f"{self._context_window} tokens"
            )

        started = time.perf_counter()
        try:
            data = await post_json(
                self._client,
                "/api/chat",
                self._payload(messages, options),
                max_retries=self._max_retries,
                retry_backoff_seconds=self._retry_backoff_seconds,
            )
        except OllamaModelNotFoundError as exc:
            raise LLMProviderUnavailableError(
                f"chat {exc}; pull it with: ollama pull {self._model}"
            ) from exc
        except OllamaUnavailableError as exc:
            raise LLMProviderUnavailableError(str(exc)) from exc
        except OllamaRequestError as exc:
            raise LLMResponseError(str(exc)) from exc

        return self._generation(data, time.perf_counter() - started)

    async def readiness(self) -> CheckStatus:
        """Is the model available, and can it handle the configured context window?"""
        metadata = await show_model(self._client, self._model)
        if metadata is None:
            return "missing"
        native = model_info_value(metadata, ".context_length")
        if native is not None and native < self._context_window:
            logger.error(
                "Chat model %s supports %d tokens of context but EASYRAG_LLM_CONTEXT_WINDOW=%d",
                self._model,
                native,
                self._context_window,
            )
            return "misconfigured"
        return "ok"

    def _payload(
        self, messages: Sequence[ChatMessage], options: GenerationOptions
    ) -> dict[str, Any]:
        # num_ctx is always explicit (ARCHITECTURE D14): Ollama's default is smaller and it
        # truncates longer prompts silently.
        model_options: dict[str, Any] = {
            "num_ctx": self._context_window,
            "temperature": options.temperature,
        }
        if options.seed is not None:
            model_options["seed"] = options.seed
        if options.max_output_tokens is not None:
            model_options["num_predict"] = options.max_output_tokens
        if options.stop:
            model_options["stop"] = list(options.stop)
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": model_options,
        }

    def _generation(self, data: dict[str, Any], elapsed_seconds: float) -> Generation:
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMResponseError("Ollama response has no message content")

        total_duration = data.get("total_duration")  # nanoseconds, server-side
        return Generation(
            text=content,
            model=str(data.get("model") or self._model),
            finish_reason=_FINISH_REASONS.get(str(data.get("done_reason")), "other"),
            prompt_tokens=_count(data.get("prompt_eval_count")),
            completion_tokens=_count(data.get("eval_count")),
            duration_seconds=(
                total_duration / 1e9 if isinstance(total_duration, int) else elapsed_seconds
            ),
        )


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
