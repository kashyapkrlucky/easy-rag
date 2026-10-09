"""Contract tests for ``LLMProvider`` implementations.

Usage — subclass and provide a ``provider`` fixture:

    class TestMyProviderContract(LLMProviderContract):
        @pytest.fixture
        def provider(self) -> LLMProvider:
            return MyProvider(...)

Prompts are chosen so that a real model and a suitably configured fake both satisfy
them (e.g. "count to 200" reliably exceeds a tiny output limit).
"""

import pytest

from easyrag.llm.base import ChatMessage, GenerationOptions, LLMProvider

GREETING = [
    ChatMessage("system", "You are a terse assistant."),
    ChatMessage("user", "Reply with exactly one word: hello"),
]
LONG_REQUEST = [ChatMessage("user", "Count from 1 to 200, separated by spaces.")]


class LLMProviderContract:
    def test_identity(self, provider: LLMProvider) -> None:
        assert isinstance(provider.model_id, str) and provider.model_id
        assert isinstance(provider.context_window, int) and provider.context_window > 0

    async def test_generates_a_well_formed_result(self, provider: LLMProvider) -> None:
        generation = await provider.generate(GREETING)

        assert isinstance(generation.text, str) and generation.text.strip()
        assert generation.model
        assert generation.finish_reason in {"stop", "length", "other"}
        assert generation.prompt_tokens is None or generation.prompt_tokens > 0
        assert generation.completion_tokens is None or generation.completion_tokens >= 0
        assert generation.duration_seconds >= 0

    async def test_generation_is_deterministic_with_default_options(
        self, provider: LLMProvider
    ) -> None:
        first = await provider.generate(GREETING)
        second = await provider.generate(GREETING)

        assert first.text == second.text

    async def test_output_limit_is_reported_as_length(self, provider: LLMProvider) -> None:
        generation = await provider.generate(LONG_REQUEST, GenerationOptions(max_output_tokens=8))

        assert generation.finish_reason == "length"
        assert generation.completion_tokens is None or generation.completion_tokens <= 8

    async def test_unicode_is_supported(self, provider: LLMProvider) -> None:
        generation = await provider.generate(
            [ChatMessage("user", "Répondez en un mot : « bonjour » — 日本語でも構いません。")]
        )

        assert isinstance(generation.text, str)

    @pytest.mark.parametrize(
        "messages",
        [[], [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")]],
        ids=["empty", "ends-with-assistant"],
    )
    async def test_invalid_conversations_are_rejected(
        self, provider: LLMProvider, messages: list[ChatMessage]
    ) -> None:
        with pytest.raises(ValueError):
            await provider.generate(messages)
