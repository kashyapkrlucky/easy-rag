from collections.abc import Sequence

import pytest

from easyrag.llm.base import (
    DEFAULT_OPTIONS,
    ChatMessage,
    GenerationOptions,
    LLMProvider,
    LLMProviderUnavailableError,
    validate_messages,
)
from easyrag.llm.fake import DEFAULT_REPLY, FakeLLMProvider
from easyrag.tokens import estimate_tokens
from tests.contracts.llm_provider import LLMProviderContract

QUESTION = [ChatMessage("user", "What is EasyRAG?")]


def counting(messages: Sequence[ChatMessage]) -> str:
    return " ".join(str(number) for number in range(1, 201))


class TestFakeLLMProviderContract(LLMProviderContract):
    @pytest.fixture
    def provider(self) -> LLMProvider:
        return FakeLLMProvider(responder=counting)


# -- options and validation ------------------------------------------------------------


def test_default_options_are_deterministic() -> None:
    assert (DEFAULT_OPTIONS.temperature, DEFAULT_OPTIONS.seed) == (0.0, 42)
    assert DEFAULT_OPTIONS.max_output_tokens is None and DEFAULT_OPTIONS.stop == ()


@pytest.mark.parametrize(
    "options",
    [
        {"temperature": -0.1},
        {"temperature": 2.5},
        {"temperature": float("nan")},
        {"max_output_tokens": 0},
        {"stop": ("",)},
    ],
)
def test_invalid_options_are_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GenerationOptions(**options)  # type: ignore[arg-type]


def test_options_are_immutable() -> None:
    with pytest.raises(AttributeError):
        DEFAULT_OPTIONS.temperature = 1.0  # type: ignore[misc]


def test_invalid_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid role"):
        validate_messages([ChatMessage("tool", "x")])  # type: ignore[arg-type]


def test_system_and_user_conversation_is_valid() -> None:
    validate_messages([ChatMessage("system", "rules"), ChatMessage("user", "question")])


# -- FakeLLMProvider behaviour ---------------------------------------------------------


async def test_scripted_replies_are_returned_in_order_then_default() -> None:
    provider = FakeLLMProvider(["first", "second"])

    texts = [(await provider.generate(QUESTION)).text for _ in range(3)]

    assert texts == ["first", "second", DEFAULT_REPLY]


async def test_scripted_exception_is_raised() -> None:
    provider = FakeLLMProvider([LLMProviderUnavailableError("Ollama is down"), "recovered"])

    with pytest.raises(LLMProviderUnavailableError):
        await provider.generate(QUESTION)
    assert (await provider.generate(QUESTION)).text == "recovered"


async def test_responder_sees_the_messages() -> None:
    provider = FakeLLMProvider(responder=lambda messages: messages[-1].content.upper())

    assert (await provider.generate(QUESTION)).text == "WHAT IS EASYRAG?"


async def test_calls_are_recorded_with_options() -> None:
    provider = FakeLLMProvider()
    options = GenerationOptions(max_output_tokens=100)

    await provider.generate(QUESTION, options)

    assert len(provider.calls) == 1
    assert provider.calls[0].messages == tuple(QUESTION)
    assert provider.calls[0].options == options


async def test_invalid_conversation_is_not_recorded() -> None:
    provider = FakeLLMProvider()

    with pytest.raises(ValueError):
        await provider.generate([])
    assert provider.calls == []


async def test_output_is_truncated_to_the_token_limit() -> None:
    provider = FakeLLMProvider(responder=counting)

    generation = await provider.generate(QUESTION, GenerationOptions(max_output_tokens=10))

    assert generation.finish_reason == "length"
    assert estimate_tokens(generation.text) <= 10
    assert counting(QUESTION).startswith(generation.text)


async def test_stop_sequences_end_the_output() -> None:
    provider = FakeLLMProvider(["Answer.\nSources: [1]"])

    generation = await provider.generate(QUESTION, GenerationOptions(stop=("\nSources",)))

    assert (generation.text, generation.finish_reason) == ("Answer.", "stop")


async def test_token_counts_are_estimated() -> None:
    provider = FakeLLMProvider(["four words of reply"])

    generation = await provider.generate(QUESTION)

    assert generation.prompt_tokens == estimate_tokens("What is EasyRAG?")
    assert generation.completion_tokens == estimate_tokens("four words of reply")
    assert generation.model == "fake/scripted"


def test_identity_is_configurable() -> None:
    provider = FakeLLMProvider(model_id="fake/x", context_window=2048)

    assert (provider.model_id, provider.context_window) == ("fake/x", 2048)
