import json
from typing import Any

import httpx2
import pytest

from easyrag.llm.base import (
    ChatMessage,
    GenerationOptions,
    LLMProvider,
    LLMProviderUnavailableError,
    LLMResponseError,
)
from easyrag.llm.ollama import OllamaLLMProvider
from easyrag.tokens import estimate_tokens
from tests.contracts.llm_provider import LLMProviderContract

MODEL = "qwen2.5:1.5b-instruct-q4_K_M"
QUESTION = [ChatMessage("system", "Be terse."), ChatMessage("user", "Say hello")]

Scripted = httpx2.Response | Exception


class FakeOllamaChat:
    """In-process stand-in for Ollama's /api/chat and /api/show (via MockTransport).

    Replies deterministically, honouring ``num_predict`` like Ollama does (truncating
    and reporting ``done_reason: "length"``). Scripted responses take precedence.
    """

    def __init__(self, context_length: int | None = 262144) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.script: list[Scripted] = []
        self.context_length = context_length

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if request.url.path == "/api/show":
            info = (
                {} if self.context_length is None else {"qwen3.context_length": self.context_length}
            )
            return httpx2.Response(200, json={"model_info": info})
        return self._chat(body)

    def _chat(self, body: dict[str, Any]) -> httpx2.Response:
        question = body["messages"][-1]["content"]
        text = " ".join(str(i) for i in range(1, 201)) if "Count" in question else "hello"
        done_reason = "stop"
        limit = body["options"].get("num_predict")
        if limit is not None and estimate_tokens(text) > limit:
            text, done_reason = text[: limit * 4], "length"
        return httpx2.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": text},
                "done": True,
                "done_reason": done_reason,
                "prompt_eval_count": sum(estimate_tokens(m["content"]) for m in body["messages"]),
                "eval_count": estimate_tokens(text),
                "total_duration": 1_500_000_000,
            },
        )

    def chat_requests(self) -> list[dict[str, Any]]:
        return [body for path, body in self.requests if path == "/api/chat"]


def make_provider(backend: FakeOllamaChat, **overrides: Any) -> OllamaLLMProvider:
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(backend.handler), base_url="http://ollama.test"
    )
    options: dict[str, Any] = {
        "model": MODEL,
        "context_window": 8192,
        "retry_backoff_seconds": 0,
        **overrides,
    }
    return OllamaLLMProvider(client, **options)


def error(status: int, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": message})


class TestOllamaLLMProviderContract(LLMProviderContract):
    @pytest.fixture
    def provider(self) -> LLMProvider:
        return make_provider(FakeOllamaChat())


# -- request ---------------------------------------------------------------------------


async def test_request_always_sets_context_window_and_deterministic_options() -> None:
    backend = FakeOllamaChat()

    await make_provider(backend, context_window=4096).generate(QUESTION)

    assert backend.chat_requests() == [
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Say hello"},
            ],
            "stream": False,
            "options": {"num_ctx": 4096, "temperature": 0.0, "seed": 42},
        }
    ]


async def test_optional_generation_options_are_mapped() -> None:
    backend = FakeOllamaChat()
    options = GenerationOptions(temperature=0.7, seed=None, max_output_tokens=256, stop=("\n\n",))

    await make_provider(backend).generate(QUESTION, options)

    assert backend.chat_requests()[0]["options"] == {
        "num_ctx": 8192,
        "temperature": 0.7,
        "num_predict": 256,
        "stop": ["\n\n"],
    }


async def test_prompt_exceeding_the_context_window_is_refused_before_sending() -> None:
    backend = FakeOllamaChat()
    long_prompt = [ChatMessage("user", "x" * 4 * 600)]  # ~600 estimated tokens

    with pytest.raises(LLMResponseError, match="exceeds the context window of 512"):
        await make_provider(backend, context_window=512).generate(long_prompt)
    assert backend.requests == []


def test_model_id_and_context_window() -> None:
    provider = make_provider(FakeOllamaChat(), context_window=4096)

    assert (provider.model_id, provider.context_window) == (f"ollama/{MODEL}", 4096)


# -- response mapping ------------------------------------------------------------------


async def test_response_fields_are_mapped() -> None:
    generation = await make_provider(FakeOllamaChat()).generate(QUESTION)

    assert generation.text == "hello"
    assert generation.model == MODEL
    assert generation.finish_reason == "stop"
    assert generation.prompt_tokens == estimate_tokens("Be terse.") + estimate_tokens("Say hello")
    assert generation.completion_tokens == estimate_tokens("hello")
    assert generation.duration_seconds == 1.5  # server-side total_duration


@pytest.mark.parametrize(
    ("done_reason", "expected"),
    [("stop", "stop"), ("length", "length"), ("load", "other"), (None, "other")],
)
async def test_done_reason_maps_to_finish_reason(done_reason: str | None, expected: str) -> None:
    backend = FakeOllamaChat()
    backend.script = [
        httpx2.Response(
            200, json={"message": {"role": "assistant", "content": "x"}, "done_reason": done_reason}
        )
    ]

    generation = await make_provider(backend).generate(QUESTION)

    assert generation.finish_reason == expected
    assert (generation.prompt_tokens, generation.completion_tokens) == (None, None)
    assert generation.model == MODEL  # falls back to the configured model


@pytest.mark.parametrize(
    "body",
    [{"done": True}, {"message": {"role": "assistant"}}, {"message": "text"}],
)
async def test_response_without_content_is_rejected(body: dict[str, Any]) -> None:
    backend = FakeOllamaChat()
    backend.script = [httpx2.Response(200, json=body)]

    with pytest.raises(LLMResponseError, match="no message content"):
        await make_provider(backend).generate(QUESTION)


# -- errors ------------------------------------------------------------------------------


async def test_missing_model_explains_how_to_pull_and_is_not_retried() -> None:
    backend = FakeOllamaChat()
    backend.script = [error(404, f'model "{MODEL}" not found, try pulling it first')]

    with pytest.raises(LLMProviderUnavailableError, match=f"ollama pull {MODEL}"):
        await make_provider(backend).generate(QUESTION)
    assert len(backend.requests) == 1


async def test_rejected_request_is_a_response_error() -> None:
    backend = FakeOllamaChat()
    backend.script = [error(400, "invalid options")]

    with pytest.raises(LLMResponseError, match="invalid options"):
        await make_provider(backend).generate(QUESTION)


async def test_transient_failures_are_retried_then_succeed() -> None:
    backend = FakeOllamaChat()
    backend.script = [error(503, "server busy"), httpx2.ReadTimeout("slow")]

    generation = await make_provider(backend, max_retries=2).generate(QUESTION)

    assert generation.text == "hello"
    assert len(backend.chat_requests()) == 3


async def test_persistent_unavailability_raises_after_retries() -> None:
    backend = FakeOllamaChat()
    backend.script = [httpx2.ConnectError("refused")] * 3

    with pytest.raises(LLMProviderUnavailableError, match="cannot reach Ollama"):
        await make_provider(backend, max_retries=2).generate(QUESTION)
    assert len(backend.requests) == 3


async def test_invalid_conversation_is_rejected_without_a_request() -> None:
    backend = FakeOllamaChat()

    with pytest.raises(ValueError):
        await make_provider(backend).generate([ChatMessage("assistant", "hi")])
    assert backend.requests == []


# -- readiness ---------------------------------------------------------------------------


async def test_readiness_ok_when_model_supports_the_context_window() -> None:
    assert await make_provider(FakeOllamaChat(context_length=32768)).readiness() == "ok"


async def test_readiness_reports_too_small_native_context() -> None:
    provider = make_provider(FakeOllamaChat(context_length=4096), context_window=8192)

    assert await provider.readiness() == "misconfigured"


async def test_readiness_ok_when_model_does_not_report_context_length() -> None:
    assert await make_provider(FakeOllamaChat(context_length=None)).readiness() == "ok"


async def test_readiness_reports_missing_model() -> None:
    backend = FakeOllamaChat()
    backend.script = [error(404, "not found")]

    assert await make_provider(backend).readiness() == "missing"
