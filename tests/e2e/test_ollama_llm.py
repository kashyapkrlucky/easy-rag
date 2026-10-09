import pytest

from easyrag.llm.base import ChatMessage, GenerationOptions, LLMProvider
from easyrag.llm.ollama import OllamaLLMProvider
from tests.contracts.llm_provider import LLMProviderContract


class TestRealOllamaLLMContract(LLMProviderContract):
    @pytest.fixture
    def provider(self, ollama_llm: OllamaLLMProvider) -> LLMProvider:
        return ollama_llm


async def test_model_is_ready(ollama_llm: OllamaLLMProvider) -> None:
    assert await ollama_llm.readiness() == "ok"


async def test_answers_from_supplied_context(ollama_llm: OllamaLLMProvider) -> None:
    # The fact is invented, so the model can only know it from the context.
    generation = await ollama_llm.generate(
        [
            ChatMessage("system", "Answer using only the provided context. Be brief."),
            ChatMessage(
                "user",
                "Context: The maintenance window for project Zephyr is every Tuesday at 03:40 UTC."
                "\n\nQuestion: When is the maintenance window for project Zephyr?",
            ),
        ],
        GenerationOptions(max_output_tokens=60),
    )

    assert "03:40" in generation.text, generation.text
    assert generation.finish_reason == "stop"
    assert generation.prompt_tokens and generation.completion_tokens
