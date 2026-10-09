import logging
from typing import Any

import pytest

from easyrag.domain.chunks import RetrievedChunk
from easyrag.embeddings.base import EmbeddingProviderUnavailableError
from easyrag.llm.base import GenerationOptions, LLMProviderUnavailableError
from easyrag.llm.fake import FakeLLMProvider
from easyrag.rag.prompt import PROMPT_VERSION, PromptBuilder
from easyrag.rag.service import RagService


def chunk(
    chunk_id: str, content: str = "Run docker compose up.", score: float = 0.8
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_path="deployment.md",
        document_title="Deployment",
        heading_path=("Deployment",),
        section_anchor="deployment",
        content=content,
        start_line=1,
        end_line=1,
        score=score,
    )


class StubRetriever:
    def __init__(self, result: list[RetrievedChunk] | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "top_k": top_k})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


OPTIONS = GenerationOptions(temperature=0.0, seed=7, max_output_tokens=200)


def service(
    retriever: StubRetriever, llm: FakeLLMProvider, *, max_context_tokens: int = 2000
) -> RagService:
    return RagService(
        retriever=retriever,
        llm=llm,
        prompt_builder=PromptBuilder(
            context_window=llm.context_window,
            max_output_tokens=200,
            max_context_tokens=max_context_tokens,
        ),
        generation_options=OPTIONS,
    )


async def test_answers_from_retrieved_chunks() -> None:
    retriever = StubRetriever([chunk("a", score=0.9), chunk("b", score=0.7)])
    llm = FakeLLMProvider(["  Use docker compose up.  "])

    answer = await service(retriever, llm).answer("  How do I deploy?  ")

    assert answer.question == "How do I deploy?"
    assert answer.text == "Use docker compose up."
    assert [s.chunk_id for s in answer.sources] == ["a", "b"]
    assert answer.truncated is False
    assert retriever.calls == [{"query": "How do I deploy?", "top_k": None}]


async def test_model_receives_the_grounded_prompt_and_configured_options() -> None:
    llm = FakeLLMProvider()

    await service(StubRetriever([chunk("a")]), llm).answer("How do I deploy?")

    (call,) = llm.calls
    assert call.options == OPTIONS
    system, user = call.messages
    assert system.role == "system" and "ONLY the information" in system.content
    assert user.role == "user"
    assert "Run docker compose up." in user.content
    assert user.content.endswith("Question: How do I deploy?")


async def test_top_k_is_passed_to_retrieval() -> None:
    retriever = StubRetriever([])

    await service(retriever, FakeLLMProvider()).answer("q", top_k=3)

    assert retriever.calls[0]["top_k"] == 3


async def test_sources_are_only_the_chunks_the_model_saw() -> None:
    retriever = StubRetriever([chunk("big", "word " * 3000), chunk("small", "tiny")])
    llm = FakeLLMProvider()

    answer = await service(retriever, llm, max_context_tokens=500).answer("q")

    assert [s.chunk_id for s in answer.sources] == ["small"]
    assert answer.trace.included_chunk_ids == ("small",)
    assert answer.trace.excluded_chunk_ids == ("big",)
    assert "word word" not in llm.calls[0].messages[1].content


async def test_truncated_generation_is_flagged() -> None:
    llm = FakeLLMProvider(["a long answer " * 200])

    answer = await service(StubRetriever([chunk("a")]), llm).answer("q")

    assert answer.truncated is True
    assert answer.trace.finish_reason == "length"


async def test_trace_records_how_the_answer_was_produced() -> None:
    retriever = StubRetriever([chunk("a", score=0.9), chunk("b", score=0.5)])
    llm = FakeLLMProvider(["Answer."], model_id="fake/model")

    trace = (await service(retriever, llm).answer("q")).trace

    assert trace.prompt_version == PROMPT_VERSION
    assert trace.llm_model == "fake/model"
    assert trace.retrieved == (("a", 0.9), ("b", 0.5))
    assert trace.included_chunk_ids == ("a", "b")
    assert trace.prompt_tokens and trace.completion_tokens
    assert trace.prompt_tokens_estimate > 0
    assert 0 <= trace.retrieval_seconds <= trace.total_seconds
    assert 0 <= trace.generation_seconds <= trace.total_seconds


async def test_trace_is_logged_without_document_text_or_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="easyrag.rag.service"):
        await service(StubRetriever([chunk("a", "SECRET DOC TEXT")]), FakeLLMProvider()).answer(
            "private question"
        )

    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1 and "included=1" in info[0]
    assert "SECRET DOC TEXT" not in caplog.text
    assert "private question" not in "\n".join(info)


async def test_blank_question_is_rejected_without_retrieval() -> None:
    retriever = StubRetriever([])

    with pytest.raises(ValueError):
        await service(retriever, FakeLLMProvider()).answer("   ")
    assert retriever.calls == []


async def test_retrieval_errors_propagate_without_calling_the_model() -> None:
    llm = FakeLLMProvider()

    with pytest.raises(EmbeddingProviderUnavailableError):
        await service(StubRetriever(EmbeddingProviderUnavailableError("down")), llm).answer("q")
    assert llm.calls == []


async def test_model_errors_propagate() -> None:
    llm = FakeLLMProvider([LLMProviderUnavailableError("Ollama is down")])

    with pytest.raises(LLMProviderUnavailableError):
        await service(StubRetriever([chunk("a")]), llm).answer("q")
