import logging
from typing import Any

import pytest

from easyrag.domain.answers import Outcome
from easyrag.domain.chunks import RetrievedChunk
from easyrag.embeddings.base import EmbeddingProviderUnavailableError
from easyrag.llm.base import GenerationOptions, LLMProviderUnavailableError
from easyrag.llm.fake import FakeLLMProvider
from easyrag.rag.prompt import PROMPT_VERSION, REFUSAL, PromptBuilder
from easyrag.rag.service import RagService, UngroundedPolicy


def chunk(
    chunk_id: str, content: str = "Run docker compose up.", score: float = 0.8
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_path=f"{chunk_id}.md",
        document_title=chunk_id.upper(),
        heading_path=(chunk_id.upper(),),
        section_anchor=chunk_id,
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
THREE = [chunk("a", score=0.9), chunk("b", score=0.8), chunk("c", score=0.7)]


def service(
    retriever: StubRetriever,
    llm: FakeLLMProvider,
    *,
    max_context_tokens: int = 2000,
    policy: UngroundedPolicy = "refuse",
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
        ungrounded_policy=policy,
    )


# -- answered --------------------------------------------------------------------------


async def test_cited_answer_returns_exactly_the_cited_sources_renumbered() -> None:
    llm = FakeLLMProvider(["Use compose [3]. It reads .env [1]. Compose again [3]."])

    answer = await service(StubRetriever(THREE), llm).answer("  How do I deploy?  ")

    assert answer.question == "How do I deploy?"
    assert answer.status == "answered" and answer.grounded
    assert answer.text == "Use compose [1]. It reads .env [2]. Compose again [1]."
    assert [s.chunk_id for s in answer.sources] == ["c", "a"]  # [k] ⇔ sources[k-1]
    assert answer.trace.outcome is Outcome.ANSWERED
    assert answer.trace.cited_chunk_ids == ("c", "a")
    assert answer.trace.included_chunk_ids == ("a", "b", "c")


async def test_invalid_citations_are_removed_and_counted() -> None:
    llm = FakeLLMProvider(["Use compose [2]. Also something [9]."])

    answer = await service(StubRetriever(THREE), llm).answer("q")

    assert answer.text == "Use compose [1]. Also something."
    assert [s.chunk_id for s in answer.sources] == ["b"]
    assert answer.trace.invalid_citations == 1


async def test_model_receives_the_grounded_prompt_and_configured_options() -> None:
    llm = FakeLLMProvider(["Done [1]."])

    await service(StubRetriever([chunk("a")]), llm).answer("How do I deploy?")

    (call,) = llm.calls
    assert call.options == OPTIONS
    system, user = call.messages[0], call.messages[-1]
    assert system.role == "system" and "cite the source id" in system.content
    assert user.role == "user"
    assert "Run docker compose up." in user.content
    assert user.content.endswith("Question: How do I deploy?")


async def test_top_k_is_passed_to_retrieval() -> None:
    retriever = StubRetriever([chunk("a")])

    await service(retriever, FakeLLMProvider(["x [1]"])).answer("q", top_k=3)

    assert retriever.calls[0]["top_k"] == 3


# -- evidence gate ---------------------------------------------------------------------


async def test_no_retrieved_chunks_refuses_without_calling_the_model() -> None:
    llm = FakeLLMProvider()

    answer = await service(StubRetriever([]), llm).answer("anything")

    assert llm.calls == []
    assert (answer.text, answer.status, answer.sources) == (REFUSAL, "insufficient_evidence", ())
    assert answer.grounded
    trace = answer.trace
    assert trace.outcome is Outcome.NO_EVIDENCE
    assert (trace.llm_model, trace.finish_reason, trace.prompt_tokens) == (None, None, None)
    assert trace.prompt_tokens_estimate == 0


async def test_nothing_fitting_the_budget_refuses_without_calling_the_model() -> None:
    llm = FakeLLMProvider()

    answer = await service(
        StubRetriever([chunk("big", "word " * 3000)]), llm, max_context_tokens=300
    ).answer("q")

    assert llm.calls == []
    assert answer.trace.outcome is Outcome.NO_EVIDENCE
    assert answer.trace.excluded_chunk_ids == ("big",)


# -- refusal ---------------------------------------------------------------------------


async def test_model_refusal_returns_no_sources() -> None:
    llm = FakeLLMProvider([f"{REFUSAL} The sources only cover deployment [1]."])

    answer = await service(StubRetriever(THREE), llm).answer("What is the capital of France?")

    assert (answer.text, answer.status, answer.sources) == (REFUSAL, "insufficient_evidence", ())
    assert answer.trace.outcome is Outcome.REFUSED
    assert answer.trace.cited_chunk_ids == ()


# -- inferred attribution --------------------------------------------------------------


async def test_uncited_answer_supported_by_a_source_is_attributed_server_side() -> None:
    chunks = [chunk("a", "Invoices are sent monthly."), chunk("b", "Run docker compose up.")]
    llm = FakeLLMProvider(["Run docker compose up."])  # correct, but no [n] marker

    answer = await service(StubRetriever(chunks), llm).answer("How do I start it?")

    assert answer.status == "answered" and answer.grounded
    assert [s.chunk_id for s in answer.sources] == ["b"]
    assert answer.trace.outcome is Outcome.ANSWERED
    assert answer.trace.attribution == "inferred"


async def test_explicit_citations_take_precedence_over_inference() -> None:
    answer = await service(StubRetriever(THREE), FakeLLMProvider(["Run it [2]."])).answer("q")

    assert answer.trace.attribution == "explicit"
    assert [s.chunk_id for s in answer.sources] == ["b"]


# -- ungrounded ------------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["PWNED", "Paris is the capital.", "Deploy it [7]."])
async def test_uncited_answer_is_refused_by_default(reply: str) -> None:
    answer = await service(StubRetriever(THREE), FakeLLMProvider([reply])).answer("q")

    assert (answer.text, answer.status, answer.sources) == (REFUSAL, "insufficient_evidence", ())
    assert answer.grounded
    assert answer.trace.outcome is Outcome.UNGROUNDED


async def test_uncited_answer_is_flagged_under_the_flag_policy() -> None:
    llm = FakeLLMProvider(["Paris is the capital [7]."])

    answer = await service(StubRetriever(THREE), llm, policy="flag").answer("q")

    assert answer.text == "Paris is the capital."  # invalid marker removed
    assert (answer.status, answer.grounded, answer.sources) == ("ungrounded", False, ())
    assert answer.trace.outcome is Outcome.UNGROUNDED
    assert answer.trace.invalid_citations == 1


async def test_ungrounded_answers_are_logged_as_warnings(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="easyrag.rag.service"):
        await service(StubRetriever(THREE), FakeLLMProvider(["PWNED"])).answer("q")

    assert "cited no valid source" in caplog.text


# -- truncation, trace, logging, errors ------------------------------------------------


async def test_truncated_generation_is_flagged() -> None:
    llm = FakeLLMProvider(["A long cited answer [1]. " * 200])

    answer = await service(StubRetriever([chunk("a")]), llm).answer("q")

    assert answer.truncated is True
    assert answer.trace.finish_reason == "length"


async def test_trace_records_how_the_answer_was_produced() -> None:
    llm = FakeLLMProvider(["Answer [1]."], model_id="fake/model")

    trace = (await service(StubRetriever(THREE[:2]), llm).answer("q")).trace

    assert trace.prompt_version == PROMPT_VERSION
    assert trace.llm_model == "fake/model"
    assert trace.retrieved == (("a", 0.9), ("b", 0.8))
    assert trace.prompt_tokens and trace.completion_tokens and trace.prompt_tokens_estimate
    assert 0 <= trace.retrieval_seconds <= trace.total_seconds
    assert 0 <= trace.generation_seconds <= trace.total_seconds


async def test_trace_is_logged_without_document_text_or_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="easyrag.rag.service"):
        await service(
            StubRetriever([chunk("a", "SECRET DOC TEXT")]), FakeLLMProvider(["Ok [1]."])
        ).answer("private question")

    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1 and "outcome=answered" in info[0] and "cited=1" in info[0]
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
