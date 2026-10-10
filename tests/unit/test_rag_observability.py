import json
import logging
from dataclasses import replace

import httpx2
import pytest

from easyrag.domain.answers import Answer, Outcome, RagTrace
from easyrag.llm.fake import FakeLLMProvider
from easyrag.observability.logging import JsonFormatter
from easyrag.observability.metrics import Metrics
from easyrag.observability.transport import InstrumentedTransport
from easyrag.rag.prompt import PromptBuilder
from easyrag.rag.service import AnswerObserver, RagService
from tests.unit.test_rag_service import OPTIONS, THREE, StubRetriever, chunk

TRACE = RagTrace(
    prompt_version="4",
    outcome=Outcome.ANSWERED,
    retrieved=(("a", 0.031), ("b", 0.02)),
    included_chunk_ids=("a", "b"),
    excluded_chunk_ids=(),
    cited_chunk_ids=("a",),
    invalid_citations=1,
    attribution="explicit",
    prompt_tokens_estimate=900,
    llm_model="fake",
    finish_reason="stop",
    prompt_tokens=850,
    completion_tokens=40,
    retrieval_seconds=0.2,
    generation_seconds=3.0,
    total_seconds=3.25,
)


def sample(metrics: Metrics, name: str, **labels: str) -> float | None:
    return metrics.registry.get_sample_value(name, labels)


def service(
    llm: FakeLLMProvider, on_answer: AnswerObserver | None, retriever: StubRetriever | None = None
) -> RagService:
    return RagService(
        retriever=retriever or StubRetriever(THREE),
        llm=llm,
        prompt_builder=PromptBuilder(
            context_window=llm.context_window, max_output_tokens=200, max_context_tokens=2000
        ),
        generation_options=OPTIONS,
        on_answer=on_answer,
    )


# -- Metrics.record_answer -------------------------------------------------------------


def test_answer_is_recorded() -> None:
    metrics = Metrics()

    metrics.record_answer(
        Answer("q", "x [1]", "answered", (), False, TRACE), retrieval_config="hybrid"
    )

    assert sample(metrics, "easyrag_rag_answers_total", outcome="answered") == 1
    assert sample(metrics, "easyrag_rag_stage_duration_seconds_sum", stage="generation") == 3.0
    assert sample(metrics, "easyrag_rag_stage_duration_seconds_sum", stage="total") == 3.25
    assert sample(metrics, "easyrag_llm_tokens_total", kind="prompt") == 850
    assert sample(metrics, "easyrag_llm_tokens_total", kind="completion") == 40
    assert sample(metrics, "easyrag_rag_prompt_tokens_sum") == 850
    assert sample(metrics, "easyrag_rag_chunks_sum", stage="retrieved") == 2
    assert sample(metrics, "easyrag_rag_chunks_sum", stage="cited") == 1
    # RRF scores are small: the buckets must resolve them.
    assert sample(metrics, "easyrag_retrieval_top_score_bucket", config="hybrid", le="0.03") == 0
    assert sample(metrics, "easyrag_retrieval_top_score_bucket", config="hybrid", le="0.035") == 1
    assert sample(metrics, "easyrag_rag_cited_answers_total", attribution="explicit") == 1
    assert sample(metrics, "easyrag_rag_invalid_citations_total") == 1
    assert sample(metrics, "easyrag_rag_truncated_answers_total") == 0


def test_no_evidence_answer_records_no_generation() -> None:
    metrics = Metrics()
    trace = replace(
        TRACE,
        outcome=Outcome.NO_EVIDENCE,
        retrieved=(),
        included_chunk_ids=(),
        cited_chunk_ids=(),
        invalid_citations=0,
        attribution=None,
        prompt_tokens_estimate=0,
        llm_model=None,
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        generation_seconds=0.0,
    )

    metrics.record_answer(
        Answer("q", "I don't know", "insufficient_evidence", (), False, trace),
        retrieval_config="hybrid",
    )

    assert sample(metrics, "easyrag_rag_answers_total", outcome="no_evidence") == 1
    assert sample(metrics, "easyrag_rag_stage_duration_seconds_count", stage="generation") is None
    assert sample(metrics, "easyrag_rag_prompt_tokens_count") == 0
    assert sample(metrics, "easyrag_retrieval_top_score_count", config="hybrid") is None
    assert sample(metrics, "easyrag_rag_chunks_sum", stage="retrieved") == 0


def test_prompt_size_falls_back_to_the_estimate() -> None:
    metrics = Metrics()
    trace = replace(TRACE, prompt_tokens=None, completion_tokens=None)

    metrics.record_answer(Answer("q", "x", "answered", (), True, trace), retrieval_config="vector")

    assert sample(metrics, "easyrag_rag_prompt_tokens_sum") == 900
    assert sample(metrics, "easyrag_llm_tokens_total", kind="prompt") is None
    assert sample(metrics, "easyrag_rag_truncated_answers_total") == 1


# -- RagService observer and log ---------------------------------------------------------


async def test_every_answer_is_observed() -> None:
    seen: list[Answer] = []

    answer = await service(FakeLLMProvider(["Compose [1]."]), seen.append).answer("q")

    assert seen == [answer]


async def test_failing_observer_does_not_fail_the_answer(caplog: pytest.LogCaptureFixture) -> None:
    def broken(answer: Answer) -> None:
        raise RuntimeError("boom")

    answer = await service(FakeLLMProvider(["Compose [1]."]), broken).answer("q")

    assert answer.status == "answered"
    assert "Answer observer failed" in caplog.text


async def test_trace_records_whether_scores_are_reranked() -> None:
    reranked = StubRetriever([replace(chunk("a"), rerank_score=0.9)])

    assert (
        await service(FakeLLMProvider(["Compose [1]."]), None, reranked).answer("q")
    ).trace.reranked
    assert not (await service(FakeLLMProvider(["Compose [1]."]), None).answer("q")).trace.reranked


async def test_trace_log_line_has_structured_fields(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="easyrag.rag.service")

    await service(FakeLLMProvider(["Compose [1]."]), None).answer("secret question")

    record = next(r for r in caplog.records if r.getMessage().startswith("RAG answer"))
    entry = json.loads(JsonFormatter().format(record))
    assert entry["rag"]["outcome"] == "answered"
    assert entry["rag"]["retrieved"] == 3
    assert entry["rag"]["top_score"] == 0.9
    assert entry["rag"]["cited"] == 1
    assert "secret question" not in json.dumps(entry)


# -- InstrumentedTransport --------------------------------------------------------------


async def test_provider_requests_are_timed_by_endpoint_and_outcome() -> None:
    metrics = Metrics()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/chat":
            return httpx2.Response(500, json={"error": "x"})
        if request.url.path == "/api/embed":
            raise httpx2.ConnectError("refused", request=request)
        return httpx2.Response(200, json={})

    transport = InstrumentedTransport(httpx2.MockTransport(handler), metrics)
    async with httpx2.AsyncClient(base_url="http://ollama", transport=transport) as client:
        await client.post("/api/show", json={})
        await client.post("/api/chat", json={})
        await client.get("/some/other/path")
        with pytest.raises(httpx2.ConnectError):
            await client.post("/api/embed", json={})

    count = "easyrag_provider_request_duration_seconds_count"
    assert sample(metrics, count, operation="/api/show", outcome="200") == 1
    assert sample(metrics, count, operation="/api/chat", outcome="500") == 1
    assert sample(metrics, count, operation="other", outcome="200") == 1
    assert sample(metrics, count, operation="/api/embed", outcome="connect_error") == 1


async def test_provider_timeouts_are_recorded() -> None:
    metrics = Metrics()

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    transport = InstrumentedTransport(httpx2.MockTransport(handler), metrics)
    async with httpx2.AsyncClient(base_url="http://ollama", transport=transport) as client:
        with pytest.raises(httpx2.ReadTimeout):
            await client.post("/api/chat", json={})

    assert (
        sample(
            metrics,
            "easyrag_provider_request_duration_seconds_count",
            operation="/api/chat",
            outcome="timeout",
        )
        == 1
    )


def test_redactions_and_unverified_answers_are_counted() -> None:
    metrics = Metrics()
    trace = replace(TRACE, outcome=Outcome.UNGROUNDED, redacted_sources=2, unverified_sentences=1)

    metrics.record_answer(
        Answer("q", "x", "insufficient_evidence", (), False, trace), retrieval_config="hybrid"
    )

    assert sample(metrics, "easyrag_rag_redacted_sources_total") == 2
    assert sample(metrics, "easyrag_rag_unverified_answers_total") == 1
