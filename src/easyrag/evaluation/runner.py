"""Run an evaluation: isolated index → retrieval metrics → answer metrics → report.

The dataset is indexed into a throwaway database on the configured PostgreSQL server
(``easyrag_eval_<id>``), so evaluating never touches the user's index.
"""

import asyncio
import dataclasses
import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import SecretStr
from sqlalchemy import create_engine, make_url, text

from easyrag.config import Settings
from easyrag.container import build_container
from easyrag.domain.answers import Answer
from easyrag.evaluation.dataset import EvalDataset, EvalQuestion
from easyrag.evaluation.metrics import (
    best_threshold,
    citation_precision,
    fact_recall,
    first_relevant_rank,
    matches,
    percentile,
    rank_metrics,
)
from easyrag.ingestion.indexer import pipeline_fingerprint
from easyrag.rag.prompt import PROMPT_VERSION, prompt_fingerprint
from easyrag.retrieval.base import Retriever
from easyrag.storage.migrate import upgrade_to_head

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


class Answerer(Protocol):
    async def answer(self, question: str, *, top_k: int | None = None) -> Answer: ...


@dataclass(frozen=True, slots=True)
class RetrievalRow:
    config: str
    question_id: str
    kind: str
    rank: int | None  # 1-based rank of the first acceptable source within top_k


@dataclass(frozen=True, slots=True)
class AnswerRow:
    question_id: str
    kind: str
    question: str
    answer: str = ""
    status: str = "error"
    outcome: str | None = None
    attribution: str | None = None
    cited: list[str] = field(default_factory=list)
    fact_recall: float | None = None
    citation_precision: float | None = None
    citation_hit: bool | None = None
    injected_leak: bool | None = None
    correct: bool = False
    retrieval_seconds: float | None = None
    generation_seconds: float | None = None
    total_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


# -- retrieval ---------------------------------------------------------------------------


async def evaluate_retrieval(
    configs: Mapping[str, Retriever], questions: Sequence[EvalQuestion], top_k: int
) -> list[RetrievalRow]:
    rows: list[RetrievalRow] = []
    for name, retriever in configs.items():
        for q in (q for q in questions if q.answerable):
            results = await retriever.retrieve(q.question, top_k=top_k)
            rows.append(RetrievalRow(name, q.id, q.kind, first_relevant_rank(results, q.expected)))
    return rows


async def top_scores(retriever: Retriever, questions: Sequence[EvalQuestion]) -> dict[str, float]:
    """Best retrieval score per question (used to calibrate the similarity threshold)."""
    scores: dict[str, float] = {}
    for q in questions:
        results = await retriever.retrieve(q.question, top_k=1)
        scores[q.id] = results[0].score if results else 0.0
    return scores


# -- answers -----------------------------------------------------------------------------


async def evaluate_answers(
    rag: Answerer, questions: Sequence[EvalQuestion], progress: Progress | None = None
) -> list[AnswerRow]:
    rows: list[AnswerRow] = []
    for number, q in enumerate(questions, start=1):
        try:
            answer = await rag.answer(q.question)
        except Exception as exc:  # one failing question must not end the evaluation
            row = AnswerRow(q.id, q.kind, q.question, error=f"{type(exc).__name__}: {exc}")
        else:
            row = score_answer(q, answer)
        rows.append(row)
        if progress:
            verdict = "ok " if row.correct else "BAD"
            progress(f"[{number:>2}/{len(questions)}] {verdict} {q.kind:12} {q.id} ({row.status})")
    return rows


def score_answer(q: EvalQuestion, answer: Answer) -> AnswerRow:
    trace = answer.trace
    leak = q.injected.casefold() in answer.text.casefold() if q.injected else None
    if q.answerable:
        recall = fact_recall(answer.text, q.facts)
        correct = answer.status == "answered" and recall == 1.0 and not leak
    else:
        recall = None
        correct = answer.status == "insufficient_evidence"
    precision = citation_precision(answer.sources, q.expected) if q.answerable else None
    return AnswerRow(
        question_id=q.id,
        kind=q.kind,
        question=q.question,
        answer=answer.text,
        status=answer.status,
        outcome=str(trace.outcome),
        attribution=trace.attribution,
        cited=[f"{c.source_path}#{c.section}" for c in answer.sources],
        fact_recall=recall,
        citation_precision=precision,
        citation_hit=any(matches(c, q.expected) for c in answer.sources) if q.answerable else None,
        injected_leak=leak,
        correct=correct,
        retrieval_seconds=trace.retrieval_seconds,
        generation_seconds=trace.generation_seconds,
        total_seconds=trace.total_seconds,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
    )


# -- summary -----------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def summarize_retrieval(rows: Sequence[RetrievalRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for config in dict.fromkeys(r.config for r in rows):
        selected = [r for r in rows if r.config == config]
        kinds = {
            k: [r.rank for r in selected if r.kind == k]
            for k in dict.fromkeys(r.kind for r in selected)
        }
        summary[config] = {
            "all": dataclasses.asdict(rank_metrics([r.rank for r in selected])),
            "by_kind": {k: dataclasses.asdict(rank_metrics(ranks)) for k, ranks in kinds.items()},
        }
    return summary


def summarize_answers(rows: Sequence[AnswerRow]) -> dict[str, Any]:
    answerable = [r for r in rows if r.kind != "unanswerable"]
    unanswerable = [r for r in rows if r.kind == "unanswerable"]
    injection = [r for r in rows if r.kind == "injection"]
    answered = [r for r in answerable if r.status == "answered"]

    def latency(attribute: str) -> dict[str, float | None]:
        values = [v for r in rows if (v := getattr(r, attribute)) is not None]
        return {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}

    return {
        "answerable": {
            "count": len(answerable),
            "answered_rate": _rate([r.status == "answered" for r in answerable]),
            "correct_rate": _rate([r.correct for r in answerable]),
            "fact_recall_mean": _mean(
                [r.fact_recall for r in answerable if r.fact_recall is not None]
            ),
            "citation_precision_mean": _mean(
                [r.citation_precision for r in answered if r.citation_precision is not None]
            ),
            "citation_hit_rate": _rate([bool(r.citation_hit) for r in answered]),
            "attribution": dict(Counter(r.attribution or "none" for r in answered)),
        },
        "unanswerable": {
            "count": len(unanswerable),
            "refusal_accuracy": _rate([r.correct for r in unanswerable]),
        },
        "injection": {
            "count": len(injection),
            "resisted_rate": _rate([r.injected_leak is False for r in injection]),
            "correct_rate": _rate([r.correct for r in injection]),
        },
        "by_kind": {
            kind: {"count": len(selected), "correct_rate": _rate([r.correct for r in selected])}
            for kind in dict.fromkeys(r.kind for r in rows)
            if (selected := [r for r in rows if r.kind == kind])
        },
        "outcomes": dict(Counter(r.outcome or "error" for r in rows)),
        "errors": [{"question": r.question_id, "error": r.error} for r in rows if r.error],
        "latency_seconds": {
            "retrieval": latency("retrieval_seconds"),
            "generation": latency("generation_seconds"),
            "total": latency("total_seconds"),
        },
        "tokens": {
            "prompt_mean": _mean([float(r.prompt_tokens) for r in rows if r.prompt_tokens]),
            "completion_mean": _mean(
                [float(r.completion_tokens) for r in rows if r.completion_tokens]
            ),
        },
    }


def summarize_calibration(
    scores: Mapping[str, float], questions: Sequence[EvalQuestion]
) -> dict[str, Any]:
    answerable = [scores[q.id] for q in questions if q.answerable and q.id in scores]
    unanswerable = [scores[q.id] for q in questions if not q.answerable and q.id in scores]
    suggestion = best_threshold(answerable, unanswerable)

    def describe(values: Sequence[float]) -> dict[str, float | None]:
        return {
            "min": min(values, default=None),
            "median": percentile(values, 0.5),
            "max": max(values, default=None),
        }

    return {
        "vector_top1_score": {
            "answerable": describe(answerable),
            "unanswerable": describe(unanswerable),
        },
        "suggested_min_score": dataclasses.asdict(suggestion) if suggestion else None,
    }


# -- orchestration -----------------------------------------------------------------------


@asynccontextmanager
async def isolated_database(database_url: str, *, keep: bool = False) -> AsyncIterator[str]:
    """Create ``easyrag_eval_<id>`` next to the configured database; drop it afterwards."""
    admin_url = make_url(database_url)
    name = f"easyrag_eval_{uuid.uuid4().hex[:10]}"

    def execute(statement: str) -> None:
        engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as connection:
                connection.execute(text(statement))
        finally:
            engine.dispose()

    await asyncio.to_thread(execute, f'CREATE DATABASE "{name}"')
    url = admin_url.set(database=name).render_as_string(hide_password=False)
    try:
        await asyncio.to_thread(upgrade_to_head, url)
        yield url
    finally:
        if keep:
            logger.info("Keeping evaluation database %s", name)
        else:
            await asyncio.to_thread(execute, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


async def run_evaluation(
    settings: Settings,
    dataset: EvalDataset,
    *,
    retrieval_only: bool = False,
    keep_database: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    say = progress or (lambda message: None)
    started = time.perf_counter()
    async with isolated_database(
        settings.database_url.get_secret_value(), keep=keep_database
    ) as url:
        eval_settings = settings.model_copy(
            update={
                "database_url": SecretStr(url),
                "docs_dir": dataset.corpus_dir,
                "index_on_startup": False,
                "watch_enabled": False,
            }
        )
        container = build_container(eval_settings)
        try:
            say(f"Indexing {dataset.corpus_dir} into an isolated database")
            index_report = await container.indexer.run()

            await container.reranker.load()
            rerank_available = container.reranker.state == "ready"
            configs: dict[str, Retriever] = dict(container.retrievers)
            if rerank_available:
                configs |= {
                    f"{n}+rerank": container.reranked(r) for n, r in container.retrievers.items()
                }
            else:
                say(f"Reranker unavailable ({container.reranker.state}); skipping rerank configs")

            say(f"Evaluating retrieval: {len(configs)} configurations")
            retrieval_rows = await evaluate_retrieval(
                configs, dataset.questions, settings.retrieval_top_k
            )
            scores = await top_scores(container.retrievers["vector"], dataset.questions)

            answer_rows: list[AnswerRow] = []
            if not retrieval_only:
                if settings.rerank_enabled and not rerank_available:
                    say("Note: reranking is enabled but unavailable; answers use retrieval order")
                say(f"Evaluating answers: {len(dataset.questions)} questions (slow on CPU)")
                answer_rows = await evaluate_answers(container.rag, dataset.questions, progress)
        finally:
            await container.aclose()

    return {
        "dataset": {
            "name": dataset.name,
            "questions": len(dataset.questions),
            "kinds": dict(Counter(q.kind for q in dataset.questions)),
        },
        "config": {
            "embedding_model": container.embeddings.model_id,
            "llm_model": container.llm.model_id,
            "llm_context_window": settings.llm_context_window,
            "llm_max_output_tokens": settings.llm_max_output_tokens,
            "prompt_version": PROMPT_VERSION,
            "prompt_fingerprint": prompt_fingerprint(),
            "pipeline_fingerprint": pipeline_fingerprint(
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                embedding_model=container.embeddings.model_id,
                dimension=container.embeddings.dimension,
            ),
            "retrieval_mode": settings.retrieval_mode,
            "retrieval_top_k": settings.retrieval_top_k,
            "retrieval_min_score": settings.retrieval_min_score,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model,
            "rag_max_context_tokens": settings.rag_max_context_tokens,
            "rag_ungrounded_policy": settings.rag_ungrounded_policy,
            "rag_injection_filter": settings.rag_injection_filter,
        },
        "run": {
            "started_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round(time.perf_counter() - started, 1),
            "retrieval_only": retrieval_only,
            "indexed_chunks": index_report.chunks_embedded + index_report.chunks_reused,
        },
        "retrieval": summarize_retrieval(retrieval_rows),
        "calibration": summarize_calibration(scores, dataset.questions),
        "answers": summarize_answers(answer_rows) if answer_rows else None,
        "rows": {
            "retrieval": [dataclasses.asdict(r) for r in retrieval_rows],
            "answers": [dataclasses.asdict(r) for r in answer_rows],
        },
    }
