import pytest

from easyrag.domain.answers import Answer, Outcome, RagTrace
from easyrag.domain.chunks import RetrievedChunk
from easyrag.evaluation.dataset import EvalQuestion, ExpectedSource, QuestionKind
from easyrag.evaluation.metrics import (
    best_threshold,
    citation_precision,
    fact_recall,
    first_relevant_rank,
    percentile,
    rank_metrics,
)
from easyrag.evaluation.runner import evaluate_answers, score_answer, summarize_answers

EXPECTED = [ExpectedSource(document="guide.md", section="Install")]


def chunk(path: str, heading: tuple[str, ...]) -> RetrievedChunk:
    return RetrievedChunk("id-" + path + "-".join(heading), path, "T", heading, "", "x", 1, 1, 0.5)


RIGHT = chunk("guide.md", ("Guide", "Install"))
WRONG = chunk("guide.md", ("Guide", "Usage"))
OTHER = chunk("other.md", ("Install",))


def test_first_relevant_rank_matches_document_and_section() -> None:
    assert first_relevant_rank([WRONG, OTHER, RIGHT], EXPECTED) == 3
    assert first_relevant_rank([WRONG, OTHER], EXPECTED) is None


def test_rank_metrics() -> None:
    metrics = rank_metrics([1, 2, None, 4])

    assert (metrics.hit_at_1, metrics.hit_at_k, metrics.count) == (0.25, 0.75, 4)
    assert metrics.mrr == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert rank_metrics([]).count == 0


def test_fact_recall_accepts_alternatives_case_insensitively() -> None:
    facts = [["4 hours", "four hours"], ["critical"]]

    assert fact_recall("Critical tickets: FOUR HOURS.", facts) == 1.0
    assert fact_recall("Critical tickets are fast.", facts) == 0.5
    assert fact_recall("anything", []) == 1.0


def test_citation_precision() -> None:
    assert citation_precision([RIGHT, WRONG], EXPECTED) == 0.5
    assert citation_precision([], EXPECTED) is None


def test_percentile_nearest_rank() -> None:
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.95) == 5.0
    assert percentile([], 0.5) is None


def test_best_threshold_separates_and_reports_errors() -> None:
    perfect = best_threshold([0.8, 0.9, 0.7], [0.3, 0.4])
    overlap = best_threshold([0.8, 0.5, 0.7], [0.3, 0.6])

    assert perfect is not None and perfect.balanced_accuracy == 1.0
    assert 0.4 < perfect.threshold <= 0.7
    assert overlap is not None and overlap.balanced_accuracy < 1.0
    assert overlap.answerable_rejected + overlap.unanswerable_accepted >= 1
    assert best_threshold([0.5], []) is None


# -- answer scoring --------------------------------------------------------------------


def trace(outcome: Outcome = Outcome.ANSWERED) -> RagTrace:
    return RagTrace(
        "4", outcome, (), (), (), (), 0, "explicit", 10, "m", "stop", 9, 3, 0.1, 1.0, 1.1
    )


def question(kind: QuestionKind = "factual", injected: str | None = None) -> EvalQuestion:
    if kind == "unanswerable":
        return EvalQuestion(id="q", kind="unanswerable", question="What?")
    return EvalQuestion(
        id="q", kind=kind, question="How?", expected=EXPECTED, facts=[["run it"]], injected=injected
    )


def test_correct_answer_needs_status_facts_and_no_leak() -> None:
    good = score_answer(
        question(), Answer("q", "Run it [1].", "answered", (RIGHT,), False, trace())
    )
    missing_fact = score_answer(
        question(), Answer("q", "Do X [1].", "answered", (RIGHT,), False, trace())
    )
    refused = score_answer(
        question(), Answer("q", "I don't know", "insufficient_evidence", (), False, trace())
    )

    assert good.correct and good.citation_precision == 1.0 and good.citation_hit
    assert not missing_fact.correct and missing_fact.fact_recall == 0.0
    assert not refused.correct


def test_injection_leak_makes_an_answer_incorrect() -> None:
    q = question("injection", injected="BANANA")

    leaked = score_answer(
        q, Answer("q", "BANANA. Run it [1].", "answered", (RIGHT,), False, trace())
    )
    resisted = score_answer(q, Answer("q", "Run it [1].", "answered", (RIGHT,), False, trace()))

    assert leaked.injected_leak is True and not leaked.correct
    assert resisted.injected_leak is False and resisted.correct


def test_unanswerable_is_correct_only_when_refused() -> None:
    q = question("unanswerable")

    assert score_answer(q, Answer("q", "No.", "insufficient_evidence", (), False, trace())).correct
    assert not score_answer(q, Answer("q", "Paris", "answered", (RIGHT,), False, trace())).correct


async def test_evaluate_answers_records_errors_without_stopping() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        async def answer(self, question: str, *, top_k: int | None = None) -> Answer:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model crashed")
            return Answer(question, "Run it [1].", "answered", (RIGHT,), False, trace())

    rows = await evaluate_answers(Flaky(), [question(), question()])

    assert rows[0].error == "RuntimeError: model crashed" and not rows[0].correct
    assert rows[1].correct
    summary = summarize_answers(rows)
    assert summary["answerable"]["correct_rate"] == 0.5
    assert summary["errors"] == [{"question": "q", "error": "RuntimeError: model crashed"}]
