"""Evaluation metrics: pure functions over labels and results."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from easyrag.domain.chunks import RetrievedChunk
from easyrag.evaluation.dataset import ExpectedSource


def matches(chunk: RetrievedChunk, expected: Sequence[ExpectedSource]) -> bool:
    """Is ``chunk`` one of the acceptable sources?"""
    path = " > ".join(chunk.heading_path)
    return any(chunk.source_path == e.document and e.section in path for e in expected)


def first_relevant_rank(
    results: Sequence[RetrievedChunk], expected: Sequence[ExpectedSource]
) -> int | None:
    """1-based rank of the first acceptable result, or None."""
    return next((rank for rank, c in enumerate(results, start=1) if matches(c, expected)), None)


@dataclass(frozen=True, slots=True)
class RankMetrics:
    hit_at_1: float
    hit_at_k: float
    mrr: float  # mean reciprocal rank (0 when not found within k)
    count: int


def rank_metrics(ranks: Sequence[int | None]) -> RankMetrics:
    if not ranks:
        return RankMetrics(0.0, 0.0, 0.0, 0)
    n = len(ranks)
    return RankMetrics(
        hit_at_1=sum(1 for r in ranks if r == 1) / n,
        hit_at_k=sum(1 for r in ranks if r is not None) / n,
        mrr=sum(1 / r for r in ranks if r is not None) / n,
        count=n,
    )


def fact_recall(answer: str, facts: Sequence[Sequence[str]]) -> float:
    """Share of key facts present in the answer (any alternative, case-insensitive)."""
    if not facts:
        return 1.0
    text = answer.casefold()
    found = sum(1 for alternatives in facts if any(a.casefold() in text for a in alternatives))
    return found / len(facts)


def citation_precision(
    cited: Sequence[RetrievedChunk], expected: Sequence[ExpectedSource]
) -> float | None:
    """Share of cited sources that are acceptable; None when nothing was cited."""
    if not cited:
        return None
    return sum(1 for chunk in cited if matches(chunk, expected)) / len(cited)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile (``fraction`` in (0, 1]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


@dataclass(frozen=True, slots=True)
class ThresholdSuggestion:
    threshold: float
    balanced_accuracy: float
    # How many answerable questions it would wrongly gate, and unanswerable ones it lets
    # through to the model.
    answerable_rejected: int
    unanswerable_accepted: int


def best_threshold(
    answerable: Sequence[float], unanswerable: Sequence[float]
) -> ThresholdSuggestion | None:
    """The score threshold that best separates answerable from unanswerable questions.

    A question passes when its top retrieval score is >= threshold. Candidates are the
    midpoints between consecutive distinct scores; ties prefer the lower threshold (which
    rejects fewer answerable questions).
    """
    if not answerable or not unanswerable:
        return None
    scores = sorted(set(answerable) | set(unanswerable))
    candidates = [scores[0] - 1e-9] + [(a + b) / 2 for a, b in pairwise(scores)]
    best: ThresholdSuggestion | None = None
    for t in candidates:
        rejected = sum(1 for s in answerable if s < t)
        accepted = sum(1 for s in unanswerable if s >= t)
        accuracy = (
            (len(answerable) - rejected) / len(answerable)
            + (len(unanswerable) - accepted) / len(unanswerable)
        ) / 2
        if best is None or accuracy > best.balanced_accuracy:
            best = ThresholdSuggestion(round(t, 4), accuracy, rejected, accepted)
    return best
