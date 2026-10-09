"""Server-side attribution: is an answer supported by the sources the model was shown?

Used when a reply contains no valid citation markers. Small local models often answer
correctly from the sources without citing them, and their markers are unreliable either
way, so grounding must not depend on the model's markers alone.

Method (deterministic, lexical): split the answer into sentences. A sentence is
*supported* by a source when at least ``SENTENCE_SUPPORT`` of its content words occur in
that source **and every number in the sentence occurs in it** (a wrong number is the
most harmful kind of hallucination). Words are lower-cased, identifiers are split on
punctuation (``postgres_password`` → postgres, password) and a plural "s" is stripped.
The answer is supported when supported sentences cover at least ``ANSWER_COVERAGE`` of
its content words; each supported sentence is attributed to the source supporting it best.

Limitations: it checks that the answer's words come from the sources, not that the
reasoning is right; and an answer that merely echoes text from a source (including an
injected instruction) counts as supported.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from easyrag.domain.chunks import RetrievedChunk

SENTENCE_SUPPORT = 0.6
ANSWER_COVERAGE = 0.7

_WORD = re.compile(r"[^\W_]+")  # letters/digits; "_", "-", "." and ":" separate words
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKER = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
_STOPWORD_TEXT = """
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each few
for from further had has have having he her here hers him his how i if in into is it
its itself just me more most my no nor not now of off on once only or other our ours
out over own same she should so some such than that the their theirs them then there
these they this those through to too under until up use used uses using very was we
were what when where which while who whom why will with would you your yours
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


@dataclass(frozen=True, slots=True)
class Attribution:
    supported: bool
    # Supporting chunks in order of first use by a supported sentence (empty if unsupported).
    sources: tuple[RetrievedChunk, ...]
    # Share of the answer's content words in supported sentences (0..1).
    coverage: float


def attribute(answer: str, chunks: Sequence[RetrievedChunk]) -> Attribution:
    chunk_words = [_content_words(chunk.content) for chunk in chunks]
    total = 0
    covered = 0
    used: list[int] = []
    for sentence in _SENTENCE_END.split(_MARKER.sub(" ", answer)):
        words = _content_words(sentence)
        if not words:
            continue
        total += len(words)
        numbers = {word for word in words if word.isdigit()}
        best, best_support = -1, 0.0
        for index, candidates in enumerate(chunk_words):
            if not numbers <= candidates:
                continue  # every number must come from the supporting source
            support = len(words & candidates) / len(words)
            if support > best_support:
                best, best_support = index, support
        if best >= 0 and best_support >= SENTENCE_SUPPORT:
            covered += len(words)
            if best not in used:
                used.append(best)

    coverage = covered / total if total else 0.0
    supported = total > 0 and coverage >= ANSWER_COVERAGE
    return Attribution(
        supported=supported,
        sources=tuple(chunks[index] for index in used) if supported else (),
        coverage=coverage,
    )


def _content_words(text: str) -> set[str]:
    return {
        _singular(word)
        for word in (w.lower() for w in _WORD.findall(text))
        if word not in _STOPWORDS and (len(word) > 2 or word.isdigit())
    }


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
