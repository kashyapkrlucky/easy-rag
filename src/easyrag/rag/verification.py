"""Verify explicitly cited answers against the sources they cite (ARCHITECTURE D39).

A model's ``[n]`` markers are not evidence by themselves: the 1.5B default model answered
"Version 2.3 removed NIMBUS_COMPRESSION [1]" citing a support-policy section, and it passed
(D34). So each sentence of a cited reply is checked against the sources *it* cites:

- a sentence with markers must have every number in its cited sources, and at least
  ``CITED_SENTENCE_SUPPORT`` of its content words (lower than the 60% used when inferring
  sources: a cited sentence already names its evidence, and paraphrase is common);
- a sentence without markers ("Then restart it.") must have its numbers in *some* source:
  a wrong number is the most harmful error, cited or not.

Small models also cite the wrong number for a correct claim ("needs 2 GB of RAM [1]" with
the RAM section shown as [3]). A cited sentence that fails but is fully supported by
another source, by the stricter standard used to infer sources (all numbers, at least
``SENTENCE_SUPPORT`` of its words), gets its citation corrected instead of being rejected.

Markers standing alone after the full stop ("... zstd. [1]") belong to the sentence before.
Pure and deterministic, like ``rag.attribution``, whose word normalisation it shares.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from easyrag.rag.attribution import SENTENCE_SUPPORT, content_words, numbers_in
from easyrag.rag.citations import marker_ids, recite, strip_markers

CITED_SENTENCE_SUPPORT = 0.4

# Captured, so splitting keeps the separators and the reply can be reassembled exactly.
_SENTENCE_END = re.compile(r"((?<=[.!?])\s+|\n+)")


@dataclass(frozen=True, slots=True)
class Verification:
    verified: bool
    # The reply with corrected citations (unchanged if none were corrected).
    reply: str
    # Sentences whose citation was corrected / that failed (model output, never documents).
    corrected: tuple[str, ...]
    failures: tuple[str, ...]


@dataclass(slots=True)
class _Sentence:
    first: int  # index of its first and last part in the split reply
    last: int
    ids: list[int]


def verify_citations(reply: str, evidence: Sequence[str]) -> Verification:
    """Check ``reply`` (raw model output, original 1-based ids) against ``evidence``.

    ``evidence[i]`` is the text source ``i + 1`` showed the model.
    """
    source_words = [content_words(text) for text in evidence]
    parts = _SENTENCE_END.split(reply)  # sentence, separator, sentence, ...
    corrected: list[str] = []
    failures: list[str] = []
    for sentence in _sentences(parts):
        text = "".join(parts[sentence.first : sentence.last + 1])
        words = content_words(strip_markers(text))
        if _supported(words, sentence.ids, source_words):
            continue
        better = _best_source(words, source_words) if sentence.ids else None
        if better is None:
            failures.append(text.strip())
            continue
        corrected.append(text.strip())
        parts[sentence.first] = recite(text, better)
        for index in range(sentence.first + 1, sentence.last + 1):
            parts[index] = ""
    return Verification(
        verified=not failures,
        reply="".join(parts),
        corrected=tuple(corrected),
        failures=tuple(failures),
    )


def _sentences(parts: list[str]) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    for index in range(0, len(parts), 2):
        segment = parts[index]
        ids = marker_ids(segment)
        if not content_words(strip_markers(segment)):
            if ids and sentences:  # a marker after the full stop: the previous sentence's
                sentences[-1].ids.extend(ids)
                sentences[-1].last = index
            continue
        sentences.append(_Sentence(index, index, ids))
    return sentences


def _supported(words: set[str], ids: list[int], source_words: list[set[str]]) -> bool:
    numbers = numbers_in(words)
    cited = [source_words[i - 1] for i in dict.fromkeys(ids) if 1 <= i <= len(source_words)]
    if not cited:
        # Uncited (or only invalid markers): only the numbers are checked, against any source.
        return not numbers or any(numbers <= candidate for candidate in source_words)
    available = set().union(*cited)
    if not numbers <= available:
        return False
    return len(words & available) / len(words) >= CITED_SENTENCE_SUPPORT


def _best_source(words: set[str], source_words: list[set[str]]) -> int | None:
    """1-based id of the source fully supporting ``words``, if any (best support wins)."""
    numbers = numbers_in(words)
    best, best_support = None, SENTENCE_SUPPORT
    for number, candidate in enumerate(source_words, start=1):
        if not numbers <= candidate:
            continue
        support = len(words & candidate) / len(words)
        if support >= best_support:
            best, best_support = number, support
    return best
