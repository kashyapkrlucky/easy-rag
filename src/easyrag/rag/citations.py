"""Citation markers and refusal detection in model output (ARCHITECTURE D13).

The prompt numbers its sources 1..n and asks the model to cite them inline as ``[n]``.
``apply_citations`` validates those markers against the number of sources, removes
invalid ones, and renumbers the valid ones by first appearance, so that in the final
answer ``[k]`` always refers to the k-th *cited* source returned to the client.
"""

import re
from dataclasses import dataclass

# [1]  [1, 2]  [1;2]  [Source 1]  [source 1, source 3] — what small models actually emit.
# Never directly after a word character, so code such as ``items[1]`` is not a citation.
_MARKER = re.compile(
    r"(?P<space>[ \t]*)(?<!\w)\[\s*(?P<ids>(?:source\s*)?\d+(?:\s*[,;]\s*(?:source\s*)?\d+)*)\s*\]",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class CitationResult:
    # Answer text with markers renumbered ([k] = k-th cited source) and invalid ones removed.
    text: str
    # Original 1-based source ids, in order of first citation (no duplicates).
    cited: tuple[int, ...]
    # Out-of-range ids found in the text, in order of appearance.
    invalid: tuple[int, ...]


def apply_citations(text: str, source_count: int) -> CitationResult:
    order: list[int] = []
    invalid: list[int] = []
    for match in _MARKER.finditer(text):
        for number in _ids(match):
            if not 1 <= number <= source_count:
                invalid.append(number)
            elif number not in order:
                order.append(number)

    renumber = {original: position for position, original in enumerate(order, start=1)}

    def rewrite(match: re.Match[str]) -> str:
        kept = list(dict.fromkeys(renumber[n] for n in _ids(match) if n in renumber))
        if not kept:
            return ""  # drop the marker together with the space before it
        return match.group("space") + "".join(f"[{number}]" for number in kept)

    return CitationResult(
        text=_MARKER.sub(rewrite, text).strip(), cited=tuple(order), invalid=tuple(invalid)
    )


def is_refusal(text: str, refusal: str) -> bool:
    """True if ``text`` starts with the refusal sentence (ignoring case, quotes, spacing).

    Deliberately strict: a refusal sentence appearing *inside* a longer answer does not
    turn that answer into a refusal.
    """
    return _normalise(text).startswith(_normalise(refusal))


def _ids(match: re.Match[str]) -> list[int]:
    return [int(number) for number in _NUMBER.findall(match.group("ids"))]


def _normalise(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    text = text.strip().lstrip("\"'*_ ").rstrip()
    return " ".join(text.split()).rstrip(".!")
