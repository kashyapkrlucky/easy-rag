"""Deterministic, structure-aware chunking.

Rules (ARCHITECTURE D7, D9):
- a chunk never spans two sections;
- within a section, whole Markdown blocks (paragraphs, lists, tables, code fences,
  quotes) are packed greedily up to the token budget;
- a block is split only when it alone exceeds the budget: first at line boundaries,
  and a single over-long line at whitespace (hard cut as a last resort);
- consecutive chunks overlap by trailing whole blocks totalling <= overlap_tokens;
- a chunk's content is a verbatim, contiguous slice of source lines, except for
  pieces of a single over-long line.

The same input and configuration always produce the same chunks and IDs.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from markdown_it import MarkdownIt

from easyrag.domain.chunks import Chunk
from easyrag.domain.documents import ParsedDocument, Section
from easyrag.ingestion.hashing import content_hash
from easyrag.tokens import estimate_tokens

_ID_SEPARATOR = "\x1f"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    # Budget for embed_text (contextual header + content), in estimated tokens.
    max_tokens: int = 400
    overlap_tokens: int = 50

    def __post_init__(self) -> None:
        if not 32 <= self.max_tokens <= 8192:
            raise ValueError("max_tokens must be between 32 and 8192")
        if not 0 <= self.overlap_tokens <= self.max_tokens // 2:
            raise ValueError("overlap_tokens must be between 0 and max_tokens / 2")


def chunk_id(source_path: str, section_anchor: str, index_in_section: int, digest: str) -> str:
    """Deterministic chunk ID.

    Uses the section anchor (unique within a document) rather than the heading path,
    so sibling sections with identical headings and text cannot collide. Contains no
    document-wide position, so editing one section never changes other sections' IDs.
    """
    key = _ID_SEPARATOR.join((source_path, section_anchor, str(index_in_section), digest))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def context_header(title: str, heading_path: Sequence[str]) -> str:
    """``"Title > Heading > Subheading"``, without repeating a title equal to the first heading."""
    parts = list(heading_path)
    if not parts or parts[0] != title:
        parts.insert(0, title)
    return " > ".join(parts)


@dataclass(frozen=True, slots=True)
class _Unit:
    """An indivisible piece of a section: a line range, or part of one over-long line."""

    start: int  # 0-based line index within the section text, inclusive
    end: int  # inclusive
    text: str | None = None  # set only for pieces of an over-long line


@dataclass(frozen=True, slots=True)
class _Piece:
    content: str
    start: int
    end: int


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._config = config if config is not None else ChunkingConfig()
        self._markdown = MarkdownIt("commonmark").enable(["table", "strikethrough"])

    @property
    def config(self) -> ChunkingConfig:
        return self._config

    def chunk(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for section in document.sections:
            header = context_header(document.title, section.heading_path)
            separator = "\n\n"
            # The header counts against the budget, but never squeezes content below half.
            budget = max(
                self._config.max_tokens - estimate_tokens(header + separator),
                self._config.max_tokens // 2,
            )
            for index, piece in enumerate(self._split_section(section, budget)):
                digest = content_hash(piece.content)
                embed_text = header + separator + piece.content
                chunks.append(
                    Chunk(
                        id=chunk_id(document.source_path, section.anchor, index, digest),
                        source_path=document.source_path,
                        ordinal=len(chunks),
                        heading_path=section.heading_path,
                        section_anchor=section.anchor,
                        content=piece.content,
                        embed_text=embed_text,
                        content_hash=digest,
                        start_line=section.body_start_line + piece.start,
                        end_line=section.body_start_line + piece.end,
                        token_estimate=estimate_tokens(embed_text),
                    )
                )
        return tuple(chunks)

    # -- section → units -------------------------------------------------------------

    def _split_section(self, section: Section, budget: int) -> list[_Piece]:
        lines = section.text.split("\n")
        units = self._units(section.text, lines, budget)
        return self._pack(lines, units, budget)

    def _units(self, text: str, lines: list[str], budget: int) -> list[_Unit]:
        ranges: list[tuple[int, int]] = []
        for token in self._markdown.parse(text):
            if token.level == 0 and token.map is not None:
                trimmed = _trim(lines, token.map[0], token.map[1] - 1)
                if trimmed is not None:
                    ranges.append(trimmed)

        # Lines that produce no block token (e.g. link reference definitions) must not be lost.
        covered = {line for start, end in ranges for line in range(start, end + 1)}
        ranges.extend(
            (index, index)
            for index, line in enumerate(lines)
            if line.strip() and index not in covered
        )

        units: list[_Unit] = []
        for start, end in sorted(ranges):
            if estimate_tokens(_join(lines, start, end)) <= budget:
                units.append(_Unit(start, end))
                continue
            for index in range(start, end + 1):  # oversize block → lines
                if not lines[index].strip():
                    continue
                if estimate_tokens(lines[index]) <= budget:
                    units.append(_Unit(index, index))
                else:  # oversize line → pieces
                    units.extend(
                        _Unit(index, index, piece) for piece in _split_line(lines[index], budget)
                    )
        return units

    # -- units → pieces ----------------------------------------------------------------

    def _pack(self, lines: list[str], units: list[_Unit], budget: int) -> list[_Piece]:
        pieces: list[_Piece] = []
        current: list[_Unit] = []

        def flush() -> None:
            if current:
                pieces.append(_piece(lines, current))

        for unit in units:
            if unit.text is not None:  # pieces of an over-long line always stand alone
                flush()
                current = []
                pieces.append(_Piece(unit.text, unit.start, unit.end))
                continue

            if current and estimate_tokens(_text(lines, [*current, unit])) > budget:
                flush()
                current = [*self._overlap(lines, current), unit]
                while len(current) > 1 and estimate_tokens(_text(lines, current)) > budget:
                    current.pop(0)
            else:
                current.append(unit)

        flush()
        return pieces

    def _overlap(self, lines: list[str], previous: list[_Unit]) -> list[_Unit]:
        """Trailing whole units of ``previous`` within the overlap budget.

        The caller trims the result until it fits beside the next unit. Since ``previous``
        plus that unit already exceeded the budget, the overlap can never be all of
        ``previous``.
        """
        tail: list[_Unit] = []
        for unit in reversed(previous):
            candidate = [unit, *tail]
            if estimate_tokens(_text(lines, candidate)) > self._config.overlap_tokens:
                break
            tail = candidate
        return tail


def _trim(lines: Sequence[str], start: int, end: int) -> tuple[int, int] | None:
    while start <= end and not lines[start].strip():
        start += 1
    while end >= start and not lines[end].strip():
        end -= 1
    return (start, end) if start <= end else None


def _join(lines: Sequence[str], start: int, end: int) -> str:
    return "\n".join(lines[start : end + 1])


def _text(lines: Sequence[str], units: Sequence[_Unit]) -> str:
    """Verbatim source slice covering ``units`` (which are contiguous and ordered)."""
    return _join(lines, units[0].start, units[-1].end)


def _piece(lines: Sequence[str], units: Sequence[_Unit]) -> _Piece:
    return _Piece(_text(lines, units), units[0].start, units[-1].end)


def _split_line(line: str, budget: int) -> list[str]:
    """Split one over-long line at whitespace; hard-cut words that alone exceed the budget."""
    pieces: list[str] = []
    current = ""
    for word in re.findall(r"\S+\s*", line):
        if estimate_tokens(current + word) <= budget:
            current += word
            continue
        if current.strip():
            pieces.append(current.rstrip())
        current = ""
        while estimate_tokens(word) > budget:
            cut = _longest_prefix_within(word, budget)
            pieces.append(word[:cut])
            word = word[cut:]
        current = word
    if current.strip():
        pieces.append(current.rstrip())
    return pieces


def _longest_prefix_within(text: str, budget: int) -> int:
    low, high = 1, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return low
