"""Markdown parsing: front matter, title and heading-delimited sections.

Built on markdown-it-py's CommonMark token stream, which provides source line maps
and correctly ignores "#" inside code fences. Only top-level headings delimit
sections; headings nested in block quotes or lists are treated as body text.
"""

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Any

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.front_matter import front_matter_plugin

from easyrag.domain.documents import ParsedDocument, Section, SourceDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    text: str
    anchor: str
    line: int  # 0-based line of the heading
    body_start: int  # 0-based first line after the heading


class MarkdownParser:
    def __init__(self) -> None:
        self._markdown = (
            MarkdownIt("commonmark").enable(["table", "strikethrough"]).use(front_matter_plugin)
        )

    def parse(self, document: SourceDocument) -> ParsedDocument:
        tokens = self._markdown.parse(document.text)
        lines = document.text.split("\n")

        front_matter: dict[str, Any] = {}
        content_start = 0
        headings: list[_Heading] = []
        anchors = _AnchorRegistry()

        for index, token in enumerate(tokens):
            if token.type == "front_matter" and token.map:
                front_matter = _parse_front_matter(token.content, document.source_path)
                content_start = token.map[1]
            elif token.type == "heading_open" and token.level == 0 and token.map:
                text = _inline_text(tokens[index + 1])
                headings.append(
                    _Heading(
                        level=int(token.tag[1]),
                        text=text,
                        anchor=anchors.claim(text),
                        line=token.map[0],
                        body_start=token.map[1],
                    )
                )

        return ParsedDocument(
            source_path=document.source_path,
            content_hash=document.content_hash,
            title=_title(front_matter, headings, document.source_path),
            front_matter=front_matter,
            sections=_sections(lines, content_start, headings),
        )


def _sections(
    lines: Sequence[str], content_start: int, headings: Sequence[_Heading]
) -> tuple[Section, ...]:
    sections: list[Section] = []
    first_heading_line = headings[0].line if headings else len(lines)

    preamble = _trimmed_range(lines, content_start, first_heading_line)
    if preamble is not None:
        first, last = preamble
        sections.append(
            Section(
                ordinal=0,
                heading_path=(),
                level=0,
                anchor="",
                text="\n".join(lines[first : last + 1]),
                start_line=first + 1,
                end_line=last + 1,
                body_start_line=first + 1,
            )
        )

    stack: list[_Heading] = []
    for position, heading in enumerate(headings):
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        stack.append(heading)

        body_end = headings[position + 1].line if position + 1 < len(headings) else len(lines)
        body = _trimmed_range(lines, heading.body_start, body_end)
        if body is None:
            continue  # a heading with no content of its own
        first, last = body
        sections.append(
            Section(
                ordinal=len(sections),
                heading_path=tuple(item.text for item in stack),
                level=heading.level,
                anchor=heading.anchor,
                text="\n".join(lines[first : last + 1]),
                start_line=heading.line + 1,
                end_line=last + 1,
                body_start_line=first + 1,
            )
        )

    return tuple(sections)


def _trimmed_range(lines: Sequence[str], start: int, end: int) -> tuple[int, int] | None:
    """First and last non-blank line indexes in ``lines[start:end]``, or None if all blank."""
    non_blank = [index for index in range(start, end) if lines[index].strip()]
    return (non_blank[0], non_blank[-1]) if non_blank else None


def _inline_text(token: Token) -> str:
    """Plain text of an inline token: formatting, links and HTML markup removed."""
    parts: list[str] = []
    for child in token.children or []:
        if child.type in ("text", "code_inline", "image"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return " ".join("".join(parts).split())


class _AnchorRegistry:
    """GitHub-style heading anchors, de-duplicated within one document."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def claim(self, heading: str) -> str:
        base = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-") or "section"
        count = self._seen.get(base, 0)
        self._seen[base] = count + 1
        return base if count == 0 else f"{base}-{count}"


def _title(front_matter: Mapping[str, Any], headings: Sequence[_Heading], source_path: str) -> str:
    title = front_matter.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    for heading in headings:
        if heading.level == 1 and heading.text:
            return heading.text
    stem = " ".join(re.split(r"[-_\s]+", PurePosixPath(source_path).stem)).strip()
    return stem[:1].upper() + stem[1:] if stem else source_path


def _parse_front_matter(raw: str, source_path: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        logger.warning("Ignoring invalid YAML front matter in %s", source_path)
        return {}
    if value is None:
        return {}
    if not isinstance(value, dict):
        logger.warning("Ignoring front matter in %s: expected a mapping", source_path)
        return {}
    return {str(key): _json_compatible(item) for key, item in value.items()}


def _json_compatible(value: Any) -> Any:
    """Convert YAML values into JSON-serialisable ones (stored as JSONB)."""
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_compatible(item) for item in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
