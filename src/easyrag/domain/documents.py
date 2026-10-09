"""Documents as they move through ingestion: loaded from disk, then parsed into sections."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A Markdown file read from the docs root, with normalised text."""

    # Relative POSIX path from the docs root, NFC-normalised, e.g. "guides/deployment.md".
    source_path: str
    # Normalised text: UTF-8 decoded, BOM removed, "\n" line endings, Unicode NFC.
    text: str
    # SHA-256 (lowercase hex) of the normalised text.
    content_hash: str
    # Size of the file on disk, before normalisation.
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SkippedFile:
    """A candidate file that could not be loaded, and why."""

    source_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class LoadResult:
    documents: tuple[SourceDocument, ...]
    skipped: tuple[SkippedFile, ...]


@dataclass(frozen=True, slots=True)
class Section:
    """Body text under one heading (or before the first heading)."""

    # Position of the section within its document, starting at 0.
    ordinal: int
    # Headings from the outermost to this section's own, e.g. ("Deployment", "Docker").
    # Empty for the preamble before the first heading.
    heading_path: tuple[str, ...]
    # Heading level 1-6; 0 for the preamble.
    level: int
    # URL fragment for the heading, unique within the document; "" for the preamble.
    anchor: str
    # Section body without the heading line, trimmed of surrounding blank lines.
    text: str
    # 1-based, inclusive line range in the source file (heading line included).
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    source_path: str
    content_hash: str
    title: str
    # JSON-compatible front matter (dates converted to ISO strings).
    front_matter: Mapping[str, Any]
    sections: tuple[Section, ...]
