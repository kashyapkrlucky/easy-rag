import re
from itertools import pairwise
from pathlib import Path
from textwrap import dedent

import pytest

from easyrag.domain.chunks import Chunk
from easyrag.domain.documents import ParsedDocument, SourceDocument
from easyrag.ingestion.chunker import Chunker, ChunkingConfig, chunk_id, context_header
from easyrag.ingestion.hashing import content_hash
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.tokens import estimate_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse(markdown: str, source_path: str = "guide.md") -> tuple[str, ParsedDocument]:
    text = dedent(markdown).lstrip("\n")
    source = SourceDocument(source_path, text, content_hash(text), len(text.encode()))
    return text, MarkdownParser().parse(source)


def chunk(
    markdown: str, *, max_tokens: int = 400, overlap_tokens: int = 0, source_path: str = "guide.md"
) -> tuple[str, ParsedDocument, tuple[Chunk, ...]]:
    text, parsed = parse(markdown, source_path)
    config = ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    return text, parsed, Chunker(config).chunk(parsed)


def paragraphs(count: int, words: int = 30, prefix: str = "p") -> str:
    return "\n\n".join(" ".join(f"{prefix}{n}w{w}" for w in range(words)) for n in range(count))


def assert_invariants(
    text: str, parsed: ParsedDocument, chunks: tuple[Chunk, ...], max_tokens: int
) -> None:
    source_lines = text.split("\n")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk IDs must be unique"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    for c in chunks:
        assert re.fullmatch(r"[0-9a-f]{32}", c.id)
        assert c.content_hash == content_hash(c.content)
        assert c.token_estimate == estimate_tokens(c.embed_text) <= max_tokens
        assert c.embed_text.endswith(c.content)
        assert 1 <= c.start_line <= c.end_line
        # Content is the verbatim source slice, or a fragment of a single over-long line.
        verbatim = "\n".join(source_lines[c.start_line - 1 : c.end_line])
        assert c.content == verbatim or (c.start_line == c.end_line and c.content in verbatim)

    for section in parsed.sections:
        inside = [c for c in chunks if c.section_anchor == section.anchor]
        assert inside, f"section {section.anchor!r} produced no chunks"
        for c in inside:  # never crosses a section boundary
            assert section.body_start_line <= c.start_line and c.end_line <= section.end_line
        # Nothing is lost: every non-blank body line lies inside some chunk's line range.
        covered = {line for c in inside for line in range(c.start_line, c.end_line + 1)}
        body = range(section.body_start_line, section.end_line + 1)
        assert {n for n in body if source_lines[n - 1].strip()} <= covered


def test_small_section_becomes_one_chunk() -> None:
    text, parsed, chunks = chunk(
        """
        # Guide
        Intro paragraph.

        ## Install
        Run the installer.

        Then restart.
        """
    )

    assert [c.content for c in chunks] == [
        "Intro paragraph.",
        "Run the installer.\n\nThen restart.",
    ]
    assert [(c.start_line, c.end_line) for c in chunks] == [(2, 2), (5, 7)]
    assert [c.heading_path for c in chunks] == [("Guide",), ("Guide", "Install")]
    assert_invariants(text, parsed, chunks, 400)


def test_long_section_is_packed_into_budgeted_chunks() -> None:
    text, parsed, chunks = chunk(f"# Guide\n\n{paragraphs(12)}\n", max_tokens=200)

    assert len(chunks) > 1
    assert all(c.heading_path == ("Guide",) for c in chunks)
    # Whole paragraphs only: no chunk starts or ends mid-paragraph.
    for c in chunks:
        assert all(
            block.startswith("p") and block.endswith("w29") for block in c.content.split("\n\n")
        )
    assert_invariants(text, parsed, chunks, 200)


def test_chunks_never_span_sections() -> None:
    text, parsed, chunks = chunk(
        f"# A\n\n{paragraphs(3, prefix='a')}\n\n# B\n\n{paragraphs(3, prefix='b')}\n",
        max_tokens=4000,
    )

    assert [c.heading_path for c in chunks] == [("A",), ("B",)]
    assert_invariants(text, parsed, chunks, 4000)


def test_code_fence_that_fits_is_kept_intact() -> None:
    code = "\n".join(f"    step_{i}()" for i in range(20))
    text, parsed, chunks = chunk(
        f"# Deploy\n\n{paragraphs(4)}\n\n```python\ndef deploy():\n{code}\n```\n", max_tokens=200
    )

    fenced = [c for c in chunks if "```python" in c.content]
    assert len(fenced) == 1
    assert fenced[0].content.count("```") == 2
    assert_invariants(text, parsed, chunks, 200)


def test_table_is_kept_intact() -> None:
    rows = "\n".join(f"| key{i} | value{i} |" for i in range(15))
    text, parsed, chunks = chunk(
        f"# Config\n\n{paragraphs(3)}\n\n| Key | Value |\n|---|---|\n{rows}\n", max_tokens=200
    )

    tables = [c for c in chunks if "| Key | Value |" in c.content]
    assert len(tables) == 1
    assert all(f"key{i}" in tables[0].content for i in range(15))
    assert_invariants(text, parsed, chunks, 200)


def test_oversized_block_is_split_at_line_boundaries() -> None:
    code = "\n".join(f"line_{i:03d} = compute({i})  # explanation for line {i}" for i in range(120))
    text, parsed, chunks = chunk(f"# Big\n\n```python\n{code}\n```\n", max_tokens=200)

    assert len(chunks) > 1
    joined = "\n".join(c.content for c in chunks)
    assert all(f"line_{i:03d}" in joined for i in range(120))
    assert_invariants(text, parsed, chunks, 200)


def test_single_over_long_line_is_split_at_whitespace() -> None:
    line = " ".join(f"word{i}" for i in range(600))
    text, parsed, chunks = chunk(f"# Long\n\n{line}\n", max_tokens=100)

    assert len(chunks) > 1
    assert " ".join(c.content for c in chunks) == line
    assert all(c.start_line == c.end_line == 3 for c in chunks)
    assert_invariants(text, parsed, chunks, 100)


def test_unbreakable_token_is_hard_cut() -> None:
    blob = "x" * 2000
    text, parsed, chunks = chunk(f"# Blob\n\n{blob}\n", max_tokens=100)

    assert "".join(c.content for c in chunks) == blob
    assert_invariants(text, parsed, chunks, 100)


def test_overlap_repeats_trailing_whole_blocks() -> None:
    short = "\n\n".join(f"Short paragraph {i}." for i in range(40))
    text, parsed, chunks = chunk(f"# Notes\n\n{short}\n", max_tokens=100, overlap_tokens=20)

    source_lines = text.split("\n")
    assert len(chunks) > 1
    for previous, following in pairwise(chunks):
        assert following.start_line <= previous.end_line, "consecutive chunks should overlap"
        shared = "\n".join(source_lines[following.start_line - 1 : previous.end_line])
        assert previous.content.endswith(shared)
        assert following.content.startswith(shared)
        assert estimate_tokens(shared) <= 20
        assert shared != previous.content  # never the entire previous chunk
    assert_invariants(text, parsed, chunks, 100)


def test_no_overlap_when_disabled() -> None:
    short = "\n\n".join(f"Short paragraph {i}." for i in range(40))
    _, _, chunks = chunk(f"# Notes\n\n{short}\n", max_tokens=100, overlap_tokens=0)

    for previous, following in pairwise(chunks):
        assert following.start_line > previous.end_line


def test_link_reference_definitions_are_not_lost() -> None:
    text, parsed, chunks = chunk(
        """
        # Links
        See [the docs][docs].

        [docs]: https://example.com/docs
        """
    )

    assert "[docs]: https://example.com/docs" in chunks[0].content
    assert_invariants(text, parsed, chunks, 400)


def test_embed_text_has_contextual_header() -> None:
    _, _, chunks = chunk(
        """
        ---
        title: Operations Handbook
        ---
        # Deployment
        ## Docker
        Use compose.
        """
    )

    assert chunks[0].embed_text == "Operations Handbook > Deployment > Docker\n\nUse compose."


def test_title_is_not_repeated_when_it_equals_the_first_heading() -> None:
    assert context_header("Guide", ("Guide", "Install")) == "Guide > Install"
    assert context_header("Guide", ()) == "Guide"
    assert context_header("Handbook", ("Guide",)) == "Handbook > Guide"


def test_chunking_is_deterministic() -> None:
    markdown = f"# A\n\n{paragraphs(10)}\n\n## B\n\n{paragraphs(5, prefix='b')}\n"

    assert chunk(markdown, max_tokens=150)[2] == chunk(markdown, max_tokens=150)[2]


def test_editing_one_section_changes_only_its_chunk_ids() -> None:
    before = (
        f"# A\n\n{paragraphs(6, prefix='a')}\n\n"
        f"# B\n\n{paragraphs(6, prefix='b')}\n\n"
        "# C\n\nc text\n"
    )
    after = before.replace("b2w5", "EDITED")

    chunks_before = chunk(before, max_tokens=150)[2]
    chunks_after = chunk(after, max_tokens=150)[2]

    def ids(chunks: tuple[Chunk, ...], anchor: str) -> list[str]:
        return [c.id for c in chunks if c.section_anchor == anchor]

    assert ids(chunks_before, "a") == ids(chunks_after, "a")
    assert ids(chunks_before, "c") == ids(chunks_after, "c")
    assert ids(chunks_before, "b") != ids(chunks_after, "b")


def test_identical_sibling_sections_get_distinct_ids() -> None:
    _, _, chunks = chunk(
        """
        # Linux
        ## Setup
        Run setup.
        # macOS
        ## Setup
        Run setup.
        """
    )

    assert len({c.id for c in chunks}) == len(chunks) == 2
    assert [c.section_anchor for c in chunks] == ["setup", "setup-1"]


def test_chunk_id_is_stable_and_well_formed() -> None:
    digest = content_hash("text")

    assert chunk_id("a.md", "intro", 0, digest) == chunk_id("a.md", "intro", 0, digest)
    assert chunk_id("a.md", "intro", 0, digest) != chunk_id("a.md", "intro", 1, digest)
    assert chunk_id("a.md", "intro", 0, digest) != chunk_id("b.md", "intro", 0, digest)
    assert re.fullmatch(r"[0-9a-f]{32}", chunk_id("a.md", "intro", 0, digest))


def test_cjk_text_respects_the_budget() -> None:
    sentence = "検索拡張生成は文書に基づいて回答します。"
    text, parsed, chunks = chunk(
        f"# 概要\n\n{(sentence * 3 + chr(10) + chr(10)) * 20}", max_tokens=120
    )

    assert len(chunks) > 1
    assert_invariants(text, parsed, chunks, 120)


def test_empty_document_produces_no_chunks() -> None:
    assert chunk("")[2] == ()


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"), [(16, 0), (10_000, 0), (100, 51), (100, -1)]
)
def test_invalid_configuration_is_rejected(max_tokens: int, overlap_tokens: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)


@pytest.mark.parametrize("document", ["README.md", "ARCHITECTURE.md", "ROADMAP.md"])
@pytest.mark.parametrize(("max_tokens", "overlap_tokens"), [(400, 50), (120, 30), (64, 0)])
def test_invariants_hold_on_real_documents(
    document: str, max_tokens: int, overlap_tokens: int
) -> None:
    text = (PROJECT_ROOT / document).read_text(encoding="utf-8")
    source = SourceDocument(document, text, content_hash(text), len(text.encode()))
    parsed = MarkdownParser().parse(source)

    chunks = Chunker(ChunkingConfig(max_tokens, overlap_tokens)).chunk(parsed)

    assert_invariants(text, parsed, chunks, max_tokens)
