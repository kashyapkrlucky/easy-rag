from textwrap import dedent

import pytest

from easyrag.domain.documents import ParsedDocument, Section, SourceDocument
from easyrag.ingestion.hashing import content_hash
from easyrag.ingestion.markdown import MarkdownParser


def parse(markdown: str, source_path: str = "guide.md") -> ParsedDocument:
    text = dedent(markdown).lstrip("\n")
    document = SourceDocument(
        source_path=source_path,
        text=text,
        content_hash=content_hash(text),
        size_bytes=len(text.encode()),
    )
    return MarkdownParser().parse(document)


def paths(parsed: ParsedDocument) -> list[tuple[str, ...]]:
    return [section.heading_path for section in parsed.sections]


def test_nested_headings_produce_heading_paths() -> None:
    parsed = parse(
        """
        # Deployment
        Overview.
        ## Docker
        Use compose.
        ### Volumes
        Mount docs.
        ## Kubernetes
        Not supported.
        # FAQ
        Questions.
        """
    )

    assert paths(parsed) == [
        ("Deployment",),
        ("Deployment", "Docker"),
        ("Deployment", "Docker", "Volumes"),
        ("Deployment", "Kubernetes"),
        ("FAQ",),
    ]
    assert [s.ordinal for s in parsed.sections] == [0, 1, 2, 3, 4]
    assert [s.level for s in parsed.sections] == [1, 2, 3, 2, 1]


def test_skipped_heading_levels() -> None:
    parsed = parse(
        """
        # Guide
        Intro.
        ### Deep
        Detail.
        ## Middle
        More.
        """
    )

    assert paths(parsed) == [("Guide",), ("Guide", "Deep"), ("Guide", "Middle")]


def test_section_text_excludes_heading_and_surrounding_blank_lines() -> None:
    parsed = parse(
        """
        # Install


        Run the installer.

        Then restart.


        # Next
        """
    )

    (section,) = parsed.sections
    assert section.text == "Run the installer.\n\nThen restart."
    assert (section.start_line, section.end_line) == (1, 6)


def test_line_numbers_point_into_the_source_file() -> None:
    parsed = parse(
        """
        ---
        title: Guide
        ---
        Preamble line.

        # First
        Body one.

        ## Second
        Body two.
        Body two continued.
        """
    )

    assert [(s.start_line, s.end_line) for s in parsed.sections] == [(4, 4), (6, 7), (9, 11)]


def test_hash_inside_code_fence_is_not_a_heading() -> None:
    parsed = parse(
        """
        # Shell

        ```bash
        # this is a comment, not a heading
        echo hi
        ```
        """
    )

    (section,) = parsed.sections
    assert section.heading_path == ("Shell",)
    assert "# this is a comment" in section.text


def test_setext_headings_are_recognised() -> None:
    parsed = parse(
        """
        Title
        =====
        Intro.

        Subtitle
        --------
        Details.
        """
    )

    assert paths(parsed) == [("Title",), ("Title", "Subtitle")]
    assert parsed.title == "Title"


def test_headings_inside_block_quotes_do_not_start_sections() -> None:
    parsed = parse(
        """
        # Notes

        > # Quoted heading
        > quoted text
        """
    )

    (section,) = parsed.sections
    assert section.heading_path == ("Notes",)
    assert "> # Quoted heading" in section.text


def test_preamble_before_first_heading_is_a_section() -> None:
    parsed = parse(
        """
        Some introduction.

        # Details
        More.
        """
    )

    preamble = parsed.sections[0]
    assert (preamble.heading_path, preamble.level, preamble.anchor) == ((), 0, "")
    assert preamble.text == "Some introduction."


def test_headings_without_content_produce_no_section_but_keep_the_path() -> None:
    parsed = parse(
        """
        # Guide
        ## Install
        Steps.
        """
    )

    assert paths(parsed) == [("Guide", "Install")]


def test_heading_text_is_plain_text() -> None:
    parsed = parse(
        """
        # Use **bold**, `code` and [links](https://example.com)
        Body.
        """
    )

    assert parsed.sections[0].heading_path == ("Use bold, code and links",)


def test_anchors_are_github_style_and_unique() -> None:
    parsed = parse(
        """
        # Docker Deployment!
        a
        ## Setup
        b
        # Other
        c
        ## Setup
        d
        ## Café & Crème
        e
        """
    )

    assert [s.anchor for s in parsed.sections] == [
        "docker-deployment",
        "setup",
        "other",
        "setup-1",
        "café--crème",
    ]


@pytest.mark.parametrize(
    ("markdown", "source_path", "expected"),
    [
        ("---\ntitle: From Front Matter\n---\n# Heading\nx\n", "a.md", "From Front Matter"),
        ("## Not level one\nx\n# Level One\ny\n", "a.md", "Level One"),
        ("No headings at all.\n", "guides/getting-started_now.md", "Getting started now"),
        ("---\ntitle: 42\n---\n# Real\nx\n", "a.md", "Real"),
    ],
)
def test_title_resolution(markdown: str, source_path: str, expected: str) -> None:
    assert parse(markdown, source_path).title == expected


def test_front_matter_is_parsed_and_made_json_compatible() -> None:
    parsed = parse(
        """
        ---
        title: Guide
        tags: [ops, docker]
        updated: 2026-10-01
        nested: {owner: platform}
        ---
        Body.
        """
    )

    assert parsed.front_matter == {
        "title": "Guide",
        "tags": ["ops", "docker"],
        "updated": "2026-10-01",
        "nested": {"owner": "platform"},
    }
    assert parsed.sections[0].text == "Body."


@pytest.mark.parametrize("front_matter", ["title: [unclosed", "- just\n- a list"])
def test_invalid_front_matter_is_ignored(front_matter: str) -> None:
    parsed = parse(f"---\n{front_matter}\n---\n# Heading\nBody.\n")

    assert parsed.front_matter == {}
    assert paths(parsed) == [("Heading",)]


def test_empty_document_has_no_sections() -> None:
    parsed = parse("", source_path="empty.md")

    assert parsed.sections == ()
    assert parsed.title == "Empty"


def test_parsing_is_deterministic() -> None:
    markdown = "# A\none\n## B\ntwo\n"

    assert parse(markdown) == parse(markdown)


def test_parsed_document_keeps_source_identity() -> None:
    parsed = parse("# A\nbody\n", source_path="x/y.md")

    assert parsed.source_path == "x/y.md"
    assert parsed.content_hash == content_hash("# A\nbody\n")
    assert all(isinstance(section, Section) for section in parsed.sections)
