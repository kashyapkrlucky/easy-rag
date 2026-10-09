import pytest

from easyrag.rag.citations import apply_citations, is_refusal
from easyrag.rag.prompt import REFUSAL


@pytest.mark.parametrize(
    ("text", "count", "expected_text", "cited", "invalid"),
    [
        ("Run compose [1].", 3, "Run compose [1].", (1,), ()),
        # Renumbered by first appearance so [k] matches the k-th returned source.
        ("A [3]. B [1]. C [3].", 3, "A [1]. B [2]. C [1].", (3, 1), ()),
        ("A [2, 3].", 3, "A [1][2].", (2, 3), ()),
        ("A [2][3].", 3, "A [1][2].", (2, 3), ()),
        ("A [2; 3].", 3, "A [1][2].", (2, 3), ()),
        ("A [Source 2].", 3, "A [1].", (2,), ()),
        ("A [source 2, source 1].", 3, "A [1][2].", (2, 1), ()),
        ("A [ 2 ].", 3, "A [1].", (2,), ()),
        # Duplicates inside one marker collapse.
        ("A [2, 2].", 3, "A [1].", (2,), ()),
        # Out-of-range markers are removed together with the preceding space.
        ("A [7].", 3, "A.", (), (7,)),
        ("A [0] and B [2].", 3, "A and B [1].", (2,), (0,)),
        ("A [2, 9].", 3, "A [1].", (2,), (9,)),
        # No markers at all.
        ("Plain answer.", 3, "Plain answer.", (), ()),
        # Not citations: Markdown links and code indexing.
        ("See [docs](http://x) and arr[1].", 3, "See [docs](http://x) and arr[1].", (), ()),
        ("Use items[0] and cite [2].", 3, "Use items[0] and cite [1].", (2,), ()),
    ],
)
def test_apply_citations(
    text: str, count: int, expected_text: str, cited: tuple[int, ...], invalid: tuple[int, ...]
) -> None:
    result = apply_citations(text, count)

    assert (result.text, result.cited, result.invalid) == (expected_text, cited, invalid)


def test_no_sources_makes_every_marker_invalid() -> None:
    result = apply_citations("Answer [1].", 0)

    assert (result.text, result.cited, result.invalid) == ("Answer.", (), (1,))


def test_multiline_answers_keep_their_layout() -> None:
    text = "Steps:\n1. Copy the file [1].\n2. Run it [2].\n\nDone [9]."

    result = apply_citations(text, 2)

    assert result.text == "Steps:\n1. Copy the file [1].\n2. Run it [2].\n\nDone."


@pytest.mark.parametrize(
    "text",
    [
        REFUSAL,
        REFUSAL.upper(),
        f'"{REFUSAL}"',
        f"  {REFUSAL}  ",
        REFUSAL.replace("'", "\u2019"),  # typographic apostrophe
        REFUSAL.rstrip("."),
        f"{REFUSAL} The sources discuss deployment only.",
        f"**{REFUSAL}**",
    ],
)
def test_refusals_are_detected(text: str) -> None:
    assert is_refusal(text, REFUSAL)


@pytest.mark.parametrize(
    "text",
    [
        "Run docker compose up [1].",
        f"The deployment uses Docker [1]. {REFUSAL}",  # refusal buried in an answer
        "I don't know.",  # different wording: handled as an uncited answer instead
        "",
    ],
)
def test_non_refusals_are_not_detected(text: str) -> None:
    assert not is_refusal(text, REFUSAL)
