import unicodedata

from easyrag.ingestion.hashing import content_hash, normalize_text


def test_line_endings_are_normalised() -> None:
    assert normalize_text("a\r\nb\rc\n") == "a\nb\nc\n"


def test_leading_bom_is_removed() -> None:
    assert normalize_text("\ufeff# Title") == "# Title"


def test_unicode_is_composed() -> None:
    decomposed = unicodedata.normalize("NFD", "café")

    assert normalize_text(decomposed) == "café"


def test_equivalent_texts_hash_identically() -> None:
    variants = [
        "\ufeff# Café\r\nBody\r\n",
        "# Café\nBody\n",
        unicodedata.normalize("NFD", "# Café\nBody\n"),
    ]

    assert len({content_hash(normalize_text(variant)) for variant in variants}) == 1


def test_content_hash_is_lowercase_sha256_hex() -> None:
    digest = content_hash("hello")

    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
