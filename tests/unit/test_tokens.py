from easyrag.tokens import estimate_tokens


def test_empty_text_has_no_tokens() -> None:
    assert estimate_tokens("") == 0


def test_alphabetic_text_is_about_four_characters_per_token() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 400) == 100


def test_wide_characters_count_one_token_each() -> None:
    assert estimate_tokens("日本語のテキスト") == 8
    assert estimate_tokens("中文 text") == 2 + 2  # 2 wide + ceil(5 / 4)


def test_estimate_is_deterministic() -> None:
    text = "Deterministic — always the same 結果."

    assert estimate_tokens(text) == estimate_tokens(text)
