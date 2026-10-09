"""Deterministic, tokenizer-free token estimation.

EasyRAG never depends on a specific model's tokenizer. Budgets (chunk sizes, prompt
context) are kept well below model limits, so an estimate is sufficient as long as it
is deterministic and does not grossly underestimate any script.
"""

import math
import unicodedata

_WIDE = frozenset({"W", "F"})


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Roughly 4 characters per token for alphabetic scripts and code, and one token per
    wide (CJK) character, which a plain ``len / 4`` would underestimate 2-4x.
    """
    wide = sum(1 for character in text if unicodedata.east_asian_width(character) in _WIDE)
    return wide + math.ceil((len(text) - wide) / 4)
