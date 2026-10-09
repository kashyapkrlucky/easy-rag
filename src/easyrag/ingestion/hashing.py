"""Text normalisation and content hashing.

Hashes are computed on normalised text so that encoding artefacts (BOM, CRLF line
endings, composed vs decomposed Unicode) never cause spurious re-indexing.
"""

import hashlib
import unicodedata


def normalize_text(text: str) -> str:
    """Strip a leading BOM, convert line endings to "\\n" and apply Unicode NFC."""
    text = text.removeprefix("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def content_hash(text: str) -> str:
    """Lowercase hex SHA-256 of the UTF-8 encoding of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
