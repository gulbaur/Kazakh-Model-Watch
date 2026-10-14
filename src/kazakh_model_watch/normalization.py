"""Shared kk-short-v1 normalization for validation and response scoring."""

import unicodedata


def normalize_answer(text: str) -> str:
    """Apply NFC, trim/collapse Unicode whitespace, then casefold.

    Zero-width characters, punctuation and distinct Kazakh letters are retained.
    Compatibility normalization (NFKC) and answer extraction are not performed.
    """
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()
