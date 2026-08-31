"""Shared validation for retrieval-augmented query text."""

from __future__ import annotations


MIN_RAG_QUERY_WEIGHT = 3

# Han ideographs count as two English-equivalent characters. Keep these ranges
# in sync with ``lightrag_webui/src/utils/queryValidation.ts``.
_HAN_CODEPOINT_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x2EBF0, 0x2EE5F),
    (0x2F800, 0x2FA1F),
    (0x30000, 0x3134F),
    (0x31350, 0x323AF),
)


class RAGQueryTooShortError(ValueError):
    """Raised when query text is too short for meaningful RAG retrieval."""


def _is_han_character(character: str) -> bool:
    codepoint = ord(character)
    return codepoint == 0x3007 or any(
        start <= codepoint <= end for start, end in _HAN_CODEPOINT_RANGES
    )


def rag_query_weight(query: str) -> int:
    """Return English-equivalent query length after trimming outer whitespace.

    Each Han ideograph counts as two; every other Unicode code point counts as
    one. Internal whitespace remains part of the query, matching the previous
    character-length contract.
    """

    return sum(2 if _is_han_character(char) else 1 for char in query.strip())


def validate_rag_query(query: str) -> str:
    """Normalize and validate query text used by RAG retrieval modes."""

    normalized = query.strip()
    if rag_query_weight(normalized) < MIN_RAG_QUERY_WEIGHT:
        raise RAGQueryTooShortError(
            "RAG query is too short. Enter at least 3 English characters or an "
            "equivalent combination where each Chinese character counts as 2."
        )
    return normalized
