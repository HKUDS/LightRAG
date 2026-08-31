"""Shared validation for query text entering the server and the core API."""

from __future__ import annotations


MIN_RAG_QUERY_WEIGHT = 3

# East Asian scripts count as two English-equivalent characters: a two-character
# CJK word carries about as much retrieval signal as a three-letter English one.
# Han ideographs, Japanese kana and Korean Hangul are all included — restricting
# the doubling to Han alone rejected ordinary two-character Japanese and Korean
# words ("ねこ", "오늘") while accepting their Han equivalents. The halfwidth
# forms are the same letters in a legacy encoding — "ﾈｺ" is the word "ネコ" — so
# they weigh the same; the rule is about how much a character says, not how wide
# it renders. Punctuation and the halfwidth sound marks stay at 1.
# Keep these ranges in sync with ``lightrag_webui/src/utils/queryValidation.ts``.
_WIDE_CODEPOINT_RANGES = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9D),  # Halfwidth Katakana letters
    (0xFFA1, 0xFFDC),  # Halfwidth Hangul letters
    (0x1B000, 0x1B16F),  # Kana Supplement / Extended-A / Small Kana Extension
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F
    (0x2EBF0, 0x2EE5F),  # Extension I
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x30000, 0x3134F),  # Extension G
    (0x31350, 0x323AF),  # Extension H
)

_TOO_SHORT_MESSAGE = (
    "RAG query is too short. Enter at least 3 English characters or an "
    "equivalent combination where each Chinese, Japanese or Korean character "
    "counts as 2."
)


class QueryValidationError(ValueError):
    """Base class for rejections of user-supplied query text."""


class EmptyQueryError(QueryValidationError):
    """Raised when query text is empty or only whitespace.

    This applies to every path, ``bypass`` and ``/api/generate`` included: an
    empty prompt has no meaning to send to an LLM, and forwarding it only buys
    a provider-side error or a hallucinated answer to nothing.
    """


class RAGQueryTooShortError(QueryValidationError):
    """Raised when query text is too short for meaningful RAG retrieval."""


def _is_wide_character(character: str) -> bool:
    codepoint = ord(character)
    return codepoint == 0x3007 or any(
        start <= codepoint <= end for start, end in _WIDE_CODEPOINT_RANGES
    )


def rag_query_weight(query: str) -> int:
    """Return English-equivalent query length after trimming outer whitespace.

    Each CJK character counts as two; every other Unicode code point counts as
    one. Internal whitespace remains part of the query, matching the previous
    character-length contract.

    Prefer :func:`meets_min_rag_query_weight` when only the comparison against
    :data:`MIN_RAG_QUERY_WEIGHT` matters: this function walks the whole string,
    which is up to 64 KiB of synchronous work on the event loop.
    """

    return sum(2 if _is_wide_character(char) else 1 for char in query.strip())


def meets_min_rag_query_weight(query: str) -> bool:
    """Return whether ``query`` reaches :data:`MIN_RAG_QUERY_WEIGHT`.

    Short-circuits as soon as the threshold is reached, so a 64 KiB query costs
    two iterations rather than 65,536. The full :func:`rag_query_weight` walk
    added ~69 ms of event-loop-blocking CPU to every max-size RAG request.
    """

    weight = 0
    for character in query.strip():
        weight += 2 if _is_wide_character(character) else 1
        if weight >= MIN_RAG_QUERY_WEIGHT:
            return True
    return False


def validate_query_not_empty(query: str) -> str:
    """Normalize query text and reject it when nothing but whitespace remains.

    Applies to every mode, including ``bypass`` and ``/api/generate``, which are
    exempt from the RAG minimum but not from having to say something.
    """

    normalized = query.strip()
    if not normalized:
        raise EmptyQueryError("Query must not be empty.")
    return normalized


def validate_rag_query(query: str) -> str:
    """Normalize and validate query text used by RAG retrieval modes."""

    normalized = validate_query_not_empty(query)
    if not meets_min_rag_query_weight(normalized):
        raise RAGQueryTooShortError(_TOO_SHORT_MESSAGE)
    return normalized
