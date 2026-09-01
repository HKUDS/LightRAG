"""Shared validation for query text entering the server and the core API."""

from __future__ import annotations


MIN_RAG_QUERY_WEIGHT = 3

# East Asian characters count as two English-equivalent ones: a two-character
# CJK word carries about as much retrieval signal as a three-letter English one.
#
# These are the East Asian Wide/Fullwidth blocks — DELIBERATELY COARSE, at block
# and plane granularity rather than per assigned character. That choice is the
# point, not a shortcut:
#
#   * It cannot rot. U+20000-U+3FFFD is the whole of planes 2 and 3, which
#     Unicode allocates to CJK ideographs, so every extension from B through J
#     and every future one is already covered. A table enumerated per assigned
#     character needs an edit — in two languages — on every Unicode release, and
#     silently under-counts real characters until someone makes it.
#   * The precision it gives up is worth nothing here. An unassigned or
#     punctuation code point inside these ranges weighs 2 instead of 1, so a
#     query of two such characters clears a minimum it should not. The cost of
#     that is one retrieval that finds nothing. Nobody types "・・" meaning to
#     ask something, and a three-character query is a coarse heuristic for
#     "did the user actually ask anything", not a correctness boundary.
#
# Keep these ranges in sync with ``lightrag_webui/src/utils/queryValidation.ts``;
# ``test_the_frontend_range_table_matches_this_one`` enforces it.
_WIDE_CODEPOINT_RANGES = (
    (0x1100, 0x115F),  # Hangul Jamo
    (0x2E80, 0x303E),  # CJK Radicals, Kangxi, CJK Symbols and Punctuation
    (0x3041, 0x33FF),  # Kana, Bopomofo, Hangul Compat Jamo, Kanbun, Enclosed CJK
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA000, 0xA4CF),  # Yi Syllables and Radicals
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFE30, 0xFE4F),  # CJK Compatibility Forms
    (0xFF00, 0xFF60),  # Fullwidth Forms
    (0xFF66, 0xFF9D),  # Halfwidth Katakana — East Asian NARROW, but the
    (0xFFA1, 0xFFDC),  # Halfwidth Hangul — same letters as the wide forms
    (0xFFE0, 0xFFE6),  # Fullwidth signs
    (0x16FE0, 0x16FE4),  # Ideographic Symbols and Punctuation
    (0x17000, 0x18AFF),  # Tangut
    (0x1AFF0, 0x1B2FF),  # Kana Extended-B/Supplement/Extended-A, Small Kana, Nushu
    (0x20000, 0x3FFFD),  # Planes 2 and 3 — every CJK extension, present and future
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
    return any(start <= codepoint <= end for start, end in _WIDE_CODEPOINT_RANGES)


def rag_query_weight(query: str) -> int:
    """Return English-equivalent query length after trimming outer whitespace.

    Each East Asian character counts as two; every other Unicode code point
    counts as one. Internal whitespace remains part of the query, matching the
    previous character-length contract.

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
