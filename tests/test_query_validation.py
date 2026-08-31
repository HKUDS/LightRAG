import pytest

from lightrag.query_validation import (
    EmptyQueryError,
    RAGQueryTooShortError,
    meets_min_rag_query_weight,
    rag_query_weight,
    validate_query_not_empty,
    validate_rag_query,
)


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "query, expected",
    [
        ("ab", 2),
        ("中", 2),
        ("中a", 3),
        ("中文", 4),
        ("  abc  ", 3),
        ("〇", 2),
        ("𠀀", 2),
        # Kana and Hangul weigh the same as Han: a two-character Japanese or
        # Korean word carries as much retrieval signal as a two-character
        # Chinese one, and counting it as 2 rejected both.
        ("ねこ", 4),
        ("データ", 6),
        ("한글", 4),
        ("오늘", 4),
    ],
)
def test_rag_query_weight_counts_cjk_as_two(query, expected):
    assert rag_query_weight(query) == expected


@pytest.mark.parametrize("query", ["", "  ", "a", "ab", "中", "〇"])
def test_validate_rag_query_rejects_weight_below_three(query):
    with pytest.raises(ValueError):
        validate_rag_query(query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("abc", "abc"),
        ("中a", "中a"),
        (" 中文 ", "中文"),
        ("ねこ", "ねこ"),
        ("한글", "한글"),
    ],
)
def test_validate_rag_query_accepts_and_strips_valid_queries(query, expected):
    assert validate_rag_query(query) == expected


@pytest.mark.parametrize("query", ["", "   ", "\t\n "])
def test_validate_query_not_empty_rejects_blank_text(query):
    with pytest.raises(EmptyQueryError):
        validate_query_not_empty(query)


@pytest.mark.parametrize("query, expected", [("a", "a"), ("  中  ", "中")])
def test_validate_query_not_empty_allows_any_non_blank_text(query, expected):
    """The non-empty rule is mode-independent; the RAG minimum is not."""
    assert validate_query_not_empty(query) == expected


def test_an_empty_rag_query_is_reported_as_empty_not_as_too_short():
    """The user-facing distinction: nothing typed vs. not enough typed."""
    with pytest.raises(EmptyQueryError):
        validate_rag_query("   ")
    with pytest.raises(RAGQueryTooShortError):
        validate_rag_query("ab")


@pytest.mark.parametrize(
    "query, expected",
    [("", False), ("ab", False), ("中", False), ("abc", True), ("中a", True)],
)
def test_meets_min_rag_query_weight_agrees_with_the_full_walk(query, expected):
    assert meets_min_rag_query_weight(query) is expected
    assert meets_min_rag_query_weight(query) is (rag_query_weight(query) >= 3)


def test_the_threshold_check_does_not_walk_the_whole_query():
    """A 64 KiB query must not cost 65,536 iterations on the event loop.

    `rag_query_weight` walking the full string added ~69 ms of synchronous CPU
    to every max-size RAG request; only the comparison against the minimum is
    ever needed, and that is decided by the third character at the latest.
    """
    consumed = 0

    class _CountingStr(str):
        """Counts how far the validator actually reads."""

        def __iter__(self):
            nonlocal consumed
            for character in str(self):
                consumed += 1
                yield character

        def strip(self):
            return self

    assert meets_min_rag_query_weight(_CountingStr("a" * 65536)) is True
    assert consumed == 3
