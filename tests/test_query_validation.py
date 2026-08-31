import pytest

from lightrag.query_validation import (
    RAGQueryTooShortError,
    rag_query_weight,
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
    ],
)
def test_rag_query_weight_counts_han_as_two(query, expected):
    assert rag_query_weight(query) == expected


@pytest.mark.parametrize("query", ["", "  ", "a", "ab", "中"])
def test_validate_rag_query_rejects_weight_below_three(query):
    with pytest.raises(RAGQueryTooShortError, match="RAG query is too short"):
        validate_rag_query(query)


@pytest.mark.parametrize(
    "query, expected", [("abc", "abc"), ("中a", "中a"), (" 中文 ", "中文")]
)
def test_validate_rag_query_accepts_and_strips_valid_queries(query, expected):
    assert validate_rag_query(query) == expected
