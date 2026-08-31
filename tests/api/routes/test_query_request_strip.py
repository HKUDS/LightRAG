"""QueryRequest applies the weighted RAG minimum after stripping whitespace."""

import importlib
import sys

import pytest
from pydantic import ValidationError

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_qr = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _qr.QueryRequest

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("query", ["   ", "\t\t\t", "  a  ", "ab ", " a ", "中"])
def test_query_request_rejects_queries_below_weighted_minimum(query):
    with pytest.raises(ValidationError):
        QueryRequest(query=query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("  abc  ", "abc"),
        ("中a", "中a"),
        ("中文", "中文"),
        ("ねこ", "ねこ"),
        ("한글", "한글"),
    ],
)
def test_query_request_accepts_english_equivalent_weight_three(query, expected):
    assert QueryRequest(query=query).query == expected


@pytest.mark.parametrize("query", ["a", "中", " a "])
def test_query_request_does_not_apply_rag_minimum_to_bypass(query):
    assert QueryRequest(query=query, mode="bypass").query == query.strip()


@pytest.mark.parametrize("mode", ["mix", "naive", "bypass"])
@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_query_request_rejects_an_empty_query_in_every_mode(mode, query):
    """`bypass` is exempt from the RAG minimum, not from carrying a prompt.

    Dropping the field's `min_length=3` for the mode-aware minimum also dropped
    the only guard that kept an empty string out of the direct-LLM path.
    """
    with pytest.raises(ValidationError):
        QueryRequest(query=query, mode=mode)
