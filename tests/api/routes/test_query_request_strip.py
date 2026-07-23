"""QueryRequest must keep min_length=3 after stripping whitespace."""

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


@pytest.mark.parametrize("query", ["   ", "\t\t\t", "  a  ", "ab ", " a "])
def test_query_request_rejects_whitespace_that_shrinks_below_min_length(query):
    with pytest.raises(ValidationError):
        QueryRequest(query=query)


def test_query_request_strips_surrounding_whitespace_when_still_long_enough():
    req = QueryRequest(query="  abc  ")
    assert req.query == "abc"
