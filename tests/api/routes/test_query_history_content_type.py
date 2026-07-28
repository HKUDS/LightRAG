"""QueryRequest conversation_history content must be a string."""

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


def test_conversation_history_rejects_non_string_content():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="hello world",
            conversation_history=[{"role": "user", "content": 123}],
        )


def test_conversation_history_rejects_missing_content():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="hello world",
            conversation_history=[{"role": "user"}],
        )


def test_conversation_history_accepts_string_content():
    req = QueryRequest(
        query="hello world",
        conversation_history=[{"role": "user", "content": "hi"}],
    )
    assert req.conversation_history == [{"role": "user", "content": "hi"}]
