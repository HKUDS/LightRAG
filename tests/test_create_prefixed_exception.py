"""Regression tests for create_prefixed_exception.

Some exceptions cannot be reconstructed from their ``args`` alone because
their constructor signatures differ (json.JSONDecodeError needs
``(msg, doc, pos)``; openai SDK ``APIStatusError`` subclasses need keyword-only
``response``/``body``). The helper must never raise while prefixing and must
surface the original type name + message. See HKUDS/LightRAG #2710 and #2794.
"""

import json

import pytest

from lightrag.utils import create_prefixed_exception


@pytest.mark.offline
def test_reconstructable_exception_keeps_type_and_prefix():
    result = create_prefixed_exception(ValueError("boom"), "`entity`")
    assert isinstance(result, ValueError)
    assert str(result) == "`entity`: boom"


@pytest.mark.offline
def test_jsondecodeerror_does_not_raise_and_preserves_message():
    try:
        json.loads('{\n  "x": "abc')  # unterminated string -> JSONDecodeError
    except json.JSONDecodeError as exc:
        result = create_prefixed_exception(exc, "`entity`")
    # JSONDecodeError(msg) is missing (doc, pos): falls back to a clean RuntimeError
    assert isinstance(result, RuntimeError)
    assert result.args  # never an empty/garbled exception
    assert "`entity`" in str(result)
    assert "JSONDecodeError" in str(result)
    assert "Unterminated string" in str(result)


@pytest.mark.offline
def test_keyword_only_constructor_exception_falls_back_cleanly():
    """Mimics openai.APIStatusError: keyword-only required args."""

    class KeywordOnlyError(Exception):
        def __init__(self, message, *, response, body):
            super().__init__(message)

    exc = KeywordOnlyError("Error code: 500", response=object(), body=None)
    result = create_prefixed_exception(exc, "chunk-1")
    assert isinstance(result, RuntimeError)
    assert str(result) == "chunk-1: KeywordOnlyError: Error code: 500"
