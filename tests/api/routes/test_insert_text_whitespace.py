"""Insert text bodies must stay non-empty after stripping whitespace."""

import importlib
import sys

import pytest
from pydantic import ValidationError

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

InsertTextRequest = _dr.InsertTextRequest
InsertTextsRequest = _dr.InsertTextsRequest

pytestmark = pytest.mark.offline


# One strip() branch; other whitespace only re-tests str.strip().
@pytest.mark.parametrize("text", ["  "])
def test_insert_text_rejects_whitespace_only(text):
    with pytest.raises(ValidationError):
        InsertTextRequest(text=text)


def test_insert_text_strips_surrounding_whitespace():
    req = InsertTextRequest(text="  hello  ")
    assert req.text == "hello"


@pytest.mark.parametrize(
    "texts",
    [
        ["ok", "  "],  # an offender after a valid entry
    ],
)
def test_insert_texts_rejects_whitespace_only_entries(texts):
    with pytest.raises(ValidationError):
        InsertTextsRequest(texts=texts)


def test_insert_texts_strips_surrounding_whitespace():
    req = InsertTextsRequest(texts=["  a  ", " b "])
    assert req.texts == ["a", "b"]
