"""Unit tests for ``trim_content_to_budget`` in ``multimodal_context``.

Companion to ``test_multimodal_surrounding_context.py``.  Uses the same
1:1 character-token tokenizer so budgets in each scenario stay readable.
"""

import json
import re

import pytest

from lightrag.multimodal_context import trim_content_to_budget
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


_MARKER_RE = re.compile(
    r"<!-- content truncated from (\d+) to (\d+) tokens, head preserved -->"
)


@pytest.mark.offline
def test_short_content_passes_through():
    tok = _tokenizer()
    content = "<table><tr><td>cell</td></tr></table>"
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=10_000, tokenizer=tok
    )
    assert out == content
    assert was_trimmed is False
    assert _MARKER_RE.search(out) is None


@pytest.mark.offline
def test_table_html_row_trim_keeps_head():
    tok = _tokenizer()
    rows_html = "".join(f"<tr><td>r{i}c0</td><td>r{i}c1</td></tr>" for i in range(10))
    body = f"<tbody>{rows_html}</tbody>"
    content = f'<table id="t-html" format="html">{body}</table>'
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=200, tokenizer=tok
    )
    assert was_trimmed is True
    assert "<table " in out
    # Marker sits outside the </table> wrapper.
    table_close = out.rfind("</table>")
    marker_match = _MARKER_RE.search(out)
    assert marker_match is not None
    assert marker_match.start() > table_close
    # Head rows preserved, tail rows dropped.
    assert "r0c0" in out
    assert "r9c0" not in out
    assert len(tok.encode(out)) <= 200


@pytest.mark.offline
def test_table_json_row_trim_keeps_head():
    tok = _tokenizer()
    rows = [[f"r{i}c0", f"r{i}c1"] for i in range(10)]
    content = '<table id="t-json" format="json">' + json.dumps(rows) + "</table>"
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=150, tokenizer=tok
    )
    assert was_trimmed is True
    assert "<table " in out
    assert "</table>" in out
    # First row preserved, last row dropped.
    assert "r0c0" in out
    assert "r9c0" not in out
    # Marker present and outside </table>.
    table_close = out.rfind("</table>")
    marker_match = _MARKER_RE.search(out)
    assert marker_match is not None
    assert marker_match.start() > table_close
    assert len(tok.encode(out)) <= 150


@pytest.mark.offline
def test_table_char_fallback_when_single_row_oversized():
    tok = _tokenizer()
    # A single huge JSON row whose serialized form alone exceeds budget.
    long_cell = "X" * 400
    content = (
        '<table id="t-big" format="json">'
        + json.dumps([[long_cell]], ensure_ascii=False)
        + "</table>"
    )
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=120, tokenizer=tok
    )
    assert was_trimmed is True
    # <table> wrapper must still be present even after char fallback.
    assert out.lstrip().startswith("<table ")
    assert "</table>" in out
    # Marker still appended outside the wrapper.
    assert _MARKER_RE.search(out) is not None
    assert len(tok.encode(out)) <= 120


@pytest.mark.offline
def test_equation_char_trim_keeps_head():
    tok = _tokenizer()
    content = "HEAD_" + "A" * 500 + "_TAIL"
    out, was_trimmed = trim_content_to_budget(
        content, kind="equations", max_tokens=100, tokenizer=tok
    )
    assert was_trimmed is True
    assert out.startswith("HEAD_")
    # Tail must have been dropped.
    assert "_TAIL" not in out
    assert _MARKER_RE.search(out) is not None
    assert len(tok.encode(out)) <= 100


@pytest.mark.offline
def test_malformed_table_falls_back_to_char_trim():
    tok = _tokenizer()
    # Missing closing </table> tag — TABLE_TAG_RE will reject this, so the
    # generic char-trim path applies (no <table> wrapper reconstruction).
    content = "<table><tr><td>" + "Z" * 500 + "</td></tr>"
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=100, tokenizer=tok
    )
    assert was_trimmed is True
    assert out.startswith("<table>")
    assert _MARKER_RE.search(out) is not None
    assert len(tok.encode(out)) <= 100


@pytest.mark.offline
def test_zero_budget_returns_input_unchanged():
    tok = _tokenizer()
    content = "x" * 5000
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=0, tokenizer=tok
    )
    assert out == content
    assert was_trimmed is False


@pytest.mark.offline
def test_negative_budget_returns_input_unchanged():
    tok = _tokenizer()
    content = "x" * 5000
    out, was_trimmed = trim_content_to_budget(
        content, kind="equations", max_tokens=-10, tokenizer=tok
    )
    assert out == content
    assert was_trimmed is False


@pytest.mark.offline
def test_tokenizer_none_returns_input_unchanged():
    content = "x" * 5000
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=100, tokenizer=None
    )
    assert out == content
    assert was_trimmed is False


@pytest.mark.offline
def test_marker_reports_original_and_final_token_counts():
    tok = _tokenizer()
    content = "x" * 500
    out, was_trimmed = trim_content_to_budget(
        content, kind="equations", max_tokens=100, tokenizer=tok
    )
    assert was_trimmed is True
    match = _MARKER_RE.search(out)
    assert match is not None
    original_in_marker = int(match.group(1))
    final_in_marker = int(match.group(2))
    assert original_in_marker == 500
    # The reported final-token count is the inner-content size (before marker),
    # so it should be strictly less than the original.
    assert final_in_marker < original_in_marker
    assert len(tok.encode(out)) <= 100


@pytest.mark.offline
def test_empty_content_returns_unchanged():
    tok = _tokenizer()
    out, was_trimmed = trim_content_to_budget(
        "", kind="tables", max_tokens=100, tokenizer=tok
    )
    assert out == ""
    assert was_trimmed is False
