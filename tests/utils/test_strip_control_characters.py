"""Unit tests for :func:`lightrag.utils.strip_control_characters`.

The helper removes C0 control / separator characters (notably the
``\\x1c``-``\\x1f`` FS/GS/RS/US separators) and Unicode surrogates while
preserving ``\\t``/``\\n``/``\\r`` and *not* touching markup or surrounding
whitespace. It is the cleaner injected at the parse-result persist points so
parser-extracted body text cannot carry these chars into chunks/storage.
"""

import pytest

from lightrag.utils import strip_control_characters

pytestmark = pytest.mark.offline


def test_removes_fs_gs_rs_us_separators():
    raw = "a\x1cb\x1dc\x1ed\x1fe"
    assert strip_control_characters(raw) == "abcde"


def test_removes_other_c0_and_del():
    # NUL, BS, VT, FF, SO..US range members, and DEL.
    raw = "x\x00\x08\x0b\x0c\x0e\x1f\x7fy"
    assert strip_control_characters(raw) == "xy"


def test_preserves_tab_newline_carriage_return():
    raw = "line1\tcol\nline2\r\nline3"
    # The whitespace control chars survive verbatim.
    assert strip_control_characters(raw) == raw


def test_noop_for_clean_text_returns_unchanged():
    clean = "  普通文本 with <table id='t'>markup</table> &lt;kept&gt;  "
    # No html.unescape, no .strip(): markup, entities and surrounding
    # whitespace are all preserved exactly.
    assert strip_control_characters(clean) == clean


def test_empty_and_falsey_input():
    assert strip_control_characters("") == ""


def test_removes_surrogates():
    raw = "ok\ud800tail"
    assert strip_control_characters(raw) == "oktail"


def test_replacement_char_can_substitute():
    raw = "a\x1fb"
    assert strip_control_characters(raw, replacement_char="_") == "a_b"


def test_cjk_separators_removed_without_inserting_spaces():
    # Critical: deletion (not space-substitution) so CJK text is not split by
    # spurious whitespace that would break the V chunker downstream.
    raw = "中\x1f文\x1c内\x1d容"
    assert strip_control_characters(raw) == "中文内容"
