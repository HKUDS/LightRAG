"""Unit tests for pure helpers in operate.py (no LLM, no embed)."""
from __future__ import annotations

from framerag.operate import _normalize_frame_name, _context_window, _safe_json


# ── #2: frame-name normalization ─────────────────────────────────────────────

def test_normalize_underscore():
    assert _normalize_frame_name("communication_tell") == "Communication_Tell"


def test_normalize_not_prefix():
    assert _normalize_frame_name("NOT_affect") == "NOT_Affect"


def test_normalize_spaces():
    # The core of bug #2: spaces must become underscores.
    assert _normalize_frame_name("Music performance") == "Music_Performance"


def test_normalize_mixed_separators():
    assert _normalize_frame_name("music - of  spheres") == "Music_Of_Spheres"


def test_normalize_empty():
    assert _normalize_frame_name("") == ""


# ── #5: sentence-aware context window ────────────────────────────────────────

CHUNK = (
    "Holmes lit his pipe. That evening he played his violin to soothe his "
    "nerves while Watson read quietly. The fog pressed against the windows."
)


def test_context_window_returns_whole_sentence():
    span = "played his violin"
    out = _context_window(CHUNK, span, window=10)
    # Even with a tiny char window, the full sentence (incl. all participants)
    # must be returned — not a slice that cuts off "Watson".
    assert "played his violin" in out
    assert "Watson" in out


def test_context_window_does_not_split_midword():
    span = "played his violin"
    out = _context_window(CHUNK, span, window=10).strip(".")
    # Must not start or end in the middle of a token.
    body = out.strip(". ")
    assert not body.startswith("layed")
    assert "violin" in body


def test_context_window_missing_span_falls_back():
    out = _context_window(CHUNK, "nonexistent span here", window=20)
    assert isinstance(out, str) and len(out) > 0


# ── _safe_json robustness ────────────────────────────────────────────────────

def test_safe_json_plain_array():
    assert _safe_json('[{"a": 1}]') == [{"a": 1}]


def test_safe_json_with_fence():
    assert _safe_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_safe_json_with_prefix_text():
    assert _safe_json('Here is the output: [{"a": 1}] done') == [{"a": 1}]


def test_safe_json_garbage_returns_none():
    assert _safe_json("not json at all") is None
