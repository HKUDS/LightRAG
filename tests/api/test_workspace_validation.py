"""Unit tests for validate_workspace_name and get_workspace_from_request.

Covers the rejection-based whitelist validator (§6.2) and the HTTP header
extraction helper (§6.2 ③).  Tests V1–V16 and G1–G6 from the design doc.
"""

import sys
from unittest.mock import MagicMock

# Prevent argparse from seeing pytest flags during config import.
_original_argv = sys.argv.copy()
sys.argv = ["lightrag-server"]

import pytest
from fastapi import HTTPException

from lightrag.api.config import validate_workspace_name
from lightrag.api.utils_api import get_workspace_from_request

sys.argv = _original_argv


# ────────────────────────────────────────────────────────────────
# validate_workspace_name  (V1–V16)
# ────────────────────────────────────────────────────────────────


class TestValidateWorkspaceNameHappy:
    """V1–V4 — valid names pass through unchanged (after strip)."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("projectA", "projectA"),  # V1
            ("v1.0", "v1.0"),  # V2 — dots allowed
            ("项目A", "项目A"),  # V3 — CJK allowed
            ("テスト", "テスト"),  # Japanese kana allowed
            ("my-project", "my-project"),  # hyphens allowed
            ("a", "a"),  # single char
            ("A" * 128, "A" * 128),  # V13 — 128 chars (boundary)
        ],
    )
    def test_valid_names_accepted(self, name, expected):
        assert validate_workspace_name(name) == expected

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("  projectA  ", "projectA"),  # V4 — whitespace stripped
            ("\t\n ws \t", "ws"),  # tabs + newlines stripped
        ],
    )
    def test_whitespace_stripped(self, name, expected):
        assert validate_workspace_name(name) == expected


class TestValidateWorkspaceNameReject:
    """V5–V16 — invalid names raise HTTPException(400)."""

    @pytest.mark.parametrize(
        "name, match_text",
        [
            ("", "empty"),  # V5 — empty string
            ("   ", "empty"),  # V6 — whitespace only
            (".", "Invalid workspace"),  # V7 — dot
            ("..", "Invalid workspace"),  # V8 — double dot
            ("a/b", "Invalid workspace"),  # V9 — slash
            ("a\\b", "Invalid workspace"),  # V10 — backslash
            ("../etc", "Invalid workspace"),  # V11 — path traversal
            ("my project", "invalid characters"),  # V15 — space
            ("name!", "invalid characters"),  # V16 — special char
            ("na#me", "invalid characters"),  # hash
            ("a@b", "invalid characters"),  # at sign
            (
                "\x00null",
                "invalid characters",
            ),  # null byte → caught by name.strip() > 0 or regex
            ("A" * 129, "too long"),  # V14 — 129 chars
        ],
    )
    def test_invalid_names_rejected(self, name, match_text):
        with pytest.raises(HTTPException) as exc_info:
            validate_workspace_name(name)
        assert exc_info.value.status_code == 400
        assert match_text.lower() in str(exc_info.value.detail).lower()

    def test_single_dot_rejected(self):
        """V7 explicit — single '.' is rejected."""
        with pytest.raises(HTTPException) as exc_info:
            validate_workspace_name(".")
        assert exc_info.value.status_code == 400

    def test_double_dot_rejected(self):
        """V8 explicit — '..' is rejected."""
        with pytest.raises(HTTPException) as exc_info:
            validate_workspace_name("..")
        assert exc_info.value.status_code == 400

    def test_dots_inside_name_accepted(self):
        """V12 — 'a..b' (dots inside, not path traversal) is OK."""
        assert validate_workspace_name("a..b") == "a..b"


# ────────────────────────────────────────────────────────────────
# get_workspace_from_request  (G1–G6)
# ────────────────────────────────────────────────────────────────


def _make_request(header_value: str | None) -> MagicMock:
    """Build a mock FastAPI Request with an optional LIGHTRAG-WORKSPACE header."""
    req = MagicMock()
    headers = {}
    if header_value is not None:
        headers["LIGHTRAG-WORKSPACE"] = header_value
    req.headers.get.side_effect = lambda key, default=None: headers.get(key, default)
    return req


class TestGetWorkspaceFromRequest:
    """G1–G6 — header extraction with validation and fallback."""

    def test_header_present_valid(self):
        """G1 — valid header returns the validated name."""
        req = _make_request("projectA")
        result = get_workspace_from_request(req)
        assert result == "projectA"

    def test_header_absent_default_provided(self):
        """G2 — no header, default_workspace set → fallback."""
        req = _make_request(None)
        result = get_workspace_from_request(req, default_workspace="default")
        assert result == "default"

    def test_header_absent_no_default(self):
        """G3 — no header, no default → HTTP 400."""
        req = _make_request(None)
        with pytest.raises(HTTPException) as exc_info:
            get_workspace_from_request(req)
        assert exc_info.value.status_code == 400
        assert "required" in str(exc_info.value.detail).lower()

    def test_header_present_empty(self):
        """G4 — header present but empty (explicitly sent) → 400, NOT fallback."""
        req = _make_request("")
        with pytest.raises(HTTPException) as exc_info:
            get_workspace_from_request(req, default_workspace="default")
        assert exc_info.value.status_code == 400
        assert "empty" in str(exc_info.value.detail).lower()

    def test_header_whitespace_only(self):
        """G5 — whitespace-only header → treated as empty, HTTP 400."""
        req = _make_request("   ")
        with pytest.raises(HTTPException) as exc_info:
            get_workspace_from_request(req, default_workspace="default")
        assert exc_info.value.status_code == 400
        assert "empty" in str(exc_info.value.detail).lower()

    def test_header_invalid_chars(self):
        """G6 — header present with invalid chars → HTTP 400."""
        req = _make_request("my project")
        with pytest.raises(HTTPException) as exc_info:
            get_workspace_from_request(req)
        assert exc_info.value.status_code == 400
        assert "invalid" in str(exc_info.value.detail).lower()
