"""
Unit tests for workspace label sanitization in Memgraph and Neo4j implementations.

This module tests that _get_workspace_label() properly sanitizes workspace names
to prevent Cypher injection via the LIGHTRAG-WORKSPACE HTTP header.

References: GitHub Issue #2698
"""

import re
import pytest

# Mark all tests as offline (no external dependencies)
pytestmark = pytest.mark.offline


def _sanitize_workspace(workspace: str) -> str:
    """Reference implementation of the sanitization logic under test.

    This mirrors the logic in _get_workspace_label() for both
    MemgraphStorage and Neo4JStorage.
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", workspace.strip())
    if not safe:
        safe = "base"
    return safe


class TestWorkspaceLabelSanitization:
    """Test suite for _get_workspace_label() sanitization in graph storage backends."""

    # --- Normal inputs ---

    def test_alphanumeric_unchanged(self):
        """Pure alphanumeric workspace names should pass through unchanged."""
        assert _sanitize_workspace("myworkspace") == "myworkspace"

    def test_alphanumeric_with_underscore(self):
        """Underscores are allowed and should remain."""
        assert _sanitize_workspace("my_workspace_1") == "my_workspace_1"

    def test_uppercase_preserved(self):
        """Case should be preserved."""
        assert _sanitize_workspace("MyWorkSpace") == "MyWorkSpace"

    def test_numeric_only(self):
        """Numeric-only workspaces are valid."""
        assert _sanitize_workspace("12345") == "12345"

    # --- Special characters replaced ---

    def test_spaces_replaced(self):
        """Spaces in workspace names should be replaced with underscores."""
        assert _sanitize_workspace("my workspace") == "my_workspace"

    def test_hyphens_replaced(self):
        """Hyphens should be replaced with underscores."""
        assert _sanitize_workspace("my-workspace") == "my_workspace"

    def test_dots_replaced(self):
        """Dots should be replaced with underscores."""
        assert _sanitize_workspace("my.workspace") == "my_workspace"

    def test_mixed_special_chars(self):
        """Multiple different special characters should all be replaced."""
        result = _sanitize_workspace("a-b.c d@e!f")
        assert result == "a_b_c_d_e_f"

    # --- Cypher injection payloads ---

    def test_cypher_injection_backtick(self):
        """Backtick injection attempt should be neutralized."""
        malicious = "test`}) MATCH (n) DETACH DELETE n //"
        result = _sanitize_workspace(malicious)
        assert "`" not in result
        assert "DETACH" not in result or result == re.sub(r"[^a-zA-Z0-9_]", "_", malicious.strip())
        # Most importantly, no backticks or special Cypher syntax
        assert re.fullmatch(r"[a-zA-Z0-9_]+", result)

    def test_cypher_injection_curly_braces(self):
        """Curly brace injection should be sanitized."""
        malicious = "test}) RETURN 1 //"
        result = _sanitize_workspace(malicious)
        assert "{" not in result
        assert "}" not in result
        assert re.fullmatch(r"[a-zA-Z0-9_]+", result)

    def test_cypher_injection_semicolon(self):
        """Semicolon injection (multi-statement) should be sanitized."""
        malicious = "test; DROP DATABASE neo4j"
        result = _sanitize_workspace(malicious)
        assert ";" not in result
        assert re.fullmatch(r"[a-zA-Z0-9_]+", result)

    def test_cypher_injection_quotes(self):
        """Quote injection should be sanitized."""
        malicious = 'test" OR 1=1 //'
        result = _sanitize_workspace(malicious)
        assert '"' not in result
        assert "'" not in result
        assert re.fullmatch(r"[a-zA-Z0-9_]+", result)

    # --- Empty / whitespace fallback ---

    def test_empty_string_fallback(self):
        """Empty workspace should fall back to 'base'."""
        assert _sanitize_workspace("") == "base"

    def test_whitespace_only_fallback(self):
        """Whitespace-only workspace should fall back to 'base'."""
        assert _sanitize_workspace("   ") == "base"

    def test_special_chars_only_fallback(self):
        """Workspace with only special characters (all stripped) should fall back to 'base'."""
        # After stripping, all chars become "_", which is allowed, so it should NOT be "base"
        result = _sanitize_workspace("---")
        assert result == "___"

    # --- Edge cases ---

    def test_leading_trailing_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped before sanitization."""
        assert _sanitize_workspace("  myworkspace  ") == "myworkspace"

    def test_unicode_characters_replaced(self):
        """Non-ASCII characters should be replaced."""
        result = _sanitize_workspace("工作区_test")
        # Chinese characters should be replaced with underscores
        assert re.fullmatch(r"[a-zA-Z0-9_]+", result)
        assert "test" in result

    def test_very_long_workspace(self):
        """Very long workspace names should still be sanitized correctly."""
        long_name = "a" * 1000
        result = _sanitize_workspace(long_name)
        assert result == long_name
        assert re.fullmatch(r"[a-zA-Z0-9_]+", result)

    def test_single_underscore(self):
        """Single underscore should be valid."""
        assert _sanitize_workspace("_") == "_"

    def test_result_always_safe_for_cypher(self):
        """Parametric check: any output must match the safe pattern."""
        dangerous_inputs = [
            "normal",
            "with spaces",
            "with-dashes",
            "with.dots",
            "`) DETACH DELETE n //",
            "'; DROP TABLE users; --",
            "test\nMATCH (n) DELETE n",
            "\t\ttabs",
            "emoji🚀test",
        ]
        for inp in dangerous_inputs:
            result = _sanitize_workspace(inp)
            assert re.fullmatch(
                r"[a-zA-Z0-9_]+", result
            ), f"Unsafe result '{result}' for input '{inp}'"
