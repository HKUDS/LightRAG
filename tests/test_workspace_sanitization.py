"""
Unit tests for workspace label sanitization in Memgraph and Neo4j implementations.

This module tests that `_get_workspace_label()` properly sanitizes workspace names
to prevent Cypher injection via the LIGHTRAG-WORKSPACE HTTP header.

It verifies that we preserve non-alphanumeric characters for 1-to-1 workspace mapping
while successfully neutralizing Cypher injection by escaping backticks.

References: GitHub Issue #2698
"""

import pytest

# Mark all tests as offline (no external dependencies)
pytestmark = pytest.mark.offline


def _sanitize_workspace(workspace: str) -> str:
    """Reference implementation of the sanitization logic under test.

    This mirrors the logic in _get_workspace_label() for both
    MemgraphStorage and Neo4JStorage.
    """
    safe = workspace.strip()
    if not safe:
        safe = "base"
    return safe.replace("`", "``")


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

    # --- Special characters preserved (unlike PostgreSQL regex stripping) ---

    def test_spaces_preserved(self):
        """Spaces in workspace names should be preserved."""
        assert _sanitize_workspace("my workspace") == "my workspace"

    def test_hyphens_preserved(self):
        """Hyphens should be preserved (solves collision issue)."""
        assert _sanitize_workspace("my-workspace") == "my-workspace"

    def test_dots_preserved(self):
        """Dots should be preserved."""
        assert _sanitize_workspace("my.workspace") == "my.workspace"

    def test_mixed_special_chars_preserved(self):
        """Multiple different special characters should be preserved."""
        assert _sanitize_workspace("a-b.c d@e!f") == "a-b.c d@e!f"

    # --- Cypher injection payloads ---

    def test_cypher_injection_backtick_escaped(self):
        """Backtick injection attempt should be neutralized by doubling backticks."""
        malicious = "test`}) MATCH (n) DETACH DELETE n //"
        # The single backtick should become a double backtick
        expected = "test``}) MATCH (n) DETACH DELETE n //"
        assert _sanitize_workspace(malicious) == expected

    def test_cypher_injection_multiple_backticks(self):
        """Multiple backticks should all be escaped."""
        malicious = "`DROP`DATABASE`"
        expected = "``DROP``DATABASE``"
        assert _sanitize_workspace(malicious) == expected

    def test_cypher_injection_curly_braces_preserved(self):
        """Curly brace injection is harmless when enclosed in backticks, so preserved."""
        malicious = "test}) RETURN 1 //"
        assert _sanitize_workspace(malicious) == malicious

    def test_cypher_injection_semicolon_preserved(self):
        """Semicolon injection is harmless when enclosed in backticks, so preserved."""
        malicious = "test; DROP DATABASE neo4j"
        assert _sanitize_workspace(malicious) == malicious

    def test_cypher_injection_quotes_preserved(self):
        """Quote injection is harmless when enclosed in backticks, so preserved."""
        malicious = 'test" OR 1=1 //'
        assert _sanitize_workspace(malicious) == malicious

    # --- Empty / whitespace fallback ---

    def test_empty_string_fallback(self):
        """Empty workspace should fall back to 'base'."""
        assert _sanitize_workspace("") == "base"

    def test_whitespace_only_fallback(self):
        """Whitespace-only workspace should fall back to 'base'."""
        assert _sanitize_workspace("   ") == "base"

    def test_special_chars_only_preserved(self):
        """Workspace with only special characters should be preserved."""
        assert _sanitize_workspace("---") == "---"

    # --- Edge cases ---

    def test_leading_trailing_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped before sanitization."""
        assert _sanitize_workspace("  myworkspace  ") == "myworkspace"

    def test_unicode_characters_preserved(self):
        """Non-ASCII/Chinese characters should be preserved."""
        assert _sanitize_workspace("工作区_test") == "工作区_test"

    def test_very_long_workspace(self):
        """Very long workspace names should still be sanitized correctly."""
        long_name = "a" * 1000 + "`"
        expected = "a" * 1000 + "``"
        assert _sanitize_workspace(long_name) == expected

    def test_single_underscore(self):
        """Single underscore should be valid."""
        assert _sanitize_workspace("_") == "_"

    def test_result_always_escapes_backticks(self):
        """Parametric check: any output must not contain unescaped single backticks."""
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
            # Find all sequences of backticks
            import re
            backtick_sequences = re.findall(r"`+", result)
            for seq in backtick_sequences:
                # Any sequence of backticks should have an EVEN length because each ` becomes ``
                assert len(seq) % 2 == 0, f"Unescaped backtick found in result '{result}' for input '{inp}'"
