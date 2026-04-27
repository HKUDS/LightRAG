"""
Unit tests for workspace label sanitization in graph implementations.

Memgraph labels are used in Cypher query strings where labels cannot be
parameterized, so the Memgraph helper must reduce workspace names to an
alphanumeric/underscore allowlist.

Neo4j keeps the existing backtick-escaping behavior to preserve one-to-one
workspace labels while preventing backtick-delimited identifier breakout.
"""

import os
import re

import pytest

# Mark all tests as offline (no external dependencies)
pytestmark = pytest.mark.offline


def _source_contains(file_name: str, pattern: str) -> bool:
    """Check that tests mirror the implementation currently in source."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, "lightrag/kg", file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return re.search(pattern, f.read()) is not None


def sanitize_memgraph(workspace: str) -> str:
    """Mirror MemgraphStorage._get_workspace_label() for dependency-free tests."""
    if not _source_contains(
        "memgraph_impl.py",
        r"re\.sub\(r[\"']\[\^A-Za-z0-9_\][\"'], [\"']_[\"'], workspace\)",
    ):
        raise RuntimeError("Could not find Memgraph allowlist sanitization logic")

    safe = workspace.strip()
    if not safe:
        return "base"
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", safe)
    if not sanitized:
        return "base"
    return sanitized


def sanitize_neo4j(workspace: str) -> str:
    """Mirror Neo4jStorage._get_workspace_label() for dependency-free tests."""
    if not _source_contains(
        "neo4j_impl.py", r"return workspace\.replace\(\"`\", \"``\"\)"
    ):
        raise RuntimeError("Could not find Neo4j backtick escaping logic")

    safe = workspace.strip()
    if not safe:
        return "base"
    return safe.replace("`", "``")


class TestMemgraphWorkspaceLabelSanitization:
    """Test suite for MemgraphStorage._get_workspace_label()."""

    def assert_logic(self, workspace: str, expected: str):
        """Helper to assert Memgraph sanitization logic."""
        assert sanitize_memgraph(workspace) == expected

    # --- Normal inputs ---

    def test_alphanumeric_unchanged(self):
        """Pure alphanumeric workspace names should pass through unchanged."""
        self.assert_logic("myworkspace", "myworkspace")

    def test_alphanumeric_with_underscore(self):
        """Underscores are allowed and should remain."""
        self.assert_logic("my_workspace_1", "my_workspace_1")

    def test_uppercase_preserved(self):
        """Case should be preserved."""
        self.assert_logic("MyWorkSpace", "MyWorkSpace")

    def test_numeric_only(self):
        """Numeric-only workspaces are valid."""
        self.assert_logic("12345", "12345")

    # --- Special characters replaced ---

    def test_spaces_replaced(self):
        """Spaces in workspace names should be replaced."""
        self.assert_logic("my workspace", "my_workspace")

    def test_hyphens_replaced(self):
        """Hyphens should be replaced."""
        self.assert_logic("my-workspace", "my_workspace")

    def test_dots_replaced(self):
        """Dots should be replaced."""
        self.assert_logic("my.workspace", "my_workspace")

    def test_mixed_special_chars_replaced(self):
        """Multiple different special characters should be replaced."""
        self.assert_logic("a-b.c d@e!f", "a_b_c_d_e_f")

    # --- Cypher injection payloads ---

    def test_cypher_injection_backtick_replaced(self):
        """Backtick injection attempt should be neutralized by replacement."""
        malicious = "test`}) MATCH (n) DETACH DELETE n //"
        expected = "test____MATCH__n__DETACH_DELETE_n___"
        self.assert_logic(malicious, expected)

    def test_cypher_injection_multiple_backticks(self):
        """Multiple backticks should all be replaced."""
        malicious = "`DROP`DATABASE`"
        expected = "_DROP_DATABASE_"
        self.assert_logic(malicious, expected)

    def test_cypher_injection_curly_braces_replaced(self):
        """Curly brace injection should be replaced for identifier safety."""
        malicious = "test}) RETURN 1 //"
        self.assert_logic(malicious, "test___RETURN_1___")

    def test_cypher_injection_semicolon_replaced(self):
        """Semicolon injection should be replaced for identifier safety."""
        malicious = "test; DROP DATABASE neo4j"
        self.assert_logic(malicious, "test__DROP_DATABASE_neo4j")

    def test_cypher_injection_quotes_replaced(self):
        """Quote injection should be replaced for identifier safety."""
        malicious = 'test" OR 1=1 //'
        self.assert_logic(malicious, "test__OR_1_1___")

    # --- Empty / whitespace fallback ---

    def test_empty_string_fallback(self):
        """Empty workspace should fall back to 'base'."""
        self.assert_logic("", "base")

    def test_whitespace_only_fallback(self):
        """Whitespace-only workspace should fall back to 'base'."""
        self.assert_logic("   ", "base")

    def test_special_chars_only_replaced(self):
        """Workspace with only special characters should be replaced with underscores."""
        self.assert_logic("---", "___")

    # --- Edge cases ---

    def test_leading_trailing_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped before sanitization."""
        self.assert_logic("  myworkspace  ", "myworkspace")

    def test_unicode_characters_replaced(self):
        """Non-ASCII/Chinese characters should be replaced."""
        self.assert_logic("工作区_test", "____test")

    def test_very_long_workspace(self):
        """Very long workspace names should still be sanitized correctly."""
        long_name = "a" * 1000 + "`"
        expected = "a" * 1000 + "_"
        self.assert_logic(long_name, expected)

    def test_single_underscore(self):
        """Single underscore should be valid."""
        self.assert_logic("_", "_")

    def test_result_uses_strict_allowlist(self):
        """Parametric check: output contains only alphanumerics and underscores."""
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
            result = sanitize_memgraph(inp)
            assert re.fullmatch(r"[A-Za-z0-9_]+", result)


class TestNeo4jWorkspaceLabelSanitization:
    """Test suite for Neo4jStorage._get_workspace_label()."""

    def test_neo4j_preserves_non_backtick_characters(self):
        assert sanitize_neo4j("my workspace") == "my workspace"
        assert sanitize_neo4j("my-workspace") == "my-workspace"
        assert sanitize_neo4j("工作区_test") == "工作区_test"

    def test_neo4j_escapes_backticks_and_falls_back(self):
        assert sanitize_neo4j("test`}) MATCH (n) DETACH DELETE n //") == (
            "test``}) MATCH (n) DETACH DELETE n //"
        )
        assert sanitize_neo4j("   ") == "base"
