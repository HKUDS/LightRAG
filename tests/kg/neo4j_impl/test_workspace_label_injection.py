#!/usr/bin/env python
"""
Unit tests for Neo4JStorage._get_workspace_label.

Verifies that the returned label is safe for both backtick-quoted
identifier context (MATCH (n:`{label}`)) and single-quoted string
literal context (labelFilter: '{label}').

These run without a live Neo4j instance.
"""

import os
import sys

import pytest

# Add the project root directory to the Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from lightrag.kg.neo4j_impl import Neo4JStorage


def _make_storage(workspace: str) -> Neo4JStorage:
    """Create a Neo4JStorage with a given workspace (no connection)."""
    return Neo4JStorage(
        namespace="test",
        global_config={},
        embedding_func=None,
        workspace=workspace if workspace else None,
    )


@pytest.mark.parametrize(
    "workspace, expected",
    [
        # Normal names pass through unchanged.
        ("my-project", "my-project"),
        ("base", "base"),
        ("workspace_123", "workspace_123"),
        # Whitespace is stripped; empty falls back to "base".
        ("  trimmed  ", "trimmed"),
        ("", "base"),
        ("   ", "base"),
        # Backticks are doubled for identifier context.
        ("ws`name", "ws``name"),
        ("a`b`c", "a``b``c"),
        # Single quotes are doubled for string-literal context (labelFilter).
        ("ws'name", "ws''name"),
        ("a'b'c", "a''b''c"),
        # Both backticks and single quotes in the same value.
        ("ws'`name", "ws''``name"),
        ("a'b`c", "a''b``c"),
        # Injection attempt: single-quote breakout for labelFilter.
        # (No path separators to pass validate_workspace; the escaping
        #  is defence-in-depth even after the validator.)
        ("base' OR 1=1", "base'' OR 1=1"),
        ("x'); CALL apoc.destroy();", "x''); CALL apoc.destroy();"),
        # Backtick injection attempt for identifier context.
        ("ws`} RETURN 0", "ws``} RETURN 0"),
    ],
)
def test_get_workspace_label(workspace, expected):
    storage = _make_storage(workspace)
    assert storage._get_workspace_label() == expected


def test_label_safe_in_backtick_identifier():
    """The returned label cannot break out of a backtick-quoted identifier."""
    storage = _make_storage("ws`} OR 1=1")
    label = storage._get_workspace_label()
    # Wrapping in backticks should produce a valid identifier with no breakout.
    identifier = f"`{label}`"
    # The identifier should not contain an unescaped backtick pair that ends
    # the quoting — every ` in label was doubled to ``.
    assert identifier.count("`") % 2 == 0


def test_label_safe_in_string_literal():
    """The returned label cannot break out of a single-quoted string literal."""
    storage = _make_storage("ws'); CALL apoc.destroy();")
    label = storage._get_workspace_label()
    # Wrapping in single quotes should produce a valid string with no breakout.
    literal = f"'{label}'"
    # Every ' in label was doubled to '', so the literal stays balanced.
    assert literal.startswith("'")
    assert literal.endswith("'")
    # The interior should have no unescaped single quote that ends the string.
    interior = literal[1:-1]
    # All single quotes in interior should be paired (escaped).
    i = 0
    while i < len(interior):
        if interior[i] == "'":
            assert interior[i + 1] == "'", f"Unescaped single quote at position {i}"
            i += 2
        else:
            i += 1
