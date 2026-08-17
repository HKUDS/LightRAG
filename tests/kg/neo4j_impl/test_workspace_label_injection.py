#!/usr/bin/env python
"""
Unit tests guarding against Cypher injection through the workspace label.

The workspace label is interpolated into two different syntactic contexts in
``neo4j_impl.py``:

1. Backtick-quoted identifiers, e.g. ``MATCH (n:`{label}`)`` — here backticks
   must be doubled and all other characters (including single quotes) are
   literal.
2. The APOC ``labelFilter`` config, which lives inside a single-quoted Cypher
   string literal. Cypher string literals are NOT escaped by doubling quotes
   (that is SQL); they use backslash escaping. Rather than escape by hand, the
   label is bound as a query parameter so Neo4j handles it safely.

These tests run without a live Neo4j instance.
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


# ---------------------------------------------------------------------------
# _get_workspace_label: backtick-identifier context
# ---------------------------------------------------------------------------


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
        ("ws`} RETURN 0", "ws``} RETURN 0"),
        # Single quotes are NOT touched here: inside backticks they are literal,
        # so doubling them would corrupt the label name.
        ("ws'name", "ws'name"),
        ("team'a", "team'a"),
        ("base' OR 1=1", "base' OR 1=1"),
        # Mixed: only backticks are escaped.
        ("ws'`name", "ws'``name"),
    ],
)
def test_get_workspace_label(workspace, expected):
    storage = _make_storage(workspace)
    assert storage._get_workspace_label() == expected


def test_label_safe_in_backtick_identifier():
    """The returned label cannot break out of a backtick-quoted identifier."""
    storage = _make_storage("ws`} OR 1=1")
    identifier = f"`{storage._get_workspace_label()}`"
    # Every ` in the label was doubled to ``, so the quoting stays balanced.
    assert identifier.count("`") % 2 == 0


# ---------------------------------------------------------------------------
# _get_raw_workspace_label: the actual label name, bound as a parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workspace, expected",
    [
        ("my-project", "my-project"),
        ("  trimmed  ", "trimmed"),
        ("", "base"),
        ("   ", "base"),
        # No escaping of any kind: the raw value is bound, never interpolated.
        ("ws`name", "ws`name"),
        ("team'a", "team'a"),
        ("x'); CALL apoc.util.sleep(1);", "x'); CALL apoc.util.sleep(1);"),
    ],
)
def test_get_raw_workspace_label(workspace, expected):
    storage = _make_storage(workspace)
    assert storage._get_raw_workspace_label() == expected


# ---------------------------------------------------------------------------
# get_knowledge_graph: labelFilter must be parameterized, never interpolated
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record

    async def consume(self):
        return None


class _FakeSession:
    """Records every (query, params) pair passed to ``run``."""

    def __init__(self):
        self.calls = []

    async def run(self, query, params=None):
        self.calls.append((query, params or {}))
        # total_nodes within limit -> the full result is used directly and the
        # truncated/limited branch is never taken (a single run call).
        return _FakeResult({"total_nodes": 0, "node_info": [], "relationships": []})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self, **kwargs):
        return self._session


@pytest.mark.asyncio
async def test_label_filter_is_parameterized_not_interpolated():
    """A malicious workspace must reach APOC labelFilter as a bound parameter.

    If the label were string-interpolated (the old behaviour), a single quote
    would close the string literal and inject Cypher. Binding it as a parameter
    makes interpolation — and therefore injection — impossible by construction.
    """
    payload = "x'); CALL apoc.util.sleep(1);"
    storage = _make_storage(payload)
    session = _FakeSession()
    storage._driver = _FakeDriver(session)
    storage._DATABASE = None

    await storage.get_knowledge_graph(node_label="some-entity", max_depth=2)

    # The subgraphAll query is the one carrying labelFilter.
    subgraph_calls = [(q, p) for q, p in session.calls if "apoc.path.subgraphAll" in q]
    assert subgraph_calls, "expected an apoc.path.subgraphAll query"

    for query, params in subgraph_calls:
        # labelFilter is bound, not interpolated: there is no single-quoted
        # string literal for the payload to break out of.
        assert "labelFilter: $label_filter" in query
        assert "labelFilter: '" not in query
        # The raw (un-escaped) label is passed as the bound value.
        assert params.get("label_filter") == payload
