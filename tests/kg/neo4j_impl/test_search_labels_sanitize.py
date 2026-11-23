#!/usr/bin/env python
"""
Unit tests for Neo4JStorage._sanitize_fulltext_query.

These run without a live Neo4j instance and guard the regression where a label
search containing Lucene reserved characters (e.g. "tb-") was misparsed by the
full-text query parser ('-' as NOT) and silently returned nothing.
"""

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Reserved characters are replaced with spaces and whitespace collapsed.
        ("tb-", "tb"),
        ("node:x", "node x"),
        ("a (b)", "a b"),
        ("C++", "C"),
        ("foo && bar", "foo bar"),
        ("name*", "name"),
        # Composed entirely of reserved characters -> empty (caller returns []).
        ("---", ""),
        ("()[]{}", ""),
        # No reserved characters -> unchanged (modulo whitespace collapsing).
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_sanitize_fulltext_query(raw, expected):
    assert Neo4JStorage._sanitize_fulltext_query(raw) == expected
