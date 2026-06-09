#!/usr/bin/env python
"""
Unit tests for Neo4JStorage._sanitize_fulltext_query.

These run without a live Neo4j instance and guard the regression where a label
search containing Lucene reserved characters (e.g. "tb-") was misparsed by the
full-text query parser ('-' as NOT) and silently returned nothing.
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


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Reserved characters are replaced with spaces and whitespace collapsed.
        ("tb-", "tb"),
        ("tb-foo", "tb foo"),
        ("node:x", "node x"),
        ("a (b)", "a b"),
        ("C++", "C"),
        ("foo && bar", "foo bar"),
        ("name*", "name"),
        # Composed entirely of reserved characters -> empty (caller returns []).
        ("---", ""),
        ("+++", ""),
        ("()[]{}", ""),
        # No reserved characters -> unchanged (modulo whitespace collapsing).
        ("machine learning", "machine learning"),
        ("  spaced   out  ", "spaced out"),
        ("学习", "学习"),
        ("机器-学习", "机器 学习"),
    ],
)
def test_sanitize_fulltext_query(raw, expected):
    assert Neo4JStorage._sanitize_fulltext_query(raw) == expected
