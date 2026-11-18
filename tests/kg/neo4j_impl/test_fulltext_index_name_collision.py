#!/usr/bin/env python
"""
Regression test: distinct workspace labels must not collide onto the same
Neo4j full-text index name.

``_normalize_index_suffix`` replaces every run of non-alphanumeric
characters with a single underscore. That is lossy: "foo-bar", "foo_bar"
and "foo bar" all normalize to "foo_bar", so two different workspaces
picked the SAME full-text index name from ``_get_fulltext_index_name``.

When the second workspace initializes, ``_create_fulltext_index`` finds an
index with that name already ONLINE (created by the first workspace,
scoped to the first workspace's label via its own backtick-quoted FOR
clause) and skips creating its own. The second workspace's
``search_labels`` then queries that same index name but filters on its own
label, which excludes every node the index actually covers, so the
full-text branch silently returns no matches for that workspace.

These tests run without a live Neo4j instance.
"""

import re

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage

pytestmark = pytest.mark.offline


def _make_storage(workspace: str) -> Neo4JStorage:
    """Create a Neo4JStorage with a given workspace (no connection)."""
    return Neo4JStorage(
        namespace="test",
        global_config={},
        embedding_func=None,
        workspace=workspace,
    )


@pytest.mark.parametrize(
    "workspace_a, workspace_b",
    [
        ("foo-bar", "foo_bar"),
        ("foo bar", "foo_bar"),
        ("team.one", "team_one"),
        ("a--b", "a_b"),
        ("tenant,1", "tenant_1"),
    ],
)
def test_distinct_workspaces_get_distinct_fulltext_index_names(
    workspace_a, workspace_b
):
    storage_a = _make_storage(workspace_a)
    storage_b = _make_storage(workspace_b)

    label_a = storage_a._get_workspace_label()
    label_b = storage_b._get_workspace_label()
    assert label_a != label_b  # sanity: these really are different workspaces

    index_name_a = storage_a._get_fulltext_index_name(label_a)
    index_name_b = storage_b._get_fulltext_index_name(label_b)

    assert index_name_a != index_name_b, (
        f"workspaces {workspace_a!r} and {workspace_b!r} collided onto the "
        f"same full-text index name {index_name_a!r}, so one of them would "
        "silently reuse the other's full-text index"
    )


def test_fulltext_index_name_is_deterministic():
    """The same workspace must always resolve to the same index name."""
    storage = _make_storage("stable-workspace")
    label = storage._get_workspace_label()
    assert storage._get_fulltext_index_name(label) == storage._get_fulltext_index_name(
        label
    )


def test_fulltext_index_name_is_a_valid_identifier():
    """The generated name must still be usable as a bare Cypher identifier."""
    storage = _make_storage("weird,workspace name!")
    label = storage._get_workspace_label()
    index_name = storage._get_fulltext_index_name(label)
    assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", index_name)
