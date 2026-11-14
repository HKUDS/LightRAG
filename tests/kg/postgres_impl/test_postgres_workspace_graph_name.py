"""PGGraphStorage must derive a graph name Apache AGE accepts.

The graph name is ``<workspace>_<namespace>`` reduced to ``[A-Za-z0-9_]``. AGE
rejects that name when it is longer than 63 bytes (create_vlabel() raises
"graph name is invalid") or starts with a digit (create_graph() raises the same
error), so a workspace longer than 41 characters, or one starting with a digit,
could never initialize. Those names now fall back to ``ws_<hash>_<namespace>``;
every name that already worked is left unchanged so no existing graph moves.
"""

import pytest

from lightrag.kg.postgres_impl import PGGraphStorage, _is_valid_age_graph_name

pytestmark = pytest.mark.offline

NAMESPACE = "chunk_entity_relation"


def _graph_name(workspace: str) -> str:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = workspace
    storage.namespace = NAMESPACE
    return storage._get_workspace_graph_name()


@pytest.mark.parametrize(
    "workspace, expected",
    [
        ("space1", "space1_chunk_entity_relation"),
        ("_team", "_team_chunk_entity_relation"),
        ("v1.0", "v1_0_chunk_entity_relation"),
        # 41 characters + "_chunk_entity_relation" is exactly 63 bytes.
        ("w" * 41, "w" * 41 + "_chunk_entity_relation"),
        ("", "chunk_entity_relation"),
        ("default", "chunk_entity_relation"),
    ],
)
def test_names_that_already_worked_are_unchanged(workspace, expected):
    assert _graph_name(workspace) == expected


@pytest.mark.parametrize(
    "workspace",
    [
        "w" * 42,
        "acme_corp_customer_support_knowledge_base_production_eu",
        "2026kb",
        "1abc",
    ],
)
def test_invalid_names_fall_back_to_a_valid_hashed_name(workspace):
    name = _graph_name(workspace)

    assert _is_valid_age_graph_name(name), name
    assert name.startswith("ws_")
    assert name.endswith("_chunk_entity_relation")
    assert name.isascii()
    assert name == _graph_name(workspace)  # deterministic


def test_long_workspaces_sharing_a_63_byte_prefix_get_separate_graphs():
    """Clipping to 63 bytes would merge these two workspaces into one graph."""
    prefix = "x" * 63

    assert _graph_name(prefix + "_eu") != _graph_name(prefix + "_us")


def test_digit_workspace_does_not_take_the_graph_of_a_ws_prefixed_one():
    assert _graph_name("2026kb") != _graph_name("ws_2026kb")


@pytest.mark.parametrize(
    "name, valid",
    [
        ("tenant_a_chunk_entity_relation", True),
        ("_team_chunk_entity_relation", True),
        ("a" * 63, True),
        ("a" * 64, False),
        ("2026kb_chunk_entity_relation", False),
    ],
)
def test_is_valid_age_graph_name(name, valid):
    assert _is_valid_age_graph_name(name) is valid
