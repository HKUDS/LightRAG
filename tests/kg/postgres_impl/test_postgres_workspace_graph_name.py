"""PGGraphStorage must derive a graph name Apache AGE accepts.

The graph name is ``<workspace>_<namespace>`` reduced to ``[A-Za-z0-9_]``. AGE
rejects it when it starts with a digit (every version) or, from AGE 1.6, when it
is longer than 63 bytes (create_vlabel() raises "graph name is invalid"). Those
names fall back to ``ws_<hash>_<namespace>``. Up to AGE 1.5 a long name worked,
clipped to 63 bytes, so initialize() keeps using such an existing clipped graph,
unless the workspace already has its hashed graph.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from lightrag.kg.postgres_impl import (
    _INITIALIZED_GRAPHS_LOOKUP,
    PGGraphStorage,
    _is_valid_age_graph_name,
)

pytestmark = pytest.mark.offline

NAMESPACE = "chunk_entity_relation"
GRAPHS_LOOKUP = _INITIALIZED_GRAPHS_LOOKUP


def _storage(workspace: str) -> PGGraphStorage:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = workspace
    storage.namespace = NAMESPACE
    return storage


def _graph_name(workspace: str) -> str:
    return _storage(workspace)._get_workspace_graph_name()


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


def test_hashed_names_follow_the_same_sanitizing_rule_as_short_names():
    """v1.0 and v1_0 share a graph; their long counterparts must too."""
    assert _graph_name("v1.0") == _graph_name("v1_0")
    assert _graph_name("v1.0" + "x" * 50) == _graph_name("v1_0" + "x" * 50)


@pytest.mark.parametrize(
    "workspace, expected",
    [
        ("w" * 42, ("w" * 42 + "_chunk_entity_relation")[:63]),
        ("w" * 41, None),  # fits in 63 bytes
        ("space1", None),
        ("2026kb", None),  # digit-leading never worked on any AGE version
        ("1" + "x" * 60, None),
        ("", None),
        ("default", None),
    ],
)
def test_legacy_clipped_graph_name(workspace, expected):
    assert _storage(workspace)._get_legacy_clipped_graph_name() == expected


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


# ---------------------------------------------------------------------------
# initialize(): resolve a legacy clipped graph before the graph is created
# ---------------------------------------------------------------------------


class _NullLock:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def _initializable_storage(monkeypatch, workspace: str, initialized_graphs=()):
    """A storage whose database reports ``initialized_graphs`` as set up."""
    monkeypatch.setattr(
        "lightrag.kg.postgres_impl.get_data_init_lock", lambda: _NullLock()
    )
    storage = _storage(workspace)
    storage.global_config = {"vector_storage": "PGVectorStorage"}
    calls: list[dict] = []

    async def query(sql, params=None, multirows=False, with_age=False, **kwargs):
        calls.append(
            {
                "sql": sql,
                "params": params,
                "with_age": with_age,
                "graph_name": kwargs.get("graph_name"),
            }
        )
        if sql == GRAPHS_LOOKUP:
            (candidates,) = params
            return [{"name": n} for n in initialized_graphs if n in candidates]
        return [{"name": "base"}, {"name": "DIRECTED"}]

    db = AsyncMock()
    db.workspace = None
    db.query = AsyncMock(side_effect=query)
    db.execute = AsyncMock()
    db._run_with_retry = AsyncMock()
    storage.db = db
    return storage, calls


@pytest.mark.asyncio
async def test_initialize_keeps_an_existing_clipped_graph(monkeypatch):
    """Upgrading from AGE 1.5 or earlier must not move the data to a new graph."""
    warning = Mock()
    monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
    clipped = ("w" * 42 + "_chunk_entity_relation")[:63]
    storage, calls = _initializable_storage(monkeypatch, "w" * 42, [clipped])

    await storage.initialize()

    assert storage.graph_name == clipped
    warning.assert_called_once()
    # Resolved before the first with_age=True call, which would create the graph.
    lookup = next(i for i, c in enumerate(calls) if c["sql"] == GRAPHS_LOOKUP)
    first_age = next(i for i, c in enumerate(calls) if c["with_age"])
    assert lookup < first_age
    assert calls[lookup]["params"] == [[_graph_name("w" * 42), clipped]]
    assert calls[first_age]["graph_name"] == clipped


@pytest.mark.asyncio
async def test_initialize_uses_the_hashed_name_when_no_clipped_graph_exists(
    monkeypatch,
):
    warning = Mock()
    monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
    storage, calls = _initializable_storage(monkeypatch, "w" * 42)

    await storage.initialize()

    assert storage.graph_name == _graph_name("w" * 42)
    assert storage.graph_name.startswith("ws_")
    assert [c for c in calls if c["sql"] == GRAPHS_LOOKUP]
    warning.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_stays_on_the_hashed_graph_when_a_clipped_one_appears(
    monkeypatch,
):
    """An older release run later creates the clipped graph; the data is hashed."""
    warning = Mock()
    monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
    hashed = _graph_name("w" * 42)
    clipped = ("w" * 42 + "_chunk_entity_relation")[:63]
    storage, calls = _initializable_storage(monkeypatch, "w" * 42, [hashed, clipped])

    await storage.initialize()

    assert storage.graph_name == hashed
    first_age = next(c for c in calls if c["with_age"])
    assert first_age["graph_name"] == hashed
    # The unused graph is named, so its data can be found.
    warning.assert_called_once()
    assert hashed in warning.call_args.args[0]
    assert clipped in warning.call_args.args[0]


@pytest.mark.asyncio
async def test_initialize_reopens_its_hashed_graph_without_a_warning(monkeypatch):
    warning = Mock()
    monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
    hashed = _graph_name("w" * 42)
    storage, _ = _initializable_storage(monkeypatch, "w" * 42, [hashed])

    await storage.initialize()

    assert storage.graph_name == hashed
    warning.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["space1", "w" * 41, "2026kb"])
async def test_initialize_skips_the_lookup_when_no_clipped_name_applies(
    monkeypatch, workspace
):
    storage, calls = _initializable_storage(monkeypatch, workspace)

    await storage.initialize()

    assert storage.graph_name == _graph_name(workspace)
    assert not [c for c in calls if c["sql"] == GRAPHS_LOOKUP]
