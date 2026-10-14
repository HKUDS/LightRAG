"""Business-layer normalization for manual relation mutations.

`acreate_entity`/`aedit_entity`/`amerge_entities` already fold a manually
supplied entity name onto the extraction naming contract
(`normalize_entity_name`) before touching storage. `aedit_relation` and
`adelete_by_relation` looked their endpoints up with the caller's raw
spelling only, so a relation whose endpoints exist under a normalized name
could not be edited or deleted with any spelling variant of that name -- only
the exact normalized string. These tests pin the fix: both lookup paths now
resolve each endpoint the same way `aedit_entity` resolves its subject.
"""

from copy import deepcopy

import pytest

from lightrag import utils_graph

pytestmark = pytest.mark.offline


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Graph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = deepcopy(nodes or {})
        self.edges = deepcopy(edges or {})
        self.removed_edges = []

    async def has_node(self, entity_name):
        return entity_name in self.nodes

    async def has_edge(self, a, b):
        return (a, b) in self.edges or (b, a) in self.edges

    async def get_node(self, entity_name):
        node = self.nodes.get(entity_name)
        return deepcopy(node) if node is not None else None

    async def get_edge(self, a, b):
        edge = self.edges.get((a, b)) or self.edges.get((b, a))
        return deepcopy(edge) if edge is not None else None

    async def upsert_node(self, entity_name, node_data):
        self.nodes[entity_name] = deepcopy(node_data)

    async def upsert_edge(self, a, b, edge_data):
        self.edges[(a, b)] = deepcopy(edge_data)

    async def remove_edges(self, pairs):
        for a, b in pairs:
            self.removed_edges.append((a, b))
            self.edges.pop((a, b), None)
            self.edges.pop((b, a), None)

    async def get_node_edges(self, entity_name):
        return []

    async def index_done_callback(self):
        return None


class _VectorStorage:
    def __init__(self):
        self.global_config = {"workspace": ""}
        self.records = {}

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


@pytest.fixture(autouse=True)
def patch_graph_lock(monkeypatch):
    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: _NoopLock(),
    )


def _seeded_relation():
    """A relation whose endpoints already carry the normalized spelling.

    Mirrors what `acreate_entity` (or automatic extraction) actually stores:
    the fullwidth-paren source name is folded to its half-width form, and
    the relation edge is written against that canonical node.
    """
    graph = _Graph(
        nodes={
            "Tesla(US)": {"entity_id": "Tesla(US)", "entity_type": "organization"},
            "SpaceX": {"entity_id": "SpaceX", "entity_type": "organization"},
        },
        edges={
            ("SpaceX", "Tesla(US)"): {
                "description": "shares a founder",
                "keywords": "founder",
                "source_id": "",
                "weight": 1.0,
                "file_path": "manual_creation",
            }
        },
    )
    return graph


@pytest.mark.asyncio
async def test_edit_relation_resolves_fullwidth_paren_spelling_variant():
    graph = _seeded_relation()
    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()

    # Fullwidth parens -- the reporter's exact raw spelling -- must resolve to
    # the normalized "Tesla(US)" node the relation was actually stored against.
    result = await utils_graph.aedit_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        "Tesla（US）",
        "SpaceX",
        {"description": "shares a founder, updated"},
    )

    assert result["src_entity"] == "SpaceX"
    assert result["tgt_entity"] == "Tesla(US)"
    assert result["graph_data"]["description"] == "shares a founder, updated"


@pytest.mark.asyncio
async def test_edit_relation_resolves_curly_quote_spelling_variant():
    graph = _Graph(
        nodes={
            "Tesla": {"entity_id": "Tesla", "entity_type": "organization"},
            "SpaceX": {"entity_id": "SpaceX", "entity_type": "organization"},
        },
        edges={
            ("SpaceX", "Tesla"): {
                "description": "shares a founder",
                "keywords": "founder",
                "source_id": "",
                "weight": 1.0,
                "file_path": "manual_creation",
            }
        },
    )
    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()

    # Curly/smart quotes normalize away entirely, leaving the plain "Tesla".
    result = await utils_graph.aedit_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        "“Tesla”",
        "SpaceX",
        {"description": "shares a founder, updated"},
    )

    assert result["src_entity"] == "SpaceX"
    assert result["tgt_entity"] == "Tesla"


@pytest.mark.asyncio
async def test_edit_relation_still_refuses_a_spelling_with_no_match():
    graph = _seeded_relation()

    with pytest.raises(ValueError, match="does not exist"):
        await utils_graph.aedit_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "NoSuchCompany",
            "SpaceX",
            {"description": "should not resolve"},
        )


@pytest.mark.asyncio
async def test_delete_relation_resolves_fullwidth_paren_spelling_variant():
    graph = _seeded_relation()
    relationships_vdb = _VectorStorage()

    result = await utils_graph.adelete_by_relation(
        graph,
        relationships_vdb,
        "Tesla（US）",
        "SpaceX",
    )

    assert result.status == "success"
    assert result.status_code == 200
    assert ("SpaceX", "Tesla(US)") in graph.removed_edges
    assert not graph.edges


@pytest.mark.asyncio
async def test_delete_relation_resolves_nbsp_spelling_variant():
    graph = _Graph(
        nodes={
            "Tesla Inc": {"entity_id": "Tesla Inc", "entity_type": "organization"},
            "SpaceX": {"entity_id": "SpaceX", "entity_type": "organization"},
        },
        edges={
            ("SpaceX", "Tesla Inc"): {
                "description": "shares a founder",
                "keywords": "founder",
                "source_id": "",
                "weight": 1.0,
                "file_path": "manual_creation",
            }
        },
    )

    # Non-breaking space between "Tesla" and "Inc" normalizes to a plain space.
    result = await utils_graph.adelete_by_relation(
        graph,
        _VectorStorage(),
        "Tesla Inc",
        "SpaceX",
    )

    assert result.status == "success"
    assert not graph.edges


@pytest.mark.asyncio
async def test_delete_relation_still_reports_not_found_for_no_match():
    graph = _seeded_relation()

    result = await utils_graph.adelete_by_relation(
        graph,
        _VectorStorage(),
        "NoSuchCompany",
        "SpaceX",
    )

    assert result.status == "not_found"
    assert result.status_code == 404


@pytest.mark.asyncio
async def test_edit_relation_prefers_exact_legacy_endpoint_key():
    """A legacy node stored under the pre-normalization spelling must still
    be reachable by that exact spelling, not only by its normalized form."""
    legacy_source = "“Legacy Co”"
    graph = _Graph(
        nodes={
            legacy_source: {"entity_id": legacy_source, "entity_type": "organization"},
            "SpaceX": {"entity_id": "SpaceX", "entity_type": "organization"},
        },
        edges={
            ("SpaceX", legacy_source): {
                "description": "shares a founder",
                "keywords": "founder",
                "source_id": "",
                "weight": 1.0,
                "file_path": "manual_creation",
            }
        },
    )

    result = await utils_graph.aedit_relation(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        legacy_source,
        "SpaceX",
        {"description": "updated"},
    )

    assert result["tgt_entity"] == legacy_source
