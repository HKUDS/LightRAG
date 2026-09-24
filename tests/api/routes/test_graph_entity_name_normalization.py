"""Business-layer normalization for manual entity mutations."""

from copy import deepcopy

import pytest

from lightrag import utils_graph
from lightrag.utils import compute_mdhash_id

pytestmark = pytest.mark.offline


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Graph:
    def __init__(self, nodes=None):
        self.nodes = deepcopy(nodes or {})
        self.edges = {}
        self.deleted_nodes = []

    async def has_node(self, entity_name):
        return entity_name in self.nodes

    async def get_node(self, entity_name):
        node = self.nodes.get(entity_name)
        return deepcopy(node) if node is not None else None

    async def upsert_node(self, entity_name, node_data):
        self.nodes[entity_name] = deepcopy(node_data)

    async def get_node_edges(self, entity_name):
        return []

    async def has_edge(self, source_entity, target_entity):
        # The default backend is an undirected ``nx.Graph``.
        return (source_entity, target_entity) in self.edges or (
            target_entity,
            source_entity,
        ) in self.edges

    async def get_edge(self, source_entity, target_entity):
        edge = self.edges.get((source_entity, target_entity))
        if edge is None:
            edge = self.edges.get((target_entity, source_entity))
        return deepcopy(edge) if edge is not None else None

    async def upsert_edge(self, source_entity, target_entity, edge_data):
        self.edges[(source_entity, target_entity)] = deepcopy(edge_data)

    async def delete_node(self, entity_name):
        self.deleted_nodes.append(entity_name)
        self.nodes.pop(entity_name, None)

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


@pytest.mark.asyncio
async def test_create_entity_uses_extraction_name_normalization():
    graph = _Graph()
    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()

    result = await utils_graph.acreate_entity(
        graph,
        entities_vdb,
        relationships_vdb,
        "  “Ａ 公 司”  ",
        {"description": "Company description", "entity_type": "organization"},
    )

    assert result["entity_name"] == "A公司"
    assert set(graph.nodes) == {"A公司"}
    entity_id = compute_mdhash_id("A公司", prefix="ent-")
    assert entities_vdb.records[entity_id]["entity_name"] == "A公司"


@pytest.mark.asyncio
async def test_create_entity_rejects_name_removed_by_normalization():
    with pytest.raises(ValueError, match="empty after normalization"):
        await utils_graph.acreate_entity(
            _Graph(),
            _VectorStorage(),
            _VectorStorage(),
            "1",
            {"description": "Invalid numeric identifier"},
        )


@pytest.mark.asyncio
async def test_edit_resolves_normalized_source_and_normalizes_rename_target():
    graph = _Graph(
        {
            "Source公司": {
                "entity_id": "Source公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            }
        }
    )
    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()
    updated_data = {"entity_name": " “Ｔ 目 标” ", "description": "renamed"}

    result = await utils_graph.aedit_entity(
        graph,
        entities_vdb,
        relationships_vdb,
        "Ｓｏｕｒｃｅ 公 司",
        updated_data,
        allow_rename=True,
    )

    assert updated_data["entity_name"] == " “Ｔ 目 标” "
    assert "Source公司" not in graph.nodes
    assert graph.nodes["T目标"]["entity_id"] == "T目标"
    assert result["entity_name"] == "T目标"
    assert result["operation_summary"]["final_entity"] == "T目标"
    assert result["operation_summary"]["renamed"] is True


@pytest.mark.asyncio
async def test_edit_prefers_exact_legacy_entity_key():
    legacy_name = "“Ａ 公 司”"
    graph = _Graph(
        {
            legacy_name: {
                "entity_id": legacy_name,
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            }
        }
    )

    result = await utils_graph.aedit_entity(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        legacy_name,
        {"description": "updated"},
        allow_rename=False,
    )

    assert set(graph.nodes) == {legacy_name}
    assert graph.nodes[legacy_name]["description"] == "updated"
    assert result["entity_name"] == legacy_name


@pytest.mark.asyncio
async def test_edit_preserves_exact_legacy_name_that_normalizes_to_empty():
    legacy_name = "1"
    graph = _Graph(
        {
            legacy_name: {
                "entity_id": legacy_name,
                "description": "old",
                "source_id": "manual_creation",
            }
        }
    )

    result = await utils_graph.aedit_entity(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        legacy_name,
        {"description": "updated"},
        allow_rename=False,
    )

    assert graph.nodes[legacy_name]["description"] == "updated"
    assert result["entity_name"] == legacy_name


@pytest.mark.asyncio
async def test_create_refuses_duplicate_exact_legacy_key():
    legacy_name = "“Ａ 公 司”"
    graph = _Graph(
        {
            legacy_name: {
                "entity_id": legacy_name,
                "description": "legacy",
            }
        }
    )

    with pytest.raises(ValueError, match="already exists"):
        await utils_graph.acreate_entity(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            legacy_name,
            {"description": "duplicate"},
        )

    assert set(graph.nodes) == {legacy_name}


@pytest.mark.asyncio
async def test_merge_normalizes_sources_and_creates_normalized_target_once():
    graph = _Graph(
        {
            "Source公司": {
                "entity_id": "Source公司",
                "description": "source",
                "entity_type": "organization",
                "source_id": "manual_creation",
            }
        }
    )
    entities_vdb = _VectorStorage()

    result = await utils_graph.amerge_entities(
        graph,
        entities_vdb,
        _VectorStorage(),
        ["Ｓｏｕｒｃｅ 公 司", "Source公司"],
        " “Ｔ 目 标” ",
    )

    assert result["entity_name"] == "T目标"
    assert set(graph.nodes) == {"T目标"}
    assert graph.nodes["T目标"]["entity_id"] == "T目标"
    assert graph.deleted_nodes == ["Source公司"]
    target_id = compute_mdhash_id("T目标", prefix="ent-")
    assert entities_vdb.records[target_id]["entity_name"] == "T目标"


@pytest.mark.asyncio
async def test_merge_preserves_exact_legacy_source_and_target_keys():
    legacy_source = "1"
    legacy_target = "“Ａ 公 司”"
    graph = _Graph(
        {
            legacy_source: {
                "entity_id": legacy_source,
                "description": "source",
                "source_id": "manual_creation",
            },
            legacy_target: {
                "entity_id": legacy_target,
                "description": "target",
                "source_id": "manual_creation",
            },
        }
    )

    result = await utils_graph.amerge_entities(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        [legacy_source],
        legacy_target,
    )

    assert result["entity_name"] == legacy_target
    assert set(graph.nodes) == {legacy_target}
    assert graph.deleted_nodes == [legacy_source]


@pytest.mark.asyncio
async def test_create_relation_accepts_the_name_create_entity_normalized():
    """The same spelling must work on both manual create endpoints.

    ``POST /graphs/entity`` stores the extraction-normalized identifier and its
    response does not carry the stored spelling back, so a caller that then
    posts the very same name to ``POST /graphs/relation`` has no way to learn
    the canonical one. Asking the existence check for the raw spelling refused
    the relation with "Target entity ... does not exist".
    """
    graph = _Graph()
    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()

    await utils_graph.acreate_entity(
        graph,
        entities_vdb,
        relationships_vdb,
        "GoodA",
        {"description": "supplier", "entity_type": "organization"},
    )
    await utils_graph.acreate_entity(
        graph,
        entities_vdb,
        relationships_vdb,
        "Tesla（US）",
        {"description": "electric vehicles", "entity_type": "organization"},
    )

    assert set(graph.nodes) == {"GoodA", "Tesla(US)"}

    result = await utils_graph.acreate_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        "GoodA",
        "Tesla（US）",
        {"description": "supplies"},
    )

    assert set(graph.edges) == {("GoodA", "Tesla(US)")}
    assert {result["src_entity"], result["tgt_entity"]} == {"GoodA", "Tesla(US)"}


@pytest.mark.asyncio
async def test_create_relation_accepts_fully_normalized_spellings():
    """Both endpoints may need resolving, not just the target."""
    graph = _Graph(
        {
            "A公司": {
                "entity_id": "A公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
            "B公司": {
                "entity_id": "B公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
        }
    )

    result = await utils_graph.acreate_relation(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        " “Ｂ 公 司” ",
        "  “Ａ 公 司”  ",
        {"description": "B supplies A"},
    )

    assert set(graph.edges) == {("B公司", "A公司")}
    assert {result["src_entity"], result["tgt_entity"]} == {"A公司", "B公司"}


@pytest.mark.asyncio
async def test_create_relation_prefers_exact_legacy_endpoint_key():
    """A node written before normalization existed keeps its own key.

    Resolving to the canonical spelling first would make a relation to a
    historical node fail, which is what the same preference in ``aedit_entity``
    and ``amerge_entities`` prevents.
    """
    legacy_name = "“Ａ 公 司”"
    graph = _Graph(
        {
            legacy_name: {
                "entity_id": legacy_name,
                "description": "legacy",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
            "B公司": {
                "entity_id": "B公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
        }
    )

    result = await utils_graph.acreate_relation(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        "B公司",
        legacy_name,
        {"description": "legacy endpoint"},
    )

    assert set(graph.nodes) == {legacy_name, "B公司"}
    assert set(graph.edges) == {("B公司", legacy_name)}
    assert {result["src_entity"], result["tgt_entity"]} == {"B公司", legacy_name}


@pytest.mark.asyncio
async def test_create_relation_rejects_endpoints_that_collapse_into_one_node():
    """Two spellings of one node are a self-loop once resolved."""
    graph = _Graph(
        {
            "Tesla(US)": {
                "entity_id": "Tesla(US)",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            }
        }
    )

    with pytest.raises(ValueError, match=r"self-loop relation on 'Tesla\(US\)'"):
        await utils_graph.acreate_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "Tesla（US）",
            "Tesla(US)",
            {"description": "would be a self-loop"},
        )

    assert graph.edges == {}


@pytest.mark.asyncio
async def test_create_relation_prefers_exact_legacy_source_key():
    """The source side resolves exactly like the target side."""
    legacy_name = "“Ａ 公 司”"
    graph = _Graph(
        {
            legacy_name: {
                "entity_id": legacy_name,
                "description": "legacy",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
            "B公司": {
                "entity_id": "B公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
        }
    )

    result = await utils_graph.acreate_relation(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        legacy_name,
        "B公司",
        {"description": "legacy endpoint"},
    )

    assert set(graph.nodes) == {legacy_name, "B公司"}
    assert set(graph.edges) == {(legacy_name, "B公司")}
    assert {result["src_entity"], result["tgt_entity"]} == {legacy_name, "B公司"}


@pytest.mark.asyncio
async def test_create_relation_rejects_endpoints_removed_by_normalization():
    """A name normalization erases leaves no identifier to look up."""
    graph = _Graph(
        {
            "B公司": {
                "entity_id": "B公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            }
        }
    )

    with pytest.raises(ValueError, match="Source entity name cannot be empty"):
        await utils_graph.acreate_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "1",
            "B公司",
            {"description": "erased source"},
        )

    with pytest.raises(ValueError, match="Target entity name cannot be empty"):
        await utils_graph.acreate_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "B公司",
            "1",
            {"description": "erased target"},
        )

    assert graph.edges == {}


@pytest.mark.asyncio
async def test_create_relation_sees_one_edge_through_either_spelling():
    """Resolved endpoints mean the second call is a duplicate, not a new edge."""
    graph = _Graph()
    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()

    for name in ("GoodA", "Tesla（US）"):
        await utils_graph.acreate_entity(
            graph,
            entities_vdb,
            relationships_vdb,
            name,
            {"description": "d", "entity_type": "organization"},
        )

    await utils_graph.acreate_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        "GoodA",
        "Tesla（US）",
        {"description": "supplies"},
    )

    with pytest.raises(ValueError, match="already exists"):
        await utils_graph.acreate_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "GoodA",
            "Tesla(US)",
            {"description": "supplies again"},
        )

    assert set(graph.edges) == {("GoodA", "Tesla(US)")}


@pytest.mark.asyncio
async def test_create_relation_locks_every_spelling(monkeypatch):
    """Both spellings are locked, so the pipeline's canonical edge lock in the
    same namespace excludes this write."""
    captured = []

    class _SpyLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _spy(keys, **kwargs):
        captured.append((list(keys), kwargs.get("namespace")))
        return _SpyLock()

    monkeypatch.setattr(utils_graph, "get_storage_keyed_lock", _spy)

    graph = _Graph(
        {
            "A公司": {
                "entity_id": "A公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
            "B公司": {
                "entity_id": "B公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            },
        }
    )

    await utils_graph.acreate_relation(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        "Ｂ公司",
        "Ａ公司",
        {"description": "d"},
    )

    assert captured == [(["A公司", "B公司", "Ａ公司", "Ｂ公司"], "GraphDB")]
