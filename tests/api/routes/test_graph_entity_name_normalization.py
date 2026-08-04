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
