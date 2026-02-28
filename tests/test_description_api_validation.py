import pytest

from lightrag.constants import SOURCE_IDS_LIMIT_METHOD_KEEP
from lightrag.operate import (
    _merge_nodes_then_upsert,
    _handle_single_relationship_extraction,
)
from lightrag import utils_graph


class DummyGraphStorage:
    def __init__(self, node=None):
        self.node = node
        self.upserted_nodes = []

    async def get_node(self, node_id):
        return self.node

    async def upsert_node(self, node_id, node_data):
        self.upserted_nodes.append((node_id, node_data))
        self.node = dict(node_data)


class DummyVectorStorage:
    def __init__(self):
        self.global_config = {"workspace": "test"}

    async def upsert(self, data):
        return None

    async def delete(self, ids):
        return None

    async def get_by_id(self, id_):
        return None

    async def index_done_callback(self):
        return True


class DummyAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_merge_nodes_then_upsert_handles_missing_legacy_description():
    graph = DummyGraphStorage(node={"source_id": "chunk-1"})
    global_config = {
        "source_ids_limit_method": SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 20,
    }

    result = await _merge_nodes_then_upsert(
        entity_name="LegacyEntity",
        nodes_data=[],
        knowledge_graph_inst=graph,
        entity_vdb=None,
        global_config=global_config,
    )

    assert result["description"] == "Entity LegacyEntity"
    assert graph.upserted_nodes[-1][1]["description"] == "Entity LegacyEntity"


@pytest.mark.asyncio
async def test_acreate_entity_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.acreate_entity(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            entity_name="EntityA",
            entity_data={"description": "   "},
        )


@pytest.mark.asyncio
async def test_acreate_relation_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.acreate_relation(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            source_entity="A",
            target_entity="B",
            relation_data={"description": ""},
        )


@pytest.mark.asyncio
async def test_aedit_entity_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.aedit_entity(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            entity_name="EntityA",
            updated_data={"description": None},
        )


@pytest.mark.asyncio
async def test_aedit_relation_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.aedit_relation(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            source_entity="A",
            target_entity="B",
            updated_data={"description": "   "},
        )


@pytest.mark.asyncio
async def test_aedit_entity_allows_updates_without_description(monkeypatch):
    async def fake_edit_impl(*args, **kwargs):
        return {"entity_name": "EntityA", "description": "kept", "source_id": "chunk-1"}

    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *a, **k: DummyAsyncContext()
    )
    monkeypatch.setattr(utils_graph, "_edit_entity_impl", fake_edit_impl)

    result = await utils_graph.aedit_entity(
        chunk_entity_relation_graph=None,
        entities_vdb=DummyVectorStorage(),
        relationships_vdb=DummyVectorStorage(),
        entity_name="EntityA",
        updated_data={"entity_type": "ORG"},
    )

    assert result["operation_summary"]["operation_status"] == "success"


@pytest.mark.asyncio
async def test_handle_single_relationship_extraction_ignores_empty_description():
    relation = await _handle_single_relationship_extraction(
        ["relation", "Alice", "Bob", "works_with", "   "],
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is None
