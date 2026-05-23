import pytest

from lightrag.constants import SOURCE_IDS_LIMIT_METHOD_KEEP
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import (
    _handle_single_entity_extraction,
    _merge_nodes_then_upsert,
    _normalize_text_extraction_record_attributes,
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
        self.upserts = []
        self.deletes = []

    async def upsert(self, data):
        self.upserts.append(data)
        return None

    async def delete(self, ids):
        self.deletes.append(ids)
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


class DummyMergeGraphStorage:
    def __init__(self):
        self.nodes = {
            "Canonical": {
                "entity_id": "Canonical",
                "description": "canonical desc",
                "entity_type": "ORG",
                "source_id": "chunk-1",
                "file_path": "canonical.md",
            },
            "Alias": {
                "entity_id": "Alias",
                "description": "alias desc",
                "entity_type": "ORG",
                "source_id": "chunk-2",
                "file_path": "alias.md",
            },
            "Neighbor": {
                "entity_id": "Neighbor",
                "description": "neighbor desc",
                "entity_type": "ORG",
                "source_id": "chunk-3",
                "file_path": "neighbor.md",
            },
        }
        self.edges = {
            ("Alias", "Neighbor"): {
                "description": "rel desc",
                "keywords": "alias",
                "source_id": "chunk-rel",
                "weight": 1.0,
                "file_path": "rel.md",
            }
        }

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        return self.nodes[node_id]

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def get_node_edges(self, node_id):
        results = []
        for src, tgt in self.edges:
            if src == node_id or tgt == node_id:
                results.append((src, tgt))
        return results

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {
            (src, tgt): data
            for (src, tgt), data in self.edges.items()
            if src != node_id and tgt != node_id
        }

    async def index_done_callback(self):
        return True


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


def test_handle_single_relationship_extraction_ignores_empty_description():
    relation = _handle_single_relationship_extraction(
        ["relation", "Alice", "Bob", "works_with", "   "],
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is None


def test_mis_prefixed_relation_row_is_recovered():
    record = _normalize_text_extraction_record_attributes(
        ["entity", "Alice", "Acme Corp", "founded", "Alice founded Acme Corp."],
        chunk_key="chunk-1",
    )

    relation = _handle_single_relationship_extraction(
        record,
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is not None
    assert relation["src_id"] == "Alice"
    assert relation["tgt_id"] == "Acme Corp"


def test_four_part_entity_row_remains_entity():
    record = _normalize_text_extraction_record_attributes(
        ["entity", "Alice", "Person", "Alice is the founder of Acme Corp."],
        chunk_key="chunk-1",
    )

    entity = _handle_single_entity_extraction(
        record,
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert entity is not None
    assert entity["entity_name"] == "Alice"


def test_malformed_recovered_relation_still_fails():
    record = _normalize_text_extraction_record_attributes(
        ["entity", "Alice", "Acme Corp", "founded", "   "],
        chunk_key="chunk-1",
    )

    relation = _handle_single_relationship_extraction(
        record,
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is None


def test_unrelated_five_part_prefix_remains_invalid():
    record = _normalize_text_extraction_record_attributes(
        ["edge", "Alice", "Acme Corp", "founded", "Alice founded Acme Corp."],
        chunk_key="chunk-1",
    )

    relation = _handle_single_relationship_extraction(
        record,
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is None


@pytest.mark.asyncio
async def test_merge_entities_preserves_file_path_in_vector_updates(monkeypatch):
    graph = DummyMergeGraphStorage()
    entities_vdb = DummyVectorStorage()
    relationships_vdb = DummyVectorStorage()

    async def fake_get_entity_info(*args, **kwargs):
        return {"entity_name": "Canonical"}

    monkeypatch.setattr(utils_graph, "get_entity_info", fake_get_entity_info)

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["Alias", "Canonical"],
        target_entity="Canonical",
    )

    relationship_payload = relationships_vdb.upserts[-1]
    entity_payload = entities_vdb.upserts[-1]

    assert next(iter(relationship_payload.values()))["file_path"] == "rel.md"
    assert set(
        next(iter(entity_payload.values()))["file_path"].split(GRAPH_FIELD_SEP)
    ) == {
        "alias.md",
        "canonical.md",
    }
