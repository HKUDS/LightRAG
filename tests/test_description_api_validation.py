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


class DummyRenameGraphStorage:
    def __init__(self, nodes=None):
        self.nodes = dict(nodes or {})
        self.upserted_nodes = []

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        node = self.nodes.get(node_id)
        return dict(node) if node is not None else None

    async def upsert_node(self, node_id, node_data):
        copied = dict(node_data)
        self.upserted_nodes.append((node_id, copied))
        self.nodes[node_id] = copied

    async def get_node_edges(self, source_node_id):
        return []

    async def get_edge(self, source_node_id, target_node_id):
        return None

    async def upsert_edge(self, source_node_id, target_node_id, edge_data):
        return None

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)

    async def index_done_callback(self):
        return None


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


class InMemoryGraphStorage:
    def __init__(self, nodes=None, edges=None):
        self.nodes = {key: dict(value) for key, value in (nodes or {}).items()}
        self.edges = {}
        for (src, tgt), edge_data in (edges or {}).items():
            self.edges[self._normalize_edge(src, tgt)] = dict(edge_data)

    @staticmethod
    def _normalize_edge(source_node_id: str, target_node_id: str) -> tuple[str, str]:
        if source_node_id <= target_node_id:
            return (source_node_id, target_node_id)
        return (target_node_id, source_node_id)

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        node = self.nodes.get(node_id)
        return dict(node) if node is not None else None

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        keys_to_delete = []
        for src, tgt in self.edges:
            if src == node_id or tgt == node_id:
                keys_to_delete.append((src, tgt))
        for edge_key in keys_to_delete:
            self.edges.pop(edge_key, None)

    async def get_node_edges(self, source_node_id):
        return [
            (src, tgt)
            for src, tgt in self.edges
            if src == source_node_id or tgt == source_node_id
        ]

    async def has_edge(self, source_node_id, target_node_id):
        return self._normalize_edge(source_node_id, target_node_id) in self.edges

    async def get_edge(self, source_node_id, target_node_id):
        edge = self.edges.get(self._normalize_edge(source_node_id, target_node_id))
        return dict(edge) if edge is not None else None

    async def upsert_edge(self, source_node_id, target_node_id, edge_data):
        edge_key = self._normalize_edge(source_node_id, target_node_id)
        self.edges[edge_key] = dict(edge_data)

    async def remove_edges(self, edges):
        for source_node_id, target_node_id in edges:
            edge_key = self._normalize_edge(source_node_id, target_node_id)
            self.edges.pop(edge_key, None)

    async def index_done_callback(self):
        return None


class InMemoryVectorStorage:
    def __init__(self):
        self.global_config = {"workspace": "test"}
        self.data = {}

    async def upsert(self, data):
        for key, value in data.items():
            self.data[key] = dict(value) if isinstance(value, dict) else value

    async def delete(self, ids):
        for item_id in ids:
            self.data.pop(item_id, None)

    async def get_by_id(self, id_):
        value = self.data.get(id_)
        return dict(value) if isinstance(value, dict) else value

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


@pytest.mark.asyncio
async def test_edit_entity_rename_syncs_name_to_new_entity_id_by_default():
    graph = DummyRenameGraphStorage(
        nodes={
            "EntityA": {
                "entity_id": "EntityA",
                "name": "EntityA",
                "description": "kept",
                "source_id": "chunk-1",
                "entity_type": "ORG",
            }
        }
    )

    result = await utils_graph._edit_entity_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=DummyVectorStorage(),
        relationships_vdb=DummyVectorStorage(),
        entity_name="EntityA",
        updated_data={"entity_name": "EntityB"},
    )

    assert result["entity_name"] == "EntityB"
    assert graph.upserted_nodes[0][0] == "EntityB"
    assert graph.upserted_nodes[0][1]["entity_id"] == "EntityB"
    assert graph.upserted_nodes[0][1]["name"] == "EntityB"


@pytest.mark.asyncio
async def test_handle_single_relationship_extraction_ignores_empty_description():
    relation = await _handle_single_relationship_extraction(
        ["relation", "Alice", "Bob", "works_with", "   "],
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is None


@pytest.mark.asyncio
async def test_entity_edit_with_stale_revision_token_is_rejected(monkeypatch):
    graph = InMemoryGraphStorage(
        nodes={
            "EntityA": {
                "entity_id": "EntityA",
                "entity_type": "ORG",
                "description": "before",
                "source_id": "chunk-1",
            }
        }
    )
    entities_vdb = InMemoryVectorStorage()
    relationships_vdb = InMemoryVectorStorage()

    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *args, **kwargs: DummyAsyncContext()
    )

    with pytest.raises(ValueError, match="revision token"):
        await utils_graph.aedit_entity(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            entity_name="EntityA",
            updated_data={"description": "after"},
            expected_revision_token="stale-token",
        )


@pytest.mark.asyncio
async def test_relation_edit_with_stale_revision_token_is_rejected(monkeypatch):
    graph = InMemoryGraphStorage(
        nodes={
            "EntityA": {"entity_id": "EntityA", "description": "A", "source_id": "chunk-a"},
            "EntityB": {"entity_id": "EntityB", "description": "B", "source_id": "chunk-b"},
        },
        edges={
            ("EntityA", "EntityB"): {
                "description": "before",
                "keywords": "k1",
                "source_id": "chunk-rel",
                "weight": 1.0,
            }
        },
    )
    entities_vdb = InMemoryVectorStorage()
    relationships_vdb = InMemoryVectorStorage()

    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *args, **kwargs: DummyAsyncContext()
    )

    with pytest.raises(ValueError, match="revision token"):
        await utils_graph.aedit_relation(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            source_entity="EntityA",
            target_entity="EntityB",
            updated_data={"description": "after"},
            expected_revision_token="stale-token",
        )


@pytest.mark.asyncio
async def test_relation_info_token_allows_edit_with_swapped_endpoints(monkeypatch):
    graph = InMemoryGraphStorage(
        nodes={
            "EntityA": {
                "entity_id": "EntityA",
                "description": "A",
                "source_id": "chunk-a",
            },
            "EntityB": {
                "entity_id": "EntityB",
                "description": "B",
                "source_id": "chunk-b",
            },
        },
        edges={
            ("EntityA", "EntityB"): {
                "description": "before",
                "keywords": "k1",
                "source_id": "chunk-rel",
                "weight": 1.0,
            }
        },
    )
    entities_vdb = InMemoryVectorStorage()
    relationships_vdb = InMemoryVectorStorage()

    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: DummyAsyncContext(),
    )

    relation_info = await utils_graph.get_relation_info(
        graph,
        relationships_vdb,
        "EntityB",
        "EntityA",
    )

    result = await utils_graph.aedit_relation(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entity="EntityA",
        target_entity="EntityB",
        updated_data={"description": "after"},
        expected_revision_token=relation_info["revision_token"],
    )

    assert result["graph_data"]["description"] == "after"


@pytest.mark.asyncio
async def test_relation_info_token_allows_delete_with_swapped_endpoints(monkeypatch):
    graph = InMemoryGraphStorage(
        nodes={
            "EntityA": {
                "entity_id": "EntityA",
                "description": "A",
                "source_id": "chunk-a",
            },
            "EntityB": {
                "entity_id": "EntityB",
                "description": "B",
                "source_id": "chunk-b",
            },
        },
        edges={
            ("EntityA", "EntityB"): {
                "description": "before",
                "keywords": "k1",
                "source_id": "chunk-rel",
                "weight": 1.0,
            }
        },
    )
    relationships_vdb = InMemoryVectorStorage()

    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: DummyAsyncContext(),
    )

    relation_info = await utils_graph.get_relation_info(
        graph,
        relationships_vdb,
        "EntityA",
        "EntityB",
    )

    result = await utils_graph.adelete_by_relation(
        chunk_entity_relation_graph=graph,
        relationships_vdb=relationships_vdb,
        source_entity="EntityB",
        target_entity="EntityA",
        expected_revision_token=relation_info["revision_token"],
    )

    assert result.status == "success"
    assert not await graph.has_edge("EntityA", "EntityB")


@pytest.mark.asyncio
async def test_relation_delete_with_stale_revision_token_returns_not_allowed_result(
    monkeypatch,
):
    graph = InMemoryGraphStorage(
        nodes={
            "EntityA": {
                "entity_id": "EntityA",
                "description": "A",
                "source_id": "chunk-a",
            },
            "EntityB": {
                "entity_id": "EntityB",
                "description": "B",
                "source_id": "chunk-b",
            },
        },
        edges={
            ("EntityA", "EntityB"): {
                "description": "before",
                "keywords": "k1",
                "source_id": "chunk-rel",
                "weight": 1.0,
            }
        },
    )
    relationships_vdb = InMemoryVectorStorage()

    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: DummyAsyncContext(),
    )

    result = await utils_graph.adelete_by_relation(
        chunk_entity_relation_graph=graph,
        relationships_vdb=relationships_vdb,
        source_entity="EntityA",
        target_entity="EntityB",
        expected_revision_token="stale-token",
    )

    assert result.status == "not_allowed"
    assert result.status_code == 409
    assert "revision token" in result.message


@pytest.mark.asyncio
async def test_merge_with_stale_revision_token_is_rejected(monkeypatch):
    graph = InMemoryGraphStorage(
        nodes={
            "SourceA": {
                "entity_id": "SourceA",
                "entity_type": "ORG",
                "description": "source-a",
                "source_id": "chunk-a",
            },
            "Target": {
                "entity_id": "Target",
                "entity_type": "ORG",
                "description": "target",
                "source_id": "chunk-t",
            },
        }
    )
    entities_vdb = InMemoryVectorStorage()
    relationships_vdb = InMemoryVectorStorage()

    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *args, **kwargs: DummyAsyncContext()
    )

    with pytest.raises(ValueError, match="revision token"):
        await utils_graph.amerge_entities(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            source_entities=["SourceA"],
            target_entity="Target",
            expected_revision_tokens={"SourceA": "stale-token"},
        )


@pytest.mark.asyncio
async def test_merge_retains_source_entity_names_as_alias_data(monkeypatch):
    graph = InMemoryGraphStorage(
        nodes={
            "SourceA": {
                "entity_id": "SourceA",
                "name": "SourceA Display",
                "entity_type": "ORG",
                "description": "source-a",
                "source_id": "chunk-a",
            },
            "SourceB": {
                "entity_id": "SourceB",
                "name": "SourceB Display",
                "entity_type": "ORG",
                "description": "source-b",
                "source_id": "chunk-b",
            },
            "Target": {
                "entity_id": "Target",
                "name": "Legacy Target Display",
                "entity_type": "ORG",
                "description": "target",
                "source_id": "chunk-t",
                "aliases": ["Legacy Target"],
            },
        }
    )
    entities_vdb = InMemoryVectorStorage()
    relationships_vdb = InMemoryVectorStorage()

    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *args, **kwargs: DummyAsyncContext()
    )

    result = await utils_graph.amerge_entities(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["SourceA", "SourceB"],
        target_entity="Target",
    )

    assert result["entity_name"] == "Target"
    assert result["graph_data"]["entity_id"] == "Target"
    assert result["graph_data"]["name"] == "Target"
    assert set(result["aliases"]) >= {"SourceA", "SourceB", "Legacy Target"}
    assert set(result["graph_data"]["aliases"]) >= {"SourceA", "SourceB", "Legacy Target"}


def test_merge_attributes_join_unique_preserves_first_seen_order():
    merged = utils_graph._merge_attributes(
        [
            {"source_id": "chunk-2<SEP>chunk-1"},
            {"source_id": "chunk-1<SEP>chunk-3"},
        ],
        {"source_id": "join_unique"},
    )

    assert merged["source_id"] == "chunk-2<SEP>chunk-1<SEP>chunk-3"
