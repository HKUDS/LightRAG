import pytest
from lightrag.base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage
from lightrag.operate import _rebuild_single_entity, _rebuild_single_relationship
from lightrag.utils import Tokenizer, TokenizerInterface


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]):
        return "".join(chr(t) for t in tokens)


class DummyGraph(BaseGraphStorage):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.workspace = ""

    async def get_node(self, node_id: str):
        return self.nodes.get(node_id)

    async def get_edge(self, source_node_id: str, target_node_id: str):
        key = (source_node_id, target_node_id)
        rev_key = (target_node_id, source_node_id)
        return self.edges.get(key) or self.edges.get(rev_key)

    async def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return (source_node_id, target_node_id) in self.edges or (
            target_node_id,
            source_node_id,
        ) in self.edges

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        self.nodes[node_id] = node_data

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict
    ) -> None:
        self.edges[(source_node_id, target_node_id)] = edge_data

    async def get_node_edges(self, source_node_id: str):
        res = []
        for src, tgt in self.edges:
            if src == source_node_id or tgt == source_node_id:
                res.append((src, tgt))
        return res

    async def index_done_callback(self) -> None:
        pass

    async def drop(self):
        return {"status": "success"}

    async def get_all_labels(self):
        return list(self.nodes.keys())

    async def get_popular_labels(self, limit: int = 300):
        return list(self.nodes.keys())[:limit]

    async def search_labels(self, query: str, limit: int = 50):
        return [k for k in self.nodes if query.lower() in k.lower()][:limit]

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ):
        from lightrag.types import KnowledgeGraph

        return KnowledgeGraph()

    async def get_all_nodes(self):
        return [{"id": k, **v} for k, v in self.nodes.items()]

    async def get_all_edges(self):
        return [{"source": k[0], "target": k[1], **v} for k, v in self.edges.items()]

    async def node_degree(self, node_id: str) -> int:
        return len(await self.get_node_edges(node_id))

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    async def remove_nodes(self, nodes: list[str]):
        for n in nodes:
            self.nodes.pop(n, None)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        for src, tgt in edges:
            self.edges.pop((src, tgt), None)
            self.edges.pop((tgt, src), None)

        self.workspace = ""


class DummyVector(BaseVectorStorage):
    def __post_init__(self):
        self.namespace = "test"

    def __init__(self):
        self.data = {}
        self.namespace = "test"

    async def upsert(self, data: dict) -> None:
        self.data.update(data)

    async def query(self, query: str, top_k: int, query_embedding: list[float] = None):
        return []

    async def delete(self, ids: list[str]):
        for i in ids:
            self.data.pop(i, None)

    async def delete_entity(self, entity_name: str) -> None:
        pass

    async def delete_entity_relation(self, entity_name: str) -> None:
        pass

    async def get_by_id(self, id: str):
        return self.data.get(id)

    async def get_by_ids(self, ids: list[str]):
        return [self.data.get(i) for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]):
        return {}

    async def index_done_callback(self) -> None:
        pass

    async def drop(self):
        self.workspace = ""
        return {"status": "success"}


class DummyKV(BaseKVStorage):
    def __init__(self):
        self.data = {}
        self.namespace = "test"

    async def get_by_id(self, id: str):
        return self.data.get(id)

    async def get_by_ids(self, ids: list[str]):
        return [self.data.get(i) for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return {k for k in keys if k not in self.data}

    async def upsert(self, data: dict) -> None:
        self.data.update(data)

    async def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.data.pop(i, None)

    async def is_empty(self) -> bool:
        return len(self.data) == 0

    async def index_done_callback(self) -> None:
        pass

    async def drop(self) -> dict[str, str]:
        return {"status": "success"}


@pytest.mark.asyncio
async def test_rebuild_single_entity_most_common_type():
    graph = DummyGraph()
    entities_vdb = DummyVector()
    llm_cache = DummyKV()
    global_config = {
        "max_source_ids_per_entity": 10,
        "max_file_paths": 10,
        "source_ids_limit_method": "KEEP",
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
    }
    await graph.upsert_node(
        "ALICE",
        {
            "description": "Initial description",
            "entity_type": "UNKNOWN",
            "source_id": "c1",
            "file_path": "f1.txt",
        },
    )

    chunk_entities = {
        "c1": {
            "ALICE": [{"entity_type": "PERSON", "description": "Alice is a person"}]
        },
        "c2": {
            "ALICE": [{"entity_type": "ORGANIZATION", "description": "Alice Org 1"}]
        },
        "c3": {
            "ALICE": [{"entity_type": "ORGANIZATION", "description": "Alice Org 2"}]
        },
        "c4": {
            "ALICE": [{"entity_type": "ORGANIZATION", "description": "Alice Org 3"}]
        },
    }

    await _rebuild_single_entity(
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        entity_name="ALICE",
        chunk_ids=["c1", "c2", "c3", "c4"],
        chunk_entities=chunk_entities,
        llm_response_cache=llm_cache,
        global_config=global_config,
    )

    updated_node = await graph.get_node("ALICE")
    assert updated_node["entity_type"] == "ORGANIZATION"


@pytest.mark.asyncio
async def test_rebuild_single_relationship_keyword_deduplication_and_formatting():
    graph = DummyGraph()
    relationships_vdb = DummyVector()
    entities_vdb = DummyVector()
    llm_cache = DummyKV()
    global_config = {
        "max_source_ids_per_relation": 10,
        "max_file_paths": 10,
        "source_ids_limit_method": "KEEP",
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
    }
    await graph.upsert_node("A", {"entity_type": "ENT", "description": "Node A"})
    await graph.upsert_node("B", {"entity_type": "ENT", "description": "Node B"})
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "Relation AB",
            "keywords": "legacy_kw",
            "weight": 1.0,
            "source_id": "c1",
            "file_path": "f1.txt",
        },
    )

    chunk_relationships = {
        "c1": {
            ("A", "B"): [
                {"keywords": "AI, Machine Learning", "description": "rel desc 1"}
            ]
        },
        "c2": {
            ("A", "B"): [
                {
                    "keywords": "Machine Learning, Deep Learning",
                    "description": "rel desc 2",
                }
            ]
        },
    }

    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=relationships_vdb,
        entities_vdb=entities_vdb,
        src="A",
        tgt="B",
        chunk_ids=["c1", "c2"],
        chunk_relationships=chunk_relationships,
        llm_response_cache=llm_cache,
        global_config=global_config,
    )

    updated_edge = await graph.get_edge("A", "B")
    assert updated_edge["keywords"] == "AI, Deep Learning, Machine Learning"
