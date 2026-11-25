"""Entity merge must redirect every incident relation, whichever endpoint
``get_node_edges`` lists first.

``BaseGraphStorage.get_node_edges`` does not promise that the queried node
comes first: NetworkX, Neo4j and PostgreSQL list it first, while MongoDB,
Memgraph and OpenSearch return each edge in its stored direction.
``_merge_entities_impl`` used to keep only tuples whose first element was the
merged entity, so on the stored-direction backends every relation stored
*towards* a source entity was dropped: it was not redirected to the target,
and it disappeared with the source node, leaving its chunk-tracking row behind
under the old key.
"""

import pytest

from lightrag import utils_graph
from lightrag.utils import make_relation_chunk_key

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _Graph:
    """Undirected graph whose edge listing order is configurable.

    ``listing="queried_first"`` mimics NetworkX / Neo4j / PostgreSQL,
    ``listing="stored"`` mimics MongoDB / Memgraph / OpenSearch.
    """

    def __init__(self, listing: str):
        self.listing = listing
        self.nodes = {
            name: {
                "entity_id": name,
                "description": f"{name} desc",
                "entity_type": "PERSON",
                "source_id": f"chunk-{name.lower()}",
                "file_path": f"{name}.txt",
            }
            for name in ("Alice", "Zoe", "Bob", "Dave")
        }
        # Stored direction: Zoe is the TARGET of the first edge and the
        # SOURCE of the second, so each listing order hides one of them from
        # a first-element filter.
        self.edges = {
            ("Alice", "Zoe"): {
                "description": "Alice knows Zoe",
                "keywords": "knows",
                "source_id": "chunk-az",
                "weight": 1.0,
                "file_path": "az.txt",
            },
            ("Zoe", "Dave"): {
                "description": "Zoe meets Dave",
                "keywords": "meets",
                "source_id": "chunk-zd",
                "weight": 1.0,
                "file_path": "zd.txt",
            },
        }

    def _find(self, a, b):
        if (a, b) in self.edges:
            return (a, b)
        if (b, a) in self.edges:
            return (b, a)
        return None

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def get_node_edges(self, node_id):
        if node_id not in self.nodes:
            return None
        listed = []
        for src, tgt in self.edges:
            if node_id not in (src, tgt):
                continue
            if self.listing == "queried_first" and tgt == node_id:
                listed.append((tgt, src))
            else:
                listed.append((src, tgt))
        return listed

    async def get_edge(self, src, tgt):
        key = self._find(src, tgt)
        return dict(self.edges[key]) if key else None

    async def upsert_edge(self, src, tgt, edge_data):
        key = self._find(src, tgt) or (src, tgt)
        self.edges[key] = dict(edge_data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {k: v for k, v in self.edges.items() if node_id not in k}

    async def index_done_callback(self):
        return True


class _Vectors:
    def __init__(self):
        self.global_config = {"workspace": "test"}

    async def upsert(self, data):
        return None

    async def delete(self, ids):
        return None

    async def index_done_callback(self):
        return True


class _KV:
    def __init__(self, data):
        self.data = dict(data)

    async def get_by_id(self, key):
        return self.data.get(key)

    async def upsert(self, data):
        self.data.update(data)

    async def delete(self, keys):
        for key in keys:
            self.data.pop(key, None)

    async def index_done_callback(self):
        return True


@pytest.mark.parametrize("listing", ["queried_first", "stored"])
async def test_merge_redirects_relations_stored_towards_the_source(listing):
    graph = _Graph(listing)
    relation_chunks = _KV(
        {
            make_relation_chunk_key("Alice", "Zoe"): {
                "chunk_ids": ["chunk-az"],
                "count": 1,
            },
            make_relation_chunk_key("Zoe", "Dave"): {
                "chunk_ids": ["chunk-zd"],
                "count": 1,
            },
        }
    )

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=_Vectors(),
        relationships_vdb=_Vectors(),
        source_entities=["Zoe"],
        target_entity="Bob",
        entity_chunks_storage=_KV({}),
        relation_chunks_storage=relation_chunks,
    )

    edges = {
        tuple(sorted(pair)): data["description"] for pair, data in graph.edges.items()
    }
    assert edges == {
        ("Alice", "Bob"): "Alice knows Zoe",
        ("Bob", "Dave"): "Zoe meets Dave",
    }
    assert relation_chunks.data == {
        make_relation_chunk_key("Alice", "Bob"): {
            "chunk_ids": ["chunk-az"],
            "count": 1,
        },
        make_relation_chunk_key("Bob", "Dave"): {
            "chunk_ids": ["chunk-zd"],
            "count": 1,
        },
    }
