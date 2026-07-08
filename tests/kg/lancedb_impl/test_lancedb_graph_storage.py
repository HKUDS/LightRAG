"""Unit tests for LanceDBGraphStorage: undirected edges, merges, BFS."""

import time

import pytest

pytest.importorskip("lancedb", reason="lancedb is required for LanceDB storage tests")

from lightrag.kg.lancedb_impl import LanceDBGraphStorage  # noqa: E402
from lightrag.types import KnowledgeGraph  # noqa: E402

pytestmark = pytest.mark.offline


def _make_storage(global_config, embedding_func, workspace="ws"):
    return LanceDBGraphStorage(
        namespace="chunk_entity_relation",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
    )


def _node(name, **extra):
    return {
        "entity_id": name,
        "entity_type": "TEST",
        "description": f"{name} description",
        "source_id": "chunk-1",
        "file_path": "test.txt",
        "created_at": int(time.time()),
        **extra,
    }


def _edge(**extra):
    return {
        "weight": 1.0,
        "description": "test edge",
        "keywords": "test",
        "source_id": "chunk-1",
        "file_path": "test.txt",
        "created_at": int(time.time()),
        **extra,
    }


@pytest.fixture
async def storage(global_config, embedding_func):
    graph = _make_storage(global_config, embedding_func)
    await graph.initialize()
    yield graph
    await graph.finalize()


async def test_node_crud_with_merge_semantics(storage):
    assert not await storage.has_node("Alice")
    assert await storage.get_node("Alice") is None
    await storage.upsert_node("Alice", _node("Alice"))
    assert await storage.has_node("Alice")
    node = await storage.get_node("Alice")
    assert node["entity_type"] == "TEST"
    # Merge: unspecified fields survive, provided fields overwrite.
    await storage.upsert_node("Alice", {"description": "updated"})
    node = await storage.get_node("Alice")
    assert node["description"] == "updated"
    assert node["entity_type"] == "TEST"
    await storage.delete_node("Alice")
    assert not await storage.has_node("Alice")


async def test_edges_are_undirected(storage):
    await storage.upsert_node("A", _node("A"))
    await storage.upsert_node("B", _node("B"))
    await storage.upsert_edge("A", "B", _edge(weight=1.5))
    assert await storage.has_edge("A", "B")
    assert await storage.has_edge("B", "A")
    assert (await storage.get_edge("A", "B"))["weight"] == 1.5
    assert (await storage.get_edge("B", "A"))["weight"] == 1.5
    # Upserting the reverse direction updates the SAME edge (merge).
    await storage.upsert_edge("B", "A", {"weight": 2.5})
    edge = await storage.get_edge("A", "B")
    assert edge["weight"] == 2.5
    assert edge["description"] == "test edge"
    assert await storage.node_degree("A") == 1
    assert await storage.node_degree("B") == 1


async def test_upsert_edge_creates_missing_source_node(storage):
    await storage.upsert_edge("Ghost", "AlsoGhost", _edge())
    assert await storage.has_node("Ghost")
    assert await storage.has_edge("Ghost", "AlsoGhost")


async def test_degrees(storage):
    for name in ("A", "B", "C"):
        await storage.upsert_node(name, _node(name))
    await storage.upsert_edge("A", "B", _edge())
    await storage.upsert_edge("A", "C", _edge())
    assert await storage.node_degree("A") == 2
    assert await storage.node_degree("B") == 1
    assert await storage.node_degree("missing") == 0
    assert await storage.edge_degree("A", "B") == 3
    assert await storage.edge_degree("A", "missing") == 2


async def test_get_node_edges(storage):
    await storage.upsert_node("A", _node("A"))
    await storage.upsert_node("B", _node("B"))
    await storage.upsert_edge("A", "B", _edge())
    edges = await storage.get_node_edges("A")
    assert edges == [("A", "B")]
    assert await storage.get_node_edges("missing") is None
    await storage.upsert_node("Lonely", _node("Lonely"))
    assert await storage.get_node_edges("Lonely") == []


async def test_get_node_edges_puts_queried_node_first(storage):
    """Regression: amerge_entities filters on tuple[0] == entity_name, so the
    queried node must come first even when it was stored as the edge target.
    """
    await storage.upsert_node("Apple", _node("Apple"))
    await storage.upsert_node("Zebra", _node("Zebra"))
    await storage.upsert_edge("Apple", "Zebra", _edge())
    assert await storage.get_node_edges("Zebra") == [("Zebra", "Apple")]
    assert await storage.get_node_edges("Apple") == [("Apple", "Zebra")]
    batch = await storage.get_nodes_edges_batch(["Zebra", "Apple"])
    assert batch["Zebra"] == [("Zebra", "Apple")]
    assert batch["Apple"] == [("Apple", "Zebra")]


async def test_batch_operations(storage):
    await storage.upsert_nodes_batch([(name, _node(name)) for name in "ABCD"])
    await storage.upsert_edges_batch(
        [
            ("A", "B", _edge(weight=1.0)),
            ("B", "C", _edge(weight=2.0)),
            ("C", "A", _edge(weight=3.0)),
        ]
    )
    nodes = await storage.get_nodes_batch(["A", "C", "missing"])
    assert set(nodes) == {"A", "C"}
    assert nodes["A"]["entity_id"] == "A"
    assert await storage.has_nodes_batch(["A", "D", "missing"]) == {"A", "D"}
    degrees = await storage.node_degrees_batch(["A", "B", "D", "missing"])
    assert degrees == {"A": 2, "B": 2, "D": 0, "missing": 0}
    edge_degrees = await storage.edge_degrees_batch([("A", "B"), ("A", "missing")])
    assert edge_degrees[("A", "B")] == 4
    assert edge_degrees[("A", "missing")] == 2
    edges = await storage.get_edges_batch(
        [{"src": "B", "tgt": "A"}, {"src": "B", "tgt": "C"}, {"src": "A", "tgt": "D"}]
    )
    assert edges[("B", "A")]["weight"] == 1.0  # reverse order resolves
    assert edges[("B", "C")]["weight"] == 2.0
    assert ("A", "D") not in edges
    node_edges = await storage.get_nodes_edges_batch(["A", "D"])
    assert len(node_edges["A"]) == 2
    assert node_edges["D"] == []


async def test_edges_batch_dedupes_canonical_pairs(storage):
    await storage.upsert_edges_batch(
        [
            ("A", "B", _edge(weight=1.0, description="first")),
            ("B", "A", _edge(weight=9.0)),
        ]
    )
    edge = await storage.get_edge("A", "B")
    # Later entry wins field-by-field, like sequential upsert_edge calls.
    assert edge["weight"] == 9.0
    assert edge["description"] == "test edge"
    assert await storage.node_degree("A") == 1


async def test_remove_nodes_removes_incident_edges(storage):
    for name in ("A", "B", "C"):
        await storage.upsert_node(name, _node(name))
    await storage.upsert_edge("A", "B", _edge())
    await storage.upsert_edge("B", "C", _edge())
    await storage.remove_nodes(["A"])
    assert not await storage.has_node("A")
    assert not await storage.has_edge("A", "B")
    assert await storage.has_edge("B", "C")


async def test_remove_edges_keeps_nodes(storage):
    await storage.upsert_node("A", _node("A"))
    await storage.upsert_node("B", _node("B"))
    await storage.upsert_edge("A", "B", _edge())
    await storage.remove_edges([("B", "A")])  # reverse order works
    assert not await storage.has_edge("A", "B")
    assert await storage.has_node("A")
    assert await storage.has_node("B")


async def test_labels_and_search(storage):
    for name in ("Apple", "Banana", "apple pie", "Pineapple"):
        await storage.upsert_node(name, _node(name))
    await storage.upsert_edge("Apple", "Banana", _edge())
    await storage.upsert_edge("Apple", "Pineapple", _edge())
    labels = await storage.get_all_labels()
    assert labels == sorted(["Apple", "Banana", "apple pie", "Pineapple"])
    popular = await storage.get_popular_labels(limit=2)
    assert popular[0] == "Apple"
    assert len(popular) == 2
    results = await storage.search_labels("apple")
    assert results[0] in ("Apple", "apple pie")  # exact-insensitive first
    assert "Pineapple" in results
    assert await storage.search_labels("  ") == []
    assert await storage.search_labels("zzz") == []


async def test_get_all_nodes_and_edges(storage):
    await storage.upsert_node("A", _node("A"))
    await storage.upsert_node("B", _node("B"))
    await storage.upsert_edge("A", "B", _edge())
    nodes = await storage.get_all_nodes()
    assert {node["id"] for node in nodes} == {"A", "B"}
    edges = await storage.get_all_edges()
    assert len(edges) == 1
    assert {edges[0]["source"], edges[0]["target"]} == {"A", "B"}
    assert edges[0]["source_id"] == "chunk-1"


async def test_get_knowledge_graph_bfs(storage):
    # A - B - C - D chain plus isolated E
    for name in "ABCDE":
        await storage.upsert_node(name, _node(name))
    await storage.upsert_edge("A", "B", _edge())
    await storage.upsert_edge("B", "C", _edge())
    await storage.upsert_edge("C", "D", _edge())

    kg = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=100)
    assert isinstance(kg, KnowledgeGraph)
    ids = {node.id for node in kg.nodes}
    assert ids == {"A", "B", "C"}  # depth 2 stops before D
    assert not kg.is_truncated
    edge_pairs = {(edge.source, edge.target) for edge in kg.edges}
    assert len(edge_pairs) == 2

    kg_missing = await storage.get_knowledge_graph("missing")
    assert kg_missing.nodes == [] and kg_missing.edges == []

    kg_capped = await storage.get_knowledge_graph("A", max_depth=5, max_nodes=2)
    assert len(kg_capped.nodes) <= 2
    assert kg_capped.is_truncated

    # Exactly max_nodes reachable nodes with nothing omitted -> NOT truncated.
    kg_exact = await storage.get_knowledge_graph("A", max_depth=5, max_nodes=4)
    assert {node.id for node in kg_exact.nodes} == {"A", "B", "C", "D"}
    assert not kg_exact.is_truncated


async def test_get_knowledge_graph_wildcard(storage, global_config):
    for name in "ABCD":
        await storage.upsert_node(name, _node(name))
    await storage.upsert_edge("A", "B", _edge())
    await storage.upsert_edge("A", "C", _edge())

    kg = await storage.get_knowledge_graph("*", max_nodes=100)
    assert {node.id for node in kg.nodes} == {"A", "B", "C", "D"}
    assert len(kg.edges) == 2
    assert not kg.is_truncated

    kg_truncated = await storage.get_knowledge_graph("*", max_nodes=2)
    assert len(kg_truncated.nodes) == 2
    assert kg_truncated.is_truncated
    # Highest-degree node must be included.
    assert "A" in {node.id for node in kg_truncated.nodes}


async def test_special_characters_in_ids(storage):
    tricky = "O'Brien \"The Boss\" 朱元璋"
    other = "partner's node"
    await storage.upsert_node(tricky, _node(tricky))
    await storage.upsert_node(other, _node(other))
    await storage.upsert_edge(tricky, other, _edge())
    assert await storage.has_node(tricky)
    assert (await storage.get_node(tricky))["entity_id"] == tricky
    assert await storage.has_edge(other, tricky)
    assert await storage.node_degree(tricky) == 1
    await storage.remove_edges([(tricky, other)])
    assert not await storage.has_edge(tricky, other)


async def test_drop_resets_graph(storage):
    await storage.upsert_node("A", _node("A"))
    await storage.upsert_edge("A", "B", _edge())
    result = await storage.drop()
    assert result == {"status": "success", "message": "data dropped"}
    assert await storage.get_all_labels() == []
    assert await storage.get_all_edges() == []
    # usable after drop
    await storage.upsert_node("C", _node("C"))
    assert await storage.has_node("C")
