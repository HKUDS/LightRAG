"""``get_knowledge_graph(label)`` must rank each BFS level by degree, then id.

The wildcard path already ranked that way; the BFS path admitted a level in the
order the per-level edge search returned its hits, which is ingestion order. The
``max_nodes`` cutoff normally lands inside a band of equal-degree leaves, so that
decided both which neighbours the caller saw and which ones were expanded next.

Unlike the Neo4j and Memgraph tie-break tests next door, the ordering here is
client-side Python, so this asserts the returned nodes rather than a query
string. See issue #3612.
"""

from collections import Counter
from unittest.mock import AsyncMock

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearch-py is required for OpenSearch storage tests",
)

from lightrag.kg.opensearch_impl import OpenSearchGraphStorage  # noqa: E402


pytestmark = pytest.mark.offline


# A three-leaf star whose leaves are deliberately not equal: Z outranks both on
# degree, and X/Y are a genuine tie that only the label can break. Discovery
# order is X, Y, Z -- the reverse of the ranking -- so an unranked level cannot
# produce the expected answer by accident.
_EDGES = [
    {"source_node_id": "A", "target_node_id": "X"},
    {"source_node_id": "A", "target_node_id": "Y"},
    {"source_node_id": "A", "target_node_id": "Z"},
    {"source_node_id": "Z", "target_node_id": "P"},
    {"source_node_id": "Z", "target_node_id": "Q"},
    {"source_node_id": "X", "target_node_id": "P"},
    {"source_node_id": "Y", "target_node_id": "Q"},
]


def _search_side_effect(edges):
    """Answer the three query shapes the client-side BFS issues: the per-level
    edge scan, the degree aggregation that ranks the level, and the final
    both-endpoints edge fetch."""

    async def _search(index=None, body=None, **kwargs):
        bool_query = body["query"]["bool"]
        if "aggs" in body:
            ids = set(bool_query["should"][0]["terms"]["source_node_id"])
            matching = [
                e
                for e in edges
                if e["source_node_id"] in ids or e["target_node_id"] in ids
            ]

            def _buckets(field):
                return [
                    {"key": key, "doc_count": count}
                    for key, count in Counter(e[field] for e in matching).items()
                ]

            return {
                "hits": {"hits": []},
                "aggregations": {
                    "source_degrees": {"buckets": _buckets("source_node_id")},
                    "target_degrees": {"buckets": _buckets("target_node_id")},
                },
            }
        if "should" in bool_query:
            ids = set(bool_query["should"][0]["terms"]["source_node_id"])
            hits = [
                {"_source": e}
                for e in edges
                if e["source_node_id"] in ids or e["target_node_id"] in ids
            ]
        else:
            ids = set(bool_query["must"][0]["terms"]["source_node_id"])
            hits = [
                {"_source": e}
                for e in edges
                if e["source_node_id"] in ids and e["target_node_id"] in ids
            ]
        return {"hits": {"hits": hits}}

    return _search


def _mget_side_effect(real_ids):
    async def _mget(index=None, body=None, **kwargs):
        return {
            "docs": [
                {"_id": nid, "found": True, "_source": {"entity_type": "person"}}
                if nid in real_ids
                else {"_id": nid, "found": False}
                for nid in body["ids"]
            ]
        }

    return _mget


def _make_storage():
    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    storage.workspace = "test"
    storage.global_config = {"max_graph_nodes": 1000}
    storage._nodes_index = "test-nodes"
    storage._edges_index = "test-edges"
    storage._indices_ready = True
    storage._ppl_graphlookup_available = False
    storage._refresh_graph_indices_if_dirty = AsyncMock(return_value=None)
    storage.client = AsyncMock()
    storage.client.search = AsyncMock(side_effect=_search_side_effect(_EDGES))
    storage.client.mget = AsyncMock(
        side_effect=_mget_side_effect({"A", "X", "Y", "Z", "P", "Q"})
    )
    storage.client.get = AsyncMock(
        return_value={"_id": "A", "_source": {"entity_type": "person"}}
    )
    return storage


@pytest.mark.asyncio
async def test_bfs_level_admits_by_degree_then_id():
    """One slot short of the whole level: the highest-degree leaf takes the
    first, and the label breaks the tie for the second."""
    storage = _make_storage()

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=3)

    assert sorted(node.id for node in result.nodes) == ["A", "X", "Z"]
    assert result.is_truncated is True


@pytest.mark.asyncio
async def test_bfs_level_that_fits_keeps_every_node():
    """The rule only decides a cutoff. With room for the whole level nothing is
    ranked away, and nothing is reported truncated."""
    storage = _make_storage()

    result = await storage.get_knowledge_graph("A", max_depth=1, max_nodes=10)

    assert sorted(node.id for node in result.nodes) == ["A", "X", "Y", "Z"]
    assert result.is_truncated is False
