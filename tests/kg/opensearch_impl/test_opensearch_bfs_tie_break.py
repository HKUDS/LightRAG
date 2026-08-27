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

            # Faithful `terms` semantics, which is what makes this stub able to
            # tell the two aggregation shapes apart: buckets are ordered by
            # doc_count and only the top `size` are returned. A key that falls
            # off the end is indistinguishable from a key with no edges at all.
            def _agg(name, field):
                spec = body["aggs"][name]
                docs, filtered = matching, "filter" in spec
                if filtered:
                    allowed = set(spec["filter"]["terms"][field])
                    docs = [e for e in matching if e[field] in allowed]
                    inner = spec["aggs"]["ids"]["terms"]
                else:
                    inner = spec["terms"]
                counts = Counter(e[field] for e in docs)
                buckets = [
                    {"key": key, "doc_count": count}
                    for key, count in counts.most_common(inner["size"])
                ]
                return (
                    {"ids": {"buckets": buckets}} if filtered else {"buckets": buckets}
                )

            return {
                "hits": {"hits": []},
                "aggregations": {
                    "source_degrees": _agg("source_degrees", "source_node_id"),
                    "target_degrees": _agg("target_degrees", "target_node_id"),
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


def _ppl_response(edges):
    return {
        "schema": [{"name": "connected_edges"}],
        "datarows": [[[dict(edge, _depth=depth) for depth, edge in edges]]],
    }


def _make_ppl_storage(edges, real_ids):
    storage = _make_storage()
    storage._ppl_graphlookup_available = True
    storage.client.transport = AsyncMock()
    storage.client.transport.perform_request = AsyncMock(
        return_value=_ppl_response(edges)
    )
    storage.client.mget = AsyncMock(side_effect=_mget_side_effect(real_ids))
    return storage


@pytest.mark.asyncio
async def test_ppl_degree_lookup_skips_levels_that_cannot_reach_the_cap():
    """Only the level straddling ``max_nodes`` is ranked. node_degrees_batch
    sends its argument as a ``terms`` clause, so passing the whole reachable
    set made a large component breach OpenSearch's index.max_terms_count and
    fail the request instead of returning a truncated subgraph."""
    edges = [(1, {"source_node_id": "A", "target_node_id": f"n{i}"}) for i in range(3)]
    edges += [
        (2, {"source_node_id": "n0", "target_node_id": f"d{i}"}) for i in range(500)
    ]
    storage = _make_ppl_storage(edges, {"A"} | {f"n{i}" for i in range(3)})
    storage.node_degrees_batch = AsyncMock(return_value={})

    await storage.get_knowledge_graph("A", max_depth=3, max_nodes=3)

    ranked = storage.node_degrees_batch.await_args.args[0]
    assert set(ranked) == {"n0", "n1", "n2"}


@pytest.mark.asyncio
async def test_ppl_degree_lookup_is_capped_on_a_single_wide_level():
    """A hub puts every neighbour on one level, so bounding by level is not
    enough on its own -- the candidate list itself carries a ceiling."""
    from lightrag.kg.opensearch_impl import _GRAPH_DEGREE_RANK_MAX_CANDIDATES

    width = _GRAPH_DEGREE_RANK_MAX_CANDIDATES + 500
    edges = [
        (1, {"source_node_id": "A", "target_node_id": f"n{i}"}) for i in range(width)
    ]
    storage = _make_ppl_storage(edges, {"A"})
    storage.node_degrees_batch = AsyncMock(return_value={})

    await storage.get_knowledge_graph("A", max_depth=1, max_nodes=1000)

    ranked = storage.node_degrees_batch.await_args.args[0]
    assert len(ranked) == _GRAPH_DEGREE_RANK_MAX_CANDIDATES


@pytest.mark.asyncio
async def test_bfs_degree_lookup_is_capped_on_a_single_wide_level():
    """The non-PPL path needs the same ceiling, and it binds harder there: the
    per-level edge query asks for ``size: 10000``, so one hub can put ~20k
    endpoints in a level -- past both ``index.max_terms_count`` and the bucket
    budget the two degree aggregations request."""
    from lightrag.kg.opensearch_impl import _GRAPH_DEGREE_RANK_MAX_CANDIDATES

    width = _GRAPH_DEGREE_RANK_MAX_CANDIDATES + 500
    neighbours = [f"n{i}" for i in range(width)]
    edges = [{"source_node_id": "A", "target_node_id": n} for n in neighbours]

    storage = _make_storage()
    storage.client.search = AsyncMock(side_effect=_search_side_effect(edges))
    storage.client.mget = AsyncMock(side_effect=_mget_side_effect({"A", *neighbours}))
    storage.node_degrees_batch = AsyncMock(return_value={})

    await storage.get_knowledge_graph("A", max_depth=1, max_nodes=1000)

    ranked = storage.node_degrees_batch.await_args.args[0]
    assert len(ranked) == _GRAPH_DEGREE_RANK_MAX_CANDIDATES


@pytest.mark.asyncio
async def test_degree_aggregation_confines_its_buckets_to_the_requested_ids():
    """``node_degrees_batch`` must not lose a requested LOW-degree node.

    The ``should`` query admits an edge when EITHER endpoint matches, so
    ``source_node_id`` can take as many distinct values as the level has
    neighbours. A bucket budget derived from ``len(node_ids)`` cannot cover
    that, and ``terms`` drops the smallest counts first -- exactly the nodes
    the ranking is trying to place last. They came back absent, scored 0, and
    the level silently reverted to label order.

    Here H is requested and busy, L is requested and quiet, and 50 unrequested
    sources each outrank L on doc_count. Under the old ``size: 2 * len(ids)``
    budget over unconfined keys, L falls off the end.
    """
    edges = [
        {"source_node_id": f"s{i}", "target_node_id": "H"}
        for i in range(50)
        for _ in range(2)
    ]
    edges.append({"source_node_id": "L", "target_node_id": "t"})

    storage = _make_storage()
    storage.client.search = AsyncMock(side_effect=_search_side_effect(edges))

    degrees = await storage.node_degrees_batch(["L", "H"])

    assert degrees["L"] == 1
    assert degrees["H"] == 100


@pytest.mark.asyncio
async def test_bfs_skips_the_degree_lookup_once_the_cap_is_full():
    """A level reached with zero slots left must not pay for a ranking.

    ``max_nodes`` lands exactly on the end of the first level, so the second
    level is discovered (it still has to be, to report truncation truthfully)
    but can admit nothing. Gating on level overflow alone ran an aggregation
    over the whole level to order nodes that were all about to be discarded.
    """
    storage = _make_storage()
    storage.node_degrees_batch = AsyncMock(return_value={})

    result = await storage.get_knowledge_graph("A", max_depth=3, max_nodes=4)

    storage.node_degrees_batch.assert_not_awaited()
    assert sorted(node.id for node in result.nodes) == ["A", "X", "Y", "Z"]
    assert result.is_truncated is True


@pytest.mark.asyncio
async def test_bfs_stops_expanding_once_truncation_is_proven():
    """Past a proven cutoff there is nothing left to discover, so the next
    level's edge search and mget are pure cost. The cut keys off
    ``truncated_by_cap`` rather than a full ``seen_nodes`` on purpose: an exact
    fill has to keep probing, which is what the test above pins."""
    storage = _make_storage()

    await storage.get_knowledge_graph("A", max_depth=3, max_nodes=3)

    # Two mgets: the start-node resolution and the one level that was
    # expanded. A third would mean the loop probed past a proven cutoff.
    assert storage.client.mget.await_count == 2


@pytest.mark.asyncio
async def test_ppl_skips_the_degree_lookup_when_no_slots_remain():
    """``max_nodes=1`` admits the start node and nothing else, so every level
    is sliced off whole by ``ranked[: max_nodes - 1]``. The scan still walked
    into the first level and ranked it."""
    edges = [(1, {"source_node_id": "A", "target_node_id": f"n{i}"}) for i in range(5)]
    storage = _make_ppl_storage(edges, {"A"} | {f"n{i}" for i in range(5)})
    storage.node_degrees_batch = AsyncMock(return_value={})

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=1)

    storage.node_degrees_batch.assert_not_awaited()
    assert [node.id for node in result.nodes] == ["A"]
    assert result.is_truncated is True
