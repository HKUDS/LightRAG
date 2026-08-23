"""``get_knowledge_graph('*')`` must rank the whole graph, not two top-N lists.

Degree came from two ``terms`` aggregations on the edge index, one per endpoint
field, each capped at its own ``size``. An entity whose in-degree and out-degree
both fell outside their respective top-N was therefore absent from the ranking
entirely, however high its undirected degree was -- an ordinary shape for a
large graph, since the cutoff normally lands in the degree-1/degree-2 band.

Edge documents now carry ``endpoints``, a multi-valued keyword holding both
ids, so one aggregation yields one bucket per entity holding true undirected
degree. The aggregation stub below implements faithful ``terms`` semantics
(buckets ordered by ``doc_count``, only the top ``size`` returned) for BOTH
shapes from the same edge list, which is what lets these tests show the old
candidate set losing an entity the new one keeps.

See issue #3613.
"""

from collections import Counter
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearch-py is required for OpenSearch storage tests",
)

from lightrag.kg.opensearch_impl import (  # noqa: E402
    _EDGE_ENDPOINTS_FIELD,
    _EDGE_ENDPOINTS_META_FLAG,
    OpenSearchGraphStorage,
    _edge_endpoints,
)


pytestmark = pytest.mark.offline


# The issue's minimal case. Every entity has undirected degree 2, so the label
# decides the single slot and "A" must win it. "A" is the one entity that holds
# neither a top-1 out-degree (that is Z, with 2) nor a top-1 in-degree (that is
# Y, with 2), so a union of two size-1 aggregations cannot see it at all.
_EDGES = [
    {"source_node_id": "Z", "target_node_id": "P"},
    {"source_node_id": "Z", "target_node_id": "Q"},
    {"source_node_id": "R", "target_node_id": "Y"},
    {"source_node_id": "S", "target_node_id": "Y"},
    {"source_node_id": "A", "target_node_id": "T"},
    {"source_node_id": "U", "target_node_id": "A"},
]


def _edge_search_side_effect(edges):
    """Answer the degree aggregation and the both-endpoints edge fetch.

    `terms` returns only the top `size` buckets, ordered by doc_count
    descending and then by key ascending, so a key that falls off the end is
    indistinguishable from one with no edges at all -- modelling that
    truncation faithfully is the whole point here.
    """

    def _top(counts, size):
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]
        return [{"key": key, "doc_count": count} for key, count in ranked]

    async def _search(index=None, body=None, **kwargs):
        aggs = (body or {}).get("aggs")
        if aggs:
            if "degrees" in aggs:
                spec = aggs["degrees"]["terms"]
                assert spec["field"] == _EDGE_ENDPOINTS_FIELD
                counts = Counter()
                for edge in edges:
                    for endpoint in _edge_endpoints(
                        edge["source_node_id"], edge["target_node_id"]
                    ):
                        counts[endpoint] += 1
                buckets = _top(counts, spec["size"])
                return {
                    "hits": {"hits": []},
                    "aggregations": {
                        "degrees": {
                            "buckets": buckets,
                            "doc_count_error_upper_bound": 0,
                        }
                    },
                }

            def _terms(name, field):
                counts = Counter(edge[field] for edge in edges)
                return {"buckets": _top(counts, aggs[name]["terms"]["size"])}

            return {
                "hits": {"hits": []},
                "aggregations": {
                    "src": _terms("src", "source_node_id"),
                    "tgt": _terms("tgt", "target_node_id"),
                },
            }

        must = body["query"]["bool"]["must"]
        sources = set(must[0]["terms"]["source_node_id"])
        targets = set(must[1]["terms"]["target_node_id"])
        return {
            "hits": {
                "hits": [
                    {"_source": edge, "sort": [i]}
                    for i, edge in enumerate(edges)
                    if edge["source_node_id"] in sources
                    and edge["target_node_id"] in targets
                ]
            }
        }

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


def _make_storage(*, endpoints_ready, edges=_EDGES, node_count=99):
    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    storage.workspace = "test"
    storage.global_config = {"max_graph_nodes": 1000}
    storage._nodes_index = "test-nodes"
    storage._edges_index = "test-edges"
    storage._indices_ready = True
    storage._ppl_graphlookup_available = False
    storage._edge_endpoints_ready = endpoints_ready
    storage._refresh_graph_indices_if_dirty = AsyncMock(return_value=None)
    storage.client = AsyncMock()
    storage.client.search = AsyncMock(side_effect=_edge_search_side_effect(edges))
    # node_count > max_nodes so the wildcard view reports truncation and takes
    # the ranked path rather than collecting every node.
    storage.client.count = AsyncMock(return_value={"count": node_count})
    storage.client.mget = AsyncMock(
        side_effect=_mget_side_effect(
            {e["source_node_id"] for e in edges} | {e["target_node_id"] for e in edges}
        )
    )
    storage.client.create_pit = AsyncMock(return_value={"pit_id": "pit"})
    storage.client.delete_pit = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_wildcard_ranking_sees_an_entity_outside_both_top_n():
    """One slot, three entities tied at undirected degree 2, "A" sorts first.

    "A" reaches the sort only because a single aggregation over `endpoints`
    buckets it once with its true degree.
    """
    storage = _make_storage(endpoints_ready=True)

    result = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=1)

    assert [node.id for node in result.nodes] == ["A"]
    assert result.is_truncated is True


@pytest.mark.asyncio
async def test_legacy_union_omits_what_the_endpoints_aggregation_ranks():
    """The defect itself, at the candidate budget the issue describes.

    Asking for one candidate, the union of two size-1 aggregations returns the
    top out-degree and the top in-degree and never sees "A"; one aggregation
    over `endpoints` returns "A" with its true undirected degree. Pinning both
    keeps the fallback from being mistaken for the fix.
    """
    legacy = _make_storage(endpoints_ready=False)
    modern = _make_storage(endpoints_ready=True)

    legacy_degrees = await legacy._degree_map_from_edge_index(1)
    modern_degrees = await modern._degree_map_from_edge_index(1)

    assert "A" not in legacy_degrees
    assert modern_degrees == {"A": 2}


@pytest.mark.asyncio
async def test_wildcard_ranking_uses_one_aggregation():
    storage = _make_storage(endpoints_ready=True)

    await storage.get_knowledge_graph("*", max_depth=1, max_nodes=1)

    agg_bodies = [
        call.kwargs["body"]
        for call in storage.client.search.await_args_list
        if "aggs" in (call.kwargs.get("body") or {})
    ]
    assert len(agg_bodies) == 1
    assert list(agg_bodies[0]["aggs"]) == ["degrees"]


@pytest.mark.asyncio
async def test_popular_labels_ranks_from_undirected_degree():
    """`get_popular_labels` ranked from the same two truncated aggregations and
    gets the same complete candidate set."""
    storage = _make_storage(endpoints_ready=True)
    storage._collect_isolated_labels = AsyncMock(return_value=[])

    labels = await storage.get_popular_labels(limit=3)

    # Every entity ties at degree 2 except the leaves, so the label orders them.
    assert labels == ["A", "Y", "Z"]


@pytest.mark.asyncio
async def test_degree_aggregation_over_fetches_per_shard():
    """`shard_size` above `size` is what keeps the merged counts honest on a
    multi-shard index; without it each shard reports only its own top `size`."""
    storage = _make_storage(endpoints_ready=True)

    await storage.get_knowledge_graph("*", max_depth=1, max_nodes=4)

    terms = next(
        call.kwargs["body"]["aggs"]["degrees"]["terms"]
        for call in storage.client.search.await_args_list
        if "degrees" in (call.kwargs.get("body") or {}).get("aggs", {})
    )
    assert terms["shard_size"] > terms["size"]


@pytest.mark.asyncio
async def test_residual_shard_error_is_reported():
    """A non-zero error bound means the degrees themselves are still
    approximate. Usable, but the operator is told rather than left guessing."""
    storage = _make_storage(endpoints_ready=True)

    async def _search(index=None, body=None, **kwargs):
        return {
            "hits": {"hits": []},
            "aggregations": {
                "degrees": {
                    "buckets": [{"key": "A", "doc_count": 2}],
                    "doc_count_error_upper_bound": 7,
                }
            },
        }

    storage.client.search = AsyncMock(side_effect=_search)

    # lightrag's logger does not propagate to the root logger caplog listens on.
    with patch("lightrag.kg.opensearch_impl.logger") as mock_logger:
        await storage._degree_map_from_edge_index(10)

    warning = mock_logger.warning.call_args[0][0]
    assert "doc_count_error_upper_bound=7" in warning


# -- write path ------------------------------------------------------------


def _upsert_storage():
    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    storage.workspace = "test"
    storage.global_config = {}
    storage._nodes_index = "test-nodes"
    storage._edges_index = "test-edges"
    storage._indices_ready = True
    storage._edge_endpoints_ready = True
    storage._ensure_indices_ready = AsyncMock(return_value=None)
    storage.has_nodes_batch = AsyncMock(side_effect=lambda ids: set(ids))
    storage.client = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_upsert_edge_writes_both_endpoints():
    storage = _upsert_storage()

    await storage.upsert_edge("A", "B", {"weight": 1.0})

    doc = storage.client.index.await_args.kwargs["body"]
    assert doc[_EDGE_ENDPOINTS_FIELD] == ["A", "B"]


@pytest.mark.asyncio
async def test_upsert_edge_deduplicates_a_self_loop():
    """`node_degree` counts the loop's single document once; two values would
    make the aggregation disagree with it."""
    storage = _upsert_storage()

    await storage.upsert_edge("A", "A", {"weight": 1.0})

    doc = storage.client.index.await_args.kwargs["body"]
    assert doc[_EDGE_ENDPOINTS_FIELD] == ["A"]


def test_edges_index_mapping_declares_endpoints_as_keyword():
    """A `dynamic: true` index would map the array as text with a `.keyword`
    subfield, which cannot be aggregated under the bare field name."""
    import inspect

    source = inspect.getsource(OpenSearchGraphStorage._create_indices_if_not_exist)
    assert '_EDGE_ENDPOINTS_FIELD: {"type": "keyword"}' in source


# -- backfill --------------------------------------------------------------


def _migration_storage():
    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    storage.workspace = "test"
    storage._edges_index = "test-edges"
    storage._edge_endpoints_ready = False
    storage.client = AsyncMock()
    storage.client.indices.exists = AsyncMock(return_value=True)
    storage.client.indices.get_mapping = AsyncMock(
        return_value={"test-edges": {"mappings": {"properties": {}, "_meta": {}}}}
    )
    return storage


@pytest.mark.asyncio
async def test_backfill_lands_mapping_values_and_flag():
    storage = _migration_storage()
    storage.client.count = AsyncMock(
        side_effect=[{"count": 12}, {"count": 0}]  # before, after
    )

    await storage._ensure_edge_endpoints_ready()

    put_bodies = [
        call.kwargs["body"]
        for call in storage.client.indices.put_mapping.await_args_list
    ]
    assert {"properties": {_EDGE_ENDPOINTS_FIELD: {"type": "keyword"}}} in put_bodies
    assert {"_meta": {_EDGE_ENDPOINTS_META_FLAG: True}} in put_bodies
    storage.client.update_by_query.assert_awaited_once()
    assert storage._edge_endpoints_ready is True


@pytest.mark.asyncio
async def test_backfill_leaves_flag_unset_when_docs_remain():
    """The flag is the promise the aggregation relies on. A partial backfill
    must keep the legacy path rather than rank from an incomplete field."""
    storage = _migration_storage()
    storage.client.count = AsyncMock(side_effect=[{"count": 12}, {"count": 3}])

    await storage._ensure_edge_endpoints_ready()

    assert storage._edge_endpoints_ready is False
    put_bodies = [
        call.kwargs["body"]
        for call in storage.client.indices.put_mapping.await_args_list
    ]
    assert {"_meta": {_EDGE_ENDPOINTS_META_FLAG: True}} not in put_bodies


@pytest.mark.asyncio
async def test_completed_backfill_skips_the_rescan():
    storage = _migration_storage()
    storage.client.indices.get_mapping = AsyncMock(
        return_value={
            "test-edges": {
                "mappings": {
                    "properties": {_EDGE_ENDPOINTS_FIELD: {"type": "keyword"}},
                    "_meta": {_EDGE_ENDPOINTS_META_FLAG: True},
                }
            }
        }
    )

    await storage._ensure_edge_endpoints_ready()

    assert storage._edge_endpoints_ready is True
    storage.client.update_by_query.assert_not_awaited()
    storage.client.indices.put_mapping.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_failure_degrades_instead_of_failing_startup():
    """The legacy aggregations are exactly what shipped before, so a cluster
    that cannot complete the backfill keeps serving them rather than refusing
    to start."""
    from opensearchpy.exceptions import OpenSearchException

    storage = _migration_storage()
    storage.client.count = AsyncMock(side_effect=OpenSearchException("boom"))

    await storage._ensure_edge_endpoints_ready()

    assert storage._edge_endpoints_ready is False
