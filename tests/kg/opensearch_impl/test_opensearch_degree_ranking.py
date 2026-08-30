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
    _EDGE_ID_CANONICAL_META_FLAG,
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


@pytest.mark.asyncio
async def test_created_edges_index_maps_endpoints_and_stamps_the_flag():
    """Pinned on the request the storage sends, not on its source text.

    Two properties of a freshly created edges index, both of which a later
    reformatting of the literal must not be able to break:

    `endpoints` is mapped `keyword` explicitly, because the index is
    `dynamic: true` and a dynamically mapped string array becomes `text` with a
    `.keyword` subfield, which cannot be aggregated under the bare field name.

    The readiness flag is in the mapping `_meta`, so an index that can never
    hold a document without the field says so durably instead of leaving the
    next startup to recount.
    """
    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    storage.workspace = "test"
    storage._nodes_index = "test-nodes"
    storage._edges_index = "test-edges"
    storage._edge_endpoints_ready = False
    storage.client = AsyncMock()
    storage.client.indices.exists = AsyncMock(return_value=False)

    await storage._create_indices_if_not_exist()

    bodies = {
        call.kwargs["index"]: call.kwargs["body"]
        for call in storage.client.indices.create.await_args_list
    }
    mappings = bodies["test-edges"]["mappings"]

    assert mappings["properties"][_EDGE_ENDPOINTS_FIELD] == {"type": "keyword"}
    assert mappings["_meta"][_EDGE_ENDPOINTS_META_FLAG] is True
    assert storage._edge_endpoints_ready is True


def test_graph_edge_properties_omit_the_endpoints_field():
    """`endpoints` is storage-internal and duplicates the top-level
    source/target, so it must not reach the API or the WebUI edge panel."""
    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)

    edge = storage._construct_graph_edge(
        "edge-1",
        {
            "source_node_id": "A",
            "target_node_id": "B",
            _EDGE_ENDPOINTS_FIELD: _edge_endpoints("A", "B"),
            "relationship": "r",
            "weight": 1.0,
            "description": "d",
            "source_ids": ["c1"],
        },
    )

    assert edge.source == "A"
    assert edge.target == "B"
    assert edge.properties == {"weight": 1.0, "description": "d"}


# -- backfill --------------------------------------------------------------


def _counter(*, probe_missing=0, backfill_counts=()):
    """Answer both count shapes from one stub.

    The coverage probe and the backfill's own count run the same query against
    the same client; `terminate_after` is what tells them apart, and it is
    capped at 1 because the probe only ever asks a boolean.
    """
    remaining = list(backfill_counts)

    async def count(*, index=None, body=None, **kwargs):
        if kwargs.get("terminate_after"):
            return {"count": min(probe_missing, 1)}
        return {"count": remaining.pop(0)}

    return AsyncMock(side_effect=count)


def _migration_storage(*, flagged=False, missing_after_flag=0, backfill_counts=()):
    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    storage.workspace = "test"
    storage._edges_index = "test-edges"
    storage._edge_endpoints_ready = False
    storage.client = AsyncMock()
    storage.client.indices.exists = AsyncMock(return_value=True)
    properties = {_EDGE_ENDPOINTS_FIELD: {"type": "keyword"}} if flagged else {}
    # A real flagged index also carries the canonical-edge-id flag, so the
    # re-stamp on the dirty path has something to preserve.
    meta = (
        {
            _EDGE_ENDPOINTS_META_FLAG: True,
            _EDGE_ID_CANONICAL_META_FLAG: True,
        }
        if flagged
        else {}
    )
    storage.client.indices.get_mapping = AsyncMock(
        return_value={
            "test-edges": {"mappings": {"properties": properties, "_meta": meta}}
        }
    )
    if flagged:
        storage.client.count = _counter(
            probe_missing=missing_after_flag, backfill_counts=backfill_counts
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
    storage = _migration_storage(flagged=True)

    await storage._ensure_edge_endpoints_ready()

    assert storage._edge_endpoints_ready is True
    # The flag's remaining job: the coverage probe still runs, the backfill and
    # the _meta write it guards do not.
    storage.client.count.assert_awaited_once()
    storage.client.update_by_query.assert_not_awaited()
    storage.client.indices.put_mapping.assert_not_awaited()


@pytest.mark.asyncio
async def test_coverage_probe_is_bounded_to_a_single_hit():
    """The question is a boolean, so the probe must not pay for counting a
    whole backlog on an index the backfill has not reached yet."""
    storage = _migration_storage(flagged=True)

    await storage._ensure_edge_endpoints_ready()

    call = storage.client.count.await_args
    assert call.kwargs["terminate_after"] == 1
    assert call.kwargs["body"]["query"] == storage._edge_missing_endpoints_query()


@pytest.mark.asyncio
async def test_flagged_index_with_missing_docs_is_backfilled_again():
    """Nothing fences writes against an old-version process, so a rolling
    deploy can index an edge without `endpoints` after the flag is stamped.
    Trusting the flag alone would leave that edge outside the aggregation
    permanently."""
    storage = _migration_storage(
        flagged=True, missing_after_flag=1, backfill_counts=(1, 0)
    )

    await storage._ensure_edge_endpoints_ready()

    storage.client.update_by_query.assert_awaited_once()
    assert storage._edge_endpoints_ready is True
    # The re-stamp must not clobber the canonical-edge-id flag sharing _meta;
    # losing it would re-trigger that whole migration on every startup.
    restamped = [
        call.kwargs["body"]["_meta"]
        for call in storage.client.indices.put_mapping.await_args_list
        if "_meta" in call.kwargs["body"]
    ]
    assert restamped == [
        {_EDGE_ENDPOINTS_META_FLAG: True, _EDGE_ID_CANONICAL_META_FLAG: True}
    ]


@pytest.mark.asyncio
async def test_flagged_index_that_cannot_be_repaired_keeps_the_legacy_ranking():
    """A backfill that does not finish must not leave the ranking claiming an
    exactness the field no longer has."""
    storage = _migration_storage(
        flagged=True, missing_after_flag=1, backfill_counts=(5, 2)
    )

    await storage._ensure_edge_endpoints_ready()

    assert storage._edge_endpoints_ready is False


@pytest.mark.asyncio
async def test_probe_failure_falls_back_to_the_flag_rather_than_below_it():
    """Revalidation guards a promise the flag already made. A search rejection
    or a transport error must not downgrade a healthy migrated index to the
    approximate ranking for the life of the process, which would make the
    guard the likeliest cause of the degradation it prevents."""
    from opensearchpy.exceptions import OpenSearchException

    storage = _migration_storage(flagged=True)
    storage.client.count = AsyncMock(side_effect=OpenSearchException("rejected"))

    await storage._ensure_edge_endpoints_ready()

    assert storage._edge_endpoints_ready is True
    storage.client.update_by_query.assert_not_awaited()


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
