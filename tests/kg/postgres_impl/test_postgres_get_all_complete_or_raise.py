"""``get_all_nodes`` / ``get_all_edges`` must be complete-or-raise.

These two methods feed whole-graph consumers — storage migration, VDB rebuild,
and the KG integrity audit — that treat the returned list as the entire graph.
The audit in particular concludes "this document owns nothing" from a document
being absent from the scan, so a silently dropped node (or an edge whose
properties, including ``source_id``, were silently blanked) is not a cosmetic
loss: it is a missing attribution that can get a contributing document falsely
certified as contribution-free, licensing a purge that leaves its graph
objects stranded (issue #3400).

Historically both methods degraded on a JSON parse failure of the AGE
``properties`` payload: ``get_all_nodes`` skipped the row with a warning and
``get_all_edges`` kept the row but blanked its properties. Corrupt properties
are data corruption; the methods now raise ``PGGraphQueryException`` instead.
"""

import pytest
from unittest.mock import AsyncMock

from lightrag.kg.postgres_impl import PGGraphQueryException, PGGraphStorage

pytestmark = pytest.mark.offline


def make_graph_storage() -> PGGraphStorage:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    return storage


@pytest.mark.asyncio
async def test_get_all_nodes_raises_on_corrupt_properties():
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[
            {"properties": '{"entity_id": "ALICE", "source_id": "chunk-1"}'},
            {"properties": '{"entity_id": "ACME", not-json'},
        ]
    )

    with pytest.raises(PGGraphQueryException) as excinfo:
        await storage.get_all_nodes()

    assert "Corrupt node properties" in excinfo.value.get_message()


@pytest.mark.asyncio
async def test_get_all_nodes_returns_every_parsable_row():
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[
            {"properties": '{"entity_id": "ALICE", "source_id": "chunk-1"}'},
            {"properties": {"entity_id": "ACME", "source_id": "chunk-2"}},
        ]
    )

    nodes = await storage.get_all_nodes()

    assert [n["id"] for n in nodes] == ["ALICE", "ACME"]


@pytest.mark.asyncio
async def test_get_all_edges_raises_on_corrupt_properties():
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[
            {
                "source": "ACME",
                "target": "ALICE",
                "properties": '{"source_id": "chunk-1"',
            },
        ]
    )

    with pytest.raises(PGGraphQueryException) as excinfo:
        await storage.get_all_edges()

    assert "Corrupt edge properties" in excinfo.value.get_message()


@pytest.mark.asyncio
async def test_get_all_edges_keeps_source_id_attribution():
    # get_all_edges() projects each endpoint with
    # ``agtype_access_operator(props, '"entity_id"')::text``. The ``::text`` cast
    # on an agtype SCALAR returns its raw string content, so endpoints arrive as
    # plain entity_ids ('ACME', not '"ACME"') -- identical to get_all_nodes() and
    # every other backend. A surrounding double quote here is therefore part of
    # the id, never an AGE serialization artifact to strip. The mock inputs mirror
    # that real form (verified end-to-end against AGE 1.7.0 and 1.8.0); a quoted
    # mock would encode a contract the backend does not actually produce.
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[
            {
                "source": "ACME",
                "target": "ALICE",
                "properties": '{"source_id": "chunk-1", "weight": 1.0}',
            },
            {
                "source": "ACME",
                "target": "BOB",
                "properties": {"source_id": "chunk-2", "weight": 1.0},
            },
            {
                # An id whose double quotes are part of the name (survives
                # normalize_entity_name because of the inner quote) must pass
                # through verbatim, not be stripped.
                "source": '"Al"ice"',
                "target": "ACME",
                "properties": {"source_id": "chunk-3", "weight": 1.0},
            },
        ]
    )

    edges = await storage.get_all_edges()

    assert edges[0]["source_id"] == "chunk-1"
    assert edges[0]["source"] == "ACME"
    assert edges[0]["target"] == "ALICE"
    # Already-parsed dict properties pass through unchanged.
    assert edges[1]["source_id"] == "chunk-2"
    assert edges[1]["target"] == "BOB"
    # A quote that is part of the id is preserved, not treated as a cast artifact.
    assert edges[2]["source"] == '"Al"ice"'
