"""_get_node_data hybrid seeding: flag-off regression, fusion, fallback."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _get_node_data


def make_graph(entity_names):
    graph = MagicMock()
    graph.get_nodes_batch = AsyncMock(
        return_value={
            name: {"entity_name": name, "description": f"desc of {name}"}
            for name in entity_names
        }
    )
    graph.node_degrees_batch = AsyncMock(
        return_value={name: 1 for name in entity_names}
    )
    graph.get_nodes_edges_batch = AsyncMock(
        return_value={name: [] for name in entity_names}
    )
    # No entities have edges in these fixtures, but
    # _find_most_related_edges_from_entities still awaits these batch calls
    # (with empty pair lists), so they must resolve rather than return a
    # plain MagicMock.
    graph.get_edges_batch = AsyncMock(return_value={})
    graph.edge_degrees_batch = AsyncMock(return_value={})
    return graph


def make_vdb(entity_names):
    vdb = MagicMock()
    vdb.cosine_better_than_threshold = 0.2
    vdb.query = AsyncMock(
        return_value=[
            {"entity_name": name, "created_at": None} for name in entity_names
        ]
    )
    return vdb


def make_keyword_storage(hits):
    ks = MagicMock()
    ks.search = AsyncMock(return_value=hits)
    return ks


@pytest.mark.asyncio
async def test_flag_off_is_vector_only_and_never_touches_keyword_storage():
    graph = make_graph(["A", "B"])
    vdb = make_vdb(["A", "B"])
    ks = make_keyword_storage([("C", 9.0)])
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = False

    node_datas, _ = await _get_node_data(
        "q", graph, vdb, param, keyword_storage=ks
    )
    ks.search.assert_not_awaited()
    assert [n["entity_name"] for n in node_datas] == ["A", "B"]


@pytest.mark.asyncio
async def test_bm25_only_entity_joins_seeds():
    graph = make_graph(["A", "B", "NVLink"])
    vdb = make_vdb(["A", "B"])
    ks = make_keyword_storage([("NVLink", 12.5)])
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = True

    node_datas, _ = await _get_node_data(
        "NVLink bandwidth?", graph, vdb, param, keyword_storage=ks
    )
    names = [n["entity_name"] for n in node_datas]
    assert "NVLink" in names


@pytest.mark.asyncio
async def test_keyword_storage_exception_falls_back_to_vector_only():
    graph = make_graph(["A"])
    vdb = make_vdb(["A"])
    ks = MagicMock()
    ks.search = AsyncMock(side_effect=RuntimeError("index corrupted"))
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = True

    node_datas, _ = await _get_node_data(
        "q", graph, vdb, param, keyword_storage=ks
    )
    assert [n["entity_name"] for n in node_datas] == ["A"]


@pytest.mark.asyncio
async def test_no_keyword_storage_behaves_as_before():
    graph = make_graph(["A"])
    vdb = make_vdb(["A"])
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = True

    node_datas, _ = await _get_node_data("q", graph, vdb, param)
    assert [n["entity_name"] for n in node_datas] == ["A"]
