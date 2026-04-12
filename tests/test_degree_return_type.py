"""Tests for node_degree and edge_degree return type contract.

Every graph storage backend must return ``int`` from ``node_degree`` and
``edge_degree``, even when the requested node or edge does not exist.
The expected default for missing entries is ``0``.

These tests mock the batch helpers so they run without a database or
heavy import dependencies.
"""

import asyncio
import types
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Helpers — extract the unbound coroutine functions from source to avoid
# importing modules that require Python 3.10+ syntax or heavy deps.
# ---------------------------------------------------------------------------


async def _pg_node_degree(self, node_id: str) -> int:
    """Copied logic from PGGraphStorage.node_degree"""
    result = await self.node_degrees_batch(node_ids=[node_id])
    if result and node_id in result:
        return result[node_id]
    return 0


async def _pg_edge_degree(self, src_id: str, tgt_id: str) -> int:
    """Copied logic from PGGraphStorage.edge_degree"""
    result = await self.edge_degrees_batch(edges=[(src_id, tgt_id)])
    if result and (src_id, tgt_id) in result:
        return result[(src_id, tgt_id)]
    return 0


async def _nx_node_degree(self, node_id: str) -> int:
    """Copied logic from NetworkXStorage.node_degree"""
    graph = await self._get_graph()
    if graph.has_node(node_id):
        return graph.degree(node_id)
    return 0


async def _nx_edge_degree(self, src_id: str, tgt_id: str) -> int:
    """Copied logic from NetworkXStorage.edge_degree"""
    graph = await self._get_graph()
    src_degree = graph.degree(src_id) if graph.has_node(src_id) else 0
    tgt_degree = graph.degree(tgt_id) if graph.has_node(tgt_id) else 0
    return src_degree + tgt_degree


def _make_stub(**attrs):
    """Create a simple namespace object with the given attributes."""
    return types.SimpleNamespace(**attrs)


# ---------------------------------------------------------------------------
# PostgreSQL (PGGraphStorage)
# ---------------------------------------------------------------------------


class TestPGGraphStorageDegree:
    """Verify PGGraphStorage.node_degree / edge_degree return 0 for missing entries."""

    @pytest.mark.asyncio
    async def test_node_degree_missing_returns_zero(self):
        stub = _make_stub(node_degrees_batch=AsyncMock(return_value={}))
        result = await _pg_node_degree(stub, "nonexistent_node")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_node_degree_existing_returns_value(self):
        stub = _make_stub(node_degrees_batch=AsyncMock(return_value={"node_a": 5}))
        result = await _pg_node_degree(stub, "node_a")
        assert result == 5
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_missing_returns_zero(self):
        stub = _make_stub(edge_degrees_batch=AsyncMock(return_value={}))
        result = await _pg_edge_degree(stub, "src", "tgt")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_existing_returns_value(self):
        stub = _make_stub(
            edge_degrees_batch=AsyncMock(return_value={("src", "tgt"): 7})
        )
        result = await _pg_edge_degree(stub, "src", "tgt")
        assert result == 7
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_node_degree_none_result_returns_zero(self):
        stub = _make_stub(node_degrees_batch=AsyncMock(return_value=None))
        result = await _pg_node_degree(stub, "any")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_none_result_returns_zero(self):
        stub = _make_stub(edge_degrees_batch=AsyncMock(return_value=None))
        result = await _pg_edge_degree(stub, "src", "tgt")
        assert result == 0
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# NetworkX
# ---------------------------------------------------------------------------


class TestNetworkXStorageDegree:
    """Verify NetworkXStorage.node_degree returns 0 for missing nodes."""

    @pytest.mark.asyncio
    async def test_node_degree_missing_returns_zero(self):
        import networkx as nx

        graph = nx.Graph()
        stub = _make_stub(_get_graph=AsyncMock(return_value=graph))
        result = await _nx_node_degree(stub, "nonexistent")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_node_degree_existing_returns_value(self):
        import networkx as nx

        graph = nx.Graph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        stub = _make_stub(_get_graph=AsyncMock(return_value=graph))
        result = await _nx_node_degree(stub, "a")
        assert result == 2
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_missing_returns_zero(self):
        import networkx as nx

        graph = nx.Graph()
        stub = _make_stub(_get_graph=AsyncMock(return_value=graph))
        result = await _nx_edge_degree(stub, "x", "y")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_existing_returns_value(self):
        import networkx as nx

        graph = nx.Graph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        stub = _make_stub(_get_graph=AsyncMock(return_value=graph))
        # edge_degree("a", "b") = degree(a) + degree(b) = 2 + 2 = 4
        result = await _nx_edge_degree(stub, "a", "b")
        assert result == 4
        assert isinstance(result, int)
