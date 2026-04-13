"""Tests for node_degree and edge_degree return type contract.

Every graph storage backend must return ``int`` from ``node_degree`` and
``edge_degree``, even when the requested node or edge does not exist.
The expected default for missing entries is ``0``.

These tests exercise the *real* methods on ``PGGraphStorage`` and
``NetworkXStorage`` (from ``lightrag/kg/``) with mocked backing stores
so they run without a database or heavy import dependencies.
"""

import asyncio
from unittest.mock import AsyncMock

import networkx as nx
import pytest

from lightrag.kg.postgres_impl import PGGraphStorage
from lightrag.kg.networkx_impl import NetworkXStorage


# ---------------------------------------------------------------------------
# Helpers — build storage instances without triggering __post_init__
# ---------------------------------------------------------------------------


def _make_pg_storage(**attrs) -> PGGraphStorage:
    """Create a PGGraphStorage instance bypassing __post_init__, then
    attach mock attributes needed for the methods under test."""
    instance = object.__new__(PGGraphStorage)
    for k, v in attrs.items():
        setattr(instance, k, v)
    return instance


def _make_nx_storage(**attrs) -> NetworkXStorage:
    """Create a NetworkXStorage instance bypassing __post_init__, then
    attach mock attributes needed for the methods under test."""
    instance = object.__new__(NetworkXStorage)
    for k, v in attrs.items():
        setattr(instance, k, v)
    return instance


# ---------------------------------------------------------------------------
# PostgreSQL (PGGraphStorage)
# ---------------------------------------------------------------------------


class TestPGGraphStorageDegree:
    """Verify PGGraphStorage.node_degree / edge_degree return 0 for missing entries."""

    @pytest.mark.asyncio
    async def test_node_degree_missing_returns_zero(self):
        storage = _make_pg_storage(
            node_degrees_batch=AsyncMock(return_value={})
        )
        result = await storage.node_degree("nonexistent_node")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_node_degree_existing_returns_value(self):
        storage = _make_pg_storage(
            node_degrees_batch=AsyncMock(return_value={"node_a": 5})
        )
        result = await storage.node_degree("node_a")
        assert result == 5
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_missing_returns_zero(self):
        storage = _make_pg_storage(
            edge_degrees_batch=AsyncMock(return_value={})
        )
        result = await storage.edge_degree("src", "tgt")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_existing_returns_value(self):
        storage = _make_pg_storage(
            edge_degrees_batch=AsyncMock(return_value={("src", "tgt"): 7})
        )
        result = await storage.edge_degree("src", "tgt")
        assert result == 7
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_node_degree_none_result_returns_zero(self):
        storage = _make_pg_storage(
            node_degrees_batch=AsyncMock(return_value=None)
        )
        result = await storage.node_degree("any")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_none_result_returns_zero(self):
        storage = _make_pg_storage(
            edge_degrees_batch=AsyncMock(return_value=None)
        )
        result = await storage.edge_degree("src", "tgt")
        assert result == 0
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# NetworkX
# ---------------------------------------------------------------------------


class TestNetworkXStorageDegree:
    """Verify NetworkXStorage.node_degree returns 0 for missing nodes."""

    @pytest.mark.asyncio
    async def test_node_degree_missing_returns_zero(self):
        graph = nx.Graph()
        storage = _make_nx_storage(_get_graph=AsyncMock(return_value=graph))
        result = await storage.node_degree("nonexistent")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_node_degree_existing_returns_value(self):
        graph = nx.Graph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        storage = _make_nx_storage(_get_graph=AsyncMock(return_value=graph))
        result = await storage.node_degree("a")
        assert result == 2
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_missing_returns_zero(self):
        graph = nx.Graph()
        storage = _make_nx_storage(_get_graph=AsyncMock(return_value=graph))
        result = await storage.edge_degree("x", "y")
        assert result == 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_degree_existing_returns_value(self):
        graph = nx.Graph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        storage = _make_nx_storage(_get_graph=AsyncMock(return_value=graph))
        # edge_degree("a", "b") = degree(a) + degree(b) = 2 + 2 = 4
        result = await storage.edge_degree("a", "b")
        assert result == 4
        assert isinstance(result, int)
