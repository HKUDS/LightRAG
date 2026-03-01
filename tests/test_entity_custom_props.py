"""Tests for custom property pass-through in acreate_entity()."""

import json
from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock, patch

from lightrag.utils_graph import acreate_entity


@asynccontextmanager
async def _noop_lock(*args, **kwargs):
    yield


def make_fake_graph():
    """Create a minimal fake KnowledgeGraphStorage."""
    graph = AsyncMock()
    graph._nodes = {}

    async def has_node(name):
        return name in graph._nodes

    async def upsert_node(name, data):
        graph._nodes[name] = data

    graph.has_node = has_node
    graph.upsert_node = upsert_node
    return graph


def make_fake_vdb():
    """Create a minimal fake VectorDBStorage with global_config."""
    vdb = AsyncMock()
    vdb.global_config = {"workspace": "test"}
    return vdb


@pytest.fixture
def storage():
    graph = make_fake_graph()
    entities_vdb = make_fake_vdb()
    relationships_vdb = make_fake_vdb()
    return graph, entities_vdb, relationships_vdb


@pytest.mark.offline
@pytest.mark.asyncio
async def test_scalar_custom_property(storage):
    """Scalar custom properties are stored as-is."""
    graph, entities_vdb, relationships_vdb = storage
    entity_data = {
        "entity_type": "PERSON",
        "description": "A test entity",
        "confidence": 0.95,
        "category": "test",
    }
    with (
        patch("lightrag.utils_graph.get_storage_keyed_lock", _noop_lock),
        patch("lightrag.utils_graph.get_entity_info", new_callable=AsyncMock),
    ):
        await acreate_entity(
            graph, entities_vdb, relationships_vdb, "TestEntity", entity_data
        )
    node = graph._nodes["TestEntity"]
    assert node["confidence"] == 0.95
    assert node["category"] == "test"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_complex_custom_property_serialized(storage):
    """List and dict custom properties are JSON-serialized."""
    graph, entities_vdb, relationships_vdb = storage
    entity_data = {
        "entity_type": "ORG",
        "description": "Another entity",
        "aliases": ["TE2", "Test2"],
        "metadata": {"source": "manual", "version": 1},
    }
    with (
        patch("lightrag.utils_graph.get_storage_keyed_lock", _noop_lock),
        patch("lightrag.utils_graph.get_entity_info", new_callable=AsyncMock),
    ):
        await acreate_entity(
            graph, entities_vdb, relationships_vdb, "TestEntity2", entity_data
        )
    node = graph._nodes["TestEntity2"]
    assert json.loads(node["aliases"]) == ["TE2", "Test2"]
    assert json.loads(node["metadata"]) == {"source": "manual", "version": 1}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reserved_keys_not_overwritten(storage):
    """Reserved keys use their computed defaults, not raw pass-through."""
    graph, entities_vdb, relationships_vdb = storage
    entity_data = {
        "entity_type": "PLACE",
        "description": "A place",
        "created_at": 999,  # should NOT overwrite the computed timestamp
    }
    with (
        patch("lightrag.utils_graph.get_storage_keyed_lock", _noop_lock),
        patch("lightrag.utils_graph.get_entity_info", new_callable=AsyncMock),
    ):
        await acreate_entity(
            graph, entities_vdb, relationships_vdb, "TestEntity3", entity_data
        )
    node = graph._nodes["TestEntity3"]
    assert node["created_at"] != 999  # computed timestamp, not the input
