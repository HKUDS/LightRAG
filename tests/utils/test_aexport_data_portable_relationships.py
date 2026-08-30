"""Tests for aexport_data's handling of vector storage backends.

``client_storage`` is a debug/snapshot escape hatch, not part of the
BaseVectorStorage contract. Two backends implement it, disagreeing on sync
vs. async: ``NanoVectorDBStorage`` (an *async* property) and
``FaissVectorDBStorage`` (a plain *sync* property) -- verified against every
``lightrag/kg/*_impl.py``; no other backend (Milvus, Qdrant, Postgres, Redis,
MongoDB, OpenSearch) defines it at all.

Before this fix, ``aexport_data`` did ``await relationships_vdb.client_storage``
unconditionally while building the supplementary "Relationships (from
VectorDB)" dump. That failed two different ways depending on backend:
``AttributeError`` on every backend without the attribute, and
``TypeError: object dict can't be used in 'await' expression`` on Faiss,
whose property returns a plain dict rather than a coroutine. Both crashed the
whole export, even though the entity/relation data -- the export's primary
content -- never touches that attribute.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.utils import aexport_data

pytestmark = pytest.mark.offline


class _FakeGraphStorage:
    """Minimal graph with one edge, enough to exercise entities + relations."""

    async def get_all_labels(self):
        return ["Alice", "Bob"]

    async def get_node(self, entity_name):
        return {"description": f"{entity_name} node", "source_id": "chunk-1"}

    async def has_edge(self, src, tgt):
        return {src, tgt} == {"Alice", "Bob"} and src == "Alice"

    async def get_edge(self, src, tgt):
        return {"description": "knows", "source_id": "chunk-1"}


class _VdbWithoutClientStorage:
    """Stands in for every backend except NanoVectorDB."""

    async def get_by_id(self, id):
        return None


class _VdbWithAsyncClientStorage:
    """Stands in for NanoVectorDBStorage: client_storage is an async property."""

    async def get_by_id(self, id):
        return None

    @property
    async def client_storage(self):
        return {"data": [{"__id__": "rel-1", "weight": 1.0}]}


class _VdbWithSyncClientStorage:
    """Stands in for FaissVectorDBStorage: client_storage is a plain sync property."""

    async def get_by_id(self, id):
        return None

    @property
    def client_storage(self):
        return {"data": [{"__id__": "rel-2", "weight": 2.0}]}


async def _run_export(tmp_path: Path, relationships_vdb) -> str:
    output_path = tmp_path / "export.csv"
    await aexport_data(
        _FakeGraphStorage(),
        _VdbWithoutClientStorage(),
        relationships_vdb,
        str(output_path),
        file_format="csv",
    )
    return output_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_export_does_not_crash_without_client_storage(tmp_path):
    content = await _run_export(tmp_path, _VdbWithoutClientStorage())

    # Entities and relations -- the export's primary content -- still land.
    assert "# ENTITIES" in content
    assert "Alice" in content
    assert "# RELATIONS" in content
    # No raw-VDB section: the backend has nothing to supply it from.
    assert "# RELATIONSHIPS" not in content


@pytest.mark.asyncio
async def test_export_includes_relationships_dump_for_async_client_storage(tmp_path):
    content = await _run_export(tmp_path, _VdbWithAsyncClientStorage())

    assert "# ENTITIES" in content
    assert "# RELATIONS" in content
    assert "# RELATIONSHIPS" in content
    assert "rel-1" in content


@pytest.mark.asyncio
async def test_export_includes_relationships_dump_for_sync_client_storage(tmp_path):
    # Faiss's client_storage is a plain sync property (returns a dict, not a
    # coroutine); awaiting it unconditionally raises TypeError. This pins
    # that the fix inspects the value rather than assuming every
    # client_storage is awaitable.
    content = await _run_export(tmp_path, _VdbWithSyncClientStorage())

    assert "# ENTITIES" in content
    assert "# RELATIONS" in content
    assert "# RELATIONSHIPS" in content
    assert "rel-2" in content
