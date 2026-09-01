"""aexport_data: the raw VectorDB relationships dump must not break the export.

``client_storage`` is a debug/snapshot escape hatch, not part of the
``BaseVectorStorage`` contract. Two backends implement it, and they disagree on
sync vs. async: ``NanoVectorDBStorage`` (an *async* property) and
``FaissVectorDBStorage`` (a plain *sync* property) -- verified against every
``lightrag/kg/*_impl.py``; no other backend (Milvus, Qdrant, Postgres, Redis,
MongoDB, OpenSearch) defines it at all.

``aexport_data`` used to do ``await relationships_vdb.client_storage``
unconditionally while building the supplementary "Relationships (from VectorDB)"
section. That failed two different ways depending on the backend:
``AttributeError`` on every backend without the attribute, and ``TypeError:
object dict can't be used in 'await' expression`` on Faiss, whose property
returns a plain dict rather than a coroutine. Both crashed the whole export,
even though the entity/relation data -- the export's primary content -- never
touches that attribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lightrag.utils import aexport_data

pytestmark = pytest.mark.offline


class _FakeGraphStorage:
    """Minimal graph with one edge, enough to exercise entities + relations."""

    async def get_all_labels(self) -> list[str]:
        return ["ALICE", "BOB"]

    async def get_node(self, entity_name: str) -> dict[str, str]:
        return {"description": f"{entity_name} node", "source_id": "chunk-1"}

    async def has_edge(self, src: str, tgt: str) -> bool:
        return (src, tgt) == ("ALICE", "BOB")

    async def get_edge(self, src: str, tgt: str) -> dict[str, str]:
        return {"description": "knows", "source_id": "chunk-1"}


class _VdbWithoutClientStorage:
    """Stands in for Milvus/Qdrant/Postgres/Redis/MongoDB/OpenSearch."""

    async def get_by_id(self, _id: str) -> None:  # pragma: no cover - unused here
        raise AssertionError("include_vector_data=False must not query the VDB")


class _VdbWithAsyncClientStorage(_VdbWithoutClientStorage):
    """Stands in for NanoVectorDBStorage: client_storage is an async property."""

    @property
    async def client_storage(self) -> dict[str, Any]:
        return {"data": [{"__id__": "rel-1", "weight": 1.0}]}


class _VdbCountingAsyncClientStorage(_VdbWithoutClientStorage):
    """Same observable shape as above, but counts descriptor invocations.

    ``@property async def`` cannot count its own accesses: the body of an
    async function does not run until the coroutine is awaited, so a counter
    inside it stays blind to exactly the access this test is about -- a
    ``hasattr`` probe that creates a coroutine and throws it away. Splitting
    the property into a sync getter that returns a coroutine keeps the
    attribute's observable behavior identical (accessing it yields an
    awaitable) while making each access visible.
    """

    def __init__(self) -> None:
        self.getter_calls = 0

    async def _snapshot(self) -> dict[str, Any]:
        return {"data": [{"__id__": "rel-1", "weight": 1.0}]}

    @property
    def client_storage(self) -> Any:
        self.getter_calls += 1
        return self._snapshot()


class _VdbWithSyncClientStorage(_VdbWithoutClientStorage):
    """Stands in for FaissVectorDBStorage: client_storage is a sync property."""

    @property
    def client_storage(self) -> dict[str, Any]:
        return {"data": [{"__id__": "rel-2", "weight": 2.0}]}


async def _run_export(tmp_path: Path, relationships_vdb: Any) -> str:
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
async def test_export_does_not_crash_without_client_storage(tmp_path: Path) -> None:
    content = await _run_export(tmp_path, _VdbWithoutClientStorage())

    # Entities and relations -- the export's primary content -- still land.
    assert "# ENTITIES" in content
    assert "ALICE" in content
    assert "# RELATIONS" in content
    # No raw-VDB section: the backend has nothing to supply it from.
    assert "# RELATIONSHIPS" not in content


@pytest.mark.asyncio
async def test_export_includes_relationships_dump_for_async_client_storage(
    tmp_path: Path,
) -> None:
    content = await _run_export(tmp_path, _VdbWithAsyncClientStorage())

    assert "# ENTITIES" in content
    assert "# RELATIONS" in content
    assert "# RELATIONSHIPS" in content
    assert "rel-1" in content


@pytest.mark.asyncio
async def test_export_probes_client_storage_on_the_class_not_the_instance(
    tmp_path: Path,
) -> None:
    """The attribute is read exactly once, to use it -- never to test for it.

    ``hasattr(relationships_vdb, "client_storage")`` would answer the presence
    question by invoking the getter and discarding what it returns; for an
    async property that means an un-awaited coroutine ("coroutine was never
    awaited") and, on a real backend, the snapshot work done twice. Reading the
    attribute off ``type(...)`` returns the property descriptor without
    invoking it.
    """
    relationships_vdb = _VdbCountingAsyncClientStorage()

    content = await _run_export(tmp_path, relationships_vdb)

    assert "rel-1" in content
    assert relationships_vdb.getter_calls == 1


@pytest.mark.asyncio
async def test_export_includes_relationships_dump_for_sync_client_storage(
    tmp_path: Path,
) -> None:
    # Faiss's client_storage is a plain sync property (returns a dict, not a
    # coroutine); awaiting it unconditionally raises TypeError. This pins that
    # the value itself decides whether to await, rather than the code assuming
    # every client_storage is awaitable.
    content = await _run_export(tmp_path, _VdbWithSyncClientStorage())

    assert "# ENTITIES" in content
    assert "# RELATIONS" in content
    assert "# RELATIONSHIPS" in content
    assert "rel-2" in content
