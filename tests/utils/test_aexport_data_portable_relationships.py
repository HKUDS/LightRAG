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

import logging
from pathlib import Path
from typing import Any

import pytest

from lightrag.utils import aexport_data

pytestmark = pytest.mark.offline


@pytest.fixture
def _propagate_lightrag_logs():
    """The ``lightrag`` logger sets propagate=False; caplog needs it on."""
    lg = logging.getLogger("lightrag")
    old = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = old


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

    async def get_by_id(self, _id: str) -> None:
        return None


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


class _VdbWithVectorBearingRows(_VdbWithoutClientStorage):
    """Faiss shape, faithfully: the embedding lives IN the metadata row.

    `FaissVectorDBStorage` writes `__vector__` into `_id_to_meta` at flush and
    reconstructs it from the index on load, and `client_storage` hands those
    rows back unfiltered. NanoVectorDB keeps vectors in a separate `matrix`,
    so only this shape can leak them into the raw dump.
    """

    @property
    def client_storage(self) -> dict[str, Any]:
        return {
            "data": [
                {
                    "__id__": "rel-3",
                    "src_id": "ALICE",
                    "__vector__": [0.125, 0.25, 0.375],
                }
            ]
        }


async def _run_export(
    tmp_path: Path,
    relationships_vdb: Any,
    *,
    include_vector_data: bool = False,
) -> str:
    output_path = tmp_path / "export.csv"
    await aexport_data(
        _FakeGraphStorage(),
        _VdbWithoutClientStorage(),
        relationships_vdb,
        str(output_path),
        file_format="csv",
        include_vector_data=include_vector_data,
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


@pytest.mark.asyncio
async def test_sync_client_storage_snapshot_warns_that_it_may_be_stale(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    _propagate_lightrag_logs: None,
) -> None:
    """The sync shape is not reload-checking, so its staleness is announced.

    `FaissVectorDBStorage.client_storage` deliberately does not funnel through
    `_get_index`, so a reader process can snapshot data older than the latest
    committed one. The dump is still worth having -- it is correct in the
    single-process case, and it is supplementary to the graph-sourced content
    -- but an export used as a backup must not omit recent records silently.
    """
    with caplog.at_level("WARNING", logger="lightrag"):
        await _run_export(tmp_path, _VdbWithSyncClientStorage())

    assert any(
        "does not check for a newer on-disk commit" in record.message
        for record in caplog.records
    ), caplog.text


@pytest.mark.asyncio
async def test_async_client_storage_snapshot_does_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    _propagate_lightrag_logs: None,
) -> None:
    """The async shape awaits a reload-if-stale path, so it has nothing to warn about."""
    with caplog.at_level("WARNING", logger="lightrag"):
        await _run_export(tmp_path, _VdbWithAsyncClientStorage())

    assert not [
        record
        for record in caplog.records
        if "does not check for a newer on-disk commit" in record.message
    ], caplog.text


@pytest.mark.asyncio
async def test_raw_dump_omits_embeddings_when_vector_data_is_not_requested(
    tmp_path: Path,
) -> None:
    """`include_vector_data=False` must mean no embeddings anywhere in the file.

    The entity and relation sections honour the flag by construction -- they
    only add a `vector_data` column when asked. The raw dump does not: it
    stringifies whatever the backend hands back, and Faiss hands back rows
    with `__vector__` in them. Left alone, the default export would emit
    embeddings and grow by orders of magnitude.
    """
    content = await _run_export(tmp_path, _VdbWithVectorBearingRows())

    assert "rel-3" in content  # the record itself is still exported
    assert "__vector__" not in content
    assert "0.375" not in content


@pytest.mark.asyncio
async def test_raw_dump_keeps_embeddings_when_vector_data_is_requested(
    tmp_path: Path,
) -> None:
    """The flag adds vector data; it must not strip what the caller asked for."""
    content = await _run_export(
        tmp_path, _VdbWithVectorBearingRows(), include_vector_data=True
    )

    assert "__vector__" in content
    assert "0.375" in content
