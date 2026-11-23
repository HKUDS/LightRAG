"""Restore a LightRAG workspace from a tagged Lance cold snapshot, zero re-embed.

Rules for callers:
- Offline promotion job: the cold snapshot is the source of truth, the hot
  vector storages are disposable derived indexes. Run against a QUIESCED
  workspace (no writers); the rebuild CLI installs a counted fail-loud
  embedding stub, and any embed attempt aborts the run.
- ``tables`` must come from ``open_snapshot``, which verifies tag, dimension,
  vector completeness and embedding-model identity BEFORE any mutation.
- Vectors come from the snapshot's embedding columns, seeded into the
  deferred-embedding buffer's vector cache (FileBackedSnapshotContract,
  *Deferred-embedding protocol*); the embedding function is never called.
  Only NanoVectorDBStorage targets support this; other backends raise
  ``StorageCapabilityError`` before anything is dropped.
- Sources of truth are restored first (full_docs/text_chunks KV + graph),
  then each vector target is dropped and rewritten. An interrupted run leaves
  a partial index that a re-run heals (idempotent); the snapshot loses nothing.
- ``before_drop(label)`` runs immediately before each vector target's drop.
  The rebuild CLI wires it to the corrupt/incompatible backup-recovery flow
  (``recover_incompatible``), so a refused container is privately backed up
  before it is destroyed; a caller that supplies no hook must not point this
  at a container whose unreadable rows are the only copy.
- Prefer an EMPTY target. Over a populated workspace the KV/graph upserts are
  additive while vector targets are replaced, so pre-existing source rows
  absent from the tag survive as orphans without vectors -- retention, not
  loss of snapshot content (Consistency without transactions). A same-snapshot
  re-run stays idempotent; there is no rejection of a populated target.
- Residues: doc_status, the LLM cache and the chunk-tracking/recovery-anchor
  KV namespaces are not in the snapshot; a restored workspace serves queries,
  but delete/retry lack their anchors until re-ingestion. README_REBUILD_VDB.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import TYPE_CHECKING, Any

import lancedb
import numpy as np
from lancedb.table import LanceTable

from lightrag.exceptions import StorageCapabilityError
from lightrag.kg.lancedb_impl import _TABLES
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.operate import _truncate_vdb_content
from lightrag.tools.rebuild_vdb import _drop_vdb, _new_stats
from lightrag.utils import logger

if TYPE_CHECKING:
    from lightrag import LightRAG

# Schema-metadata key the cold export stamps on every table (lancedb_impl
# _schemas); empty bytes when the exporting model was unknown.
_EMBEDDING_MODEL_KEY = b"lightrag.embedding_model"

# Content-less chunks legitimately carry no vector; every other row must.
_NULL_VECTOR_FILTERS = {
    "entities": "embedding IS NULL",
    "relations": "embedding IS NULL",
    "chunks": "embedding IS NULL AND content <> ''",
}
_CHUNK_KEYS = ("content", "full_doc_id", "chunk_order_index", "tokens", "file_path")
_NODE_KEYS = ("entity_type", "description", "source_id", "file_path")
_EDGE_KEYS = ("keywords", "description", "source_id", "weight", "file_path")


class RestoreError(RuntimeError):
    """A snapshot or restore input was refused; nothing has been mutated."""


def open_snapshot(
    dest: str, tag: str, *, dim: int, model_name: str | None = None,
    storage_options: dict[str, str] | None = None,
) -> dict[str, LanceTable]:
    """Open the four snapshot tables checked out at ``tag``, fully validated.
    Raises ``RestoreError`` (blank tag, missing table/tag, empty table, dim
    mismatch, missing vector, embedding-model mismatch) before the caller may
    mutate anything. ``model_name`` is the target's embedding model: a recorded
    snapshot model that differs is an incompatible query space and refuses;
    an unrecorded model (older snapshot) or an unnamed target only warns,
    because the identity cannot be verified either way.
    """
    if not isinstance(tag, str) or not tag.strip():
        raise RestoreError("tag must be a non-empty string")
    db = lancedb.connect(dest, storage_options=storage_options or {})
    existing = set(db.list_tables())
    tables: dict[str, LanceTable] = {}
    for name in _TABLES:
        if name not in existing:
            raise RestoreError(f"snapshot table {name!r} is missing at {dest}")
        table = db.open_table(name)
        if tag not in table.tags.list():
            raise RestoreError(f"tag {tag!r} not found on snapshot table {name!r}")
        table.checkout(tag)
        tables[name] = table
    for name, table in tables.items():
        if table.count_rows() == 0:
            raise RestoreError(f"snapshot table {name!r} is empty at tag {tag!r}")
    for name, null_filter in _NULL_VECTOR_FILTERS.items():
        field = tables[name].schema.field("embedding").type
        if field.list_size != dim:
            raise RestoreError(f"snapshot table {name!r} stores dimension {field.list_size}, configured embedding dimension is {dim}")
        missing = tables[name].count_rows(filter=null_filter)
        if missing:
            raise RestoreError(f"snapshot table {name!r} has {missing} vector-eligible row(s) without a stored vector at tag {tag!r}")
    for name, table in tables.items():
        recorded = (table.schema.metadata or {}).get(_EMBEDDING_MODEL_KEY, b"").decode()
        if recorded and model_name and recorded != model_name:
            raise RestoreError(
                f"snapshot table {name!r} was embedded with model {recorded!r}, "
                f"configured embedding model is {model_name!r}; vectors from another "
                "embedding space are an incompatible query space"
            )
        if not recorded or not model_name:
            logger.warning(
                f"Snapshot table {name!r}: embedding model cannot be verified "
                f"(snapshot={recorded!r}, target={model_name!r}); proceeding on the "
                "operator's assurance that the spaces match"
            )
    return tables


def _pages(table: LanceTable, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Row pages of a checked-out (immutable) table, bounded by batch_size."""
    for start in range(0, table.count_rows(), batch_size):
        yield table.take_offsets(list(range(start, start + batch_size))).to_arrow().to_pylist()


def _entity_payload(row: dict[str, Any], global_config: dict[str, Any]) -> dict[str, Any]:
    name = row["entity_name"]
    content = _truncate_vdb_content(f"{name}\n{row['description']}", global_config, f"entity:{name}")
    return {"entity_name": name, "content": content,
            **{key: row[key] for key in ("entity_type", "description", "source_id", "file_path")}}


def _relation_payload(row: dict[str, Any], global_config: dict[str, Any]) -> dict[str, Any]:
    src, tgt = row["src_id"], row["tgt_id"]
    content = _truncate_vdb_content(f"{row['keywords']}\t{src}\n{tgt}\n{row['description']}", global_config, f"relationship:{src}-{tgt}")
    return {"src_id": src, "tgt_id": tgt, "content": content,
            **{key: row[key] for key in ("keywords", "description", "source_id", "weight", "file_path")}}


def _chunk_payload(row: dict[str, Any], global_config: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _CHUNK_KEYS}


async def restore_workspace_snapshot(
    tables: dict[str, LanceTable], rag: LightRAG, global_config: dict[str, Any], *,
    batch_size: int = 500, concurrency: int = 3,
    before_drop: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Restore sources + the three vector targets; returns per-target stats.
    Caller rules are in the module docstring. ``rag`` may be any object
    exposing the six INITIALIZED storages and ``embedding_func``.
    ``before_drop(label)`` is awaited right before that target's drop; the
    rebuild CLI wires it to the backup-recovery flow so corrupt or refused
    containers are preserved before destruction. Failures propagate; nothing
    is swallowed.
    """
    graph = rag.chunk_entity_relation_graph
    storages = (rag.full_docs, rag.text_chunks, graph, rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb)
    if any(storage is None for storage in storages):
        raise RestoreError("target storages are missing; call await rag.initialize_storages() first")
    if batch_size < 1:
        raise RestoreError(f"batch_size must be >= 1, got {batch_size}")
    if concurrency < 1:
        raise RestoreError(f"concurrency must be >= 1, got {concurrency}")
    for vdb in (rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb):
        if not getattr(vdb, "persists_vectors", True) or not isinstance(vdb, NanoVectorDBStorage):
            raise StorageCapabilityError(f"{type(vdb).__name__} cannot restore precomputed vectors; snapshot restore supports NanoVectorDBStorage targets")

    for rows in _pages(tables["docs"], batch_size):
        await rag.full_docs.upsert({r["id"]: {"content": r["content"]} for r in rows})
    for rows in _pages(tables["chunks"], batch_size):
        await rag.text_chunks.upsert({r["id"]: _chunk_payload(r, global_config) for r in rows})
    for rows in _pages(tables["entities"], batch_size):
        await graph.upsert_nodes_batch([(r["entity_name"], {k: r[k] for k in _NODE_KEYS}) for r in rows])
    for rows in _pages(tables["relations"], batch_size):
        await graph.upsert_edges_batch([(r["src_id"], r["tgt_id"], {k: r[k] for k in _EDGE_KEYS}) for r in rows])
    for storage in (rag.full_docs, rag.text_chunks, graph):
        await storage.index_done_callback()

    semaphore = asyncio.Semaphore(concurrency)

    async def restore_target(label: str, vdb: NanoVectorDBStorage, table: LanceTable,
                             build_payload) -> dict[str, Any]:
        async with semaphore:
            stats = _new_stats(label, table.count_rows())
            if before_drop is not None:
                await before_drop(label)
            await _drop_vdb(vdb, label)
            for rows in _pages(table, batch_size):
                vectors = {r["id"]: r["embedding"] for r in rows if r["embedding"] is not None}
                stats["skipped"] += len(rows) - len(vectors)
                if not vectors:
                    continue
                await vdb.upsert({r["id"]: build_payload(r, global_config)
                                  for r in rows if r["embedding"] is not None})
                # The deferred-embedding protocol's vector cache: a pending doc
                # whose vector is already set is materialized without embedding.
                for rid, vector in vectors.items():
                    vdb._pending_upserts[rid].vector = np.asarray(vector, dtype=np.float32)
                await vdb.index_done_callback()
                stats["rebuilt"] += len(vectors)
            return stats

    targets = (
        ("entities", rag.entities_vdb, tables["entities"], _entity_payload),
        ("relationships", rag.relationships_vdb, tables["relations"], _relation_payload),
        ("chunks", rag.chunks_vdb, tables["chunks"], _chunk_payload),
    )
    # TaskGroup, not gather: a failing target cancels and awaits its siblings,
    # so an interrupted restore never leaves concurrent writers for the re-run.
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(restore_target(*target)) for target in targets]
    stats = dict(zip((t[0] for t in targets), (task.result() for task in tasks)))
    logger.info("Snapshot restore complete: " + ", ".join(f"{label}={s['rebuilt']} vectors" for label, s in stats.items()))
    return stats
