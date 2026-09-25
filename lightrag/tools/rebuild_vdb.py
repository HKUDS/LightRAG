#!/usr/bin/env python3
"""
Offline Vector Storage (VDB) Rebuild Tool for LightRAG

The knowledge graph and the text_chunks KV store are the authoritative data
sources in LightRAG. If a vector storage write fails at runtime (e.g. during
an entity editing with WebUI), graph and vector storage drift apart:
graph records lose their vector counterparts (and stale vector records may
remain). This tool restores consistency by dropping each vector storage and
rebuilding it from scratch from its authoritative source:

    entities_vdb       <- graph nodes
    relationships_vdb  <- graph edges
    chunks_vdb         <- text_chunks KV store

It can also be used after changing the embedding model or embedding
dimension. Run it with the updated embedding configuration and rebuild all
vector storages so stored vectors match the embedding space the server will
query.

That second use is the reason this tool tolerates one specific startup
failure. A vector storage whose container was written in a different embedding
space REFUSES to attach (``VectorSpaceMismatchError``) -- that refusal is what
stops the server serving an empty or foreign index, and it is the condition
this tool exists to clear. So the three vector targets are initialized
individually and a typed refusal is *recorded* rather than aborting the run;
the rebuild then opens with ``drop()`` on the refused container, which
re-provisions it in the current embedding space, and re-initializes it. A typed
corrupt file-backed snapshot also permits entering the menu, but every file it
names is backed up before any confirmed rebuild drops it. Source checks and
user confirmation precede recovery; normal startup still refuses corruption.
Other failures, including outages and bad credentials, abort without dropping
anything.

The authoritative SOURCES (graph storage and the ``text_chunks`` KV store) keep
the server-identical init path and still abort the run on any failure -- they
are what the rebuild reads from, and rebuilding vectors out of a half-migrated
source is worse than not rebuilding at all.

A diagnostic consistency check mode is also provided so users can decide
whether a (potentially expensive, full re-embedding) rebuild is needed. The
check itself only issues read queries and does not run a rebuild (no drop +
re-embed). It is NOT, however, strictly side-effect-free: the tool
initializes every storage on startup — the same initialization the server
performs — and for some backends that includes schema/DDL setup and one-time
legacy migrations (e.g. Qdrant upserts data into the new collection,
PostgreSQL batch-inserts into the new table, Milvus may create a temp
collection and drop/rename the original). Run it like a server startup, not
like a pure read.

IMPORTANT: Shut down the LightRAG Server (and any other writers) before
running this tool.

Usage:
    lightrag-rebuild-vdb
    # or
    python -m lightrag.tools.rebuild_vdb

Configuration is read from .env / environment variables, exactly like the
LightRAG server (LIGHTRAG_GRAPH_STORAGE, LIGHTRAG_VECTOR_STORAGE,
LIGHTRAG_KV_STORAGE, WORKSPACE, WORKING_DIR, EMBEDDING_* ...). The embedding
function is constructed through the same factory the server uses, so rebuilt
vectors live in exactly the same embedding space.
"""

import asyncio
import os
import sys
import shutil
import time
from uuid import uuid4
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.constants import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_WORKING_DIR,
)
from lightrag.config_store import (
    EMBEDDING_TARGETS,
    create_configuration_storage,
    describe_configuration_container,
    preflight_configuration_anchor,
    read_embedding_baselines,
    record_embedding_baseline,
    resolve_config_dir,
    resolve_configuration_storage,
    verify_configuration_identity,
)
from lightrag.exceptions import (
    ConfigurationAnchorLockError,
    ConfigurationStorageError,
    CorruptStorageSnapshotError,
    ReferencesIntactFlushError,
    StorageCapabilityError,
    VectorSpaceMismatchError,
    WorkingDirectoryInUseError,
)
from lightrag.kg import STORAGE_ENV_REQUIREMENTS
from lightrag.private_file import PrivateFileError, open_private_file
from lightrag.kg.anchor_lock import (
    acquire_anchor_lock_shared,
    release_anchor_lock_shared,
)
from lightrag.kg.working_dir_lock import (
    acquire_working_dir_lock,
    release_working_dir_lock,
    uses_working_dir,
)
from lightrag.namespace import NameSpace
from lightrag.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    get_env_value,
    logger,
    make_relation_vdb_ids,
    safe_vdb_operation_with_exception,
    setup_logger,
)

# NOTE: .env loading and logger setup are deferred to main() so that importing
# this module as a library (see README "Library usage") has no side effects on
# the caller's environment or logging configuration.

DEFAULT_BATCH_SIZE = 500

# Flush deferred-embedding backends (nano/faiss compute embeddings in
# index_done_callback) every N batches to bound memory usage.
FLUSH_EVERY_N_BATCHES = 10

# Cap for listing missing items in the consistency report
MAX_REPORTED_MISSING = 20

# ANSI color codes for terminal output
BOLD_CYAN = "\033[1;36m"
BOLD_RED = "\033[1;31m"
BOLD_GREEN = "\033[1;32m"
RESET = "\033[0m"

ProgressCallback = Callable[[int, int], None]


def _new_stats(label: str, source_total: int) -> Dict[str, Any]:
    return {
        "label": label,
        "source_total": source_total,
        "prepared": 0,
        "rebuilt": 0,
        # Records upserted but not yet confirmed flushed to disk. For
        # deferred-embedding backends (nano/faiss) the embedding+persist
        # happens in index_done_callback, so a record only counts as
        # "rebuilt" once a flush succeeds.
        "staged": 0,
        "skipped": 0,
        "duplicates": 0,
        "batches": 0,
        "failed_batches": 0,
        "errors": [],
    }


def _ensure_vector_rebuild_supported(vdb) -> None:
    if not getattr(vdb, "persists_vectors", True):
        storage_name = type(vdb).__name__
        raise StorageCapabilityError(
            f"{storage_name} does not persist vectors and cannot be used as a "
            "rebuild target. Configure a persistent vector storage before "
            "calling the rebuild library API."
        )


async def _drop_vdb(vdb, label: str) -> None:
    drop_result = await vdb.drop()
    if not isinstance(drop_result, dict) or drop_result.get("status") != "success":
        raise RuntimeError(f"Failed to drop {label} vector storage: {drop_result}")
    logger.info(f"Dropped {label} vector storage")


def backup_corrupt_snapshot(paths: tuple[str, ...]) -> List[str]:
    """Preserve every file of a corrupt storage before a confirmed rebuild.

    Copies each path that exists to an exclusive ``<path>.corrupt-<token>``
    sibling and flushes it before the caller may drop the originals. One token
    per recovery, so a multi-file storage's backups read as one set. Any error
    aborts recovery with the originals untouched. Backups are never deleted
    automatically, including after a failed rebuild.

    A storage whose files have all vanished is REFUSED rather than reported as
    backed up: there is nothing left to preserve, so nothing may be destroyed.
    A file that is merely absent from a present set is skipped -- a torn pair
    is exactly the shape some corruption takes.

    "Absent" is decided by OPENING the source and catching ``FileNotFoundError``
    from that one call, not by a prior ``exists()``. A file that disappears
    between the check and the open would otherwise abort mid-way through, and,
    worse, only the open can tell "this half of a torn pair is missing" apart
    from "the directory is gone" or "this file cannot be read" -- both of which
    must abort rather than be silently skipped as nothing-to-preserve.

    A backup holds the same documents and metadata as the store it copies and
    is kept indefinitely by design, so it must never be more readable than
    what it copies. ``open_private_file`` is what enforces that, on both
    platforms and from the instant each file exists -- read it before relying
    on this. A backup that cannot be created private, or proven private,
    removes itself and aborts the recovery with the originals intact.
    """
    token = uuid4().hex[:8]
    backups: List[str] = []
    for path in paths:
        absolute = os.path.abspath(path)
        try:
            source = open(absolute, "rb")
        except FileNotFoundError:
            # The ONLY tolerated failure, and scoped to the open alone.
            continue
        backup = f"{absolute}.corrupt-{token}"
        with source, open_private_file(backup) as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        # Appended only once the copy is durable, so the returned list never
        # names a backup that does not hold its source.
        backups.append(backup)
    if not backups:
        raise FileNotFoundError(
            f"Refusing to recover: none of {list(paths)} exists to be backed up"
        )
    if os.name != "nt":
        for parent in sorted({os.path.dirname(backup) for backup in backups}):
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    return backups


async def clear_vector_space_refusal(vdb, label: str) -> None:
    """Make a vector storage that refused to attach usable again.

    A storage raises ``VectorSpaceMismatchError`` from ``initialize()`` when
    its container holds vectors from another embedding space. The way out is
    the one this tool is built around, and it is two steps, in this order:

    1. ``drop()`` -- destroys the foreign vectors and re-provisions the
       container in the CURRENT embedding space, recording this process's
       provenance marker. Every backend must be able to serve ``drop()`` while
       refused; a backend that raises its refusal before it has a client or a
       lock is wedged, not fail-closed.
    2. ``initialize()`` again -- the container now matches, so this is the
       ordinary init path and leaves a fully live instance. Doing the rebuild
       against the half-initialized object instead would depend on which step
       of ``initialize()`` happened to run before the refusal.

    Destructive by construction: it exists to delete a container the operator
    has already been told is unusable. Call it only for a target whose
    ``initialize()`` raised ``VectorSpaceMismatchError`` -- never to paper over
    another failure.
    """
    logger.warning(
        f"Rebuild {label}: dropping the incompatible vector container before "
        f"the rebuild (its vectors were written in a different embedding space)"
    )
    await _drop_vdb(vdb, label)
    await vdb.initialize()
    logger.info(
        f"Rebuild {label}: vector storage re-provisioned in the current embedding space"
    )


async def _flush(vdb, stats: Dict[str, Any]) -> None:
    """Flush staged records to disk and credit them as rebuilt.

    Deferred-embedding backends (nano/faiss) compute embeddings and persist
    inside ``index_done_callback``, so an embedder outage surfaces here rather
    than in ``upsert``. Treat such a failure the same way as a failed upsert
    batch: record it, drop the staged count, and continue (sources are never
    modified, so the user can re-run). ``rebuilt`` is only incremented after a
    flush succeeds. That is not the whole story on a per-item backend, which
    keeps a retryable failure buffered and returns normally: ``rebuilt`` then
    counts a record the server never took. Mid-rebuild that heals -- the next
    flush retries it -- so it is not treated as an error here. What must not
    heal silently is a residue left by the LAST flush, and ``commit_baseline``
    asks the storage directly before recording anything.
    """
    if stats["staged"] == 0:
        return
    label = stats["label"]
    failure: BaseException | None = None
    try:
        await vdb.index_done_callback()
    except ReferencesIntactFlushError as e:
        # A backend that can prove its raise lost nothing. Whether anything
        # LANDED is the buffer's answer to give, not this exception's -- and
        # OpenSearch raises this both when a bulk failed with every operation
        # still buffered AND when the bulk was accepted and only the refresh
        # after it failed. Reading both as a failed flush reports durable
        # vectors as lost, and ``commit_baseline`` then withholds the
        # baseline, leaving the server refusing to start over an index that
        # is in fact rebuilt.
        if await _retained_after_flush(vdb):
            failure = e
        else:
            logger.warning(
                f"Rebuild {label}: the flush of {stats['staged']} staged "
                f"record(s) reported {type(e).__name__}, but the storage "
                f"kept nothing buffered -- the write landed and only a step "
                f"after it failed ({e})"
            )
    except Exception as e:
        failure = e

    if failure is None:
        stats["rebuilt"] += stats["staged"]
    else:
        logger.error(
            f"Rebuild {label}: flush of {stats['staged']} staged record(s) "
            f"failed: {failure}"
        )
        stats["failed_batches"] += 1
        stats["errors"].append(
            {
                "batch": f"flush@batch-{stats['batches']}",
                "records_lost": stats["staged"],
                "error_type": type(failure).__name__,
                "error_msg": str(failure),
            }
        )
    stats["staged"] = 0


async def _retained_after_flush(vdb) -> bool:
    """Whether the storage still holds operations the server never took.

    A backend that cannot be asked keeps the conservative reading of its own
    raise: unanswerable is treated as retained, so a flush is never credited
    on an assumption.
    """
    has_pending = getattr(vdb, "has_pending_index_ops", None)
    if has_pending is None:
        return True
    try:
        return bool(await has_pending(include_deletes=True))
    except Exception as e:  # pragma: no cover - defensive
        logger.error(
            f"Could not ask {type(vdb).__name__} whether its flush retained "
            f"anything ({type(e).__name__}: {e}); treating it as retained"
        )
        return True


async def _upsert_batch(
    vdb,
    batch_payload: Dict[str, Dict[str, Any]],
    batch_no: int,
    total_batches: int,
    stats: Dict[str, Any],
) -> None:
    """Upsert one batch; collect the error and continue on persistent failure."""
    label = stats["label"]
    try:
        await safe_vdb_operation_with_exception(
            operation=lambda payload=batch_payload: vdb.upsert(payload),
            operation_name=f"rebuild_{label}_upsert",
            entity_name=f"batch {batch_no}/{total_batches}",
            max_retries=3,
            retry_delay=0.2,
        )
        stats["staged"] += len(batch_payload)
    except Exception as e:
        logger.error(f"Rebuild {label}: batch {batch_no}/{total_batches} failed: {e}")
        stats["failed_batches"] += 1
        stats["errors"].append(
            {
                "batch": batch_no,
                "records_lost": len(batch_payload),
                "error_type": type(e).__name__,
                "error_msg": str(e),
            }
        )
    stats["batches"] += 1
    if stats["batches"] % FLUSH_EVERY_N_BATCHES == 0:
        await _flush(vdb, stats)


async def _drop_and_upsert(
    vdb,
    payloads: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
    *,
    batch_size: int,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Drop the VDB, then upsert ``payloads`` in batches with periodic flushes."""
    await _drop_vdb(vdb, stats["label"])

    items = list(payloads.items())
    total_batches = (len(items) + batch_size - 1) // batch_size
    for batch_no, start in enumerate(range(0, len(items), batch_size), start=1):
        batch_payload = dict(items[start : start + batch_size])
        await _upsert_batch(vdb, batch_payload, batch_no, total_batches, stats)
        if progress_callback:
            progress_callback(batch_no, total_batches)

    # Final flush persists any remaining deferred embeddings
    await _flush(vdb, stats)
    return stats


async def rebuild_entities_vdb(
    graph,
    entities_vdb,
    global_config: Dict[str, Any],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Rebuild the entities vector storage from graph nodes (authoritative source).

    Payloads mirror the authoritative write point in
    operate._merge_nodes_then_upsert field for field.
    """
    _ensure_vector_rebuild_supported(entities_vdb)
    from lightrag.operate import _truncate_vdb_content

    nodes = await graph.get_all_nodes()
    stats = _new_stats("entities", len(nodes))

    payloads: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        entity_name = node.get("entity_id") or node.get("id")
        if entity_name is None or not str(entity_name).strip():
            stats["skipped"] += 1
            logger.warning(
                f"Rebuild entities: skipping graph node without entity id: {node!r}"
            )
            continue
        entity_name = str(entity_name)
        description = node.get("description") or ""
        entity_content = _truncate_vdb_content(
            f"{entity_name}\n{description}",
            global_config,
            f"entity:{entity_name}",
        )
        entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
        if entity_vdb_id in payloads:
            stats["duplicates"] += 1
            continue
        payloads[entity_vdb_id] = {
            "entity_name": entity_name,
            "entity_type": node.get("entity_type") or "",
            "content": entity_content,
            "source_id": node.get("source_id") or "",
            "description": description,
            "file_path": node.get("file_path") or "",
        }

    stats["prepared"] = len(payloads)
    return await _drop_and_upsert(
        entities_vdb,
        payloads,
        stats,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )


async def rebuild_relationships_vdb(
    graph,
    relationships_vdb,
    global_config: Dict[str, Any],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Rebuild the relationships vector storage from graph edges (authoritative source).

    Payloads mirror the authoritative write point in
    operate._merge_edges_then_upsert field for field: endpoints are sorted and
    the VDB id is the normalized ``rel-`` hash. Backends that return each
    undirected edge once per direction (e.g. Neo4j, Memgraph) are deduplicated
    by that normalized id.
    """
    _ensure_vector_rebuild_supported(relationships_vdb)
    from lightrag.operate import _truncate_vdb_content

    edges = await graph.get_all_edges()
    stats = _new_stats("relationships", len(edges))

    payloads: Dict[str, Dict[str, Any]] = {}
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None or not str(src).strip() or not str(tgt).strip():
            stats["skipped"] += 1
            logger.warning(
                f"Rebuild relationships: skipping graph edge without endpoints: {edge!r}"
            )
            continue
        src_id, tgt_id = str(src), str(tgt)
        # Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
        if src_id > tgt_id:
            src_id, tgt_id = tgt_id, src_id

        rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        if rel_vdb_id in payloads:
            stats["duplicates"] += 1
            continue

        description = edge.get("description") or ""
        keywords = edge.get("keywords") or ""
        try:
            weight = float(edge.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        rel_content = _truncate_vdb_content(
            f"{keywords}\t{src_id}\n{tgt_id}\n{description}",
            global_config,
            f"relationship:{src_id}-{tgt_id}",
        )
        payloads[rel_vdb_id] = {
            "src_id": src_id,
            "tgt_id": tgt_id,
            "source_id": edge.get("source_id") or "",
            "content": rel_content,
            "keywords": keywords,
            "description": description,
            "weight": weight,
            "file_path": edge.get("file_path") or "",
        }

    stats["prepared"] = len(payloads)
    return await _drop_and_upsert(
        relationships_vdb,
        payloads,
        stats,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )


async def enumerate_kv_keys(kv) -> List[str]:
    """List all keys of a KV storage instance.

    BaseKVStorage has no enumeration API, so this uses backend-specific
    scans (same patterns as lightrag/tools/clean_llm_query_cache.py).
    When a new KV backend is added, this function must be extended.
    """
    storage_name = type(kv).__name__

    if storage_name == "JsonKVStorage":
        async with kv._storage_lock:
            return list(kv._data.keys())

    if storage_name == "RedisKVStorage":
        keys: List[str] = []
        prefix = f"{kv.final_namespace}:"
        async with kv._get_redis_connection() as redis:
            cursor = 0
            while True:
                cursor, batch = await redis.scan(cursor, match=f"{prefix}*", count=1000)
                for key in batch:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    keys.append(key[len(prefix) :] if key.startswith(prefix) else key)
                if cursor == 0:
                    break
        return keys

    if storage_name == "PGKVStorage":
        from lightrag.kg.postgres_impl import namespace_to_table_name

        table_name = namespace_to_table_name(kv.namespace)
        query = f"SELECT id FROM {table_name} WHERE workspace = $1"
        rows = await kv.db.query(query, [kv.workspace], multirows=True)
        return [row["id"] for row in (rows or [])]

    if storage_name == "MongoKVStorage":
        keys = []
        cursor = kv._data.find({}, {"_id": 1})
        async for doc in cursor:
            keys.append(doc["_id"])
        return keys

    if storage_name == "OpenSearchKVStorage":
        keys = []
        async for hits in kv._iter_raw_docs(batch_size=1000):
            keys.extend(hit["_id"] for hit in hits)
        return keys

    raise ValueError(f"Unsupported KV storage type for key enumeration: {storage_name}")


async def rebuild_chunks_vdb(
    text_chunks_kv,
    chunks_vdb,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Rebuild the chunks vector storage from the text_chunks KV store.

    The KV store is enumerated directly (not via doc_status.chunks_list)
    because ainsert_custom_kg writes chunks without a doc_status record.
    The ingestion pipeline upserts the full chunk record into chunks_vdb,
    so each KV record is passed through as the payload.

    Every key in the text_chunks namespace is a chunk record, and chunks use
    several id schemes that no single prefix matches — ``chunk-<hash>``
    (custom KG), ``{doc_id}-chunk-{order}`` (text pipeline), and
    ``{doc_id}-mm-<modality>-{order}`` (multimodal). Rather than pattern-match
    keys (and silently drop a scheme), all keys are enumerated and the
    per-record ``content`` check below is the only filter.
    """
    _ensure_vector_rebuild_supported(chunks_vdb)
    chunk_ids = [str(key) for key in await enumerate_kv_keys(text_chunks_kv)]
    stats = _new_stats("chunks", len(chunk_ids))

    await _drop_vdb(chunks_vdb, "chunks")

    total_batches = (len(chunk_ids) + batch_size - 1) // batch_size
    for batch_no, start in enumerate(range(0, len(chunk_ids), batch_size), start=1):
        batch_ids = chunk_ids[start : start + batch_size]
        records = await text_chunks_kv.get_by_ids(batch_ids)

        batch_payload: Dict[str, Dict[str, Any]] = {}
        for chunk_id, record in zip(batch_ids, records):
            if not isinstance(record, dict) or not record.get("content"):
                stats["skipped"] += 1
                logger.warning(
                    f"Rebuild chunks: skipping chunk without content: {chunk_id}"
                )
                continue
            payload = dict(record)
            payload.pop("_id", None)
            payload.setdefault("full_doc_id", "")
            payload.setdefault("file_path", "")
            batch_payload[chunk_id] = payload

        stats["prepared"] += len(batch_payload)
        if batch_payload:
            await _upsert_batch(
                chunks_vdb, batch_payload, batch_no, total_batches, stats
            )
        if progress_callback:
            progress_callback(batch_no, total_batches)

    await _flush(chunks_vdb, stats)
    return stats


async def check_vdb_consistency(
    graph,
    entities_vdb,
    relationships_vdb,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    incompatible: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Read-only diagnosis: find graph records with no vector counterpart.

    Only the graph -> VDB direction is covered; stale reverse orphans
    (records present in the VDB but absent from the graph) can only be
    eliminated by a full rebuild. Relations are probed with both candidate
    ids from make_relation_vdb_ids so legacy reverse-order ids are not
    misreported as missing.

    ``incompatible`` names targets (``"entities"`` / ``"relationships"``) whose
    storage refused to attach because of incompatible vectors or a corrupt
    snapshot, mapped to the refusal message. Such a target is NOT
    probed: its storage cannot answer, and probing it anyway would report
    every graph record as missing -- a true statement about that container
    that reads as a routine drift report and buries the one fact that matters,
    which is that the container is unusable and a rebuild is mandatory rather
    than optional. The report carries the refusal under ``"incompatible"``
    instead, and ``consistent`` is False.
    """
    incompatible = dict(incompatible or {})
    report: Dict[str, Any] = {
        "graph_entities": 0,
        "graph_relations": 0,
        "missing_entities": 0,
        "missing_relations": 0,
        "missing_entity_names": [],
        "missing_relation_pairs": [],
        "skipped_nodes": 0,
        "skipped_edges": 0,
        "incompatible": incompatible,
    }

    # Entities: one candidate id per graph node
    nodes = await graph.get_all_nodes()
    entity_items: List[tuple] = []
    seen_entity_ids: set = set()
    for node in nodes:
        entity_name = node.get("entity_id") or node.get("id")
        if entity_name is None or not str(entity_name).strip():
            report["skipped_nodes"] += 1
            continue
        entity_name = str(entity_name)
        entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
        if entity_vdb_id in seen_entity_ids:
            continue
        seen_entity_ids.add(entity_vdb_id)
        entity_items.append((entity_vdb_id, entity_name))

    report["graph_entities"] = len(entity_items)
    if "entities" in incompatible:
        entity_items = []
    for start in range(0, len(entity_items), batch_size):
        batch = entity_items[start : start + batch_size]
        results = await entities_vdb.get_by_ids([vdb_id for vdb_id, _ in batch])
        for (vdb_id, entity_name), record in zip(batch, results):
            if record is None:
                report["missing_entities"] += 1
                if len(report["missing_entity_names"]) < MAX_REPORTED_MISSING:
                    report["missing_entity_names"].append(entity_name)

    # Relations: both candidate ids (normalized + legacy reverse) per edge
    edges = await graph.get_all_edges()
    relation_items: List[tuple] = []
    seen_relation_ids: set = set()
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None or not str(src).strip() or not str(tgt).strip():
            report["skipped_edges"] += 1
            continue
        candidate_ids = make_relation_vdb_ids(str(src), str(tgt))
        if candidate_ids[0] in seen_relation_ids:
            continue
        seen_relation_ids.add(candidate_ids[0])
        relation_items.append((candidate_ids, f"{src} ~ {tgt}"))

    report["graph_relations"] = len(relation_items)
    if "relationships" in incompatible:
        relation_items = []
    for start in range(0, len(relation_items), batch_size):
        batch = relation_items[start : start + batch_size]
        flat_ids: List[str] = []
        for candidate_ids, _ in batch:
            flat_ids.extend(candidate_ids)
        results = await relationships_vdb.get_by_ids(flat_ids)

        offset = 0
        for candidate_ids, pair_label in batch:
            candidate_records = results[offset : offset + len(candidate_ids)]
            offset += len(candidate_ids)
            if all(record is None for record in candidate_records):
                report["missing_relations"] += 1
                if len(report["missing_relation_pairs"]) < MAX_REPORTED_MISSING:
                    report["missing_relation_pairs"].append(pair_label)

    report["consistent"] = (
        not incompatible
        and report["missing_entities"] == 0
        and report["missing_relations"] == 0
    )
    return report


class RebuildTool:
    """Interactive CLI for the offline VDB rebuild."""

    def __init__(self):
        self.graph = None
        self.entities_vdb = None
        self.relationships_vdb = None
        self.chunks_vdb = None
        self.text_chunks = None
        # The configuration storage: where each target's embedding baseline is
        # recorded AFTER that target's rebuild is durable and verified, and
        # never before. See docs/design/ConfigurationStorageContract.md.
        self.configuration_storage = None
        # Whether this run holds the configuration-directory claim; see
        # ``setup_storages``.
        self._holds_working_dir = False
        self.config_dir = ""
        # The shared anchor lock (``lightrag/kg/anchor_lock.py``) and the
        # anchor it guards, read before anything opens. The tool VERIFIES the
        # container's identity against it and never writes either.
        self._holds_anchor_lock = False
        self.anchor_working_dir = ""
        self.anchor = None
        self.global_config: Dict[str, Any] = {}
        self.embedding_func: EmbeddingFunc | None = None
        self.embedding_available = False
        self.workspace = ""
        self.batch_size = DEFAULT_BATCH_SIZE
        self.storage_names: Dict[str, str] = {}
        # Vector targets that refused initialization, label -> diagnostic.
        # Corrupt targets additionally retain the typed error, whose
        # ``artifacts`` name the files the recovery must preserve first.
        # Entries are cleared after confirmed recovery reinitializes a target.
        self.incompatible_vdbs: Dict[str, str] = {}
        self.corrupt_vdbs: Dict[str, CorruptStorageSnapshotError] = {}

    # ------------------------------------------------------------------
    # Configuration / setup
    # ------------------------------------------------------------------

    def resolve_storage_names(self) -> Dict[str, str]:
        kv = os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        return {
            "graph": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
            "vector": os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
            "kv": kv,
            # Its own category, resolved exactly as the server resolves it --
            # the tool must open the SAME container the server records into,
            # or its post-rebuild baseline lands where nothing reads it.
            "config": resolve_configuration_storage(
                os.getenv("LIGHTRAG_CONFIG_STORAGE", ""), kv_storage=kv
            ),
        }

    def resolve_config_dir(self) -> str:
        return resolve_config_dir(
            os.getenv("LIGHTRAG_CONFIG_DIR", ""),
            os.getenv("WORKING_DIR", DEFAULT_WORKING_DIR),
        )

    def check_env_vars(self, storage_name: str) -> None:
        """Warn about missing env vars (initialization is the real validation)."""
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            print(
                f"⚠️  Warning: {storage_name} normally requires: "
                f"{', '.join(missing_vars)} (may be provided via config.ini)"
            )

    def build_embedding_func(self) -> EmbeddingFunc | None:
        """Build the embedding function through the server's factory.

        Returns None when the api extra is unavailable; check-only mode
        still works without it.
        """
        try:
            from lightrag.api.config import global_args
            from lightrag.api.lightrag_server import (
                create_embedding_function_from_args,
            )
        except ImportError as e:
            print(f"\n⚠️  Could not import the LightRAG API package: {e}")
            print('   Rebuild requires the api extra: pip install "lightrag-hku[api]"')
            print("   Continuing in CHECK-ONLY mode (no embedding available).")
            return None

        embedding_func = create_embedding_function_from_args(global_args)
        print(
            f"- Embedding: binding={global_args.embedding_binding} "
            f"model={embedding_func.model_name} dim={embedding_func.embedding_dim}"
        )
        return embedding_func

    def build_global_config(self) -> Dict[str, Any]:
        global_config: Dict[str, Any] = {
            "working_dir": os.getenv("WORKING_DIR", DEFAULT_WORKING_DIR),
            # Backend selection, mirroring LightRAG._build_global_config. PG
            # storages derive enable_vector from global_config["vector_storage"],
            # so this must carry the real backend name for a mixed config like
            # PGGraphStorage + QdrantVectorDBStorage to resolve correctly. Keep all
            # three names for parity.
            "kv_storage": self.storage_names["kv"],
            "vector_storage": self.storage_names["vector"],
            "graph_storage": self.storage_names["graph"],
            # Read by the JSON backend when it opens the configuration
            # container; ignored by the server backends.
            "config_dir": self.config_dir,
            "embedding_batch_num": get_env_value(
                "EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM, int
            ),
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": get_env_value(
                    "COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, float
                )
            },
            "embedding_func": self.embedding_func,
        }

        # Content truncation parity with the server pipeline
        # (_truncate_vdb_content is a no-op when these keys are absent)
        max_token_size = getattr(self.embedding_func, "max_token_size", None)
        if max_token_size:
            try:
                from lightrag.utils import TiktokenTokenizer

                global_config["tokenizer"] = TiktokenTokenizer(
                    os.getenv("TIKTOKEN_MODEL_NAME", "gpt-4o-mini")
                )
                global_config["embedding_token_limit"] = max_token_size
            except Exception as e:
                logger.warning(f"Tokenizer unavailable, skipping truncation: {e}")
        return global_config

    async def setup_storages(self) -> bool:
        """Instantiate and initialize all storages. Returns False on failure."""
        from lightrag.kg.factory import get_storage_class

        self.storage_names = self.resolve_storage_names()
        self.config_dir = self.resolve_config_dir()
        self.workspace = os.getenv("WORKSPACE", "")

        # The anchor first (steps 0a-0c, as a start runs them): the shared
        # lock, a strict read, and a refusal when it binds another backend
        # type -- before the directory claim and before anything opens.
        if not self.preflight_anchor():
            return False

        # Claim the working directory FIRST, before building anything. This
        # tool is a SECOND process tree: a server running on the same
        # directory keeps its own in-memory copy of a file-backed
        # configuration namespace and publishes by rewriting the whole file,
        # so the two would overwrite each other's baselines -- and the record
        # this tool writes is the one that says the rebuild happened. The
        # confirmation prompt is not a substitute; it asks the operator, and
        # this asks the filesystem. Taken on the resolved storage NAME so the
        # refusal precedes the environment checks and the embedding function:
        # the answer does not depend on them, so neither should the wait.
        self._holds_working_dir = uses_working_dir(self.storage_names["config"])
        if self._holds_working_dir:
            try:
                acquire_working_dir_lock(self.config_dir)
            except WorkingDirectoryInUseError as e:
                self._holds_working_dir = False
                print(f"\n✗ {e}")
                return False

        print("\nChecking configuration...")
        for storage_name in set(self.storage_names.values()):
            self.check_env_vars(storage_name)

        self.embedding_func = self.build_embedding_func()
        self.embedding_available = self.embedding_func is not None
        if not self.embedding_available:
            # Vector storages require an embedding_func even for read paths;
            # use a stub that fails loudly if an embedding is ever requested.
            async def _no_embedding(*_args, **_kwargs):
                raise RuntimeError(
                    "Embedding is not available in check-only mode. "
                    'Install the api extra: pip install "lightrag-hku[api]"'
                )

            # model_name must match the server's embedding function: Qdrant /
            # PostgreSQL derive the collection/table name from
            # model_name + embedding_dim. Omitting it falls back to the legacy
            # name, so check-only mode would probe the wrong collection (and
            # could create an empty legacy one) and misreport records as missing.
            self.embedding_func = EmbeddingFunc(
                embedding_dim=get_env_value("EMBEDDING_DIM", 1024, int),
                func=_no_embedding,
                model_name=get_env_value("EMBEDDING_MODEL", None, special_none=True),
            )

        self.global_config = self.build_global_config()

        graph_cls = get_storage_class(self.storage_names["graph"])
        vector_cls = get_storage_class(self.storage_names["vector"])
        kv_cls = get_storage_class(self.storage_names["kv"])
        config_cls = get_storage_class(self.storage_names["config"])

        # Namespaces and meta_fields must match LightRAG's own storage setup
        self.graph = graph_cls(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
        )
        self.entities_vdb = vector_cls(
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        self.relationships_vdb = vector_cls(
            namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        self.chunks_vdb = vector_cls(
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"},
        )
        self.text_chunks = kv_cls(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
        )
        self.configuration_storage = create_configuration_storage(
            config_cls,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
        )

        print("\nInitializing storages...")
        try:
            # Authoritative sources first, on the server-identical path: any
            # failure here aborts, migrations included. Rebuilding vectors out
            # of a half-migrated graph or chunk store is worse than not
            # rebuilding. The configuration storage is a source too: a rebuild
            # that cannot record its baseline afterwards is not a clean
            # recovery, so failing to open it aborts here rather than after
            # the vectors were dropped.
            await self.configuration_storage.initialize()
            # The container must be the one the anchor binds before anything
            # else opens: a baseline recorded into the wrong container is a
            # rebuild that protects nothing.
            if not await self.verify_identity():
                return False
            for storage in (self.graph, self.text_chunks):
                await storage.initialize()
            # Vector targets, one at a time, tolerating ONLY the typed
            # embedding-space refusal or a typed corrupt snapshot.
            # Neither is deleted during setup. Corrupt files require a backup
            # after source checks and explicit rebuild confirmation.
            for label, vdb in self.vector_targets().items():
                try:
                    await vdb.initialize()
                except CorruptStorageSnapshotError as e:
                    # Recovery DESTROYS the container, so it is offered only to
                    # a raiser that named the local files to preserve first.
                    # A raiser with none (a server-backed store) cannot be
                    # preserved, so it must not be destroyed either.
                    if not e.artifacts:
                        raise
                    self.corrupt_vdbs[label] = e
                    self.incompatible_vdbs[label] = str(e)
                    print(f"⚠️  {label} snapshot is corrupt: {e}")
                except VectorSpaceMismatchError as e:
                    self.incompatible_vdbs[label] = str(e)
                    print(f"⚠️  {label} vector storage refused to attach: {e}")
        except Exception as e:
            print(f"✗ Storage initialization failed: {e}")
            for storage_name in set(self.storage_names.values()):
                required = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
                if required:
                    print(f"  {storage_name} requires: {', '.join(required)}")
            return False

        print(f"- Graph Storage:  {self.storage_names['graph']}")
        print(f"- Vector Storage: {self.storage_names['vector']}")
        print(f"- KV Storage:     {self.storage_names['kv']}")
        print(
            f"- Configuration:  "
            f"{describe_configuration_container(self.storage_names['config'], self.config_dir)}"
        )
        print(f"- Workspace:      {self.workspace if self.workspace else '(default)'}")
        print(f"- Working Dir:    {self.global_config['working_dir']}")
        print("- Connection Status: ✓ Success")
        if not await self.print_baselines():
            return False
        if self.incompatible_vdbs:
            print(
                f"\n{BOLD_RED}⚠️  {len(self.incompatible_vdbs)} vector storage(s) cannot attach: "
                f"incompatible vectors or corrupt snapshots{RESET}"
            )
            for label in self.incompatible_vdbs:
                print(f"    - {label}")
            print(
                "  They are unusable until rebuilt. A rebuild (menu options 2-4)\n"
                "  drops the incompatible container and re-embeds from the\n"
                "  authoritative sources; nothing else can repair them."
            )
        return True

    def preflight_anchor(self) -> bool:
        """Take the shared anchor lock and refuse on a backend-type mismatch.

        Refuses exactly as a start would. Never creates or rewrites the
        anchor: only a server or SDK start binds.
        """
        self.anchor_working_dir = os.getenv("WORKING_DIR", DEFAULT_WORKING_DIR)
        try:
            acquire_anchor_lock_shared(self.anchor_working_dir)
        except ConfigurationAnchorLockError as e:
            print(f"\n✗ {e}")
            return False
        self._holds_anchor_lock = True
        try:
            self.anchor = preflight_configuration_anchor(
                working_dir=self.anchor_working_dir,
                backend=self.storage_names["config"],
                container=describe_configuration_container(
                    self.storage_names["config"], self.config_dir
                ),
            )
        except ConfigurationStorageError as e:
            print(f"\n✗ {e}")
            self.release_anchor_lock()
            return False
        return True

    async def verify_identity(self) -> bool:
        """Verify the configuration container against the anchor; write nothing."""
        try:
            binding = await verify_configuration_identity(
                self.configuration_storage,
                self.anchor,
                working_dir=self.anchor_working_dir,
                backend=self.storage_names["config"],
                container=describe_configuration_container(
                    self.storage_names["config"], self.config_dir
                ),
            )
        except ConfigurationStorageError as e:
            print(f"✗ {e}")
            return False
        if binding.action == "unanchored":
            print(
                "- Storage identity: no anchor on record, so the configuration "
                "container's identity was not verified (only a server start "
                "binds one)"
            )
        else:
            print(f"- Storage identity: {binding.storage_uuid} (matches the anchor)")
        return True

    def release_anchor_lock(self) -> None:
        if self._holds_anchor_lock:
            self._holds_anchor_lock = False
            release_anchor_lock_shared(self.anchor_working_dir)

    async def print_baselines(self) -> bool:
        """Report each target's recorded embedding baseline against the config.

        Diagnostic: the server refuses to START on a recorded mismatch, and
        this tool is the way out of that refusal, so it lists what the server
        would refuse on rather than refusing itself. A configuration read that
        cannot complete aborts (returns False): a rebuild that could not record
        its baseline afterwards would leave the server refusing anyway.
        """
        try:
            recorded = await read_embedding_baselines(
                self.configuration_storage, self.workspace
            )
        except ConfigurationStorageError as e:
            print(f"✗ Could not read the recorded embedding baselines: {e}")
            return False
        print("- Embedding baselines (recorded -> configured):")
        mismatched = []
        for target in EMBEDDING_TARGETS:
            baseline = recorded.get(target)
            if baseline is None:
                print(f"    {target:14s} (none recorded)")
                continue
            differs = baseline.differs_from(self.embedding_func)
            flag = "  ✗ MISMATCH" if differs else ""
            print(
                f"    {target:14s} model={baseline.model!r} dim={baseline.dim} "
                f"origin={baseline.origin}{flag}"
            )
            if differs:
                mismatched.append(target)
        if mismatched:
            print(
                f"\n{BOLD_RED}⚠️  The server refuses to start this workspace: "
                f"{', '.join(mismatched)} were adopted under another embedding "
                f"space.{RESET}\n  Rebuilding them (menu options 2-4) re-embeds "
                f"from the authoritative sources and records the configured "
                f"space as their baseline."
            )
        return True

    async def commit_baseline(self, label: str, stats: Dict[str, Any]) -> None:
        """Record ``label``'s baseline once its rebuild is durable and verified.

        Configuration LAST: the record is written only when every batch and
        every flush of that one target succeeded, and only that target's
        record moves. A rebuild whose baseline could not be recorded is
        reported as a failed rebuild -- the stale record keeps the server
        refusing, which is the safe direction, and re-running converges.

        "Every flush succeeded" is not what a returning ``index_done_callback``
        proves. A per-item backend keeps its retryable failures (408/429/5xx)
        buffered and returns normally, so the last flush of a rebuild can leave
        vectors that never reached the server with nothing left to retry them.
        The baseline would then say the target was adopted in the configured
        space while its index is incomplete -- and no later check catches that:
        the startup precheck sees a matching record, and the coverage gate only
        refuses an EMPTY index. So the storage is asked directly, and a
        retained operation is a failed rebuild.
        """
        if stats["errors"]:
            print(
                f"  ⚠️  {label}: baseline NOT recorded -- the rebuild reported "
                f"errors, so the previous record stays and the server keeps "
                f"refusing until a clean rebuild."
            )
            return
        if not await self._vectors_are_durable(label, stats):
            return
        try:
            baseline = await record_embedding_baseline(
                self.configuration_storage,
                workspace=self.workspace,
                target=label,
                embedding_func=self.embedding_func,
            )
        except ConfigurationStorageError as e:
            print(f"  ✗ {label}: rebuilt, but the baseline could not be recorded: {e}")
            stats["errors"].append(
                {
                    "batch": "baseline",
                    "records_lost": 0,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                }
            )
            return
        print(
            f"  ✓ {label}: baseline recorded (model={baseline.model!r} "
            f"dim={baseline.dim})"
        )

    async def _vectors_are_durable(self, label: str, stats: Dict[str, Any]) -> bool:
        """Whether ``label``'s vector storage retained nothing after its flush.

        Records an error on ``stats`` and returns False when something is still
        buffered, or when the storage could not answer -- an unreadable answer
        is not a durable one. A backend that buffers nothing answers False and
        costs a method call.
        """
        vdb = self.vector_targets().get(label)
        has_pending = getattr(vdb, "has_pending_index_ops", None)
        if has_pending is None:
            return True
        try:
            retained = bool(await has_pending(include_deletes=True))
        except Exception as e:
            print(
                f"  ✗ {label}: rebuilt, but the vector storage could not say "
                f"whether anything is still buffered: {e}"
            )
            stats["errors"].append(
                {
                    "batch": "durability",
                    "records_lost": 0,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                }
            )
            return False
        if not retained:
            return True
        print(
            f"  ✗ {label}: rebuilt, but the vector storage still holds "
            f"operations its last flush could not deliver -- the baseline is "
            f"NOT recorded, so the server keeps refusing until a clean rebuild."
        )
        stats["errors"].append(
            {
                "batch": "durability",
                "records_lost": 0,
                "error_type": "RetainedVectorOps",
                "error_msg": (
                    "the vector storage retained operations after its final "
                    "flush; the rebuilt index is incomplete"
                ),
            }
        )
        return False

    def vector_targets(self) -> Dict[str, Any]:
        """The three rebuild targets, keyed by the label used in reports."""
        return {
            "entities": self.entities_vdb,
            "relationships": self.relationships_vdb,
            "chunks": self.chunks_vdb,
        }

    async def recover_incompatible(self, labels: List[str]) -> None:
        """Drop + re-initialize each named target that refused to attach.

        The caller must check sources and obtain explicit confirmation first.
        A corrupt target's files are backed up before any destructive action.
        Runs immediately before the rebuild of those targets, so a refused
        container is destroyed only once the operator has confirmed the
        rebuild. The rebuild helpers drop again straight after; ``drop()`` is
        idempotent, and paying it twice is cheaper than threading the refusal
        state through the library API.
        """
        targets = self.vector_targets()
        for label in labels:
            if label not in self.incompatible_vdbs:
                continue
            if label in self.corrupt_vdbs:
                error = self.corrupt_vdbs[label]
                logger.warning(
                    f"Rebuild {label}: backing up the corrupt files before the "
                    f"rebuild drops them ({', '.join(error.artifacts)})"
                )
                try:
                    for backup in backup_corrupt_snapshot(error.artifacts):
                        print(f"  ✓ {label}: corrupt file preserved at {backup}")
                except PrivateFileError as privacy_error:
                    # The refusal alone would leave the operator stuck: every
                    # re-run reaches this same branch. Name the way out that
                    # actually works -- an absent container reads as a first
                    # start, so moving the files aside turns this into an
                    # ordinary rebuild with nothing left to preserve.
                    raise PrivateFileError(
                        f"{privacy_error}\n\nNothing has been changed. To "
                        f"recover {label} by hand, move these files somewhere "
                        f"you control and re-run this tool -- a container that "
                        f"is absent reads as a first start, so the rebuild "
                        f"then proceeds normally:\n"
                        + "\n".join(f"    {artifact}" for artifact in error.artifacts)
                    ) from privacy_error
                await _drop_vdb(targets[label], label)
                await targets[label].initialize()
                logger.info(f"Rebuild {label}: corrupt container re-provisioned")
                del self.corrupt_vdbs[label]
                del self.incompatible_vdbs[label]
                print(f"  ✓ {label}: corrupt container dropped and re-provisioned")
                continue
            await clear_vector_space_refusal(targets[label], label)
            del self.incompatible_vdbs[label]
            print(f"  ✓ {label}: incompatible container dropped and re-provisioned")

    def all_storages(self):
        return [
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.text_chunks,
            self.configuration_storage,
        ]

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    def print_header(self):
        print("\n" + "=" * 60)
        print(f"{BOLD_CYAN}LightRAG Offline Vector Storage Rebuild Tool{RESET}")
        print("=" * 60)
        print("\nAuthoritative sources: graph storage + text_chunks KV store")
        print("Targets: entities_vdb, relationships_vdb, chunks_vdb")
        print("\n" + "=" * 60)
        print(f"{BOLD_RED}⚠️  IMPORTANT: STOP THE LIGHTRAG SERVER FIRST{RESET}")
        print("=" * 60)
        print("\nThis tool drops and rewrites vector storages. Running it while")
        print("the LightRAG Server (or any other writer) is active can corrupt")
        print("data or silently lose concurrent updates - for ALL backends.")

    def confirm_server_stopped(self) -> bool:
        confirm = (
            input("\nHas the LightRAG Server been shut down? (yes/no): ")
            .strip()
            .lower()
        )
        if confirm != "yes":
            print("\n✓ Operation cancelled - please shut down the server first")
            return False
        return True

    def make_progress_printer(self, label: str) -> ProgressCallback:
        def _print_progress(done: int, total: int):
            total = max(total, 1)
            bar_length = 40
            filled = int(bar_length * done / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(
                f"\r  {label}: [{bar}] {done}/{total} batches",
                end="" if done < total else "\n",
                flush=True,
            )

        return _print_progress

    def print_rebuild_section(self, label: str) -> None:
        """Visually separate each vector storage's rebuild output.

        The per-storage drop/flush logs (and the progress bar) otherwise run
        together across entities/relationships/chunks; this header marks where
        one storage's rebuild starts.
        """
        print(f"\n{BOLD_CYAN}{'─' * 60}{RESET}")
        print(f"{BOLD_CYAN}▶ Rebuilding {label} vector storage{RESET}")
        print(f"{BOLD_CYAN}{'─' * 60}{RESET}")

    def print_rebuild_stats(self, stats: Dict[str, Any]):
        print(f"\n  {BOLD_CYAN}{stats['label']}{RESET}:")
        print(f"    Source records:  {stats['source_total']:,}")
        print(f"    Rebuilt:         {stats['rebuilt']:,}")
        if stats["skipped"]:
            print(f"    Skipped (dirty): {stats['skipped']:,}")
        if stats["duplicates"]:
            print(f"    Deduplicated:    {stats['duplicates']:,}")
        if stats["errors"]:
            print(f"    {BOLD_RED}Failed batches:  {stats['failed_batches']}{RESET}")
            for err in stats["errors"][:5]:
                print(
                    f"      - batch {err['batch']}: {err['error_type']}: "
                    f"{err['error_msg'][:120]}"
                )
            if len(stats["errors"]) > 5:
                print(f"      ... and {len(stats['errors']) - 5} more")

    def print_check_report(self, report: Dict[str, Any]):
        incompatible = report.get("incompatible") or {}
        print("\n" + "=" * 60)
        print("📊 Consistency Report (graph -> vector storage)")
        print("=" * 60)
        if incompatible:
            print(
                f"\n{BOLD_RED}✗ Vector storage refused to attach — these storages were "
                f"not probed:{RESET}"
            )
            for label, message in incompatible.items():
                print(f"    - {label}: {message}")
            print(
                "\n  Their containers hold another embedding space's vectors, or\n"
                "  cannot be read back at all, so every record is unreachable by\n"
                "  construction and a 'missing' count for them would say nothing\n"
                "  about drift. A rebuild (menu options 2-4) is required, not\n"
                "  optional."
            )
        print(f"  Graph entities:    {report['graph_entities']:,}")
        print(f"  Graph relations:   {report['graph_relations']:,}")
        if "entities" not in incompatible:
            print(f"  Missing entities:  {report['missing_entities']:,}")
        if "relationships" not in incompatible:
            print(f"  Missing relations: {report['missing_relations']:,}")
        if report["missing_entity_names"]:
            print("\n  Missing entities (first few):")
            for name in report["missing_entity_names"]:
                print(f"    - {name}")
        if report["missing_relation_pairs"]:
            print("\n  Missing relations (first few):")
            for pair in report["missing_relation_pairs"]:
                print(f"    - {pair}")
        if report["consistent"]:
            print(f"\n{BOLD_GREEN}✓ No missing vector records detected.{RESET}")
            print("  Note: this check only covers the graph -> VDB direction; stale")
            print(
                "  VDB-only records (reverse orphans) require a full rebuild to clear."
            )
        elif incompatible:
            print(f"\n{BOLD_RED}✗ Vector storage unusable (see above).{RESET}")
            print("  Run a rebuild (menu options 2-4) to restore service.")
        else:
            print(f"\n{BOLD_RED}✗ Inconsistencies detected.{RESET}")
            print("  Run a rebuild (menu options 2-4) to restore consistency.")

    async def print_source_counts(self, include_graph: bool, include_chunks: bool):
        if include_graph:
            nodes = await self.graph.get_all_nodes()
            edges = await self.graph.get_all_edges()
            print(f"  Graph nodes: {len(nodes):,}")
            print(f"  Graph edges: {len(edges):,} (before deduplication)")
        if include_chunks:
            chunk_ids = await enumerate_kv_keys(self.text_chunks)
            print(f"  Text chunks: {len(chunk_ids):,}")

    def confirm_rebuild(self, targets: str) -> bool:
        print("\n" + "=" * 60)
        print(f"{BOLD_RED}⚠️  WARNING: {targets} will be DROPPED and rebuilt!{RESET}")
        print("=" * 60)
        print("\nAll affected records will be re-embedded, which may incur")
        print("significant embedding API cost and time on large datasets.")
        print("If interrupted, simply re-run this tool (sources are read-only).")
        if self.corrupt_vdbs:
            print("Selected corrupt files will be backed up before deletion.")
            print("Verify the source counts and source integrity before proceeding;")
            print("a readable source is not proof that it contains every lost row.")
        confirm = input("\nProceed with the rebuild? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("\n✓ Rebuild cancelled")
            return False
        return True

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    async def run_check(self):
        print("\nRunning consistency check (read queries only; no rebuild)...")
        start = time.time()
        report = await check_vdb_consistency(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            batch_size=self.batch_size,
            incompatible=self.incompatible_vdbs,
        )
        self.print_check_report(report)
        print(f"\n(check took {time.time() - start:.1f}s)")

    async def run_rebuild_entities_relations(self) -> List[Dict[str, Any]]:
        all_stats = []
        self.print_rebuild_section("entities")
        stats = await rebuild_entities_vdb(
            self.graph,
            self.entities_vdb,
            self.global_config,
            batch_size=self.batch_size,
            progress_callback=self.make_progress_printer("entities"),
        )
        await self.commit_baseline("entities", stats)
        all_stats.append(stats)
        self.print_rebuild_section("relationships")
        stats = await rebuild_relationships_vdb(
            self.graph,
            self.relationships_vdb,
            self.global_config,
            batch_size=self.batch_size,
            progress_callback=self.make_progress_printer("relationships"),
        )
        await self.commit_baseline("relationships", stats)
        all_stats.append(stats)
        return all_stats

    async def run_rebuild_chunks(self) -> List[Dict[str, Any]]:
        self.print_rebuild_section("chunks")
        stats = await rebuild_chunks_vdb(
            self.text_chunks,
            self.chunks_vdb,
            batch_size=self.batch_size,
            progress_callback=self.make_progress_printer("chunks"),
        )
        await self.commit_baseline("chunks", stats)
        return [stats]

    def report_rebuild(self, all_stats: List[Dict[str, Any]]) -> bool:
        """Print the rebuild report and return True if any batch/flush failed."""
        print("\n" + "=" * 60)
        print("📊 Rebuild Report")
        print("=" * 60)
        had_errors = False
        for stats in all_stats:
            self.print_rebuild_stats(stats)
            had_errors = had_errors or bool(stats["errors"])
        print()
        if had_errors:
            print(f"{BOLD_RED}⚠️  Rebuild finished with errors (see above).{RESET}")
            print("   Sources were not modified - re-run this tool to retry.")
        else:
            print(f"{BOLD_GREEN}✓ Rebuild completed successfully.{RESET}")
        return had_errors

    async def run(self) -> bool:
        """Run the interactive tool. Returns True on success, False on failure.

        Failure (-> non-zero exit) covers storage-init failure, any unhandled
        exception, interruption, and a rebuild that finished with batch/flush
        errors. A clean user cancellation (server not stopped) is a success.
        This is a disaster-recovery entry point, so a partial rebuild after the
        target VDB was already dropped must NOT look like a clean recovery.
        """
        success = True
        try:
            # Initialize shared storage (REQUIRED for storage classes to work)
            from lightrag.kg.shared_storage import initialize_share_data

            initialize_share_data(workers=1)

            self.print_header()
            if not self.confirm_server_stopped():
                return True  # deliberate user cancellation, not a failure

            if not await self.setup_storages():
                return False

            while True:
                print("\n=== Rebuild Options ===")
                print("[1] Consistency check (diagnose only; no rebuild)")
                if self.embedding_available:
                    print("[2] Rebuild entities + relationships VDB")
                    print("[3] Rebuild chunks VDB")
                    print("[4] Rebuild ALL vector storages")
                else:
                    print("[2-4] (unavailable - embedding requires the api extra)")
                print("[0] Exit")

                choice = input("\nSelect option: ").strip()
                if choice == "" or choice == "0":
                    print("\n✓ Exiting")
                    return success
                if choice == "1":
                    await self.run_check()
                    continue
                if choice not in ("2", "3", "4"):
                    print("✗ Invalid choice. Please enter 0, 1, 2, 3 or 4")
                    continue
                if not self.embedding_available:
                    print(
                        "✗ Rebuild unavailable in check-only mode. "
                        'Install the api extra: pip install "lightrag-hku[api]"'
                    )
                    continue

                include_graph = choice in ("2", "4")
                include_chunks = choice in ("3", "4")
                targets = {
                    "2": "entities_vdb + relationships_vdb",
                    "3": "chunks_vdb",
                    "4": "ALL vector storages",
                }[choice]

                print("\nCounting source records...")
                await self.print_source_counts(include_graph, include_chunks)

                if not self.confirm_rebuild(targets):
                    continue

                start = time.time()
                selected = (["entities", "relationships"] if include_graph else []) + (
                    ["chunks"] if include_chunks else []
                )
                await self.recover_incompatible(selected)
                all_stats: List[Dict[str, Any]] = []
                if include_graph:
                    all_stats.extend(await self.run_rebuild_entities_relations())
                if include_chunks:
                    all_stats.extend(await self.run_rebuild_chunks())
                # Sticky: once any rebuild reports errors the session is a
                # failure, even if a later retry succeeds (a partial rebuild
                # after a drop must surface a non-zero exit to automation).
                if self.report_rebuild(all_stats):
                    success = False
                print(f"(rebuild took {time.time() - start:.1f}s)")

        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
            return False
        except Exception as e:
            print(f"\n✗ Rebuild tool failed: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            for storage in self.all_storages():
                if storage is not None:
                    try:
                        await storage.finalize()
                    except Exception:
                        pass
            try:
                from lightrag.kg.shared_storage import finalize_share_data

                finalize_share_data()
            except Exception:
                pass
            # Last, after every storage that writes under it is down.
            if self._holds_working_dir:
                self._holds_working_dir = False
                release_working_dir_lock(self.config_dir)
            # Taken before the directory claim, so given back after it.
            self.release_anchor_lock()


async def async_main() -> bool:
    """Async main entry point. Returns True on success, False on failure."""
    tool = RebuildTool()
    return await tool.run()


def main():
    """Synchronous entry point for CLI command.

    Exits non-zero on failure (storage-init failure, unhandled error,
    interruption, or a rebuild that finished with errors) so automation and
    operators do not mistake a partial/failed recovery for a clean one.
    """
    # Load environment and configure logging only when run as a tool, never on import.
    load_dotenv(dotenv_path=".env", override=False)
    setup_logger("lightrag", level="INFO")
    success = asyncio.run(async_main())
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
