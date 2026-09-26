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
re-provisions it in the current embedding space, and re-initializes it. Only
the typed refusal is tolerated: a cluster outage, a bad credential or a
corrupt file still aborts, because dropping a vector storage on a false
positive destroys data the graph may not be able to rebuild.

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
import time
from typing import Any, AsyncIterator, Callable, Dict, List

from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.constants import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
)
from lightrag.exceptions import StorageCapabilityError, VectorSpaceMismatchError
from lightrag.kg import STORAGE_ENV_REQUIREMENTS
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

ProgressCallback = Callable[[int, int | None], None]


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
    flush succeeds, so it never overstates what was actually persisted.
    """
    if stats["staged"] == 0:
        return
    label = stats["label"]
    try:
        await vdb.index_done_callback()
        stats["rebuilt"] += stats["staged"]
    except Exception as e:
        logger.error(
            f"Rebuild {label}: flush of {stats['staged']} staged record(s) failed: {e}"
        )
        stats["failed_batches"] += 1
        stats["errors"].append(
            {
                "batch": f"flush@batch-{stats['batches']}",
                "records_lost": stats["staged"],
                "error_type": type(e).__name__,
                "error_msg": str(e),
            }
        )
    finally:
        stats["staged"] = 0


async def _upsert_batch(
    vdb,
    batch_payload: Dict[str, Dict[str, Any]],
    batch_no: int,
    total_batches: int | None,
    stats: Dict[str, Any],
) -> None:
    """Upsert one batch; collect the error and continue on persistent failure."""
    label = stats["label"]
    batch_label = (
        f"{batch_no}/{total_batches}" if total_batches is not None else str(batch_no)
    )
    try:
        await safe_vdb_operation_with_exception(
            operation=lambda payload=batch_payload: vdb.upsert(payload),
            operation_name=f"rebuild_{label}_upsert",
            entity_name=f"batch {batch_label}",
            max_retries=3,
            retry_delay=0.2,
        )
        stats["staged"] += len(batch_payload)
    except Exception as e:
        logger.error(f"Rebuild {label}: batch {batch_label} failed: {e}")
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


async def _preflight_batches(
    batches: AsyncIterator[list[Any]],
) -> tuple[list[Any] | None, AsyncIterator[list[Any]]]:
    """Advance a bounded source iterator before any destructive target drop."""
    try:
        first = await anext(batches)
    except StopAsyncIteration:
        first = None
    return first, batches


async def _stream_upsert(
    vdb,
    source_batches: AsyncIterator[list[Any]],
    first_batch: list[Any] | None,
    stats: Dict[str, Any],
    prepare_batch,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Drop only after preflight, then transform and upsert bounded batches."""
    await _drop_vdb(vdb, stats["label"])
    batch_no = 0

    async def consume(source_batch: list[Any]) -> None:
        nonlocal batch_no
        batch_no += 1
        stats["source_total"] += len(source_batch)
        payload = await prepare_batch(source_batch, stats)
        stats["prepared"] += len(payload)
        if payload:
            await _upsert_batch(vdb, payload, batch_no, None, stats)
        if progress_callback:
            progress_callback(batch_no, None)

    if first_batch is not None:
        await consume(first_batch)
    async for source_batch in source_batches:
        await consume(source_batch)
    if progress_callback:
        progress_callback(batch_no, batch_no)
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

    batches = graph.iter_labels(batch_size)
    first_batch, batches = await _preflight_batches(batches)
    stats = _new_stats("entities", 0)

    async def prepare(labels: list[str], stats: Dict[str, Any]):
        nodes = await graph.get_nodes_batch(
            [str(label) for label in labels if label is not None]
        )
        payloads: Dict[str, Dict[str, Any]] = {}
        for label in labels:
            if label is None:
                stats["skipped"] += 1
                continue
            entity_name = str(label)
            node = nodes.get(label) or nodes.get(entity_name)
            if not isinstance(node, dict) or not entity_name.strip():
                stats["skipped"] += 1
                logger.warning(f"Rebuild entities: skipping unreadable node {label!r}")
                continue
            description = node.get("description") or ""
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
            if entity_vdb_id in payloads:
                stats["duplicates"] += 1
                continue
            payloads[entity_vdb_id] = {
                "entity_name": entity_name,
                "entity_type": node.get("entity_type") or "",
                "content": _truncate_vdb_content(
                    f"{entity_name}\n{description}",
                    global_config,
                    f"entity:{entity_name}",
                ),
                "source_id": node.get("source_id") or "",
                "description": description,
                "file_path": node.get("file_path") or "",
            }
        return payloads

    return await _stream_upsert(
        entities_vdb,
        batches,
        first_batch,
        stats,
        prepare,
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

    batches = graph.iter_edges(batch_size)
    first_batch, batches = await _preflight_batches(batches)
    stats = _new_stats("relationships", 0)

    async def prepare(edges: list[dict], stats: Dict[str, Any]):
        payloads: Dict[str, Dict[str, Any]] = {}
        for edge in edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src is None or tgt is None or not str(src).strip() or not str(tgt).strip():
                stats["skipped"] += 1
                continue
            src_id, tgt_id = sorted((str(src), str(tgt)))
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
            payloads[rel_vdb_id] = {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": edge.get("source_id") or "",
                "content": _truncate_vdb_content(
                    f"{keywords}\t{src_id}\n{tgt_id}\n{description}",
                    global_config,
                    f"relationship:{src_id}-{tgt_id}",
                ),
                "keywords": keywords,
                "description": description,
                "weight": weight,
                "file_path": edge.get("file_path") or "",
            }
        return payloads

    return await _stream_upsert(
        relationships_vdb,
        batches,
        first_batch,
        stats,
        prepare,
        progress_callback=progress_callback,
    )


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
    batches = text_chunks_kv.iter_keys(batch_size)
    first_batch, batches = await _preflight_batches(batches)
    stats = _new_stats("chunks", 0)

    async def prepare(keys: list[str], stats: Dict[str, Any]):
        batch_ids = [str(key) for key in keys]
        records = await text_chunks_kv.get_by_ids(batch_ids)
        record_map: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if record is None:
                continue
            if not isinstance(record, dict) or not record.get("_id"):
                raise RuntimeError("Chunk source returned an unidentifiable record")
            record_id = str(record["_id"])
            if record_id not in batch_ids or record_id in record_map:
                raise RuntimeError("Chunk source returned duplicate or unexpected ids")
            record_map[record_id] = record

        batch_payload: Dict[str, Dict[str, Any]] = {}
        for chunk_id in batch_ids:
            record = record_map.get(chunk_id)
            if record is None:
                raise RuntimeError("Chunk source returned an incomplete batch")
            if not record.get("content"):
                stats["skipped"] += 1
                continue
            payload = dict(record)
            payload.pop("_id", None)
            payload.setdefault("full_doc_id", "")
            payload.setdefault("file_path", "")
            batch_payload[chunk_id] = payload
        return batch_payload

    return await _stream_upsert(
        chunks_vdb,
        batches,
        first_batch,
        stats,
        prepare,
        progress_callback=progress_callback,
    )


async def check_vdb_consistency(
    graph,
    entities_vdb,
    relationships_vdb,
    *,
    text_chunks_kv=None,
    chunks_vdb=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    incompatible: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Read-only, bounded, bidirectional consistency diagnosis."""
    incompatible = dict(incompatible or {})
    report: Dict[str, Any] = {"targets": {}, "incompatible": incompatible}

    async def identify(records, requested: set[str]) -> dict[str, dict]:
        if len(records) != len(requested):
            raise StorageCapabilityError("incomplete_probe_response")
        found: dict[str, dict] = {}
        for record in records:
            if record is None:
                continue
            if not isinstance(record, dict) or record.get("id") is None:
                raise StorageCapabilityError("unidentifiable_probe_record")
            record_id = str(record["id"])
            if record_id not in requested or record_id in found:
                raise StorageCapabilityError("duplicate_or_unexpected_probe_id")
            found[record_id] = record
        return found

    async def evaluate(label, vdb, batches, candidates_for, display_for):
        target = {
            "status": "inconclusive",
            "source_count": 0,
            "target_count": None,
            "missing": 0,
            "missing_examples": [],
            "reason": None,
        }
        report["targets"][label] = target
        blocked_status = None
        if label in incompatible:
            blocked_status = "incompatible"
            target.update(status=blocked_status, reason=incompatible[label])
        elif not getattr(vdb, "persists_vectors", True):
            blocked_status = "not_applicable"
            target.update(status=blocked_status, reason="non_persisting_backend")
        try:
            async for source_batch in batches:
                logical = []
                requested: list[str] = []
                for item in source_batch:
                    candidate_ids = [str(value) for value in candidates_for(item)]
                    if not candidate_ids:
                        continue
                    logical.append((candidate_ids, display_for(item)))
                    requested.extend(candidate_ids)
                requested = list(dict.fromkeys(requested))
                target["source_count"] += len(logical)
                if blocked_status is not None:
                    continue
                found = await identify(
                    await vdb.get_by_ids(requested), set(requested)
                )
                for candidate_ids, display in logical:
                    if not any(candidate in found for candidate in candidate_ids):
                        target["missing"] += 1
                        if len(target["missing_examples"]) < MAX_REPORTED_MISSING:
                            target["missing_examples"].append(display)
            if blocked_status is not None:
                return
            target["target_count"] = await vdb.get_exact_count()
        except Exception as exc:
            target["reason"] = (str(exc) or type(exc).__name__)[:240]
            return
        target["status"] = (
            "consistent"
            if target["missing"] == 0
            and target["source_count"] == target["target_count"]
            else "inconsistent"
        )

    await evaluate(
        "entities",
        entities_vdb,
        graph.iter_labels(batch_size),
        lambda label: [compute_mdhash_id(str(label), prefix="ent-")],
        lambda label: str(label),
    )
    await evaluate(
        "relationships",
        relationships_vdb,
        graph.iter_edges(batch_size),
        lambda edge: make_relation_vdb_ids(str(edge["source"]), str(edge["target"])),
        lambda edge: f"{edge['source']} ~ {edge['target']}",
    )
    if text_chunks_kv is not None and chunks_vdb is not None:
        await evaluate(
            "chunks",
            chunks_vdb,
            text_chunks_kv.iter_keys(batch_size),
            lambda key: [str(key)],
            lambda key: str(key),
        )

    # Keep the established summary fields for library callers while exposing
    # the richer per-target contract above.
    entities = report["targets"]["entities"]
    relationships = report["targets"]["relationships"]
    report.update(
        graph_entities=entities["source_count"],
        graph_relations=relationships["source_count"],
        missing_entities=entities["missing"],
        missing_relations=relationships["missing"],
        missing_entity_names=entities["missing_examples"],
        missing_relation_pairs=relationships["missing_examples"],
        consistent=all(
            target["status"] in ("consistent", "not_applicable")
            for target in report["targets"].values()
        ),
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
        self.global_config: Dict[str, Any] = {}
        self.embedding_func: EmbeddingFunc | None = None
        self.embedding_available = False
        self.workspace = ""
        self.batch_size = DEFAULT_BATCH_SIZE
        self.storage_names: Dict[str, str] = {}
        # Vector targets whose initialize() refused with
        # VectorSpaceMismatchError, label -> refusal message. Cleared per
        # target by clear_vector_space_refusal() once its container has been
        # dropped and re-provisioned in the current embedding space.
        self.incompatible_vdbs: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Configuration / setup
    # ------------------------------------------------------------------

    def resolve_storage_names(self) -> Dict[str, str]:
        return {
            "graph": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
            "vector": os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
            "kv": os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        }

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
            "working_dir": os.getenv("WORKING_DIR", "./rag_storage"),
            # Backend selection, mirroring LightRAG._build_global_config. PG
            # storages derive enable_vector from global_config["vector_storage"],
            # so this must carry the real backend name for a mixed config like
            # PGGraphStorage + QdrantVectorDBStorage to resolve correctly. Keep all
            # three names for parity.
            "kv_storage": self.storage_names["kv"],
            "vector_storage": self.storage_names["vector"],
            "graph_storage": self.storage_names["graph"],
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
        self.workspace = os.getenv("WORKSPACE", "")

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

        print("\nInitializing storages...")
        try:
            # Authoritative sources first, on the server-identical path: any
            # failure here aborts, migrations included. Rebuilding vectors out
            # of a half-migrated graph or chunk store is worse than not
            # rebuilding.
            for storage in (self.graph, self.text_chunks):
                await storage.initialize()
            # Vector targets, one at a time, tolerating ONLY the typed
            # embedding-space refusal. This is the condition the tool exists to
            # clear, so aborting on it would leave the operator with no
            # sanctioned way out; every other failure still aborts, because
            # dropping a vector storage on a false positive destroys data.
            for label, vdb in self.vector_targets().items():
                try:
                    await vdb.initialize()
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
        print(f"- Workspace:      {self.workspace if self.workspace else '(default)'}")
        print(f"- Working Dir:    {self.global_config['working_dir']}")
        print("- Connection Status: ✓ Success")
        if self.incompatible_vdbs:
            print(
                f"\n{BOLD_RED}⚠️  {len(self.incompatible_vdbs)} vector storage(s) hold "
                f"vectors from a different embedding space:{RESET}"
            )
            for label in self.incompatible_vdbs:
                print(f"    - {label}")
            print(
                "  They are unusable until rebuilt. A rebuild (menu options 2-4)\n"
                "  drops the incompatible container and re-embeds from the\n"
                "  authoritative sources; nothing else can repair them."
            )
        return True

    def vector_targets(self) -> Dict[str, Any]:
        """The three rebuild targets, keyed by the label used in reports."""
        return {
            "entities": self.entities_vdb,
            "relationships": self.relationships_vdb,
            "chunks": self.chunks_vdb,
        }

    async def recover_incompatible(self, labels: List[str]) -> None:
        """Drop + re-initialize each named target that refused to attach.

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
        def _print_progress(done: int, total: int | None):
            if total is None:
                print(f"\r  {label}: processed {done} batches", end="", flush=True)
                return
            if total == 0:
                print(f"\r  {label}: 0/0 batches")
                return
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
        print("📊 Consistency Report")
        print("=" * 60)
        if incompatible:
            print(
                f"\n{BOLD_RED}✗ Embedding space mismatch — these vector storages were "
                f"not probed:{RESET}"
            )
            for label, message in incompatible.items():
                print(f"    - {label}: {message}")
            print(
                "\n  Their stored vectors belong to a different embedding model or\n"
                "  dimension, so 'missing' counts for them would say nothing about\n"
                "  drift — every record is unreachable by construction. A rebuild\n"
                "  (menu options 2-4) is required, not optional."
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
        print("\n  Per-target status:")
        for label, target in report["targets"].items():
            counts = (
                f"source={target['source_count']:,}, target={target['target_count']:,}"
                if target["target_count"] is not None
                else f"source={target['source_count']:,}, target=unknown"
            )
            reason = f" ({target['reason']})" if target["reason"] else ""
            print(f"    - {label}: {target['status']} [{counts}]{reason}")
        if report["consistent"]:
            print(f"\n{BOLD_GREEN}✓ Every applicable target is consistent.{RESET}")
        elif incompatible:
            print(f"\n{BOLD_RED}✗ Vector storage unusable (see above).{RESET}")
            print("  Run a rebuild (menu options 2-4) to restore service.")
        else:
            print(f"\n{BOLD_RED}✗ Inconsistencies detected.{RESET}")
            print("  Run a rebuild (menu options 2-4) to restore consistency.")

    async def print_source_counts(self, include_graph: bool, include_chunks: bool):
        if include_graph:
            node_count = 0
            async for batch in self.graph.iter_labels(self.batch_size):
                node_count += len(batch)
            edge_count = 0
            async for batch in self.graph.iter_edges(self.batch_size):
                edge_count += len(batch)
            print(f"  Graph nodes: {node_count:,}")
            print(f"  Graph edges: {edge_count:,}")
        if include_chunks:
            chunk_count = 0
            async for batch in self.text_chunks.iter_keys(self.batch_size):
                chunk_count += len(batch)
            print(f"  Text chunks: {chunk_count:,}")

    def confirm_rebuild(self, targets: str) -> bool:
        print("\n" + "=" * 60)
        print(f"{BOLD_RED}⚠️  WARNING: {targets} will be DROPPED and rebuilt!{RESET}")
        print("=" * 60)
        print("\nAll affected records will be re-embedded, which may incur")
        print("significant embedding API cost and time on large datasets.")
        print("If interrupted, simply re-run this tool (sources are read-only).")
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
            text_chunks_kv=self.text_chunks,
            chunks_vdb=self.chunks_vdb,
            batch_size=self.batch_size,
            incompatible=self.incompatible_vdbs,
        )
        self.print_check_report(report)
        print(f"\n(check took {time.time() - start:.1f}s)")

    async def run_rebuild_entities_relations(self) -> List[Dict[str, Any]]:
        all_stats = []
        self.print_rebuild_section("entities")
        all_stats.append(
            await rebuild_entities_vdb(
                self.graph,
                self.entities_vdb,
                self.global_config,
                batch_size=self.batch_size,
                progress_callback=self.make_progress_printer("entities"),
            )
        )
        self.print_rebuild_section("relationships")
        all_stats.append(
            await rebuild_relationships_vdb(
                self.graph,
                self.relationships_vdb,
                self.global_config,
                batch_size=self.batch_size,
                progress_callback=self.make_progress_printer("relationships"),
            )
        )
        return all_stats

    async def run_rebuild_chunks(self) -> List[Dict[str, Any]]:
        self.print_rebuild_section("chunks")
        return [
            await rebuild_chunks_vdb(
                self.text_chunks,
                self.chunks_vdb,
                batch_size=self.batch_size,
                progress_callback=self.make_progress_printer("chunks"),
            )
        ]

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
