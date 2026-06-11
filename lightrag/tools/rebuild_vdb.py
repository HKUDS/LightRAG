#!/usr/bin/env python3
"""
Offline Vector Storage (VDB) Rebuild Tool for LightRAG

The knowledge graph and the text_chunks KV store are the authoritative data
sources in LightRAG. If a vector storage write fails at runtime (e.g. during
an entity merge, see issue #2917), graph and vector storage drift apart:
graph records lose their vector counterparts (and stale vector records may
remain). This tool restores consistency by dropping each vector storage and
rebuilding it from scratch from its authoritative source:

    entities_vdb       <- graph nodes
    relationships_vdb  <- graph edges
    chunks_vdb         <- text_chunks KV store

A read-only consistency check mode is also provided so users can decide
whether a (potentially expensive, full re-embedding) rebuild is needed.

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
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.constants import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
)
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

ProgressCallback = Callable[[int, int], None]


def _strip_agtype_quotes(value: Any) -> Any:
    """Strip surrounding double quotes from PostgreSQL/AGE agtype text casts.

    PGGraphStorage.get_all_edges() extracts entity ids via an
    ``agtype::text`` cast, which leaves string values wrapped in double
    quotes (e.g. ``'"Alice"'``). Other backends return plain strings.
    """
    if (
        isinstance(value, str)
        and len(value) >= 2
        and value[0] == '"'
        and value[-1] == '"'
    ):
        return value[1:-1]
    return value


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


async def _drop_vdb(vdb, label: str) -> None:
    drop_result = await vdb.drop()
    if not isinstance(drop_result, dict) or drop_result.get("status") != "success":
        raise RuntimeError(f"Failed to drop {label} vector storage: {drop_result}")
    logger.info(f"Dropped {label} vector storage")


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
    from lightrag.operate import _truncate_vdb_content

    nodes = await graph.get_all_nodes()
    stats = _new_stats("entities", len(nodes))

    payloads: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        entity_name = _strip_agtype_quotes(node.get("entity_id") or node.get("id"))
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
    from lightrag.operate import _truncate_vdb_content

    edges = await graph.get_all_edges()
    stats = _new_stats("relationships", len(edges))

    payloads: Dict[str, Dict[str, Any]] = {}
    for edge in edges:
        src = _strip_agtype_quotes(edge.get("source"))
        tgt = _strip_agtype_quotes(edge.get("target"))
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
) -> Dict[str, Any]:
    """Read-only diagnosis: find graph records with no vector counterpart.

    Only the graph -> VDB direction is covered; stale reverse orphans
    (records present in the VDB but absent from the graph) can only be
    eliminated by a full rebuild. Relations are probed with both candidate
    ids from make_relation_vdb_ids so legacy reverse-order ids are not
    misreported as missing.
    """
    report: Dict[str, Any] = {
        "graph_entities": 0,
        "graph_relations": 0,
        "missing_entities": 0,
        "missing_relations": 0,
        "missing_entity_names": [],
        "missing_relation_pairs": [],
        "skipped_nodes": 0,
        "skipped_edges": 0,
    }

    # Entities: one candidate id per graph node
    nodes = await graph.get_all_nodes()
    entity_items: List[tuple] = []
    seen_entity_ids: set = set()
    for node in nodes:
        entity_name = _strip_agtype_quotes(node.get("entity_id") or node.get("id"))
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
        src = _strip_agtype_quotes(edge.get("source"))
        tgt = _strip_agtype_quotes(edge.get("target"))
        if src is None or tgt is None or not str(src).strip() or not str(tgt).strip():
            report["skipped_edges"] += 1
            continue
        candidate_ids = make_relation_vdb_ids(str(src), str(tgt))
        if candidate_ids[0] in seen_relation_ids:
            continue
        seen_relation_ids.add(candidate_ids[0])
        relation_items.append((candidate_ids, f"{src} ~ {tgt}"))

    report["graph_relations"] = len(relation_items)
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
        report["missing_entities"] == 0 and report["missing_relations"] == 0
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
            print(
                "   Rebuild requires the api extra: " 'pip install "lightrag-hku[api]"'
            )
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
            # storages derive enable_vector from global_config["vector_storage"]
            # (ClientManager.get_config defaults enable_vector=True when it is
            # absent), so a legal mixed config like PGGraphStorage +
            # QdrantVectorDBStorage would otherwise make the tool wrongly try to
            # initialize pgvector. Keep all three names for parity.
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
            for storage in self.all_storages():
                await storage.initialize()
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
        return True

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
        print("\n" + "=" * 60)
        print("📊 Consistency Report (graph -> vector storage)")
        print("=" * 60)
        print(f"  Graph entities:    {report['graph_entities']:,}")
        print(f"  Graph relations:   {report['graph_relations']:,}")
        print(f"  Missing entities:  {report['missing_entities']:,}")
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
        confirm = input("\nProceed with the rebuild? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("\n✓ Rebuild cancelled")
            return False
        return True

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    async def run_check(self):
        print("\nRunning read-only consistency check...")
        start = time.time()
        report = await check_vdb_consistency(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            batch_size=self.batch_size,
        )
        self.print_check_report(report)
        print(f"\n(check took {time.time() - start:.1f}s)")

    async def run_rebuild_entities_relations(self) -> List[Dict[str, Any]]:
        all_stats = []
        all_stats.append(
            await rebuild_entities_vdb(
                self.graph,
                self.entities_vdb,
                self.global_config,
                batch_size=self.batch_size,
                progress_callback=self.make_progress_printer("entities"),
            )
        )
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
        return [
            await rebuild_chunks_vdb(
                self.text_chunks,
                self.chunks_vdb,
                batch_size=self.batch_size,
                progress_callback=self.make_progress_printer("chunks"),
            )
        ]

    def report_rebuild(self, all_stats: List[Dict[str, Any]]):
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

    async def run(self):
        try:
            # Initialize shared storage (REQUIRED for storage classes to work)
            from lightrag.kg.shared_storage import initialize_share_data

            initialize_share_data(workers=1)

            self.print_header()
            if not self.confirm_server_stopped():
                return

            if not await self.setup_storages():
                return

            while True:
                print("\n=== Rebuild Options ===")
                print("[1] Consistency check (read-only)")
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
                    return
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
                all_stats: List[Dict[str, Any]] = []
                if include_graph:
                    all_stats.extend(await self.run_rebuild_entities_relations())
                if include_chunks:
                    all_stats.extend(await self.run_rebuild_chunks())
                self.report_rebuild(all_stats)
                print(f"(rebuild took {time.time() - start:.1f}s)")

        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
        except Exception as e:
            print(f"\n✗ Rebuild tool failed: {e}")
            import traceback

            traceback.print_exc()
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


async def async_main():
    """Async main entry point"""
    tool = RebuildTool()
    await tool.run()


def main():
    """Synchronous entry point for CLI command"""
    # Load environment and configure logging only when run as a tool, never on import.
    load_dotenv(dotenv_path=".env", override=False)
    setup_logger("lightrag", level="INFO")
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
