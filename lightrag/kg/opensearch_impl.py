"""
OpenSearch Storage Implementation for LightRAG

This module provides OpenSearch-based storage backends for LightRAG,
including KV storage, document status storage, graph storage, and vector storage.

Requirements:
    - opensearch-py >= 3.0.0
    - OpenSearch 3.x or higher with k-NN plugin enabled
"""

import os
import re
import ssl as ssl_module
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Union, final
import numpy as np
import configparser

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..utils import logger, compute_mdhash_id, _cooperative_yield, merge_source_ids
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock, get_namespace_lock

import pipmaster as pm

if not pm.is_installed("opensearch-py"):
    pm.install("opensearch-py")

from opensearchpy import AsyncOpenSearch, helpers  # type: ignore
from opensearchpy.exceptions import (  # type: ignore
    OpenSearchException,
    NotFoundError,
    RequestError,
    ConflictError,
)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


def _get_opensearch_env(key, fallback):
    cfg_key = key.replace("OPENSEARCH_", "").lower()
    return os.environ.get(key, config.get("opensearch", cfg_key, fallback=fallback))


def _get_index_number_of_shards() -> int:
    return int(_get_opensearch_env("OPENSEARCH_NUMBER_OF_SHARDS", "1"))


def _get_index_number_of_replicas() -> int:
    return int(_get_opensearch_env("OPENSEARCH_NUMBER_OF_REPLICAS", "0"))


def _sanitize_index_name(name: str) -> str:
    """Sanitize a string to be a valid OpenSearch index name."""
    sanitized = re.sub(r"[^a-z0-9_-]", "_", name.lower())
    if sanitized and sanitized[0] in "-_+":
        sanitized = "x" + sanitized
    return sanitized


# HTTP statuses that indicate a transient failure where retrying makes sense:
# request timeout, rate limit, and the standard 5xx server-error range.
# A missing status (None) typically means a network or parse error before the
# server responded, which is also retriable.
_RETRYABLE_BULK_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# Cap the length of error summaries dumped to logs so a multi-MB mapping
# explanation can't flood the log file.
_BULK_ERROR_SUMMARY_MAX_LEN = 200


@dataclass(frozen=True)
class _FailedBulkOp:
    """Structured representation of a non-retryable per-action bulk failure."""

    op: str
    doc_id: str
    status: int | None
    error: str


@dataclass
class _PendingVectorDoc:
    """Buffered vector upsert waiting for embedding and/or bulk flush."""

    source: dict[str, Any]
    content: str
    vector: list[float] | None = None


def _summarize_bulk_error(error: Any) -> str:
    """Turn an opensearch-py per-action ``error`` payload into a short string.

    The field may be a string, dict (``{"type": ..., "reason": ...}``) or
    something else entirely. We prefer ``reason`` / ``type`` from dicts to
    keep the log readable.
    """
    if error is None:
        return ""
    if isinstance(error, str):
        summary = error
    elif isinstance(error, dict):
        reason = error.get("reason") or error.get("type")
        summary = reason if isinstance(reason, str) else repr(error)
    else:
        summary = repr(error)
    if len(summary) > _BULK_ERROR_SUMMARY_MAX_LEN:
        summary = summary[: _BULK_ERROR_SUMMARY_MAX_LEN - 3] + "..."
    return summary


def _extract_bulk_failed_ids(
    failed: list[Any] | None,
) -> tuple[set[str], list[_FailedBulkOp]]:
    """Split an opensearch-py bulk ``failed`` list into retryable / dead ops.

    ``async_bulk(raise_on_error=False)`` returns ``(success, failed)`` where
    ``failed`` is a list of per-action error dicts shaped like::

        {"index":  {"_id": "...", "status": 500, "error": {...}}}
        {"delete": {"_id": "...", "status": 404, ...}}
        {"create": {"_id": "...", "status": 409, ...}}

    Returns ``(retryable, non_retryable)``:
      * ``retryable``     — ``set[str]`` of ids that should be retried on
        the next flush (408 / 429 / 5xx, plus a missing status which
        usually means a network-level failure before the server responded).
      * ``non_retryable`` — ``list[_FailedBulkOp]`` of permanent failures
        (most 4xx, mapping errors, etc.) carrying op-name, id, status and
        a short ``error`` summary so callers can log meaningful context.
        ``404`` on a delete is treated as success-equivalent and dropped
        from both sets.

    Unrecognised or malformed entries are skipped so a stray dict shape
    never crashes the flush path.
    """
    retryable: set[str] = set()
    non_retryable: list[_FailedBulkOp] = []
    if not failed:
        return retryable, non_retryable
    for entry in failed:
        if not isinstance(entry, dict):
            continue
        for op_name, op_payload in entry.items():
            if not isinstance(op_payload, dict):
                continue
            doc_id = op_payload.get("_id")
            if not isinstance(doc_id, str):
                continue
            status = op_payload.get("status")
            # Deleting a missing doc is not a real failure -- the row is
            # already gone, so we don't carry it forward on every flush.
            if op_name == "delete" and status == 404:
                continue
            if status is None or status in _RETRYABLE_BULK_STATUSES:
                retryable.add(doc_id)
            else:
                non_retryable.append(
                    _FailedBulkOp(
                        op=op_name,
                        doc_id=doc_id,
                        status=status if isinstance(status, int) else None,
                        error=_summarize_bulk_error(op_payload.get("error")),
                    )
                )
    return retryable, non_retryable


# Flush-time bulk batching limits. opensearch-py's helpers.async_bulk already
# splits a request by payload-byte budget (primary) and record count
# (secondary) via _ActionChunker -- semantically identical to MongoDB's
# _chunk_by_budget. We only expose those two limiter dimensions as env vars and
# pass them through as `max_chunk_bytes` / `chunk_size`, mirroring the MONGO_*
# knobs (lightrag/kg/mongo_impl.py) so behaviour stays consistent across
# backends. Defaults are tuned for OpenSearch: 100 MiB sits at the typical
# `http.max_content_length` ceiling, while the record caps match Mongo's.
DEFAULT_OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES = 100 * 1024 * 1024  # 100 MiB
DEFAULT_OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH = 128
DEFAULT_OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH = 1000

# Sentinel "effectively unbounded" byte budget when payload splitting is
# disabled (env value <= 0). async_bulk needs a positive int here, so we use a
# large finite value in place of Mongo's float("inf").
_OPENSEARCH_UNBOUNDED_PAYLOAD_BYTES = 1 << 62


def _resolve_bulk_batch_limits() -> tuple[int, int, int]:
    """Resolve flush-time bulk batching limits from env, with module defaults.

    Shared by every OpenSearch write path so the byte/record caps that bound a
    single ``async_bulk`` request are consistent across all of them. A
    non-positive value disables that splitting dimension (see
    ``_run_chunked_async_bulk``). Returns
    ``(upsert_payload_bytes, upsert_records, delete_records)``.
    """
    upsert_payload_bytes = int(
        _get_opensearch_env(
            "OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES",
            str(DEFAULT_OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES),
        )
    )
    upsert_records = int(
        _get_opensearch_env(
            "OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH",
            str(DEFAULT_OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH),
        )
    )
    delete_records = int(
        _get_opensearch_env(
            "OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH",
            str(DEFAULT_OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH),
        )
    )
    if upsert_payload_bytes <= 0:
        logger.warning(
            f"OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES={upsert_payload_bytes} is non-positive, disable payload-size splitting"
        )
    if upsert_records <= 0:
        logger.warning(
            f"OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH={upsert_records} is non-positive, disable upsert record-count splitting"
        )
    if delete_records <= 0:
        logger.warning(
            f"OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH={delete_records} is non-positive, disable delete record-count splitting"
        )
    return upsert_payload_bytes, upsert_records, delete_records


async def _run_chunked_async_bulk(
    client: Any,
    actions: list[dict[str, Any]],
    *,
    max_payload_bytes: int,
    max_records_per_batch: int,
    log_prefix: str,
    what: str,
    raise_on_error: bool = False,
    **bulk_kwargs: Any,
) -> tuple[int, list[Any]]:
    """Run ``helpers.async_bulk`` with payload-size/record-count bounded chunks.

    A thin wrapper that mirrors ``mongo_impl._run_batched_bulk_write`` in shape,
    but delegates the actual splitting to opensearch-py's ``_ActionChunker``
    (byte budget primary, record count secondary, oversized single action
    emitted as its own chunk -- the same semantics as Mongo's
    ``_chunk_by_budget``). A non-positive limit disables that dimension. Extra
    keyword arguments (e.g. ``refresh``) are forwarded to ``async_bulk``.
    Returns ``async_bulk``'s ``(success, failed)`` tuple (``failed`` is empty
    when ``raise_on_error=True``).
    """
    if not actions:
        return 0, []
    chunk_size = max_records_per_batch if max_records_per_batch > 0 else len(actions)
    max_chunk_bytes = (
        max_payload_bytes
        if max_payload_bytes > 0
        else _OPENSEARCH_UNBOUNDED_PAYLOAD_BYTES
    )
    if len(actions) > chunk_size:
        # Log format aligned with mongo_impl's flush split log
        # (max_payload=/batch= field names, raw configured values). Unlike
        # Mongo we cannot report the final batch count up front: async_bulk's
        # _ActionChunker decides it at stream time by byte budget, so this
        # record-count condition only catches count-driven splits.
        logger.info(
            f"{log_prefix} {what} split for {len(actions)} records "
            f"(max_payload={max_payload_bytes} batch={max_records_per_batch})"
        )
    return await helpers.async_bulk(
        client,
        actions,
        chunk_size=chunk_size,
        max_chunk_bytes=max_chunk_bytes,
        raise_on_error=raise_on_error,
        **bulk_kwargs,
    )


# Index _meta flag marking that an edges index has been migrated to canonical
# (sorted-pair) document ids. Guards the one-time reindex in
# PGGraphStorage-style startup so it runs at most once per index.
_EDGE_ID_CANONICAL_META_FLAG = "edge_id_canonical_v1"

# Emit a migration progress line every this many scanned edges, so operators
# watching a large-index reindex see liveness and an X/total denominator.
_EDGE_MIGRATION_PROGRESS_INTERVAL = 50_000


def _canonical_edge_id(source_node_id: str, target_node_id: str) -> str:
    """Direction-independent edge document ``_id``.

    ``hash(sorted(src, tgt))`` collapses an edge and its reverse onto the same
    ``_id``, so concurrent ``(A,B)``/``(B,A)`` writes overwrite one document
    (last-write-wins) instead of racing into two separate docs. This makes
    ``upsert_edge`` idempotent by construction — no ``exists(reverse)``
    read-then-write and no lock needed. The canonical id is always one of the
    two directed ids ``hash("src-tgt")``/``hash("tgt-src")``, so the
    bidirectional ``mget`` in ``has_edge``/``get_edge`` keeps finding it.
    """
    lo, hi = sorted((source_node_id, target_node_id))
    return compute_mdhash_id(f"{lo}-{hi}", prefix="edge-")


def _edge_source_id_list(doc: dict[str, Any]) -> list[str]:
    """Return an edge doc's source ids, from the ``source_ids`` array or by
    splitting the ``GRAPH_FIELD_SEP``-joined ``source_id`` string."""
    sids = doc.get("source_ids")
    if not sids and doc.get("source_id"):
        sids = doc["source_id"].split(GRAPH_FIELD_SEP)
    return list(sids or [])


def _coerce_weight(weight: Any) -> float | None:
    """Coerce a (possibly string) edge weight to float, or None if non-numeric."""
    if weight is None:
        return None
    try:
        return float(weight)
    except (TypeError, ValueError):
        return None


def _merge_edge_payloads(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge edge-doc relation payloads when consolidating legacy duplicates.

    ``docs[0]`` is the survivor/base; the rest are duplicates folded into it.
    Mirrors ``mongo_impl``'s dedupe merge and ``operate.py``'s
    ``_merge_edges_then_upsert`` field semantics (minus LLM description
    summarisation): ``source_id``/``source_ids``/``file_path``/``description``
    union their ``GRAPH_FIELD_SEP`` components, ``keywords`` are comma-set-
    unioned, and ``weight`` is **summed** (duplicate docs carry separate
    accumulated weight). Returns only the merged fields (to be layered onto the
    surviving doc).

    Idempotent across fail-fast retries: the union fields union split components
    (re-merging an already-merged base is a no-op), and the weight sum counts the
    base's current weight once plus each duplicate only while its source_ids are
    not already folded into the base — so a retry (whose base already contains
    them) does not double-count. Legacy string weights are coerced; non-numeric
    values are skipped so a bad value cannot crash the migration.
    """
    source_ids: list[str] = []
    file_paths: list[str] = []
    descriptions: list[str] = []
    keywords: set[str] = set()
    for d in docs:
        source_ids = merge_source_ids(source_ids, _edge_source_id_list(d))
        fp = d.get("file_path")
        file_paths = merge_source_ids(
            file_paths, fp.split(GRAPH_FIELD_SEP) if fp else []
        )
        desc = d.get("description")
        descriptions = merge_source_ids(
            descriptions, desc.split(GRAPH_FIELD_SEP) if desc else []
        )
        kw = d.get("keywords")
        if kw:
            keywords.update(k.strip() for k in kw.split(",") if k.strip())

    # Idempotent summed weight: base (docs[0]) counts once; each duplicate adds
    # its weight only if its source_ids are not already folded into the base.
    base = docs[0] if docs else {}
    base_sids = set(_edge_source_id_list(base))
    weights: list[float] = []
    bw = _coerce_weight(base.get("weight"))
    if bw is not None:
        weights.append(bw)
    for d in docs[1:]:
        d_sids = set(_edge_source_id_list(d))
        if not d_sids or d_sids <= base_sids:
            continue  # no new trackable evidence -> don't (re-)add its weight
        dw = _coerce_weight(d.get("weight"))
        if dw is not None:
            weights.append(dw)

    merged: dict[str, Any] = {}
    if source_ids:
        merged["source_ids"] = source_ids
        merged["source_id"] = GRAPH_FIELD_SEP.join(source_ids)
    if file_paths:
        merged["file_path"] = GRAPH_FIELD_SEP.join(file_paths)
    if descriptions:
        merged["description"] = GRAPH_FIELD_SEP.join(descriptions)
    if keywords:
        merged["keywords"] = ",".join(sorted(keywords))
    if weights:
        merged["weight"] = sum(weights)
    return merged


# Detected at first connection; True when OpenSearch >= 3.3.0.
_shard_doc_supported: bool | None = None


def _pit_sort_with_field(field: str) -> list[dict]:
    """Return PIT sort clause with a unique field as primary sort.

    Used purely as a pagination tiebreaker — order is fixed to asc since the
    business sort (when present) is applied separately by the caller.

    >= 3.3.0: _shard_doc only (most efficient, already unique within PIT).
    < 3.3.0:  field + _doc (field is unique, _doc for efficiency).
    """
    if _shard_doc_supported:
        return [{"_shard_doc": "asc"}]
    return [{field: {"order": "asc"}}, {"_doc": "asc"}]


def _pit_sort_with_composite_key(*fields: str) -> list[dict]:
    """Return PIT sort clause with multiple fields forming a composite unique key.

    >= 3.3.0: _shard_doc (most efficient, ignores the fields).
    < 3.3.0:  field1 + field2 + ... + _doc (composite is unique, _doc for efficiency).
    """
    if _shard_doc_supported:
        return [{"_shard_doc": "asc"}]
    return [{f: {"order": "asc"}} for f in fields] + [{"_doc": "asc"}]


async def _detect_shard_doc_support(client: AsyncOpenSearch) -> bool:
    """Check if the cluster supports _shard_doc (OpenSearch >= 3.3.0)."""
    try:
        info = await client.info()
        version_str = info.get("version", {}).get("number", "0.0.0")
        # Strip pre-release suffixes (e.g. "3.3.0-SNAPSHOT" → "3", "3", "0")
        parts = [p.split("-")[0] for p in version_str.split(".")]
        major = int(parts[0]) if parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        supported = (major > 3) or (major == 3 and minor >= 3)
        logger.info(
            f"OpenSearch version {version_str}: "
            f"_shard_doc {'supported' if supported else 'not supported, using field+_doc fallback'}"
        )
        return supported
    except Exception as e:
        logger.warning(
            f"Failed to detect OpenSearch version, assuming _shard_doc not supported: {e}"
        )
        return False


class ClientManager:
    """Singleton manager for OpenSearch client connections."""

    _instances = {"client": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncOpenSearch:
        """Get or create a shared AsyncOpenSearch client with reference counting."""
        global _shard_doc_supported
        async with cls._lock:
            if cls._instances["client"] is None:
                hosts_str = _get_opensearch_env("OPENSEARCH_HOSTS", "localhost:9200")
                hosts = [h.strip() for h in hosts_str.split(",") if h.strip()]
                username = _get_opensearch_env("OPENSEARCH_USER", "admin")
                password = _get_opensearch_env("OPENSEARCH_PASSWORD", "admin")
                use_ssl = _get_opensearch_env("OPENSEARCH_USE_SSL", "true").lower() in (
                    "true",
                    "1",
                    "yes",
                )
                verify_certs = _get_opensearch_env(
                    "OPENSEARCH_VERIFY_CERTS", "false"
                ).lower() in ("true", "1", "yes")
                timeout = int(_get_opensearch_env("OPENSEARCH_TIMEOUT", "30"))
                max_retries = int(_get_opensearch_env("OPENSEARCH_MAX_RETRIES", "3"))

                ssl_context = None
                if use_ssl and not verify_certs:
                    ssl_context = ssl_module.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl_module.CERT_NONE

                client = AsyncOpenSearch(
                    hosts=hosts,
                    http_auth=(username, password) if username else None,
                    use_ssl=use_ssl,
                    verify_certs=verify_certs,
                    ssl_context=ssl_context,
                    ssl_show_warn=False,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_on_timeout=True,
                )
                cls._instances["client"] = client
                cls._instances["ref_count"] = 0
                _shard_doc_supported = await _detect_shard_doc_support(client)
                logger.info(f"OpenSearch client connected to {hosts}")

            cls._instances["ref_count"] += 1
            return cls._instances["client"]

    @classmethod
    async def release_client(cls, client: AsyncOpenSearch):
        """Release a client reference. Closes the connection when ref count reaches 0."""
        global _shard_doc_supported
        async with cls._lock:
            if client is not None and client is cls._instances["client"]:
                cls._instances["ref_count"] -= 1
                if cls._instances["ref_count"] <= 0:
                    try:
                        await cls._instances["client"].close()
                    except Exception:
                        pass
                    cls._instances["client"] = None
                    cls._instances["ref_count"] = 0
                    _shard_doc_supported = None
                    logger.info("OpenSearch client connection closed")


def _resolve_workspace(workspace: str, namespace: str):
    """Resolve effective workspace from env or parameter."""
    opensearch_workspace = os.environ.get("OPENSEARCH_WORKSPACE")
    if opensearch_workspace and opensearch_workspace.strip():
        effective = opensearch_workspace.strip()
        logger.info(
            f"Using OPENSEARCH_WORKSPACE: '{effective}' (overriding '{workspace}/{namespace}')"
        )
        return effective
    return workspace


def _build_index_name(workspace: str, namespace: str) -> tuple[str, str, str]:
    """Build index name and return (effective_workspace, final_namespace, index_name)."""
    effective = _resolve_workspace(workspace, namespace)
    if effective:
        final_ns = f"{effective}_{namespace}"
    else:
        final_ns = namespace
        effective = ""
    index_name = _sanitize_index_name(final_ns)
    return effective, final_ns, index_name


async def _mget_optional_doc(
    client: AsyncOpenSearch,
    index_name: str,
    doc_id: str,
    source_excludes: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fetch a single document via mget and return None when it is absent.

    ``source_excludes`` is forwarded to OpenSearch's ``_source_excludes`` so
    callers can ask the server to omit specific fields (e.g. ``["vector"]``)
    and save network bandwidth.
    """
    kwargs: dict[str, Any] = {"index": index_name, "body": {"ids": [doc_id]}}
    if source_excludes:
        kwargs["_source_excludes"] = source_excludes
    response = await client.mget(**kwargs)
    docs = response.get("docs", [])
    if not docs:
        return None
    doc = docs[0]
    if not doc.get("found"):
        return None
    return doc


def _is_missing_index_error(exc: Exception) -> bool:
    """Return True when an OpenSearch exception means the target index is missing."""
    return "index_not_found_exception" in str(exc)


async def _verify_mirrored_id_mapping(client: AsyncOpenSearch, index_name: str) -> None:
    """Fail-fast when an existing index lacks the __mirrored_id keyword mapping.

    Only enforced on OpenSearch < 3.3.0, where __mirrored_id serves as the
    cross-shard pagination tiebreaker. Indices created by older LightRAG
    releases will be missing this mapping; sorting by a missing field on a
    multi-shard index can drop or duplicate documents during PIT pagination.
    """
    if _shard_doc_supported:
        return
    try:
        mapping = await client.indices.get_mapping(index=index_name)
    except OpenSearchException:
        return
    props = mapping.get(index_name, {}).get("mappings", {}).get("properties", {})
    if "__mirrored_id" not in props:
        raise RuntimeError(
            f"Index '{index_name}' lacks the '__mirrored_id' keyword mapping "
            f"required for stable PIT pagination on OpenSearch < 3.3.0. "
            f"This index was likely created by an older LightRAG release. "
            f"Please reindex the data, or upgrade the cluster to OpenSearch >= 3.3.0."
        )


@final
@dataclass
class OpenSearchKVStorage(BaseKVStorage):
    """Key-Value storage using OpenSearch. Uses dynamic mapping to support varied schemas."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)
    _index_ready: bool = field(default=False, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )
        # Pending writes are flushed via _flush_pending_kv_ops() during
        # index_done_callback() / finalize(). Buffering many small upsert()
        # invocations into a single async_bulk roundtrip avoids the per-call
        # HTTP overhead profiled in issue #2785; the lock-everywhere model
        # mirrors what #3043 introduced for OpenSearchVectorDBStorage.
        self._pending_upserts: dict[str, dict[str, Any]] = {}
        self._pending_kv_deletes: set[str] = set()
        # Namespace-keyed lock (multi-process aware) is assigned in
        # initialize(). All buffer reads / writes and the flush itself
        # acquire this lock so an in-flight flush cannot interleave with
        # concurrent get_by_id / upsert / delete on the same workspace.
        self._flush_lock = None
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
            self._max_delete_records_per_batch,
        ) = _resolve_bulk_batch_limits()

    async def initialize(self):
        """Initialize client connection and create index if needed."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_index_if_not_exists()
            self._index_ready = True
            logger.debug(
                f"[{self.workspace}] OpenSearch KV storage initialized: {self._index_name}"
            )
        if self._flush_lock is None:
            self._flush_lock = get_namespace_lock(
                self.namespace, workspace=self.workspace
            )

    async def _ensure_index_ready(self):
        """Recreate the KV index after drop before the next write."""
        if self._index_ready:
            return
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            if not self._index_ready:
                await self._create_index_if_not_exists()
                self._index_ready = True

    def _mark_index_missing(self):
        """Mark the KV index as unavailable for subsequent read short-circuiting."""
        self._index_ready = False

    async def _create_index_if_not_exists(self):
        try:
            if not await self.client.indices.exists(index=self._index_name):
                # Use dynamic mapping so any namespace schema works
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "__mirrored_id": {"type": "keyword"},
                        },
                    },
                    "settings": {
                        "index": {
                            "number_of_shards": _get_index_number_of_shards(),
                            "number_of_replicas": _get_index_number_of_replicas(),
                        },
                    },
                }
                await self.client.indices.create(index=self._index_name, body=body)
                logger.info(f"[{self.workspace}] Created index: {self._index_name}")
            else:
                await _verify_mirrored_id_mapping(self.client, self._index_name)
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating index: {e}")
            raise

    async def finalize(self):
        """Flush pending writes and release the OpenSearch client connection.

        Regular flush failures (any ``Exception``) are captured so they
        can be re-surfaced as a ``RuntimeError`` that names the unflushed
        buffer counts -- otherwise ``LightRAG.finalize_storages()`` would
        log the storage as successfully finalized while writes silently
        failed to reach OpenSearch.

        ``BaseException`` subclasses other than ``Exception`` (notably
        ``asyncio.CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit``)
        are NOT caught: they propagate through the ``finally`` block so
        shutdown cancellation is honoured and not silently swallowed.
        The client is released in ``finally`` so it does not leak whether
        the flush succeeded, failed, or was cancelled.
        """
        flush_error: Exception | None = None
        try:
            try:
                await self._flush_pending_kv_ops()
            except Exception as e:
                # _flush_pending_kv_ops leaves the buffers intact on raise.
                flush_error = e
        finally:
            if self.client is not None:
                await ClientManager.release_client(self.client)
                self.client = None

        # Reached only when no BaseException propagated through the
        # finally above. Snapshot remaining buffer state to report
        # concrete counts.
        pending_upserts = len(self._pending_upserts)
        pending_deletes = len(self._pending_kv_deletes)

        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] OpenSearchKVStorage.finalize() flush "
                f"raised; {pending_upserts} pending upserts and "
                f"{pending_deletes} pending deletes were left buffered "
                f"(client released, data lost)"
            ) from flush_error
        if pending_upserts or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] OpenSearchKVStorage.finalize() left "
                f"{pending_upserts} pending upserts and {pending_deletes} "
                f"pending deletes buffered after final flush attempt "
                f"(transient bulk failure); these writes have been lost"
            )

    async def _iter_raw_docs(
        self, batch_size: int = 1000
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield raw OpenSearch hits using PIT + search_after pagination."""
        if not self._index_ready:
            return

        try:
            pit = await self.client.create_pit(
                index=self._index_name, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": {"match_all": {}},
                        "size": batch_size,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": _pit_sort_with_field("__mirrored_id"),
                    }
                    if search_after:
                        body["search_after"] = search_after

                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break

                    yield hits

                    search_after = hits[-1]["sort"]
                    if len(hits) < batch_size:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
            logger.error(f"[{self.workspace}] Error scanning documents: {e}")
            raise

    def _materialize_pending_kv_doc(
        self, doc_id: str, source: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a get_by_id-shaped view of a buffered upsert.

        Mirrors the post-processing applied to mget hits: drops the
        ``__mirrored_id`` PIT sort key, attaches the ``_id`` field and
        ensures ``create_time`` / ``update_time`` defaults are populated.
        The buffer entry itself is not mutated.
        """
        doc = {k: v for k, v in source.items() if k != "__mirrored_id"}
        doc["_id"] = doc_id
        doc.setdefault("create_time", 0)
        doc.setdefault("update_time", 0)
        return doc

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a document by its ID, with read-your-writes against the buffer.

        Priority: pending delete (tombstone) → pending upsert (buffered
        write) → OpenSearch via mget. The buffered path strips
        ``__mirrored_id`` so the returned dict has the same shape as the
        mget path.
        """
        async with self._flush_lock:
            if id in self._pending_kv_deletes:
                return None
            pending = self._pending_upserts.get(id)
            if pending is not None:
                return self._materialize_pending_kv_doc(id, pending)
            if not self._index_ready:
                return None
        try:
            response = await _mget_optional_doc(self.client, self._index_name, id)
            if response is None:
                return None
            doc = response["_source"]
            doc.pop("__mirrored_id", None)
            doc["_id"] = response["_id"]
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
            return doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting document {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple documents by IDs (read-your-writes), preserving order.

        Buffer is consulted under the lock with the same three-tier
        priority as ``get_by_id``; remaining ids fall through to mget
        outside the lock so the network call does not stall the flush.
        """
        if not ids:
            return []
        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_kv_deletes:
                    buffered[doc_id] = None
                    continue
                pending = self._pending_upserts.get(doc_id)
                if pending is not None:
                    buffered[doc_id] = self._materialize_pending_kv_doc(doc_id, pending)
                    continue
                remaining.append(doc_id)
            index_ready = self._index_ready

        doc_map: dict[str, dict[str, Any] | None] = {}
        if remaining and index_ready:
            try:
                response = await self.client.mget(
                    index=self._index_name, body={"ids": remaining}
                )
                for doc in response["docs"]:
                    if doc.get("found"):
                        data = doc["_source"]
                        data.pop("__mirrored_id", None)
                        data["_id"] = doc["_id"]
                        data.setdefault("create_time", 0)
                        data.setdefault("update_time", 0)
                        doc_map[doc["_id"]] = data
            except OpenSearchException as e:
                if _is_missing_index_error(e):
                    self._mark_index_missing()
                else:
                    logger.error(f"[{self.workspace}] Error getting documents: {e}")

        return [
            buffered[doc_id] if doc_id in buffered else doc_map.get(doc_id)
            for doc_id in ids
        ]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return the subset of keys that do not exist in storage.

        Buffer-aware: buffered upserts count as "exists" (and so are
        removed from the missing set), buffered deletes count as
        "missing" and are NOT queried via mget (a persisted-but-pending-
        delete row would otherwise be misclassified as existing).
        """
        async with self._flush_lock:
            pending_upserts = set(self._pending_upserts)
            pending_deletes = set(self._pending_kv_deletes)
            index_ready = self._index_ready

        # Buffered upserts shadow OpenSearch -- they will exist after flush.
        to_check = keys - pending_upserts - pending_deletes
        if not to_check:
            # All keys are accounted for by the buffer alone.
            return keys - pending_upserts
        if not index_ready:
            return keys - pending_upserts
        try:
            response = await self.client.mget(
                index=self._index_name,
                body={"ids": list(to_check)},
                _source=False,
            )
            existing_on_server = {
                doc["_id"] for doc in response["docs"] if doc.get("found")
            }
            return (keys - pending_upserts) - existing_on_server
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return keys - pending_upserts
            logger.error(f"[{self.workspace}] Error filtering keys: {e}")
            return keys - pending_upserts

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer documents for batched flush.

        Time-stamping and ``__mirrored_id`` injection happen eagerly so the
        persisted shape matches what reads expect; the actual ``async_bulk``
        call is deferred to ``_flush_pending_kv_ops()`` invoked from
        ``index_done_callback`` / ``finalize``.

        Multi-worker note: the buffer is process-local. Other workers will
        not see these writes until ``index_done_callback()`` flushes them.
        """
        if not data:
            return
        await self._ensure_index_ready()
        logger.debug(
            f"[{self.workspace}] Buffering {len(data)} documents for {self.namespace}"
        )
        current_time = int(time.time())

        # Construct sources outside the lock (no IO; just dict shuffling)
        # so we hold the lock only for the buffer-swap step.
        prepared: list[tuple[str, dict[str, Any]]] = []
        for i, (doc_id, doc_data) in enumerate(data.items(), start=1):
            doc_data["update_time"] = current_time
            doc_data.setdefault("create_time", current_time)
            source = {k: v for k, v in doc_data.items() if k != "_id"}
            source["__mirrored_id"] = doc_id
            prepared.append((doc_id, source))
            await _cooperative_yield(i)

        # Buffer: an upsert cancels any pending delete on the same id.
        async with self._flush_lock:
            for doc_id, source in prepared:
                self._pending_kv_deletes.discard(doc_id)
                self._pending_upserts[doc_id] = source

    async def delete(self, ids: list[str]) -> None:
        """Buffer document deletes for batched flush.

        A delete cancels any pending upsert on the same id; the actual
        bulk delete is performed by ``_flush_pending_kv_ops`` during the
        next ``index_done_callback`` / ``finalize`` call.

        ``_index_ready`` is intentionally NOT checked here: even if the
        index has been marked missing, the buffered upsert (if any) must
        still be invalidated, otherwise a subsequent flush would resurrect
        a logically-deleted key.
        """
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_upserts.pop(doc_id, None)
                self._pending_kv_deletes.add(doc_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for {len(ids)} documents in {self.namespace}"
        )

    async def _flush_pending_kv_ops(self) -> None:
        """Flush buffered upserts + deletes via a single async_bulk call.

        Concurrency contract: the entire flush runs under ``_flush_lock``;
        ``upsert`` / ``delete`` / reads / ``drop`` all acquire the same lock
        so an in-flight flush cannot interleave with concurrent buffer
        mutations.

        Failure handling mirrors the Vector-side helper:
          * If ``_ensure_index_ready`` raises, the buffers are left intact
            and the next flush retries.
          * If ``async_bulk`` raises, the buffers are left intact.
          * Per-doc retryable failures (408 / 429 / 5xx) stay in the buffer.
          * Per-doc non-retryable failures (most 4xx) are cleared and a
            sample is logged at WARNING with op / id / status / error.
        """
        async with self._flush_lock:
            if not self._pending_upserts and not self._pending_kv_deletes:
                return
            if self.client is None:
                return

            await self._ensure_index_ready()

            pending_upserts = self._pending_upserts
            pending_deletes = self._pending_kv_deletes

            # Deletes are flushed before upserts so a delete followed (in time)
            # by an upsert on the same id still ends as an index; the two
            # buffers are disjoint anyway (upsert/delete pop each other), so
            # running them as separate async_bulk requests is safe and lets the
            # delete record-count cap differ from the upsert cap (mirrors
            # mongo_impl's separate upsert/delete phases).
            delete_actions: list[dict[str, Any]] = [
                {
                    "_op_type": "delete",
                    "_index": self._index_name,
                    "_id": doc_id,
                }
                for doc_id in pending_deletes
            ]
            index_actions: list[dict[str, Any]] = [
                {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": doc_id,
                    "_source": source,
                }
                for doc_id, source in pending_upserts.items()
            ]

            try:
                log_prefix = f"[{self.workspace}] {self.namespace} flush:"
                del_success, del_failed = await _run_chunked_async_bulk(
                    self.client,
                    delete_actions,
                    max_payload_bytes=self._max_upsert_payload_bytes,
                    max_records_per_batch=self._max_delete_records_per_batch,
                    log_prefix=log_prefix,
                    what="delete",
                    raise_on_error=False,
                )
                idx_success, idx_failed = await _run_chunked_async_bulk(
                    self.client,
                    index_actions,
                    max_payload_bytes=self._max_upsert_payload_bytes,
                    max_records_per_batch=self._max_upsert_records_per_batch,
                    log_prefix=log_prefix,
                    what="upsert",
                    raise_on_error=False,
                )
                success = del_success + idx_success
                failed = list(del_failed) + list(idx_failed)
            except OpenSearchException as e:
                logger.error(
                    f"[{self.workspace}] Error flushing KV ops "
                    f"(upserts={len(pending_upserts)}, "
                    f"deletes={len(pending_deletes)}): {e}"
                )
                raise

            retryable_ids, non_retryable_ops = _extract_bulk_failed_ids(failed)
            non_retryable_ids = {op.doc_id for op in non_retryable_ops}

            # Clear successful + non-retryable entries; keep retryable ones.
            for doc_id in list(pending_upserts.keys()):
                if doc_id not in retryable_ids:
                    pending_upserts.pop(doc_id, None)
            new_deletes: set[str] = set()
            for doc_id in pending_deletes:
                if doc_id in retryable_ids:
                    new_deletes.add(doc_id)
            pending_deletes.clear()
            pending_deletes.update(new_deletes)

            if retryable_ids:
                logger.warning(
                    f"[{self.workspace}] {len(retryable_ids)} KV ops will "
                    f"retry on the next flush (transient failure)"
                )
            if non_retryable_ops:
                sample = non_retryable_ops[:5]
                sample_text = ", ".join(
                    f"{op.op}/{op.doc_id}/status={op.status}/{op.error}"
                    for op in sample
                )
                logger.warning(
                    f"[{self.workspace}] {len(non_retryable_ops)} KV ops "
                    f"failed permanently and were dropped (non-retryable status). "
                    f"Sample: {sample_text}"
                )
                if len(non_retryable_ops) > len(sample):
                    logger.debug(
                        f"[{self.workspace}] Remaining permanent failures: "
                        + ", ".join(
                            f"{op.op}/{op.doc_id}/status={op.status}/{op.error}"
                            for op in non_retryable_ops[len(sample) :]
                        )
                    )
            logger.debug(
                f"[{self.workspace}] Flushed KV ops: {success} ok, "
                f"retry={len(retryable_ids)}, dropped={len(non_retryable_ids)}"
            )

    async def index_done_callback(self) -> None:
        """Flush pending KV ops and refresh the index for search visibility.

        Flush runs first so a previously-missing index gets recreated by
        ``_flush_pending_kv_ops`` (via ``_ensure_index_ready``) before any
        buffered writes are abandoned. The refresh step is skipped only
        when the index is still not ready after the flush attempt.
        """
        await self._flush_pending_kv_ops()
        if not self._index_ready:
            return
        try:
            await self.client.indices.refresh(index=self._index_name)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
        except Exception:
            pass

    async def is_empty(self) -> bool:
        """Return True if the index (plus pending buffer) contains no docs.

        Buffer-aware: a pending upsert makes is_empty False immediately,
        avoiding the counterintuitive "I just upserted but is_empty
        returned True" case. Pending deletes alone are not enough to flip
        the answer because we cannot tell whether other persisted rows
        survive without flushing.
        """
        async with self._flush_lock:
            if self._pending_upserts:
                return False
            index_ready = self._index_ready
        if not index_ready:
            return True
        try:
            response = await self.client.count(index=self._index_name)
            return response["count"] == 0
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
            return True

    async def drop(self) -> dict[str, str]:
        """Delete the entire index, discarding pending buffers.

        Runs entirely under ``_flush_lock`` so a concurrent flush / upsert
        cannot land writes against an index that is being deleted.
        """
        async with self._flush_lock:
            # Pending writes are meaningless once the index is dropped.
            self._pending_upserts.clear()
            self._pending_kv_deletes.clear()
            try:
                try:
                    await self.client.indices.delete(index=self._index_name)
                    logger.info(f"[{self.workspace}] Dropped index: {self._index_name}")
                except NotFoundError:
                    logger.info(
                        f"[{self.workspace}] Index already missing during drop: {self._index_name}"
                    )
                self._mark_index_missing()
                return {
                    "status": "success",
                    "message": f"Index {self._index_name} dropped",
                }
            except OpenSearchException as e:
                self._mark_index_missing()
                logger.error(f"[{self.workspace}] Error dropping index: {e}")
                return {"status": "error", "message": str(e)}
            except Exception as e:
                self._mark_index_missing()
                logger.error(f"[{self.workspace}] Unexpected error dropping index: {e}")
                return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchDocStatusStorage(DocStatusStorage):
    """Document status storage using OpenSearch."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)
    _index_ready: bool = field(default=False, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
            self._max_delete_records_per_batch,
        ) = _resolve_bulk_batch_limits()

    def _prepare_doc_status_data(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw OpenSearch document to DocProcessingStatus-compatible dict."""
        data = doc.copy()
        data.pop("_id", None)
        data.pop("__mirrored_id", None)
        if "file_path" not in data:
            data["file_path"] = "no-file-path"
        data.setdefault("metadata", {})
        data.setdefault("error_msg", None)
        if "error" in data:
            if not data.get("error_msg"):
                data["error_msg"] = data.pop("error")
            else:
                data.pop("error", None)
        return data

    async def initialize(self):
        """Initialize client connection and create doc status index."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_index_if_not_exists()
            self._index_ready = True
            logger.debug(
                f"[{self.workspace}] OpenSearch DocStatus storage initialized: {self._index_name}"
            )

    async def _ensure_index_ready(self):
        """Recreate the doc status index after drop before the next write."""
        if self._index_ready:
            return
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            if not self._index_ready:
                await self._create_index_if_not_exists()
                self._index_ready = True

    def _mark_index_missing(self):
        """Mark the doc status index as unavailable for subsequent read short-circuiting."""
        self._index_ready = False

    async def _create_index_if_not_exists(self):
        try:
            if not await self.client.indices.exists(index=self._index_name):
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "__mirrored_id": {"type": "keyword"},
                            "status": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "track_id": {"type": "keyword"},
                            "content_hash": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                        },
                    },
                    "settings": {
                        "index": {
                            "number_of_shards": _get_index_number_of_shards(),
                            "number_of_replicas": _get_index_number_of_replicas(),
                        },
                    },
                }
                await self.client.indices.create(index=self._index_name, body=body)
                logger.info(
                    f"[{self.workspace}] Created doc status index: {self._index_name}"
                )
            else:
                await _verify_mirrored_id_mapping(self.client, self._index_name)
                await self._ensure_content_hash_mapping()
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating doc status index: {e}")
            raise

    async def _ensure_content_hash_mapping(self) -> None:
        """Add the content_hash keyword mapping to a pre-existing doc status index.

        Indices created by older LightRAG releases lack content_hash entirely.
        put_mapping is idempotent for new fields, so this is safe to call every
        startup; we only fail loudly when the cluster reports a mapping conflict
        (which would indicate dynamic mapping already coerced content_hash to a
        different type).
        """
        try:
            mapping = await self.client.indices.get_mapping(index=self._index_name)
        except OpenSearchException:
            return
        props = (
            mapping.get(self._index_name, {}).get("mappings", {}).get("properties", {})
        )
        if "content_hash" in props:
            return
        try:
            await self.client.indices.put_mapping(
                index=self._index_name,
                body={"properties": {"content_hash": {"type": "keyword"}}},
            )
            logger.info(
                f"[{self.workspace}] Added content_hash keyword mapping to {self._index_name}"
            )
        except OpenSearchException as e:
            logger.warning(
                f"[{self.workspace}] Failed to add content_hash mapping to "
                f"{self._index_name}: {e}"
            )

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        """Get a document status record by ID."""
        if not self._index_ready:
            return None
        try:
            response = await _mget_optional_doc(self.client, self._index_name, id)
            if response is None:
                return None
            doc = response["_source"]
            doc["_id"] = response["_id"]
            return doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting doc status {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple document status records by IDs."""
        if not self._index_ready:
            return [None] * len(ids)
        try:
            response = await self.client.mget(index=self._index_name, body={"ids": ids})
            doc_map = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["_id"] = doc["_id"]
                    doc_map[doc["_id"]] = data
            return [doc_map.get(id) for id in ids]
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return [None] * len(ids)
            logger.error(f"[{self.workspace}] Error getting doc statuses: {e}")
            return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return the subset of keys that do not exist in storage."""
        if not self._index_ready:
            return keys
        try:
            response = await self.client.mget(
                index=self._index_name, body={"ids": list(keys)}, _source=False
            )
            existing_ids = {doc["_id"] for doc in response["docs"] if doc.get("found")}
            return keys - existing_ids
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return keys
            logger.error(f"[{self.workspace}] Error filtering keys: {e}")
            return keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update document status records."""
        if not data:
            return
        await self._ensure_index_ready()
        logger.debug(f"[{self.workspace}] Upserting {len(data)} doc statuses")
        actions = []
        for i, (k, v) in enumerate(data.items(), start=1):
            v.setdefault("chunks_list", [])
            source = {fk: fv for fk, fv in v.items() if fk != "_id"}
            source["__mirrored_id"] = k
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": k,
                    "_source": source,
                }
            )
            await _cooperative_yield(i)
        try:
            # DocStatus needs refresh="wait_for" because get_docs_by_status
            # (search-based) is called immediately after enqueue upserts.
            await _run_chunked_async_bulk(
                self.client,
                actions,
                max_payload_bytes=self._max_upsert_payload_bytes,
                max_records_per_batch=self._max_upsert_records_per_batch,
                log_prefix=f"[{self.workspace}] {self.namespace} upsert:",
                what="doc-status upsert",
                raise_on_error=False,
                refresh="wait_for",
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting doc statuses: {e}")

    async def get_status_counts(self) -> dict[str, int]:
        """Get document counts grouped by status."""
        if not self._index_ready:
            return {}
        try:
            body = {
                "size": 0,
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }
            response = await self.client.search(index=self._index_name, body=body)
            return {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["status_counts"]["buckets"]
            }
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {}
            logger.error(f"[{self.workspace}] Error getting status counts: {e}")
            return {}

    async def _search_all_docs(self, query: dict) -> dict[str, DocProcessingStatus]:
        """Fetch all documents matching a query using PIT + search_after."""
        if not self._index_ready:
            return {}
        result = {}
        batch_size = 10000
        try:
            pit = await self.client.create_pit(
                index=self._index_name, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": query,
                        "size": batch_size,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": _pit_sort_with_field("__mirrored_id"),
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        try:
                            data = self._prepare_doc_status_data(hit["_source"])
                            result[hit["_id"]] = DocProcessingStatus(**data)
                        except (KeyError, TypeError) as e:
                            logger.error(
                                f"[{self.workspace}] Error parsing doc {hit['_id']}: {e}"
                            )
                    search_after = hits[-1]["sort"]
                    if len(hits) < batch_size:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {}
            logger.error(f"[{self.workspace}] Error fetching docs: {e}")
        return result

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching a specific processing status."""
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching any of the given statuses in a single query.

        Uses OpenSearch's terms query (multi-value equivalent of term) to fetch
        all matching statuses in one PIT + search_after pass instead of one
        full scan per status.
        """
        if not statuses:
            return {}
        status_values = [s.value for s in statuses]
        return await self._search_all_docs({"terms": {"status": status_values}})

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching a specific track ID."""
        return await self._search_all_docs({"term": {"track_id": track_id}})

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination using PIT + search_after."""
        if not self._index_ready:
            return [], 0
        status_filter_values = self.resolve_status_filter_values(
            status_filter=status_filter,
            status_filters=status_filters,
        )
        page = max(1, page)
        page_size = max(10, min(200, page_size))
        if sort_field == "id":
            sort_field = "_id"
        if sort_field not in ("created_at", "updated_at", "_id", "file_path"):
            sort_field = "updated_at"
        sort_order = "asc" if sort_direction.lower() == "asc" else "desc"

        query = {"match_all": {}}
        if status_filter_values is not None:
            if len(status_filter_values) == 1:
                query = {"term": {"status": next(iter(status_filter_values))}}
            else:
                query = {"terms": {"status": sorted(status_filter_values)}}

        skip_count = (page - 1) * page_size

        try:
            count_resp = await self.client.count(
                index=self._index_name, body={"query": query}
            )
            total_count = count_resp.get("count", 0)
            if total_count == 0 or skip_count >= total_count:
                return [], total_count

            sort_clause = [{sort_field: {"order": sort_order}}] + _pit_sort_with_field(
                "__mirrored_id"
            )

            pit = await self.client.create_pit(
                index=self._index_name, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                skipped = 0
                while skipped < skip_count:
                    batch = min(page_size, skip_count - skipped)
                    body = {
                        "query": query,
                        "sort": sort_clause,
                        "size": batch,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                    }
                    if search_after:
                        body["search_after"] = search_after
                    resp = await self.client.search(body=body)
                    hits = resp["hits"]["hits"]
                    if not hits:
                        return [], total_count
                    search_after = hits[-1]["sort"]
                    skipped += len(hits)

                body = {
                    "query": query,
                    "sort": sort_clause,
                    "size": page_size,
                    "pit": {"id": pit_id, "keep_alive": "1m"},
                }
                if search_after:
                    body["search_after"] = search_after
                response = await self.client.search(body=body)
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass

            documents = []
            for hit in response["hits"]["hits"]:
                try:
                    data = self._prepare_doc_status_data(hit["_source"])
                    documents.append((hit["_id"], DocProcessingStatus(**data)))
                except (KeyError, TypeError) as e:
                    logger.error(
                        f"[{self.workspace}] Error parsing doc {hit['_id']}: {e}"
                    )
            return documents, total_count
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return [], 0
            logger.error(f"[{self.workspace}] Error in paginated query: {e}")
            return [], 0

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get document counts for all statuses including an 'all' total."""
        if not self._index_ready:
            return {}
        try:
            body = {
                "size": 0,
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }
            response = await self.client.search(index=self._index_name, body=body)
            counts = {}
            total = 0
            for bucket in response["aggregations"]["status_counts"]["buckets"]:
                counts[bucket["key"]] = bucket["doc_count"]
                total += bucket["doc_count"]
            counts["all"] = total
            return counts
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {}
            logger.error(f"[{self.workspace}] Error getting all status counts: {e}")
            return {}

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Find a document status record by its file_path field."""
        if not self._index_ready:
            return None
        try:
            body = {"query": {"term": {"file_path": file_path}}, "size": 1}
            response = await self.client.search(index=self._index_name, body=body)
            hits = response["hits"]["hits"]
            if hits:
                doc = hits[0]["_source"]
                doc["_id"] = hits[0]["_id"]
                return doc
            return None
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting doc by file_path: {e}")
            return None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find an existing record whose canonical basename matches.

        The caller is responsible for passing an already-canonical basename;
        stored ``file_path`` values are canonicalized by the business layer, so
        this lookup performs an exact term query against the file_path keyword
        field.
        """
        if not basename:
            return None
        if basename == "unknown_source":
            return None
        if not self._index_ready:
            return None
        try:
            body = {"query": {"term": {"file_path": basename}}, "size": 1}
            response = await self.client.search(index=self._index_name, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                return None
            hit = hits[0]
            doc = hit["_source"]
            return hit["_id"], doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting doc by file_basename: {e}")
            return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find an existing record whose content_hash field matches.

        Uses the content_hash keyword mapping created by
        ``_create_index_if_not_exists`` / ``_ensure_content_hash_mapping``.
        Empty values short-circuit so legacy rows without the field cannot
        accidentally match via type coercion.
        """
        if not content_hash:
            return None
        if not self._index_ready:
            return None
        try:
            body = {"query": {"term": {"content_hash": content_hash}}, "size": 1}
            response = await self.client.search(index=self._index_name, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                return None
            hit = hits[0]
            doc = hit["_source"]
            return hit["_id"], doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting doc by content_hash: {e}")
            return None

    async def index_done_callback(self) -> None:
        """Refresh index to make recently indexed documents searchable."""
        if not self._index_ready:
            return
        try:
            await self.client.indices.refresh(index=self._index_name)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
        except Exception:
            pass

    async def is_empty(self) -> bool:
        """Return True if the index contains no documents."""
        if not self._index_ready:
            return True
        try:
            response = await self.client.count(index=self._index_name)
            return response["count"] == 0
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete document status records by IDs."""
        if not ids:
            return
        if not self._index_ready:
            return
        if isinstance(ids, set):
            ids = list(ids)
        try:
            # DocStatus needs refresh="wait_for" because downstream readers
            # (get_docs_by_status, get_docs_paginated, etc.) are search-based
            # and callers like _validate_and_fix_document_consistency() may
            # query immediately after deletion without index_done_callback().
            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]
            await _run_chunked_async_bulk(
                self.client,
                actions,
                max_payload_bytes=self._max_upsert_payload_bytes,
                max_records_per_batch=self._max_delete_records_per_batch,
                log_prefix=f"[{self.workspace}] {self.namespace} delete:",
                what="doc-status delete",
                raise_on_error=False,
                refresh="wait_for",
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
            logger.error(f"[{self.workspace}] Error deleting doc statuses: {e}")

    async def drop(self) -> dict[str, str]:
        """Delete the entire doc status index."""
        try:
            try:
                await self.client.indices.delete(index=self._index_name)
                logger.info(
                    f"[{self.workspace}] Dropped doc status index: {self._index_name}"
                )
            except NotFoundError:
                logger.info(
                    f"[{self.workspace}] Doc status index already missing during drop: {self._index_name}"
                )
            self._mark_index_missing()
            return {"status": "success", "message": f"Index {self._index_name} dropped"}
        except OpenSearchException as e:
            self._mark_index_missing()
            logger.error(f"[{self.workspace}] Error dropping doc status index: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            self._mark_index_missing()
            logger.error(
                f"[{self.workspace}] Unexpected error dropping doc status index: {e}"
            )
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchGraphStorage(BaseGraphStorage):
    """Graph storage using OpenSearch with separate nodes and edges indices.

    Supports two BFS traversal strategies:
    - PPL graphlookup (server-side BFS, requires OpenSearch SQL plugin with Calcite engine)
    - Application-level batched BFS (fallback, works on any OpenSearch 3.x+)

    The strategy is auto-detected during initialize() and can be overridden via
    the OPENSEARCH_USE_PPL_GRAPHLOOKUP environment variable (true/false).
    """

    client: AsyncOpenSearch = field(default=None)
    _nodes_index: str = field(default="", init=False)
    _edges_index: str = field(default="", init=False)
    _indices_ready: bool = field(default=False, init=False)
    _nodes_dirty: bool = field(default=False, init=False)
    _edges_dirty: bool = field(default=False, init=False)
    _ppl_graphlookup_available: bool = field(default=False, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, base_name = _build_index_name(
            self.workspace, self.namespace
        )
        self._nodes_index = f"{base_name}-nodes"
        self._edges_index = f"{base_name}-edges"
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
            self._max_delete_records_per_batch,
        ) = _resolve_bulk_batch_limits()

    async def initialize(self):
        """Initialize client, create indices, and detect PPL graphlookup support."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_indices_if_not_exist()
            await self._migrate_edges_to_canonical_id_if_needed()
            self._indices_ready = True
            self._nodes_dirty = False
            self._edges_dirty = False
            await self._detect_ppl_graphlookup()
            logger.debug(
                f"[{self.workspace}] OpenSearch Graph storage initialized: "
                f"{self._nodes_index}, {self._edges_index} "
                f"(PPL graphlookup: {self._ppl_graphlookup_available})"
            )

    async def _ensure_indices_ready(self):
        """Recreate graph indices after drop before the next write."""
        if self._indices_ready:
            return
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            if not self._indices_ready:
                await self._create_indices_if_not_exist()
                self._indices_ready = True

    def _mark_indices_missing(self):
        """Mark graph indices as unavailable for subsequent read short-circuiting."""
        self._indices_ready = False
        self._nodes_dirty = False
        self._edges_dirty = False

    async def _refresh_graph_indices_if_dirty(
        self, *, refresh_nodes: bool = False, refresh_edges: bool = False
    ) -> None:
        """Refresh graph indices only when prior writes made search views stale."""
        if not self._indices_ready:
            return
        if not (
            (refresh_nodes and self._nodes_dirty)
            or (refresh_edges and self._edges_dirty)
        ):
            return

        try:
            async with get_data_init_lock():
                if refresh_nodes and self._nodes_dirty:
                    await self.client.indices.refresh(index=self._nodes_index)
                    self._nodes_dirty = False
                if refresh_edges and self._edges_dirty:
                    await self.client.indices.refresh(index=self._edges_index)
                    self._edges_dirty = False
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
                return
            raise

    async def _detect_ppl_graphlookup(self):
        """Detect whether PPL graphlookup command is available on this cluster."""
        env_override = os.environ.get("OPENSEARCH_USE_PPL_GRAPHLOOKUP", "").lower()
        if env_override == "true":
            self._ppl_graphlookup_available = True
            return
        if env_override == "false":
            self._ppl_graphlookup_available = False
            return
        # Auto-detect by sending a minimal PPL query
        try:
            await self.client.transport.perform_request(
                "POST",
                "/_plugins/_ppl",
                body={"query": f"source = {self._edges_index} | head 0"},
            )
            # PPL endpoint works; now test graphlookup syntax with a no-op query
            await self.client.transport.perform_request(
                "POST",
                "/_plugins/_ppl",
                body={
                    "query": (
                        f"source = {self._edges_index} | head 1 "
                        f"| graphLookup {self._edges_index} "
                        f"start=source_node_id edge=target_node_id-->source_node_id "
                        f"maxDepth=0 as _gl_probe"
                    )
                },
            )
            self._ppl_graphlookup_available = True
            logger.info(
                f"[{self.workspace}] PPL graphlookup is available, using server-side BFS"
            )
        except Exception:
            self._ppl_graphlookup_available = False
            logger.info(
                f"[{self.workspace}] PPL graphlookup not available, using client-side BFS"
            )

    async def _create_indices_if_not_exist(self):
        try:
            if not await self.client.indices.exists(index=self._nodes_index):
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "entity_id": {"type": "keyword"},
                            "entity_type": {"type": "keyword"},
                            "description": {"type": "text"},
                            "source_id": {"type": "text"},
                            "source_ids": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "created_at": {"type": "long"},
                        },
                    },
                    "settings": {
                        "index": {
                            "number_of_shards": _get_index_number_of_shards(),
                            "number_of_replicas": _get_index_number_of_replicas(),
                        }
                    },
                }
                await self.client.indices.create(index=self._nodes_index, body=body)
                logger.info(
                    f"[{self.workspace}] Created nodes index: {self._nodes_index}"
                )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise

        try:
            if not await self.client.indices.exists(index=self._edges_index):
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "source_node_id": {"type": "keyword"},
                            "target_node_id": {"type": "keyword"},
                            "relationship": {"type": "keyword"},
                            "description": {"type": "text"},
                            "weight": {"type": "float"},
                            "keywords": {"type": "text"},
                            "source_id": {"type": "text"},
                            "source_ids": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "created_at": {"type": "long"},
                        },
                    },
                    "settings": {
                        "index": {
                            "number_of_shards": _get_index_number_of_shards(),
                            "number_of_replicas": _get_index_number_of_replicas(),
                        }
                    },
                }
                await self.client.indices.create(index=self._edges_index, body=body)
                logger.info(
                    f"[{self.workspace}] Created edges index: {self._edges_index}"
                )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise

    async def _migrate_edges_to_canonical_id_if_needed(self) -> None:
        """One-time reindex of edge docs onto canonical (sorted-pair) ``_id``s.

        Legacy edges were keyed by ``hash("src-tgt")`` in the *call* direction,
        so an edge could live under either orientation's id. After
        ``upsert_edge`` switched to a canonical sorted-pair id, a fresh write
        lands on a different ``_id`` than a legacy reverse-direction doc,
        leaving two documents for one edge (``node_degree``/``get_node_edges``
        double-count). This re-keys every non-canonical doc onto its canonical
        ``_id`` and deletes the stale id.

        **Fail-fast.** Runs in ``initialize`` inside ``get_data_init_lock``
        (which serialises one deployment's worker pool — only the first worker
        migrates, the rest skip via the ``_meta`` flag). On any non-benign
        per-item error (e.g. 429/503) it raises, so the service does not start
        until the index is fully canonical; the next startup rescans (the flag
        is only set on full success). Because the service is gated on a complete
        migration and every later write is canonical, there is no need for a
        per-write reverse-orientation cleanup.

        The canonical write uses ``op_type=create`` (insert-only): a legacy
        reciprocal duplicate (both directed docs present) collapses onto the
        existing forward/canonical doc (create 409, benign) and the reverse copy
        is deleted. A create that fails fast happens *before* any delete, so a
        source row is never dropped without its canonical counterpart existing.

        Assumes no concurrent *old-version* writer adds non-canonical docs after
        this completes (true for stop-the-world / single-deployment restarts). A
        true rolling deploy with two code versions writing the same index could
        leave a straggler reverse doc; the remedy is to clear the ``_meta`` flag
        and let the next startup re-migrate.
        """
        try:
            if not await self.client.indices.exists(index=self._edges_index):
                logger.debug(
                    f"[{self.workspace}] Edge index {self._edges_index} does not "
                    f"exist yet; skipping canonical edge-id migration"
                )
                return
            mapping = await self.client.indices.get_mapping(index=self._edges_index)
            meta = (
                mapping.get(self._edges_index, {}).get("mappings", {}).get("_meta", {})
            )
            if meta.get(_EDGE_ID_CANONICAL_META_FLAG):
                logger.info(
                    f"[{self.workspace}] Edge index {self._edges_index} already on "
                    f"canonical ids; skipping migration"
                )
                return

            # Count upfront so operators get an X/total denominator; best-effort
            # (migration still works if count is unavailable).
            try:
                total = (await self.client.count(index=self._edges_index)).get("count")
            except OpenSearchException:
                total = None
            logger.info(
                f"[{self.workspace}] Starting canonical edge-id migration for "
                f"{self._edges_index}"
                + (f" (~{total} edges to scan)" if total is not None else "")
            )

            scanned = 0
            migrated = 0
            # Each entry is (canonical_id, old_id, source) for one non-canonical
            # doc to be re-keyed. Flush roughly one bulk chunk at a time so a huge
            # index does not buffer every action in memory before writing.
            pending: list[tuple[str, str, dict[str, Any]]] = []
            flush_at = max(self._max_upsert_records_per_batch, 1)
            next_progress = _EDGE_MIGRATION_PROGRESS_INTERVAL

            async def _flush_pending() -> None:
                nonlocal pending
                if not pending:
                    return
                batch, pending = pending, []

                # Phase 1 — create the canonical docs. op_type=create
                # (insert-only): a create 409 means the canonical doc already
                # exists (forward legacy doc), which is benign — the reverse
                # source row is then safe to drop. raise_on_error=False so a 409
                # does not abort.
                create_actions = [
                    {
                        "_op_type": "create",
                        "_index": self._edges_index,
                        "_id": canonical,
                        "_source": source,
                    }
                    for canonical, _old_id, source in batch
                ]
                _success, errors = await _run_chunked_async_bulk(
                    self.client,
                    create_actions,
                    max_payload_bytes=self._max_upsert_payload_bytes,
                    max_records_per_batch=self._max_upsert_records_per_batch,
                    log_prefix=f"[{self.workspace}] {self.namespace} edges:",
                    what="canonical edge-id migration (create)",
                    raise_on_error=False,
                )
                # A create 409 means the canonical doc already exists (a forward
                # legacy doc, i.e. a reciprocal duplicate): merge this reverse
                # doc's relation payload into it (below) so deleting the reverse
                # loses no evidence. Any other create error (e.g. 429/503) fails
                # fast BEFORE any delete, so no edge is dropped without its
                # canonical counterpart in place; the flag stays unset and the
                # next startup rescans.
                conflicted_canonicals: list[str] = []
                real_create_errors = []
                for e in errors:
                    info = e.get("create") if isinstance(e, dict) else None
                    if info is not None and info.get("status") == 409:
                        if info.get("_id"):
                            conflicted_canonicals.append(info["_id"])
                        continue
                    real_create_errors.append(e)
                if real_create_errors:
                    raise RuntimeError(
                        f"Canonical edge-id migration: {len(real_create_errors)} "
                        f"create error(s) in {self._edges_index}; aborting startup "
                        f"(no source rows deleted)"
                    )

                if conflicted_canonicals:
                    source_by_canonical = {
                        canonical: source for canonical, _old_id, source in batch
                    }
                    for canonical in conflicted_canonicals:
                        reverse_source = source_by_canonical.get(canonical)
                        if reverse_source is not None:
                            await self._merge_into_canonical_edge(
                                canonical, reverse_source
                            )

                # Phase 2 — every create succeeded, 409'd-then-merged, so the
                # canonical now exists for all; delete the old ids. delete 404 is
                # benign (another run already removed it); any other delete error
                # fails fast.
                delete_actions = [
                    {"_op_type": "delete", "_index": self._edges_index, "_id": old_id}
                    for _canonical, old_id, _source in batch
                ]
                _ds, derrors = await _run_chunked_async_bulk(
                    self.client,
                    delete_actions,
                    max_payload_bytes=self._max_upsert_payload_bytes,
                    max_records_per_batch=self._max_delete_records_per_batch,
                    log_prefix=f"[{self.workspace}] {self.namespace} edges:",
                    what="canonical edge-id migration (delete)",
                    raise_on_error=False,
                )
                real_delete_errors = [
                    e
                    for e in derrors
                    if not (
                        isinstance(e, dict) and e.get("delete", {}).get("status") == 404
                    )
                ]
                if real_delete_errors:
                    raise RuntimeError(
                        f"Canonical edge-id migration: {len(real_delete_errors)} "
                        f"delete error(s) in {self._edges_index}; aborting startup"
                    )

            scroll_id = None
            try:
                response = await self.client.search(
                    index=self._edges_index,
                    body={"query": {"match_all": {}}, "sort": ["_doc"]},
                    scroll="5m",
                    size=1000,
                )
                while True:
                    scroll_id = response.get("_scroll_id")
                    hits = response.get("hits", {}).get("hits", [])
                    if not hits:
                        break
                    for hit in hits:
                        scanned += 1
                        source = hit.get("_source", {})
                        src = source.get("source_node_id")
                        tgt = source.get("target_node_id")
                        if not src or not tgt:
                            continue
                        canonical = _canonical_edge_id(src, tgt)
                        if hit["_id"] == canonical:
                            continue
                        # Queue (canonical, old_id, source); the create/delete
                        # split happens in _flush_pending so a failed create never
                        # takes its source row with it.
                        pending.append((canonical, hit["_id"], source))
                        migrated += 1
                    if len(pending) >= flush_at:
                        await _flush_pending()
                    if scanned >= next_progress:
                        logger.info(
                            f"[{self.workspace}] Canonical edge-id migration "
                            f"progress: scanned {scanned}"
                            + (f"/{total}" if total is not None else "")
                            + f", migrated {migrated} so far"
                        )
                        next_progress += _EDGE_MIGRATION_PROGRESS_INTERVAL
                    response = await self.client.scroll(
                        scroll_id=scroll_id, scroll="5m"
                    )
                await _flush_pending()
            finally:
                if scroll_id is not None:
                    try:
                        await self.client.clear_scroll(scroll_id=scroll_id)
                    except OpenSearchException:
                        pass

            if migrated:
                # Make migrated docs visible to subsequent searches in one go.
                try:
                    await self.client.indices.refresh(index=self._edges_index)
                except OpenSearchException:
                    pass

            logger.info(
                f"[{self.workspace}] Canonical edge-id migration complete for "
                f"{self._edges_index}: scanned {scanned}, migrated {migrated}"
            )
            # Mark complete (only reached on full success) so subsequent startups
            # skip the full scan. Legacy reciprocal duplicates collapsed onto one
            # canonical doc: the reverse doc's relation payload was merged into
            # the existing canonical (see _merge_into_canonical_edge) and the
            # reverse orientation deleted — no relation evidence lost.
            await self.client.indices.put_mapping(
                index=self._edges_index,
                body={"_meta": {**meta, _EDGE_ID_CANONICAL_META_FLAG: True}},
            )
        except OpenSearchException as e:
            # Fail fast: a transport/cluster error during migration must abort
            # startup (flag stays unset) rather than serve a half-migrated index.
            logger.error(
                f"[{self.workspace}] Canonical edge-id migration failed for "
                f"{self._edges_index}: {e}; aborting startup"
            )
            raise

    async def _merge_into_canonical_edge(
        self, canonical_id: str, reverse_source: dict[str, Any]
    ) -> None:
        """Merge a legacy reverse-orientation doc's payload into an existing
        canonical doc (the create-409 reciprocal-duplicate case) so deleting the
        reverse loses no relation evidence (mirrors mongo_impl's dedupe merge).

        Uses optimistic concurrency (``if_seq_no``/``if_primary_term``) so a
        concurrent live write during a rolling deploy is never clobbered: on a
        version conflict we re-read — now including that write — and re-merge.
        The merge is idempotent (see ``_merge_edge_payloads``), so a fail-fast
        retry over an already-merged canonical is a no-op.
        """
        for _attempt in range(3):
            try:
                current = await self.client.get(
                    index=self._edges_index, id=canonical_id
                )
            except NotFoundError:
                # Canonical vanished between the create-409 and now; recreate it
                # from the reverse source (nothing to merge against).
                await self.client.index(
                    index=self._edges_index, id=canonical_id, body=reverse_source
                )
                return
            base = current.get("_source", {})
            merged = {**base, **_merge_edge_payloads([base, reverse_source])}
            try:
                await self.client.index(
                    index=self._edges_index,
                    id=canonical_id,
                    body=merged,
                    if_seq_no=current["_seq_no"],
                    if_primary_term=current["_primary_term"],
                )
                return
            except ConflictError:
                # A concurrent write changed the canonical doc; re-read and
                # re-merge so we never overwrite that write with stale data.
                continue
        raise RuntimeError(
            f"Canonical edge-id migration: could not merge into {canonical_id} "
            f"after retries in {self._edges_index}; aborting startup"
        )

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    # --- Basic queries ---

    async def has_node(self, node_id: str) -> bool:
        """Check whether a node exists in the graph."""
        if not self._indices_ready:
            return False
        try:
            return await self.client.exists(index=self._nodes_index, id=node_id)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check whether an edge exists between two nodes (bidirectional).

        Uses mget with the two candidate edge IDs so the check is real-time
        (translog-backed), consistent with has_node() and independent of the
        index refresh cycle.
        """
        if not self._indices_ready:
            return False
        try:
            forward_id = compute_mdhash_id(
                f"{source_node_id}-{target_node_id}", prefix="edge-"
            )
            reverse_id = compute_mdhash_id(
                f"{target_node_id}-{source_node_id}", prefix="edge-"
            )
            response = await self.client.mget(
                index=self._edges_index, body={"ids": [forward_id, reverse_id]}
            )
            return any(doc.get("found") for doc in response.get("docs", []))
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return False

    async def node_degree(self, node_id: str) -> int:
        """Count the number of edges connected to a node."""
        if not self._indices_ready:
            return 0
        try:
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            response = await self.client.count(
                index=self._edges_index,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"source_node_id": node_id}},
                                {"term": {"target_node_id": node_id}},
                            ]
                        }
                    }
                },
            )
            return response.get("count", 0)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Sum of degrees of both endpoint nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get a node document by ID, or None if not found."""
        if not self._indices_ready:
            return None
        try:
            response = await _mget_optional_doc(self.client, self._nodes_index, node_id)
            if response is None:
                return None
            doc = response["_source"]
            doc["_id"] = response["_id"]
            return doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get an edge between two nodes (bidirectional), or None.

        Uses mget with the two candidate edge IDs so the read is real-time
        (translog-backed), consistent with get_node() and independent of the
        index refresh cycle.
        """
        if not self._indices_ready:
            return None
        try:
            forward_id = compute_mdhash_id(
                f"{source_node_id}-{target_node_id}", prefix="edge-"
            )
            reverse_id = compute_mdhash_id(
                f"{target_node_id}-{source_node_id}", prefix="edge-"
            )
            response = await self.client.mget(
                index=self._edges_index, body={"ids": [forward_id, reverse_id]}
            )
            for doc in response.get("docs", []):
                if doc.get("found"):
                    result = doc["_source"]
                    result["_id"] = doc["_id"]
                    return result
            return None
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all (source, target) edge tuples connected to a node."""
        if not self._indices_ready:
            return None
        try:
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            query = {
                "bool": {
                    "should": [
                        {"term": {"source_node_id": source_node_id}},
                        {"term": {"target_node_id": source_node_id}},
                    ]
                }
            }
            edges = []
            pit = await self.client.create_pit(
                index=self._edges_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": query,
                        "_source": ["source_node_id", "target_node_id"],
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": _pit_sort_with_composite_key(
                            "source_node_id", "target_node_id"
                        ),
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        edges.append(
                            (
                                hit["_source"]["source_node_id"],
                                hit["_source"]["target_node_id"],
                            )
                        )
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            return edges
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return None

    # --- Batch operations ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch multiple nodes by ID."""
        if not self._indices_ready:
            return {}
        try:
            response = await self.client.mget(
                index=self._nodes_index, body={"ids": node_ids}
            )
            result = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["_id"] = doc["_id"]
                    result[doc["_id"]] = data
            return result
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return {}

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Batch-fetch edge counts for multiple nodes using aggregations."""
        if not node_ids:
            return {}
        if not self._indices_ready:
            return {}
        try:
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            # Use a single query with aggregations for both source and target
            body = {
                "size": 0,
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"source_node_id": node_ids}},
                            {"terms": {"target_node_id": node_ids}},
                        ]
                    }
                },
                "aggs": {
                    "source_degrees": {
                        "terms": {
                            "field": "source_node_id",
                            "size": len(node_ids) * 2,
                        }
                    },
                    "target_degrees": {
                        "terms": {
                            "field": "target_node_id",
                            "size": len(node_ids) * 2,
                        }
                    },
                },
            }
            response = await self.client.search(index=self._edges_index, body=body)
            result = {}
            for bucket in response["aggregations"]["source_degrees"]["buckets"]:
                if bucket["key"] in node_ids:
                    result[bucket["key"]] = (
                        result.get(bucket["key"], 0) + bucket["doc_count"]
                    )
            for bucket in response["aggregations"]["target_degrees"]["buckets"]:
                if bucket["key"] in node_ids:
                    result[bucket["key"]] = (
                        result.get(bucket["key"], 0) + bucket["doc_count"]
                    )
            return result
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return {}

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Batch-fetch edge tuples for multiple nodes."""
        result = {nid: [] for nid in node_ids}
        if not self._indices_ready:
            return result
        try:
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            query = {
                "bool": {
                    "should": [
                        {"terms": {"source_node_id": node_ids}},
                        {"terms": {"target_node_id": node_ids}},
                    ]
                }
            }
            pit = await self.client.create_pit(
                index=self._edges_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": query,
                        "_source": ["source_node_id", "target_node_id"],
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": _pit_sort_with_composite_key(
                            "source_node_id", "target_node_id"
                        ),
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        src = hit["_source"]["source_node_id"]
                        tgt = hit["_source"]["target_node_id"]
                        if src in result:
                            result[src].append((src, tgt))
                        if tgt in result:
                            result[tgt].append((src, tgt))
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            pass
        return result

    # --- Upsert operations ---

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert or update a node. Adds entity_id for PPL compatibility."""
        try:
            await self._ensure_indices_ready()
            doc = {k: v for k, v in node_data.items() if k != "_id"}
            doc["entity_id"] = node_id
            if node_data.get("source_id", ""):
                doc["source_ids"] = node_data["source_id"].split(GRAPH_FIELD_SEP)
            # No per-operation refresh: node reads use ID-based mget/exists
            # (translog, real-time). Search visibility after index_done_callback().
            await self.client.index(index=self._nodes_index, id=node_id, body=doc)
            self._nodes_dirty = True
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting node {node_id}: {e}")

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert or update an edge keyed by a canonical (sorted-pair) ``_id``.

        The canonical id collapses ``(src, tgt)`` and ``(tgt, src)`` onto one
        document, so this is idempotent by construction: concurrent
        reciprocal writers overwrite the same ``_id`` (last-write-wins) instead
        of racing into two docs. No ``exists(reverse)`` read-then-write needed.

        New writes are always canonical, and the startup migration is fail-fast
        (the service does not start until every legacy doc is on its canonical
        id), so there is no need to delete a reverse-orientation doc on each
        write — the index is canonical before any write happens.
        """
        try:
            await self._ensure_indices_ready()
            # Ensure source node exists (don't overwrite if it already has data)
            if not await self.has_node(source_node_id):
                await self.upsert_node(source_node_id, {})

            doc = {k: v for k, v in edge_data.items() if k != "_id"}
            doc["source_node_id"] = source_node_id
            doc["target_node_id"] = target_node_id
            if edge_data.get("source_id", ""):
                doc["source_ids"] = edge_data["source_id"].split(GRAPH_FIELD_SEP)

            edge_id = _canonical_edge_id(source_node_id, target_node_id)
            await self.client.index(index=self._edges_index, id=edge_id, body=doc)
            self._edges_dirty = True
        except OpenSearchException as e:
            logger.error(
                f"[{self.workspace}] Error upserting edge {source_node_id}->{target_node_id}: {e}"
            )

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Batch insert/update multiple nodes using the OpenSearch bulk API.

        Args:
            nodes: List of (node_id, node_data) tuples.
        """
        if not nodes:
            return
        try:
            await self._ensure_indices_ready()
            actions = []
            for node_id, node_data in nodes:
                doc = {k: v for k, v in node_data.items() if k != "_id"}
                doc["entity_id"] = node_id
                if node_data.get("source_id", ""):
                    doc["source_ids"] = node_data["source_id"].split(GRAPH_FIELD_SEP)
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": self._nodes_index,
                        "_id": node_id,
                        "_source": doc,
                    }
                )
            await _run_chunked_async_bulk(
                self.client,
                actions,
                max_payload_bytes=self._max_upsert_payload_bytes,
                max_records_per_batch=self._max_upsert_records_per_batch,
                log_prefix=f"[{self.workspace}] {self.namespace} nodes:",
                what="node upsert",
                raise_on_error=True,
            )
            self._nodes_dirty = True
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error during batch node upsert: {e}")

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        """Check existence of multiple nodes using a single mget request.

        Args:
            node_ids: List of node IDs to check.

        Returns:
            Set of node_ids that exist in the graph.
        """
        if not node_ids:
            return set()
        if not self._indices_ready:
            return set()
        try:
            response = await self.client.mget(
                index=self._nodes_index, body={"ids": node_ids}
            )
            return {doc["_id"] for doc in response.get("docs", []) if doc.get("found")}
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return set()

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Batch insert/update multiple edges using the OpenSearch bulk API.

        Each edge is keyed by its canonical (sorted-pair) ``_id`` (see
        ``_canonical_edge_id``), so reciprocal directions collapse onto one
        document with no reverse-direction look-up. Edges that map to the same
        canonical id within this batch are deduplicated last-write-wins.

        Args:
            edges: List of (source_node_id, target_node_id, edge_data) tuples.
        """
        if not edges:
            return
        try:
            await self._ensure_indices_ready()

            # Ensure all source nodes exist (mirrors upsert_edge behaviour)
            source_ids = list({src for src, _tgt, _data in edges})
            existing_sources = await self.has_nodes_batch(source_ids)
            missing_sources = [
                (nid, {}) for nid in source_ids if nid not in existing_sources
            ]
            if missing_sources:
                await self.upsert_nodes_batch(missing_sources)

            # Key every edge by its canonical id and dedupe within the batch
            # (last-write-wins) so a single bulk request carries one action per
            # logical edge regardless of direction.
            actions_by_id: dict[str, dict[str, Any]] = {}
            for src, tgt, edge_data in edges:
                doc = {k: v for k, v in edge_data.items() if k != "_id"}
                doc["source_node_id"] = src
                doc["target_node_id"] = tgt
                if edge_data.get("source_id", ""):
                    doc["source_ids"] = edge_data["source_id"].split(GRAPH_FIELD_SEP)
                edge_id = _canonical_edge_id(src, tgt)
                actions_by_id[edge_id] = {
                    "_op_type": "index",
                    "_index": self._edges_index,
                    "_id": edge_id,
                    "_source": doc,
                }
            actions = list(actions_by_id.values())
            await _run_chunked_async_bulk(
                self.client,
                actions,
                max_payload_bytes=self._max_upsert_payload_bytes,
                max_records_per_batch=self._max_upsert_records_per_batch,
                log_prefix=f"[{self.workspace}] {self.namespace} edges:",
                what="edge upsert",
                raise_on_error=True,
            )
            self._edges_dirty = True
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error during batch edge upsert: {e}")

    # --- Delete operations ---

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its connected edges.

        Marks node and edge search views dirty so refresh happens lazily on the
        next search/count-based graph read. Uses conflicts="proceed" to
        tolerate already-deleted matches.
        """
        try:
            # Refresh edge search view so delete_by_query sees all un-flushed writes.
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            # Delete all edges referencing this node
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"source_node_id": node_id}},
                            {"term": {"target_node_id": node_id}},
                        ]
                    }
                }
            }
            await self.client.delete_by_query(
                index=self._edges_index,
                body=body,
                params={"conflicts": "proceed"},
            )
            # Delete the node
            try:
                await self.client.delete(index=self._nodes_index, id=node_id)
            except NotFoundError:
                pass
            self._nodes_dirty = True
            self._edges_dirty = True
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error deleting node {node_id}: {e}")

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Batch-delete multiple nodes and their connected edges.

        Marks node and edge search views dirty so refresh happens lazily on the
        next search/count-based graph read. Uses conflicts="proceed" to
        tolerate already-deleted matches.
        """
        if not nodes:
            return
        logger.info(f"[{self.workspace}] Deleting {len(nodes)} nodes")
        try:
            # Refresh edge search view so delete_by_query sees all un-flushed writes.
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            # Delete edges
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"source_node_id": nodes}},
                            {"terms": {"target_node_id": nodes}},
                        ]
                    }
                }
            }
            await self.client.delete_by_query(
                index=self._edges_index,
                body=body,
                params={"conflicts": "proceed"},
            )
            # Delete nodes
            actions = [
                {"_op_type": "delete", "_index": self._nodes_index, "_id": nid}
                for nid in nodes
            ]
            await _run_chunked_async_bulk(
                self.client,
                actions,
                max_payload_bytes=self._max_upsert_payload_bytes,
                max_records_per_batch=self._max_delete_records_per_batch,
                log_prefix=f"[{self.workspace}] {self.namespace} nodes:",
                what="node delete",
                raise_on_error=False,
            )
            self._nodes_dirty = True
            self._edges_dirty = True
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error removing nodes: {e}")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Batch-delete multiple edges by deterministic ID (real-time).

        New writes key edges by their canonical (sorted-pair) id, but we still
        delete *both* directed candidates per edge:
          forward  = compute_mdhash_id("src-tgt", prefix="edge-")
          reverse  = compute_mdhash_id("tgt-src", prefix="edge-")
        The canonical id is always one of these two, and deleting the other is a
        harmless 404 — this keeps deletes effective for any legacy doc not yet
        collapsed by the canonical-id migration. The raw bulk API does not raise
        on a 404 delete.

        Marks edge search views dirty so refresh happens lazily on the next
        search/count-based graph read.
        """
        if not edges:
            return
        logger.info(f"[{self.workspace}] Deleting {len(edges)} edges")
        try:
            operations = []
            for src, tgt in edges:
                for edge_id in (
                    compute_mdhash_id(f"{src}-{tgt}", prefix="edge-"),
                    compute_mdhash_id(f"{tgt}-{src}", prefix="edge-"),
                ):
                    operations.append(
                        {
                            "delete": {
                                "_index": self._edges_index,
                                "_id": edge_id,
                            }
                        }
                    )
            # The raw bulk API does not auto-chunk (unlike helpers.async_bulk),
            # so split the operation list by the delete record-count cap to keep
            # each request bounded (mirrors mongo_impl's chunked delete_many).
            chunk = (
                self._max_delete_records_per_batch
                if self._max_delete_records_per_batch > 0
                else len(operations)
            )
            if len(operations) > chunk:
                logger.info(
                    f"[{self.workspace}] {self.namespace} edges: edge delete "
                    f"{len(operations)} ops split into bulk chunks (chunk={chunk})"
                )
            for i in range(0, len(operations), chunk):
                await self.client.bulk(body=operations[i : i + chunk])
            self._edges_dirty = True
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error removing edges: {e}")

    # --- Query operations ---

    async def get_all_labels(self) -> list[str]:
        """Get all node IDs (entity names) sorted alphabetically."""
        if not self._indices_ready:
            return []
        try:
            await self._refresh_graph_indices_if_dirty(refresh_nodes=True)
            labels = []
            pit = await self.client.create_pit(
                index=self._nodes_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": {"match_all": {}},
                        "_source": False,
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": _pit_sort_with_field("entity_id"),
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        labels.append(hit["_id"])
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            labels.sort()
            return labels
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return []

    async def _collect_node_ids(
        self, limit: int, exclude_ids: set[str] | None = None
    ) -> list[str]:
        """Collect up to `limit` node IDs, optionally skipping known IDs."""
        if limit <= 0:
            return []

        excluded = exclude_ids or set()
        if not excluded and limit <= 10000:
            body = {
                "query": {"match_all": {}},
                "_source": False,
                "size": limit,
            }
            resp = await self.client.search(index=self._nodes_index, body=body)
            return [hit["_id"] for hit in resp["hits"]["hits"]]

        node_ids: list[str] = []
        pit = await self.client.create_pit(
            index=self._nodes_index, params={"keep_alive": "1m"}
        )
        pit_id = pit["pit_id"]
        try:
            search_after = None
            while len(node_ids) < limit:
                body = {
                    "query": {"match_all": {}},
                    "_source": False,
                    "size": 10000,
                    "pit": {"id": pit_id, "keep_alive": "1m"},
                    "sort": _pit_sort_with_field("entity_id"),
                }
                if search_after:
                    body["search_after"] = search_after
                resp = await self.client.search(body=body)
                hits = resp["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    node_id = hit["_id"]
                    if node_id in excluded:
                        continue
                    node_ids.append(node_id)
                    if len(node_ids) >= limit:
                        break
                search_after = hits[-1].get("sort")
                if len(hits) < 10000:
                    break
        finally:
            try:
                await self.client.delete_pit(body={"pit_id": [pit_id]})
            except Exception:
                pass

        return node_ids

    @staticmethod
    def _edge_rank_key(edge: dict[str, Any]) -> tuple[int, float]:
        """Rank traversal edges by shallower depth first, then higher weight."""
        depth = edge.get("_depth", edge.get("depth", 0))
        try:
            depth_value = int(depth)
        except (TypeError, ValueError):
            depth_value = 0

        weight = edge.get("weight", 0)
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            weight_value = 0.0

        return (depth_value, -weight_value)

    async def _append_edges_between_nodes(
        self, node_ids: list[str], result: KnowledgeGraph
    ) -> None:
        """Append all edges whose source and target are both in `node_ids`."""
        if not node_ids:
            return

        edge_query = {
            "bool": {
                "must": [
                    {"terms": {"source_node_id": node_ids}},
                    {"terms": {"target_node_id": node_ids}},
                ]
            }
        }
        seen_edges = set()
        pit = await self.client.create_pit(
            index=self._edges_index, params={"keep_alive": "1m"}
        )
        pit_id = pit["pit_id"]
        try:
            search_after = None
            while True:
                edge_body = {
                    "query": edge_query,
                    "size": 10000,
                    "pit": {"id": pit_id, "keep_alive": "1m"},
                    "sort": _pit_sort_with_composite_key(
                        "source_node_id", "target_node_id"
                    ),
                }
                if search_after:
                    edge_body["search_after"] = search_after
                edge_resp = await self.client.search(body=edge_body)
                hits = edge_resp["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    e = hit["_source"]
                    eid = f"{e['source_node_id']}-{e['target_node_id']}"
                    if eid not in seen_edges:
                        seen_edges.add(eid)
                        result.edges.append(self._construct_graph_edge(eid, e))
                search_after = hits[-1].get("sort")
                if len(hits) < 10000:
                    break
        finally:
            try:
                await self.client.delete_pit(body={"pit_id": [pit_id]})
            except Exception:
                pass

    def _construct_graph_node(self, node_id, node_data: dict) -> KnowledgeGraphNode:
        return KnowledgeGraphNode(
            id=node_id,
            labels=[node_id],
            properties={
                k: v
                for k, v in node_data.items()
                if k
                not in (
                    "_id",
                    "entity_id",
                    "source_ids",
                    "connected_edges",
                    "edge_count",
                )
            },
        )

    def _construct_graph_edge(self, edge_id: str, edge: dict) -> KnowledgeGraphEdge:
        return KnowledgeGraphEdge(
            id=edge_id,
            type=edge.get("relationship", ""),
            source=edge["source_node_id"],
            target=edge["target_node_id"],
            properties={
                k: v
                for k, v in edge.items()
                if k
                not in (
                    "_id",
                    "source_node_id",
                    "target_node_id",
                    "relationship",
                    "source_ids",
                )
            },
        )

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """Retrieve a subgraph via PPL graphlookup (if available) or client-side BFS."""
        if not self._indices_ready:
            return KnowledgeGraph()
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        result = KnowledgeGraph()
        start = time.perf_counter()

        try:
            await self._refresh_graph_indices_if_dirty(
                refresh_nodes=True, refresh_edges=True
            )
            if node_label == "*":
                result = await self._get_knowledge_graph_all(max_nodes)
            elif self._ppl_graphlookup_available:
                result = await self._bfs_subgraph_ppl(node_label, max_depth, max_nodes)
            else:
                result = await self._bfs_subgraph(node_label, max_depth, max_nodes)

            duration = time.perf_counter() - start
            logger.info(
                f"[{self.workspace}] Subgraph query in {duration:.4f}s | "
                f"Nodes: {len(result.nodes)} | Edges: {len(result.edges)} | Truncated: {result.is_truncated}"
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
                return KnowledgeGraph()
            logger.error(f"[{self.workspace}] Graph query failed: {e}")

        return result

    async def _get_knowledge_graph_all(self, max_nodes: int) -> KnowledgeGraph:
        """Get all nodes (up to max_nodes, ranked by degree) and their interconnecting edges."""
        result = KnowledgeGraph()
        if not self._indices_ready:
            return result
        try:
            total = (await self.client.count(index=self._nodes_index))["count"]
            result.is_truncated = total > max_nodes

            if result.is_truncated:
                # Get top nodes by degree
                body = {
                    "size": 0,
                    "aggs": {
                        "src": {
                            "terms": {
                                "field": "source_node_id",
                                "size": max_nodes,
                            }
                        },
                        "tgt": {
                            "terms": {
                                "field": "target_node_id",
                                "size": max_nodes,
                            }
                        },
                    },
                }
                resp = await self.client.search(index=self._edges_index, body=body)
                degree_map = {}
                for bucket in resp["aggregations"]["src"]["buckets"]:
                    degree_map[bucket["key"]] = (
                        degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                    )
                for bucket in resp["aggregations"]["tgt"]["buckets"]:
                    degree_map[bucket["key"]] = (
                        degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                    )
                top_ids = sorted(degree_map, key=degree_map.get, reverse=True)[
                    :max_nodes
                ]
                if len(top_ids) < max_nodes:
                    top_ids.extend(
                        await self._collect_node_ids(
                            max_nodes - len(top_ids), exclude_ids=set(top_ids)
                        )
                    )
            else:
                top_ids = await self._collect_node_ids(max_nodes)

            # Fetch node data
            if top_ids:
                node_resp = await self.client.mget(
                    index=self._nodes_index, body={"ids": top_ids}
                )
                found_node_ids = []
                for doc in node_resp["docs"]:
                    if doc.get("found"):
                        found_node_ids.append(doc["_id"])
                        result.nodes.append(
                            self._construct_graph_node(doc["_id"], doc["_source"])
                        )

                await self._append_edges_between_nodes(found_node_ids, result)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
                return result
            logger.error(f"[{self.workspace}] Error in get_knowledge_graph_all: {e}")
        return result

    async def _bfs_subgraph_ppl(
        self, start_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """Server-side BFS using PPL graphlookup command.

        Queries the nodes index for the start node, then uses graphLookup to traverse
        the edges index with bidirectional BFS. Uses `flatten` to unnest results and
        `depthField` for depth-based sorting. Falls back to client-side BFS on failure.
        """
        result = KnowledgeGraph()

        # Verify start node exists
        start_node = await self.get_node(start_label)
        if not start_node:
            return result

        result.nodes.append(self._construct_graph_node(start_label, start_node))

        if max_depth == 0:
            return result

        # PPL maxDepth=0 means 1 hop (direct match), so max_depth-1
        ppl_depth = max(0, max_depth - 1)
        escaped = self._escape_ppl(start_label)
        ppl_query = (
            f"source = {self._nodes_index}"
            f" | where entity_id = '{escaped}'"
            f" | graphLookup {self._edges_index}"
            f" start=entity_id"
            f" edge=target_node_id<->source_node_id"
            f" maxDepth={ppl_depth}"
            f" depthField=_depth"
            f" usePIT=true"
            f" as connected_edges"
        )

        try:
            resp = await self.client.transport.perform_request(
                "POST",
                "/_plugins/_ppl",
                body={"query": ppl_query},
            )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] PPL graphlookup failed, falling back to client BFS: {e}"
            )
            return await self._bfs_subgraph(start_label, max_depth, max_nodes)

        # Parse PPL response — schema-driven to avoid fragile positional access
        try:
            datarows = resp.get("datarows", [])
            schema = [col["name"] for col in resp.get("schema", [])]
            ce_idx = (
                schema.index("connected_edges") if "connected_edges" in schema else -1
            )

            # Collect all edge rows from connected_edges arrays
            all_edge_rows = []
            for row in datarows:
                edges_arr = row[ce_idx] if ce_idx >= 0 else []
                if isinstance(edges_arr, list):
                    all_edge_rows.extend(edges_arr)

            if not all_edge_rows:
                return result

            if isinstance(all_edge_rows[0], dict):
                sorted_edge_rows = sorted(all_edge_rows, key=self._edge_rank_key)
            else:
                # Positional array — column positions are unknown, fall back to client BFS
                logger.warning(
                    f"[{self.workspace}] PPL returned positional arrays, falling back to client BFS"
                )
                return await self._bfs_subgraph(start_label, max_depth, max_nodes)

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.warning(
                f"[{self.workspace}] Error parsing PPL response, falling back: {e}"
            )
            return await self._bfs_subgraph(start_label, max_depth, max_nodes)

        ordered_node_ids = [start_label]
        discovered_nodes = {start_label}
        for edge_row in sorted_edge_rows:
            for node_id in (
                edge_row.get("source_node_id"),
                edge_row.get("target_node_id"),
            ):
                if not node_id or node_id in discovered_nodes:
                    continue
                discovered_nodes.add(node_id)
                if len(ordered_node_ids) < max_nodes:
                    ordered_node_ids.append(node_id)

        result.is_truncated = len(discovered_nodes) > max_nodes

        # Batch fetch node data (start node already added)
        new_node_ids = [nid for nid in ordered_node_ids if nid != start_label]
        if new_node_ids:
            node_resp = await self.client.mget(
                index=self._nodes_index, body={"ids": new_node_ids}
            )
            for doc in node_resp["docs"]:
                if doc.get("found"):
                    result.nodes.append(
                        self._construct_graph_node(doc["_id"], doc["_source"])
                    )

        await self._append_edges_between_nodes(ordered_node_ids, result)

        return result

    @staticmethod
    def _escape_ppl(value: str) -> str:
        """Escape a string for safe inclusion in a PPL single-quoted literal.

        Escapes backslashes, single quotes, and control characters that could
        interfere with PPL query parsing.
        """
        value = value.replace("\\", "\\\\").replace("'", "\\'")
        # Strip control characters that could break the PPL string literal
        value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        return value

    @staticmethod
    def _escape_wildcard(value: str) -> str:
        """Escape OpenSearch wildcard special characters in user input.

        Escapes \\, *, and ? so they are treated as literal characters
        rather than wildcard operators, preventing DoS via expensive patterns.
        """
        # Escape backslash first, then wildcard metacharacters
        return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")

    async def _bfs_subgraph(
        self, start_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """BFS traversal from a starting node, batching neighbor lookups per level."""
        result = KnowledgeGraph()
        seen_nodes = set()

        # Verify start node exists
        start_node = await self.get_node(start_label)
        if not start_node:
            return result

        seen_nodes.add(start_label)
        result.nodes.append(self._construct_graph_node(start_label, start_node))

        current_level = [start_label]
        for _ in range(max_depth):
            if not current_level or len(seen_nodes) >= max_nodes:
                break

            # Batch fetch all edges for current level
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"source_node_id": current_level}},
                            {"terms": {"target_node_id": current_level}},
                        ]
                    }
                },
                "_source": ["source_node_id", "target_node_id"],
                "size": 10000,
            }
            try:
                resp = await self.client.search(index=self._edges_index, body=body)
            except OpenSearchException:
                break

            next_level = set()
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]["source_node_id"]
                tgt = hit["_source"]["target_node_id"]
                if src not in seen_nodes:
                    next_level.add(src)
                if tgt not in seen_nodes:
                    next_level.add(tgt)

            # Limit to max_nodes
            new_ids = []
            for nid in next_level:
                if len(seen_nodes) + len(new_ids) >= max_nodes:
                    break
                new_ids.append(nid)

            if new_ids:
                # Batch fetch node data
                node_resp = await self.client.mget(
                    index=self._nodes_index, body={"ids": new_ids}
                )
                for doc in node_resp["docs"]:
                    if doc.get("found"):
                        seen_nodes.add(doc["_id"])
                        result.nodes.append(
                            self._construct_graph_node(doc["_id"], doc["_source"])
                        )

            current_level = new_ids

        # Fetch all edges between seen nodes using PIT scrolling
        all_ids = list(seen_nodes)
        if all_ids:
            try:
                await self._append_edges_between_nodes(all_ids, result)
            except OpenSearchException:
                pass

        result.is_truncated = len(seen_nodes) >= max_nodes
        return result

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes with their properties."""
        if not self._indices_ready:
            return []
        try:
            await self._refresh_graph_indices_if_dirty(refresh_nodes=True)
            nodes = []
            pit = await self.client.create_pit(
                index=self._nodes_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": {"match_all": {}},
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": _pit_sort_with_field("entity_id"),
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        node = hit["_source"]
                        node["id"] = hit["_id"]
                        nodes.append(node)
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            return nodes
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return []

    async def get_all_edges(self) -> list[dict]:
        """Get all edges with source/target fields added."""
        if not self._indices_ready:
            return []
        try:
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            edges = []
            pit = await self.client.create_pit(
                index=self._edges_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": {"match_all": {}},
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": _pit_sort_with_composite_key(
                            "source_node_id", "target_node_id"
                        ),
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        edge = hit["_source"]
                        edge["source"] = edge.get("source_node_id")
                        edge["target"] = edge.get("target_node_id")
                        edges.append(edge)
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            return edges
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return []

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get node labels ranked by edge degree (most connected first)."""
        if not self._indices_ready:
            return []
        try:
            await self._refresh_graph_indices_if_dirty(refresh_edges=True)
            body = {
                "size": 0,
                "aggs": {
                    "src": {"terms": {"field": "source_node_id", "size": limit * 2}},
                    "tgt": {"terms": {"field": "target_node_id", "size": limit * 2}},
                },
            }
            response = await self.client.search(index=self._edges_index, body=body)
            degree_map = {}
            for bucket in response["aggregations"]["src"]["buckets"]:
                degree_map[bucket["key"]] = (
                    degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                )
            for bucket in response["aggregations"]["tgt"]["buckets"]:
                degree_map[bucket["key"]] = (
                    degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                )
            sorted_labels = sorted(degree_map, key=degree_map.get, reverse=True)[:limit]
            return sorted_labels
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search node labels with wildcard and prefix matching."""
        query = query.strip()
        if not query:
            return []
        if not self._indices_ready:
            return []
        try:
            await self._refresh_graph_indices_if_dirty(refresh_nodes=True)
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"entity_id": {"value": query, "boost": 10}}},
                            {
                                "prefix": {
                                    "entity_id": {"value": query.lower(), "boost": 5}
                                }
                            },
                            {
                                "wildcard": {
                                    "entity_id": {
                                        "value": f"*{self._escape_wildcard(query.lower())}*",
                                        "case_insensitive": True,
                                        "boost": 2,
                                    }
                                }
                            },
                        ]
                    }
                },
                "_source": False,
                "size": limit,
            }
            response = await self.client.search(index=self._nodes_index, body=body)
            return [hit["_id"] for hit in response["hits"]["hits"]]
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
            return []

    async def index_done_callback(self) -> None:
        """Refresh both node and edge indices."""
        if not self._indices_ready:
            return
        try:
            await self._refresh_graph_indices_if_dirty(
                refresh_nodes=True, refresh_edges=True
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_indices_missing()
                return
        except Exception:
            pass

    async def drop(self) -> dict[str, str]:
        """Delete both node and edge indices."""
        errors = []
        for idx in (self._nodes_index, self._edges_index):
            try:
                await self.client.indices.delete(index=idx)
                logger.info(f"[{self.workspace}] Dropped graph index: {idx}")
            except NotFoundError:
                logger.info(
                    f"[{self.workspace}] Graph index already missing during drop: {idx}"
                )
            except OpenSearchException as e:
                errors.append(f"{idx}: {e}")
                logger.error(
                    f"[{self.workspace}] Error dropping graph index {idx}: {e}"
                )
            except Exception as e:
                errors.append(f"{idx}: {e}")
                logger.error(
                    f"[{self.workspace}] Unexpected error dropping graph index {idx}: {e}"
                )

        self._mark_indices_missing()

        if errors:
            return {
                "status": "error",
                "message": "Failed to drop graph indices: " + "; ".join(errors),
            }

        try:
            logger.info(f"[{self.workspace}] Dropped graph indices")
            return {"status": "success", "message": "Graph indices dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error finalizing graph drop: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchVectorDBStorage(BaseVectorStorage):
    """Vector storage using OpenSearch k-NN plugin with corrected cosine score handling."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)
    _index_ready: bool = field(default=False, init=False)

    def __init__(
        self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    def __post_init__(self):
        self._validate_embedding_func()
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # Pending writes are flushed via _flush_pending_vector_ops() during
        # index_done_callback() / finalize(). This batches many small upsert()
        # invocations into a single async_bulk roundtrip. See issue #2785.
        self._pending_vector_docs: dict[str, _PendingVectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        # Namespace-keyed lock (multi-process safe) is initialised in
        # initialize(). All buffer reads / writes and any destructive server
        # mutation (delete_by_query, drop, finalize) are serialised through
        # this lock to keep in-process readers race-free during a flush and
        # to order cross-worker flushes against the same OpenSearch index.
        self._flush_lock = None
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
            self._max_delete_records_per_batch,
        ) = _resolve_bulk_batch_limits()

    async def initialize(self):
        """Initialize client and create k-NN vector index."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_knn_index_if_not_exists()
            self._index_ready = True
            logger.debug(
                f"[{self.workspace}] OpenSearch Vector storage initialized: {self._index_name}"
            )
        if self._flush_lock is None:
            self._flush_lock = get_namespace_lock(
                self.namespace, workspace=self.workspace
            )

    async def _ensure_index_ready(self):
        """Recreate the vector index before the next write if it is missing."""
        if self._index_ready:
            return
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            if not self._index_ready:
                await self._create_knn_index_if_not_exists()
                self._index_ready = True

    def _mark_index_missing(self):
        """Mark the vector index as unavailable for subsequent read short-circuiting."""
        self._index_ready = False

    async def _create_knn_index_if_not_exists(self):
        try:
            if await self.client.indices.exists(index=self._index_name):
                # Validate existing index dimension
                try:
                    mapping = await self.client.indices.get_mapping(
                        index=self._index_name
                    )
                    existing_dim = (
                        mapping[self._index_name]["mappings"]["properties"]
                        .get("vector", {})
                        .get("dimension")
                    )
                    expected_dim = self.embedding_func.embedding_dim
                    if existing_dim is not None and existing_dim != expected_dim:
                        raise ValueError(
                            f"Vector dimension mismatch! Index '{self._index_name}' has "
                            f"dimension {existing_dim}, but current embedding model expects "
                            f"dimension {expected_dim}. Please drop the existing index or "
                            f"use an embedding model with matching dimensions."
                        )
                except (KeyError, TypeError):
                    logger.warning(
                        f"[{self.workspace}] Could not read vector mapping for index "
                        f"'{self._index_name}'; skipping dimension validation"
                    )
                return

            ef_construction = int(
                _get_opensearch_env("OPENSEARCH_KNN_EF_CONSTRUCTION", "200")
            )
            m = int(_get_opensearch_env("OPENSEARCH_KNN_M", "16"))
            ef_search = int(_get_opensearch_env("OPENSEARCH_KNN_EF_SEARCH", "100"))

            body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": ef_search,
                        "number_of_shards": _get_index_number_of_shards(),
                        "number_of_replicas": _get_index_number_of_replicas(),
                    }
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": self.embedding_func.embedding_dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": ef_construction,
                                    "m": m,
                                },
                            },
                        },
                        "content": {"type": "text"},
                        "entity_name": {"type": "keyword"},
                        "src_id": {"type": "keyword"},
                        "tgt_id": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "created_at": {"type": "long"},
                    },
                    "dynamic": True,
                },
            }
            await self.client.indices.create(index=self._index_name, body=body)
            logger.info(
                f"[{self.workspace}] Created k-NN index: {self._index_name} "
                f"(dim={self.embedding_func.embedding_dim})"
            )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                logger.error(f"[{self.workspace}] Error creating k-NN index: {e}")
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating k-NN index: {e}")
            raise

    async def finalize(self):
        """Flush pending writes and release the OpenSearch client connection.

        Regular flush failures (any ``Exception``) are captured so they
        can be re-surfaced as a ``RuntimeError`` that names the unflushed
        buffer counts -- otherwise ``LightRAG.finalize_storages()`` would
        log the storage as successfully finalized while writes silently
        failed to reach OpenSearch.

        ``BaseException`` subclasses other than ``Exception`` (notably
        ``asyncio.CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit``)
        are NOT caught: they propagate through the ``finally`` block so
        shutdown cancellation is honoured and not silently swallowed.
        The client is released in ``finally`` so it does not leak whether
        the flush succeeded, failed, or was cancelled.
        """
        flush_error: Exception | None = None
        try:
            try:
                await self._flush_pending_vector_ops()
            except Exception as e:
                flush_error = e
        finally:
            if self.client is not None:
                await ClientManager.release_client(self.client)
                self.client = None

        pending_docs = len(self._pending_vector_docs)
        pending_deletes = len(self._pending_vector_deletes)

        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] OpenSearchVectorDBStorage.finalize() "
                f"flush raised; {pending_docs} pending upserts and "
                f"{pending_deletes} pending deletes were left buffered "
                f"(client released, data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] OpenSearchVectorDBStorage.finalize() "
                f"left {pending_docs} pending upserts and {pending_deletes} "
                f"pending deletes buffered after final flush attempt "
                f"(transient bulk failure); these writes have been lost"
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vector docs for embedding and batched flush.

        Docs are buffered in ``self._pending_vector_docs`` and flushed in a
        single ``async_bulk`` call during ``index_done_callback()`` /
        ``finalize()``. This is a behavioral change relative to per-call
        ``async_bulk``: writes are not durable in OpenSearch until the next
        flush, which matches the contract used by other LightRAG storage
        backends ("changes will be persisted during the next
        index_done_callback").

        Embedding is deferred to the flush path so repeated upserts of the
        same id and many small upsert calls can be embedded once in a single
        batch. Flush holds the namespace lock while embedding and bulk
        indexing so cross-worker destructive mutations cannot interleave with
        partially-flushed vector writes.
        """
        if not data:
            return
        await self._ensure_index_ready()
        logger.debug(
            f"[{self.workspace}] Buffering {len(data)} vectors for {self.namespace}"
        )
        current_time = int(time.time())

        pending_docs: list[tuple[str, _PendingVectorDoc]] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            content = v["content"]
            source = {
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            pending_docs.append(
                (
                    k,
                    _PendingVectorDoc(
                        source=source,
                        content=content,
                    ),
                )
            )
            await _cooperative_yield(i)

        # Buffer: an upsert overrides a pending delete on the same id.
        async with self._flush_lock:
            for doc_id, pending_doc in pending_docs:
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = pending_doc

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered vector upserts and deletes via a single async_bulk call.

        Concurrency contract: the entire flush, including deferred embedding,
        runs under ``_flush_lock`` (a ``get_namespace_lock`` instance), and so
        do all buffer reads / writes and destructive server mutations on this
        storage. That keeps the operation sequential within the process and
        orders concurrent cross-worker flushes against the same OpenSearch
        index.

        Embedding deliberately runs *inside* this lock (not in ``upsert`` or
        lock-free): it makes deferred embedding and bulk indexing atomic
        against concurrent upserts and destructive mutations (``drop`` /
        ``delete_entity_relation``). This is what lets
        ``index_done_callback`` / ``finalize`` promise that every buffered
        vector is embedded and persisted on return. Moving embedding out of
        the lock to avoid blocking reads would let a destructive op
        interleave between embed and bulk and resurrect or drop vectors out
        of order -- do not do it.

        Failure handling:
          * If ``_ensure_index_ready`` raises, the buffers are left intact
            and the next flush will retry.
          * If embedding raises, the buffers are left intact and the next
            flush will retry. Model providers already retry internally, so
            this is treated like a persistence failure.
          * If ``async_bulk`` itself raises (network / parse error), the
            buffers are left intact and the next flush will retry. Index
            ops are idempotent on ``_id`` and a re-issued delete on a
            missing doc is filtered out as 404 by ``_extract_bulk_failed_ids``.
          * Per-doc retryable failures (408 / 429 / 5xx) stay in the
            buffer for the next flush.
          * Per-doc non-retryable failures (most 4xx, mapping errors) are
            cleared from the buffer and logged with a sample of
            (op, id, status, error) so operators can diagnose them.
        """
        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self.client is None:
                return

            # If the index disappeared between writes (e.g. read path
            # marked it missing), recreate it now. Failure leaves the
            # buffers untouched and bubbles up to the caller.
            await self._ensure_index_ready()

            pending_docs = self._pending_vector_docs
            pending_deletes = self._pending_vector_deletes

            docs_to_embed = [
                (doc_id, pending_doc)
                for doc_id, pending_doc in pending_docs.items()
                if pending_doc.vector is None
            ]
            if docs_to_embed:
                contents = [pending_doc.content for _, pending_doc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                # TEMP diagnostic (remove later): confirm deferred batching is
                # actually coalescing per-id upserts. defer working -> docs >>
                # batches; eager/per-id -> docs == batches == 1 every flush.
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: embedding "
                    f"{len(docs_to_embed)} vectors in {len(batches)} batch(es) "
                    f"(batch_num={self._max_batch_size})"
                )
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error embedding pending vector ops "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                # Explicit check (not assert): a count mismatch would silently
                # truncate via zip() under `python -O`, mis-pairing vectors with
                # docs. Raise instead so buffers stay intact for the next flush.
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((_, pending_doc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pending_doc.vector = embedding.tolist()
                    await _cooperative_yield(i)

            # Deletes and upserts go as separate async_bulk requests so the
            # delete record-count cap can differ from the upsert cap (mirrors
            # mongo_impl's separate upsert/delete phases). The two buffers are
            # disjoint -- delete() pops from pending_docs and upsert() discards
            # from pending_deletes -- so request ordering is irrelevant.
            delete_actions: list[dict[str, Any]] = [
                {
                    "_op_type": "delete",
                    "_index": self._index_name,
                    "_id": doc_id,
                }
                for doc_id in pending_deletes
            ]
            committed_doc_ids: set[str] = set()
            index_actions: list[dict[str, Any]] = []
            for doc_id, pending_doc in pending_docs.items():
                if pending_doc.vector is None:
                    continue
                committed_doc_ids.add(doc_id)
                index_actions.append(
                    {
                        "_op_type": "index",
                        "_index": self._index_name,
                        "_id": doc_id,
                        "_source": {
                            **pending_doc.source,
                            "vector": pending_doc.vector,
                        },
                    }
                )
            if not delete_actions and not index_actions:
                return

            try:
                # No per-operation refresh: search visibility is established
                # by the refresh in index_done_callback().
                log_prefix = f"[{self.workspace}] {self.namespace} flush:"
                _, del_failed = await _run_chunked_async_bulk(
                    self.client,
                    delete_actions,
                    max_payload_bytes=self._max_upsert_payload_bytes,
                    max_records_per_batch=self._max_delete_records_per_batch,
                    log_prefix=log_prefix,
                    what="delete",
                    raise_on_error=False,
                )
                _, idx_failed = await _run_chunked_async_bulk(
                    self.client,
                    index_actions,
                    max_payload_bytes=self._max_upsert_payload_bytes,
                    max_records_per_batch=self._max_upsert_records_per_batch,
                    log_prefix=log_prefix,
                    what="upsert",
                    raise_on_error=False,
                )
                failed = list(del_failed) + list(idx_failed)
            except OpenSearchException as e:
                logger.error(
                    f"[{self.workspace}] Error flushing vector ops "
                    f"(upserts={len(pending_docs)}, "
                    f"deletes={len(pending_deletes)}): {e}"
                )
                # Bulk did not return per-doc statuses, so keep everything
                # buffered for the next flush.
                raise

            retryable_ids, non_retryable_ops = _extract_bulk_failed_ids(failed)

            # Clear successful and non-retryable entries; keep retryable ones
            # in place for the next flush.
            for doc_id in committed_doc_ids:
                if doc_id not in retryable_ids:
                    pending_docs.pop(doc_id, None)
            new_deletes: set[str] = set()
            for doc_id in pending_deletes:
                if doc_id in retryable_ids:
                    new_deletes.add(doc_id)
            pending_deletes.clear()
            pending_deletes.update(new_deletes)

            if retryable_ids:
                logger.warning(
                    f"[{self.workspace}] {len(retryable_ids)} vector ops will "
                    f"retry on the next flush (transient failure)"
                )
            if non_retryable_ops:
                sample = non_retryable_ops[:5]
                sample_text = ", ".join(
                    f"{op.op}/{op.doc_id}/status={op.status}/{op.error}"
                    for op in sample
                )
                logger.warning(
                    f"[{self.workspace}] {len(non_retryable_ops)} vector ops "
                    f"failed permanently and were dropped (non-retryable status). "
                    f"Sample: {sample_text}"
                )

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """k-NN similarity search with cosine score conversion for lucene engine."""
        if not self._index_ready:
            return []
        if query_embedding is not None:
            query_vector = (
                query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else list(query_embedding)
            )
        else:
            embedding = await self.embedding_func([query], context="query", _priority=5)
            query_vector = embedding[0].tolist()

        search_body = {
            "size": top_k,
            "query": {"knn": {"vector": {"vector": query_vector, "k": top_k}}},
            "_source": {"excludes": ["vector"]},
        }
        try:
            response = await self.client.search(
                index=self._index_name, body=search_body
            )
            results = []
            for hit in response["hits"]["hits"]:
                # OpenSearch k-NN with lucene engine and cosinesimil space type
                # returns scores that can be used directly as similarity measure.
                score = hit["_score"]

                if score >= self.cosine_better_than_threshold:
                    doc = hit["_source"]
                    doc["id"] = hit["_id"]
                    doc["distance"] = score
                    results.append(doc)
            logger.info(
                f"[{self.workspace}] Vector query on {self._index_name}: "
                f"top_k={top_k}, threshold={self.cosine_better_than_threshold}, "
                f"total_hits={len(response['hits']['hits'])}, "
                f"passed_filter={len(results)}, "
                f"score_range=[{min((h['_score'] for h in response['hits']['hits']), default=0):.4f}, "
                f"{max((h['_score'] for h in response['hits']['hits']), default=0):.4f}]"
            )
            return results
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return []
            logger.error(f"[{self.workspace}] Error querying vectors: {e}")
            return []

    async def index_done_callback(self) -> None:
        """Flush pending vector ops and refresh the index for k-NN visibility.

        Flush runs first so that a previously-missing index gets recreated
        by ``_flush_pending_vector_ops`` (via ``_ensure_index_ready``)
        before any buffered writes are abandoned. The refresh step is
        skipped only when the index is still not ready after the flush
        attempt -- refreshing a half-built index is pointless.

        Durability contract: each call embeds and bulk-indexes the *entire*
        pending buffer in one shot. Deferred embedding runs inside
        ``_flush_pending_vector_ops``'s ``_flush_lock`` section (not in
        ``upsert``) precisely so this callback can guarantee every buffered
        vector is embedded and flushed together; only transient per-doc
        failures stay buffered for the next flush. Do not move embedding
        out of the lock -- see ``_flush_pending_vector_ops`` for why.
        """
        await self._flush_pending_vector_ops()
        if not self._index_ready:
            return
        try:
            await self.client.indices.refresh(index=self._index_name)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
        except Exception:
            pass

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a vector document by ID, with read-your-writes against the buffer.

        The ``vector`` field is stripped from the result to match every other
        LightRAG vector backend (see ``NanoVectorDBStorage.get_by_id``).
        Callers that need the embedding itself must use ``get_vectors_by_ids``.
        """
        # Buffer lookups happen under the namespace lock so an in-flight
        # flush is observed as either "completely before" or "completely
        # after" -- never as a snapshot-swapped intermediate state.
        async with self._flush_lock:
            if id in self._pending_vector_deletes:
                return None
            pending = self._pending_vector_docs.get(id)
            if pending is not None:
                # pending.source is built in upsert from created_at + meta_fields
                # and never carries the embedding, so no "vector" strip is needed
                # here (unlike the mget path below, which excludes it server-side).
                doc = dict(pending.source)
                doc["id"] = id
                return doc
            if not self._index_ready:
                return None
        # Network IO outside the lock so mget RTT doesn't block flush.
        try:
            response = await _mget_optional_doc(
                self.client,
                self._index_name,
                id,
                source_excludes=["vector"],
            )
            if response is None:
                return None
            doc = response["_source"]
            doc.pop("vector", None)  # defensive in case _source_excludes is ignored
            doc["id"] = response["_id"]
            return doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting vector {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector documents by IDs (read-your-writes), preserving order.

        The ``vector`` field is stripped from each result; see ``get_by_id``.
        """
        if not ids:
            return []
        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    # pending.source never carries the embedding; see get_by_id.
                    doc = dict(pending.source)
                    doc["id"] = doc_id
                    buffered[doc_id] = doc
                    continue
                remaining.append(doc_id)
            index_ready = self._index_ready

        doc_map: dict[str, dict[str, Any] | None] = {}
        if remaining and index_ready:
            try:
                response = await self.client.mget(
                    index=self._index_name,
                    body={"ids": remaining},
                    _source_excludes=["vector"],
                )
                for doc in response["docs"]:
                    if doc.get("found"):
                        data = doc["_source"]
                        data.pop("vector", None)
                        data["id"] = doc["_id"]
                        doc_map[doc["_id"]] = data
            except OpenSearchException as e:
                if _is_missing_index_error(e):
                    self._mark_index_missing()
                else:
                    logger.error(
                        f"[{self.workspace}] Error getting vectors by ids: {e}"
                    )

        return [
            buffered[doc_id] if doc_id in buffered else doc_map.get(doc_id)
            for doc_id in ids
        ]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings for given IDs, with read-your-writes."""
        if not ids:
            return {}
        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            docs_to_embed: list[tuple[str, _PendingVectorDoc]] = []
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    if pending.vector is None:
                        docs_to_embed.append((doc_id, pending))
                    else:
                        result[doc_id] = pending.vector
                    continue
                remaining.append(doc_id)
            index_ready = self._index_ready

            if docs_to_embed:
                contents = [pending_doc.content for _, pending_doc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error lazily embedding pending vectors "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                # Explicit check (not assert): see _flush_pending_vector_ops.
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((doc_id, pending_doc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pending_doc.vector = embedding.tolist()
                    result[doc_id] = pending_doc.vector
                    await _cooperative_yield(i)

        if not remaining:
            return result
        if not index_ready:
            return result
        try:
            response = await self.client.mget(
                index=self._index_name,
                body={"ids": remaining},
                _source_includes=["vector"],
            )
            for doc in response["docs"]:
                if doc.get("found") and "vector" in doc.get("_source", {}):
                    result[doc["_id"]] = doc["_source"]["vector"]
            return result
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return result
            logger.error(f"[{self.workspace}] Error getting vectors: {e}")
            return result

    async def delete(self, ids: list[str]) -> None:
        """Buffer vector deletes for batched flush.

        A delete cancels any pending upsert for the same id; the actual bulk
        delete is performed by ``_flush_pending_vector_ops`` during the next
        ``index_done_callback`` / ``finalize`` call.
        """
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for {len(ids)} vectors in {self.namespace}"
        )

    async def delete_entity(self, entity_name: str) -> None:
        """Buffer an entity vector delete by computing its hash ID."""
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        async with self._flush_lock:
            self._pending_vector_docs.pop(entity_id, None)
            self._pending_vector_deletes.add(entity_id)
        logger.debug(f"[{self.workspace}] Buffered delete for entity {entity_name}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt.

        The whole method runs under ``_flush_lock`` so the ``delete_by_query``
        cannot interleave with an in-flight bulk indexing of a related doc.
        Buffered upserts that match are pruned in-memory; persisted rows are
        removed via the server-side ``delete_by_query``.

        Buffer semantics — post-prune with caller short-circuit contract:
            Matching pending upserts are pruned **only after** the
            server-side ``delete_by_query`` succeeds (or returns the
            equivalent of "index already missing"). On any other server
            failure the exception is re-raised and the pending buffer
            stays intact so a higher-level retry can still observe the
            buffered relation vectors. Correctness relies on the caller
            short-circuiting before ``index_done_callback`` can run;
            ``adelete_by_entity`` in ``utils_graph.py`` honors this.

            Previously this method pre-pruned the buffer and swallowed
            ``OpenSearchException`` into a ``logger.error`` — that
            combination silently dropped both the buffered relation
            vectors and the server-side failure signal, leaving the
            caller's graph + vector store permanently inconsistent.
        """

        def _prune_pending() -> None:
            for doc_id in [
                k
                for k, v in self._pending_vector_docs.items()
                if v.source.get("src_id") == entity_name
                or v.source.get("tgt_id") == entity_name
            ]:
                self._pending_vector_docs.pop(doc_id, None)

        async with self._flush_lock:
            if not self._index_ready:
                # No server state to mutate; buffer prune is the only
                # delete intent we can record.
                _prune_pending()
                return

            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"src_id": entity_name}},
                            {"term": {"tgt_id": entity_name}},
                        ]
                    }
                }
            }
            try:
                # conflicts="proceed" tolerates stale search view after refresh removal.
                await self.client.delete_by_query(
                    index=self._index_name, body=body, params={"conflicts": "proceed"}
                )
            except OpenSearchException as e:
                if _is_missing_index_error(e):
                    # Index gone is equivalent to "all rows already
                    # deleted" — safe to prune pending and treat as
                    # success.
                    self._mark_index_missing()
                    _prune_pending()
                    return
                logger.error(
                    f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
                )
                raise

            # Server-side delete succeeded — safe to prune the pending
            # buffer so subsequent flushes don't re-upsert the deleted
            # relations.
            _prune_pending()
            logger.debug(
                f"[{self.workspace}] Deleted relations for entity {entity_name}"
            )

    async def drop(self) -> dict[str, str]:
        """Delete and recreate the vector index, discarding pending buffers.

        Runs entirely under ``_flush_lock`` so a concurrent flush / upsert
        cannot land writes against an index that is being deleted and
        rebuilt.
        """
        async with self._flush_lock:
            # Pending writes are meaningless once the index is dropped.
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()
            try:
                try:
                    await self.client.indices.delete(index=self._index_name)
                    logger.info(
                        f"[{self.workspace}] Dropped vector index: {self._index_name}"
                    )
                except NotFoundError:
                    logger.info(
                        f"[{self.workspace}] Vector index already missing during drop: {self._index_name}"
                    )
                # Recreate the index
                await self._create_knn_index_if_not_exists()
                self._index_ready = True
                logger.info(
                    f"[{self.workspace}] Dropped and recreated vector index: {self._index_name}"
                )
                return {
                    "status": "success",
                    "message": f"Vector index {self._index_name} dropped and recreated",
                }
            except OpenSearchException as e:
                self._mark_index_missing()
                logger.error(f"[{self.workspace}] Error dropping vector index: {e}")
                return {"status": "error", "message": str(e)}
            except Exception as e:
                self._mark_index_missing()
                logger.error(
                    f"[{self.workspace}] Unexpected error dropping vector index: {e}"
                )
                return {"status": "error", "message": str(e)}
