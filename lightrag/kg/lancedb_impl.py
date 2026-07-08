"""LanceDB unified storage backend for LightRAG.

Implements all four storage types on top of a single embedded LanceDB
database directory:

- ``LanceDBKVStorage``        (BaseKVStorage)
- ``LanceDBVectorStorage``    (BaseVectorStorage)
- ``LanceDBGraphStorage``     (BaseGraphStorage)
- ``LanceDBDocStatusStorage`` (DocStatusStorage)

LanceDB is an embedded, serverless vector database built on the Lance
columnar format. This backend keeps vectors, the knowledge graph, KV data
and document status in one local directory with no external service.

Storage layout (one Lance table per namespace, workspace-prefixed):

- KV:         ``{workspace}_{namespace}``            (id, payload, create_time, update_time)
- Vector:     ``{workspace}_{namespace}[_{model}]``  (id, vector, content, src_id, tgt_id, created_at, payload)
- Graph:      ``{workspace}_{namespace}_nodes``      (id, payload)
              ``{workspace}_{namespace}_edges``      (id, src, tgt, payload)
- Doc status: ``{workspace}_{namespace}``            (id, status, file_path, track_id, content_hash, payload)

Full-fidelity records are stored as JSON strings in the ``payload`` column;
columns that need SQL filtering (ids, src/tgt, status, ...) are typed.

Concurrency model: all storage instances in a process share one
``AsyncConnection`` (and one ``AsyncTable`` handle per table) through
``ClientManager``. Writes to a table are serialized with a per-table
``asyncio.Lock`` because concurrent ``merge_insert`` calls with overlapping
keys can commit duplicate rows under LanceDB's optimistic concurrency.
Multi-process deployments (e.g. gunicorn workers) are not recommended with
this backend; use a server-based backend for those.
"""

from __future__ import annotations

import asyncio
import configparser
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Iterable, Iterator, final

import numpy as np

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..constants import DEFAULT_QUERY_PRIORITY
from ..exceptions import StorageNotInitializedError
from ..kg.shared_storage import get_data_init_lock, get_namespace_lock
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import (
    _cooperative_yield,
    compute_mdhash_id,
    get_pinyin_sort_key,
    logger,
    validate_workspace,
)

import pipmaster as pm

if not pm.is_installed("lancedb"):
    pm.install("lancedb")

import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402
from lancedb.index import FTS  # noqa: E402

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Max number of ids per SQL IN (...) clause; larger sets are chunked.
_SQL_IN_CHUNK = 500


def _get_lancedb_env(key: str, fallback: str | None = None) -> str | None:
    """Env var first, then [lancedb] section of config.ini, then fallback."""
    cfg_key = key.replace("LANCEDB_", "").lower()
    return os.environ.get(key, config.get("lancedb", cfg_key, fallback=fallback))


def _resolve_lancedb_uri(global_config: dict[str, Any]) -> str:
    uri = _get_lancedb_env("LANCEDB_URI")
    if uri and uri.strip():
        uri = uri.strip()
    else:
        working_dir = global_config.get("working_dir") or "."
        uri = os.path.join(working_dir, "lancedb")
    # Normalize local paths so different spellings of the same directory
    # share one client (and its per-table write locks). Leave remote URIs
    # (s3://, az://, ...) untouched.
    if "://" not in uri:
        uri = os.path.abspath(uri)
    return uri


def _resolve_workspace(workspace: str) -> str:
    """LANCEDB_WORKSPACE env override beats the constructor parameter."""
    env_workspace = _get_lancedb_env("LANCEDB_WORKSPACE")
    if env_workspace and env_workspace.strip():
        return env_workspace.strip()
    return workspace or ""


def _build_table_name(workspace: str, namespace: str) -> tuple[str, str, str]:
    """Return (effective_workspace, final_namespace, sanitized_table_name)."""
    effective = _resolve_workspace(workspace)
    final_namespace = f"{effective}_{namespace}" if effective else namespace
    return effective, final_namespace, _sanitize_table_name(final_namespace)


def _sanitize_table_name(name: str) -> str:
    # Lowercase to avoid case-only collisions on case-insensitive filesystems
    # (LanceDB tables are directories).
    sanitized = re.sub(r"[^a-z0-9_-]", "_", name.lower())
    if sanitized and sanitized[0] in "-_":
        sanitized = "x" + sanitized
    # Folding is lossy: distinct names like "TeamA" and "teama" (both valid
    # workspaces) would otherwise share a table. Disambiguate any name the
    # folding changed with a short hash of the original.
    if sanitized != name:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        sanitized = f"{sanitized}_{digest}"
    return sanitized


def _sql_quote(value: Any) -> str:
    """Quote a value as a SQL string literal (single quotes doubled)."""
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(column: str, values: Iterable[Any]) -> str:
    return f"{column} IN ({', '.join(_sql_quote(v) for v in values)})"


def _iter_chunks(values: list[Any], size: int = _SQL_IN_CHUNK) -> Iterator[list[Any]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _json_dumps(data: dict[str, Any]) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except TypeError as e:
        raise TypeError(
            f"LanceDB storage requires JSON-serializable records: {e}"
        ) from e


def _json_loads(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    return json.loads(payload)


def _status_value(status: Any) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _kv_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("payload", pa.string()),
            pa.field("create_time", pa.int64()),
            pa.field("update_time", pa.int64()),
        ]
    )


def _vector_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("content", pa.string()),
            pa.field("src_id", pa.string()),
            pa.field("tgt_id", pa.string()),
            pa.field("created_at", pa.int64()),
            pa.field("payload", pa.string()),
        ]
    )


def _graph_node_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("payload", pa.string()),
        ]
    )


def _graph_edge_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("src", pa.string()),
            pa.field("tgt", pa.string()),
            pa.field("payload", pa.string()),
        ]
    )


def _doc_status_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("track_id", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("payload", pa.string()),
        ]
    )


def _canonical_edge_id(source_node_id: str, target_node_id: str) -> str:
    lo, hi = sorted((source_node_id, target_node_id))
    return compute_mdhash_id(f"{lo}-{hi}", prefix="edge-")


class LanceDBClient:
    """Shared per-URI LanceDB connection with table-handle cache and locks.

    All storage instances pointing at the same database directory share one
    connection and one ``AsyncTable`` handle per table so that writes made
    through one storage are immediately visible to the others. Per-table
    ``asyncio.Lock``s serialize writes: concurrent ``merge_insert`` calls
    with the same keys would otherwise commit duplicate rows.
    """

    def __init__(self, uri: str):
        self.uri = uri
        self._connection = None
        self._tables: dict[str, Any] = {}
        self._table_locks: dict[str, asyncio.Lock] = {}
        self._creation_lock = asyncio.Lock()
        self._writes_since_optimize: dict[str, int] = {}
        interval_raw = _get_lancedb_env("LANCEDB_READ_CONSISTENCY_INTERVAL", "0")
        try:
            self._read_consistency_interval = timedelta(seconds=float(interval_raw))
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid LANCEDB_READ_CONSISTENCY_INTERVAL={interval_raw!r}, using 0"
            )
            self._read_consistency_interval = timedelta(0)
        threshold_raw = _get_lancedb_env("LANCEDB_OPTIMIZE_THRESHOLD", "64")
        try:
            self.optimize_threshold = int(threshold_raw)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid LANCEDB_OPTIMIZE_THRESHOLD={threshold_raw!r}, using 64"
            )
            self.optimize_threshold = 64

    async def connect(self) -> None:
        if self._connection is None:
            # Only makedirs for local paths; remote URIs (s3://, az://, ...)
            # are managed by LanceDB and must not be created on the local FS
            # (os.makedirs would otherwise make a bogus "s3:" dir in the cwd).
            if "://" not in self.uri:
                os.makedirs(self.uri, exist_ok=True)
            self._connection = await lancedb.connect_async(
                self.uri,
                read_consistency_interval=self._read_consistency_interval,
            )
            logger.info(f"LanceDB connected: {self.uri}")

    async def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._tables.clear()
        self._writes_since_optimize.clear()

    def table(self, name: str):
        table = self._tables.get(name)
        if table is None:
            raise StorageNotInitializedError(f"LanceDB table {name}")
        return table

    def table_lock(self, name: str) -> asyncio.Lock:
        lock = self._table_locks.get(name)
        if lock is None:
            lock = self._table_locks.setdefault(name, asyncio.Lock())
        return lock

    async def get_or_create_table(self, name: str, schema: pa.Schema):
        table = self._tables.get(name)
        if table is not None:
            return table
        async with self._creation_lock:
            table = self._tables.get(name)
            if table is None:
                table = await self._connection.create_table(
                    name, schema=schema, exist_ok=True
                )
                self._tables[name] = table
        return table

    async def drop_and_recreate_table(self, name: str, schema: pa.Schema):
        async with self._creation_lock:
            await self._connection.drop_table(name, ignore_missing=True)
            table = await self._connection.create_table(name, schema=schema)
            self._tables[name] = table
            self._writes_since_optimize[name] = 0
        return table

    def bump_writes(self, name: str) -> None:
        self._writes_since_optimize[name] = self._writes_since_optimize.get(name, 0) + 1

    async def maybe_optimize(self, name: str) -> None:
        """Compact fragments and fold new rows into indexes periodically.

        Every LanceDB write commits a new table version; without periodic
        optimize() small fragments accumulate and slow scans down.
        Best-effort: failures are logged, never raised.
        """
        if self.optimize_threshold <= 0:
            return
        if self._writes_since_optimize.get(name, 0) < self.optimize_threshold:
            return
        try:
            table = self._tables.get(name)
            if table is not None:
                await table.optimize()
                self._writes_since_optimize[name] = 0
        except Exception as e:
            logger.warning(f"LanceDB optimize failed for table {name}: {e}")


class ClientManager:
    """Reference-counted shared LanceDB clients, keyed by database URI."""

    _instances: dict[str, dict[str, Any]] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls, uri: str) -> LanceDBClient:
        async with cls._lock:
            entry = cls._instances.get(uri)
            if entry is None:
                client = LanceDBClient(uri)
                await client.connect()
                entry = {"client": client, "ref_count": 0}
                cls._instances[uri] = entry
            entry["ref_count"] += 1
            return entry["client"]

    @classmethod
    async def release_client(cls, client: LanceDBClient | None) -> None:
        if client is None:
            return
        async with cls._lock:
            entry = cls._instances.get(client.uri)
            if entry is None or entry["client"] is not client:
                await client.close()
                return
            entry["ref_count"] -= 1
            if entry["ref_count"] <= 0:
                await client.close()
                del cls._instances[client.uri]


async def _fetch_rows(
    table,
    where: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run a plain (non-vector) scan; plain scans return all rows by default."""
    query = table.query()
    if where:
        query = query.where(where)
    if columns is not None:
        query = query.select(columns)
    if limit is not None:
        query = query.limit(limit)
    return await query.to_list()


async def _fetch_rows_by_ids(
    table,
    ids: list[str],
    columns: list[str] | None = None,
    id_column: str = "id",
) -> list[dict[str, Any]]:
    if not ids:
        return []
    rows: list[dict[str, Any]] = []
    for chunk in _iter_chunks(ids):
        rows.extend(await _fetch_rows(table, where=_sql_in(id_column, chunk), columns=columns))
    return rows


@final
@dataclass
class LanceDBKVStorage(BaseKVStorage):
    """Key-value storage: JSON payload per id, workspace-prefixed table."""

    def __init__(
        self,
        namespace: str,
        global_config: dict[str, Any],
        embedding_func=None,
        workspace: str | None = None,
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        validate_workspace(self.workspace)
        self.workspace, self.final_namespace, self._table_name = _build_table_name(
            self.workspace, self.namespace
        )
        self._client: LanceDBClient | None = None

    async def initialize(self):
        async with get_data_init_lock():
            if self._client is None:
                self._client = await ClientManager.get_client(
                    _resolve_lancedb_uri(self.global_config)
                )
            await self._client.get_or_create_table(self._table_name, _kv_schema())

    async def finalize(self):
        if self._client is not None:
            await ClientManager.release_client(self._client)
            self._client = None

    def _table(self):
        if self._client is None:
            raise StorageNotInitializedError(type(self).__name__)
        return self._client.table(self._table_name)

    @staticmethod
    def _format_record(record_id: str, payload: str | None) -> dict[str, Any]:
        record = _json_loads(payload)
        record.setdefault("create_time", 0)
        record.setdefault("update_time", 0)
        record["_id"] = record_id
        return record

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await _fetch_rows(
            self._table(), where=f"id = {_sql_quote(id)}", columns=["id", "payload"], limit=1
        )
        if not rows:
            return None
        return self._format_record(id, rows[0].get("payload"))

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows = await _fetch_rows_by_ids(self._table(), ids, columns=["id", "payload"])
        by_id = {row["id"]: row.get("payload") for row in rows}
        return [
            self._format_record(rid, by_id[rid]) if rid in by_id else None
            for rid in ids
        ]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        rows = await _fetch_rows_by_ids(self._table(), list(keys), columns=["id"])
        return keys - {row["id"] for row in rows}

    async def get_all_keys(self) -> list[str]:
        """List every id in this namespace's table.

        BaseKVStorage has no enumeration contract; the offline maintenance
        tools (rebuild_vdb / clean_llm_query_cache / migrate_llm_cache) rely
        on this backend-specific scan. The table name is workspace-prefixed,
        so a plain scan already returns only this workspace's ids.
        """
        rows = await _fetch_rows(self._table(), columns=["id"])
        return [row["id"] for row in rows]

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        table = self._table()
        async with self._client.table_lock(self._table_name):
            existing_rows = await _fetch_rows_by_ids(
                table, list(data.keys()), columns=["id", "create_time"]
            )
            existing_create_times = {
                row["id"]: row.get("create_time") for row in existing_rows
            }
            current_time = int(time.time())
            rows: list[dict[str, Any]] = []
            for i, (key, value) in enumerate(data.items(), 1):
                record = dict(value)
                if self.namespace.endswith("text_chunks"):
                    record.setdefault("llm_cache_list", [])
                create_time = (
                    record.get("create_time")
                    or existing_create_times.get(key)
                    or current_time
                )
                record["create_time"] = create_time
                record["update_time"] = current_time
                record["_id"] = key
                rows.append(
                    {
                        "id": key,
                        "payload": _json_dumps(record),
                        "create_time": create_time,
                        "update_time": current_time,
                    }
                )
                await _cooperative_yield(i)
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
            self._client.bump_writes(self._table_name)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        table = self._table()
        try:
            async with self._client.table_lock(self._table_name):
                for chunk in _iter_chunks(list(ids)):
                    await table.delete(_sql_in("id", chunk))
                self._client.bump_writes(self._table_name)
        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to delete from {self._table_name}: {e}")
            raise

    async def is_empty(self) -> bool:
        return await self._table().count_rows() == 0

    async def index_done_callback(self) -> None:
        # Writes are committed immediately; just compact periodically.
        if self._client is not None:
            await self._client.maybe_optimize(self._table_name)

    async def drop(self) -> dict[str, str]:
        try:
            if self._client is None:
                raise StorageNotInitializedError(type(self).__name__)
            await self._client.drop_and_recreate_table(self._table_name, _kv_schema())
            logger.info(f"[{self.workspace}] Dropped KV table {self._table_name}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to drop {self._table_name}: {e}")
            return {"status": "error", "message": str(e)}


@dataclass
class _PendingLanceVectorDoc:
    """Buffered vector upsert awaiting embedding at flush time."""

    source: dict[str, Any]  # {"created_at": int, **meta_fields}
    content: str
    vector: list[float] | None = None


@final
@dataclass
class LanceDBVectorStorage(BaseVectorStorage):
    """Vector storage with deferred embedding and optional CJK-friendly FTS.

    ``upsert()`` buffers records without embedding them; the flush in
    ``index_done_callback()`` embeds all pending contents in batches of
    ``embedding_batch_num`` and writes them in one ``merge_insert``. Reads
    (``get_by_id``/``get_by_ids``/``get_vectors_by_ids``) see pending
    records; ``query()`` only sees flushed data.
    """

    def __init__(
        self,
        namespace: str,
        global_config: dict[str, Any],
        embedding_func=None,
        workspace: str | None = None,
        meta_fields: set[str] | None = None,
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
        validate_workspace(self.workspace)
        self._validate_embedding_func()
        self.workspace, self.final_namespace, base_table_name = _build_table_name(
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
        # Embedding-model suffix isolates tables across embedding models/dims,
        # since a Lance fixed-size vector column cannot change dimension.
        model_suffix = self._generate_collection_suffix()
        if model_suffix:
            self._table_name = _sanitize_table_name(f"{base_table_name}_{model_suffix}")
        else:
            self._table_name = base_table_name
        self._client: LanceDBClient | None = None
        self._buffer_lock = None
        self._pending_docs: dict[str, _PendingLanceVectorDoc] = {}
        fts_enabled_raw = _get_lancedb_env("LANCEDB_ENABLE_FTS", "true") or "true"
        self._fts_enabled = fts_enabled_raw.strip().lower() == "true"
        self._fts_tokenizer = (
            _get_lancedb_env("LANCEDB_FTS_TOKENIZER", "ngram") or "ngram"
        ).strip()

    async def initialize(self):
        async with get_data_init_lock():
            if self._buffer_lock is None:
                self._buffer_lock = get_namespace_lock(
                    self.namespace, workspace=self.workspace
                )
            if self._client is None:
                self._client = await ClientManager.get_client(
                    _resolve_lancedb_uri(self.global_config)
                )
            dim = self.embedding_func.embedding_dim
            table = await self._client.get_or_create_table(
                self._table_name, _vector_schema(dim)
            )
            await self._check_vector_dim(table, dim)
            await self._ensure_fts_index(table)

    async def finalize(self):
        flush_error: Exception | None = None
        try:
            if self._client is not None and self._pending_docs:
                async with self._buffer_lock:
                    await self._flush_pending_locked()
        except Exception as e:
            flush_error = e
        if self._client is not None:
            await ClientManager.release_client(self._client)
            self._client = None
        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] LanceDB vector flush failed during finalize: {flush_error}"
            ) from flush_error

    def _table(self):
        if self._client is None:
            raise StorageNotInitializedError(type(self).__name__)
        return self._client.table(self._table_name)

    async def _check_vector_dim(self, table, dim: int) -> None:
        schema = await table.schema()
        vector_field = schema.field("vector")
        stored_dim = getattr(vector_field.type, "list_size", None)
        if stored_dim is not None and stored_dim != dim:
            raise ValueError(
                f"Vector dimension mismatch for table {self._table_name}: "
                f"stored={stored_dim}, embedding_func={dim}. "
                "Clear the LanceDB directory or use a different table/model suffix."
            )

    def _fts_config(self) -> FTS:
        if self._fts_tokenizer == "ngram":
            # Bigram tokenization handles CJK text without external language
            # models while still matching Latin-script terms.
            return FTS(
                base_tokenizer="ngram",
                ngram_min_length=2,
                ngram_max_length=2,
                stem=False,
                remove_stop_words=False,
                ascii_folding=True,
                lower_case=True,
            )
        return FTS(base_tokenizer=self._fts_tokenizer)

    async def _ensure_fts_index(self, table) -> None:
        """Best-effort FTS index on content; vector contract works without it."""
        if not self._fts_enabled:
            return
        try:
            indices = await table.list_indices()
            for index in indices:
                columns = getattr(index, "columns", None) or []
                if "content" in columns:
                    return
            await table.create_index("content", config=self._fts_config())
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Failed to create FTS index on {self._table_name}: {e}"
            )

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> dict[str, Any]:
        record = _json_loads(row.get("payload"))
        record["content"] = row.get("content")
        if row.get("src_id") is not None:
            record["src_id"] = row["src_id"]
        if row.get("tgt_id") is not None:
            record["tgt_id"] = row["tgt_id"]
        record["id"] = row["id"]
        record["created_at"] = row.get("created_at")
        return record

    _NON_VECTOR_COLUMNS = ["id", "content", "src_id", "tgt_id", "created_at", "payload"]

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        if self._buffer_lock is None:
            raise StorageNotInitializedError(type(self).__name__)
        current_time = int(time.time())
        pending: dict[str, _PendingLanceVectorDoc] = {}
        for i, (key, value) in enumerate(data.items(), 1):
            content = value["content"]
            source = {
                "created_at": current_time,
                **{k: v for k, v in value.items() if k in self.meta_fields},
            }
            pending[key] = _PendingLanceVectorDoc(source=source, content=content)
            await _cooperative_yield(i)
        async with self._buffer_lock:
            self._pending_docs.update(pending)
        logger.debug(
            f"[{self.workspace}] Buffered {len(pending)} vector docs for {self.final_namespace}"
        )

    async def _embed_contents(self, contents: list[str]) -> np.ndarray:
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch, context="document") for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) != len(contents):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(contents)}, got {len(embeddings)}"
            )
        return embeddings

    async def _flush_pending_locked(self) -> None:
        """Embed and persist pending docs; caller holds ``_buffer_lock``.

        On failure the pending buffer is kept intact so the next
        ``index_done_callback`` retries, and the error propagates so the
        pipeline can abort instead of marking documents processed.
        """
        if not self._pending_docs:
            return
        snapshot = list(self._pending_docs.items())
        to_embed = [(doc_id, doc) for doc_id, doc in snapshot if doc.vector is None]
        if to_embed:
            embeddings = await self._embed_contents([doc.content for _, doc in to_embed])
            for (_, doc), embedding in zip(to_embed, embeddings):
                doc.vector = embedding.astype(np.float32).tolist()
        rows: list[dict[str, Any]] = []
        for i, (doc_id, doc) in enumerate(snapshot, 1):
            meta = dict(doc.source)
            created_at = meta.pop("created_at", None)
            content = meta.pop("content", doc.content)
            src_id = meta.pop("src_id", None)
            tgt_id = meta.pop("tgt_id", None)
            rows.append(
                {
                    "id": doc_id,
                    "vector": doc.vector,
                    "content": content,
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "created_at": created_at,
                    "payload": _json_dumps(meta),
                }
            )
            await _cooperative_yield(i)
        table = self._table()
        async with self._client.table_lock(self._table_name):
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
            self._client.bump_writes(self._table_name)
        # Only drop the exact entries we flushed; a re-upsert that landed
        # while embedding was in flight must survive for the next flush.
        for doc_id, doc in snapshot:
            if self._pending_docs.get(doc_id) is doc:
                self._pending_docs.pop(doc_id, None)
        logger.info(
            f"[{self.workspace}] Flushed {len(rows)} vector docs to {self._table_name}"
        )

    async def index_done_callback(self) -> None:
        if self._buffer_lock is None:
            raise StorageNotInitializedError(type(self).__name__)
        async with self._buffer_lock:
            await self._flush_pending_locked()
        await self._client.maybe_optimize(self._table_name)

    async def drop_pending_index_ops(self) -> None:
        if self._buffer_lock is None:
            return
        async with self._buffer_lock:
            self._pending_docs.clear()

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            query_vector = (
                query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else list(query_embedding)
            )
        else:
            embeddings = await self.embedding_func(
                [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
            )
            query_vector = embeddings[0].tolist()
        table = self._table()
        rows = await (
            table.query()
            .nearest_to(query_vector)
            .distance_type("cosine")
            .limit(top_k)
            .select(self._NON_VECTOR_COLUMNS + ["_distance"])
            .to_list()
        )
        results = []
        for row in rows:
            # LanceDB returns cosine distance; LightRAG thresholds similarity.
            similarity = 1.0 - row["_distance"]
            if similarity < self.cosine_better_than_threshold:
                continue
            record = self._row_to_record(row)
            record["distance"] = similarity
            results.append(record)
        return results

    async def full_text_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """BM25 full-text search over the content column (extra capability).

        Not part of the LightRAG storage contract; requires the FTS index
        (enabled by default, CJK-friendly bigram tokenizer).
        """
        table = self._table()
        rows = await (
            table.query()
            .nearest_to_text(query)
            .limit(top_k)
            .select(self._NON_VECTOR_COLUMNS + ["_score"])
            .to_list()
        )
        results = []
        for row in rows:
            record = self._row_to_record(row)
            record["score"] = row.get("_score")
            results.append(record)
        return results

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._buffer_lock:
            pending = self._pending_docs.get(id)
            if pending is not None:
                return {**pending.source, "id": id}
        rows = await _fetch_rows(
            self._table(),
            where=f"id = {_sql_quote(id)}",
            columns=self._NON_VECTOR_COLUMNS,
            limit=1,
        )
        if not rows:
            return None
        return self._row_to_record(rows[0])

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        found: dict[str, dict[str, Any]] = {}
        remaining: list[str] = []
        async with self._buffer_lock:
            for record_id in ids:
                pending = self._pending_docs.get(record_id)
                if pending is not None:
                    found[record_id] = {**pending.source, "id": record_id}
                else:
                    remaining.append(record_id)
        rows = await _fetch_rows_by_ids(
            self._table(), remaining, columns=self._NON_VECTOR_COLUMNS
        )
        for row in rows:
            found[row["id"]] = self._row_to_record(row)
        return [found.get(record_id) for record_id in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        async with self._buffer_lock:
            pending_to_embed = [
                (record_id, doc)
                for record_id in ids
                if (doc := self._pending_docs.get(record_id)) is not None
                and doc.vector is None
            ]
            if pending_to_embed:
                embeddings = await self._embed_contents(
                    [doc.content for _, doc in pending_to_embed]
                )
                for (_, doc), embedding in zip(pending_to_embed, embeddings):
                    # Cache on the pending doc so the flush reuses it.
                    doc.vector = embedding.astype(np.float32).tolist()
            for record_id in ids:
                doc = self._pending_docs.get(record_id)
                if doc is not None and doc.vector is not None:
                    result[record_id] = doc.vector
                else:
                    remaining.append(record_id)
        rows = await _fetch_rows_by_ids(self._table(), remaining, columns=["id", "vector"])
        for row in rows:
            vector = row.get("vector")
            if vector is not None:
                result[row["id"]] = list(vector)
        return result

    async def delete(self, ids: list[str]):
        if not ids:
            return
        table = self._table()
        try:
            async with self._buffer_lock:
                for record_id in ids:
                    self._pending_docs.pop(record_id, None)
            async with self._client.table_lock(self._table_name):
                for chunk in _iter_chunks(list(ids)):
                    await table.delete(_sql_in("id", chunk))
                self._client.bump_writes(self._table_name)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to delete {len(ids)} vectors from {self._table_name}: {e}"
            )
            raise

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        quoted = _sql_quote(entity_name)
        table = self._table()
        try:
            async with self._client.table_lock(self._table_name):
                await table.delete(f"src_id = {quoted} OR tgt_id = {quoted}")
                self._client.bump_writes(self._table_name)
            async with self._buffer_lock:
                stale = [
                    doc_id
                    for doc_id, doc in self._pending_docs.items()
                    if doc.source.get("src_id") == entity_name
                    or doc.source.get("tgt_id") == entity_name
                ]
                for doc_id in stale:
                    self._pending_docs.pop(doc_id, None)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to delete relations of {entity_name}: {e}"
            )
            raise

    async def drop(self) -> dict[str, str]:
        try:
            if self._client is None:
                raise StorageNotInitializedError(type(self).__name__)
            if self._buffer_lock is not None:
                async with self._buffer_lock:
                    self._pending_docs.clear()
            table = await self._client.drop_and_recreate_table(
                self._table_name, _vector_schema(self.embedding_func.embedding_dim)
            )
            await self._ensure_fts_index(table)
            logger.info(f"[{self.workspace}] Dropped vector table {self._table_name}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to drop {self._table_name}: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class LanceDBGraphStorage(BaseGraphStorage):
    """Graph storage: nodes and edges tables with canonical undirected edges.

    Each undirected edge is stored once under a canonical id derived from
    the sorted endpoint pair; the ``src``/``tgt`` columns keep the latest
    call direction. Node/edge upserts merge properties into the existing
    payload (NetworkX/Mongo semantics).
    """

    def __init__(
        self,
        namespace: str,
        global_config: dict[str, Any],
        embedding_func=None,
        workspace: str | None = None,
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        validate_workspace(self.workspace)
        self.workspace, self.final_namespace, base_table_name = _build_table_name(
            self.workspace, self.namespace
        )
        self._nodes_table_name = f"{base_table_name}_nodes"
        self._edges_table_name = f"{base_table_name}_edges"
        self._client: LanceDBClient | None = None

    async def initialize(self):
        async with get_data_init_lock():
            if self._client is None:
                self._client = await ClientManager.get_client(
                    _resolve_lancedb_uri(self.global_config)
                )
            await self._client.get_or_create_table(
                self._nodes_table_name, _graph_node_schema()
            )
            await self._client.get_or_create_table(
                self._edges_table_name, _graph_edge_schema()
            )

    async def finalize(self):
        if self._client is not None:
            await ClientManager.release_client(self._client)
            self._client = None

    def _nodes_table(self):
        if self._client is None:
            raise StorageNotInitializedError(type(self).__name__)
        return self._client.table(self._nodes_table_name)

    def _edges_table(self):
        if self._client is None:
            raise StorageNotInitializedError(type(self).__name__)
        return self._client.table(self._edges_table_name)

    async def has_node(self, node_id: str) -> bool:
        return await self._nodes_table().count_rows(f"id = {_sql_quote(node_id)}") > 0

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        edge_id = _canonical_edge_id(source_node_id, target_node_id)
        return await self._edges_table().count_rows(f"id = {_sql_quote(edge_id)}") > 0

    async def node_degree(self, node_id: str) -> int:
        quoted = _sql_quote(node_id)
        return await self._edges_table().count_rows(
            f"src = {quoted} OR tgt = {quoted}"
        )

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree, tgt_degree = await asyncio.gather(
            self.node_degree(src_id), self.node_degree(tgt_id)
        )
        return src_degree + tgt_degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        rows = await _fetch_rows(
            self._nodes_table(), where=f"id = {_sql_quote(node_id)}", limit=1
        )
        if not rows:
            return None
        return _json_loads(rows[0].get("payload"))

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        edge_id = _canonical_edge_id(source_node_id, target_node_id)
        rows = await _fetch_rows(
            self._edges_table(), where=f"id = {_sql_quote(edge_id)}", limit=1
        )
        if not rows:
            return None
        return _json_loads(rows[0].get("payload"))

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not await self.has_node(source_node_id):
            return None
        quoted = _sql_quote(source_node_id)
        rows = await _fetch_rows(
            self._edges_table(),
            where=f"src = {quoted} OR tgt = {quoted}",
            columns=["src", "tgt"],
        )
        # Queried node first in every tuple (NetworkX semantics) — callers
        # like amerge_entities filter on tuple[0] == entity_name.
        return [
            (source_node_id, row["tgt"] if row["src"] == source_node_id else row["src"])
            for row in rows
        ]

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        rows = await _fetch_rows_by_ids(self._nodes_table(), node_ids)
        return {row["id"]: _json_loads(row.get("payload")) for row in rows}

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        rows = await _fetch_rows_by_ids(self._nodes_table(), node_ids, columns=["id"])
        return {row["id"] for row in rows}

    async def _edges_touching(self, node_ids: list[str]) -> list[dict[str, Any]]:
        """All edge rows with any endpoint in ``node_ids`` (deduped by id)."""
        edges: dict[str, dict[str, Any]] = {}
        table = self._edges_table()
        for chunk in _iter_chunks(node_ids):
            rows = await _fetch_rows(
                table, where=f"{_sql_in('src', chunk)} OR {_sql_in('tgt', chunk)}"
            )
            for row in rows:
                edges[row["id"]] = row
        return list(edges.values())

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        result = {node_id: 0 for node_id in node_ids}
        requested = set(node_ids)
        for row in await self._edges_touching(node_ids):
            if row["src"] in requested:
                result[row["src"]] += 1
            # A self-loop counts once, matching count_rows on src OR tgt.
            if row["tgt"] in requested and row["tgt"] != row["src"]:
                result[row["tgt"]] += 1
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        node_ids = list({node for pair in edge_pairs for node in pair})
        degrees = await self.node_degrees_batch(node_ids)
        return {
            (src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}
        id_to_pairs: dict[str, list[tuple[str, str]]] = {}
        for pair in pairs:
            edge_id = _canonical_edge_id(pair["src"], pair["tgt"])
            id_to_pairs.setdefault(edge_id, []).append((pair["src"], pair["tgt"]))
        rows = await _fetch_rows_by_ids(self._edges_table(), list(id_to_pairs.keys()))
        result: dict[tuple[str, str], dict] = {}
        for row in rows:
            payload = _json_loads(row.get("payload"))
            for requested_pair in id_to_pairs.get(row["id"], []):
                result[requested_pair] = dict(payload)
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        result: dict[str, list[tuple[str, str]]] = {node_id: [] for node_id in node_ids}
        requested = set(node_ids)
        for row in await self._edges_touching(node_ids):
            # Queried node first in each tuple, matching get_node_edges.
            if row["src"] in requested:
                result[row["src"]].append((row["src"], row["tgt"]))
            if row["tgt"] in requested and row["tgt"] != row["src"]:
                result[row["tgt"]].append((row["tgt"], row["src"]))
        return result

    async def _merge_upsert_nodes_locked(
        self, nodes: list[tuple[str, dict[str, str]]]
    ) -> None:
        """Merge node properties into existing payloads; caller holds lock."""
        table = self._nodes_table()
        node_ids = list({node_id for node_id, _ in nodes})
        rows = await _fetch_rows_by_ids(table, node_ids)
        merged: dict[str, dict[str, Any]] = {
            row["id"]: _json_loads(row.get("payload")) for row in rows
        }
        for i, (node_id, node_data) in enumerate(nodes, 1):
            current = merged.setdefault(node_id, {})
            current.update(node_data)
            await _cooperative_yield(i)
        await (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(
                [
                    {"id": node_id, "payload": _json_dumps(payload)}
                    for node_id, payload in merged.items()
                ]
            )
        )
        self._client.bump_writes(self._nodes_table_name)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        async with self._client.table_lock(self._nodes_table_name):
            await self._merge_upsert_nodes_locked([(node_id, node_data)])

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        if not nodes:
            return
        async with self._client.table_lock(self._nodes_table_name):
            await self._merge_upsert_nodes_locked(nodes)

    async def _ensure_source_nodes(self, source_ids: list[str]) -> None:
        existing = await self.has_nodes_batch(source_ids)
        missing = [node_id for node_id in source_ids if node_id not in existing]
        if missing:
            await self.upsert_nodes_batch([(node_id, {}) for node_id in missing])

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        await self.upsert_edges_batch([(source_node_id, target_node_id, edge_data)])

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        if not edges:
            return
        await self._ensure_source_nodes(list({src for src, _, _ in edges}))
        # Deduplicate by canonical id within the batch: apply in call order so
        # later writes win field-by-field, matching sequential upsert_edge.
        deduped: dict[str, tuple[str, str, dict[str, str]]] = {}
        for src, tgt, edge_data in edges:
            edge_id = _canonical_edge_id(src, tgt)
            if edge_id in deduped:
                _, _, merged_data = deduped[edge_id]
                merged_data = {**merged_data, **edge_data}
            else:
                merged_data = dict(edge_data)
            deduped[edge_id] = (src, tgt, merged_data)
        table = self._edges_table()
        async with self._client.table_lock(self._edges_table_name):
            rows = await _fetch_rows_by_ids(table, list(deduped.keys()))
            existing = {row["id"]: _json_loads(row.get("payload")) for row in rows}
            out_rows = []
            for i, (edge_id, (src, tgt, edge_data)) in enumerate(deduped.items(), 1):
                payload = existing.get(edge_id, {})
                payload.update(edge_data)
                payload["source_node_id"] = src
                payload["target_node_id"] = tgt
                out_rows.append(
                    {
                        "id": edge_id,
                        "src": src,
                        "tgt": tgt,
                        "payload": _json_dumps(payload),
                    }
                )
                await _cooperative_yield(i)
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(out_rows)
            )
            self._client.bump_writes(self._edges_table_name)

    async def delete_node(self, node_id: str) -> None:
        await self.remove_nodes([node_id])

    async def remove_nodes(self, nodes: list[str]):
        if not nodes:
            return
        edges_table = self._edges_table()
        nodes_table = self._nodes_table()
        async with self._client.table_lock(self._edges_table_name):
            for chunk in _iter_chunks(nodes):
                await edges_table.delete(
                    f"{_sql_in('src', chunk)} OR {_sql_in('tgt', chunk)}"
                )
            self._client.bump_writes(self._edges_table_name)
        async with self._client.table_lock(self._nodes_table_name):
            for chunk in _iter_chunks(nodes):
                await nodes_table.delete(_sql_in("id", chunk))
            self._client.bump_writes(self._nodes_table_name)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        if not edges:
            return
        edge_ids = list({_canonical_edge_id(src, tgt) for src, tgt in edges})
        table = self._edges_table()
        async with self._client.table_lock(self._edges_table_name):
            for chunk in _iter_chunks(edge_ids):
                await table.delete(_sql_in("id", chunk))
            self._client.bump_writes(self._edges_table_name)

    async def get_all_labels(self) -> list[str]:
        rows = await _fetch_rows(self._nodes_table(), columns=["id"])
        return sorted(row["id"] for row in rows)

    async def get_all_nodes(self) -> list[dict]:
        rows = await _fetch_rows(self._nodes_table())
        return [
            {**_json_loads(row.get("payload")), "id": row["id"]} for row in rows
        ]

    async def get_all_edges(self) -> list[dict]:
        rows = await _fetch_rows(self._edges_table())
        return [
            {
                **_json_loads(row.get("payload")),
                "source": row["src"],
                "target": row["tgt"],
            }
            for row in rows
        ]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        node_rows = await _fetch_rows(self._nodes_table(), columns=["id"])
        degrees = {row["id"]: 0 for row in node_rows}
        edge_rows = await _fetch_rows(self._edges_table(), columns=["src", "tgt"])
        for row in edge_rows:
            if row["src"] in degrees:
                degrees[row["src"]] += 1
            if row["tgt"] in degrees and row["tgt"] != row["src"]:
                degrees[row["tgt"]] += 1
        ranked = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))
        return [label for label, _ in ranked[:limit]]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        rows = await _fetch_rows(self._nodes_table(), columns=["id"])
        matches: list[tuple[str, int]] = []
        for row in rows:
            label = str(row["id"])
            label_lower = label.lower()
            if query_lower not in label_lower:
                continue
            if label_lower == query_lower:
                score = 1000
            elif label_lower.startswith(query_lower):
                score = 500
            else:
                score = 100 - len(label)
                if f" {query_lower}" in label_lower or f"_{query_lower}" in label_lower:
                    score += 50
            matches.append((label, score))
        matches.sort(key=lambda item: (-item[1], item[0]))
        return [label for label, _ in matches[:limit]]

    @staticmethod
    def _construct_graph_node(node_id: str, node_data: dict[str, Any]) -> KnowledgeGraphNode:
        return KnowledgeGraphNode(
            id=node_id, labels=[node_id], properties=dict(node_data)
        )

    @staticmethod
    def _construct_graph_edge(row: dict[str, Any]) -> KnowledgeGraphEdge:
        payload = _json_loads(row.get("payload"))
        return KnowledgeGraphEdge(
            id=f"{row['src']}-{row['tgt']}",
            type=payload.get("relationship", ""),
            source=row["src"],
            target=row["tgt"],
            properties={
                k: v
                for k, v in payload.items()
                if k not in ("source_node_id", "target_node_id", "relationship")
            },
        )

    async def _append_edges_between(
        self, node_ids: set[str], result: KnowledgeGraph
    ) -> None:
        seen_edges: set[str] = set()
        table = self._edges_table()
        for chunk in _iter_chunks(list(node_ids)):
            rows = await _fetch_rows(table, where=_sql_in("src", chunk))
            for row in rows:
                if row["tgt"] not in node_ids or row["id"] in seen_edges:
                    continue
                seen_edges.add(row["id"])
                result.edges.append(self._construct_graph_edge(row))

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        global_max = self.global_config.get("max_graph_nodes", 1000)
        max_nodes = global_max if max_nodes is None else min(max_nodes, global_max)
        if node_label == "*":
            return await self._get_knowledge_graph_all(max_nodes)
        return await self._get_knowledge_graph_bfs(node_label, max_depth, max_nodes)

    async def _get_knowledge_graph_all(self, max_nodes: int) -> KnowledgeGraph:
        result = KnowledgeGraph()
        total = await self._nodes_table().count_rows()
        if total > max_nodes:
            result.is_truncated = True
            selected = set(await self.get_popular_labels(limit=max_nodes))
            nodes = await self.get_nodes_batch(list(selected))
        else:
            rows = await _fetch_rows(self._nodes_table())
            nodes = {row["id"]: _json_loads(row.get("payload")) for row in rows}
            selected = set(nodes.keys())
        for node_id, node_data in nodes.items():
            result.nodes.append(self._construct_graph_node(node_id, node_data))
        await self._append_edges_between(selected, result)
        return result

    async def _get_knowledge_graph_bfs(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        result = KnowledgeGraph()
        start_node = await self.get_node(node_label)
        if start_node is None:
            logger.warning(f"[{self.workspace}] Graph node not found: {node_label}")
            return result
        result.nodes.append(self._construct_graph_node(node_label, start_node))
        seen: set[str] = {node_label}
        result_ids: set[str] = {node_label}
        frontier: list[str] = [node_label]
        truncated = False
        for _ in range(max_depth):
            if not frontier:
                break
            neighbor_ids: list[str] = []
            for row in await self._edges_touching(frontier):
                for endpoint in (row["src"], row["tgt"]):
                    if endpoint in seen:
                        continue
                    if len(seen) >= max_nodes:
                        # A reachable neighbor exists but the cap keeps it out.
                        truncated = True
                        break
                    seen.add(endpoint)
                    neighbor_ids.append(endpoint)
                if truncated:
                    break
            if truncated or not neighbor_ids:
                break
            fetched = await self.get_nodes_batch(neighbor_ids)
            next_frontier = []
            for node_id in neighbor_ids:
                node_data = fetched.get(node_id)
                if node_data is None:
                    continue  # edge endpoint without node row — skip
                result.nodes.append(self._construct_graph_node(node_id, node_data))
                result_ids.add(node_id)
                next_frontier.append(node_id)
            frontier = next_frontier
        await self._append_edges_between(result_ids, result)
        result.is_truncated = truncated
        return result

    async def index_done_callback(self) -> None:
        if self._client is not None:
            await self._client.maybe_optimize(self._nodes_table_name)
            await self._client.maybe_optimize(self._edges_table_name)

    async def drop(self) -> dict[str, str]:
        try:
            if self._client is None:
                raise StorageNotInitializedError(type(self).__name__)
            await self._client.drop_and_recreate_table(
                self._nodes_table_name, _graph_node_schema()
            )
            await self._client.drop_and_recreate_table(
                self._edges_table_name, _graph_edge_schema()
            )
            logger.info(f"[{self.workspace}] Dropped graph tables {self.final_namespace}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to drop graph tables: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class LanceDBDocStatusStorage(DocStatusStorage):
    """Document status storage with typed columns for status/dedup lookups."""

    def __init__(
        self,
        namespace: str,
        global_config: dict[str, Any],
        embedding_func=None,
        workspace: str | None = None,
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        validate_workspace(self.workspace)
        self.workspace, self.final_namespace, self._table_name = _build_table_name(
            self.workspace, self.namespace
        )
        self._client: LanceDBClient | None = None

    async def initialize(self):
        async with get_data_init_lock():
            if self._client is None:
                self._client = await ClientManager.get_client(
                    _resolve_lancedb_uri(self.global_config)
                )
            await self._client.get_or_create_table(
                self._table_name, _doc_status_schema()
            )

    async def finalize(self):
        if self._client is not None:
            await ClientManager.release_client(self._client)
            self._client = None

    def _table(self):
        if self._client is None:
            raise StorageNotInitializedError(type(self).__name__)
        return self._client.table(self._table_name)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await _fetch_rows(
            self._table(),
            where=f"id = {_sql_quote(id)}",
            columns=["id", "payload"],
            limit=1,
        )
        if not rows:
            return None
        return _json_loads(rows[0].get("payload"))

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows = await _fetch_rows_by_ids(self._table(), ids, columns=["id", "payload"])
        by_id = {row["id"]: _json_loads(row.get("payload")) for row in rows}
        return [by_id.get(record_id) for record_id in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        rows = await _fetch_rows_by_ids(self._table(), list(keys), columns=["id"])
        return keys - {row["id"] for row in rows}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        table = self._table()
        rows: list[dict[str, Any]] = []
        for i, (key, value) in enumerate(data.items(), 1):
            record = dict(value)
            record.setdefault("chunks_list", [])
            rows.append(
                {
                    "id": key,
                    "status": _status_value(record.get("status")),
                    # Normalize like _prepare_doc_status_data's read path so the
                    # typed column agrees with the payload's normalized file_path.
                    "file_path": record.get("file_path") or "no-file-path",
                    "track_id": record.get("track_id"),
                    "content_hash": record.get("content_hash"),
                    "payload": _json_dumps(record),
                }
            )
            await _cooperative_yield(i)
        async with self._client.table_lock(self._table_name):
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
            self._client.bump_writes(self._table_name)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        table = self._table()
        try:
            async with self._client.table_lock(self._table_name):
                for chunk in _iter_chunks(list(ids)):
                    await table.delete(_sql_in("id", chunk))
                self._client.bump_writes(self._table_name)
        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to delete doc status rows: {e}")
            raise

    async def is_empty(self) -> bool:
        return await self._table().count_rows() == 0

    @staticmethod
    def _prepare_doc_status_data(doc: dict[str, Any]) -> dict[str, Any]:
        data = doc.copy()
        data.pop("content", None)  # deprecated field
        if not data.get("file_path"):
            data["file_path"] = "no-file-path"
        data.setdefault("metadata", {})
        data.setdefault("error_msg", None)
        if "error" in data:  # legacy field rename
            if not data.get("error_msg"):
                data["error_msg"] = data.pop("error")
            else:
                data.pop("error", None)
        return data

    def _build_doc_status(
        self, doc_id: str, payload: str | None
    ) -> DocProcessingStatus | None:
        try:
            data = self._prepare_doc_status_data(_json_loads(payload))
            return DocProcessingStatus(**data)
        except (KeyError, TypeError) as e:
            logger.error(f"[{self.workspace}] Invalid doc status record {doc_id}: {e}")
            return None

    def _rows_to_statuses(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, DocProcessingStatus]:
        result: dict[str, DocProcessingStatus] = {}
        for row in rows:
            status = self._build_doc_status(row["id"], row.get("payload"))
            if status is not None:
                result[row["id"]] = status
        return result

    async def get_status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in DocStatus}
        rows = await _fetch_rows(self._table(), columns=["status"])
        for row in rows:
            status = row.get("status")
            if status is not None:
                counts[status] = counts.get(status, 0) + 1
        return counts

    async def get_all_status_counts(self) -> dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        if not statuses:
            return {}
        where = _sql_in("status", [_status_value(status) for status in statuses])
        rows = await _fetch_rows(self._table(), where=where, columns=["id", "payload"])
        return self._rows_to_statuses(rows)

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        rows = await _fetch_rows(
            self._table(),
            where=f"track_id = {_sql_quote(track_id)}",
            columns=["id", "payload"],
        )
        return self._rows_to_statuses(rows)

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        page = max(1, page)
        page_size = max(10, min(200, page_size))
        if sort_field not in ("created_at", "updated_at", "id", "file_path"):
            sort_field = "updated_at"
        reverse = sort_direction.lower() != "asc"
        status_values = self.resolve_status_filter_values(
            status_filter=status_filter, status_filters=status_filters
        )
        where = _sql_in("status", sorted(status_values)) if status_values else None
        rows = await _fetch_rows(self._table(), where=where, columns=["id", "payload"])
        docs = list(self._rows_to_statuses(rows).items())

        def sort_key(item: tuple[str, DocProcessingStatus]):
            doc_id, status = item
            if sort_field == "id":
                return doc_id
            if sort_field == "file_path":
                return get_pinyin_sort_key(getattr(status, sort_field, "") or "")
            return getattr(status, sort_field, "") or ""

        docs.sort(key=sort_key, reverse=reverse)
        total_count = len(docs)
        start = (page - 1) * page_size
        return docs[start : start + page_size], total_count

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        rows = await _fetch_rows(
            self._table(),
            where=f"file_path = {_sql_quote(file_path)}",
            columns=["id", "payload"],
            limit=1,
        )
        if not rows:
            return None
        return _json_loads(rows[0].get("payload"))

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        if not basename or basename == "unknown_source":
            return None
        rows = await _fetch_rows(
            self._table(),
            where=f"file_path = {_sql_quote(basename)}",
            columns=["id", "payload"],
            limit=1,
        )
        if not rows:
            return None
        return rows[0]["id"], _json_loads(rows[0].get("payload"))

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        if not content_hash:
            return None
        rows = await _fetch_rows(
            self._table(),
            where=f"content_hash = {_sql_quote(content_hash)}",
            columns=["id", "payload"],
            limit=1,
        )
        if not rows:
            return None
        return rows[0]["id"], _json_loads(rows[0].get("payload"))

    async def index_done_callback(self) -> None:
        if self._client is not None:
            await self._client.maybe_optimize(self._table_name)

    async def drop(self) -> dict[str, str]:
        try:
            if self._client is None:
                raise StorageNotInitializedError(type(self).__name__)
            await self._client.drop_and_recreate_table(
                self._table_name, _doc_status_schema()
            )
            logger.info(f"[{self.workspace}] Dropped doc status table {self._table_name}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to drop {self._table_name}: {e}")
            return {"status": "error", "message": str(e)}
