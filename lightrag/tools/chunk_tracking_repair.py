#!/usr/bin/env python3
"""Offline chunk-tracking repair tool.

The tool removes orphan tracking keys by replacing ``entity_chunks`` and/or
``relation_chunks``. It retains authoritative rows for objects that still exist
in the graph and supplements them from cached extraction results. This is
deliberately offline-only: a standalone process cannot share the API server's
in-memory pipeline reservations, and dropping a whole namespace while any
server, worker, or SDK writer is active can silently lose concurrent updates.

Run without ``--apply`` to build and print a read-only repair plan. Before an
apply, stop every writer that uses the same backing storages and workspace.
The complete plan is computed before the first drop, so a source read failure
leaves tracking untouched. Before apply, the plan is durably sealed on local
disk. An interrupted apply is resumed from that pre-drop snapshot instead of
re-reading a partially rebuilt namespace.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lightrag.base import CURSOR_END, CURSOR_START, DocStatus
from lightrag.constants import RELATION_NO_EVIDENCE_SOURCE_IDS
from lightrag.utils import (
    EmbeddingFunc,
    has_chunk_tracking_row,
    logger,
    make_relation_chunk_key,
    setup_logger,
)

_CHUNK_SCAN_BATCH = 200
_UPSERT_BATCH = 500
_PLAN_SCHEMA_VERSION = 1
_BOLD_YELLOW = "\033[1;33m"
_RESET = "\033[0m"
_CONNECTION_ID_ENV_KEYS = (
    "MEMGRAPH_URI",
    "MONGO_DATABASE",
    "MONGO_URI",
    "NEO4J_DATABASE",
    "NEO4J_URI",
    "OPENSEARCH_HOSTS",
    "POSTGRES_DATABASE",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "REDIS_URI",
)


class _DiskPlan:
    """SQLite-backed object universe and replacement rows.

    SQLite performs uniqueness and joins on disk, so Python retains at most one
    scan/upsert batch plus the largest individual tracking row. Apply plans can
    outlive the process and are removed only after successful replay.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        create: bool = True,
        remove_on_close: bool = True,
    ):
        if path is None:
            fd, generated = tempfile.mkstemp(
                prefix="lightrag-chunk-tracking-", suffix=".sqlite3"
            )
            os.close(fd)
            self.path = Path(generated)
        else:
            self.path = Path(path).expanduser().resolve()
            if create:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            elif not self.path.is_file():
                raise FileNotFoundError(f"Repair plan does not exist: {self.path}")
        self._remove_on_close = remove_on_close
        self.connection = sqlite3.connect(self.path)
        self._closed = False
        self.connection.execute("PRAGMA synchronous=FULL")
        if create:
            self.connection.executescript(
                """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE objects (
                namespace TEXT NOT NULL,
                object_key TEXT NOT NULL,
                PRIMARY KEY (namespace, object_key)
            ) WITHOUT ROWID;
            CREATE TABLE rows (
                namespace TEXT NOT NULL,
                object_key TEXT NOT NULL,
                PRIMARY KEY (namespace, object_key)
            ) WITHOUT ROWID;
            CREATE TABLE attribution (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                object_key TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                UNIQUE (namespace, object_key, chunk_id)
            );
            CREATE INDEX attribution_row
                ON attribution(namespace, object_key, sequence);
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
            )
        else:
            integrity = self.connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ValueError(f"Repair plan failed integrity check: {integrity}")

    def close(self, *, remove: bool | None = None) -> None:
        if self._closed:
            return
        self.connection.close()
        self._closed = True
        should_remove = self._remove_on_close if remove is None else remove
        if should_remove:
            for candidate in (self.path, Path(f"{self.path}-journal")):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError as exc:
                    # Cleanup is not part of the tracking commit. A completed
                    # repair must not be reported as failed merely because its
                    # already-unneeded local recovery file could not be removed.
                    logger.warning(
                        f"Unable to remove chunk-tracking repair plan {candidate}: {exc}"
                    )

    def __del__(self):
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            pass

    def set_metadata(self, **values: str) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)", values.items()
        )

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def sync(self) -> None:
        """Commit the plan and force its file and directory entry to disk."""
        self.connection.commit()
        file_fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        if os.name != "nt":
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def retain_after_close(self) -> None:
        self._remove_on_close = False

    def remove_after_close(self) -> None:
        self._remove_on_close = True

    def add_objects(self, namespace: str, keys) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO objects VALUES (?, ?)",
            ((namespace, key) for key in keys),
        )

    def add_chunks(self, chunk_ids) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO chunks VALUES (?)",
            ((chunk_id,) for chunk_id in chunk_ids if chunk_id),
        )

    def object_exists(self, namespace: str, key: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM objects WHERE namespace = ? AND object_key = ?",
                (namespace, key),
            ).fetchone()
            is not None
        )

    def add_row(self, namespace: str, key: str, chunk_ids=()) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO rows VALUES (?, ?)", (namespace, key)
        )
        self.connection.executemany(
            """INSERT OR IGNORE INTO attribution(namespace, object_key, chunk_id)
               VALUES (?, ?, ?)""",
            ((namespace, key, chunk_id) for chunk_id in chunk_ids),
        )

    def count(self, table: str, namespace: str | None = None) -> int:
        if table not in {"objects", "rows", "chunks"}:
            raise ValueError(f"Unsupported plan table: {table}")
        if namespace is None:
            row = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        else:
            row = self.connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE namespace = ?", (namespace,)
            ).fetchone()
        return int(row[0])

    def iter_keys(self, table: str, namespace: str, batch_size: int):
        after = ""
        while True:
            rows = self.connection.execute(
                f"""SELECT object_key FROM {table}
                    WHERE namespace = ? AND object_key > ?
                    ORDER BY object_key LIMIT ?""",
                (namespace, after, batch_size),
            ).fetchall()
            if not rows:
                return
            batch = [row[0] for row in rows]
            yield batch
            after = batch[-1]

    def iter_chunks(self, batch_size: int):
        after = ""
        while True:
            rows = self.connection.execute(
                "SELECT chunk_id FROM chunks WHERE chunk_id > ? ORDER BY chunk_id LIMIT ?",
                (after, batch_size),
            ).fetchall()
            if not rows:
                return
            batch = [row[0] for row in rows]
            yield batch
            after = batch[-1]

    def row_payload(self, namespace: str, keys: list[str]) -> dict[str, dict]:
        payload: dict[str, dict] = {}
        for key in keys:
            chunk_ids = [
                row[0]
                for row in self.connection.execute(
                    """SELECT chunk_id FROM attribution
                       WHERE namespace = ? AND object_key = ? ORDER BY sequence""",
                    (namespace, key),
                )
            ]
            payload[key] = {"chunk_ids": chunk_ids, "count": len(chunk_ids)}
        return payload


@dataclass
class ChunkTrackingRepairReport:
    """Counts and warnings for a repair plan or completed apply."""

    scanned_documents: int = 0
    documents_with_chunks: int = 0
    scanned_chunks: int = 0
    chunks_with_cache: int = 0
    chunks_with_attribution: int = 0
    chunks_without_cache: int = 0
    graph_entities: int = 0
    graph_relations: int = 0
    existing_entity_rows: int = 0
    existing_relation_rows: int = 0
    malformed_entity_rows: int = 0
    malformed_relation_rows: int = 0
    entity_rows_planned: int = 0
    relation_rows_planned: int = 0
    entity_rows_written: int = 0
    relation_rows_written: int = 0
    entities_without_tracking_row: int = 0
    relations_without_tracking_row: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkTrackingRepairPlan:
    """Stable replacement payload computed before any destructive mutation."""

    report: ChunkTrackingRepairReport
    disk: _DiskPlan = field(repr=False)
    completed: bool = field(default=False, repr=False)

    @property
    def unsafe_empty_namespaces(self) -> list[str]:
        """Namespaces a server restart would repopulate from graph source_id."""
        unsafe: list[str] = []
        if self.report.graph_entities and not self.report.entity_rows_planned:
            unsafe.append("entity_chunks")
        if self.report.graph_relations and not self.report.relation_rows_planned:
            unsafe.append("relation_chunks")
        return unsafe

    def blockers(
        self,
        namespaces: set[str],
        *,
        allow_empty_graph: bool = False,
        allow_missing_rows: bool = False,
    ) -> list[str]:
        """Return reasons an apply must be refused before its first drop."""
        blockers = [
            namespace
            for namespace in self.unsafe_empty_namespaces
            if namespace in namespaces
        ]
        if (
            not allow_empty_graph
            and not self.report.graph_entities
            and not self.report.graph_relations
        ):
            blockers.append("graph is empty or unavailable")
        if not allow_missing_rows:
            if (
                "entity_chunks" in namespaces
                and self.report.entities_without_tracking_row
            ):
                blockers.append(
                    f"{self.report.entities_without_tracking_row} graph entity/entities "
                    "have no planned tracking row"
                )
            if (
                "relation_chunks" in namespaces
                and self.report.relations_without_tracking_row
            ):
                blockers.append(
                    f"{self.report.relations_without_tracking_row} graph relation(s) "
                    "have no planned tracking row"
                )
        return blockers

    def close(self, *, remove: bool | None = None) -> None:
        self.disk.close(remove=remove)

    @property
    def path(self) -> Path:
        return self.disk.path

    @property
    def state(self) -> str:
        if self.completed:
            return "complete"
        return self.disk.get_metadata("state") or "building"

    def seal(self, rag) -> None:
        """Persist a complete pre-drop snapshot and its storage identity."""
        self.disk.set_metadata(
            schema_version=str(_PLAN_SCHEMA_VERSION),
            state="ready",
            storage_identity=json.dumps(
                _repair_storage_identity(rag), sort_keys=True, separators=(",", ":")
            ),
            report=json.dumps(
                self.report.as_dict(), sort_keys=True, separators=(",", ":")
            ),
        )
        self.disk.sync()

    def prepare_apply(
        self,
        namespaces: set[str],
        *,
        allow_empty_graph: bool,
        allow_missing_rows: bool,
    ) -> None:
        """Durably record the exact destructive operation before its drop."""
        options = {
            "namespaces": sorted(namespaces),
            "allow_empty_graph": allow_empty_graph,
            "allow_missing_rows": allow_missing_rows,
        }
        encoded = json.dumps(options, sort_keys=True, separators=(",", ":"))
        prior = self.disk.get_metadata("apply_options")
        if self.state == "applying" and prior != encoded:
            raise ValueError("Resume options do not match the sealed repair plan")
        # From this point forward a failure may leave tracking partial, so the
        # sealed snapshot must survive normal cleanup and object destruction.
        self.disk.retain_after_close()
        self.disk.set_metadata(state="applying", apply_options=encoded)
        self.disk.sync()
        logger.info(f"Durable chunk-tracking recovery plan: {self.path}")

    def resume_options(self) -> tuple[set[str], bool, bool]:
        if self.state != "applying":
            raise ValueError(
                "Repair plan has no interrupted apply to resume; build a new plan"
            )
        raw = self.disk.get_metadata("apply_options")
        if raw is None:
            raise ValueError("Repair plan is missing its apply options")
        options = json.loads(raw)
        return (
            set(options["namespaces"]),
            bool(options["allow_empty_graph"]),
            bool(options["allow_missing_rows"]),
        )

    def mark_complete(self) -> None:
        # The tracking namespaces are already committed at this point. This is
        # local cleanup state only and must not turn a successful repair into a
        # reported failure because of an unrelated plan-file metadata write.
        self.completed = True
        self.disk.remove_after_close()


def _repair_storage_identity(rag) -> dict[str, Any]:
    def configured_value(
        module: Any,
        env_key: str,
        section: str,
        option: str,
        fallback: str | None,
    ) -> str | None:
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
        backend_config = getattr(module, "config", None)
        if backend_config is None:
            return fallback
        return backend_config.get(section, option, fallback=fallback)

    def connection_target(storage) -> str | None:
        """Hash the initialized backend's effective non-secret target."""
        module_name = type(storage).__module__
        module = sys.modules.get(module_name)
        values: dict[str, Any] | None = None
        if module_name.endswith(".redis_impl"):
            values = {"uri": getattr(storage, "_redis_url", None)}
        elif module_name.endswith(".mongo_impl") and module is not None:
            database = getattr(getattr(storage, "db", None), "name", None)
            values = {
                "uri": configured_value(
                    module,
                    "MONGO_URI",
                    "mongodb",
                    "uri",
                    "mongodb://root:root@localhost:27017/",
                ),
                "database": database
                or configured_value(
                    module,
                    "MONGO_DATABASE",
                    "mongodb",
                    "database",
                    "LightRAG",
                ),
            }
        elif module_name.endswith(".neo4j_impl") and module is not None:
            values = {
                "uri": configured_value(module, "NEO4J_URI", "neo4j", "uri", None),
                "database": getattr(storage, "_DATABASE", None),
            }
        elif module_name.endswith(".memgraph_impl") and module is not None:
            values = {
                "uri": configured_value(
                    module,
                    "MEMGRAPH_URI",
                    "memgraph",
                    "uri",
                    "bolt://localhost:7687",
                ),
                "database": getattr(storage, "_DATABASE", None),
            }
        elif module_name.endswith(".opensearch_impl") and module is not None:
            values = {
                "hosts": configured_value(
                    module,
                    "OPENSEARCH_HOSTS",
                    "opensearch",
                    "hosts",
                    "localhost:9200",
                ),
                "use_ssl": configured_value(
                    module,
                    "OPENSEARCH_USE_SSL",
                    "opensearch",
                    "use_ssl",
                    "true",
                ),
            }
        elif module_name.endswith((".postgres_impl", ".pgtable_impl")):
            database = getattr(storage, "db", None)
            if database is not None:
                values = {
                    "host": getattr(database, "host", None),
                    "port": getattr(database, "port", None),
                    "database": getattr(database, "database", None),
                }
        if values is None:
            return None
        return hashlib.sha256(
            json.dumps(values, sort_keys=True, default=str).encode()
        ).hexdigest()

    def identify(storage) -> dict[str, str]:
        identity = {
            "class": f"{type(storage).__module__}.{type(storage).__qualname__}",
            "workspace": str(getattr(storage, "workspace", "")),
            "namespace": str(getattr(storage, "namespace", "")),
        }
        # Some backends derive the actual storage target without updating the
        # public workspace/namespace fields. Redis, for example, lets
        # REDIS_WORKSPACE override workspace and uses final_namespace for every
        # key operation. Bind a recovery plan to that effective target so a
        # changed override cannot redirect resume into another workspace.
        if hasattr(storage, "final_namespace"):
            identity["final_namespace"] = str(storage.final_namespace)
        target_fingerprint = connection_target(storage)
        if target_fingerprint is not None:
            identity["connection_target_fingerprint"] = target_fingerprint
        return identity

    connection_values = {key: os.environ.get(key) for key in _CONNECTION_ID_ENV_KEYS}
    connection_fingerprint = hashlib.sha256(
        json.dumps(connection_values, sort_keys=True).encode()
    ).hexdigest()
    return {
        "working_dir": str(Path(rag.working_dir).expanduser().resolve()),
        "workspace": str(rag.workspace or ""),
        "graph": identify(rag.chunk_entity_relation_graph),
        "entity_chunks": identify(rag.entity_chunks),
        "relation_chunks": identify(rag.relation_chunks),
        "connection_fingerprint": connection_fingerprint,
    }


def load_chunk_tracking_repair_plan(
    rag, path: str | os.PathLike[str]
) -> ChunkTrackingRepairPlan:
    """Open and validate a durable plan without reading damaged tracking."""
    _require_storages(rag)
    disk = _DiskPlan(path, create=False, remove_on_close=False)
    try:
        version = disk.get_metadata("schema_version")
        if version != str(_PLAN_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported repair plan schema version: {version or 'missing'}"
            )
        identity = disk.get_metadata("storage_identity")
        expected = json.dumps(
            _repair_storage_identity(rag), sort_keys=True, separators=(",", ":")
        )
        if identity != expected:
            raise ValueError(
                "Repair plan storage identity does not match the configured "
                "working directory, workspace, or storage namespaces"
            )
        raw_report = disk.get_metadata("report")
        if raw_report is None:
            raise ValueError("Repair plan is incomplete and has no sealed report")
        report = ChunkTrackingRepairReport(**json.loads(raw_report))
        plan = ChunkTrackingRepairPlan(report, disk)
        plan.resume_options()
        return plan
    except BaseException:
        disk.close(remove=False)
        raise


async def build_chunk_tracking_repair_plan(
    rag,
    *,
    plan_path: str | os.PathLike[str] | None = None,
    durable: bool = False,
) -> ChunkTrackingRepairPlan:
    """Build a complete, read-only replacement plan for one workspace.

    The graph is an existence oracle only. Existing authoritative rows are
    retained because rename, merge, and explicit creation may produce keys or
    attribution the extraction cache cannot reproduce. Cached extraction adds
    chunk-granular attribution; graph ``source_id`` and document-level anchors
    are never seeds.
    """
    _require_storages(rag)
    report = ChunkTrackingRepairReport()
    disk = _DiskPlan(
        plan_path,
        remove_on_close=not durable,
    )
    try:
        (
            report.scanned_documents,
            report.documents_with_chunks,
        ) = await _collect_corpus_chunk_ids(rag, disk)
        report.scanned_chunks = disk.count("chunks")

        await _collect_graph_objects(rag, disk)
        report.graph_entities = disk.count("objects", "entity_chunks")
        report.graph_relations = disk.count("objects", "relation_chunks")

        (
            report.existing_entity_rows,
            report.malformed_entity_rows,
        ) = await _collect_existing_rows(
            rag.entity_chunks,
            disk,
            "entity_chunks",
            excluded_ids=RELATION_NO_EVIDENCE_SOURCE_IDS,
        )
        (
            report.existing_relation_rows,
            report.malformed_relation_rows,
        ) = await _collect_existing_rows(
            rag.relation_chunks,
            disk,
            "relation_chunks",
            excluded_ids=RELATION_NO_EVIDENCE_SOURCE_IDS,
        )
        if report.scanned_chunks and rag.llm_response_cache is not None:
            (
                report.chunks_with_cache,
                report.chunks_with_attribution,
                report.chunks_without_cache,
            ) = await _accumulate_cached_attribution(rag, disk)
        else:
            report.chunks_without_cache = report.scanned_chunks
            if rag.llm_response_cache is None:
                report.warnings.append(
                    "LLM response cache is not configured, so no chunk-level "
                    "attribution can be recovered."
                )

        report.entity_rows_planned = disk.count("rows", "entity_chunks")
        report.relation_rows_planned = disk.count("rows", "relation_chunks")
        report.entities_without_tracking_row = (
            report.graph_entities - report.entity_rows_planned
        )
        report.relations_without_tracking_row = (
            report.graph_relations - report.relation_rows_planned
        )
    except BaseException:
        disk.close(remove=True)
        raise

    _add_plan_warnings(report)
    plan = ChunkTrackingRepairPlan(report, disk)
    if plan.unsafe_empty_namespaces:
        report.warnings.append(
            f"Refusing apply while {', '.join(plan.unsafe_empty_namespaces)} would "
            "be empty although the graph contains corresponding objects. A server "
            "restart would re-seed the empty namespace from KEEP-truncated graph "
            "source_id; restore the extraction cache or re-ingest first."
        )
    if not report.graph_entities and not report.graph_relations:
        report.warnings.append(
            "The graph is empty or unavailable. Apply is blocked by default because "
            "the tool cannot distinguish a legitimately empty graph from a wrong "
            "backend/workspace or a backend whose graph index is unavailable."
        )

    for warning in report.warnings:
        logger.warning(f"Chunk tracking repair plan: {warning}")
    # Seal only after the complete report is final, but still before apply can
    # record its options or drop a namespace.
    plan.seal(rag)
    return plan


async def apply_chunk_tracking_repair_plan(
    rag,
    plan: ChunkTrackingRepairPlan,
    *,
    namespaces: set[str] | None = None,
    allow_empty_graph: bool = False,
    allow_missing_rows: bool = False,
) -> ChunkTrackingRepairReport:
    """Apply a precomputed plan, failing before drop for unsafe empty output."""
    _require_storages(rag)
    namespaces = namespaces or {"entity_chunks", "relation_chunks"}
    invalid = namespaces - {"entity_chunks", "relation_chunks"}
    if invalid:
        raise ValueError(f"Unknown tracking namespace(s): {', '.join(sorted(invalid))}")
    blockers = plan.blockers(
        namespaces,
        allow_empty_graph=allow_empty_graph,
        allow_missing_rows=allow_missing_rows,
    )
    if blockers:
        raise ValueError(
            f"Refusing to drop tracking: {', '.join(blockers)}. Review the plan, "
            "correct the backend/cache configuration, narrow --namespace, or use "
            "an explicit allow override only after reviewing the reported loss."
        )

    plan.prepare_apply(
        namespaces,
        allow_empty_graph=allow_empty_graph,
        allow_missing_rows=allow_missing_rows,
    )

    if "entity_chunks" in namespaces:
        await _drop_tracking_namespace(rag.entity_chunks, "entity_chunks")
        plan.report.entity_rows_written = await _write_tracking_rows(
            rag.entity_chunks, plan.disk, "entity_chunks"
        )
    if "relation_chunks" in namespaces:
        await _drop_tracking_namespace(rag.relation_chunks, "relation_chunks")
        plan.report.relation_rows_written = await _write_tracking_rows(
            rag.relation_chunks, plan.disk, "relation_chunks"
        )
    plan.mark_complete()
    logger.info(
        "Chunk tracking repair completed: "
        f"{plan.report.entity_rows_written} entity row(s), "
        f"{plan.report.relation_rows_written} relation row(s) retained/rebuilt; "
        f"cache available for {plan.report.chunks_with_cache}/"
        f"{plan.report.scanned_chunks} chunk(s)"
    )
    return plan.report


def _require_storages(rag) -> None:
    required = (
        "doc_status",
        "text_chunks",
        "chunk_entity_relation_graph",
        "entity_chunks",
        "relation_chunks",
    )
    missing = [name for name in required if getattr(rag, name, None) is None]
    if missing:
        raise ValueError(
            f"Chunk tracking repair requires configured storages: {', '.join(missing)}"
        )


async def _collect_corpus_chunk_ids(rag, disk: _DiskPlan) -> tuple[int, int]:
    position = CURSOR_START
    scanned = 0
    contributing_docs = 0
    while position is not CURSOR_END:
        page = await rag.doc_status.get_docs_by_statuses_page(
            list(DocStatus),
            limit=_CHUNK_SCAN_BATCH,
            position=position,
            strict=True,
        )
        doc_ids = list(page.docs)
        docs = await rag.doc_status.get_full_docs_by_ids(doc_ids, strict=True)
        if len(docs) != len(doc_ids):
            missing = set(doc_ids) - set(docs)
            raise RuntimeError(
                "Doc-status rows disappeared during the offline scan: "
                + ", ".join(sorted(missing))
            )
        scanned += len(doc_ids)
        for doc_id in doc_ids:
            chunks_list = getattr(docs[doc_id], "chunks_list", None) or []
            if chunks_list:
                contributing_docs += 1
                disk.add_chunks(chunks_list)
        position = page.next_position
    return scanned, contributing_docs


async def _collect_existing_rows(
    storage,
    disk: _DiskPlan,
    namespace: str,
    *,
    excluded_ids: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    """Retain authoritative rows for objects that still exist in the graph.

    Rename, merge, and explicit creation can intentionally produce tracking keys
    or attribution that the extraction cache cannot reproduce. Existing rows are
    therefore the authority for current graph objects; rebuilding from cache only
    would turn an orphan sweep into provenance loss.
    """
    retained = 0
    malformed = 0
    for batch in disk.iter_keys("objects", namespace, _CHUNK_SCAN_BATCH):
        stored_rows = await storage.get_by_ids(batch)
        if len(stored_rows) != len(batch):
            raise RuntimeError(
                f"{type(storage).__name__}.get_by_ids returned {len(stored_rows)} "
                f"row(s) for {len(batch)} requested tracking key(s)"
            )
        for key, stored in zip(batch, stored_rows):
            if stored is None:
                continue
            if not has_chunk_tracking_row(stored):
                malformed += 1
                continue
            disk.add_row(
                namespace,
                key,
                dict.fromkeys(
                    chunk_id
                    for chunk_id in stored["chunk_ids"]
                    if chunk_id and chunk_id not in excluded_ids
                ),
            )
            retained += 1
    return retained, malformed


async def _collect_graph_objects(rag, disk: _DiskPlan) -> None:
    graph = rag.chunk_entity_relation_graph
    async for labels in graph.iter_labels(_CHUNK_SCAN_BATCH):
        disk.add_objects("entity_chunks", labels)
    async for edges in graph.iter_edges(_CHUNK_SCAN_BATCH):
        keys = []
        for edge in edges:
            src = edge.get("source") or edge.get("src_id") or edge.get("src")
            tgt = edge.get("target") or edge.get("tgt_id") or edge.get("tgt")
            if src and tgt:
                keys.append(make_relation_chunk_key(src, tgt))
        disk.add_objects("relation_chunks", keys)


async def _accumulate_cached_attribution(
    rag,
    disk: _DiskPlan,
) -> tuple[int, int, int]:
    from lightrag.operate import (
        _get_cached_extraction_results,
        _rebuild_from_extraction_result,
    )

    with_cache = 0
    with_attribution = 0
    for batch in disk.iter_chunks(_CHUNK_SCAN_BATCH):
        missing = await rag.text_chunks.filter_keys(set(batch))
        batch = [chunk_id for chunk_id in batch if chunk_id not in missing]
        if not batch:
            continue
        cached_results, chunk_data_by_id = await _get_cached_extraction_results(
            rag.llm_response_cache,
            set(batch),
            text_chunks_storage=rag.text_chunks,
        )
        for chunk_id, results in cached_results.items():
            parsed_any = False
            attributed_any = False
            for extraction_result, timestamp in results:
                try:
                    entities, relationships = await _rebuild_from_extraction_result(
                        text_chunks_storage=rag.text_chunks,
                        chunk_id=chunk_id,
                        extraction_result=extraction_result,
                        timestamp=timestamp,
                        chunk_data=chunk_data_by_id.get(chunk_id),
                    )
                except Exception as exc:
                    logger.warning(
                        "Chunk tracking repair: unusable cached extraction "
                        f"result for chunk {chunk_id}: {exc}"
                    )
                    continue
                parsed_any = True
                if entities or relationships:
                    attributed_any = True
                for entity_name in entities:
                    if disk.object_exists("entity_chunks", entity_name):
                        disk.add_row("entity_chunks", entity_name, [chunk_id])
                for src, tgt in relationships:
                    storage_key = make_relation_chunk_key(src, tgt)
                    if disk.object_exists("relation_chunks", storage_key):
                        disk.add_row("relation_chunks", storage_key, [chunk_id])
            if parsed_any:
                with_cache += 1
            if attributed_any:
                with_attribution += 1
    return with_cache, with_attribution, disk.count("chunks") - with_cache


def _add_plan_warnings(report: ChunkTrackingRepairReport) -> None:
    if report.chunks_without_cache:
        report.warnings.append(
            f"{report.chunks_without_cache} of {report.scanned_chunks} chunk(s) "
            "had no usable cached extraction result; those chunks cannot add "
            "attribution beyond any authoritative row already retained."
        )
    cached_without_attribution = (
        report.chunks_with_cache - report.chunks_with_attribution
    )
    if cached_without_attribution:
        report.warnings.append(
            f"{cached_without_attribution} cached chunk(s) parsed successfully but "
            "contained no entity or relation extraction records."
        )
    if report.entities_without_tracking_row or report.relations_without_tracking_row:
        report.warnings.append(
            f"{report.entities_without_tracking_row} entity/entities and "
            f"{report.relations_without_tracking_row} relation(s) present in the "
            "graph have neither a usable existing tracking row nor matching "
            "cached chunk-level attribution."
        )
    if report.malformed_entity_rows or report.malformed_relation_rows:
        report.warnings.append(
            f"Ignored {report.malformed_entity_rows} malformed entity tracking "
            f"row(s) and {report.malformed_relation_rows} malformed relation "
            "tracking row(s); only rows carrying a chunk_ids list are authoritative."
        )


async def _drop_tracking_namespace(storage, label: str) -> None:
    result = await storage.drop()
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError(f"Failed to drop {label} during repair: {result}")


def _require_no_pending_tracking_writes(storage, label: str) -> None:
    pending = {
        attr: len(value)
        for attr in ("_pending_upserts", "_pending_kv_deletes")
        if (value := getattr(storage, attr, None))
    }
    if pending:
        details = ", ".join(f"{name}={count}" for name, count in pending.items())
        raise RuntimeError(
            f"{label} flush left buffered operations ({details}); "
            "the durable repair plan was retained for resume"
        )


async def _write_tracking_rows(storage, disk: _DiskPlan, label: str) -> int:
    written = 0
    flush_buffer = getattr(storage, "_flush_pending_kv_ops", None)
    for keys in disk.iter_keys("rows", label, _UPSERT_BATCH):
        payload = disk.row_payload(label, keys)
        await storage.upsert(payload)
        # OpenSearch KV upsert is process-buffered. Flush its private buffer per
        # repair batch so backend memory stays bounded without forcing an index
        # refresh after every batch. A retryable bulk failure is retained in the
        # buffer and must fail this apply before the plan can be marked complete.
        if flush_buffer is not None:
            await flush_buffer()
            _require_no_pending_tracking_writes(storage, label)
        written += len(payload)
    await storage.index_done_callback()
    _require_no_pending_tracking_writes(storage, label)
    logger.info(f"Chunk tracking repair: rebuilt {written} {label} row(s)")
    return written


def _print_report(report: ChunkTrackingRepairReport, *, applied: bool) -> None:
    phase = "Apply result" if applied else "Repair plan"
    print(f"\n{phase}:")
    print(
        f"  Corpus: {report.scanned_documents} total document(s), "
        f"{report.documents_with_chunks} with chunks, "
        f"{report.scanned_chunks} distinct chunk(s)"
    )
    print(
        f"  Cache: {report.chunks_with_cache} usable chunk(s), "
        f"{report.chunks_with_attribution} with extracted objects, "
        f"{report.chunks_without_cache} unavailable or unusable"
    )
    print(
        f"  Graph: {report.graph_entities} entity/entities, "
        f"{report.graph_relations} relation(s)"
    )
    print(
        f"  Existing current-object rows: {report.existing_entity_rows}/"
        f"{report.graph_entities} entity, {report.existing_relation_rows}/"
        f"{report.graph_relations} relation"
    )
    print(
        f"  Planned rows: {report.entity_rows_planned}/{report.graph_entities} "
        f"entity, {report.relation_rows_planned}/{report.graph_relations} relation"
    )
    if applied:
        print(
            f"  Written rows: {report.entity_rows_written} entity, "
            f"{report.relation_rows_written} relation"
        )
    for warning in report.warnings:
        print(f"  WARNING: {warning}")


def _confirm_offline(assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(
        "Have all LightRAG servers, workers, and SDK writers for this workspace "
        "been stopped? (yes/no): "
    )
    return answer.strip().lower() == "yes"


def _confirm_apply(assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(
        "Type REPAIR to drop and rebuild entity_chunks and relation_chunks: "
    )
    return answer.strip() == "REPAIR"


def _storage_description(storage) -> str:
    details: list[str] = []
    workspace = getattr(storage, "workspace", None)
    namespace = getattr(storage, "namespace", None)
    if workspace is not None:
        details.append(f"workspace={workspace!r}")
    if namespace is not None:
        details.append(f"namespace={namespace!r}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{type(storage).__name__}{suffix}"


def _run_mode(args: argparse.Namespace) -> str:
    if not args.apply:
        return "DRY RUN ONLY"
    if getattr(args, "resume_plan", None):
        return "APPLY (RESUME)"
    return "APPLY"


def _print_run_status(message: str) -> None:
    print(f"{_BOLD_YELLOW}{message}{_RESET}")


async def _build_rag():
    from lightrag import LightRAG

    async def _noop_llm(*_args, **_kwargs) -> str:
        raise RuntimeError("chunk_tracking_repair never calls the LLM")

    async def _noop_embed(_texts):
        raise RuntimeError("chunk_tracking_repair never embeds")

    return LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        workspace=os.getenv("WORKSPACE", ""),
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        doc_status_storage=os.getenv(
            "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
        ),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=8192,
            func=_noop_embed,
            model_name=os.getenv("EMBEDDING_MODEL"),
        ),
    )


def _new_recovery_plan_path(rag) -> Path:
    directory = (
        Path(rag.working_dir).expanduser().resolve() / ".chunk_tracking_repair_plans"
    )
    return directory / f"plan-{uuid.uuid4().hex}.sqlite3"


async def run(args: argparse.Namespace) -> bool:
    """Run the offline tool; return False for any unsafe or failed apply."""
    print("LightRAG Offline Chunk-Tracking Repair Tool")
    _print_run_status(f"Run mode: {_run_mode(args)}")
    print("STOP every server, worker, and SDK writer for the target workspace.")
    if not _confirm_offline(args.yes):
        print("Operation cancelled. Stop all writers before running this tool.")
        return True

    rag = await _build_rag()
    await rag.initialize_storages()
    plan: ChunkTrackingRepairPlan | None = None
    try:
        print(f"Working directory: {rag.working_dir}")
        print(f"Requested workspace: {rag.workspace or '(default)'}")
        print(f"Graph storage: {_storage_description(rag.chunk_entity_relation_graph)}")
        print(f"Tracking KV storage: {_storage_description(rag.entity_chunks)}")
        print(f"Doc-status storage: {_storage_description(rag.doc_status)}")

        resume_path = getattr(args, "resume_plan", None)
        requested_namespace = getattr(args, "namespace", None)
        if resume_path:
            if not args.apply:
                raise ValueError("--resume-plan requires --apply")
            plan = load_chunk_tracking_repair_plan(rag, resume_path)
            namespaces, allow_empty_graph, allow_missing_rows = plan.resume_options()
            if requested_namespace is not None:
                requested_namespaces = (
                    {"entity_chunks", "relation_chunks"}
                    if requested_namespace == "both"
                    else {f"{requested_namespace}_chunks"}
                )
                if requested_namespaces != namespaces:
                    raise ValueError(
                        "--namespace does not match the interrupted repair plan"
                    )
            print(f"Resuming durable repair plan: {plan.path}")
        else:
            namespaces = (
                {"entity_chunks", "relation_chunks"}
                if requested_namespace in (None, "both")
                else {f"{requested_namespace}_chunks"}
            )
            allow_empty_graph = args.allow_empty_graph
            allow_missing_rows = args.allow_missing_rows
            requested_plan_path = getattr(args, "plan_file", None)
            durable = bool(args.apply)
            plan_path = (
                Path(requested_plan_path)
                if requested_plan_path
                else _new_recovery_plan_path(rag)
                if durable
                else None
            )
            plan = await build_chunk_tracking_repair_plan(
                rag,
                plan_path=plan_path,
                durable=durable,
            )
        _print_report(plan.report, applied=False)
        blockers = plan.blockers(
            namespaces,
            allow_empty_graph=allow_empty_graph,
            allow_missing_rows=allow_missing_rows,
        )
        if blockers:
            print(f"Unsafe plan: {', '.join(blockers)}.")
            print("No tracking namespace was modified.")
            return False
        if not args.apply:
            _print_run_status("Dry run only. Re-run with --apply to rebuild tracking.")
            return True
        if not resume_path:
            print(f"Durable recovery plan: {plan.path}")
        if not _confirm_apply(args.yes):
            print("Apply cancelled; tracking was not modified.")
            return True

        report = await apply_chunk_tracking_repair_plan(
            rag,
            plan,
            namespaces=namespaces,
            allow_empty_graph=allow_empty_graph,
            allow_missing_rows=allow_missing_rows,
        )
        _print_report(report, applied=True)
        print("Repair completed successfully. It is now safe to restart LightRAG.")
        return True
    except Exception as exc:
        if plan is not None and plan.completed:
            logger.exception(
                "Chunk tracking repair committed but final reporting failed"
            )
            print(f"Repair storage writes completed, but final reporting failed: {exc}")
            return True
        logger.exception("Chunk tracking repair failed")
        print(f"Repair failed: {exc}")
        if args.apply:
            if plan is not None and plan.state == "applying":
                print(
                    "Keep every LightRAG writer stopped. Resume the sealed plan "
                    f"with: lightrag-repair-chunk-tracking --apply --resume-plan {plan.path}"
                )
            else:
                print("No tracking namespace was modified.")
        return False
    finally:
        if plan is not None:
            plan.close(remove=plan.state != "applying")
        try:
            await rag.finalize_storages()
        except Exception as exc:
            if plan is None or not plan.completed:
                raise
            logger.exception(
                "Chunk tracking repair committed but storage finalization failed"
            )
            print(
                "Repair storage writes completed, but storage finalization failed: "
                f"{exc}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline repair of entity/relation chunk tracking."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Drop and rebuild tracking; default is a read-only repair plan.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm all writers are stopped and accept the destructive apply.",
    )
    parser.add_argument(
        "--namespace",
        choices=("both", "entity", "relation"),
        default=None,
        help="Tracking namespace to rebuild (default: both).",
    )
    plan_group = parser.add_mutually_exclusive_group()
    plan_group.add_argument(
        "--plan-file",
        help="Path for the new durable recovery plan used by --apply.",
    )
    plan_group.add_argument(
        "--resume-plan",
        help="Resume an interrupted --apply from this sealed plan file.",
    )
    parser.add_argument(
        "--allow-empty-graph",
        action="store_true",
        help=(
            "Allow clearing the selected tracking namespace when the graph is "
            "empty; use only after independently verifying backend/workspace."
        ),
    )
    parser.add_argument(
        "--allow-missing-rows",
        action="store_true",
        help=(
            "Allow a plan that leaves current graph objects without tracking "
            "rows; review the reported denominator first."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)
    setup_logger("lightrag", level="INFO")
    if not asyncio.run(run(_parse_args())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
