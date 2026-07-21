"""Database-backed workspace control plane and request runtime selection.

The LightRAG storage interfaces already accept a ``workspace`` value.  This
module keeps the API control plane in the deployment's configured database as
well: PostgreSQL deployments use PostgreSQL and MongoDB deployments use
MongoDB.  There is deliberately no SQLite fallback, because splitting a
production deployment's control-plane metadata from its configured data store
breaks backup, availability, and multi-replica guarantees.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Mapping
from uuid import UUID, uuid4

from fastapi import Header, HTTPException, Request

from lightrag.api.routers.document_routes import (
    DocumentManager,
    _acquire_destructive_busy,
    _release_destructive_busy,
)
from lightrag.utils import logger


DEFAULT_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"
WORKSPACE_HEADER = "X-LightRAG-Workspace-ID"

_STATIC_STORAGE_WORKSPACE_ENV_VARS = (
    "POSTGRES_WORKSPACE",
    "NEO4J_WORKSPACE",
    "MONGODB_WORKSPACE",
    "OPENSEARCH_WORKSPACE",
    "MILVUS_WORKSPACE",
    "QDRANT_WORKSPACE",
    "REDIS_WORKSPACE",
    "MEMGRAPH_WORKSPACE",
)


class WorkspaceError(RuntimeError):
    """Base error raised by workspace lifecycle operations."""


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when a public workspace id does not exist."""


class WorkspaceStateError(WorkspaceError):
    """Raised when an operation is not valid in the workspace's state."""


class WorkspaceBusyError(WorkspaceError):
    """Raised when a destructive workspace operation cannot get its lock."""


@dataclass(frozen=True)
class WorkspaceRecord:
    """Persistent workspace metadata.

    ``id`` is the public UUID accepted from the API header. ``storage_key`` is
    an immutable, storage-safe key passed into every LightRAG storage backend.
    The default record deliberately retains the pre-workspace storage key from
    the current deployment (usually the empty string) so existing data remains
    reachable without an implicit migration.
    """

    id: str
    storage_key: str
    display_name: str
    description: str | None
    status: str
    is_default: bool
    created_at: str
    updated_at: str
    deleted_at: str | None = None


@dataclass
class WorkspaceContext:
    """The request-scoped runtime objects for one workspace."""

    workspace: WorkspaceRecord
    rag: Any
    doc_manager: DocumentManager


class WorkspaceRegistry:
    """Storage-agnostic contract for persistent workspace metadata."""

    async def initialize(self, default_storage_key: str) -> WorkspaceRecord:
        raise NotImplementedError

    async def finalize(self) -> None:
        return None

    async def create(
        self, display_name: str, description: str | None = None
    ) -> WorkspaceRecord:
        raise NotImplementedError

    async def list(self, include_deleted: bool = False) -> list[WorkspaceRecord]:
        raise NotImplementedError

    async def get(
        self, workspace_id: str | UUID, *, include_deleted: bool = False
    ) -> WorkspaceRecord:
        raise NotImplementedError

    async def get_default(self) -> WorkspaceRecord:
        raise NotImplementedError

    async def update(
        self,
        workspace_id: str | UUID,
        *,
        display_name: str | None = None,
        description: str | None | object = None,
        description_provided: bool = False,
    ) -> WorkspaceRecord:
        raise NotImplementedError

    async def mark_deleting(self, workspace_id: str | UUID) -> WorkspaceRecord:
        raise NotImplementedError

    async def mark_deleted(self, workspace_id: str | UUID) -> WorkspaceRecord:
        raise NotImplementedError

    async def mark_failed(self, workspace_id: str | UUID) -> WorkspaceRecord:
        raise NotImplementedError

    async def create_job(
        self, workspace_id: str, job_type: str, status: str, detail: str | None = None
    ) -> str:
        raise NotImplementedError

    async def update_job(
        self, job_id: str, status: str, detail: str | None = None
    ) -> None:
        raise NotImplementedError


class _WorkspaceRegistryBase(WorkspaceRegistry):
    """Shared lifecycle rules, backed by database-specific primitive operations."""

    async def initialize(self, default_storage_key: str) -> WorkspaceRecord:
        await self._initialize_storage()
        record = await self._find_default_record()
        if record is None:
            candidate = WorkspaceRecord(
                id=DEFAULT_WORKSPACE_ID,
                storage_key=default_storage_key,
                display_name="Default workspace",
                description=None,
                status="active",
                is_default=True,
                created_at=_utc_now(),
                updated_at=_utc_now(),
            )
            try:
                record = await self._insert_workspace(candidate)
            except Exception:
                # A second API replica may have created the singleton first.
                record = await self._find_default_record()
                if record is None:
                    raise
        if record.storage_key != default_storage_key:
            raise WorkspaceStateError(
                "The persisted default workspace does not match the configured "
                "WORKSPACE value. Restore the previous WORKSPACE setting or "
                "migrate the default workspace explicitly."
            )
        return record

    async def create(
        self, display_name: str, description: str | None = None
    ) -> WorkspaceRecord:
        display_name = _validate_display_name(display_name)
        now = _utc_now()
        return await self._insert_workspace(
            WorkspaceRecord(
                id=str(uuid4()),
                storage_key=f"ws_{uuid4().hex}",
                display_name=display_name,
                description=_normalize_description(description),
                status="active",
                is_default=False,
                created_at=now,
                updated_at=now,
            )
        )

    async def get(
        self, workspace_id: str | UUID, *, include_deleted: bool = False
    ) -> WorkspaceRecord:
        workspace_id = _normalize_workspace_id(workspace_id)
        record = await self._find_workspace_record(workspace_id, include_deleted)
        if record is None:
            raise WorkspaceNotFoundError(f"Workspace '{workspace_id}' was not found")
        return record

    async def get_default(self) -> WorkspaceRecord:
        record = await self._find_default_record()
        if record is None:
            raise WorkspaceNotFoundError("The default workspace has not been initialized")
        return record

    async def update(
        self,
        workspace_id: str | UUID,
        *,
        display_name: str | None = None,
        description: str | None | object = None,
        description_provided: bool = False,
    ) -> WorkspaceRecord:
        workspace_id = _normalize_workspace_id(workspace_id)
        existing = await self.get(workspace_id)
        if existing.status != "active":
            raise WorkspaceStateError(
                f"Workspace '{workspace_id}' cannot be updated while status={existing.status}"
            )
        changes: dict[str, Any] = {}
        if display_name is not None:
            changes["display_name"] = _validate_display_name(display_name)
        if description_provided:
            changes["description"] = _normalize_description(
                description if isinstance(description, str) else None
            )
        if not changes:
            return existing
        changes["updated_at"] = _utc_now()
        updated = await self._update_workspace_record(workspace_id, changes)
        if updated is None:
            raise WorkspaceNotFoundError(f"Workspace '{workspace_id}' was not found")
        return updated

    async def mark_deleting(self, workspace_id: str | UUID) -> WorkspaceRecord:
        return await self._set_status(workspace_id, "deleting")

    async def mark_deleted(self, workspace_id: str | UUID) -> WorkspaceRecord:
        return await self._set_status(workspace_id, "deleted", deleted_at=_utc_now())

    async def mark_failed(self, workspace_id: str | UUID) -> WorkspaceRecord:
        return await self._set_status(workspace_id, "failed")

    async def _set_status(
        self, workspace_id: str | UUID, status: str, *, deleted_at: str | None = None
    ) -> WorkspaceRecord:
        workspace_id = _normalize_workspace_id(workspace_id)
        changes: dict[str, Any] = {"status": status, "updated_at": _utc_now()}
        if deleted_at is not None:
            changes["deleted_at"] = deleted_at
        updated = await self._update_workspace_record(workspace_id, changes)
        if updated is None:
            raise WorkspaceNotFoundError(f"Workspace '{workspace_id}' was not found")
        return updated

    async def create_job(
        self, workspace_id: str, job_type: str, status: str, detail: str | None = None
    ) -> str:
        workspace_id = _normalize_workspace_id(workspace_id)
        job_id = str(uuid4())
        now = _utc_now()
        await self._insert_job(
            {
                "id": job_id,
                "workspace_id": workspace_id,
                "job_type": job_type,
                "status": status,
                "detail": detail,
                "created_at": now,
                "updated_at": now,
            }
        )
        return job_id

    async def update_job(
        self, job_id: str, status: str, detail: str | None = None
    ) -> None:
        await self._update_job_record(
            str(UUID(job_id)),
            {"status": status, "detail": detail, "updated_at": _utc_now()},
        )

    async def _initialize_storage(self) -> None:
        raise NotImplementedError

    async def _insert_workspace(self, record: WorkspaceRecord) -> WorkspaceRecord:
        raise NotImplementedError

    async def _find_workspace_record(
        self, workspace_id: str, include_deleted: bool
    ) -> WorkspaceRecord | None:
        raise NotImplementedError

    async def _find_default_record(self) -> WorkspaceRecord | None:
        raise NotImplementedError

    async def _update_workspace_record(
        self, workspace_id: str, changes: Mapping[str, Any]
    ) -> WorkspaceRecord | None:
        raise NotImplementedError

    async def _insert_job(self, job: Mapping[str, Any]) -> None:
        raise NotImplementedError

    async def _update_job_record(self, job_id: str, changes: Mapping[str, Any]) -> None:
        raise NotImplementedError


class PostgresWorkspaceRegistry(_WorkspaceRegistryBase):
    """Workspace metadata in the configured PostgreSQL database."""

    _DDL = (
        """
        CREATE TABLE IF NOT EXISTS lightrag_workspaces (
            id UUID PRIMARY KEY,
            storage_key TEXT NOT NULL UNIQUE,
            display_name VARCHAR(255) NOT NULL,
            description TEXT NULL,
            status VARCHAR(16) NOT NULL,
            is_default BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            deleted_at TIMESTAMPTZ NULL,
            CHECK (status IN ('provisioning', 'active', 'deleting', 'deleted', 'failed'))
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_lightrag_workspaces_one_default
            ON lightrag_workspaces(is_default) WHERE is_default = TRUE
        """,
        """
        CREATE TABLE IF NOT EXISTS lightrag_workspace_jobs (
            id UUID PRIMARY KEY,
            workspace_id UUID NOT NULL REFERENCES lightrag_workspaces(id),
            job_type VARCHAR(64) NOT NULL,
            status VARCHAR(32) NOT NULL,
            detail TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_lightrag_workspace_jobs_workspace
            ON lightrag_workspace_jobs(workspace_id, created_at DESC)
        """,
    )

    def __init__(self, vector_storage: str):
        self.vector_storage = vector_storage
        self._db: Any | None = None

    async def _initialize_storage(self) -> None:
        if self._db is None:
            from lightrag.kg.postgres_impl import ClientManager

            self._db = await ClientManager.get_client(
                vector_storage=self.vector_storage
            )
        for statement in self._DDL:
            await self._db.execute(statement)

    async def finalize(self) -> None:
        if self._db is not None:
            from lightrag.kg.postgres_impl import ClientManager

            await ClientManager.release_client(self._db)
            self._db = None

    async def _insert_workspace(self, record: WorkspaceRecord) -> WorkspaceRecord:
        row = await self._query_one(
            """
            INSERT INTO lightrag_workspaces
                (id, storage_key, display_name, description, status, is_default,
                 created_at, updated_at, deleted_at)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::timestamptz, $8::timestamptz,
                    $9::timestamptz)
            RETURNING *
            """,
            [
                record.id,
                record.storage_key,
                record.display_name,
                record.description,
                record.status,
                record.is_default,
                _postgres_value("created_at", record.created_at),
                _postgres_value("updated_at", record.updated_at),
                _postgres_value("deleted_at", record.deleted_at),
            ],
        )
        assert row is not None
        return _record_from_mapping(row)

    async def list(self, include_deleted: bool = False) -> list[WorkspaceRecord]:
        where = "" if include_deleted else "WHERE status != 'deleted'"
        rows = await self._query_many(
            f"""
            SELECT * FROM lightrag_workspaces
            {where}
            ORDER BY is_default DESC, created_at ASC
            """
        )
        return [_record_from_mapping(row) for row in rows]

    async def _find_workspace_record(
        self, workspace_id: str, include_deleted: bool
    ) -> WorkspaceRecord | None:
        where = "" if include_deleted else "AND status != 'deleted'"
        row = await self._query_one(
            f"SELECT * FROM lightrag_workspaces WHERE id = $1::uuid {where}",
            [workspace_id],
        )
        return _record_from_mapping(row) if row else None

    async def _find_default_record(self) -> WorkspaceRecord | None:
        row = await self._query_one(
            "SELECT * FROM lightrag_workspaces WHERE is_default = TRUE"
        )
        return _record_from_mapping(row) if row else None

    async def _update_workspace_record(
        self, workspace_id: str, changes: Mapping[str, Any]
    ) -> WorkspaceRecord | None:
        columns = tuple(changes)
        assignments = ", ".join(f"{column} = ${index}" for index, column in enumerate(columns, 1))
        values = [_postgres_value(column, changes[column]) for column in columns]
        row = await self._query_one(
            f"""
            UPDATE lightrag_workspaces SET {assignments}
            WHERE id = ${len(values) + 1}::uuid
            RETURNING *
            """,
            [*values, workspace_id],
        )
        return _record_from_mapping(row) if row else None

    async def _insert_job(self, job: Mapping[str, Any]) -> None:
        await self._execute(
            """
            INSERT INTO lightrag_workspace_jobs
                (id, workspace_id, job_type, status, detail, created_at, updated_at)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6::timestamptz, $7::timestamptz)
            """,
            [
                job["id"],
                job["workspace_id"],
                job["job_type"],
                job["status"],
                job["detail"],
                _postgres_value("created_at", job["created_at"]),
                _postgres_value("updated_at", job["updated_at"]),
            ],
        )

    async def _update_job_record(self, job_id: str, changes: Mapping[str, Any]) -> None:
        columns = tuple(changes)
        assignments = ", ".join(f"{column} = ${index}" for index, column in enumerate(columns, 1))
        values = [_postgres_value(column, changes[column]) for column in columns]
        await self._execute(
            f"UPDATE lightrag_workspace_jobs SET {assignments} WHERE id = ${len(values) + 1}::uuid",
            [*values, job_id],
        )

    async def _query_one(
        self, sql: str, params: list[Any] | None = None
    ) -> Mapping[str, Any] | None:
        if self._db is None:
            raise WorkspaceStateError("Workspace registry is not initialized")
        return await self._db.query(sql, params)

    async def _query_many(
        self, sql: str, params: list[Any] | None = None
    ) -> list[Mapping[str, Any]]:
        if self._db is None:
            raise WorkspaceStateError("Workspace registry is not initialized")
        rows = await self._db.query(sql, params, multirows=True)
        return rows or []

    async def _execute(self, sql: str, params: list[Any] | None = None) -> None:
        if self._db is None:
            raise WorkspaceStateError("Workspace registry is not initialized")

        async def operation(connection: Any) -> None:
            await connection.execute(sql, *(params or ()))

        await self._db._run_with_retry(operation)


class MongoWorkspaceRegistry(_WorkspaceRegistryBase):
    """Workspace metadata in the configured MongoDB database."""

    def __init__(self):
        self._db: Any | None = None
        self._workspaces: Any | None = None
        self._jobs: Any | None = None

    async def _initialize_storage(self) -> None:
        if self._db is None:
            from lightrag.kg.mongo_impl import ClientManager

            self._db = await ClientManager.get_client()
            self._workspaces = self._db["lightrag_workspaces"]
            self._jobs = self._db["lightrag_workspace_jobs"]
            await self._workspaces.create_index("storage_key", unique=True)
            await self._workspaces.create_index(
                "is_default",
                unique=True,
                partialFilterExpression={"is_default": True},
                name="idx_lightrag_workspaces_one_default",
            )
            await self._jobs.create_index(
                [("workspace_id", 1), ("created_at", -1)],
                name="idx_lightrag_workspace_jobs_workspace",
            )

    async def finalize(self) -> None:
        if self._db is not None:
            from lightrag.kg.mongo_impl import ClientManager

            await ClientManager.release_client(self._db)
            self._db = self._workspaces = self._jobs = None

    async def _insert_workspace(self, record: WorkspaceRecord) -> WorkspaceRecord:
        collection = self._require_collection(self._workspaces)
        await collection.insert_one(_workspace_document(record))
        return record

    async def list(self, include_deleted: bool = False) -> list[WorkspaceRecord]:
        collection = self._require_collection(self._workspaces)
        query = {} if include_deleted else {"status": {"$ne": "deleted"}}
        rows = await collection.find(query).sort(
            [("is_default", -1), ("created_at", 1)]
        ).to_list(length=None)
        return [_record_from_mapping(row) for row in rows]

    async def _find_workspace_record(
        self, workspace_id: str, include_deleted: bool
    ) -> WorkspaceRecord | None:
        collection = self._require_collection(self._workspaces)
        query: dict[str, Any] = {"_id": workspace_id}
        if not include_deleted:
            query["status"] = {"$ne": "deleted"}
        row = await collection.find_one(query)
        return _record_from_mapping(row) if row else None

    async def _find_default_record(self) -> WorkspaceRecord | None:
        row = await self._require_collection(self._workspaces).find_one(
            {"is_default": True}
        )
        return _record_from_mapping(row) if row else None

    async def _update_workspace_record(
        self, workspace_id: str, changes: Mapping[str, Any]
    ) -> WorkspaceRecord | None:
        from pymongo import ReturnDocument

        row = await self._require_collection(self._workspaces).find_one_and_update(
            {"_id": workspace_id},
            {"$set": dict(changes)},
            return_document=ReturnDocument.AFTER,
        )
        return _record_from_mapping(row) if row else None

    async def _insert_job(self, job: Mapping[str, Any]) -> None:
        document = dict(job)
        document["_id"] = document.pop("id")
        await self._require_collection(self._jobs).insert_one(document)

    async def _update_job_record(self, job_id: str, changes: Mapping[str, Any]) -> None:
        await self._require_collection(self._jobs).update_one(
            {"_id": job_id}, {"$set": dict(changes)}
        )

    @staticmethod
    def _require_collection(collection: Any | None) -> Any:
        if collection is None:
            raise WorkspaceStateError("Workspace registry is not initialized")
        return collection


class InMemoryWorkspaceRegistry(_WorkspaceRegistryBase):
    """Test-only registry that never persists outside the current process."""

    def __init__(self):
        self._workspaces: dict[str, WorkspaceRecord] = {}
        self._jobs: dict[str, dict[str, Any]] = {}

    async def _initialize_storage(self) -> None:
        return None

    async def _insert_workspace(self, record: WorkspaceRecord) -> WorkspaceRecord:
        if record.id in self._workspaces or any(
            item.storage_key == record.storage_key for item in self._workspaces.values()
        ):
            raise WorkspaceStateError("Workspace already exists")
        if record.is_default and any(
            item.is_default for item in self._workspaces.values()
        ):
            raise WorkspaceStateError("Default workspace already exists")
        self._workspaces[record.id] = record
        return record

    async def list(self, include_deleted: bool = False) -> list[WorkspaceRecord]:
        records = [
            record
            for record in self._workspaces.values()
            if include_deleted or record.status != "deleted"
        ]
        return sorted(records, key=lambda record: (not record.is_default, record.created_at))

    async def _find_workspace_record(
        self, workspace_id: str, include_deleted: bool
    ) -> WorkspaceRecord | None:
        record = self._workspaces.get(workspace_id)
        if record and (include_deleted or record.status != "deleted"):
            return record
        return None

    async def _find_default_record(self) -> WorkspaceRecord | None:
        return next(
            (record for record in self._workspaces.values() if record.is_default), None
        )

    async def _update_workspace_record(
        self, workspace_id: str, changes: Mapping[str, Any]
    ) -> WorkspaceRecord | None:
        record = self._workspaces.get(workspace_id)
        if record is None:
            return None
        updated = WorkspaceRecord(
            id=record.id,
            storage_key=record.storage_key,
            display_name=changes.get("display_name", record.display_name),
            description=changes.get("description", record.description),
            status=changes.get("status", record.status),
            is_default=record.is_default,
            created_at=record.created_at,
            updated_at=changes.get("updated_at", record.updated_at),
            deleted_at=changes.get("deleted_at", record.deleted_at),
        )
        self._workspaces[workspace_id] = updated
        return updated

    async def _insert_job(self, job: Mapping[str, Any]) -> None:
        self._jobs[str(job["id"])] = dict(job)

    async def _update_job_record(self, job_id: str, changes: Mapping[str, Any]) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(changes)


def create_workspace_registry(
    *, kv_storage: str, vector_storage: str, doc_status_storage: str, graph_storage: str
) -> WorkspaceRegistry:
    """Select a registry in the same database family as configured RAG storage.

    ``LIGHTRAG_WORKSPACE_REGISTRY_BACKEND`` may explicitly select ``postgres``
    or ``mongodb``.  ``auto`` (the default) resolves from the configured
    storage classes.  There is intentionally no SQLite mode or file fallback.
    """

    configured = os.getenv("LIGHTRAG_WORKSPACE_REGISTRY_BACKEND", "auto").strip().lower()
    if configured in {"", "auto"}:
        storage_classes = {
            kv_storage,
            vector_storage,
            doc_status_storage,
            graph_storage,
        }
        if kv_storage == "PGKVStorage":
            configured = "postgres"
        elif kv_storage == "MongoKVStorage":
            configured = "mongodb"
        elif any(name.startswith("PG") for name in storage_classes):
            configured = "postgres"
        elif any(name.startswith("Mongo") for name in storage_classes):
            configured = "mongodb"
        else:
            raise WorkspaceStateError(
                "Multi-workspace mode requires a durable workspace registry in the "
                "configured database. Set LIGHTRAG_WORKSPACE_REGISTRY_BACKEND to "
                "postgres or mongodb and configure that database; SQLite is not supported."
            )
    if configured in {"postgres", "postgresql"}:
        return PostgresWorkspaceRegistry(vector_storage)
    if configured in {"mongodb", "mongo"}:
        return MongoWorkspaceRegistry()
    raise WorkspaceStateError(
        "LIGHTRAG_WORKSPACE_REGISTRY_BACKEND must be auto, postgres, or mongodb"
    )


class WorkspaceRuntimeManager:
    """Create, cache, bind, and delete isolated LightRAG runtime contexts."""

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        default_rag: Any,
        default_doc_manager: DocumentManager,
        rag_factory: Callable[[str], Any],
        input_dir: str | Path,
    ):
        self.registry = registry
        self.default_rag = default_rag
        self.default_doc_manager = default_doc_manager
        self.rag_factory = rag_factory
        self.input_dir = Path(input_dir)
        self.default_workspace: WorkspaceRecord | None = None
        self._contexts: dict[str, WorkspaceContext] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._current: contextvars.ContextVar[WorkspaceContext | None] = (
            contextvars.ContextVar("lightrag_workspace_context", default=None)
        )

    async def initialize(self) -> WorkspaceRecord:
        validate_dynamic_workspace_configuration()
        self.default_workspace = await self.registry.initialize(
            self.default_rag.workspace
        )
        self._contexts[self.default_workspace.id] = WorkspaceContext(
            workspace=self.default_workspace,
            rag=self.default_rag,
            doc_manager=self.default_doc_manager,
        )
        return self.default_workspace

    def current_context(self) -> WorkspaceContext:
        context = self._current.get()
        if context is not None:
            return context
        if self.default_workspace is None:
            raise WorkspaceStateError("Workspace runtime manager is not initialized")
        return self._contexts[self.default_workspace.id]

    def activate(self, context: WorkspaceContext) -> contextvars.Token:
        return self._current.set(context)

    def deactivate(self, token: contextvars.Token) -> None:
        self._current.reset(token)

    def rag_proxy(self) -> "WorkspaceRAGProxy":
        return WorkspaceRAGProxy(self)

    def document_manager_proxy(self) -> "WorkspaceDocumentManagerProxy":
        return WorkspaceDocumentManagerProxy(self)

    async def get_context(
        self, workspace_id: str | UUID | None = None, *, allow_deleting: bool = False
    ) -> WorkspaceContext:
        if workspace_id is None:
            workspace = await self.registry.get_default()
        else:
            workspace = await self.registry.get(workspace_id)

        if workspace.status != "active" and not (
            allow_deleting and workspace.status == "deleting"
        ):
            raise WorkspaceStateError(
                f"Workspace '{workspace.id}' is not available (status={workspace.status})"
            )
        return await self._context_for_workspace(workspace)

    async def _context_for_workspace(
        self, workspace: WorkspaceRecord
    ) -> WorkspaceContext:
        existing = self._contexts.get(workspace.id)
        if existing is not None:
            return existing

        lock = self._locks.setdefault(workspace.id, asyncio.Lock())
        async with lock:
            existing = self._contexts.get(workspace.id)
            if existing is not None:
                return existing
            rag = self.rag_factory(workspace.storage_key)
            doc_manager = DocumentManager(
                str(self.input_dir), workspace=workspace.storage_key
            )
            try:
                await rag.initialize_storages()
            except Exception:
                await rag.finalize_storages()
                raise
            context = WorkspaceContext(
                workspace=workspace, rag=rag, doc_manager=doc_manager
            )
            self._contexts[workspace.id] = context
            logger.info(
                "Initialized workspace runtime id=%s storage_key=%s",
                workspace.id,
                workspace.storage_key or "<legacy-default>",
            )
            return context

    async def create_workspace(
        self, display_name: str, description: str | None = None
    ) -> WorkspaceRecord:
        workspace = await self.registry.create(display_name, description)
        try:
            await self._context_for_workspace(workspace)
        except Exception:
            await self.registry.mark_failed(workspace.id)
            raise
        return workspace

    async def update_workspace(
        self,
        workspace_id: str | UUID,
        *,
        display_name: str | None = None,
        description: str | None | object = None,
        description_provided: bool = False,
    ) -> WorkspaceRecord:
        return await self.registry.update(
            workspace_id,
            display_name=display_name,
            description=description,
            description_provided=description_provided,
        )

    async def request_workspace_deletion(
        self, workspace_id: str | UUID
    ) -> tuple[WorkspaceRecord, str]:
        workspace = await self.registry.get(workspace_id)
        if workspace.is_default:
            raise WorkspaceStateError(
                "The default workspace cannot be deleted. Create and promote a new "
                "default workspace before deleting legacy data."
            )
        if workspace.status != "active":
            raise WorkspaceStateError(
                f"Workspace '{workspace.id}' cannot be deleted while status={workspace.status}"
            )
        context = await self._context_for_workspace(workspace)
        try:
            acquired, reason = await _acquire_destructive_busy(context.rag)
        except ValueError as exc:
            # Lightweight API/unit-test contexts can exist before
            # shared_storage is bootstrapped. Production contexts initialize
            # it through LightRAG.initialize_storages(); without it there is
            # no cross-request pipeline state to coordinate.
            if "Shared dictionaries not initialized" not in str(exc):
                raise
            acquired, reason = True, None
        if not acquired:
            raise WorkspaceBusyError(reason or "Workspace pipeline is busy")
        try:
            workspace = await self.registry.mark_deleting(workspace.id)
            job_id = await self.registry.create_job(workspace.id, "delete", "queued")
            return workspace, job_id
        except Exception:
            await self._release_delete_guard(context.rag)
            raise

    async def delete_workspace(self, workspace_id: str | UUID, job_id: str) -> None:
        """Run the idempotent destructive portion of workspace deletion.

        The destructive pipeline flag is acquired in ``request_workspace_deletion``
        and held until this function completes, so queued writes cannot resurrect
        storage rows after they are dropped.
        """

        workspace = await self.registry.get(workspace_id)
        context = await self._context_for_workspace(workspace)
        await self.registry.update_job(job_id, "running")
        try:
            storages = [
                context.rag.text_chunks,
                context.rag.full_docs,
                context.rag.full_entities,
                context.rag.full_relations,
                context.rag.entity_chunks,
                context.rag.relation_chunks,
                context.rag.entities_vdb,
                context.rag.relationships_vdb,
                context.rag.chunks_vdb,
                context.rag.chunk_entity_relation_graph,
                context.rag.llm_response_cache,
                context.rag.doc_status,
            ]
            results = await asyncio.gather(
                *(storage.drop() for storage in storages if storage is not None),
                return_exceptions=True,
            )
            errors = [
                str(result)
                for result in results
                if isinstance(result, Exception)
                or (isinstance(result, dict) and result.get("status") != "success")
            ]
            if errors:
                raise WorkspaceError("; ".join(errors))

            await self._remove_workspace_files(workspace)
            await context.rag.finalize_storages()
            self._contexts.pop(workspace.id, None)
            await self.registry.mark_deleted(workspace.id)
            await self.registry.update_job(job_id, "completed")
            logger.info("Deleted workspace id=%s", workspace.id)
        except Exception as exc:
            logger.error("Workspace deletion failed id=%s: %s", workspace.id, exc)
            await self.registry.mark_failed(workspace.id)
            await self.registry.update_job(job_id, "failed", str(exc))
        finally:
            await self._release_delete_guard(context.rag)

    @staticmethod
    async def _release_delete_guard(rag: Any) -> None:
        try:
            await _release_destructive_busy(rag)
        except ValueError as exc:
            if "Shared dictionaries not initialized" not in str(exc):
                raise

    async def _remove_workspace_files(self, workspace: WorkspaceRecord) -> None:
        if not workspace.storage_key:
            raise WorkspaceStateError(
                "Refusing to recursively remove the legacy root input directory"
            )
        base_input_dir = self.input_dir.resolve()
        workspace_dir = (base_input_dir / workspace.storage_key).resolve()
        try:
            workspace_dir.relative_to(base_input_dir)
        except ValueError as exc:
            raise WorkspaceStateError(
                "Workspace input path escaped the configured input directory"
            ) from exc
        if workspace_dir == base_input_dir:
            raise WorkspaceStateError("Refusing to remove the configured input root")
        if workspace_dir.exists():
            await asyncio.to_thread(shutil.rmtree, workspace_dir)

    async def finalize_non_default_contexts(self) -> None:
        default_id = self.default_workspace.id if self.default_workspace else None
        for workspace_id, context in list(self._contexts.items()):
            if workspace_id == default_id:
                continue
            try:
                await context.rag.finalize_storages()
            finally:
                self._contexts.pop(workspace_id, None)


class WorkspaceRAGProxy:
    """Delegate existing routers to the RAG bound by the request dependency."""

    def __init__(self, manager: WorkspaceRuntimeManager):
        self._manager = manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager.current_context().rag, name)

    def _resolve_workspace_value(self) -> Any:
        """Return the concrete RAG for background tasks scheduled by a route."""

        return self._manager.current_context().rag

    @property
    def workspace_id(self) -> str:
        """Public UUID of the request-selected workspace."""

        return self._manager.current_context().workspace.id


class WorkspaceDocumentManagerProxy:
    """Delegate existing document routes to the request workspace manager."""

    def __init__(self, manager: WorkspaceRuntimeManager):
        self._manager = manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager.current_context().doc_manager, name)

    def _resolve_workspace_value(self) -> DocumentManager:
        """Return the concrete document manager for a background task."""

        return self._manager.current_context().doc_manager


async def bind_workspace_context(
    request: Request,
    workspace_id: str | None = Header(
        default=None,
        alias=WORKSPACE_HEADER,
        description=(
            "Optional public workspace UUID. When omitted, the server uses the "
            "configured default workspace."
        ),
    ),
) -> AsyncGenerator[WorkspaceContext, None]:
    """FastAPI dependency that makes a header-selected workspace current.

    It is attached to the existing document/query/graph/Ollama routers, which
    preserves all current paths while making the header appear in OpenAPI.
    """

    manager: WorkspaceRuntimeManager = request.app.state.workspace_manager
    try:
        normalized_id = _normalize_workspace_id(workspace_id) if workspace_id else None
        context = await manager.get_context(normalized_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkspaceStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    token = manager.activate(context)
    try:
        yield context
    finally:
        manager.deactivate(token)


def validate_dynamic_workspace_configuration() -> None:
    """Reject backend-specific overrides that would collapse all workspaces."""

    import os

    forced = [
        name
        for name in _STATIC_STORAGE_WORKSPACE_ENV_VARS
        if os.getenv(name, "").strip()
    ]
    if forced:
        raise WorkspaceStateError(
            "Multi-workspace mode cannot run with fixed storage workspace "
            f"overrides: {', '.join(forced)}. Remove those variables so every "
            "backend receives the request workspace dynamically."
        )


def _record_from_mapping(row: Mapping[str, Any]) -> WorkspaceRecord:
    row_id = row.get("id", row.get("_id"))
    return WorkspaceRecord(
        id=str(row_id),
        storage_key=str(row["storage_key"]),
        display_name=str(row["display_name"]),
        description=row.get("description"),
        status=str(row["status"]),
        is_default=bool(row["is_default"]),
        created_at=_serialize_timestamp(row["created_at"]),
        updated_at=_serialize_timestamp(row["updated_at"]),
        deleted_at=(
            _serialize_timestamp(row["deleted_at"])
            if row.get("deleted_at") is not None
            else None
        ),
    )


def _workspace_document(record: WorkspaceRecord) -> dict[str, Any]:
    return {
        "_id": record.id,
        "storage_key": record.storage_key,
        "display_name": record.display_name,
        "description": record.description,
        "status": record.status,
        "is_default": record.is_default,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "deleted_at": record.deleted_at,
    }


def _postgres_value(column: str, value: Any) -> Any:
    if column.endswith("_at") and isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


def _serialize_timestamp(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _normalize_workspace_id(value: str | UUID) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("X-LightRAG-Workspace-ID must be a valid UUID") from exc


def _validate_display_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Workspace display_name cannot be empty")
    if len(normalized) > 255:
        raise ValueError("Workspace display_name must not exceed 255 characters")
    return normalized


def _normalize_description(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if len(normalized) > 10_000:
        raise ValueError("Workspace description must not exceed 10000 characters")
    return normalized or None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
