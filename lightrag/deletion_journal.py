"""Crash-safe journal for deferred batch document deletion.

The journal intentionally lives under ``working_dir`` rather than in a
particular KV backend.  LightRAG deployments may mix PostgreSQL, JSON, Redis,
or another graph store, while ``working_dir`` is the shared persistent state
location already required by the application.  Writes use replace-after-fsync
so a container restart can only observe the previous complete journal or the
new complete journal, never a partial JSON document.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ``flock``/``msvcrt.locking`` protect separate server processes. The asyncio
# lock closes the same-process hole because POSIX advisory locks are process-
# scoped and may otherwise let two tasks in one worker re-enter the same lock.
_in_process_batch_locks: dict[str, asyncio.Lock] = {}
_in_process_batch_locks_guard = threading.Lock()


class DeferredDeletionStage(str, Enum):
    PREPARED = "PREPARED"
    REBUILD_PENDING = "REBUILD_PENDING"
    FINALIZATION_PENDING = "FINALIZATION_PENDING"
    COMMITTED = "COMMITTED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_relation_key(source: str, target: str) -> tuple[str, str]:
    return (source, target) if source <= target else (target, source)


def _ordered_union(existing: list[str], incoming: list[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *incoming]))


_WINDOWS_RESERVED_COMPONENT_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CLOCK$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_WINDOWS_INVALID_COMPONENT_CHARACTERS = frozenset('<>:"/\\\\|?*')


def _is_windows_reserved_component(component: str) -> bool:
    """Return whether a path component aliases a Windows device name.

    Win32 treats device basenames as reserved even when an extension follows
    (for example ``CON.json`` and ``LPT9.backup``), and strips a trailing space
    from that basename. Validate before adding our ``.json``/``.lock`` suffixes.
    """
    basename = component.split(".", 1)[0].rstrip(" ").upper()
    return basename in _WINDOWS_RESERVED_COMPONENT_BASENAMES


def _workspace_component(workspace: str) -> str:
    """Return a deterministic workspace directory name valid on Windows and POSIX."""
    workspace = workspace or "default"
    is_safe_component = (
        workspace not in {".", ".."}
        and not workspace.endswith((".", " "))
        and not _is_windows_reserved_component(workspace)
        and not any(
            character in _WINDOWS_INVALID_COMPONENT_CHARACTERS or ord(character) < 32
            for character in workspace
        )
    )
    if is_safe_component:
        return workspace

    digest = hashlib.sha256(workspace.encode("utf-8")).hexdigest()
    return f"workspace-{digest}"


@dataclass
class DeferredDeletionJournal:
    """All state needed to safely resume a deferred deletion rebuild."""

    batch_id: str
    document_ids: list[str]
    delete_llm_cache: bool
    delete_file: bool = False
    source_file_paths: dict[str, str] = field(default_factory=dict)
    # Claimed durably before best-effort source cleanup. This intentionally
    # provides at-most-once semantics after crashes.
    source_file_cleanup_claimed: bool = False
    stage: DeferredDeletionStage = DeferredDeletionStage.PREPARED
    entities_to_rebuild: dict[str, list[str]] = field(default_factory=dict)
    relationships_to_rebuild: dict[tuple[str, str], list[str]] = field(
        default_factory=dict
    )
    document_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Save INTENT before calling storage and COMPLETED plus its rebuild result
    # afterwards. An interrupted INTENT has an unknowable outcome and must not
    # be retried as another destructive document delete.
    document_delete_checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_document_index: int = 0
    attempt_count: int = 0
    error_detail: str | None = None
    recovery_stage: DeferredDeletionStage | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    @classmethod
    def new(
        cls,
        batch_id: str,
        document_ids: list[str],
        delete_llm_cache: bool,
        delete_file: bool = False,
    ) -> "DeferredDeletionJournal":
        if not batch_id:
            raise ValueError("batch_id is required")
        if not document_ids:
            raise ValueError("document_ids is required")
        return cls(
            batch_id=batch_id,
            document_ids=list(dict.fromkeys(document_ids)),
            delete_llm_cache=delete_llm_cache,
            delete_file=delete_file,
        )

    def aggregate_targets(
        self,
        *,
        entities: dict[str, list[str]],
        relationships: dict[tuple[str, str], list[str]],
    ) -> None:
        for entity_name, chunk_ids in entities.items():
            self.entities_to_rebuild[entity_name] = _ordered_union(
                self.entities_to_rebuild.get(entity_name, []), chunk_ids
            )
        for (source, target), chunk_ids in relationships.items():
            relation_key = _canonical_relation_key(source, target)
            self.relationships_to_rebuild[relation_key] = _ordered_union(
                self.relationships_to_rebuild.get(relation_key, []), chunk_ids
            )
        self.updated_at = _utc_now()

    def mark_document_delete_intent(self, doc_id: str) -> None:
        if doc_id in self.document_delete_checkpoints:
            raise ValueError(
                f"document deletion checkpoint already exists for {doc_id!r}"
            )
        self.document_delete_checkpoints[doc_id] = {"state": "INTENT"}
        self.updated_at = _utc_now()

    def record_document_delete_completed(
        self,
        doc_id: str,
        *,
        entities: dict[str, list[str]],
        relationships: dict[tuple[str, str], list[str]],
    ) -> None:
        checkpoint = self.document_delete_checkpoints.get(doc_id)
        if checkpoint != {"state": "INTENT"}:
            raise ValueError(f"document deletion intent is missing for {doc_id!r}")
        self.document_delete_checkpoints[doc_id] = {
            "state": "COMPLETED",
            "entities_to_rebuild": {
                name: list(chunk_ids) for name, chunk_ids in entities.items()
            },
            "relationships_to_rebuild": [
                {"source": source, "target": target, "chunk_ids": list(chunk_ids)}
                for (source, target), chunk_ids in relationships.items()
            ],
        }
        self.aggregate_targets(entities=entities, relationships=relationships)

    def mark_rebuild_pending(self) -> None:
        if self.stage not in {
            DeferredDeletionStage.PREPARED,
            DeferredDeletionStage.FAILED_RETRYABLE,
        }:
            raise ValueError(f"cannot move {self.stage} to REBUILD_PENDING")
        self.stage = DeferredDeletionStage.REBUILD_PENDING
        self.error_detail = None
        self.updated_at = _utc_now()

    def mark_failed_retryable(self, error_detail: str) -> None:
        if self.stage is not DeferredDeletionStage.FAILED_RETRYABLE:
            self.recovery_stage = self.stage
        self.stage = DeferredDeletionStage.FAILED_RETRYABLE
        self.error_detail = error_detail
        self.attempt_count += 1
        self.updated_at = _utc_now()

    def mark_finalization_pending(self) -> None:
        if self.stage is not DeferredDeletionStage.REBUILD_PENDING:
            raise ValueError("journal must be REBUILD_PENDING before finalization")
        self.stage = DeferredDeletionStage.FINALIZATION_PENDING
        self.error_detail = None
        self.updated_at = _utc_now()

    def mark_committed(self) -> None:
        if self.stage is not DeferredDeletionStage.FINALIZATION_PENDING:
            raise ValueError("journal must be FINALIZATION_PENDING before COMMITTED")
        self.stage = DeferredDeletionStage.COMMITTED
        self.error_detail = None
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "document_ids": self.document_ids,
            "delete_llm_cache": self.delete_llm_cache,
            "delete_file": self.delete_file,
            "source_file_paths": self.source_file_paths,
            "source_file_cleanup_claimed": self.source_file_cleanup_claimed,
            "stage": self.stage.value,
            "entities_to_rebuild": self.entities_to_rebuild,
            "relationships_to_rebuild": [
                {"source": source, "target": target, "chunk_ids": chunk_ids}
                for (source, target), chunk_ids in self.relationships_to_rebuild.items()
            ],
            "document_metadata": self.document_metadata,
            "document_delete_checkpoints": self.document_delete_checkpoints,
            "current_document_index": self.current_document_index,
            "attempt_count": self.attempt_count,
            "error_detail": self.error_detail,
            "recovery_stage": self.recovery_stage.value
            if self.recovery_stage
            else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeferredDeletionJournal":
        relationships = {
            _canonical_relation_key(entry["source"], entry["target"]): list(
                entry.get("chunk_ids", [])
            )
            for entry in data.get("relationships_to_rebuild", [])
        }
        return cls(
            batch_id=data["batch_id"],
            document_ids=list(data["document_ids"]),
            delete_llm_cache=bool(data["delete_llm_cache"]),
            delete_file=bool(data.get("delete_file", False)),
            source_file_paths={
                doc_id: file_path
                for doc_id, file_path in data.get("source_file_paths", {}).items()
                if isinstance(file_path, str)
            },
            source_file_cleanup_claimed=bool(
                data.get("source_file_cleanup_claimed", False)
            ),
            stage=DeferredDeletionStage(data["stage"]),
            entities_to_rebuild={
                name: list(chunk_ids)
                for name, chunk_ids in data.get("entities_to_rebuild", {}).items()
            },
            relationships_to_rebuild=relationships,
            document_metadata=dict(data.get("document_metadata", {})),
            document_delete_checkpoints={
                doc_id: dict(checkpoint)
                for doc_id, checkpoint in data.get(
                    "document_delete_checkpoints", {}
                ).items()
                if isinstance(doc_id, str) and isinstance(checkpoint, dict)
            },
            current_document_index=int(data.get("current_document_index", 0)),
            attempt_count=int(data.get("attempt_count", 0)),
            error_detail=data.get("error_detail"),
            recovery_stage=(
                DeferredDeletionStage(data["recovery_stage"])
                if data.get("recovery_stage")
                else None
            ),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


def _is_windows() -> bool:
    """Small seam for platform-specific locking and its non-Windows test coverage."""
    return os.name == "nt"


class DeferredDeletionJournalStore:
    """Atomic filesystem persistence for deferred deletion journals."""

    def __init__(self, working_dir: str | Path, workspace: str) -> None:
        journal_root = (Path(working_dir).resolve() / "deletion_journals").resolve()
        directory = journal_root / _workspace_component(workspace)
        if not directory.resolve().is_relative_to(journal_root):
            raise ValueError("workspace journal directory escapes working_dir")
        self.directory = directory

    def _path(self, batch_id: str) -> Path:
        if (
            not batch_id
            or batch_id in {".", ".."}
            or batch_id.endswith((".", " "))
            or _is_windows_reserved_component(batch_id)
            or any(
                character in _WINDOWS_INVALID_COMPONENT_CHARACTERS
                or ord(character) < 32
                for character in batch_id
            )
        ):
            raise ValueError("invalid batch_id")
        return self.directory / f"{batch_id}.json"

    def _lock_path(self, batch_id: str) -> Path:
        self._path(batch_id)  # validate before constructing another filesystem name
        return self.directory / f".{batch_id}.lock"

    @staticmethod
    def _in_process_lock(lock_path: Path) -> asyncio.Lock:
        key = str(lock_path.resolve())
        with _in_process_batch_locks_guard:
            return _in_process_batch_locks.setdefault(key, asyncio.Lock())

    @staticmethod
    def _acquire_os_lock(lock_path: Path) -> int:
        """Acquire an advisory lock released automatically when the process dies.

        POSIX uses ``flock``. Windows uses a one-byte ``msvcrt.locking`` lock.
        Both are OS-owned file locks, so a process crash closes the descriptor
        and releases the lock rather than requiring a stale-lease timeout.
        """
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError(
                    f"deletion journal lock is not a regular file: {lock_path}"
                )
            if not _is_windows():
                os.fchmod(descriptor, 0o600)
            if _is_windows():
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                getattr(msvcrt, "locking")(descriptor, getattr(msvcrt, "LK_LOCK"), 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _release_os_lock(descriptor: int) -> None:
        try:
            if _is_windows():
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                getattr(msvcrt, "locking")(descriptor, getattr(msvcrt, "LK_UNLCK"), 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @asynccontextmanager
    async def batch_lock(self, batch_id: str):
        """Serialize one batch across tasks and processes for run/resume.

        The lock file is intentionally retained: it contains no journal state,
        and retaining it avoids an unlink/recreate race. Its advisory lock is
        released by the OS on normal close *and on process crash* on POSIX and
        Windows, so a restart cannot deadlock on an abandoned lease.
        """
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        lock_path = self._lock_path(batch_id)
        async with self._in_process_lock(lock_path):
            # ``to_thread`` cannot be cancelled once its worker started.  Keep
            # a done callback so cancellation while waiting for a contended
            # OS lock cannot orphan a descriptor that acquires later.
            acquire_task = asyncio.create_task(
                asyncio.to_thread(self._acquire_os_lock, lock_path)
            )
            acquisition_abandoned = False
            release_scheduled = False

            def release_if_unclaimed(task: asyncio.Task[int]) -> None:
                nonlocal release_scheduled
                if not acquisition_abandoned or release_scheduled or task.cancelled():
                    return
                try:
                    descriptor = task.result()
                except BaseException:
                    return
                release_scheduled = True
                asyncio.create_task(
                    asyncio.to_thread(self._release_os_lock, descriptor)
                )

            acquire_task.add_done_callback(release_if_unclaimed)
            try:
                descriptor = await asyncio.shield(acquire_task)
            except BaseException:
                # A cancellation can race with a successful worker result. Mark
                # the acquisition abandoned and invoke the callback directly
                # when it already completed; otherwise its done callback will
                # release the descriptor after flock/msvcrt returns.
                acquisition_abandoned = True
                if acquire_task.done():
                    release_if_unclaimed(acquire_task)
                raise
            try:
                yield
            finally:
                await asyncio.to_thread(self._release_os_lock, descriptor)

    def save(self, journal: DeferredDeletionJournal) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        destination = self._path(journal.batch_id)
        payload = json.dumps(journal.to_dict(), sort_keys=True, separators=(",", ":"))
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory, prefix=f".{journal.batch_id}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            os.chmod(destination, 0o600)
        finally:
            if descriptor != -1:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        # POSIX requires syncing the directory entry after replace. Windows has
        # no directory file descriptor equivalent; ``os.replace`` + file fsync
        # still provides an atomic complete-file journal there.
        if os.name != "nt":
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def load(self, batch_id: str) -> DeferredDeletionJournal | None:
        path = self._path(batch_id)
        try:
            if path.is_symlink():
                raise OSError(f"refusing symlinked deletion journal: {path}")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, encoding="utf-8") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise OSError(f"deletion journal is not a regular file: {path}")
                return DeferredDeletionJournal.from_dict(json.load(handle))
        except FileNotFoundError:
            return None

    def delete(self, batch_id: str) -> None:
        try:
            self._path(batch_id).unlink()
        except FileNotFoundError:
            return

    def list_unfinished(self) -> list[DeferredDeletionJournal]:
        """Return durable journals that require an explicit operator decision."""
        if not self.directory.exists():
            return []
        journals = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                # Reuse the secure no-follow loader rather than opening the
                # glob result directly; operator list endpoints must not follow
                # a journal-shaped symlink planted in working_dir.
                journal = self.load(path.stem)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                journal is not None
                and journal.stage is not DeferredDeletionStage.COMMITTED
            ):
                journals.append(journal)
        return journals
