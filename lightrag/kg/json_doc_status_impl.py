from dataclasses import dataclass
import heapq
import json
import os
import uuid
from typing import Any, ClassVar, Union, final

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    CursorPosition,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    FailureGenerationMode,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.file_atomic import reap_orphan_tmp_files
from lightrag.utils import (
    _cooperative_yield,
    load_json,
    logger,
    validate_workspace,
    write_json,
    get_pinyin_sort_key,
)
from lightrag.exceptions import (
    StorageControlPlaneError,
    StorageNotInitializedError,
    StorageRecordNotFoundError,
)
from .shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)


@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON-file-backed document-status storage, sharing memory across processes.

    Uses the **same shared-memory + dirty-flag protocol** as
    ``JsonKVStorage`` — see that class's docstring for the canonical
    description of:
        * how ``self._data`` is a cross-process
          ``multiprocessing.Manager().dict()`` proxy obtained via
          ``get_namespace_data``;
        * how ``try_initialize_namespace`` ensures exactly one process
          reads the JSON file on first init;
        * how ``set_all_update_flags`` marks dirty state (semantics
          *reversed* from the file-backed classes
          ``NanoVectorDBStorage`` / ``FaissVectorDBStorage`` /
          ``NetworkXStorage``);
        * how ``index_done_callback`` flushes and calls
          ``clear_all_update_flags``;
        * why ``_storage_lock`` wraps **every** ``self._data`` access
          (not just commit / reload).

    Differences from ``JsonKVStorage`` (in this class only):
        * ``upsert`` calls ``index_done_callback`` synchronously after
          mutating shared memory, so doc-status changes hit disk
          immediately rather than being deferred to the pipeline's
          batched ``_insert_done()``. Rationale: doc-status is the
          recovery anchor for the ingest pipeline — if the process
          crashes after an in-memory upsert but before the next batch
          commit, the doc must still be visible as PENDING/PROCESSING
          on restart. The other writes (``delete``, ``drop``) follow
          the standard deferred-commit pattern.
        * Pre-upsert preparation (``chunks_list`` default) runs
          *outside* the lock because it only mutates the caller-
          supplied dict, not the shared store.
        * Read methods are richer (``get_docs_by_status`` /
          ``get_docs_by_track_id`` / ``get_docs_paginated`` /
          ``get_doc_by_file_path`` / etc.), but they all follow the
          same "acquire ``_storage_lock``, scan ``self._data``, copy
          values out before returning" template.

    Non-pipeline write paths:
        * ``drop`` — destructive, **not** serialized; the caller must
          hold the pipeline ``busy`` reservation (the
          ``/documents/clear`` endpoint does this).

    Memory-bounding capability boundary (Phase 1): the paged scheduling
    API below is a true keyset sweep (bounded page memory via a bounded
    heap), but this backend inherently keeps the WHOLE store resident in
    shared memory and rewrites the whole file on flush — that residency
    is an independent, documented limitation, not something the page API
    can fix. Deployments with very large doc counts should use a server
    backend (Redis/PG/...).
    """

    supports_bounded_scheduling_pages: ClassVar[bool] = True
    supports_failure_generation: ClassVar[bool] = True
    supports_strict_doc_identity_lookup: ClassVar[bool] = True
    supports_strict_point_reads: ClassVar[bool] = True

    # Scheduling control-plane doc (failure-generation counter + mode
    # marker + version stamps) lives in a SEPARATE sidecar file, never in
    # ``self._data`` — a reserved key inside the data dict would leak into
    # every full-scan reader. Version stamps guard marker integrity: a
    # mismatch reads as MIGRATING (never LEGACY).
    _CTRL_SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self):
        # Reject path traversal before using workspace in a file path
        validate_workspace(self.workspace)
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            # Default behavior when workspace is empty
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._ctrl_file_name = os.path.join(
            workspace_dir, f"kv_store_{self.namespace}_scheduling_ctrl.json"
        )
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

        reap_orphan_tmp_files(self._file_name, self.workspace or "_")
        reap_orphan_tmp_files(self._ctrl_file_name, self.workspace or "_")

    async def initialize(self):
        """Bind to the shared namespace dict and load from disk on first init.

        Same protocol as ``JsonKVStorage.initialize``: a global init
        lock (``try_initialize_namespace``) elects one process to read
        the JSON file into the shared ``self._data``; other processes
        skip the read and see the same shared dict.
        """
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        async with get_data_init_lock():
            # check need_init must before get_namespace_data
            need_init = await try_initialize_namespace(
                self.namespace, workspace=self.workspace
            )
            self._data = await get_namespace_data(
                self.namespace, workspace=self.workspace
            )
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    self._data.update(loaded_data)
                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} doc status load {self.namespace} with {len(loaded_data)} records"
                    )

        # Scheduling control-plane bootstrap (Phase 1). JSON is a single-host
        # backend whose process fleet upgrades atomically with the code, so
        # the migrate-then-publish-ENFORCED step is safe at init time (server
        # backends require the coordinated stop-write upgrade instead).
        # Counter calibration runs on EVERY init: a restore-from-backup can
        # roll the ctrl file behind persisted rows, and the reservation
        # counter must stay monotonic — ``max(counter, max persisted)``.
        async with self._storage_lock:
            max_gen = 0
            for row in self._data.values():
                if not isinstance(row, dict):
                    continue
                try:
                    gen = int(row.get("failure_generation") or 0)
                except (TypeError, ValueError):
                    gen = 0
                max_gen = max(max_gen, gen)
            ctrl = load_json(self._ctrl_file_name)
            if not isinstance(ctrl, dict):
                ctrl = {
                    "schema_version": self._CTRL_SCHEMA_VERSION,
                    "mode": FailureGenerationMode.ENFORCED.value,
                    "failure_generation_counter": max_gen,
                }
                write_json(ctrl, self._ctrl_file_name)
                logger.info(
                    f"[{self.workspace}] Published failure-generation marker "
                    f"(mode=enforced, counter={max_gen}) for {self.namespace}"
                )
            elif ctrl.get("schema_version") == self._CTRL_SCHEMA_VERSION:
                try:
                    counter = int(ctrl.get("failure_generation_counter") or 0)
                except (TypeError, ValueError):
                    counter = 0
                if counter < max_gen:
                    ctrl["failure_generation_counter"] = max_gen
                    write_json(ctrl, self._ctrl_file_name)
                    logger.warning(
                        f"[{self.workspace}] failure-generation counter behind "
                        f"persisted rows ({counter} < {max_gen}); recalibrated"
                    )
            # An unknown schema_version is left untouched: mode reads report
            # MIGRATING (never LEGACY) until an explicit migration handles it.

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        ordered_results: list[dict[str, Any] | None] = []
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            for id in ids:
                data = self._data.get(id, None)
                if data:
                    ordered_results.append(data.copy())
                else:
                    ordered_results.append(None)
        return ordered_results

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status.value: 0 for status in DocStatus}
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            for doc in self._data.values():
                counts[doc["status"]] += 1
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching any of the given statuses in a single pass.

        Acquires the storage lock once and scans the in-memory dict once,
        filtering against a set of status values.  More efficient than N separate
        get_docs_by_status() calls, which would acquire the lock N times and scan
        the data N times.  ``strict=True`` raises on any record that cannot be
        converted (complete-or-raise scheduling contract, see base class).
        """
        if not statuses:
            return {}
        status_values = {s.value for s in statuses}
        result = {}
        async with self._storage_lock:
            for k, v in self._data.items():
                try:
                    # Read ``status`` INSIDE the try: a record missing it (or
                    # that is not a mapping) is then skipped in relaxed mode and
                    # raised under strict — symmetric with the other required
                    # fields below, instead of crashing every relaxed caller.
                    if v["status"] not in status_values:
                        continue
                    data = v.copy()
                    data.pop("content", None)
                    if not data.get("file_path"):
                        data["file_path"] = "no-file-path"
                    if "metadata" not in data:
                        data["metadata"] = {}
                    if "error_msg" not in data:
                        data["error_msg"] = None
                    result[k] = DocProcessingStatus(**data)
                except (KeyError, TypeError) as e:
                    logger.error(
                        f"[{self.workspace}] Missing required field for document {k}: {e}"
                    )
                    if strict:
                        raise
                    continue
        return result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
        result = {}
        async with self._storage_lock:
            for k, v in self._data.items():
                if v.get("track_id") == track_id:
                    try:
                        # Make a copy of the data to avoid modifying the original
                        data = v.copy()
                        # Remove deprecated content field if it exists
                        data.pop("content", None)
                        # Normalize missing or null file_path
                        if not data.get("file_path"):
                            data["file_path"] = "no-file-path"
                        # Ensure new fields exist with default values
                        if "metadata" not in data:
                            data["metadata"] = {}
                        if "error_msg" not in data:
                            data["error_msg"] = None
                        result[k] = DocProcessingStatus(**data)
                    except KeyError as e:
                        logger.error(
                            f"[{self.workspace}] Missing required field for document {k}: {e}"
                        )
                        continue
        return result

    async def index_done_callback(self) -> None:
        """Flush dirty shared memory to disk and clear all dirty flags.

        Identical commit protocol to ``JsonKVStorage.index_done_callback``
        (snapshot the shared dict → ``write_json`` → if sanitization
        happened reload the cleaned data → ``clear_all_update_flags``).
        See ``JsonKVStorage`` docstring for details.
        """
        async with self._storage_lock:
            if self.storage_updated.value:
                # DictProxy.copy() is a single Manager RPC that marshals the
                # whole mapping server-side; dict(proxy) would walk the mapping
                # protocol and fetch every value with its own RPC. Plain dicts
                # (single-process mode) copy cheaply and identically — write_json
                # only reads its argument, so the shallow copy is safe there too.
                data_dict = self._data.copy()
                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} doc status writting {len(data_dict)} records to {self.namespace}"
                )

                # Write JSON and check if sanitization was applied
                needs_reload = write_json(data_dict, self._file_name)

                # If data was sanitized, reload cleaned data to update shared memory
                if needs_reload:
                    logger.info(
                        f"[{self.workspace}] Reloading sanitized data into shared memory for {self.namespace}"
                    )
                    cleaned_data = load_json(self._file_name)
                    if cleaned_data is not None:
                        self._data.clear()
                        self._data.update(cleaned_data)

                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert/update doc-status records and **persist immediately**.

        Differs from ``JsonKVStorage.upsert`` in that it calls
        ``index_done_callback`` synchronously at the end, so changes
        are flushed to disk before this coroutine returns. Rationale:
        doc-status is the recovery anchor for the ingest pipeline — if
        the process crashes after an in-memory upsert but before the
        next batch commit, the doc must still be visible as
        PENDING/PROCESSING on restart.

        Steps:
            1. Pre-process the caller's dict (default ``chunks_list``)
               **outside** the lock — only mutates the caller-supplied
               value dicts, not shared state.
            2. Under ``_storage_lock``, ``self._data.update(data)`` and
               ``set_all_update_flags`` to mark every process dirty.
            3. Await ``index_done_callback`` for an immediate flush.

        See ``JsonKVStorage`` class docstring for the shared-memory +
        dirty-flag protocol that underpins step 2.
        """
        if not data:
            return
        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        # Prepare data outside the lock: this only mutates the caller-supplied
        # dict values, not shared storage state, so no lock needed here.
        for i, (doc_id, doc_data) in enumerate(data.items(), start=1):
            if "chunks_list" not in doc_data:
                doc_data["chunks_list"] = []
            await _cooperative_yield(i)
        async with self._storage_lock:
            self._data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)

        await self.index_done_callback()

    async def is_empty(self) -> bool:
        """Check if the storage is empty

        Returns:
            bool: True if storage is empty, False otherwise

        Raises:
            StorageNotInitializedError: If storage is not initialized
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            return len(self._data) == 0

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._storage_lock:
            return self._data.get(id)

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """
        status_filter_values = self.resolve_status_filter_values(
            status_filter=status_filter,
            status_filters=status_filters,
        )

        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        # For JSON storage, we load all data and sort/filter in memory
        all_docs = []

        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                # Apply status filter
                if (
                    status_filter_values is not None
                    and doc_data.get("status") not in status_filter_values
                ):
                    continue

                try:
                    # Prepare document data
                    data = doc_data.copy()
                    data.pop("content", None)
                    if not data.get("file_path"):
                        data["file_path"] = "no-file-path"
                    if "metadata" not in data:
                        data["metadata"] = {}
                    if "error_msg" not in data:
                        data["error_msg"] = None

                    doc_status = DocProcessingStatus(**data)

                    # Add sort key for sorting
                    if sort_field == "id":
                        doc_status._sort_key = doc_id
                    elif sort_field == "file_path":
                        # Use pinyin sorting for file_path field to support Chinese characters
                        file_path_value = getattr(doc_status, sort_field, "")
                        doc_status._sort_key = get_pinyin_sort_key(file_path_value)
                    else:
                        doc_status._sort_key = getattr(doc_status, sort_field, "")

                    all_docs.append((doc_id, doc_status))

                except KeyError as e:
                    logger.error(
                        f"[{self.workspace}] Error processing document {doc_id}: {e}"
                    )
                    continue

        # Sort documents
        reverse_sort = sort_direction.lower() == "desc"
        all_docs.sort(
            key=lambda x: getattr(x[1], "_sort_key", ""), reverse=reverse_sort
        )

        # Remove sort key from documents
        for doc_id, doc in all_docs:
            if hasattr(doc, "_sort_key"):
                delattr(doc, "_sort_key")

        total_count = len(all_docs)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = all_docs[start_idx:end_idx]

        return paginated_docs, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        counts = await self.get_status_counts()

        # Add 'all' field with total count
        total_count = sum(counts.values())
        counts["all"] = total_count

        return counts

    async def delete(self, doc_ids: list[str]) -> None:
        """Remove doc-status records from shared memory.

        Unlike ``upsert``, ``delete`` does **not** force an immediate
        flush — it follows the standard deferred-commit pattern.
        Persistence happens at the next ``index_done_callback``
        (driven by the pipeline's ``_insert_done()`` at end of batch).

        Only calls ``set_all_update_flags`` if at least one key was
        actually present (avoids creating spurious dirty state for
        no-op deletes).

        Args:
            doc_ids: List of document IDs to be deleted from storage
        """
        async with self._storage_lock:
            any_deleted = False
            for doc_id in doc_ids:
                result = self._data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
            Returns the same format as get_by_ids method
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                if doc_data.get("file_path") == file_path:
                    # Return complete document data, consistent with get_by_ids method
                    return doc_data

        return None

    @staticmethod
    def _is_duplicate_row(row: Any) -> bool:
        """True for duplicate-attempt marker rows (``dup-*`` records and
        post-parse content duplicates): ``metadata.is_duplicate == true``."""
        if not isinstance(row, dict):
            return False
        metadata = row.get("metadata")
        return bool(isinstance(metadata, dict) and metadata.get("is_duplicate"))

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find the PRIMARY record whose canonical basename matches.

        The caller is responsible for passing an already-canonical basename.
        Stored ``file_path`` values are canonicalized by the business layer, so
        this lookup intentionally performs an exact match only.

        ``file_path`` is one-to-many (duplicate-attempt ``dup-*`` rows keep the
        same canonical basename); this returns the single primary
        (``metadata.is_duplicate != true``) row — callers doing identity
        checks / dedup want the document, never a duplicate marker. When only
        duplicate markers remain (primary deleted) the basename is free again
        and this returns ``None``.
        """
        if not basename:
            return None
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        if basename == "unknown_source":
            return None
        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                if doc_data.get("file_path") == basename and not self._is_duplicate_row(
                    doc_data
                ):
                    return doc_id, doc_data
        return None

    async def get_doc_by_file_basename_strict(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Fail-closed basename lookup (see base contract).

        The in-memory scan has no transport failure surface and never
        swallows errors, so the aligned legacy lookup already satisfies the
        strict contract: ``None`` here IS confirmed absence (uninitialized
        storage raises instead of returning a miss).
        """
        return await self.get_doc_by_file_basename(basename)

    async def get_by_id_strict(self, id: str) -> Union[dict[str, Any], None]:
        """Strict point read: complete-or-raise (base contract).

        In-memory shared dict — a miss is a confirmed absence; the only
        failure surface is uninitialized storage, which raises.
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            row = self._data.get(id)
        return row.copy() if isinstance(row, dict) else row

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find an existing record whose content_hash field matches."""
        if not content_hash:
            return None
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                if doc_data.get("content_hash") == content_hash:
                    return doc_id, doc_data
        return None

    # ------------------------------------------------------------------
    # Memory-bounding scheduling API (Phase 1)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_sort_key(doc_id: str, row: dict[str, Any]) -> tuple[str, str]:
        """(created_at, id) keyset key. A missing/non-string created_at sorts
        deterministically first as "" — such legacy rows are consumed (and
        skipped/raised per strictness) without ever moving mid-sweep."""
        created = row.get("created_at")
        return (created if isinstance(created, str) else "", doc_id)

    @staticmethod
    def _encode_cursor(key: tuple[str, str]) -> str:
        return json.dumps(list(key), ensure_ascii=False)

    @staticmethod
    def _decode_cursor(opaque: str) -> tuple[str, str]:
        try:
            decoded = json.loads(opaque)
            created, doc_id = decoded
            if not isinstance(created, str) or not isinstance(doc_id, str):
                raise ValueError("cursor fields must be strings")
        except (ValueError, TypeError) as e:
            raise StorageControlPlaneError(
                f"Malformed scheduling cursor for JsonDocStatusStorage: {e}"
            ) from e
        return (created, doc_id)

    def _scheduling_record_from_row(
        self, doc_id: str, row: dict[str, Any], *, strict: bool
    ) -> DocSchedulingRecord | None:
        """Project one raw row; strict raises on unusable rows, relaxed
        returns None (the caller still counts the row as consumed)."""
        try:
            status = DocStatus(str(row["status"]))
            created_at = row["created_at"]
            updated_at = row.get("updated_at", created_at)
            if not isinstance(created_at, str) or not isinstance(updated_at, str):
                raise TypeError("created_at/updated_at must be strings")
            metadata = row.get("metadata")
            return DocSchedulingRecord(
                id=doc_id,
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                file_path=row.get("file_path") or "no-file-path",
                track_id=row.get("track_id"),
                has_custom_chunk_journal=isinstance(metadata, dict)
                and isinstance(metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"[{self.workspace}] Unusable scheduling row {doc_id}: {e}")
            if strict:
                raise
            return None

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        max_failure_generation: int | None = None,
        strict: bool = False,
    ) -> DocStatusPage:
        """Bounded keyset page over the in-memory store.

        Selection uses a bounded heap (``heapq.nsmallest``) so page memory is
        O(limit) even though each page re-scans the resident dict — the
        O(total) residency itself is this backend's documented boundary.

        Consumed-position contract: every selected candidate is consumed —
        returned, dropped by the ``max_failure_generation`` predicate, or
        skipped as unusable in relaxed mode — so the cursor advances past
        filtered pages instead of re-reading them; fewer candidates than
        ``limit`` proves exhaustion (CURSOR_END).
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if not statuses or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        last_key: tuple[str, str] | None = None
        if isinstance(position, CursorAfter):
            last_key = self._decode_cursor(position.opaque)
        status_values = {s.value for s in statuses}
        failed_value = DocStatus.FAILED.value

        def _candidates():
            for doc_id, row in self._data.items():
                if not isinstance(row, dict):
                    if strict:
                        raise TypeError(f"doc_status record {doc_id} is not a mapping")
                    continue
                if str(row.get("status")) not in status_values:
                    continue
                key = self._row_sort_key(doc_id, row)
                if last_key is not None and key <= last_key:
                    continue
                yield key, doc_id, row

        async with self._storage_lock:
            selected = heapq.nsmallest(limit, _candidates(), key=lambda t: t[0])
            docs: dict[str, DocSchedulingRecord] = {}
            for key, doc_id, row in selected:
                if (
                    max_failure_generation is not None
                    and str(row.get("status")) == failed_value
                ):
                    try:
                        generation = int(row.get("failure_generation") or 0)
                    except (TypeError, ValueError):
                        generation = 0
                    if generation > max_failure_generation:
                        continue  # consumed by the cohort predicate
                record = self._scheduling_record_from_row(doc_id, row, strict=strict)
                if record is None:
                    continue  # relaxed skip is still consumed
                docs[doc_id] = record
        if len(selected) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            next_position = CursorAfter(self._encode_cursor(selected[-1][0]))
        return DocStatusPage(docs=docs, next_position=next_position)

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        """Fail-closed status count (full scan — documented JSON boundary)."""
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        if not statuses:
            return 0
        status_values = {s.value for s in statuses}
        count = 0
        async with self._storage_lock:
            for doc_id, row in self._data.items():
                if not isinstance(row, dict) or "status" not in row:
                    if strict:
                        raise TypeError(
                            f"doc_status record {doc_id} has no readable status"
                        )
                    continue
                if str(row.get("status")) in status_values:
                    count += 1
        return count

    async def update_doc_status_fields(
        self,
        doc_id: str,
        fields: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> None:
        """Targeted field update with an immediate flush (doc-status is the
        pipeline's recovery anchor — same rationale as ``upsert``)."""
        if "created_at" in fields:
            raise ValueError(
                "created_at is an immutable scheduling sort key and cannot "
                "be changed via update_doc_status_fields"
            )
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            row = self._data.get(doc_id)
            if row is None:
                if missing_ok:
                    return
                raise StorageRecordNotFoundError(doc_id)
            self._data[doc_id] = {**row, **fields}
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()

    # ------------------------------------------------------------------
    # failure_generation write side (Phase 1)
    # ------------------------------------------------------------------

    def _read_ctrl_locked(self) -> dict[str, Any]:
        """Read + validate the control-plane doc; caller holds the lock.

        Raises a control-plane error instead of degrading: a missing or
        version-mismatched marker on a workspace this backend already
        initialized is corruption, never "new LEGACY workspace".
        """
        ctrl = load_json(self._ctrl_file_name)
        if not isinstance(ctrl, dict) or ctrl.get("schema_version") != (
            self._CTRL_SCHEMA_VERSION
        ):
            raise StorageControlPlaneError(
                f"[{self.workspace}] failure-generation marker missing or "
                f"version-mismatched for {self.namespace}; refusing (never "
                "degrades to LEGACY full-snapshot behaviour)"
            )
        return ctrl

    async def get_failure_generation_mode(self) -> FailureGenerationMode:
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            ctrl = load_json(self._ctrl_file_name)
        if not isinstance(ctrl, dict):
            return FailureGenerationMode.MIGRATING
        if ctrl.get("schema_version") != self._CTRL_SCHEMA_VERSION:
            return FailureGenerationMode.MIGRATING
        try:
            return FailureGenerationMode(str(ctrl.get("mode")))
        except ValueError:
            return FailureGenerationMode.MIGRATING

    async def reserve_failure_generation(self) -> int:
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            return self._reserve_generation_locked()

    def _reserve_generation_locked(self) -> int:
        ctrl = self._read_ctrl_locked()
        try:
            counter = int(ctrl.get("failure_generation_counter") or 0)
        except (TypeError, ValueError) as e:
            raise StorageControlPlaneError(
                f"[{self.workspace}] failure-generation counter corrupt: {e}"
            ) from e
        counter += 1
        ctrl["failure_generation_counter"] = counter
        # Durability no weaker than the FAILED status write: persist the
        # reservation before any row can publish it. A crash after this
        # write leaves a permanent hole — allowed; reuse is not.
        write_json(ctrl, self._ctrl_file_name)
        return counter

    async def mark_doc_failed(self, doc_id: str, fields: dict[str, Any]) -> int | None:
        """FAILED transition funnel: reserve + publish under ONE lock.

        The single storage lock makes reserve-before-publish atomic here
        (window → 0). Idempotent per attempt: an already-FAILED row whose
        ``failure_attempt_id`` equals the current attempt keeps its
        generation; anything else assigns a fresh one. A caller-supplied
        ``created_at`` is ignored for existing rows (immutable sort key);
        a missing row is conditionally created (enqueue/parse errors can
        fail before the PENDING row landed).
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            existing = self._data.get(doc_id)
            if isinstance(existing, dict):
                current_attempt = existing.get("processing_attempt_id") or fields.get(
                    "processing_attempt_id"
                )
            else:
                current_attempt = fields.get("processing_attempt_id")
            if (
                isinstance(existing, dict)
                and str(existing.get("status")) == DocStatus.FAILED.value
                and current_attempt
                and existing.get("failure_attempt_id") == current_attempt
            ):
                try:
                    return int(existing.get("failure_generation") or 0)
                except (TypeError, ValueError):
                    return 0
            generation = self._reserve_generation_locked()
            row = {**existing, **fields} if isinstance(existing, dict) else dict(fields)
            if isinstance(existing, dict) and "created_at" in existing:
                row["created_at"] = existing["created_at"]
            row["status"] = DocStatus.FAILED.value
            row["failure_generation"] = generation
            if current_attempt:
                row["failure_attempt_id"] = current_attempt
                row.setdefault("processing_attempt_id", current_attempt)
            if "chunks_list" not in row:
                row["chunks_list"] = []
            self._data[doc_id] = row
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()
        return generation

    async def ensure_processing_attempt_id(self, doc_id: str) -> str:
        """Atomic (single-lock) mint-or-reuse of the row's attempt id."""
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        minted = False
        async with self._storage_lock:
            row = self._data.get(doc_id)
            if row is None:
                raise StorageRecordNotFoundError(doc_id)
            attempt = row.get("processing_attempt_id")
            if not attempt:
                attempt = uuid.uuid4().hex
                self._data[doc_id] = {**row, "processing_attempt_id": attempt}
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                minted = True
        if minted:
            await self.index_done_callback()
        return str(attempt)

    async def drop(self) -> dict[str, str]:
        """Clear shared memory and immediately persist the empty state.

        This method will:
            1. Clear the shared ``self._data`` dict under
               ``_storage_lock`` (visible to all processes immediately).
            2. ``set_all_update_flags`` so every process knows there is
               dirty state pending persistence.
            3. Call ``index_done_callback`` synchronously to flush the
               empty state to disk and clear the dirty flags.

        Caller contract:
            ``drop`` is destructive and **not** serialized by this
            storage class. The caller must hold the pipeline ``busy``
            reservation (the ``/documents/clear`` endpoint does this)
            before invoking it. See class docstring,
            *Non-pipeline write paths*.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                self._data.clear()
                await set_all_update_flags(self.namespace, workspace=self.workspace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
