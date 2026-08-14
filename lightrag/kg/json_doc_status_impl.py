from dataclasses import dataclass
import copy
import hashlib
import heapq
import json
import os
from typing import Any, ClassVar, Sequence, Union, final

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
    SourceAbsent,
    SourceConflict,
    SourceConflictPage,
    SourceConflictRepairResult,
    SourceConflictSummary,
    SourceResolution,
    SourceUnique,
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
    SourceConflictRepairCASError,
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

    supports_strict_point_reads: ClassVar[bool] = True

    # Bounded upper limit on the sample of conflicting doc IDs surfaced by the
    # source-conflict listing/repair APIs — never materialize the whole set.
    _CONFLICT_SAMPLE_CAP: ClassVar[int] = 32

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
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

        reap_orphan_tmp_files(self._file_name, self.workspace or "_")

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
                    ordered_results.append(
                        copy.deepcopy(data) if isinstance(data, dict) else data
                    )
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
                    result[k] = self._doc_processing_status_from_row(v)
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
        """Get all documents with a specific track_id.

        Relaxed read path (no ``strict`` variant): this backs the
        ``/documents/track_status`` display endpoint, not the scheduling
        control plane, so an unusable row is skipped-and-logged rather than
        failing the whole listing. That matters because ``track_id`` groups a
        whole upload BATCH — raising here would hide every healthy sibling
        document behind one bad row.
        """
        result = {}
        async with self._storage_lock:
            for k, v in self._data.items():
                if not isinstance(v, dict):
                    # Logged, never silently dropped: a non-mapping record is
                    # corruption an operator needs to see, and this is the only
                    # place it surfaces for this track_id. The sibling scans
                    # report it too (``get_docs_by_statuses`` reaches it via the
                    # TypeError from its in-``try`` ``v["status"]`` subscript).
                    logger.error(
                        f"[{self.workspace}] doc_status record {k} is not a mapping"
                    )
                    continue
                if v.get("track_id") != track_id:
                    continue
                try:
                    result[k] = self._doc_processing_status_from_row(v)
                except (KeyError, TypeError) as e:
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
            data = self._data.get(id)
            return copy.deepcopy(data) if isinstance(data, dict) else data

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

        # Everything below — filtering, hydratability validation, sorting,
        # page selection, and the final page's deep-copy hydration — runs
        # inside ONE uninterrupted lock hold. Nothing in this block awaits,
        # so no concurrent upsert/delete can land between the scan and the
        # final hydration; splitting these into separate lock acquisitions
        # let a status update/delete land in that gap, producing a response
        # that mixed two snapshots (e.g. a status=pending filter returning an
        # already-updated "processed" row, or overcounting a row that a
        # concurrent delete then removed before it could be hydrated).
        #
        # The scan and validation below are still cheap per document — no
        # nested-structure copying, just a throwaway shallow
        # DocProcessingStatus construction (see ``_row_is_hydratable``) — so
        # holding the lock across them does not reintroduce the "deep-copy
        # the whole corpus" cost. Only the final page (bounded by page_size,
        # <=200) is ever deep-copied.
        async with self._storage_lock:
            projections: list[tuple[Any, str]] = []
            for doc_id, doc_data in self._data.items():
                if (
                    status_filter_values is not None
                    and doc_data.get("status") not in status_filter_values
                ):
                    continue
                # Validate via a real (throwaway) construction attempt rather
                # than checking a fixed set of required-field names: a row
                # with an extra/unexpected field fails construction too, and
                # must be excluded from total_count for the same reason a
                # missing field is — otherwise total_count could count rows
                # the page-hydration step below can never actually return.
                if not self._row_is_hydratable(doc_id, doc_data):
                    continue

                if sort_field == "id":
                    sort_key = doc_id
                elif sort_field == "file_path":
                    # Use pinyin sorting for file_path field to support Chinese
                    # characters; "no-file-path" mirrors the fallback applied
                    # during hydration (_doc_processing_status_from_row) so
                    # sort order matches what the hydrated page would use.
                    sort_key = get_pinyin_sort_key(
                        doc_data.get("file_path") or "no-file-path"
                    )
                else:
                    sort_key = doc_data.get(sort_field, "")
                projections.append((sort_key, doc_id))

            total_count = len(projections)

            reverse_sort = sort_direction.lower() == "desc"
            projections.sort(key=lambda t: t[0], reverse=reverse_sort)

            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_doc_ids = [doc_id for _, doc_id in projections[start_idx:end_idx]]

            # Deep-copy and hydrate ONLY the requested page. Each doc_id here
            # was validated as hydratable above in this same uninterrupted
            # critical section, so the row is guaranteed present and usable.
            paginated_docs: list[tuple[str, DocProcessingStatus]] = []
            for doc_id in page_doc_ids:
                row = self._data[doc_id]
                # Pagination never surfaces chunks_list (DocStatusResponse
                # doesn't expose it — see the PG backend's paged CTE, which
                # excludes the column for the same reason), so drop it before
                # hydration instead of deep-copying a potentially large
                # chunk-id list only to discard it.
                row_without_chunks = {
                    k: v for k, v in row.items() if k != "chunks_list"
                }
                doc_status = self._doc_processing_status_from_row(row_without_chunks)
                paginated_docs.append((doc_id, doc_status))

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
                    return (
                        copy.deepcopy(doc_data)
                        if isinstance(doc_data, dict)
                        else doc_data
                    )

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
                    return doc_id, copy.deepcopy(doc_data)
        return None

    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> SourceResolution:
        """Typed source resolution (see base contract).

        Collects up to two PRIMARY (``metadata.is_duplicate != true``) rows for
        the canonical basename and maps 0/1/≥2 → Absent/Unique/Conflict. The
        in-memory scan has no transport failure surface, so a returned
        ``SourceAbsent`` IS confirmed absence (uninitialized storage raises).
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        if not canonical_source_key or canonical_source_key == "unknown_source":
            return SourceAbsent()
        candidates: list[tuple[str, dict[str, Any]]] = []
        async with self._storage_lock:
            for doc_id, row in self._data.items():
                if not isinstance(row, dict):
                    continue
                if row.get("file_path") == canonical_source_key and not (
                    self._is_duplicate_row(row)
                ):
                    candidates.append((doc_id, dict(row)))
                    if len(candidates) >= 2:
                        break
        if not candidates:
            return SourceAbsent()
        if len(candidates) == 1:
            doc_id, row = candidates[0]
            return SourceUnique(
                doc_id=doc_id,
                doc=self._scheduling_record_from_raw(doc_id, row),
            )
        return SourceConflict(
            candidate_count=None,
            sample_doc_ids=tuple(sorted(doc_id for doc_id, _ in candidates)),
        )

    async def get_by_id_strict(self, id: str) -> Union[dict[str, Any], None]:
        """Strict point read: complete-or-raise (base contract).

        In-memory shared dict — a miss is a confirmed absence; the only
        failure surface is uninitialized storage, which raises.
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            row = self._data.get(id)
        return copy.deepcopy(row) if isinstance(row, dict) else row

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find an existing record whose content_hash field matches.

        ``exclude_doc_id`` skips that row AND any row that merely points at it
        (``is_duplicate`` naming it as ``original_doc_id``) so the duplicate
        check can neither be answered with the row being processed nor with a
        record of that row (see base contract). JSON has no content_hash index,
        so this is a single full dict scan either way — both exclusions are
        predicates in the same pass, which is also why skipping a pointer row
        cannot truncate the search: the scan keeps looking for the earliest
        remaining holder.

        Returns the EARLIEST match by ``(created_at, id)`` rather than the
        first in dict order: insertion order is not creation order once rows
        are reloaded or rewritten, and the base contract's "earliest other
        holder" is what keeps the ``original_doc_id`` recorded on a duplicate
        stable across runs.
        """
        if not content_hash:
            return None
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        best: tuple[tuple[str, str], str, dict[str, Any]] | None = None
        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                if doc_id == exclude_doc_id:
                    continue
                if self._row_points_at_as_duplicate(doc_data, exclude_doc_id):
                    continue
                if doc_data.get("content_hash") != content_hash:
                    continue
                sort_key = self._row_sort_key(doc_id, doc_data)
                if best is None or sort_key < best[0]:
                    best = (sort_key, doc_id, doc_data)
        if best is None:
            return None
        return best[1], copy.deepcopy(best[2]) if isinstance(best[2], dict) else best[2]

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
            # doc_status holds ``status`` as EITHER a DocStatus member (the
            # in-memory pipeline path) or its raw string value (JSON-reloaded).
            # ``DocStatus(str(member))`` would break on a member (str(enum) is
            # "DocStatus.PENDING", not "pending"), so normalise enum-tolerantly
            # — same as get_docs_by_statuses' membership test and the base
            # _scheduling_record_from_raw helper.
            raw_status = row["status"]
            status = (
                raw_status
                if isinstance(raw_status, DocStatus)
                else DocStatus(raw_status)
            )
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

    @staticmethod
    def _normalize_status_row(data: dict[str, Any]) -> dict[str, Any]:
        """Field normalisation shared by hydration and validation: pop the
        deprecated ``content`` field, default missing/falsy ``file_path``,
        default missing ``metadata``/``error_msg``. Mutates and returns
        ``data`` in place — the caller decides whether ``data`` is a deep
        copy (safe to expose) or a shallow/throwaway one (validation only,
        see ``_row_is_hydratable``)."""
        data.pop("content", None)  # deprecated inline content, never a status field
        if not data.get("file_path"):
            data["file_path"] = "no-file-path"
        data.setdefault("metadata", {})
        data.setdefault("error_msg", None)
        return data

    @classmethod
    def _doc_processing_status_from_row(
        cls, row: dict[str, Any]
    ) -> DocProcessingStatus:
        """Normalise a raw doc_status row into a FULL DocProcessingStatus.

        Single source of the raw → status construction shared by
        ``get_docs_by_statuses``, ``get_docs_paginated`` and the
        ``get_full_docs_by_ids`` hydration path. Raises ``KeyError``/
        ``TypeError`` on a malformed row (missing required fields); the
        caller decides strict (raise) vs relaxed (skip). Fields the
        dataclass does not declare are tolerated, not treated as malformed
        — see ``DocProcessingStatus.from_stored``.

        Deep-copies ``row`` before use: the returned ``DocProcessingStatus``
        carries nested mutable fields (``metadata``, ``chunks_list``) that
        must not alias the live storage row, or a caller mutating them would
        corrupt persisted state without going through ``upsert`` — the same
        class of bug fixed for the other read paths.
        """
        data = cls._normalize_status_row(copy.deepcopy(row))
        return DocProcessingStatus.from_stored(data)

    def _row_is_hydratable(self, doc_id: str, row: dict[str, Any]) -> bool:
        """True if ``row`` can be hydrated into a full DocProcessingStatus.

        Used by ``get_docs_paginated``'s validation pass to keep
        ``total_count`` consistent with what the page-hydration step can
        actually return: a row missing a required field would fail
        construction — counting it as present without confirming that
        would let ``total_count`` overstate what pages can actually
        deliver. (Fields the dataclass does not declare are tolerated,
        not treated as malformed.)

        Attempts the SAME normalisation + construction as
        ``_doc_processing_status_from_row``, but on a shallow copy that is
        discarded immediately (never returned to a caller), so it validates
        constructibility without paying the deep-copy cost across every
        matching document — only the final page (bounded by page_size) is
        ever deep-copied.
        """
        try:
            DocProcessingStatus.from_stored(self._normalize_status_row(dict(row)))
            return True
        except (KeyError, TypeError) as e:
            logger.error(f"[{self.workspace}] Error processing document {doc_id}: {e}")
            return False

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        """Bounded keyset page over the in-memory store.

        Selection uses a bounded heap (``heapq.nsmallest``) so page memory is
        O(limit) even though each page re-scans the resident dict — the
        O(total) residency itself is this backend's documented boundary.

        Consumed-position contract: every selected candidate is consumed —
        returned or skipped as unusable in relaxed mode — so the cursor
        advances past filtered pages instead of re-reading them; fewer
        candidates than ``limit`` proves exhaustion (CURSOR_END).

        Strict failure surface (backend divergence, by design): because this
        backend candidate-selects by scanning the WHOLE resident dict, a
        ``strict=True`` page raises on a malformed (non-mapping) record even
        when that record's status is NOT part of the requested sweep —
        server-side backends (PG/Mongo/OpenSearch/Redis) never see
        out-of-scope rows and would not fail on the same corruption. An
        operator whose JSON-backed sweep fails closed should inspect the
        whole store, not just the swept statuses.
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

        def _candidates():
            for doc_id, row in self._data.items():
                if not isinstance(row, dict):
                    if strict:
                        raise TypeError(f"doc_status record {doc_id} is not a mapping")
                    continue
                # Membership (NOT str()) so a DocStatus member matches by its
                # str-enum value — str(member) is "DocStatus.PENDING", which
                # would wrongly exclude the in-memory pipeline's enum status.
                if row.get("status") not in status_values:
                    continue
                key = self._row_sort_key(doc_id, row)
                if last_key is not None and key <= last_key:
                    continue
                yield key, doc_id, row

        async with self._storage_lock:
            selected = heapq.nsmallest(limit, _candidates(), key=lambda t: t[0])
            docs: dict[str, DocSchedulingRecord] = {}
            for key, doc_id, row in selected:
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
                # Membership (NOT str()) so an enum status member matches by
                # its str-enum value (see get_docs_by_statuses_page).
                if row.get("status") in status_values:
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
    # Strict batch read
    # ------------------------------------------------------------------

    async def get_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocSchedulingRecord]:
        """Batch strict read (see base contract).

        In-memory dict lookup: a missing id is a confirmed absence and is
        omitted. ``strict=True`` raises on a malformed (non-mapping) record;
        uninitialized storage always raises.
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        result: dict[str, DocSchedulingRecord] = {}
        async with self._storage_lock:
            for doc_id in doc_ids:
                row = self._data.get(doc_id)
                if row is None:
                    continue  # confirmed absent
                if not isinstance(row, dict):
                    if strict:
                        raise TypeError(f"doc_status record {doc_id} is not a mapping")
                    continue
                record = self._scheduling_record_from_row(doc_id, row, strict=strict)
                if record is not None:
                    result[doc_id] = record
        return result

    async def get_full_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocProcessingStatus]:
        """Batch hydration to full DocProcessingStatus (see base contract).

        In-memory dict lookup: a missing id is a confirmed absence and is
        omitted. ``strict=True`` raises on a malformed record (mirrors
        ``get_docs_by_statuses(strict=True)``); uninitialized storage always
        raises.
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        result: dict[str, DocProcessingStatus] = {}
        async with self._storage_lock:
            for doc_id in doc_ids:
                row = self._data.get(doc_id)
                if row is None:
                    continue  # confirmed absent
                try:
                    result[doc_id] = self._doc_processing_status_from_row(row)
                except (KeyError, TypeError) as e:
                    logger.error(
                        f"[{self.workspace}] Unusable doc_status row hydrating "
                        f"{doc_id}: {e}"
                    )
                    if strict:
                        raise
                    continue
        return result

    # ------------------------------------------------------------------
    # Source-conflict listing and explicit CAS repair
    # ------------------------------------------------------------------

    @staticmethod
    def _conflict_fingerprint(sorted_doc_ids: list[str]) -> str:
        """Deterministic digest over candidate doc IDs in stable sort order."""
        digest = hashlib.sha256()
        for doc_id in sorted_doc_ids:
            digest.update(doc_id.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _primary_candidates_locked(self, canonical_source_key: str) -> list[str]:
        """Sorted primary (non-duplicate) doc IDs for a canonical key. The
        caller holds ``_storage_lock``."""
        return sorted(
            doc_id
            for doc_id, row in self._data.items()
            if isinstance(row, dict)
            and row.get("file_path") == canonical_source_key
            and not self._is_duplicate_row(row)
        )

    @staticmethod
    def _decode_conflict_cursor(opaque: str) -> str:
        try:
            key = json.loads(opaque)
            if not isinstance(key, str):
                raise ValueError("conflict cursor must be a string")
        except (ValueError, TypeError) as e:
            raise StorageControlPlaneError(
                f"Malformed source-conflict cursor for JsonDocStatusStorage: {e}"
            ) from e
        return key

    async def list_source_conflicts_page(
        self,
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
    ) -> SourceConflictPage:
        """Page canonical source keys with >1 primary candidate (see base)."""
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if position is CURSOR_END:
            return SourceConflictPage(conflicts=(), next_position=CURSOR_END)
        last_key: str | None = None
        if isinstance(position, CursorAfter):
            last_key = self._decode_conflict_cursor(position.opaque)
        async with self._storage_lock:
            groups: dict[str, list[str]] = {}
            for doc_id, row in self._data.items():
                if not isinstance(row, dict):
                    continue
                file_path = row.get("file_path")
                if not isinstance(file_path, str) or file_path in (
                    "",
                    "unknown_source",
                    "no-file-path",
                ):
                    continue
                if self._is_duplicate_row(row):
                    continue
                groups.setdefault(file_path, []).append(doc_id)
        conflict_keys = sorted(k for k, v in groups.items() if len(v) >= 2)
        page_keys = [k for k in conflict_keys if last_key is None or k > last_key][
            :limit
        ]
        conflicts = tuple(
            SourceConflictSummary(
                canonical_source_key=k,
                candidate_count=len(groups[k]),
                sample_doc_ids=tuple(sorted(groups[k])[: self._CONFLICT_SAMPLE_CAP]),
            )
            for k in page_keys
        )
        if len(page_keys) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            next_position = CursorAfter(json.dumps(page_keys[-1], ensure_ascii=False))
        return SourceConflictPage(conflicts=conflicts, next_position=next_position)

    async def repair_source_conflict(
        self,
        canonical_source_key: str,
        *,
        primary_doc_id: str,
        expected_candidate_count: int,
        expected_candidate_fingerprint: str,
        dry_run: bool = True,
    ) -> SourceConflictRepairResult:
        """Demote all-but-one primary to duplicate, CAS-guarded (see base).

        The single storage lock makes the re-read + CAS + demotion atomic:
        dry-run reports the current count/fingerprint; commit refuses when
        they no longer match the operator-echoed expectation.
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            candidates = self._primary_candidates_locked(canonical_source_key)
            count = len(candidates)
            fingerprint = self._conflict_fingerprint(candidates)
            if primary_doc_id not in candidates:
                raise ValueError(
                    f"primary_doc_id {primary_doc_id!r} is not a current primary "
                    f"candidate for {canonical_source_key!r}"
                )
            demoted = [d for d in candidates if d != primary_doc_id]
            if dry_run:
                return SourceConflictRepairResult(
                    canonical_source_key=canonical_source_key,
                    primary_doc_id=primary_doc_id,
                    candidate_count=count,
                    fingerprint=fingerprint,
                    demoted_sample_doc_ids=tuple(demoted[: self._CONFLICT_SAMPLE_CAP]),
                    committed=False,
                )
            if (
                count != expected_candidate_count
                or fingerprint != expected_candidate_fingerprint
            ):
                raise SourceConflictRepairCASError(
                    f"[{self.workspace}] source-conflict repair CAS failed for "
                    f"{canonical_source_key!r}: candidate set changed "
                    f"(count {count} vs {expected_candidate_count})"
                )
            for doc_id in demoted:
                row = dict(self._data[doc_id])
                metadata = dict(row.get("metadata") or {})
                metadata["is_duplicate"] = True
                metadata["original_doc_id"] = primary_doc_id
                row["metadata"] = metadata
                self._data[doc_id] = row
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()
        return SourceConflictRepairResult(
            canonical_source_key=canonical_source_key,
            primary_doc_id=primary_doc_id,
            candidate_count=count,
            fingerprint=fingerprint,
            demoted_sample_doc_ids=tuple(demoted[: self._CONFLICT_SAMPLE_CAP]),
            committed=True,
        )

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
