from dataclasses import dataclass
import os
from typing import Any, Union, final

from lightrag.base import (
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.file_atomic import reap_orphan_tmp_files
from lightrag.utils import (
    _cooperative_yield,
    load_json,
    logger,
    validate_workspace,
    write_json,
    get_pinyin_sort_key,
)
from lightrag.exceptions import StorageNotInitializedError
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
    """

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
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching any of the given statuses in a single pass.

        Acquires the storage lock once and scans the in-memory dict once,
        filtering against a set of status values.  More efficient than N separate
        get_docs_by_status() calls, which would acquire the lock N times and scan
        the data N times.
        """
        if not statuses:
            return {}
        status_values = {s.value for s in statuses}
        result = {}
        async with self._storage_lock:
            for k, v in self._data.items():
                if v["status"] not in status_values:
                    continue
                try:
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
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )
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

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find an existing record whose canonical basename matches.

        The caller is responsible for passing an already-canonical basename.
        Stored ``file_path`` values are canonicalized by the business layer, so
        this lookup intentionally performs an exact match only.
        """
        if not basename:
            return None
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        if basename == "unknown_source":
            return None
        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                if doc_data.get("file_path") == basename:
                    return doc_id, doc_data
        return None

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
