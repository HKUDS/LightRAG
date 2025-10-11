from dataclasses import dataclass
import os
from typing import Any, Union, final

from lightrag.base import (
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.utils import (
    load_json,
    logger,
    write_json,
    get_pinyin_sort_key,
)
from lightrag.exceptions import StorageNotInitializedError
from .shared_storage import (
    get_namespace_data,
    get_storage_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)


@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        
        # Get composite workspace (supports multi-tenant isolation)
        composite_workspace = self._get_composite_workspace()
        
        if composite_workspace and composite_workspace != "_":
            # Include composite workspace in the file path for data isolation
            # For multi-tenant: tenant_id:kb_id:workspace
            # For single-tenant: just workspace
            workspace_dir = os.path.join(working_dir, composite_workspace)
            self.final_namespace = f"{composite_workspace}_{self.namespace}"
        else:
            # Default behavior when workspace is empty
            workspace_dir = working_dir
            self.final_namespace = self.namespace
            composite_workspace = "_"

        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

    async def initialize(self):
        """Initialize storage data"""
        self._storage_lock = get_storage_lock()
        self.storage_updated = await get_update_flag(self.final_namespace)
        async with get_data_init_lock():
            # check need_init must before get_namespace_data
            need_init = await try_initialize_namespace(self.final_namespace)
            self._data = await get_namespace_data(self.final_namespace)
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
            for doc_id, doc in self._data.items():
                try:
                    status = doc.get("status")
                    if status in counts:
                        counts[status] += 1
                    else:
                        # Log warning for unknown status but don't fail
                        logger.warning(
                            f"[{self.workspace}] Unknown status '{status}' for document {doc_id}"
                        )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error counting status for document {doc_id}: {e}"
                    )
                    continue
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        result = {}
        async with self._storage_lock:
            for k, v in self._data.items():
                if v["status"] == status.value:
                    try:
                        # Make a copy of the data to avoid modifying the original
                        data = v.copy()
                        # Remove deprecated content field if it exists
                        data.pop("content", None)
                        # If file_path is not in data, use document id as file path
                        if "file_path" not in data:
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
                        # If file_path is not in data, use document id as file path
                        if "file_path" not in data:
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
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )
                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} doc status writting {len(data_dict)} records to {self.namespace}"
                )
                write_json(data_dict, self._file_name)
                await clear_all_update_flags(self.final_namespace)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed
        """
        if not data:
            return
        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            # Ensure chunks_list field exists for new documents
            for doc_id, doc_data in data.items():
                if "chunks_list" not in doc_data:
                    doc_data["chunks_list"] = []
            self._data.update(data)
            await set_all_update_flags(self.final_namespace)

        await self.index_done_callback()

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._storage_lock:
            return self._data.get(id)

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
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

        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                # Apply status filter
                if (
                    status_filter is not None
                    and doc_data.get("status") != status_filter.value
                ):
                    continue

                try:
                    # Prepare document data
                    data = doc_data.copy()
                    data.pop("content", None)
                    if "file_path" not in data:
                        data["file_path"] = "no-file-path"
                    if "metadata" not in data:
                        data["metadata"] = {}
                    if "error_msg" not in data:
                        data["error_msg"] = None

                    # Filter data to only include valid fields for DocProcessingStatus
                    # This prevents TypeError if extra fields are present in the JSON
                    valid_fields = DocProcessingStatus.__dataclass_fields__.keys()
                    filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                    
                    doc_status = DocProcessingStatus(**filtered_data)

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

                except (KeyError, TypeError, ValueError) as e:
                    logger.error(
                        f"[{self.workspace}] Error processing document {doc_id}: {e}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Unexpected error processing document {doc_id}: {e}"
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
        """Delete specific records from storage by their IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        async with self._storage_lock:
            any_deleted = False
            for doc_id in doc_ids:
                result = self._data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.final_namespace)

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

    async def get_doc_by_external_id(
        self, external_id: str
    ) -> Union[dict[str, Any], None]:
        """Get document by external ID for idempotency checks.

        Args:
            external_id: The external ID to search for (client-provided unique identifier)

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
            Returns the same format as get_by_id method
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with self._storage_lock:
            for doc_id, doc_data in self._data.items():
                if doc_data.get("external_id") == external_id:
                    return doc_data

        return None

    async def drop(self) -> dict[str, str]:
        """Drop all document status data from storage and clean up resources

        This method will:
        1. Clear all document status data from memory
        2. Update flags to notify other processes
        3. Trigger index_done_callback to save the empty state

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                self._data.clear()
                await set_all_update_flags(self.final_namespace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
