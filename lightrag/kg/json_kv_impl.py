import os
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import (
    BaseKVStorage,
)
from lightrag.utils import (
    load_json,
    logger,
    write_json,
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
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
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

    async def initialize(self):
        """Initialize storage data"""
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
                    # Migrate legacy cache structure if needed
                    if self.namespace.endswith("_cache"):
                        loaded_data = await self._migrate_legacy_cache_structure(
                            loaded_data
                        )

                    self._data.update(loaded_data)
                    data_count = len(loaded_data)

                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} KV load {self.namespace} with {data_count} records"
                    )

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )

                # Calculate data count - all data is now flattened
                data_count = len(data_dict)

                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} KV writting {data_count} records to {self.namespace}"
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

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            result = self._data.get(id)
            if result:
                # Create a copy to avoid modifying the original data
                result = dict(result)
                # Ensure time fields are present, provide default values for old data
                result.setdefault("create_time", 0)
                result.setdefault("update_time", 0)
                # Ensure _id field contains the clean ID
                result["_id"] = id
            return result

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            results = []
            for id in ids:
                data = self._data.get(id, None)
                if data:
                    # Create a copy to avoid modifying the original data
                    result = {k: v for k, v in data.items()}
                    # Ensure time fields are present, provide default values for old data
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    # Ensure _id field contains the clean ID
                    result["_id"] = id
                    results.append(result)
                else:
                    results.append(None)
            return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed
        """
        if not data:
            return

        import time

        current_time = int(time.time())  # Get current Unix timestamp

        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonKVStorage")
        async with self._storage_lock:
            # Add timestamps to data based on whether key exists
            for k, v in data.items():
                # For text_chunks namespace, ensure llm_cache_list field exists
                if self.namespace.endswith("text_chunks"):
                    if "llm_cache_list" not in v:
                        v["llm_cache_list"] = []

                # Add timestamps based on whether key exists
                if k in self._data:  # Key exists, only update update_time
                    v["update_time"] = current_time
                else:  # New key, set both create_time and update_time
                    v["create_time"] = current_time
                    v["update_time"] = current_time

                v["_id"] = k

            self._data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def delete(self, ids: list[str]) -> None:
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
            for doc_id in ids:
                result = self._data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def is_empty(self) -> bool:
        """Check if the storage is empty

        Returns:
            bool: True if storage contains no data, False otherwise
        """
        async with self._storage_lock:
            return len(self._data) == 0

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources
           This action will persistent the data to disk immediately.

        This method will:
        1. Clear all data from memory
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
                await set_all_update_flags(self.namespace, workspace=self.workspace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def _migrate_legacy_cache_structure(self, data: dict) -> dict:
        """Migrate legacy nested cache structure to flattened structure

        Args:
            data: Original data dictionary that may contain legacy structure

        Returns:
            Migrated data dictionary with flattened cache keys (sanitized if needed)
        """
        from lightrag.utils import generate_cache_key

        # Early return if data is empty
        if not data:
            return data

        # Check first entry to see if it's already in new format
        first_key = next(iter(data.keys()))
        if ":" in first_key and len(first_key.split(":")) == 3:
            # Already in flattened format, return as-is
            return data

        migrated_data = {}
        migration_count = 0

        for key, value in data.items():
            # Check if this is a legacy nested cache structure
            if isinstance(value, dict) and all(
                isinstance(v, dict) and "return" in v for v in value.values()
            ):
                # This looks like a legacy cache mode with nested structure
                mode = key
                for cache_hash, cache_entry in value.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    migrated_data[flattened_key] = cache_entry
                    migration_count += 1
            else:
                # Keep non-cache data or already flattened cache data as-is
                migrated_data[key] = value

        if migration_count > 0:
            logger.info(
                f"[{self.workspace}] Migrated {migration_count} legacy cache entries to flattened structure"
            )
            # Persist migrated data immediately and check if sanitization was applied
            needs_reload = write_json(migrated_data, self._file_name)

            # If data was sanitized during write, reload cleaned data
            if needs_reload:
                logger.info(
                    f"[{self.workspace}] Reloading sanitized migration data for {self.namespace}"
                )
                cleaned_data = load_json(self._file_name)
                if cleaned_data is not None:
                    return cleaned_data  # Return cleaned data to update shared memory

        return migrated_data

    async def finalize(self):
        """Finalize storage resources
        Persistence cache data to disk before exiting
        """
        if self.namespace.endswith("_cache"):
            await self.index_done_callback()
