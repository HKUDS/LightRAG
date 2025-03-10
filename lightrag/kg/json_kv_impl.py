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
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

    async def initialize(self):
        """Initialize storage data"""
        self._storage_lock = get_storage_lock()
        self.storage_updated = await get_update_flag(self.namespace)
        async with get_data_init_lock():
            # check need_init must before get_namespace_data
            need_init = await try_initialize_namespace(self.namespace)
            self._data = await get_namespace_data(self.namespace)
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    self._data.update(loaded_data)

                    # Calculate data count based on namespace
                    if self.namespace.endswith("cache"):
                        # For cache namespaces, sum the cache entries across all cache types
                        data_count = sum(
                            len(first_level_dict)
                            for first_level_dict in loaded_data.values()
                            if isinstance(first_level_dict, dict)
                        )
                    else:
                        # For non-cache namespaces, use the original count method
                        data_count = len(loaded_data)

                    logger.info(
                        f"Process {os.getpid()} KV load {self.namespace} with {data_count} records"
                    )

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )

                # Calculate data count based on namespace
                if self.namespace.endswith("cache"):
                    # # For cache namespaces, sum the cache entries across all cache types
                    data_count = sum(
                        len(first_level_dict)
                        for first_level_dict in data_dict.values()
                        if isinstance(first_level_dict, dict)
                    )
                else:
                    # For non-cache namespaces, use the original count method
                    data_count = len(data_dict)

                logger.info(
                    f"Process {os.getpid()} KV writting {data_count} records to {self.namespace}"
                )
                write_json(data_dict, self._file_name)
                await clear_all_update_flags(self.namespace)

    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage

        Returns:
            Dictionary containing all stored data
        """
        async with self._storage_lock:
            return dict(self._data)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            return self._data.get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            return [
                (
                    {k: v for k, v in self._data[id].items()}
                    if self._data.get(id, None)
                    else None
                )
                for id in ids
            ]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        logger.info(f"Inserting {len(data)} records to {self.namespace}")
        async with self._storage_lock:
            self._data.update(data)
            await set_all_update_flags(self.namespace)

    async def delete(self, ids: list[str]) -> None:
        async with self._storage_lock:
            for doc_id in ids:
                self._data.pop(doc_id, None)
            await set_all_update_flags(self.namespace)
        await self.index_done_callback()
