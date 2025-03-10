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
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

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
                    logger.info(
                        f"Process {os.getpid()} doc status load {self.namespace} with {len(loaded_data)} records"
                    )

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        async with self._storage_lock:
            for id in ids:
                data = self._data.get(id, None)
                if data:
                    result.append(data)
        return result

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status.value: 0 for status in DocStatus}
        async with self._storage_lock:
            for doc in self._data.values():
                counts[doc["status"]] += 1
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
                        # If content is missing, use content_summary as content
                        if "content" not in data and "content_summary" in data:
                            data["content"] = data["content_summary"]
                        result[k] = DocProcessingStatus(**data)
                    except KeyError as e:
                        logger.error(f"Missing required field for document {k}: {e}")
                        continue
        return result

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )
                logger.info(
                    f"Process {os.getpid()} doc status writting {len(data_dict)} records to {self.namespace}"
                )
                write_json(data_dict, self._file_name)
                await clear_all_update_flags(self.namespace)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        logger.info(f"Inserting {len(data)} records to {self.namespace}")
        async with self._storage_lock:
            self._data.update(data)
            await set_all_update_flags(self.namespace)

        await self.index_done_callback()

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._storage_lock:
            return self._data.get(id)

    async def delete(self, doc_ids: list[str]):
        async with self._storage_lock:
            for doc_id in doc_ids:
                self._data.pop(doc_id, None)
            await set_all_update_flags(self.namespace)
        await self.index_done_callback()

    async def drop(self) -> None:
        """Drop the storage"""
        async with self._storage_lock:
            self._data.clear()
            await set_all_update_flags(self.namespace)
        await self.index_done_callback()
