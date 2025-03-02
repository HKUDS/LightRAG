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
    try_initialize_namespace,
)


@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._storage_lock = get_storage_lock()
        self._data = None

    async def initialize(self):
        """Initialize storage data"""
        # check need_init must before get_namespace_data
        need_init = try_initialize_namespace(self.namespace)
        self._data = await get_namespace_data(self.namespace)
        if need_init:
            loaded_data = load_json(self._file_name) or {}
            async with self._storage_lock:
                self._data.update(loaded_data)
                logger.info(f"Load KV {self.namespace} with {len(loaded_data)} data")

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            data_dict = (
                dict(self._data) if hasattr(self._data, "_getvalue") else self._data
            )
            write_json(data_dict, self._file_name)

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
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        async with self._storage_lock:
            left_data = {k: v for k, v in data.items() if k not in self._data}
            self._data.update(left_data)

    async def delete(self, ids: list[str]) -> None:
        async with self._storage_lock:
            for doc_id in ids:
                self._data.pop(doc_id, None)
        await self.index_done_callback()
