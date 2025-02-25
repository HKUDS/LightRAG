import asyncio
import os
from dataclasses import dataclass
from typing import Any, final
import threading
from multiprocessing import Manager

from lightrag.base import (
    BaseKVStorage,
)
from lightrag.utils import (
    load_json,
    logger,
    write_json,
)

# Global variables for shared memory management
_init_lock = threading.Lock()
_manager = None
_shared_kv_data = None


def _get_manager():
    """Get or create the global manager instance"""
    global _manager, _shared_kv_data
    with _init_lock:
        if _manager is None:
            try:
                _manager = Manager()
                _shared_kv_data = _manager.dict()
            except Exception as e:
                logger.error(f"Failed to initialize shared memory manager: {e}")
                raise RuntimeError(f"Shared memory initialization failed: {e}")
    return _manager


@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._lock = asyncio.Lock()
        
        # Ensure manager is initialized
        _get_manager()
        
        # Get or create namespace data
        if self.namespace not in _shared_kv_data:
            with _init_lock:
                if self.namespace not in _shared_kv_data:
                    try:
                        initial_data = load_json(self._file_name) or {}
                        _shared_kv_data[self.namespace] = initial_data
                    except Exception as e:
                        logger.error(f"Failed to initialize shared data for namespace {self.namespace}: {e}")
                        raise RuntimeError(f"Shared data initialization failed: {e}")
        
        try:
            self._data = _shared_kv_data[self.namespace]
            logger.info(f"Load KV {self.namespace} with {len(self._data)} data")
        except Exception as e:
            logger.error(f"Failed to access shared memory: {e}")
            raise RuntimeError(f"Cannot access shared memory: {e}")

    async def index_done_callback(self) -> None:
        write_json(self._data, self._file_name)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return self._data.get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [
            (
                {k: v for k, v in self._data[id].items()}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._data.pop(doc_id, None)
        await self.index_done_callback()
