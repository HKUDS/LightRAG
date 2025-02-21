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


@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data: dict[str, Any] = load_json(self._file_name) or {}
        logger.info(f"Loaded document status storage with {len(self._data)} records")

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        return set(keys) - set(self._data.keys())

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for id in ids:
            data = self._data.get(id, None)
            if data:
                result.append(data)
        return result

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status.value: 0 for status in DocStatus}
        for doc in self._data.values():
            counts[doc["status"]] += 1
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        result = {}
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
        write_json(self._data, self._file_name)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        self._data.update(data)
        await self.index_done_callback()

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        return self._data.get(id)

    async def delete(self, doc_ids: list[str]):
        for doc_id in doc_ids:
            self._data.pop(doc_id, None)
        await self.index_done_callback()

    async def drop(self) -> None:
        """Drop the storage"""
        self._data.clear()
