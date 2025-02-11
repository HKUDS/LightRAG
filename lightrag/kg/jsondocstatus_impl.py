"""
JsonDocStatus Storage Module
=======================

This module provides a storage interface for graphs using NetworkX, a popular Python library for creating, manipulating, and studying the structure, dynamics, and functions of complex networks.

The `NetworkXStorage` class extends the `BaseGraphStorage` class from the LightRAG library, providing methods to load, save, manipulate, and query graphs using NetworkX.

Author: lightrag team
Created: 2024-01-25
License: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Version: 1.0.0

Dependencies:
    - NetworkX
    - NumPy
    - LightRAG
    - graspologic

Features:
    - Load and save graphs in various formats (e.g., GEXF, GraphML, JSON)
    - Query graph nodes and edges
    - Calculate node and edge degrees
    - Embed nodes using various algorithms (e.g., Node2Vec)
    - Remove nodes and edges from the graph

Usage:
    from lightrag.storage.networkx_storage import NetworkXStorage

"""

from dataclasses import dataclass
import os
from typing import Any, Union

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


@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data: dict[str, Any] = load_json(self._file_name) or {}
        logger.info(f"Loaded document status storage with {len(self._data)} records")

    async def filter_keys(self, data: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        return set(data) - set(self._data.keys())

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for id in ids:
            data = self._data.get(id, None)
            if data:
                result.append(data)
        return result

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status: 0 for status in DocStatus}
        for doc in self._data.values():
            counts[doc["status"]] += 1
        return counts

    async def get_failed_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        return {
            k: DocProcessingStatus(**v)
            for k, v in self._data.items()
            if v["status"] == DocStatus.FAILED
        }

    async def get_pending_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        return {
            k: DocProcessingStatus(**v)
            for k, v in self._data.items()
            if v["status"] == DocStatus.PENDING
        }

    async def get_processed_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all processed documents"""
        return {
            k: DocProcessingStatus(**v)
            for k, v in self._data.items()
            if v["status"] == DocStatus.PROCESSED
        }
    
    async def get_processing_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all processing documents"""
        return {
            k: DocProcessingStatus(**v)
            for k, v in self._data.items()
            if v["status"] == DocStatus.PROCESSING
        }

    async def index_done_callback(self):
        """Save data to file after indexing"""
        write_json(self._data, self._file_name)

    async def upsert(self, data: dict[str, Any]) -> None:
        """Update or insert document status

        Args:
            data: Dictionary of document IDs and their status data
        """
        self._data.update(data)
        await self.index_done_callback()

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        return self._data.get(id)

    async def delete(self, doc_ids: list[str]):
        """Delete document status by IDs"""
        for doc_id in doc_ids:
            self._data.pop(doc_id, None)
        await self.index_done_callback()
