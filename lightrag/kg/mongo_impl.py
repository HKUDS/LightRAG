import os
import re
import json
import time
from dataclasses import dataclass, field
import numpy as np
import configparser
import asyncio

from typing import Any, Union, final

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..utils import logger, compute_mdhash_id, _cooperative_yield, merge_source_ids
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from .._version import __version__
from ..kg.shared_storage import get_data_init_lock, get_namespace_lock

import pipmaster as pm

if not pm.is_installed("pymongo"):
    pm.install("pymongo")

from pymongo import AsyncMongoClient  # type: ignore
from pymongo import UpdateOne  # type: ignore
from pymongo.asynchronous.database import AsyncDatabase  # type: ignore
from pymongo.asynchronous.collection import AsyncCollection  # type: ignore
from pymongo.operations import SearchIndexModel  # type: ignore
from pymongo.driver_info import DriverInfo  # type: ignore
from pymongo.errors import (  # type: ignore
    PyMongoError,
    DuplicateKeyError,
    BulkWriteError,
)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

GRAPH_BFS_MODE = os.getenv("MONGO_GRAPH_BFS_MODE", "bidirectional")

# Flush-time batching limits shared by every MongoDB upsert path
# (MongoVectorDBStorage, MongoKVStorage, MongoGraphStorage).
# The payload-byte budget is the primary limiter; the record-count caps are a
# secondary guard that only binds when individual records are small.
# Upsert and delete have separate count caps on purpose: upsert records each
# carry a full embedding vector and are far heavier than delete _ids, so the
# upsert batch count is kept much smaller than the delete one.
# MongoDB caps a single BSON document at 16MB and a single bulk command message
# at 48MB; a 16MB JSON estimate (which overestimates the real BSON size) keeps
# every bulk_write comfortably below the wire limit and bounds peak memory.
DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16MB
DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH = 128
DEFAULT_MONGO_DELETE_MAX_RECORDS_PER_BATCH = 1000

# Separator joining the sorted (src, tgt) pair into a single canonical edge_key
# string. \x1f (ASCII unit separator) cannot appear in normal entity names, so
# collisions are effectively impossible (same pragmatic choice as the PG impl's
# \x01-delimited advisory-lock key).
_EDGE_KEY_SEP = "\x1f"
# MongoDB duplicate-key error code, raised when an upsert insert races the
# unique edge_key index (another writer inserted the same edge first).
_DUPLICATE_KEY_CODE = 11000


def _canonical_edge_key(source_node_id: str, target_node_id: str) -> str:
    """Direction-independent key for an undirected edge.

    ``sep.join(sorted((src, tgt)))`` maps ``(A,B)`` and ``(B,A)`` to the same
    string, so a unique index on ``edge_key`` lets MongoDB reject the second of
    two racing inserts (the classic ``$or``-upsert duplicate gap) regardless of
    direction, while reads keep using the bidirectional ``$or`` filter.
    """
    lo, hi = sorted((source_node_id, target_node_id))
    return f"{lo}{_EDGE_KEY_SEP}{hi}"


def _estimate_doc_bytes(doc: Any) -> int:
    """Estimate a document's serialized byte size via compact JSON.

    JSON overestimates the real BSON size MongoDB writes (a JSON float string is
    far longer than the 8 bytes a BSON double encodes), so callers stay
    conservatively below server limits and never underestimate.

    This is a splitting *heuristic*, not the exact wire size: upsert callers pass
    only the dominant payload field (the ``$set`` body / ``update_doc``), not the
    full ``UpdateOne`` op (filter, ``$setOnInsert``, ``$or`` wrapper). Those extras
    are tiny next to an embedding/document body, and the 16MB estimate budget sits
    far under MongoDB's 48MB bulk-command limit, so the under-count is immaterial;
    the server stays the final arbiter.
    """
    return len(
        json.dumps(doc, ensure_ascii=False, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )


def _chunk_by_budget(
    items: list[Any],
    size_of,
    max_payload_bytes: int,
    max_records_per_batch: int,
) -> list[tuple[list[Any], int]]:
    """Split items into batches by estimated payload size (primary) and count.

    The byte budget is the primary limiter: items accumulate until adding the
    next one would exceed ``max_payload_bytes``, then a new batch starts.
    ``size_of(item)`` returns an item's estimated serialized byte size. A single
    item larger than the byte budget is emitted as its own batch rather than
    raising; the server stays the final arbiter. A non-positive limit disables
    that dimension. Returns ``(batch, summed_estimated_bytes)`` tuples (the
    estimate is used for logging).
    """
    if not items:
        return []

    payload_limit = max_payload_bytes if max_payload_bytes > 0 else float("inf")
    records_limit = max_records_per_batch if max_records_per_batch > 0 else float("inf")

    batches: list[tuple[list[Any], int]] = []
    current: list[Any] = []
    # JSON array overhead ("[]")
    current_bytes = 2

    for item in items:
        item_bytes = size_of(item)
        # If current batch not empty, a comma is needed before next element.
        separator_overhead = 1 if current else 0
        next_bytes = current_bytes + separator_overhead + item_bytes

        if current and (len(current) >= records_limit or next_bytes > payload_limit):
            batches.append((current, current_bytes))
            current = []
            current_bytes = 2
            next_bytes = current_bytes + item_bytes

        current.append(item)
        current_bytes = next_bytes

    if current:
        batches.append((current, current_bytes))

    return batches


def _resolve_upsert_batch_limits() -> tuple[int, int]:
    """Resolve flush-time upsert batching limits from env, with module defaults.

    Shared by every MongoDB upsert path so the byte/record caps that bound a
    single ``bulk_write`` are consistent across all of them. A non-positive
    value disables that splitting dimension.
    """
    max_payload_bytes = int(
        os.getenv(
            "MONGO_UPSERT_MAX_PAYLOAD_BYTES",
            str(DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES),
        )
    )
    max_records_per_batch = int(
        os.getenv(
            "MONGO_UPSERT_MAX_RECORDS_PER_BATCH",
            str(DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH),
        )
    )
    if max_payload_bytes <= 0:
        logger.warning(
            f"MONGO_UPSERT_MAX_PAYLOAD_BYTES={max_payload_bytes} is non-positive, disable payload-size splitting"
        )
    if max_records_per_batch <= 0:
        logger.warning(
            f"MONGO_UPSERT_MAX_RECORDS_PER_BATCH={max_records_per_batch} is non-positive, disable upsert record-count splitting"
        )
    return max_payload_bytes, max_records_per_batch


async def _run_batched_bulk_write(
    collection,
    ops: list[tuple[Any, int, str]],
    *,
    max_payload_bytes: int,
    max_records_per_batch: int,
    ordered: bool,
    log_prefix: str,
    what: str,
) -> None:
    """Execute UpdateOne ops as payload-size/record-count bounded bulk_write batches.

    ``ops`` is a list of ``(operation, estimated_bytes, id_for_log)`` triples.
    Splitting keeps each bulk command below MongoDB's 48MB message ceiling and
    bounds the in-memory op list. Fail-fast: a batch failure raises and no
    further batches run, so callers must treat the whole write as retryable
    (UpdateOne(..., upsert=True) is idempotent).
    """
    if not ops:
        return

    batches = _chunk_by_budget(
        ops, lambda triple: triple[1], max_payload_bytes, max_records_per_batch
    )
    if len(batches) > 1:
        logger.info(
            f"{log_prefix} {what} split into {len(batches)} batches "
            f"for {len(ops)} records"
        )
    for batch_index, (batch, estimated_bytes) in enumerate(batches, 1):
        if (
            len(batch) == 1
            and max_payload_bytes > 0
            and estimated_bytes > max_payload_bytes
        ):
            logger.warning(
                f"{log_prefix} {what}: single record id={batch[0][2]} "
                f"estimated {estimated_bytes} bytes exceeds {max_payload_bytes}"
            )
        logger.debug(
            f"{log_prefix} {what} batch {batch_index}/{len(batches)}: "
            f"records={len(batch)}, estimated_payload_bytes={estimated_bytes}"
        )
        await collection.bulk_write([triple[0] for triple in batch], ordered=ordered)


class ClientManager:
    _instances = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncMongoClient:
        async with cls._lock:
            if cls._instances["db"] is None:
                uri = os.environ.get(
                    "MONGO_URI",
                    config.get(
                        "mongodb",
                        "uri",
                        fallback="mongodb://root:root@localhost:27017/",
                    ),
                )
                database_name = os.environ.get(
                    "MONGO_DATABASE",
                    config.get("mongodb", "database", fallback="LightRAG"),
                )
                client = AsyncMongoClient(
                    uri,
                    driver=DriverInfo(name="LightRAG", version=__version__),
                )
                db = client.get_database(database_name)
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: AsyncDatabase):
        async with cls._lock:
            if db is not None:
                if db is cls._instances["db"]:
                    cls._instances["ref_count"] -= 1
                    if cls._instances["ref_count"] == 0:
                        cls._instances["db"] = None


@final
@dataclass
class MongoKVStorage(BaseKVStorage):
    db: AsyncDatabase = field(default=None)
    _data: AsyncCollection = field(default=None)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        # Check for MONGODB_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all MongoDB storage instances
        mongodb_workspace = os.environ.get("MONGODB_WORKSPACE")
        if mongodb_workspace and mongodb_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = mongodb_workspace.strip()
            logger.info(
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
            self.workspace = effective_workspace
            logger.debug(
                f"Final namespace with workspace prefix: '{self.final_namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            self.workspace = ""
            logger.debug(
                f"[{self.workspace}] Final namespace (no workspace): '{self.namespace}'"
            )

        self._collection_name = self.final_namespace
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
        ) = _resolve_upsert_batch_limits()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            self._data = await get_or_create_collection(self.db, self._collection_name)
            logger.debug(
                f"[{self.workspace}] Use MongoDB as KV {self._collection_name}"
            )

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._data = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        # Unified handling for flattened keys
        doc = await self._data.find_one({"_id": id})
        if doc:
            # Ensure time fields are present, provide default values for old data
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
        return doc

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        cursor = self._data.find({"_id": {"$in": ids}})
        docs = await cursor.to_list(length=None)

        doc_map: dict[str, dict[str, Any]] = {}
        for doc in docs:
            if not doc:
                continue
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
            doc_map[str(doc.get("_id"))] = doc

        ordered_results: list[dict[str, Any] | None] = []
        for id_value in ids:
            ordered_results.append(doc_map.get(str(id_value)))
        return ordered_results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        cursor = self._data.find({"_id": {"$in": list(keys)}}, {"_id": 1})
        existing_ids = {str(x["_id"]) async for x in cursor}
        return keys - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        # Unified handling for all namespaces with flattened keys. KV docs
        # (full_docs, text_chunks, llm_response_cache) can be large, so the
        # upsert is split into payload-bounded bulk_write batches.
        operations: list[tuple[Any, int, str]] = []
        current_time = int(time.time())  # Get current Unix timestamp

        for i, (k, v) in enumerate(data.items(), start=1):
            # For text_chunks namespace, ensure llm_cache_list field exists
            if self.namespace.endswith("text_chunks"):
                if "llm_cache_list" not in v:
                    v["llm_cache_list"] = []

            # Create a copy of v for $set operation, excluding create_time to avoid conflicts
            v_for_set = v.copy()
            v_for_set["_id"] = k  # Use flattened key as _id
            v_for_set["update_time"] = current_time  # Always update update_time

            # Remove create_time from $set to avoid conflict with $setOnInsert
            v_for_set.pop("create_time", None)

            operations.append(
                (
                    UpdateOne(
                        {"_id": k},
                        {
                            "$set": v_for_set,  # Update all fields except create_time
                            "$setOnInsert": {
                                "create_time": current_time
                            },  # Set create_time only on insert
                        },
                        upsert=True,
                    ),
                    _estimate_doc_bytes(v_for_set),
                    k,
                )
            )
            await _cooperative_yield(i)

        # ordered=False (intentional): the old single bulk_write used pymongo's
        # default ordered=True, but every op targets a distinct flattened _id, so
        # the writes are order-independent. ordered=False lets the server apply
        # them in parallel and is the right choice for idempotent upserts.
        await _run_batched_bulk_write(
            self._data,
            operations,
            max_payload_bytes=self._max_upsert_payload_bytes,
            max_records_per_batch=self._max_upsert_records_per_batch,
            ordered=False,
            log_prefix=f"[{self.workspace}] {self.namespace} upsert:",
            what="upsert",
        )

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        try:
            # Use count_documents with limit 1 for efficiency
            count = await self._data.count_documents({}, limit=1)
            return count == 0
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete documents with specified IDs

        Args:
            ids: List of document IDs to be deleted
        """
        if not ids:
            return

        # Convert to list if it's a set (MongoDB BSON cannot encode sets)
        if isinstance(ids, set):
            ids = list(ids)

        try:
            result = await self._data.delete_many({"_id": {"$in": ids}})
            logger.info(
                f"[{self.workspace}] Deleted {result.deleted_count} documents from {self.namespace}"
            )
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error deleting documents from {self.namespace}: {e}"
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            result = await self._data.delete_many({})
            deleted_count = result.deleted_count

            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} documents from doc status {self._collection_name}"
            )
            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped",
            }
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error dropping doc status {self._collection_name}: {e}"
            )
            return {"status": "error", "message": str(e)}


@final
@dataclass
class MongoDocStatusStorage(DocStatusStorage):
    db: AsyncDatabase = field(default=None)
    _data: AsyncCollection = field(default=None)

    def _prepare_doc_status_data(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Normalize and migrate a raw Mongo document to DocProcessingStatus-compatible dict."""
        # Make a copy of the data to avoid modifying the original
        data = doc.copy()
        # Remove deprecated content field if it exists
        data.pop("content", None)
        # Remove MongoDB _id field if it exists
        data.pop("_id", None)
        # If file_path is not in data, use document id as file path
        if "file_path" not in data:
            data["file_path"] = "no-file-path"
        # Ensure new fields exist with default values
        if "metadata" not in data:
            data["metadata"] = {}
        if "error_msg" not in data:
            data["error_msg"] = None
        # Backward compatibility: migrate legacy 'error' field to 'error_msg'
        if "error" in data:
            if "error_msg" not in data or data["error_msg"] in (None, ""):
                data["error_msg"] = data.pop("error")
            else:
                data.pop("error", None)
        return data

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        # Check for MONGODB_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all MongoDB storage instances
        mongodb_workspace = os.environ.get("MONGODB_WORKSPACE")
        if mongodb_workspace and mongodb_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = mongodb_workspace.strip()
            logger.info(
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
            self.workspace = effective_workspace
            logger.debug(
                f"Final namespace with workspace prefix: '{self.final_namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            self.workspace = ""
            logger.debug(f"Final namespace (no workspace): '{self.final_namespace}'")

        self._collection_name = self.final_namespace

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            self._data = await get_or_create_collection(self.db, self._collection_name)

            # Create and migrate all indexes including Chinese collation for file_path
            await self.create_and_migrate_indexes_if_not_exists()

            logger.debug(
                f"[{self.workspace}] Use MongoDB as DocStatus {self._collection_name}"
            )

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._data = None

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        return await self._data.find_one({"_id": id})

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        cursor = self._data.find({"_id": {"$in": ids}})
        docs = await cursor.to_list(length=None)

        doc_map: dict[str, dict[str, Any]] = {}
        for doc in docs:
            if not doc:
                continue
            doc_map[str(doc.get("_id"))] = doc

        ordered_results: list[dict[str, Any] | None] = []
        for id_value in ids:
            ordered_results.append(doc_map.get(str(id_value)))
        return ordered_results

    async def filter_keys(self, data: set[str]) -> set[str]:
        cursor = self._data.find({"_id": {"$in": list(data)}}, {"_id": 1})
        existing_ids = {str(x["_id"]) async for x in cursor}
        return data - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        update_tasks: list[Any] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            # Ensure chunks_list field exists and is an array
            if "chunks_list" not in v:
                v["chunks_list"] = []
            data[k]["_id"] = k
            update_tasks.append(
                self._data.update_one({"_id": k}, {"$set": v}, upsert=True)
            )
            await _cooperative_yield(i)
        await asyncio.gather(*update_tasks)

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        cursor = await self._data.aggregate(pipeline, allowDiskUse=True)
        result = await cursor.to_list()
        counts = {}
        for doc in result:
            counts[doc["_id"]] = doc["count"]
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching any of the given statuses in a single query.

        Uses MongoDB's $in operator to fetch all matching statuses in one
        round-trip instead of one find() call per status.
        """
        if not statuses:
            return {}
        status_values = [s.value for s in statuses]
        cursor = self._data.find({"status": {"$in": status_values}})
        docs = await cursor.to_list(length=None)
        result = {}
        for doc in docs:
            try:
                data = self._prepare_doc_status_data(doc)
                result[doc["_id"]] = DocProcessingStatus(**data)
            except KeyError as e:
                logger.error(
                    f"[{self.workspace}] Missing required field for document {doc['_id']}: {e}"
                )
                continue
        return result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
        cursor = self._data.find({"track_id": track_id})
        result = await cursor.to_list()
        processed_result = {}
        for doc in result:
            try:
                data = self._prepare_doc_status_data(doc)
                processed_result[doc["_id"]] = DocProcessingStatus(**data)
            except KeyError as e:
                logger.error(
                    f"[{self.workspace}] Missing required field for document {doc['_id']}: {e}"
                )
                continue
        return processed_result

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        try:
            # Use count_documents with limit 1 for efficiency
            count = await self._data.count_documents({}, limit=1)
            return count == 0
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            result = await self._data.delete_many({})
            deleted_count = result.deleted_count

            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} documents from doc status {self._collection_name}"
            )
            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped",
            }
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error dropping doc status {self._collection_name}: {e}"
            )
            return {"status": "error", "message": str(e)}

    async def delete(self, ids: list[str]) -> None:
        await self._data.delete_many({"_id": {"$in": ids}})

    async def create_and_migrate_indexes_if_not_exists(self):
        """Create indexes to optimize pagination queries and migrate file_path indexes for Chinese collation"""
        try:
            # Get indexes for the current collection only
            indexes_cursor = await self._data.list_indexes()
            existing_indexes = await indexes_cursor.to_list(length=None)
            existing_index_names = {idx.get("name", "") for idx in existing_indexes}

            # Define collation configuration for Chinese pinyin sorting
            collation_config = {"locale": "zh", "numericOrdering": True}

            # Use workspace-specific index names to avoid cross-workspace conflicts
            workspace_prefix = f"{self.workspace}_" if self.workspace != "" else ""

            # 1. Define all indexes needed with workspace-specific names
            all_indexes = [
                # Original pagination indexes
                {
                    "name": f"{workspace_prefix}status_updated_at",
                    "keys": [("status", 1), ("updated_at", -1)],
                },
                {
                    "name": f"{workspace_prefix}status_created_at",
                    "keys": [("status", 1), ("created_at", -1)],
                },
                {"name": f"{workspace_prefix}updated_at", "keys": [("updated_at", -1)]},
                {"name": f"{workspace_prefix}created_at", "keys": [("created_at", -1)]},
                {"name": f"{workspace_prefix}id", "keys": [("_id", 1)]},
                {"name": f"{workspace_prefix}track_id", "keys": [("track_id", 1)]},
                # New file_path indexes with Chinese collation and workspace-specific names
                {
                    "name": f"{workspace_prefix}file_path_zh_collation",
                    "keys": [("file_path", 1)],
                    "collation": collation_config,
                },
                {
                    "name": f"{workspace_prefix}status_file_path_zh_collation",
                    "keys": [("status", 1), ("file_path", 1)],
                    "collation": collation_config,
                },
                # Partial index on content_hash for content-based dedup lookups.
                # Mirrors the PG partial index: skip legacy/empty values so the
                # index stays small and a content_hash="" query is a guaranteed miss.
                {
                    "name": f"{workspace_prefix}content_hash",
                    "keys": [("content_hash", 1)],
                    "partialFilterExpression": {
                        "content_hash": {"$exists": True, "$type": "string", "$gt": ""}
                    },
                },
            ]

            # 2. Handle legacy index cleanup: only drop old indexes that exist in THIS collection
            legacy_index_names = [
                "file_path_zh_collation",
                "status_file_path_zh_collation",
                "status_updated_at",
                "status_created_at",
                "updated_at",
                "created_at",
                "id",
                "track_id",
                "content_hash",
            ]

            for legacy_name in legacy_index_names:
                if (
                    legacy_name in existing_index_names
                    and legacy_name
                    != f"{workspace_prefix}{legacy_name.replace(workspace_prefix, '')}"
                ):
                    try:
                        await self._data.drop_index(legacy_name)
                        logger.debug(
                            f"[{self.workspace}] Migrated: dropped legacy index '{legacy_name}' from collection {self._collection_name}"
                        )
                        existing_index_names.discard(legacy_name)
                    except PyMongoError as drop_error:
                        logger.warning(
                            f"[{self.workspace}] Failed to drop legacy index '{legacy_name}' from collection {self._collection_name}: {drop_error}"
                        )

            # 3. Create all needed indexes with workspace-specific names
            for index_info in all_indexes:
                index_name = index_info["name"]
                if index_name not in existing_index_names:
                    create_kwargs = {"name": index_name}
                    if "collation" in index_info:
                        create_kwargs["collation"] = index_info["collation"]
                    if "partialFilterExpression" in index_info:
                        create_kwargs["partialFilterExpression"] = index_info[
                            "partialFilterExpression"
                        ]

                    try:
                        await self._data.create_index(
                            index_info["keys"], **create_kwargs
                        )
                        logger.debug(
                            f"[{self.workspace}] Created index '{index_name}' for collection {self._collection_name}"
                        )
                    except PyMongoError as create_error:
                        # If creation still fails, log the error but continue with other indexes
                        logger.error(
                            f"[{self.workspace}] Failed to create index '{index_name}' for collection {self._collection_name}: {create_error}"
                        )
                else:
                    logger.debug(
                        f"[{self.workspace}] Index '{index_name}' already exists for collection {self._collection_name}"
                    )

        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error creating/migrating indexes for {self._collection_name}: {e}"
            )

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
            sort_field: Field to sort by ('created_at', 'updated_at', '_id')
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

        if sort_field not in ["created_at", "updated_at", "_id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        # Build query filter
        query_filter = {}
        if status_filter_values is not None:
            query_filter["status"] = {"$in": sorted(status_filter_values)}

        # Get total count
        total_count = await self._data.count_documents(query_filter)

        # Calculate skip value
        skip = (page - 1) * page_size

        # Build sort criteria
        sort_direction_value = 1 if sort_direction.lower() == "asc" else -1
        sort_criteria = [(sort_field, sort_direction_value)]

        # Query for paginated data with Chinese collation for file_path sorting
        if sort_field == "file_path":
            # Use Chinese collation for pinyin sorting
            cursor = (
                self._data.find(query_filter)
                .sort(sort_criteria)
                .collation({"locale": "zh", "numericOrdering": True})
                .skip(skip)
                .limit(page_size)
            )
        else:
            # Use default sorting for other fields
            cursor = (
                self._data.find(query_filter)
                .sort(sort_criteria)
                .skip(skip)
                .limit(page_size)
            )
        result = await cursor.to_list(length=page_size)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for doc in result:
            try:
                doc_id = doc["_id"]

                data = self._prepare_doc_status_data(doc)

                doc_status = DocProcessingStatus(**data)
                documents.append((doc_id, doc_status))
            except KeyError as e:
                logger.error(
                    f"[{self.workspace}] Missing required field for document {doc['_id']}: {e}"
                )
                continue

        return documents, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        cursor = await self._data.aggregate(pipeline, allowDiskUse=True)
        result = await cursor.to_list()

        counts = {}
        total_count = 0
        for doc in result:
            counts[doc["_id"]] = doc["count"]
            total_count += doc["count"]

        # Add 'all' field with total count
        counts["all"] = total_count

        return counts

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
            Returns the same format as get_by_id method
        """
        return await self._data.find_one({"file_path": file_path})

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Mongo-native override of basename-based document lookup.

        The caller is responsible for passing an already-canonical basename;
        stored ``file_path`` values are canonicalized by the business layer, so
        this lookup performs an exact match only and relies on the file_path
        index created by ``create_and_migrate_indexes_if_not_exists``.
        """
        if not basename:
            return None
        if basename == "unknown_source":
            return None

        try:
            doc = await self._data.find_one({"file_path": basename})
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error in get_doc_by_file_basename: {e}")
            return None
        if not doc:
            return None
        doc_id = doc.get("_id")
        if doc_id is None:
            return None
        return str(doc_id), doc

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Mongo-native override of content-hash document lookup.

        Uses the partial ``content_hash`` index. Empty strings are treated as a
        miss to align with the partial-index predicate; legacy rows missing the
        field cannot match a non-empty query because ``find_one`` requires an
        exact value.
        """
        if not content_hash:
            return None

        try:
            doc = await self._data.find_one({"content_hash": content_hash})
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error in get_doc_by_content_hash: {e}")
            return None
        if not doc:
            return None
        doc_id = doc.get("_id")
        if doc_id is None:
            return None
        return str(doc_id), doc


@final
@dataclass
class MongoGraphStorage(BaseGraphStorage):
    """
    A concrete implementation using MongoDB's $graphLookup to demonstrate multi-hop queries.
    """

    db: AsyncDatabase = field(default=None)
    # node collection storing node_id, node_properties
    collection: AsyncCollection = field(default=None)
    # edge collection storing source_node_id, target_node_id, and edge_properties
    edgeCollection: AsyncCollection = field(default=None)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        # Check for MONGODB_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all MongoDB storage instances
        mongodb_workspace = os.environ.get("MONGODB_WORKSPACE")
        if mongodb_workspace and mongodb_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = mongodb_workspace.strip()
            logger.info(
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
            self.workspace = effective_workspace
            logger.debug(
                f"Final namespace with workspace prefix: '{self.final_namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            self.workspace = ""
            logger.debug(f"Final namespace (no workspace): '{self.final_namespace}'")

        self._collection_name = self.final_namespace
        self._edge_collection_name = f"{self._collection_name}_edges"
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
        ) = _resolve_upsert_batch_limits()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            self.collection = await get_or_create_collection(
                self.db, self._collection_name
            )
            self.edge_collection = await get_or_create_collection(
                self.db, self._edge_collection_name
            )

            # Create Atlas Search index for better search performance if possible
            await self.create_search_index_if_not_exists()

            # Fail-fast: migrate legacy edges to a canonical edge_key and build
            # the unique index before serving (upsert relies on it). Raises on
            # failure so startup aborts rather than serving a half-migrated graph.
            await self.create_edge_indexes_and_migrate_if_not_exists()

            logger.debug(
                f"[{self.workspace}] Use MongoDB as KG {self._collection_name}"
            )

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self.collection = None
            self.edge_collection = None

    async def create_edge_indexes_and_migrate_if_not_exists(self) -> None:
        """Create the unique ``edge_key`` index, migrating legacy edges first.

        Fail-fast one-time migration (mirrors the OpenSearch canonical-id work):

          1. dedupe legacy reciprocal duplicate docs, **merging provenance** —
             ``source_id`` / ``source_ids`` / ``file_path`` are unioned into the
             survivor so no chunk evidence is lost;
          2. backfill the canonical ``edge_key`` on every remaining doc;
          3. build the partial unique index on ``edge_key``.

        The index doubles as the completion flag: if it already exists we skip.
        Anything failing raises, so ``initialize``/startup aborts rather than
        serving a half-migrated collection (the upsert filter relies on every doc
        having ``edge_key``). Runs inside ``get_data_init_lock``, so only the
        first worker of a deployment migrates; the rest skip on the index.

        Assumes no concurrent *old-version* writer adds ``edge_key``-less docs
        after this completes (true for stop-the-world / single-deployment
        restarts). A true rolling deploy with mixed code versions writing one
        collection could leave a straggler duplicate; the remedy is to drop the
        ``edge_key_unique`` index and let the next startup re-migrate.
        """
        workspace_prefix = f"{self.workspace}_" if self.workspace != "" else ""
        index_name = f"{workspace_prefix}edge_key_unique"

        indexes_cursor = await self.edge_collection.list_indexes()
        existing_indexes = await indexes_cursor.to_list(length=None)
        if any(idx.get("name") == index_name for idx in existing_indexes):
            logger.debug(
                f"[{self.workspace}] Edge collection {self._edge_collection_name} "
                f"already has {index_name}; skipping edge_key migration"
            )
            return

        logger.info(
            f"[{self.workspace}] Starting edge_key migration for "
            f"{self._edge_collection_name}"
        )
        removed = await self._dedupe_legacy_edges()
        backfilled = await self._backfill_edge_keys()
        # The unique index is the completion flag — only created on full success.
        # unique build raises if any duplicate slipped through (e.g. a concurrent
        # old-version writer), which fails startup so the next run retries.
        await self.edge_collection.create_index(
            [("edge_key", 1)],
            name=index_name,
            unique=True,
            partialFilterExpression={"edge_key": {"$exists": True, "$type": "string"}},
        )
        logger.info(
            f"[{self.workspace}] edge_key migration complete for "
            f"{self._edge_collection_name}: removed {removed} duplicate doc(s), "
            f"backfilled {backfilled} edge_key(s)"
        )

    async def _dedupe_legacy_edges(self) -> int:
        """Collapse duplicate docs for the same undirected edge into one.

        Groups by the canonical (sorted) endpoint pair; for each group with more
        than one doc, keeps the newest by ``created_at`` and unions
        ``source_ids``/``source_id``/``file_path`` from every duplicate into it
        before deleting the rest. Returns the number of docs removed.
        """
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "lo": {
                            "$cond": [
                                {"$lte": ["$source_node_id", "$target_node_id"]},
                                "$source_node_id",
                                "$target_node_id",
                            ]
                        },
                        "hi": {
                            "$cond": [
                                {"$lte": ["$source_node_id", "$target_node_id"]},
                                "$target_node_id",
                                "$source_node_id",
                            ]
                        },
                    },
                    "docs": {
                        "$push": {
                            "_id": "$_id",
                            "source_id": "$source_id",
                            "source_ids": "$source_ids",
                            "file_path": "$file_path",
                            "created_at": "$created_at",
                        }
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$match": {"count": {"$gt": 1}}},
        ]
        removed = 0
        cursor = await self.edge_collection.aggregate(pipeline, allowDiskUse=True)
        async for group in cursor:
            docs = group["docs"]
            survivor = max(docs, key=lambda d: d.get("created_at") or 0)
            others = [d for d in docs if d["_id"] != survivor["_id"]]
            if not others:
                continue

            # Union provenance across ALL docs (survivor included) so no
            # chunk/file evidence is dropped when the duplicates are deleted.
            all_source_ids: list[str] = []
            all_file_paths: list[str] = []
            for d in docs:
                sids = d.get("source_ids")
                if not sids and d.get("source_id"):
                    sids = d["source_id"].split(GRAPH_FIELD_SEP)
                all_source_ids = merge_source_ids(all_source_ids, sids or [])
                fp = d.get("file_path")
                fps = fp.split(GRAPH_FIELD_SEP) if fp else []
                all_file_paths = merge_source_ids(all_file_paths, fps)

            set_fields: dict[str, Any] = {}
            if all_source_ids:
                set_fields["source_ids"] = all_source_ids
                set_fields["source_id"] = GRAPH_FIELD_SEP.join(all_source_ids)
            if all_file_paths:
                set_fields["file_path"] = GRAPH_FIELD_SEP.join(all_file_paths)
            if set_fields:
                await self.edge_collection.update_one(
                    {"_id": survivor["_id"]}, {"$set": set_fields}
                )
            await self.edge_collection.delete_many(
                {"_id": {"$in": [d["_id"] for d in others]}}
            )
            removed += len(others)
        return removed

    async def _backfill_edge_keys(self) -> int:
        """Set the canonical ``edge_key`` on every doc that lacks it. Returns the
        modified count. Runs after dedupe, so each canonical pair has one doc and
        the backfilled keys are unique."""
        result = await self.edge_collection.update_many(
            {"edge_key": {"$exists": False}},
            [
                {
                    "$set": {
                        "edge_key": {
                            "$cond": [
                                {"$lte": ["$source_node_id", "$target_node_id"]},
                                {
                                    "$concat": [
                                        "$source_node_id",
                                        _EDGE_KEY_SEP,
                                        "$target_node_id",
                                    ]
                                },
                                {
                                    "$concat": [
                                        "$target_node_id",
                                        _EDGE_KEY_SEP,
                                        "$source_node_id",
                                    ]
                                },
                            ]
                        }
                    }
                }
            ],
        )
        return result.modified_count

    # Sample entity document
    # "source_ids" is Array representation of "source_id" split by GRAPH_FIELD_SEP

    # {
    #     "_id" : "CompanyA",
    #     "entity_id" : "CompanyA",
    #     "entity_type" : "Organization",
    #     "description" : "A major technology company",
    #     "source_id" : "chunk-eeec0036b909839e8ec4fa150c939eec",
    #     "source_ids": ["chunk-eeec0036b909839e8ec4fa150c939eec"],
    #     "file_path" : "custom_kg",
    #     "created_at" : 1749904575
    # }

    # Sample relation document
    # {
    #     "_id" : ObjectId("6856ac6e7c6bad9b5470b678"), // MongoDB build-in ObjectId
    #     "description" : "CompanyA develops ProductX",
    #     "source_node_id" : "CompanyA",
    #     "target_node_id" : "ProductX",
    #     "relationship": "Develops", // To distinguish multiple same-target relations
    #     "weight" : Double("1"),
    #     "keywords" : "develop, produce",
    #     "source_id" : "chunk-eeec0036b909839e8ec4fa150c939eec",
    #     "source_ids": ["chunk-eeec0036b909839e8ec4fa150c939eec"],
    #     "file_path" : "custom_kg",
    #     "created_at" : 1749904575
    # }

    #
    # -------------------------------------------------------------------------
    # BASIC QUERIES
    # -------------------------------------------------------------------------
    #

    async def has_node(self, node_id: str) -> bool:
        """
        Check if node_id is present in the collection by looking up its doc.
        No real need for $graphLookup here, but let's keep it direct.
        """
        doc = await self.collection.find_one({"_id": node_id}, {"_id": 1})
        return doc is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if there's a direct single-hop edge between source_node_id and target_node_id.
        """
        doc = await self.edge_collection.find_one(
            {
                "$or": [
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                    },
                    {
                        "source_node_id": target_node_id,
                        "target_node_id": source_node_id,
                    },
                ]
            },
            {"_id": 1},
        )
        return doc is not None

    #
    # -------------------------------------------------------------------------
    # DEGREES
    # -------------------------------------------------------------------------
    #

    async def node_degree(self, node_id: str) -> int:
        """
        Returns the total number of edges connected to node_id (both inbound and outbound).
        """
        return await self.edge_collection.count_documents(
            {"$or": [{"source_node_id": node_id}, {"target_node_id": node_id}]}
        )

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        return src_degree + trg_degree

    #
    # -------------------------------------------------------------------------
    # GETTERS
    # -------------------------------------------------------------------------
    #

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """
        Return the full node document, or None if missing.
        """
        return await self.collection.find_one({"_id": node_id})

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        return await self.edge_collection.find_one(
            {
                "$or": [
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                    },
                    {
                        "source_node_id": target_node_id,
                        "target_node_id": source_node_id,
                    },
                ]
            }
        )

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """
        Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found
        """
        cursor = self.edge_collection.find(
            {
                "$or": [
                    {"source_node_id": source_node_id},
                    {"target_node_id": source_node_id},
                ]
            },
            {"source_node_id": 1, "target_node_id": 1},
        )

        return [
            (e.get("source_node_id"), e.get("target_node_id")) async for e in cursor
        ]

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        result = {}

        async for doc in self.collection.find({"_id": {"$in": node_ids}}):
            result[doc.get("_id")] = doc
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        # merge the outbound and inbound results with the same "_id" and sum the "degree"
        merged_results = {}

        # Outbound degrees
        outbound_pipeline = [
            {"$match": {"source_node_id": {"$in": node_ids}}},
            {"$group": {"_id": "$source_node_id", "degree": {"$sum": 1}}},
        ]

        cursor = await self.edge_collection.aggregate(
            outbound_pipeline, allowDiskUse=True
        )
        async for doc in cursor:
            merged_results[doc.get("_id")] = doc.get("degree")

        # Inbound degrees
        inbound_pipeline = [
            {"$match": {"target_node_id": {"$in": node_ids}}},
            {"$group": {"_id": "$target_node_id", "degree": {"$sum": 1}}},
        ]

        cursor = await self.edge_collection.aggregate(
            inbound_pipeline, allowDiskUse=True
        )
        async for doc in cursor:
            merged_results[doc.get("_id")] = merged_results.get(
                doc.get("_id"), 0
            ) + doc.get("degree")

        return merged_results

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Batch retrieve edges for multiple nodes.
        For each node, returns both outgoing and incoming edges to properly represent
        the undirected graph nature.

        Args:
            node_ids: List of node IDs (entity_id) for which to retrieve edges.

        Returns:
            A dictionary mapping each node ID to its list of edge tuples (source, target).
            For each node, the list includes both:
            - Outgoing edges: (queried_node, connected_node)
            - Incoming edges: (connected_node, queried_node)
        """
        result = {node_id: [] for node_id in node_ids}

        # Query outgoing edges (where node is the source)
        outgoing_cursor = self.edge_collection.find(
            {"source_node_id": {"$in": node_ids}},
            {"source_node_id": 1, "target_node_id": 1},
        )
        async for edge in outgoing_cursor:
            source = edge["source_node_id"]
            target = edge["target_node_id"]
            result[source].append((source, target))

        # Query incoming edges (where node is the target)
        incoming_cursor = self.edge_collection.find(
            {"target_node_id": {"$in": node_ids}},
            {"source_node_id": 1, "target_node_id": 1},
        )
        async for edge in incoming_cursor:
            source = edge["source_node_id"]
            target = edge["target_node_id"]
            result[target].append((source, target))

        return result

    #
    # -------------------------------------------------------------------------
    # UPSERTS
    # -------------------------------------------------------------------------
    #

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Insert or update a node document.
        """
        update_doc = {"$set": {**node_data}}
        if node_data.get("source_id", ""):
            update_doc["$set"]["source_ids"] = node_data["source_id"].split(
                GRAPH_FIELD_SEP
            )

        await self.collection.update_one({"_id": node_id}, update_doc, upsert=True)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Upsert the undirected edge between source_node_id and target_node_id.

        Matches on the canonical ``edge_key`` (direction-independent) instead of
        the old bidirectional ``$or`` filter, so the unique ``edge_key`` index can
        reject a racing duplicate insert. If two writers race the first insert,
        the loser hits a ``DuplicateKeyError``; we retry once, which now matches
        the just-inserted doc and updates it.
        """
        # Ensure source node exists
        await self.upsert_node(source_node_id, {})

        # Copy so we never mutate the caller's edge_data dict.
        set_doc: dict = {**edge_data}
        if edge_data.get("source_id", ""):
            set_doc["source_ids"] = edge_data["source_id"].split(GRAPH_FIELD_SEP)
        set_doc["source_node_id"] = source_node_id
        set_doc["target_node_id"] = target_node_id
        edge_key = _canonical_edge_key(source_node_id, target_node_id)
        set_doc["edge_key"] = edge_key
        update_doc = {"$set": set_doc}

        for attempt in range(2):
            try:
                await self.edge_collection.update_one(
                    {"edge_key": edge_key}, update_doc, upsert=True
                )
                return
            except DuplicateKeyError:
                # Another writer inserted this edge between our filter miss and
                # insert. Retry once: the doc now exists, so the upsert becomes a
                # plain update. A second failure is unexpected — let it surface.
                if attempt == 1:
                    raise

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Batch insert/update multiple nodes using a single bulk_write() call.

        Args:
            nodes: List of (node_id, node_data) tuples.
        """
        if not nodes:
            return
        ops: list[tuple[Any, int, str]] = []
        for node_id, node_data in nodes:
            update_doc: dict = {"$set": {**node_data}}
            if node_data.get("source_id", ""):
                update_doc["$set"]["source_ids"] = node_data["source_id"].split(
                    GRAPH_FIELD_SEP
                )
            ops.append(
                (
                    UpdateOne({"_id": node_id}, update_doc, upsert=True),
                    _estimate_doc_bytes(update_doc),
                    node_id,
                )
            )
        await _run_batched_bulk_write(
            self.collection,
            ops,
            max_payload_bytes=self._max_upsert_payload_bytes,
            max_records_per_batch=self._max_upsert_records_per_batch,
            ordered=True,
            log_prefix=f"[{self.workspace}] {self.namespace} nodes:",
            what="node upsert",
        )

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        """Check existence of multiple nodes using a single $in query.

        Args:
            node_ids: List of node IDs to check.

        Returns:
            Set of node_ids that exist in the graph.
        """
        if not node_ids:
            return set()
        cursor = self.collection.find({"_id": {"$in": node_ids}}, {"_id": 1})
        return {doc["_id"] async for doc in cursor}

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Batch insert/update multiple edges using a single bulk_write() call.

        Also ensures source nodes exist (matching upsert_edge() behaviour) via a
        separate bulk_write on the node collection for any source nodes that need
        to be created as empty placeholders.

        Args:
            edges: List of (source_node_id, target_node_id, edge_data) tuples.
        """
        if not edges:
            return

        # Ensure all source nodes exist (mirrors upsert_edge's upsert_node call)
        source_node_ids = list(dict.fromkeys(src for src, _tgt, _data in edges))
        node_ops: list[tuple[Any, int, str]] = [
            (
                UpdateOne({"_id": src}, {"$setOnInsert": {"_id": src}}, upsert=True),
                _estimate_doc_bytes({"_id": src}),
                src,
            )
            for src in source_node_ids
        ]
        await _run_batched_bulk_write(
            self.collection,
            node_ops,
            max_payload_bytes=self._max_upsert_payload_bytes,
            max_records_per_batch=self._max_upsert_records_per_batch,
            ordered=False,
            log_prefix=f"[{self.workspace}] {self.namespace} edges:",
            what="source-node placeholder upsert",
        )

        # Key every edge by its canonical edge_key and dedupe within the batch
        # (last-write-wins). Deduping collapses reciprocal directions onto one op,
        # which both matches the unique index and avoids an intra-batch
        # duplicate-key error from two ops inserting the same edge_key.
        deduped_ops: dict[str, tuple[Any, int, str]] = {}
        for source_node_id, target_node_id, edge_data in edges:
            update_doc: dict = {"$set": {**edge_data}}
            if edge_data.get("source_id", ""):
                update_doc["$set"]["source_ids"] = edge_data["source_id"].split(
                    GRAPH_FIELD_SEP
                )
            update_doc["$set"]["source_node_id"] = source_node_id
            update_doc["$set"]["target_node_id"] = target_node_id
            edge_key = _canonical_edge_key(source_node_id, target_node_id)
            update_doc["$set"]["edge_key"] = edge_key
            deduped_ops[edge_key] = (
                UpdateOne({"edge_key": edge_key}, update_doc, upsert=True),
                _estimate_doc_bytes(update_doc),
                f"{source_node_id}->{target_node_id}",
            )
        edge_ops = list(deduped_ops.values())

        # ordered=False so a duplicate-key race on one edge does not block the
        # rest. If concurrent writers (another process bypassing the keyed lock)
        # win an insert, our upsert hits 11000; retry once — the docs now exist so
        # the upserts update instead of inserting. A non-11000 error re-raises.
        async def _run_edge_bulk() -> None:
            await _run_batched_bulk_write(
                self.edge_collection,
                edge_ops,
                max_payload_bytes=self._max_upsert_payload_bytes,
                max_records_per_batch=self._max_upsert_records_per_batch,
                ordered=False,
                log_prefix=f"[{self.workspace}] {self.namespace} edges:",
                what="edge upsert",
            )

        try:
            await _run_edge_bulk()
        except BulkWriteError as e:
            details = e.details or {}
            write_errors = details.get("writeErrors", [])
            # Retry ONLY when every failure is a duplicate-key race; a
            # writeConcern failure (durability problem, empty writeErrors) or any
            # other write error must surface, not be masked by a blind retry.
            dup_only = (
                bool(write_errors)
                and all(we.get("code") == _DUPLICATE_KEY_CODE for we in write_errors)
                and not details.get("writeConcernErrors")
            )
            if not dup_only:
                raise
            logger.debug(
                f"[{self.workspace}] {self.namespace} edges: {len(write_errors)} "
                f"duplicate-key race(s) on edge upsert; retrying as updates"
            )
            await _run_edge_bulk()

    #
    # -------------------------------------------------------------------------
    # DELETION
    # -------------------------------------------------------------------------
    #

    async def delete_node(self, node_id: str) -> None:
        """
        1) Remove node's doc entirely.
        2) Remove inbound & outbound edges from any doc that references node_id.
        """
        # Remove all edges
        await self.edge_collection.delete_many(
            {"$or": [{"source_node_id": node_id}, {"target_node_id": node_id}]}
        )

        # Remove the node doc
        await self.collection.delete_one({"_id": node_id})

    #
    # -------------------------------------------------------------------------
    # QUERY
    # -------------------------------------------------------------------------
    #

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node _ids(entity names) in the database
        Returns:
            [id1, id2, ...]  # Alphabetically sorted id list
        """

        # Use aggregation with allowDiskUse for large datasets
        pipeline = [{"$project": {"_id": 1}}, {"$sort": {"_id": 1}}]
        cursor = await self.collection.aggregate(pipeline, allowDiskUse=True)
        labels = []
        async for doc in cursor:
            labels.append(doc["_id"])
        return labels

    def _construct_graph_node(
        self, node_id, node_data: dict[str, str]
    ) -> KnowledgeGraphNode:
        return KnowledgeGraphNode(
            id=node_id,
            labels=[node_id],
            properties={
                k: v
                for k, v in node_data.items()
                if k
                not in [
                    "_id",
                    "connected_edges",
                    "source_ids",
                    "edge_count",
                ]
            },
        )

    def _construct_graph_edge(self, edge_id: str, edge: dict[str, str]):
        return KnowledgeGraphEdge(
            id=edge_id,
            type=edge.get("relationship", ""),
            source=edge["source_node_id"],
            target=edge["target_node_id"],
            properties={
                k: v
                for k, v in edge.items()
                if k
                not in [
                    "_id",
                    "source_node_id",
                    "target_node_id",
                    "relationship",
                    "source_ids",
                    "edge_key",
                ]
            },
        )

    async def _fetch_nodes_by_ids(
        self, node_ids: list[str], projection: dict[str, int] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch nodes by ID while preserving the requested order."""
        if not node_ids:
            return []

        cursor = self.collection.find({"_id": {"$in": node_ids}}, projection)
        docs_by_id = {}
        async for doc in cursor:
            docs_by_id[str(doc["_id"])] = doc
        return [docs_by_id[node_id] for node_id in node_ids if node_id in docs_by_id]

    async def get_knowledge_graph_all_by_degree(
        self, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """
        It's possible that the node with one or multiple relationships is retrieved,
        while its neighbor is not.  Then this node might seem like disconnected in UI.
        """

        total_node_count = await self.collection.count_documents({})
        result = KnowledgeGraph()
        seen_edges = set()

        result.is_truncated = total_node_count > max_nodes
        if result.is_truncated:
            # Get all node_ids ranked by degree if max_nodes exceeds total node count
            pipeline = [
                {"$project": {"source_node_id": 1, "_id": 0}},
                {"$group": {"_id": "$source_node_id", "degree": {"$sum": 1}}},
                {
                    "$unionWith": {
                        "coll": self._edge_collection_name,
                        "pipeline": [
                            {"$project": {"target_node_id": 1, "_id": 0}},
                            {
                                "$group": {
                                    "_id": "$target_node_id",
                                    "degree": {"$sum": 1},
                                }
                            },
                        ],
                    }
                },
                {"$group": {"_id": "$_id", "degree": {"$sum": "$degree"}}},
                {"$sort": {"degree": -1}},
                {"$limit": max_nodes},
            ]
            cursor = await self.edge_collection.aggregate(pipeline, allowDiskUse=True)

            node_ids = []
            async for doc in cursor:
                node_id = str(doc["_id"])
                node_ids.append(node_id)

            if len(node_ids) < max_nodes:
                remaining = max_nodes - len(node_ids)
                cursor = self.collection.find(
                    {"_id": {"$nin": node_ids}},
                    {"source_ids": 0},
                ).limit(remaining)
                async for doc in cursor:
                    node_ids.append(str(doc["_id"]))

            docs = await self._fetch_nodes_by_ids(node_ids, {"source_ids": 0})
            for doc in docs:
                result.nodes.append(self._construct_graph_node(doc["_id"], doc))

            # As node count reaches the limit, only need to fetch the edges that directly connect to these nodes
            edge_cursor = self.edge_collection.find(
                {
                    "$and": [
                        {"source_node_id": {"$in": node_ids}},
                        {"target_node_id": {"$in": node_ids}},
                    ]
                }
            )
        else:
            # All nodes and edges are needed
            cursor = self.collection.find({}, {"source_ids": 0})

            async for doc in cursor:
                node_id = str(doc["_id"])
                result.nodes.append(self._construct_graph_node(doc["_id"], doc))

            edge_cursor = self.edge_collection.find({})

        async for edge in edge_cursor:
            edge_id = f"{edge['source_node_id']}-{edge['target_node_id']}"
            if edge_id not in seen_edges:
                seen_edges.add(edge_id)
                result.edges.append(self._construct_graph_edge(edge_id, edge))

        return result

    async def _bidirectional_bfs_nodes(
        self,
        node_labels: list[str],
        seen_nodes: set[str],
        result: KnowledgeGraph,
        depth: int,
        max_depth: int,
        max_nodes: int,
    ) -> KnowledgeGraph:
        if depth > max_depth or len(result.nodes) > max_nodes:
            return result

        cursor = self.collection.find({"_id": {"$in": node_labels}})

        async for node in cursor:
            node_id = node["_id"]
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                result.nodes.append(self._construct_graph_node(node_id, node))
                if len(result.nodes) > max_nodes:
                    return result

        # Collect neighbors
        # Get both inbound and outbound one hop nodes
        cursor = self.edge_collection.find(
            {
                "$or": [
                    {"source_node_id": {"$in": node_labels}},
                    {"target_node_id": {"$in": node_labels}},
                ]
            }
        )

        neighbor_nodes = []
        async for edge in cursor:
            if edge["source_node_id"] not in seen_nodes:
                neighbor_nodes.append(edge["source_node_id"])
            if edge["target_node_id"] not in seen_nodes:
                neighbor_nodes.append(edge["target_node_id"])

        if neighbor_nodes:
            result = await self._bidirectional_bfs_nodes(
                neighbor_nodes, seen_nodes, result, depth + 1, max_depth, max_nodes
            )

        return result

    async def get_knowledge_subgraph_bidirectional_bfs(
        self,
        node_label: str,
        depth: int,
        max_depth: int,
        max_nodes: int,
    ) -> KnowledgeGraph:
        seen_nodes = set()
        seen_edges = set()
        result = KnowledgeGraph()

        result = await self._bidirectional_bfs_nodes(
            [node_label], seen_nodes, result, depth, max_depth, max_nodes
        )

        # Get all edges from seen_nodes
        all_node_ids = list(seen_nodes)
        cursor = self.edge_collection.find(
            {
                "$and": [
                    {"source_node_id": {"$in": all_node_ids}},
                    {"target_node_id": {"$in": all_node_ids}},
                ]
            }
        )

        async for edge in cursor:
            edge_id = f"{edge['source_node_id']}-{edge['target_node_id']}"
            if edge_id not in seen_edges:
                result.edges.append(self._construct_graph_edge(edge_id, edge))
                seen_edges.add(edge_id)

        return result

    async def get_knowledge_subgraph_in_out_bound_bfs(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        seen_nodes = set()
        seen_edges = set()
        result = KnowledgeGraph()
        project_doc = {
            "source_ids": 0,
            "created_at": 0,
            "entity_type": 0,
            "file_path": 0,
        }

        # Verify if starting node exists
        start_node = await self.collection.find_one({"_id": node_label})
        if not start_node:
            logger.warning(
                f"[{self.workspace}] Starting node with label {node_label} does not exist!"
            )
            return result

        seen_nodes.add(node_label)
        result.nodes.append(self._construct_graph_node(node_label, start_node))

        if max_depth == 0:
            return result

        # In MongoDB, depth = 0 means one-hop
        max_depth = max_depth - 1

        pipeline = [
            {"$match": {"_id": node_label}},
            {"$project": project_doc},
            {
                "$graphLookup": {
                    "from": self._edge_collection_name,
                    "startWith": "$_id",
                    "connectFromField": "target_node_id",
                    "connectToField": "source_node_id",
                    "maxDepth": max_depth,
                    "depthField": "depth",
                    "as": "connected_edges",
                },
            },
            {
                "$unionWith": {
                    "coll": self._collection_name,
                    "pipeline": [
                        {"$match": {"_id": node_label}},
                        {"$project": project_doc},
                        {
                            "$graphLookup": {
                                "from": self._edge_collection_name,
                                "startWith": "$_id",
                                "connectFromField": "source_node_id",
                                "connectToField": "target_node_id",
                                "maxDepth": max_depth,
                                "depthField": "depth",
                                "as": "connected_edges",
                            }
                        },
                    ],
                }
            },
        ]

        cursor = await self.collection.aggregate(pipeline, allowDiskUse=True)
        node_edges = []

        # Two records for node_label are returned capturing outbound and inbound connected_edges
        async for doc in cursor:
            if doc.get("connected_edges", []):
                node_edges.extend(doc.get("connected_edges"))

        # Sort the connected edges by depth ascending and weight descending
        # And stores the source_node_id and target_node_id in sequence to retrieve the neighbouring nodes
        node_edges = sorted(
            node_edges,
            key=lambda x: (x["depth"], -x["weight"]),
        )

        # As order matters, we need to use another list to store the node_id
        # And only take the first max_nodes ones
        node_ids = []
        for edge in node_edges:
            if len(node_ids) < max_nodes and edge["source_node_id"] not in seen_nodes:
                node_ids.append(edge["source_node_id"])
                seen_nodes.add(edge["source_node_id"])

            if len(node_ids) < max_nodes and edge["target_node_id"] not in seen_nodes:
                node_ids.append(edge["target_node_id"])
                seen_nodes.add(edge["target_node_id"])

        # Filter out all the node whose id is same as node_label so that we do not check existence next step
        cursor = self.collection.find({"_id": {"$in": node_ids}})

        async for doc in cursor:
            result.nodes.append(self._construct_graph_node(str(doc["_id"]), doc))

        for edge in node_edges:
            if (
                edge["source_node_id"] not in seen_nodes
                or edge["target_node_id"] not in seen_nodes
            ):
                continue

            edge_id = f"{edge['source_node_id']}-{edge['target_node_id']}"
            if edge_id not in seen_edges:
                result.edges.append(self._construct_graph_edge(edge_id, edge))
                seen_edges.add(edge_id)

        return result

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maximum nodes to return, Defaults to global_config max_graph_nodes

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit

        If a graph is like this and starting from B:
        A → B ← C ← F, B -> E, C → D

        Outbound BFS:
        B → E

        Inbound BFS:
        A → B
        C → B
        F → C

        Bidirectional BFS:
        A → B
        B → E
        F → C
        C → B
        C → D
        """
        # Use global_config max_graph_nodes as default if max_nodes is None
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        result = KnowledgeGraph()
        start = time.perf_counter()

        try:
            # Optimize pipeline to avoid memory issues with large datasets
            if node_label == "*":
                result = await self.get_knowledge_graph_all_by_degree(
                    max_depth, max_nodes
                )
            elif GRAPH_BFS_MODE == "in_out_bound":
                result = await self.get_knowledge_subgraph_in_out_bound_bfs(
                    node_label, max_depth, max_nodes
                )
            else:
                result = await self.get_knowledge_subgraph_bidirectional_bfs(
                    node_label, 0, max_depth, max_nodes
                )

            duration = time.perf_counter() - start

            logger.info(
                f"[{self.workspace}] Subgraph query successful in {duration:.4f} seconds | Node count: {len(result.nodes)} | Edge count: {len(result.edges)} | Truncated: {result.is_truncated}"
            )

        except PyMongoError as e:
            # Handle memory limit errors specifically
            if "memory limit" in str(e).lower() or "sort exceeded" in str(e).lower():
                logger.warning(
                    f"[{self.workspace}] MongoDB memory limit exceeded, falling back to simple query: {str(e)}"
                )
                # Fallback to a simple query without complex aggregation
                try:
                    simple_cursor = self.collection.find({}).limit(max_nodes)
                    async for doc in simple_cursor:
                        result.nodes.append(
                            self._construct_graph_node(str(doc["_id"]), doc)
                        )
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Fallback query completed | Node count: {len(result.nodes)}"
                    )
                except PyMongoError as fallback_error:
                    logger.error(
                        f"[{self.workspace}] Fallback query also failed: {str(fallback_error)}"
                    )
            else:
                logger.error(f"[{self.workspace}] MongoDB query failed: {str(e)}")

        return result

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """
        logger.info(f"[{self.workspace}] Deleting {len(nodes)} nodes")
        if not nodes:
            return

        # 1. Remove all edges referencing these nodes
        await self.edge_collection.delete_many(
            {
                "$or": [
                    {"source_node_id": {"$in": nodes}},
                    {"target_node_id": {"$in": nodes}},
                ]
            }
        )

        # 2. Delete the node documents
        await self.collection.delete_many({"_id": {"$in": nodes}})

        logger.debug(f"[{self.workspace}] Successfully deleted nodes: {nodes}")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        logger.info(f"[{self.workspace}] Deleting {len(edges)} edges")
        if not edges:
            return

        all_edge_pairs = []
        for source_id, target_id in edges:
            all_edge_pairs.append(
                {"source_node_id": source_id, "target_node_id": target_id}
            )
            all_edge_pairs.append(
                {"source_node_id": target_id, "target_node_id": source_id}
            )

        await self.edge_collection.delete_many({"$or": all_edge_pairs})

        logger.debug(f"[{self.workspace}] Successfully deleted edges: {edges}")

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        cursor = self.collection.find({})
        nodes = []
        async for node in cursor:
            node_dict = dict(node)
            # Add node id (entity_id) to the dictionary for easier access
            node_dict["id"] = node_dict.get("_id")
            nodes.append(node_dict)
        return nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        cursor = self.edge_collection.find({})
        edges = []
        async for edge in cursor:
            edge_dict = dict(edge)
            edge_dict["source"] = edge_dict.get("source_node_id")
            edge_dict["target"] = edge_dict.get("target_node_id")
            edges.append(edge_dict)
        return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels(entity names) by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels(entity names) sorted by degree (highest first)
        """
        try:
            # Use aggregation pipeline to count edges per node and sort by degree
            pipeline = [
                # Count outbound edges
                {"$group": {"_id": "$source_node_id", "out_degree": {"$sum": 1}}},
                # Union with inbound edges count
                {
                    "$unionWith": {
                        "coll": self._edge_collection_name,
                        "pipeline": [
                            {
                                "$group": {
                                    "_id": "$target_node_id",
                                    "in_degree": {"$sum": 1},
                                }
                            }
                        ],
                    }
                },
                # Group by node_id and sum degrees
                {
                    "$group": {
                        "_id": "$_id",
                        "total_degree": {
                            "$sum": {
                                "$add": [
                                    {"$ifNull": ["$out_degree", 0]},
                                    {"$ifNull": ["$in_degree", 0]},
                                ]
                            }
                        },
                    }
                },
                # Sort by degree descending, then by label ascending
                {"$sort": {"total_degree": -1, "_id": 1}},
                # Limit results
                {"$limit": limit},
                # Project only the label
                {"$project": {"_id": 1}},
            ]

            cursor = await self.edge_collection.aggregate(pipeline, allowDiskUse=True)
            labels = []
            async for doc in cursor:
                if doc.get("_id"):
                    labels.append(doc["_id"])

            logger.debug(
                f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            return []

    async def _try_atlas_text_search(self, query_strip: str, limit: int) -> list[str]:
        """Try Atlas Search using simple text search."""
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "entity_id_search_idx",
                        "text": {"query": query_strip, "path": "_id"},
                    }
                },
                {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
                {"$limit": limit},
            ]
            cursor = await self.collection.aggregate(pipeline)
            labels = [doc["_id"] async for doc in cursor if doc.get("_id")]
            if labels:
                logger.debug(
                    f"[{self.workspace}] Atlas text search returned {len(labels)} results"
                )
                return labels
            return []
        except PyMongoError as e:
            logger.debug(f"[{self.workspace}] Atlas text search failed: {e}")
            return []

    async def _try_atlas_autocomplete_search(
        self, query_strip: str, limit: int
    ) -> list[str]:
        """Try Atlas Search using autocomplete for prefix matching."""
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "entity_id_search_idx",
                        "autocomplete": {
                            "query": query_strip,
                            "path": "_id",
                            "fuzzy": {"maxEdits": 1, "prefixLength": 1},
                        },
                    }
                },
                {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
                {"$limit": limit},
            ]
            cursor = await self.collection.aggregate(pipeline)
            labels = [doc["_id"] async for doc in cursor if doc.get("_id")]
            if labels:
                logger.debug(
                    f"[{self.workspace}] Atlas autocomplete search returned {len(labels)} results"
                )
                return labels
            return []
        except PyMongoError as e:
            logger.debug(f"[{self.workspace}] Atlas autocomplete search failed: {e}")
            return []

    async def _try_atlas_compound_search(
        self, query_strip: str, limit: int
    ) -> list[str]:
        """Try Atlas Search using compound query for comprehensive matching."""
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "entity_id_search_idx",
                        "compound": {
                            "should": [
                                {
                                    "text": {
                                        "query": query_strip,
                                        "path": "_id",
                                        "score": {"boost": {"value": 10}},
                                    }
                                },
                                {
                                    "autocomplete": {
                                        "query": query_strip,
                                        "path": "_id",
                                        "score": {"boost": {"value": 5}},
                                        "fuzzy": {"maxEdits": 1, "prefixLength": 1},
                                    }
                                },
                                {
                                    "wildcard": {
                                        "query": f"*{query_strip}*",
                                        "path": "_id",
                                        "score": {"boost": {"value": 2}},
                                    }
                                },
                            ],
                            "minimumShouldMatch": 1,
                        },
                    }
                },
                {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
                {"$sort": {"score": {"$meta": "searchScore"}}},
                {"$limit": limit},
            ]
            cursor = await self.collection.aggregate(pipeline)
            labels = [doc["_id"] async for doc in cursor if doc.get("_id")]
            if labels:
                logger.debug(
                    f"[{self.workspace}] Atlas compound search returned {len(labels)} results"
                )
                return labels
            return []
        except PyMongoError as e:
            logger.debug(f"[{self.workspace}] Atlas compound search failed: {e}")
            return []

    async def _fallback_regex_search(self, query_strip: str, limit: int) -> list[str]:
        """Fallback to regex-based search when Atlas Search fails."""
        try:
            logger.debug(
                f"[{self.workspace}] Using regex fallback search for: '{query_strip}'"
            )

            escaped_query = re.escape(query_strip)
            regex_condition = {"_id": {"$regex": escaped_query, "$options": "i"}}
            cursor = self.collection.find(regex_condition, {"_id": 1}).limit(limit * 2)
            docs = await cursor.to_list(length=limit * 2)

            # Extract labels
            labels = []
            for doc in docs:
                doc_id = doc.get("_id")
                if doc_id:
                    labels.append(doc_id)

            # Sort results to prioritize exact matches and starts-with matches
            def sort_key(label):
                label_lower = label.lower()
                query_lower_strip = query_strip.lower()

                if label_lower == query_lower_strip:
                    return (0, label_lower)  # Exact match - highest priority
                elif label_lower.startswith(query_lower_strip):
                    return (1, label_lower)  # Starts with - medium priority
                else:
                    return (2, label_lower)  # Contains - lowest priority

            labels.sort(key=sort_key)
            labels = labels[:limit]  # Apply final limit after sorting

            logger.debug(
                f"[{self.workspace}] Regex fallback search returned {len(labels)} results (limit: {limit})"
            )
            return labels

        except Exception as e:
            logger.error(f"[{self.workspace}] Regex fallback search failed: {e}")
            import traceback

            logger.error(f"[{self.workspace}] Traceback: {traceback.format_exc()}")
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """
        Search labels(entity names) with progressive fallback strategy:
        1. Atlas text search (simple and fast)
        2. Atlas autocomplete search (prefix matching with fuzzy)
        3. Atlas compound search (comprehensive matching)
        4. Regex fallback (when Atlas Search is unavailable)
        """
        query_strip = query.strip()
        if not query_strip:
            return []

        # First check if we have any nodes at all
        try:
            node_count = await self.collection.count_documents({})
            if node_count == 0:
                logger.debug(
                    f"[{self.workspace}] No nodes found in collection {self._collection_name}"
                )
                return []
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error counting nodes: {e}")
            return []

        # Progressive search strategy
        search_methods = [
            ("text", self._try_atlas_text_search),
            ("autocomplete", self._try_atlas_autocomplete_search),
            ("compound", self._try_atlas_compound_search),
        ]

        # Try Atlas Search methods in order
        for method_name, search_method in search_methods:
            try:
                labels = await search_method(query_strip, limit)
                if labels:
                    logger.debug(
                        f"[{self.workspace}] Search successful using {method_name} method: {len(labels)} results"
                    )
                    return labels
                else:
                    logger.debug(
                        f"[{self.workspace}] {method_name} search returned no results, trying next method"
                    )
            except Exception as e:
                logger.debug(
                    f"[{self.workspace}] {method_name} search failed: {e}, trying next method"
                )
                continue

        # If all Atlas Search methods fail, use regex fallback
        logger.info(
            f"[{self.workspace}] All Atlas Search methods failed, using regex fallback search for: '{query_strip}'"
        )
        return await self._fallback_regex_search(query_strip, limit)

    async def _check_if_index_needs_rebuild(
        self, indexes: list, index_name: str
    ) -> bool:
        """Check if the existing index needs to be rebuilt due to configuration issues."""
        for index in indexes:
            if index["name"] == index_name:
                # Check if the index has the old problematic configuration
                definition = index.get("latestDefinition", {})
                mappings = definition.get("mappings", {})
                fields = mappings.get("fields", {})
                id_field = fields.get("_id", {})

                # If it's the old single-type autocomplete configuration, rebuild
                if (
                    isinstance(id_field, dict)
                    and id_field.get("type") == "autocomplete"
                ):
                    logger.info(
                        f"[{self.workspace}] Found old index configuration for '{index_name}', will rebuild"
                    )
                    return True

                # If it's not a list (multi-type configuration), rebuild
                if not isinstance(id_field, list):
                    logger.info(
                        f"[{self.workspace}] Index '{index_name}' needs upgrade to multi-type configuration"
                    )
                    return True

                logger.info(
                    f"[{self.workspace}] Index '{index_name}' has correct configuration"
                )
                return False
        return True  # Index doesn't exist, needs creation

    async def _safely_drop_old_index(self, index_name: str):
        """Safely drop the old search index."""
        try:
            await self.collection.drop_search_index(index_name)
            logger.info(
                f"[{self.workspace}] Successfully dropped old search index '{index_name}'"
            )
        except PyMongoError as e:
            logger.warning(
                f"[{self.workspace}] Could not drop old index '{index_name}': {e}"
            )

    async def _create_improved_search_index(self, index_name: str):
        """Create an improved search index with multiple field types."""
        search_index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "_id": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "token",
                            },
                            {
                                "type": "autocomplete",
                                "maxGrams": 15,
                                "minGrams": 2,
                            },
                        ]
                    },
                },
                "analyzer": "lucene.standard",  # Index-level analyzer for text processing
            },
            name=index_name,
            type="search",
        )

        await self.collection.create_search_index(search_index_model)
        logger.info(
            f"[{self.workspace}] Created improved Atlas Search index '{index_name}' for collection {self._collection_name}. "
        )
        logger.info(
            f"[{self.workspace}] Index will be built asynchronously, using regex fallback until ready."
        )

    async def create_search_index_if_not_exists(self):
        """Creates an improved Atlas Search index for entity search, rebuilding if necessary."""
        index_name = "entity_id_search_idx"

        try:
            # Check if we're using MongoDB Atlas (has search index capabilities)
            indexes_cursor = await self.collection.list_search_indexes()
            indexes = await indexes_cursor.to_list(length=None)

            # Check if we need to rebuild the index
            needs_rebuild = await self._check_if_index_needs_rebuild(
                indexes, index_name
            )

            if needs_rebuild:
                # Check if index exists and drop it
                index_exists = any(idx["name"] == index_name for idx in indexes)
                if index_exists:
                    await self._safely_drop_old_index(index_name)

                # Create the improved search index (async, no waiting)
                await self._create_improved_search_index(index_name)
            else:
                logger.info(
                    f"[{self.workspace}] Atlas Search index '{index_name}' already exists with correct configuration"
                )

        except PyMongoError as e:
            # This is expected if not using MongoDB Atlas or if search indexes are not supported
            logger.info(
                f"[{self.workspace}] Could not create Atlas Search index for {self._collection_name}: {e}. "
                "This is normal if not using MongoDB Atlas - search will use regex fallback."
            )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Unexpected error creating Atlas Search index for {self._collection_name}: {e}"
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            result = await self.collection.delete_many({})
            deleted_count = result.deleted_count

            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} documents from graph {self._collection_name}"
            )

            result = await self.edge_collection.delete_many({})
            edge_count = result.deleted_count
            logger.info(
                f"[{self.workspace}] Dropped {edge_count} edges from graph {self._edge_collection_name}"
            )

            return {
                "status": "success",
                "message": f"{deleted_count} documents and {edge_count} edges dropped",
            }
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error dropping graph {self._collection_name}: {e}"
            )
            return {"status": "error", "message": str(e)}


@dataclass
class _PendingVectorDoc:
    """Buffered vector upsert waiting for embedding and/or bulk flush."""

    source: dict[str, Any]
    content: str
    vector: list[float] | None = None


@final
@dataclass
class MongoVectorDBStorage(BaseVectorStorage):
    db: AsyncDatabase | None = field(default=None)
    _data: AsyncCollection | None = field(default=None)
    _index_name: str = field(default="", init=False)

    def __init__(
        self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    def __post_init__(self):
        self._validate_embedding_func()

        # Check for MONGODB_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all MongoDB storage instances
        mongodb_workspace = os.environ.get("MONGODB_WORKSPACE")
        if mongodb_workspace and mongodb_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = mongodb_workspace.strip()
            logger.info(
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
            self.workspace = effective_workspace
            logger.debug(
                f"Final namespace with workspace prefix: '{self.final_namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            self.workspace = ""
            logger.debug(f"Final namespace (no workspace): '{self.final_namespace}'")

        # Set index name based on workspace for backward compatibility
        if effective_workspace:
            # Use collection-specific index name for workspaced collections to avoid conflicts
            self._index_name = f"vector_knn_index_{self.final_namespace}"
        else:
            # Keep original index name for backward compatibility with existing deployments
            self._index_name = "vector_knn_index"

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._collection_name = self.final_namespace
        self._max_batch_size = self.global_config["embedding_batch_num"]

        # Flush-time batching limits (see module-level DEFAULT_MONGO_* constants).
        # A non-positive value disables that splitting dimension. The two upsert
        # limits are shared with KV/graph via _resolve_upsert_batch_limits(); the
        # delete count cap is vector-specific (only the VDB batches deletes here).
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
        ) = _resolve_upsert_batch_limits()
        self._max_delete_records_per_batch = int(
            os.getenv(
                "MONGO_DELETE_MAX_RECORDS_PER_BATCH",
                str(DEFAULT_MONGO_DELETE_MAX_RECORDS_PER_BATCH),
            )
        )
        if self._max_delete_records_per_batch <= 0:
            logger.warning(
                f"MONGO_DELETE_MAX_RECORDS_PER_BATCH={self._max_delete_records_per_batch} is non-positive, disable delete record-count splitting"
            )

        # Deferred-embedding buffers and the per-namespace flush lock.
        # Constructed in initialize() once shared-storage primitives are
        # available; keyed on final_namespace so two instances pointing at
        # the same MongoDB collection (e.g. with the MONGODB_WORKSPACE env
        # override) share a single writer lock.
        self._pending_vector_docs: dict[str, _PendingVectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        self._flush_lock = None

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            self._data = await get_or_create_collection(self.db, self._collection_name)

            # Ensure vector index exists
            await self.create_vector_index_if_not_exists()

            logger.debug(
                f"[{self.workspace}] Use MongoDB as VDB {self._collection_name}"
            )

        if self._flush_lock is None:
            self._flush_lock = get_namespace_lock(
                namespace=self.final_namespace, workspace=""
            )

    async def finalize(self):
        """Flush pending vector ops, release the Mongo client, surface unflushed data."""
        flush_error: Exception | None = None
        try:
            await self._flush_pending_vector_ops()
        except Exception as e:
            flush_error = e

        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._data = None

        pending_docs = len(self._pending_vector_docs)
        pending_deletes = len(self._pending_vector_deletes)

        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] MongoVectorDBStorage.finalize() flush raised; "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes were left buffered (client released, data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] MongoVectorDBStorage.finalize() left "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes buffered after final flush attempt (these writes have been lost)"
            )

    async def create_vector_index_if_not_exists(self):
        """Creates an Atlas Vector Search index."""
        try:
            indexes_cursor = await self._data.list_search_indexes()
            indexes = await indexes_cursor.to_list(length=None)
            for index in indexes:
                if index["name"] == self._index_name:
                    # Check if the existing index has matching vector dimensions
                    existing_dim = None
                    definition = index.get("latestDefinition", {})
                    fields = definition.get("fields", [])
                    for field in fields:
                        if (
                            field.get("type") == "vector"
                            and field.get("path") == "vector"
                        ):
                            existing_dim = field.get("numDimensions")
                            break

                    expected_dim = self.embedding_func.embedding_dim

                    if existing_dim is not None and existing_dim != expected_dim:
                        error_msg = (
                            f"Vector dimension mismatch! Index '{self._index_name}' has "
                            f"dimension {existing_dim}, but current embedding model expects "
                            f"dimension {expected_dim}. Please drop the existing index or "
                            f"use an embedding model with matching dimensions."
                        )
                        logger.error(f"[{self.workspace}] {error_msg}")
                        raise ValueError(error_msg)

                    logger.info(
                        f"[{self.workspace}] vector index {self._index_name} already exists with matching dimensions ({expected_dim})"
                    )
                    return

            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": self.embedding_func.embedding_dim,  # Ensure correct dimensions
                            "path": "vector",
                            "similarity": "cosine",  # Options: euclidean, cosine, dotProduct
                        }
                    ]
                },
                name=self._index_name,
                type="vectorSearch",
            )

            await self._data.create_search_index(search_index_model)
            logger.info(
                f"[{self.workspace}] Vector index {self._index_name} created successfully."
            )

        except PyMongoError as e:
            error_msg = f"[{self.workspace}] Error creating vector index {self._index_name}: {e}"
            logger.error(error_msg)
            raise SystemExit(
                f"Failed to create MongoDB vector index. Program cannot continue. {error_msg}"
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vector docs for embedding and batched flush.

        Embedding deliberately does NOT happen here: repeated upserts of
        the same id, or many small batches, collapse into a single
        flush-time embedding pass. Reads observe pending docs via the
        same lock for read-your-writes.
        """
        if not data:
            return

        current_time = int(time.time())

        pending_docs: list[tuple[str, _PendingVectorDoc]] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            source = {
                "_id": k,
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            pending_docs.append(
                (
                    k,
                    _PendingVectorDoc(source=source, content=v["content"]),
                )
            )
            await _cooperative_yield(i)

        # Installing a fresh _PendingVectorDoc invalidates any vector
        # cached by a prior get_vectors_by_ids() call on a stale revision.
        async with self._flush_lock:
            for doc_id, pdoc in pending_docs:
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = pdoc

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Queries the vector database using Atlas Vector Search.

        Reads from the server-side index only; buffered upserts and deletes
        are NOT visible until ``index_done_callback`` / ``finalize`` flushes
        them. Callers that need read-your-writes for a freshly upserted id
        should use ``get_by_id`` / ``get_by_ids`` (which consult the buffer)
        or flush first. Matches the deferred-embedding contract used by
        OpenSearch / FAISS / Nano.
        """
        if query_embedding is not None:
            # Convert numpy array to list if needed for MongoDB compatibility
            if hasattr(query_embedding, "tolist"):
                query_vector = query_embedding.tolist()
            else:
                query_vector = list(query_embedding)
        else:
            # Generate the embedding
            embedding = await self.embedding_func(
                [query], context="query", _priority=5
            )  # higher priority for query
            # Convert numpy array to a list to ensure compatibility with MongoDB
            query_vector = embedding[0].tolist()

        # Define the aggregation pipeline with the converted query vector
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._index_name,  # Use stored index name for consistency
                    "path": "vector",
                    "queryVector": query_vector,
                    "numCandidates": 100,  # Adjust for performance
                    "limit": top_k,
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": self.cosine_better_than_threshold}}},
            {"$project": {"vector": 0}},
        ]

        # Execute the aggregation pipeline
        cursor = await self._data.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)

        # Format and return the results with created_at field
        return [
            {
                **doc,
                "id": doc["_id"],
                "distance": doc.get("score", None),
                "created_at": doc.get("created_at"),  # Include created_at field
            }
            for doc in results
        ]

    async def index_done_callback(self) -> None:
        """Flush buffered vector ops; Mongo persists automatically once written."""
        await self._flush_pending_vector_ops()

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered vector upserts and deletes in batched bulk writes.

        Embedding runs *inside* this lock (not in `upsert` or lock-free):
        it makes deferred embedding and the bulk write atomic against
        concurrent upserts and destructive mutations. Any failure (embed
        or server write) raises and leaves both buffers intact; the next
        `index_done_callback` retries automatically.

        Concurrency invariant: ``_flush_lock`` is a non-reentrant asyncio
        lock. Callers MUST NOT hold it when invoking this method --
        re-entry would deadlock. The only in-tree callers are
        ``index_done_callback`` and ``finalize``, both lock-free.
        """
        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self._data is None:
                return

            pending_docs = self._pending_vector_docs
            pending_deletes = self._pending_vector_deletes

            docs_to_embed: list[tuple[str, _PendingVectorDoc]] = [
                (doc_id, pdoc)
                for doc_id, pdoc in pending_docs.items()
                if pdoc.vector is None
            ]

            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: embedding "
                    f"{len(docs_to_embed)} vectors in {len(batches)} batch(es) "
                    f"(batch_num={self._max_batch_size})"
                )
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error embedding pending vector ops "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise

                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((_, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pdoc.vector = np.array(embedding, dtype=np.float32).tolist()
                    await _cooperative_yield(i)

            # Assemble final upsert payload. After the embed loop above every
            # pending doc has a non-None vector (count-mismatch was checked),
            # so we can iterate without re-guarding. Each full_doc carries its
            # own "_id" (from source), matching the UpdateOne filter key.
            ids_to_commit: list[str] = list(pending_docs.keys())
            list_data: list[dict[str, Any]] = [
                {**pending_docs[doc_id].source, "vector": pending_docs[doc_id].vector}
                for doc_id in ids_to_commit
            ]

            try:
                if list_data:
                    # Split the upsert into batches that stay under the server-side
                    # bulk-command message limit and bound peak memory. Fail-fast:
                    # any batch failure raises immediately and the full buffer is
                    # retained for the next flush (upsert/delete are idempotent).
                    # Logging is kept aligned with MilvusVectorDBStorage; the
                    # batching maths is shared via _chunk_by_budget.
                    upsert_batches = _chunk_by_budget(
                        list_data,
                        _estimate_doc_bytes,
                        self._max_upsert_payload_bytes,
                        self._max_upsert_records_per_batch,
                    )
                    if len(upsert_batches) > 1:
                        logger.info(
                            f"[{self.workspace}] {self.namespace} flush: upsert split into "
                            f"{len(upsert_batches)} batches for {len(list_data)} records "
                            f"(max_payload={self._max_upsert_payload_bytes} batch={self._max_upsert_records_per_batch})"
                        )
                    for batch_index, (records_batch, estimated_bytes) in enumerate(
                        upsert_batches, 1
                    ):
                        if (
                            len(records_batch) == 1
                            and self._max_upsert_payload_bytes > 0
                            and estimated_bytes > self._max_upsert_payload_bytes
                        ):
                            logger.warning(
                                f"[{self.workspace}] {self.namespace} flush: single record "
                                f"id={records_batch[0].get('_id')} estimated {estimated_bytes} bytes "
                                f"exceeds {self._max_upsert_payload_bytes}"
                            )
                        logger.debug(
                            f"[{self.workspace}] MongoDB upsert batch {batch_index}/{len(upsert_batches)}: "
                            f"records={len(records_batch)}, estimated_payload_bytes={estimated_bytes}"
                        )
                        await self._data.bulk_write(
                            [
                                UpdateOne(
                                    {"_id": doc["_id"]}, {"$set": doc}, upsert=True
                                )
                                for doc in records_batch
                            ],
                            ordered=False,
                        )
                if pending_deletes:
                    # Chunk deletes by record count; _ids are short strings so a
                    # count cap is enough to stay under the bulk message limit.
                    # delete_many($in) is the 1:1 equivalent of a batched delete.
                    delete_ids = list(pending_deletes)
                    delete_chunk = (
                        self._max_delete_records_per_batch
                        if self._max_delete_records_per_batch > 0
                        else len(delete_ids)
                    )
                    for i in range(0, len(delete_ids), delete_chunk):
                        await self._data.delete_many(
                            {"_id": {"$in": delete_ids[i : i + delete_chunk]}}
                        )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error flushing vector ops "
                    f"(upserts={len(pending_docs)}, "
                    f"deletes={len(pending_deletes)}): {e}"
                )
                raise

            # On success, clear the buffers in-place so external references
            # (e.g. drop()) see the cleared state.
            for doc_id in ids_to_commit:
                pending_docs.pop(doc_id, None)
            pending_deletes.clear()

    async def delete(self, ids: list[str]) -> None:
        """Buffer vector deletes for batched flush."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for {len(ids)} vectors in {self.namespace}"
        )

    async def delete_entity(self, entity_name: str) -> None:
        """Buffer an entity vector delete by computing its hash ID."""
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        async with self._flush_lock:
            self._pending_vector_docs.pop(entity_id, None)
            self._pending_vector_deletes.add(entity_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for entity {entity_name} (id={entity_id})"
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt.

        The whole method runs under ``_flush_lock`` so the server-side find
        + delete cannot interleave with an in-flight bulk write. Server-side
        failures are re-raised (no log-and-swallow): the caller decides
        whether to retry.

        Buffer semantics — post-prune with caller short-circuit contract:
            Matching pending upserts in ``_pending_vector_docs`` are
            pruned **only after** the server-side ``delete_many``
            succeeds. On failure the pending buffer stays intact and
            the exception propagates so the caller (``adelete_by_entity``
            in ``utils_graph.py``) can short-circuit before
            ``_persist_graph_updates`` flushes a half-cleaned buffer.
        """

        def _prune_pending() -> None:
            for doc_id in [
                k
                for k, v in self._pending_vector_docs.items()
                if v.source.get("src_id") == entity_name
                or v.source.get("tgt_id") == entity_name
            ]:
                self._pending_vector_docs.pop(doc_id, None)

        async with self._flush_lock:
            if self._data is None:
                # No server state to mutate; buffer prune is the only
                # delete intent we can record.
                _prune_pending()
                return

            # _id is the only field we need from the find; project to keep
            # the cursor light.
            relations_cursor = self._data.find(
                {"$or": [{"src_id": entity_name}, {"tgt_id": entity_name}]},
                {"_id": 1},
            )
            relations = await relations_cursor.to_list(length=None)

            if not relations:
                # No server rows to delete — still safe to prune any
                # pending upserts so they can't re-create the relation.
                _prune_pending()
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
                return

            relation_ids = [relation["_id"] for relation in relations]
            await self._data.delete_many({"_id": {"$in": relation_ids}})
            # Server-side delete succeeded — safe to prune the pending
            # buffer so subsequent flushes don't re-upsert the deleted
            # relations.
            _prune_pending()
            logger.debug(
                f"[{self.workspace}] Deleted {len(relation_ids)} relations for {entity_name}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID, with read-your-writes against the buffer.

        Pending buffer hits never include the `vector` field; server-side
        fallback projects it out for parity.
        """
        async with self._flush_lock:
            if id in self._pending_vector_deletes:
                return None
            pending = self._pending_vector_docs.get(id)
            if pending is not None:
                doc = dict(pending.source)
                # Surface both _id (Mongo native) and id (API expectation).
                doc.setdefault("_id", id)
                doc["id"] = id
                return doc

        try:
            result = await self._data.find_one({"_id": id}, {"vector": 0})
            if result:
                result_dict = dict(result)
                if "_id" in result_dict and "id" not in result_dict:
                    result_dict["id"] = result_dict["_id"]
                return result_dict
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs (read-your-writes), preserving order."""
        if not ids:
            return []

        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    doc = dict(pending.source)
                    doc.setdefault("_id", doc_id)
                    doc["id"] = doc_id
                    buffered[doc_id] = doc
                    continue
                remaining.append(doc_id)

        formatted_map: dict[str, dict[str, Any]] = {}
        if remaining:
            try:
                cursor = self._data.find({"_id": {"$in": remaining}}, {"vector": 0})
                results = await cursor.to_list(length=None)
                for result in results:
                    result_dict = dict(result)
                    if "_id" in result_dict and "id" not in result_dict:
                        result_dict["id"] = result_dict["_id"]
                    key = str(result_dict.get("id", result_dict.get("_id")))
                    formatted_map[key] = result_dict
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error retrieving vector data for IDs {remaining}: {e}"
                )
                return []

        return [
            buffered[doc_id] if doc_id in buffered else formatted_map.get(str(doc_id))
            for doc_id in ids
        ]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings for given IDs, with read-your-writes.

        Pending docs whose vector hasn't been embedded yet are embedded
        lazily inside the lock; the resulting vector is cached on the
        buffered `_PendingVectorDoc` so the next flush won't re-embed.

        Visibility caveat for ids not in the buffer: the server-side
        ``find`` fallback runs *outside* ``_flush_lock``. A concurrent
        ``delete()`` that lands between lock release and the cursor
        read only buffers the delete -- the old vector is still on disk
        until the next flush, so this method may return a stale vector
        for an id that has been buffered for deletion. This is
        best-effort read-after-uncommitted-delete and matches the
        ``query()`` contract: callers needing strict consistency must
        ``index_done_callback()`` first.
        """
        if not ids:
            return {}

        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            docs_to_embed: list[tuple[str, _PendingVectorDoc]] = []
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    if pending.vector is None:
                        docs_to_embed.append((doc_id, pending))
                    else:
                        result[doc_id] = pending.vector
                    continue
                remaining.append(doc_id)

            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error lazily embedding pending vectors "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((doc_id, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pdoc.vector = np.array(embedding, dtype=np.float32).tolist()
                    result[doc_id] = pdoc.vector
                    await _cooperative_yield(i)

        if not remaining:
            return result

        try:
            cursor = self._data.find(
                {"_id": {"$in": remaining}}, {"_id": 1, "vector": 1}
            )
            results = await cursor.to_list(length=None)
            for row in results:
                if row and "vector" in row and "_id" in row:
                    result[row["_id"]] = row["vector"]
            return result
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error getting vectors: {e}")
            return result

    async def drop(self) -> dict[str, str]:
        """Drop all documents and recreate the vector index. Destructive.

        MUST only be called when ``pipeline_status`` is idle (see the
        Pipeline concurrency contract in ``AGENTS.md``); the only
        in-tree caller ``clear_documents`` enforces this.

        Caveat — only this instance's buffers are cleared. Other
        ``MongoVectorDBStorage`` instances aliased onto the same
        ``final_namespace`` (multi-worker processes, or distinct
        workspaces collapsed by ``MONGODB_WORKSPACE``) keep their own
        buffers; a sibling whose prior flush failed and left buffers
        intact will, on its next flush, bulk-write those stale rows into
        the freshly recreated collection. Direct callers bypassing the
        idle precondition MUST flush every aliased instance first.

        Returns:
            dict[str, str]: ``{"status": "success"|"error", "message": str}``
        """
        try:
            async with self._flush_lock:
                # Discard any buffered writes before the collection is wiped;
                # a concurrent flush would otherwise resurrect them.
                self._pending_vector_docs.clear()
                self._pending_vector_deletes.clear()

                # Delete all documents
                result = await self._data.delete_many({})
                deleted_count = result.deleted_count

                # Recreate vector index
                await self.create_vector_index_if_not_exists()

            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} documents from vector storage {self._collection_name} and recreated vector index"
            )
            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped and vector index recreated",
            }
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error dropping vector storage {self._collection_name}: {e}"
            )
            return {"status": "error", "message": str(e)}


async def get_or_create_collection(db: AsyncDatabase, collection_name: str):
    collection_names = await db.list_collection_names()

    if collection_name not in collection_names:
        collection = await db.create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")
        return collection
    else:
        logger.debug(f"Collection '{collection_name}' already exists.")
        return db.get_collection(collection_name)
