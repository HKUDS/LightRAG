import os
import re
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
from ..utils import logger, compute_mdhash_id
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock

import pipmaster as pm

if not pm.is_installed("pymongo"):
    pm.install("pymongo")

from pymongo import AsyncMongoClient  # type: ignore
from pymongo import UpdateOne  # type: ignore
from pymongo.asynchronous.database import AsyncDatabase  # type: ignore
from pymongo.asynchronous.collection import AsyncCollection  # type: ignore
from pymongo.operations import SearchIndexModel  # type: ignore
from pymongo.errors import PyMongoError  # type: ignore

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

GRAPH_BFS_MODE = os.getenv("MONGO_GRAPH_BFS_MODE", "bidirectional")


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
                client = AsyncMongoClient(uri)
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
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding passed workspace: '{self.workspace}')"
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

        # Unified handling for all namespaces with flattened keys
        # Use bulk_write for better performance

        operations = []
        current_time = int(time.time())  # Get current Unix timestamp

        for k, v in data.items():
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
                UpdateOne(
                    {"_id": k},
                    {
                        "$set": v_for_set,  # Update all fields except create_time
                        "$setOnInsert": {
                            "create_time": current_time
                        },  # Set create_time only on insert
                    },
                    upsert=True,
                )
            )

        if operations:
            await self._data.bulk_write(operations)

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
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding passed workspace: '{self.workspace}')"
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
        for k, v in data.items():
            # Ensure chunks_list field exists and is an array
            if "chunks_list" not in v:
                v["chunks_list"] = []
            data[k]["_id"] = k
            update_tasks.append(
                self._data.update_one({"_id": k}, {"$set": v}, upsert=True)
            )
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
        cursor = self._data.find({"status": status.value})
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
        if status_filter is not None:
            query_filter["status"] = status_filter.value

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
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding passed workspace: '{self.workspace}')"
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

            logger.debug(
                f"[{self.workspace}] Use MongoDB as KG {self._collection_name}"
            )

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self.collection = None
            self.edge_collection = None

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
        """
        Upsert an edge between source_node_id and target_node_id with optional 'relation'.
        If an edge with the same target exists, we remove it and re-insert with updated data.
        """
        # Ensure source node exists
        await self.upsert_node(source_node_id, {})

        update_doc = {"$set": edge_data}
        if edge_data.get("source_id", ""):
            update_doc["$set"]["source_ids"] = edge_data["source_id"].split(
                GRAPH_FIELD_SEP
            )

        edge_data["source_node_id"] = source_node_id
        edge_data["target_node_id"] = target_node_id

        await self.edge_collection.update_one(
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
            update_doc,
            upsert=True,
        )

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
        Get all existing node _id in the database
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
                ]
            },
        )

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

            cursor = self.collection.find({"_id": {"$in": node_ids}}, {"source_ids": 0})
            async for doc in cursor:
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
        min_degree: int = 0,
        include_orphans: bool = False,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maximum nodes to return, Defaults to global_config max_graph_nodes
            min_degree: Minimum degree (connections) for nodes to be included. 0=all nodes
            include_orphans: Include orphan nodes (degree=0) even when min_degree > 0

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
        """Get popular labels by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
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
        Search labels with progressive fallback strategy:
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
        # Check for MONGODB_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all MongoDB storage instances
        mongodb_workspace = os.environ.get("MONGODB_WORKSPACE")
        if mongodb_workspace and mongodb_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = mongodb_workspace.strip()
            logger.info(
                f"Using MONGODB_WORKSPACE environment variable: '{effective_workspace}' (overriding passed workspace: '{self.workspace}')"
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

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._data = None

    async def create_vector_index_if_not_exists(self):
        """Creates an Atlas Vector Search index."""
        try:
            indexes_cursor = await self._data.list_search_indexes()
            indexes = await indexes_cursor.to_list(length=None)
            for index in indexes:
                if index["name"] == self._index_name:
                    logger.info(
                        f"[{self.workspace}] vector index {self._index_name} already exist"
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
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        # Add current time as Unix timestamp
        current_time = int(time.time())

        list_data = [
            {
                "_id": k,
                "created_at": current_time,  # Add created_at field as Unix timestamp
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = np.array(embeddings[i], dtype=np.float32).tolist()

        update_tasks = []
        for doc in list_data:
            update_tasks.append(
                self._data.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
            )
        await asyncio.gather(*update_tasks)

        return list_data

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Queries the vector database using Atlas Vector Search."""
        if query_embedding is not None:
            # Convert numpy array to list if needed for MongoDB compatibility
            if hasattr(query_embedding, "tolist"):
                query_vector = query_embedding.tolist()
            else:
                query_vector = list(query_embedding)
        else:
            # Generate the embedding
            embedding = await self.embedding_func(
                [query], _priority=5
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
        # Mongo handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        logger.debug(
            f"[{self.workspace}] Deleting {len(ids)} vectors from {self.namespace}"
        )
        if not ids:
            return

        # Convert to list if it's a set (MongoDB BSON cannot encode sets)
        if isinstance(ids, set):
            ids = list(ids)

        try:
            result = await self._data.delete_many({"_id": {"$in": ids}})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {result.deleted_count} vectors from {self.namespace}"
            )
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {str(e)}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name

        Args:
            entity_name: Name of the entity to delete
        """
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            result = await self._data.delete_one({"_id": entity_id})
            if result.deleted_count > 0:
                logger.debug(
                    f"[{self.workspace}] Successfully deleted entity {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] Entity {entity_name} not found in storage"
                )
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error deleting entity {entity_name}: {str(e)}"
            )

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: Name of the entity whose relations should be deleted
        """
        try:
            # Find relations where entity appears as source or target
            relations_cursor = self._data.find(
                {"$or": [{"src_id": entity_name}, {"tgt_id": entity_name}]}
            )
            relations = await relations_cursor.to_list(length=None)

            if not relations:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
                return

            # Extract IDs of relations to delete
            relation_ids = [relation["_id"] for relation in relations]
            logger.debug(
                f"[{self.workspace}] Found {len(relation_ids)} relations for entity {entity_name}"
            )

            # Delete the relations
            result = await self._data.delete_many({"_id": {"$in": relation_ids}})
            logger.debug(
                f"[{self.workspace}] Deleted {result.deleted_count} relations for {entity_name}"
            )
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {str(e)}"
            )

        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error searching by prefix in {self.namespace}: {str(e)}"
            )
            return []

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Search for the specific ID in MongoDB
            result = await self._data.find_one({"_id": id})
            if result:
                # Format the result to include id field expected by API
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
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Query MongoDB for multiple IDs
            cursor = self._data.find({"_id": {"$in": ids}})
            results = await cursor.to_list(length=None)

            # Format results to include id field expected by API and preserve ordering
            formatted_map: dict[str, dict[str, Any]] = {}
            for result in results:
                result_dict = dict(result)
                if "_id" in result_dict and "id" not in result_dict:
                    result_dict["id"] = result_dict["_id"]
                key = str(result_dict.get("id", result_dict.get("_id")))
                formatted_map[key] = result_dict

            ordered_results: list[dict[str, Any] | None] = []
            for id_value in ids:
                ordered_results.append(formatted_map.get(str(id_value)))

            return ordered_results
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        try:
            # Query MongoDB for the specified IDs, only retrieving the vector field
            cursor = self._data.find({"_id": {"$in": ids}}, {"vector": 1})
            results = await cursor.to_list(length=None)

            vectors_dict = {}
            for result in results:
                if result and "vector" in result and "_id" in result:
                    # MongoDB stores vectors as arrays, so they should already be lists
                    vectors_dict[result["_id"]] = result["vector"]

            return vectors_dict
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection and recreating vector index.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
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
