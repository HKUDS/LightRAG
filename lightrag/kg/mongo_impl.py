import os
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

import pipmaster as pm

if not pm.is_installed("pymongo"):
    pm.install("pymongo")

from pymongo import AsyncMongoClient  # type: ignore
from pymongo.asynchronous.database import AsyncDatabase  # type: ignore
from pymongo.asynchronous.collection import AsyncCollection  # type: ignore
from pymongo.operations import SearchIndexModel  # type: ignore
from pymongo.errors import PyMongoError  # type: ignore

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))
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

    def __post_init__(self):
        self._collection_name = self.namespace

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()
            self._data = await get_or_create_collection(self.db, self._collection_name)
            logger.debug(f"Use MongoDB as KV {self._collection_name}")

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
        docs = await cursor.to_list()
        # Ensure time fields are present for all documents
        for doc in docs:
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
        return docs

    async def filter_keys(self, keys: set[str]) -> set[str]:
        cursor = self._data.find({"_id": {"$in": list(keys)}}, {"_id": 1})
        existing_ids = {str(x["_id"]) async for x in cursor}
        return keys - existing_ids

    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage

        Returns:
            Dictionary containing all stored data
        """
        cursor = self._data.find({})
        result = {}
        async for doc in cursor:
            doc_id = doc.pop("_id")
            # Ensure time fields are present for all documents
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
            result[doc_id] = doc
        return result

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        # Unified handling for all namespaces with flattened keys
        # Use bulk_write for better performance
        from pymongo import UpdateOne

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
                f"Deleted {result.deleted_count} documents from {self.namespace}"
            )
        except PyMongoError as e:
            logger.error(f"Error deleting documents from {self.namespace}: {e}")

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        """Delete specific records from storage by cache mode

        Args:
            modes (list[str]): List of cache modes to be dropped from storage

        Returns:
            bool: True if successful, False otherwise
        """
        if not modes:
            return False

        try:
            # Build regex pattern to match flattened key format: mode:cache_type:hash
            pattern = f"^({'|'.join(modes)}):"
            result = await self._data.delete_many({"_id": {"$regex": pattern}})
            logger.info(f"Deleted {result.deleted_count} documents by modes: {modes}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache by modes {modes}: {e}")
            return False

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            result = await self._data.delete_many({})
            deleted_count = result.deleted_count

            logger.info(
                f"Dropped {deleted_count} documents from doc status {self._collection_name}"
            )
            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped",
            }
        except PyMongoError as e:
            logger.error(f"Error dropping doc status {self._collection_name}: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class MongoDocStatusStorage(DocStatusStorage):
    db: AsyncDatabase = field(default=None)
    _data: AsyncCollection = field(default=None)

    def __post_init__(self):
        self._collection_name = self.namespace

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()
            self._data = await get_or_create_collection(self.db, self._collection_name)
            logger.debug(f"Use MongoDB as DocStatus {self._collection_name}")

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._data = None

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        return await self._data.find_one({"_id": id})

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        cursor = self._data.find({"_id": {"$in": ids}})
        return await cursor.to_list()

    async def filter_keys(self, data: set[str]) -> set[str]:
        cursor = self._data.find({"_id": {"$in": list(data)}}, {"_id": 1})
        existing_ids = {str(x["_id"]) async for x in cursor}
        return data - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"Inserting {len(data)} to {self.namespace}")
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
        cursor = self._data.aggregate(pipeline, allowDiskUse=True)
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
        return {
            doc["_id"]: DocProcessingStatus(
                content=doc["content"],
                content_summary=doc.get("content_summary"),
                content_length=doc["content_length"],
                status=doc["status"],
                created_at=doc.get("created_at"),
                updated_at=doc.get("updated_at"),
                chunks_count=doc.get("chunks_count", -1),
                file_path=doc.get("file_path", doc["_id"]),
                chunks_list=doc.get("chunks_list", []),
            )
            for doc in result
        }

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            result = await self._data.delete_many({})
            deleted_count = result.deleted_count

            logger.info(
                f"Dropped {deleted_count} documents from doc status {self._collection_name}"
            )
            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped",
            }
        except PyMongoError as e:
            logger.error(f"Error dropping doc status {self._collection_name}: {e}")
            return {"status": "error", "message": str(e)}

    async def delete(self, ids: list[str]) -> None:
        await self._data.delete_many({"_id": {"$in": ids}})


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

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._collection_name = self.namespace
        self._edge_collection_name = f"{self._collection_name}_edges"

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()
            self.collection = await get_or_create_collection(
                self.db, self._collection_name
            )
            self.edge_collection = await get_or_create_collection(
                self.db, self._edge_collection_name
            )
            logger.debug(f"Use MongoDB as KG {self._collection_name}")

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

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all nodes that are associated with the given chunk_ids.

        Args:
            chunk_ids (list[str]): A list of chunk IDs to find associated nodes for.

        Returns:
            list[dict]: A list of nodes, where each node is a dictionary of its properties.
                        An empty list if no matching nodes are found.
        """
        if not chunk_ids:
            return []

        cursor = self.collection.find({"source_ids": {"$in": chunk_ids}})
        return [doc async for doc in cursor]

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all edges that are associated with the given chunk_ids.

        Args:
            chunk_ids (list[str]): A list of chunk IDs to find associated edges for.

        Returns:
            list[dict]: A list of edges, where each edge is a dictionary of its properties.
                        An empty list if no matching edges are found.
        """
        if not chunk_ids:
            return []

        cursor = self.edge_collection.find({"source_ids": {"$in": chunk_ids}})

        edges = []
        async for edge in cursor:
            edge["source"] = edge["source_node_id"]
            edge["target"] = edge["target_node_id"]
            edges.append(edge)

        return edges

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
        self, max_depth: int = 3, max_nodes: int = MAX_GRAPH_NODES
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
        depth: int = 0,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
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
        depth=0,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
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
        self, node_label: str, max_depth: int = 3, max_nodes: int = MAX_GRAPH_NODES
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
            logger.warning(f"Starting node with label {node_label} does not exist!")
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
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000

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
                f"Subgraph query successful in {duration:.4f} seconds | Node count: {len(result.nodes)} | Edge count: {len(result.edges)} | Truncated: {result.is_truncated}"
            )

        except PyMongoError as e:
            # Handle memory limit errors specifically
            if "memory limit" in str(e).lower() or "sort exceeded" in str(e).lower():
                logger.warning(
                    f"MongoDB memory limit exceeded, falling back to simple query: {str(e)}"
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
                        f"Fallback query completed | Node count: {len(result.nodes)}"
                    )
                except PyMongoError as fallback_error:
                    logger.error(f"Fallback query also failed: {str(fallback_error)}")
            else:
                logger.error(f"MongoDB query failed: {str(e)}")

        return result

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """
        logger.info(f"Deleting {len(nodes)} nodes")
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

        logger.debug(f"Successfully deleted nodes: {nodes}")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        logger.info(f"Deleting {len(edges)} edges")
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

        logger.debug(f"Successfully deleted edges: {edges}")

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            result = await self.collection.delete_many({})
            deleted_count = result.deleted_count

            logger.info(
                f"Dropped {deleted_count} documents from graph {self._collection_name}"
            )

            result = await self.edge_collection.delete_many({})
            edge_count = result.deleted_count
            logger.info(
                f"Dropped {edge_count} edges from graph {self._edge_collection_name}"
            )

            return {
                "status": "success",
                "message": f"{deleted_count} documents and {edge_count} edges dropped",
            }
        except PyMongoError as e:
            logger.error(f"Error dropping graph {self._collection_name}: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class MongoVectorDBStorage(BaseVectorStorage):
    db: AsyncDatabase | None = field(default=None)
    _data: AsyncCollection | None = field(default=None)

    def __post_init__(self):
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._collection_name = self.namespace
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()
            self._data = await get_or_create_collection(self.db, self._collection_name)

            # Ensure vector index exists
            await self.create_vector_index_if_not_exists()

            logger.debug(f"Use MongoDB as VDB {self._collection_name}")

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._data = None

    async def create_vector_index_if_not_exists(self):
        """Creates an Atlas Vector Search index."""
        try:
            index_name = "vector_knn_index"

            indexes_cursor = await self._data.list_search_indexes()
            indexes = await indexes_cursor.to_list(length=None)
            for index in indexes:
                if index["name"] == index_name:
                    logger.debug("vector index already exist")
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
                name=index_name,
                type="vectorSearch",
            )

            await self._data.create_search_index(search_index_model)
            logger.info("Vector index created successfully.")

        except PyMongoError as _:
            logger.debug("vector index already exist")

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"Inserting {len(data)} to {self.namespace}")
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
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Queries the vector database using Atlas Vector Search."""
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
                    "index": "vector_knn_index",  # Ensure this matches the created index name
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
        logger.debug(f"Deleting {len(ids)} vectors from {self.namespace}")
        if not ids:
            return

        # Convert to list if it's a set (MongoDB BSON cannot encode sets)
        if isinstance(ids, set):
            ids = list(ids)

        try:
            result = await self._data.delete_many({"_id": {"$in": ids}})
            logger.debug(
                f"Successfully deleted {result.deleted_count} vectors from {self.namespace}"
            )
        except PyMongoError as e:
            logger.error(
                f"Error while deleting vectors from {self.namespace}: {str(e)}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name

        Args:
            entity_name: Name of the entity to delete
        """
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            result = await self._data.delete_one({"_id": entity_id})
            if result.deleted_count > 0:
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found in storage")
        except PyMongoError as e:
            logger.error(f"Error deleting entity {entity_name}: {str(e)}")

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
                logger.debug(f"No relations found for entity {entity_name}")
                return

            # Extract IDs of relations to delete
            relation_ids = [relation["_id"] for relation in relations]
            logger.debug(
                f"Found {len(relation_ids)} relations for entity {entity_name}"
            )

            # Delete the relations
            result = await self._data.delete_many({"_id": {"$in": relation_ids}})
            logger.debug(f"Deleted {result.deleted_count} relations for {entity_name}")
        except PyMongoError as e:
            logger.error(f"Error deleting relations for {entity_name}: {str(e)}")

        except PyMongoError as e:
            logger.error(f"Error searching by prefix in {self.namespace}: {str(e)}")
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
            logger.error(f"Error retrieving vector data for ID {id}: {e}")
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

            # Format results to include id field expected by API
            formatted_results = []
            for result in results:
                result_dict = dict(result)
                if "_id" in result_dict and "id" not in result_dict:
                    result_dict["id"] = result_dict["_id"]
                formatted_results.append(result_dict)

            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving vector data for IDs {ids}: {e}")
            return []

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
                f"Dropped {deleted_count} documents from vector storage {self._collection_name} and recreated vector index"
            )
            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped and vector index recreated",
            }
        except PyMongoError as e:
            logger.error(f"Error dropping vector storage {self._collection_name}: {e}")
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
