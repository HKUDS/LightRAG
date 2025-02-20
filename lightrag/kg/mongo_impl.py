import os
from dataclasses import dataclass, field
import numpy as np
import configparser
import asyncio

from typing import Any, List, Union, final

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..namespace import NameSpace, is_namespace
from ..utils import logger
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
import pipmaster as pm

if not pm.is_installed("pymongo"):
    pm.install("pymongo")

if not pm.is_installed("motor"):
    pm.install("motor")

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)
from pymongo.operations import SearchIndexModel
from pymongo.errors import PyMongoError

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


class ClientManager:
    _instances = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncIOMotorDatabase:
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
                client = AsyncIOMotorClient(uri)
                db = client.get_database(database_name)
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: AsyncIOMotorDatabase):
        async with cls._lock:
            if db is not None:
                if db is cls._instances["db"]:
                    cls._instances["ref_count"] -= 1
                    if cls._instances["ref_count"] == 0:
                        cls._instances["db"] = None


@final
@dataclass
class MongoKVStorage(BaseKVStorage):
    db: AsyncIOMotorDatabase = field(default=None)
    _data: AsyncIOMotorCollection = field(default=None)

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
        return await self._data.find_one({"_id": id})

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        cursor = self._data.find({"_id": {"$in": ids}})
        return await cursor.to_list()

    async def filter_keys(self, keys: set[str]) -> set[str]:
        cursor = self._data.find({"_id": {"$in": list(keys)}}, {"_id": 1})
        existing_ids = {str(x["_id"]) async for x in cursor}
        return keys - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            update_tasks: list[Any] = []
            for mode, items in data.items():
                for k, v in items.items():
                    key = f"{mode}_{k}"
                    data[mode][k]["_id"] = f"{mode}_{k}"
                    update_tasks.append(
                        self._data.update_one(
                            {"_id": key}, {"$setOnInsert": v}, upsert=True
                        )
                    )
            await asyncio.gather(*update_tasks)
        else:
            update_tasks = []
            for k, v in data.items():
                data[k]["_id"] = k
                update_tasks.append(
                    self._data.update_one({"_id": k}, {"$set": v}, upsert=True)
                )
            await asyncio.gather(*update_tasks)

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            res = {}
            v = await self._data.find_one({"_id": mode + "_" + id})
            if v:
                res[id] = v
                logger.debug(f"llm_response_cache find one by:{id}")
                return res
            else:
                return None
        else:
            return None

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass


@final
@dataclass
class MongoDocStatusStorage(DocStatusStorage):
    db: AsyncIOMotorDatabase = field(default=None)
    _data: AsyncIOMotorCollection = field(default=None)

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
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        update_tasks: list[Any] = []
        for k, v in data.items():
            data[k]["_id"] = k
            update_tasks.append(
                self._data.update_one({"_id": k}, {"$set": v}, upsert=True)
            )
        await asyncio.gather(*update_tasks)

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        cursor = self._data.aggregate(pipeline)
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
            )
            for doc in result
        }

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass


@final
@dataclass
class MongoGraphStorage(BaseGraphStorage):
    """
    A concrete implementation using MongoDB's $graphLookup to demonstrate multi-hop queries.
    """

    db: AsyncIOMotorDatabase = field(default=None)
    collection: AsyncIOMotorCollection = field(default=None)

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._collection_name = self.namespace

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()
            self.collection = await get_or_create_collection(
                self.db, self._collection_name
            )
            logger.debug(f"Use MongoDB as KG {self._collection_name}")

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self.collection = None

    #
    # -------------------------------------------------------------------------
    # HELPER: $graphLookup pipeline
    # -------------------------------------------------------------------------
    #

    async def _graph_lookup(
        self, start_node_id: str, max_depth: int = None
    ) -> List[dict]:
        """
        Performs a $graphLookup starting from 'start_node_id' and returns
        all reachable documents (including the start node itself).

        Pipeline Explanation:
        - 1) $match: We match the start node document by _id = start_node_id.
        - 2) $graphLookup:
            "from": same collection,
            "startWith": "$edges.target" (the immediate neighbors in 'edges'),
            "connectFromField": "edges.target",
            "connectToField": "_id",
            "as": "reachableNodes",
            "maxDepth": max_depth (if provided),
            "depthField": "depth" (used for debugging or filtering).
        - 3) We add an $project or $unwind as needed to extract data.
        """
        pipeline = [
            {"$match": {"_id": start_node_id}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": "$edges.target",
                    "connectFromField": "edges.target",
                    "connectToField": "_id",
                    "as": "reachableNodes",
                    "depthField": "depth",
                }
            },
        ]

        # If you want a limited depth (e.g., only 1 or 2 hops), set maxDepth
        if max_depth is not None:
            pipeline[1]["$graphLookup"]["maxDepth"] = max_depth

        # Return the matching doc plus a field "reachableNodes"
        cursor = self.collection.aggregate(pipeline)
        results = await cursor.to_list(None)

        # If there's no matching node, results = [].
        # Otherwise, results[0] is the start node doc,
        # plus results[0]["reachableNodes"] is the array of connected docs.
        return results

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
        Check if there's a direct single-hop edge from source_node_id to target_node_id.

        We'll do a $graphLookup with maxDepth=0 from the source node—meaning
        “Look up zero expansions.” Actually, for a direct edge check, we can do maxDepth=1
        and then see if the target node is in the "reachableNodes" at depth=0.

        But typically for a direct edge, we might just do a find_one.
        Below is a demonstration approach.
        """
        # We can do a single-hop graphLookup (maxDepth=0 or 1).
        # Then check if the target_node appears among the edges array.
        pipeline = [
            {"$match": {"_id": source_node_id}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": "$edges.target",
                    "connectFromField": "edges.target",
                    "connectToField": "_id",
                    "as": "reachableNodes",
                    "depthField": "depth",
                    "maxDepth": 0,  # means: do not follow beyond immediate edges
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "reachableNodes._id": 1,  # only keep the _id from the subdocs
                }
            },
        ]
        cursor = self.collection.aggregate(pipeline)
        results = await cursor.to_list(None)
        if not results:
            return False

        # results[0]["reachableNodes"] are the immediate neighbors
        reachable_ids = [d["_id"] for d in results[0].get("reachableNodes", [])]
        return target_node_id in reachable_ids

    #
    # -------------------------------------------------------------------------
    # DEGREES
    # -------------------------------------------------------------------------
    #

    async def node_degree(self, node_id: str) -> int:
        """
        Returns the total number of edges connected to node_id (both inbound and outbound).
        The easiest approach is typically two queries:
         - count of edges array in node_id's doc
         - count of how many other docs have node_id in their edges.target.

        But we'll do a $graphLookup demonstration for inbound edges:
        1) Outbound edges: direct from node's edges array
        2) Inbound edges: we can do a special $graphLookup from all docs
           or do an explicit match.

        For demonstration, let's do this in two steps (with second step $graphLookup).
        """
        # --- 1) Outbound edges (direct from doc) ---
        doc = await self.collection.find_one({"_id": node_id}, {"edges": 1})
        if not doc:
            return 0
        outbound_count = len(doc.get("edges", []))

        # --- 2) Inbound edges:
        # A simple way is: find all docs where "edges.target" == node_id.
        # But let's do a $graphLookup from `node_id` in REVERSE.
        # There's a trick to do "reverse" graphLookups: you'd store
        # reversed edges or do a more advanced pipeline. Typically you'd do
        # a direct match. We'll just do a direct match for inbound.
        inbound_count_pipeline = [
            {"$match": {"edges.target": node_id}},
            {
                "$project": {
                    "matchingEdgesCount": {
                        "$size": {
                            "$filter": {
                                "input": "$edges",
                                "as": "edge",
                                "cond": {"$eq": ["$$edge.target", node_id]},
                            }
                        }
                    }
                }
            },
            {"$group": {"_id": None, "totalInbound": {"$sum": "$matchingEdgesCount"}}},
        ]
        inbound_cursor = self.collection.aggregate(inbound_count_pipeline)
        inbound_result = await inbound_cursor.to_list(None)
        inbound_count = inbound_result[0]["totalInbound"] if inbound_result else 0

        return outbound_count + inbound_count

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        If your graph can hold multiple edges from the same src to the same tgt
        (e.g. different 'relation' values), you can sum them. If it's always
        one edge, this is typically 1 or 0.

        We'll do a single-hop $graphLookup from src_id,
        then count how many edges reference tgt_id at depth=0.
        """
        pipeline = [
            {"$match": {"_id": src_id}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": "$edges.target",
                    "connectFromField": "edges.target",
                    "connectToField": "_id",
                    "as": "neighbors",
                    "depthField": "depth",
                    "maxDepth": 0,
                }
            },
            {"$project": {"edges": 1, "neighbors._id": 1, "neighbors.type": 1}},
        ]
        cursor = self.collection.aggregate(pipeline)
        results = await cursor.to_list(None)
        if not results:
            return 0

        # We can simply count how many edges in `results[0].edges` have target == tgt_id.
        edges = results[0].get("edges", [])
        count = sum(1 for e in edges if e.get("target") == tgt_id)
        return count

    #
    # -------------------------------------------------------------------------
    # GETTERS
    # -------------------------------------------------------------------------
    #

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """
        Return the full node document (including "edges"), or None if missing.
        """
        return await self.collection.find_one({"_id": node_id})

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        pipeline = [
            {"$match": {"_id": source_node_id}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": "$edges.target",
                    "connectFromField": "edges.target",
                    "connectToField": "_id",
                    "as": "neighbors",
                    "depthField": "depth",
                    "maxDepth": 0,
                }
            },
            {"$project": {"edges": 1}},
        ]
        cursor = self.collection.aggregate(pipeline)
        docs = await cursor.to_list(None)
        if not docs:
            return None

        for e in docs[0].get("edges", []):
            if e.get("target") == target_node_id:
                return e
        return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """
        Return a list of (source_id, target_id) for direct edges from source_node_id.
        Demonstrates $graphLookup at maxDepth=0, though direct doc retrieval is simpler.
        """
        pipeline = [
            {"$match": {"_id": source_node_id}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": "$edges.target",
                    "connectFromField": "edges.target",
                    "connectToField": "_id",
                    "as": "neighbors",
                    "depthField": "depth",
                    "maxDepth": 0,
                }
            },
            {"$project": {"_id": 0, "edges": 1}},
        ]
        cursor = self.collection.aggregate(pipeline)
        result = await cursor.to_list(None)
        if not result:
            return None

        edges = result[0].get("edges", [])
        return [(source_node_id, e["target"]) for e in edges]

    #
    # -------------------------------------------------------------------------
    # UPSERTS
    # -------------------------------------------------------------------------
    #

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Insert or update a node document. If new, create an empty edges array.
        """
        # By default, preserve existing 'edges'.
        # We'll only set 'edges' to [] on insert (no overwrite).
        update_doc = {"$set": {**node_data}, "$setOnInsert": {"edges": []}}
        await self.collection.update_one({"_id": node_id}, update_doc, upsert=True)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge from source_node_id -> target_node_id with optional 'relation'.
        If an edge with the same target exists, we remove it and re-insert with updated data.
        """
        # Ensure source node exists
        await self.upsert_node(source_node_id, {})

        # Remove existing edge (if any)
        await self.collection.update_one(
            {"_id": source_node_id}, {"$pull": {"edges": {"target": target_node_id}}}
        )

        # Insert new edge
        new_edge = {"target": target_node_id}
        new_edge.update(edge_data)
        await self.collection.update_one(
            {"_id": source_node_id}, {"$push": {"edges": new_edge}}
        )

    #
    # -------------------------------------------------------------------------
    # DELETION
    # -------------------------------------------------------------------------
    #

    async def delete_node(self, node_id: str) -> None:
        """
        1) Remove node's doc entirely.
        2) Remove inbound edges from any doc that references node_id.
        """
        # Remove inbound edges from all other docs
        await self.collection.update_many({}, {"$pull": {"edges": {"target": node_id}}})

        # Remove the node doc
        await self.collection.delete_one({"_id": node_id})

    #
    # -------------------------------------------------------------------------
    # EMBEDDINGS (NOT IMPLEMENTED)
    # -------------------------------------------------------------------------
    #

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        """
        Placeholder for demonstration, raises NotImplementedError.
        """
        raise NotImplementedError("Node embedding is not used in lightrag.")

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
        # Use MongoDB's distinct and aggregation to get all unique labels
        pipeline = [
            {"$group": {"_id": "$_id"}},  # Group by _id
            {"$sort": {"_id": 1}},  # Sort alphabetically
        ]

        cursor = self.collection.aggregate(pipeline)
        labels = []
        async for doc in cursor:
            labels.append(doc["_id"])
        return labels

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        """
        Get complete connected subgraph for specified node (including the starting node itself)

        Args:
            node_label: Label of the nodes to start from
            max_depth: Maximum depth of traversal (default: 5)

        Returns:
            KnowledgeGraph object containing nodes and edges of the subgraph
        """
        label = node_label
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        try:
            if label == "*":
                # Get all nodes and edges
                async for node_doc in self.collection.find({}):
                    node_id = str(node_doc["_id"])
                    if node_id not in seen_nodes:
                        result.nodes.append(
                            KnowledgeGraphNode(
                                id=node_id,
                                labels=[node_doc.get("_id")],
                                properties={
                                    k: v
                                    for k, v in node_doc.items()
                                    if k not in ["_id", "edges"]
                                },
                            )
                        )
                        seen_nodes.add(node_id)

                        # Process edges
                        for edge in node_doc.get("edges", []):
                            edge_id = f"{node_id}-{edge['target']}"
                            if edge_id not in seen_edges:
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id,
                                        type=edge.get("relation", ""),
                                        source=node_id,
                                        target=edge["target"],
                                        properties={
                                            k: v
                                            for k, v in edge.items()
                                            if k not in ["target", "relation"]
                                        },
                                    )
                                )
                                seen_edges.add(edge_id)
            else:
                # Verify if starting node exists
                start_nodes = self.collection.find({"_id": label})
                start_nodes_exist = await start_nodes.to_list(length=1)
                if not start_nodes_exist:
                    logger.warning(f"Starting node with label {label} does not exist!")
                    return result

                # Use $graphLookup for traversal
                pipeline = [
                    {
                        "$match": {"_id": label}
                    },  # Start with nodes having the specified label
                    {
                        "$graphLookup": {
                            "from": self._collection_name,
                            "startWith": "$edges.target",
                            "connectFromField": "edges.target",
                            "connectToField": "_id",
                            "maxDepth": max_depth,
                            "depthField": "depth",
                            "as": "connected_nodes",
                        }
                    },
                ]

                async for doc in self.collection.aggregate(pipeline):
                    # Add the start node
                    node_id = str(doc["_id"])
                    if node_id not in seen_nodes:
                        result.nodes.append(
                            KnowledgeGraphNode(
                                id=node_id,
                                labels=[
                                    doc.get(
                                        "_id",
                                    )
                                ],
                                properties={
                                    k: v
                                    for k, v in doc.items()
                                    if k
                                    not in [
                                        "_id",
                                        "edges",
                                        "connected_nodes",
                                        "depth",
                                    ]
                                },
                            )
                        )
                        seen_nodes.add(node_id)

                    # Add edges from start node
                    for edge in doc.get("edges", []):
                        edge_id = f"{node_id}-{edge['target']}"
                        if edge_id not in seen_edges:
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=edge_id,
                                    type=edge.get("relation", ""),
                                    source=node_id,
                                    target=edge["target"],
                                    properties={
                                        k: v
                                        for k, v in edge.items()
                                        if k not in ["target", "relation"]
                                    },
                                )
                            )
                            seen_edges.add(edge_id)

                    # Add connected nodes and their edges
                    for connected in doc.get("connected_nodes", []):
                        node_id = str(connected["_id"])
                        if node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[connected.get("_id")],
                                    properties={
                                        k: v
                                        for k, v in connected.items()
                                        if k not in ["_id", "edges", "depth"]
                                    },
                                )
                            )
                            seen_nodes.add(node_id)

                            # Add edges from connected nodes
                            for edge in connected.get("edges", []):
                                edge_id = f"{node_id}-{edge['target']}"
                                if edge_id not in seen_edges:
                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            id=edge_id,
                                            type=edge.get("relation", ""),
                                            source=node_id,
                                            target=edge["target"],
                                            properties={
                                                k: v
                                                for k, v in edge.items()
                                                if k not in ["target", "relation"]
                                            },
                                        )
                                    )
                                    seen_edges.add(edge_id)

            logger.info(
                f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
            )

        except PyMongoError as e:
            logger.error(f"MongoDB query failed: {str(e)}")

        return result

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass


@final
@dataclass
class MongoVectorDBStorage(BaseVectorStorage):
    db: AsyncIOMotorDatabase | None = field(default=None)
    _data: AsyncIOMotorCollection | None = field(default=None)

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

            indexes = await self._data.list_search_indexes().to_list(length=None)
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
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        list_data = [
            {
                "_id": k,
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

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Queries the vector database using Atlas Vector Search."""
        # Generate the embedding
        embedding = await self.embedding_func([query])

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
        cursor = self._data.aggregate(pipeline)
        results = await cursor.to_list()

        # Format and return the results
        return [
            {**doc, "id": doc["_id"], "distance": doc.get("score", None)}
            for doc in results
        ]

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError


async def get_or_create_collection(db: AsyncIOMotorDatabase, collection_name: str):
    collection_names = await db.list_collection_names()

    if collection_name not in collection_names:
        collection = await db.create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")
        return collection
    else:
        logger.debug(f"Collection '{collection_name}' already exists.")
        return db.get_collection(collection_name)
