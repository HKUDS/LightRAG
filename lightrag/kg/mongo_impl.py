import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
import pipmaster as pm
import numpy as np

if not pm.is_installed("pymongo"):
    pm.install("pymongo")

if not pm.is_installed("motor"):
    pm.install("motor")

from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Any, Union, List, Tuple

from ..utils import logger
from ..base import BaseKVStorage, BaseGraphStorage
from ..namespace import NameSpace, is_namespace


@dataclass
class MongoKVStorage(BaseKVStorage):
    def __post_init__(self):
        client = MongoClient(
            os.environ.get("MONGO_URI", "mongodb://root:root@localhost:27017/")
        )
        database = client.get_database(os.environ.get("MONGO_DATABASE", "LightRAG"))
        self._data = database.get_collection(self.namespace)
        logger.info(f"Use MongoDB as KV {self.namespace}")

    async def get_by_id(self, id: str) -> dict[str, Any]:
        return self._data.find_one({"_id": id})

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return list(self._data.find({"_id": {"$in": ids}}))

    async def filter_keys(self, data: set[str]) -> set[str]:
        existing_ids = [
            str(x["_id"]) for x in self._data.find({"_id": {"$in": data}}, {"_id": 1})
        ]
        return set([s for s in data if s not in existing_ids])

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            for mode, items in data.items():
                for k, v in tqdm_async(items.items(), desc="Upserting"):
                    key = f"{mode}_{k}"
                    result = self._data.update_one(
                        {"_id": key}, {"$setOnInsert": v}, upsert=True
                    )
                    if result.upserted_id:
                        logger.debug(f"\nInserted new document with key: {key}")
                    data[mode][k]["_id"] = key
        else:
            for k, v in tqdm_async(data.items(), desc="Upserting"):
                self._data.update_one({"_id": k}, {"$set": v}, upsert=True)
                data[k]["_id"] = k

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            res = {}
            v = self._data.find_one({"_id": mode + "_" + id})
            if v:
                res[id] = v
                logger.debug(f"llm_response_cache find one by:{id}")
                return res
            else:
                return None
        else:
            return None

    async def drop(self) -> None:
        """Drop the collection"""
        await self._data.drop()


@dataclass
class MongoGraphStorage(BaseGraphStorage):
    """
    A concrete implementation using MongoDB’s $graphLookup to demonstrate multi-hop queries.
    """

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.client = AsyncIOMotorClient(
            os.environ.get("MONGO_URI", "mongodb://root:root@localhost:27017/")
        )
        self.db = self.client[os.environ.get("MONGO_DATABASE", "LightRAG")]
        self.collection = self.db[os.environ.get("MONGO_KG_COLLECTION", "MDB_KG")]

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

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """
        Return the full node document (including "edges"), or None if missing.
        """
        return await self.collection.find_one({"_id": node_id})

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """
        Return the first edge dict from source_node_id to target_node_id if it exists.
        Uses a single-hop $graphLookup as demonstration, though a direct find is simpler.
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

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[List[Tuple[str, str]], None]:
        """
        Return a list of (target_id, relation) for direct edges from source_node_id.
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
        return [(e["target"], e["relation"]) for e in edges]

    #
    # -------------------------------------------------------------------------
    # UPSERTS
    # -------------------------------------------------------------------------
    #

    async def upsert_node(self, node_id: str, node_data: dict):
        """
        Insert or update a node document. If new, create an empty edges array.
        """
        # By default, preserve existing 'edges'.
        # We'll only set 'edges' to [] on insert (no overwrite).
        update_doc = {"$set": {**node_data}, "$setOnInsert": {"edges": []}}
        await self.collection.update_one({"_id": node_id}, update_doc, upsert=True)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict
    ):
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

    async def delete_node(self, node_id: str):
        """
        1) Remove node’s doc entirely.
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

    async def embed_nodes(self, algorithm: str) -> Tuple[np.ndarray, List[str]]:
        """
        Placeholder for demonstration, raises NotImplementedError.
        """
        raise NotImplementedError("Node embedding is not used in lightrag.")
