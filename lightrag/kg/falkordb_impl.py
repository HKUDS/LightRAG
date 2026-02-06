import os
import re
import asyncio
from dataclasses import dataclass
from typing import final
import configparser
from concurrent.futures import ThreadPoolExecutor

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
import pipmaster as pm

if not pm.is_installed("falkordb"):
    pm.install("falkordb")

import falkordb
import redis.exceptions

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


# Set falkordb logger level to ERROR to suppress warning logs
logging.getLogger("falkordb").setLevel(logging.ERROR)


@final
@dataclass
class FalkorDBStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Check FALKORDB_WORKSPACE environment variable and override workspace if set
        falkordb_workspace = os.environ.get("FALKORDB_WORKSPACE")
        if falkordb_workspace and falkordb_workspace.strip():
            workspace = falkordb_workspace

        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._db = None
        self._graph = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_workspace_label(self) -> str:
        """Get workspace label, return 'base' for compatibility when workspace is empty"""
        workspace = getattr(self, "workspace", None)
        return workspace if workspace else "base"

    async def initialize(self):
        HOST = os.environ.get(
            "FALKORDB_HOST", config.get("falkordb", "host", fallback="localhost")
        )
        PORT = int(
            os.environ.get(
                "FALKORDB_PORT", config.get("falkordb", "port", fallback=6379)
            )
        )
        PASSWORD = os.environ.get(
            "FALKORDB_PASSWORD", config.get("falkordb", "password", fallback=None)
        )
        USERNAME = os.environ.get(
            "FALKORDB_USERNAME", config.get("falkordb", "username", fallback=None)
        )
        GRAPH_NAME = os.environ.get(
            "FALKORDB_GRAPH_NAME",
            config.get(
                "falkordb",
                "graph_name",
                fallback=re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace),
            ),
        )

        try:
            # Create FalkorDB connection
            self._db = falkordb.FalkorDB(
                host=HOST,
                port=PORT,
                password=PASSWORD,
                username=USERNAME,
            )

            # Select the graph (creates if doesn't exist)
            self._graph = self._db.select_graph(GRAPH_NAME)

            # Test connection with a simple query
            await self._run_query("RETURN 1")

            # Create index for workspace nodes on entity_id if it doesn't exist
            workspace_label = self._get_workspace_label()
            try:
                index_query = (
                    f"CREATE INDEX FOR (n:`{workspace_label}`) ON (n.entity_id)"
                )
                await self._run_query(index_query)
                logger.info(
                    f"Created index for {workspace_label} nodes on entity_id in FalkorDB"
                )
            except Exception as e:
                # Index may already exist, which is not an error
                logger.debug(f"Index creation may have failed or already exists: {e}")

            logger.info(f"Connected to FalkorDB at {HOST}:{PORT}, graph: {GRAPH_NAME}")

        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB at {HOST}:{PORT}: {e}")
            raise

    async def finalize(self):
        """Close the FalkorDB connection and release all resources"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._db:
            # FalkorDB doesn't have an explicit close method for the client
            self._db = None
            self._graph = None

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure connection is closed when context manager exits"""
        await self.finalize()

    async def _run_query(self, query: str, params: dict = None):
        """Run a query asynchronously using thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self._graph.query(query, params or {})
        )

    async def index_done_callback(self) -> None:
        # FalkorDB handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given label exists in the database

        Args:
            node_id: Label of the node to check

        Returns:
            bool: True if node exists, False otherwise

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        workspace_label = self._get_workspace_label()
        try:
            query = f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN count(n) > 0 AS node_exists"
            result = await self._run_query(query, {"entity_id": node_id.strip()})
            return result.result_set[0][0] if result.result_set else False
        except Exception as e:
            logger.error(f"Error checking node existence for {node_id}: {str(e)}")
            raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            bool: True if edge exists, False otherwise

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        workspace_label = self._get_workspace_label()
        try:
            query = (
                f"MATCH (a:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(b:`{workspace_label}` {{entity_id: $target_entity_id}}) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            result = await self._run_query(
                query,
                {
                    "source_entity_id": source_node_id,
                    "target_entity_id": target_node_id,
                },
            )
            return result.result_set[0][0] if result.result_set else False
        except Exception as e:
            logger.error(
                f"Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
            )
            raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        workspace_label = self._get_workspace_label()
        try:
            query = f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN n"
            result = await self._run_query(query, {"entity_id": node_id})

            if result.result_set and len(result.result_set) > 0:
                node = result.result_set[0][0]  # Get the first node
                # Convert FalkorDB node to dictionary
                node_dict = {key: value for key, value in node.properties.items()}
                return node_dict
            return None
        except Exception as e:
            logger.error(f"Error getting node for {node_id}: {str(e)}")
            raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        workspace_label = self._get_workspace_label()
        query = f"""
        UNWIND $node_ids AS id
        MATCH (n:`{workspace_label}` {{entity_id: id}})
        RETURN n.entity_id AS entity_id, n
        """
        result = await self._run_query(query, {"node_ids": node_ids})
        nodes = {}

        if result.result_set and len(result.result_set) > 0:
            for record in result.result_set:
                entity_id = record[0]
                node = record[1]
                node_dict = {key: value for key, value in node.properties.items()}
                nodes[entity_id] = node_dict

        return nodes

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node with the given label.
        If multiple nodes have the same label, returns the degree of the first node.
        If no node is found, returns 0.

        Args:
            node_id: The label of the node

        Returns:
            int: The number of relationships the node has, or 0 if no node found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
                MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(r) AS degree
            """
            result = await self._run_query(query, {"entity_id": node_id})

            if result.result_set and len(result.result_set) > 0:
                degree = result.result_set[0][0]
                return degree
            else:
                logger.warning(f"No node found with label '{node_id}'")
                return 0
        except Exception as e:
            logger.error(f"Error getting node degree for {node_id}: {str(e)}")
            raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.

        Args:
            node_ids: List of node labels (entity_id values) to look up.

        Returns:
            A dictionary mapping each node_id to its degree (number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        workspace_label = self._get_workspace_label()
        query = f"""
            UNWIND $node_ids AS id
            MATCH (n:`{workspace_label}` {{entity_id: id}})
            OPTIONAL MATCH (n)-[r]-()
            RETURN n.entity_id AS entity_id, COUNT(r) AS degree
        """
        result = await self._run_query(query, {"node_ids": node_ids})
        degrees = {}

        if result.result_set and len(result.result_set) > 0:
            for record in result.result_set:
                entity_id = record[0]
                degrees[entity_id] = record[1]

        # For any node_id that did not return a record, set degree to 0.
        for nid in node_ids:
            if nid not in degrees:
                logger.warning(f"No node found with label '{nid}'")
                degrees[nid] = 0

        return degrees

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

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        return degrees

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edge_pairs: List of (src, tgt) tuples.

        Returns:
            A dictionary mapping each (src, tgt) tuple to the sum of their degrees.
        """
        # Collect unique node IDs from all edge pairs.
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        # Get degrees for all nodes in one go.
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Sum up degrees for each edge pair.
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            dict: Edge properties if found, default properties if not found or on error

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
            MATCH (start:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(end:`{workspace_label}` {{entity_id: $target_entity_id}})
            RETURN properties(r) as edge_properties
            """
            result = await self._run_query(
                query,
                {
                    "source_entity_id": source_node_id,
                    "target_entity_id": target_node_id,
                },
            )

            if result.result_set and len(result.result_set) > 0:
                edge_result = result.result_set[0][0]  # Get properties dict

                # Ensure required keys exist with defaults
                required_keys = {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }
                for key, default_value in required_keys.items():
                    if key not in edge_result:
                        edge_result[key] = default_value
                        logger.warning(
                            f"Edge between {source_node_id} and {target_node_id} "
                            f"missing {key}, using default: {default_value}"
                        )

                return edge_result

            # Return None when no edge found
            return None

        except Exception as e:
            logger.error(
                f"Error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
            )
            raise

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query.

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """

        workspace_label = self._get_workspace_label()
        query = f"""
        UNWIND $pairs AS pair
        MATCH (start:`{workspace_label}` {{entity_id: pair.src}})-[r]-(end:`{workspace_label}` {{entity_id: pair.tgt}})
        RETURN pair.src AS src_id, pair.tgt AS tgt_id, properties(r) AS edge_properties
        """
        result = await self._run_query(query, {"pairs": pairs})
        edges_dict = {}

        if result.result_set and len(result.result_set) > 0:
            for record in result.result_set:
                if record and len(record) >= 3:
                    src = record[0]
                    tgt = record[1]
                    edge_props = record[2] if record[2] else {}

                    edge_result = {}
                    for key, default in {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        edge_result[key] = edge_props.get(key, default)

                    edges_dict[(src, tgt)] = edge_result

        # Add default properties for pairs not found
        for pair_dict in pairs:
            src = pair_dict["src"]
            tgt = pair_dict["tgt"]
            if (src, tgt) not in edges_dict:
                edges_dict[(src, tgt)] = {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }

        return edges_dict

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found

        Raises:
            ValueError: If source_node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            workspace_label = self._get_workspace_label()
            query = f"""MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
                    OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`)
                    WHERE connected.entity_id IS NOT NULL
                    RETURN n, r, connected"""
            result = await self._run_query(query, {"entity_id": source_node_id})

            edges = []
            if result.result_set:
                for record in result.result_set:
                    source_node = record[0]
                    connected_node = record[2]

                    # Skip if either node is None
                    if not source_node or not connected_node:
                        continue

                    source_label = source_node.properties.get("entity_id")
                    target_label = connected_node.properties.get("entity_id")

                    if source_label and target_label:
                        edges.append((source_label, target_label))

            return edges
        except Exception as e:
            logger.error(f"Error in get_node_edges for {source_node_id}: {str(e)}")
            raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Batch retrieve edges for multiple nodes in one query using UNWIND.
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
        workspace_label = self._get_workspace_label()
        query = f"""
            UNWIND $node_ids AS id
            MATCH (n:`{workspace_label}` {{entity_id: id}})
            OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`)
            RETURN id AS queried_id, n.entity_id AS node_entity_id,
                   connected.entity_id AS connected_entity_id,
                   startNode(r).entity_id AS start_entity_id
        """
        result = await self._run_query(query, {"node_ids": node_ids})

        # Initialize the dictionary with empty lists for each node ID
        edges_dict = {node_id: [] for node_id in node_ids}

        # Process results to include both outgoing and incoming edges
        if result.result_set:
            for record in result.result_set:
                queried_id = record[0]
                node_entity_id = record[1]
                connected_entity_id = record[2]
                start_entity_id = record[3]

                # Skip if either node is None
                if not node_entity_id or not connected_entity_id:
                    continue

                # Determine the actual direction of the edge
                # If the start node is the queried node, it's an outgoing edge
                # Otherwise, it's an incoming edge
                if start_entity_id == node_entity_id:
                    # Outgoing edge: (queried_node -> connected_node)
                    edges_dict[queried_id].append((node_entity_id, connected_entity_id))
                else:
                    # Incoming edge: (connected_node -> queried_node)
                    edges_dict[queried_id].append((connected_entity_id, node_entity_id))

        return edges_dict

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        workspace_label = self._get_workspace_label()
        query = f"""
        UNWIND $chunk_ids AS chunk_id
        MATCH (n:`{workspace_label}`)
        WHERE n.source_id IS NOT NULL AND chunk_id IN split(n.source_id, $sep)
        RETURN DISTINCT n
        """
        result = await self._run_query(
            query, {"chunk_ids": chunk_ids, "sep": GRAPH_FIELD_SEP}
        )
        nodes = []

        if result.result_set:
            for record in result.result_set:
                node = record[0]
                node_dict = {key: value for key, value in node.properties.items()}
                # Add node id (entity_id) to the dictionary for easier access
                node_dict["id"] = node_dict.get("entity_id")
                nodes.append(node_dict)

        return nodes

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        workspace_label = self._get_workspace_label()
        query = f"""
        UNWIND $chunk_ids AS chunk_id
        MATCH (a:`{workspace_label}`)-[r]-(b:`{workspace_label}`)
        WHERE r.source_id IS NOT NULL AND chunk_id IN split(r.source_id, $sep)
        RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
        """
        result = await self._run_query(
            query, {"chunk_ids": chunk_ids, "sep": GRAPH_FIELD_SEP}
        )
        edges = []

        if result.result_set:
            for record in result.result_set:
                edge_properties = record[2]
                edge_properties["source"] = record[0]
                edge_properties["target"] = record[1]
                edges.append(edge_properties)

        return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((redis.exceptions.RedisError, Exception)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the FalkorDB database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        workspace_label = self._get_workspace_label()
        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError(
                "FalkorDB: node properties must contain an 'entity_id' field"
            )

        try:
            query = f"""
            MERGE (n:`{workspace_label}` {{entity_id: $entity_id}})
            SET n += $properties
            SET n:`{entity_type}`
            """
            await self._run_query(
                query, {"entity_id": node_id, "properties": properties}
            )
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((redis.exceptions.RedisError, Exception)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        Ensures both source and target nodes exist and are unique before creating the edge.
        Uses entity_id property to uniquely identify nodes.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge

        Raises:
            ValueError: If either source or target node does not exist or is not unique
        """
        try:
            edge_properties = edge_data
            workspace_label = self._get_workspace_label()
            query = f"""
            MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})
            WITH source
            MATCH (target:`{workspace_label}` {{entity_id: $target_entity_id}})
            MERGE (source)-[r:DIRECTED]-(target)
            SET r += $properties
            RETURN r, source, target
            """
            await self._run_query(
                query,
                {
                    "source_entity_id": source_node_id,
                    "target_entity_id": target_node_id,
                    "properties": edge_properties,
                },
            )
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

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
            max_nodes: Maximum nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        workspace_label = self._get_workspace_label()
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        try:
            if node_label == "*":
                # Get all nodes with highest degree
                query = f"""
                MATCH (n:`{workspace_label}`)
                OPTIONAL MATCH (n)-[r]-()
                WITH n, COALESCE(count(r), 0) AS degree
                ORDER BY degree DESC
                LIMIT $max_nodes
                WITH collect(n) AS nodes
                UNWIND nodes AS node
                OPTIONAL MATCH (node)-[rel]-(connected)
                WHERE connected IN nodes
                RETURN collect(DISTINCT node) AS filtered_nodes,
                       collect(DISTINCT rel) AS relationships
                """
                graph_result = await self._run_query(query, {"max_nodes": max_nodes})
            else:
                # Get subgraph starting from specific node
                # Simple BFS implementation since FalkorDB might not have APOC
                query = f"""
                MATCH path = (start:`{workspace_label}` {{entity_id: $entity_id}})-[*0..{max_depth}]-(connected)
                WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
                UNWIND path_nodes AS node
                WITH collect(DISTINCT node) AS all_nodes, path_rels
                UNWIND path_rels AS rel
                WITH all_nodes, collect(DISTINCT rel) AS all_rels
                RETURN all_nodes[0..{max_nodes}] AS filtered_nodes, all_rels AS relationships
                """
                graph_result = await self._run_query(query, {"entity_id": node_label})

            if graph_result.result_set:
                record = graph_result.result_set[0]
                nodes_list = record[0] if record[0] else []
                relationships_list = record[1] if record[1] else []

                # Check if truncated
                if len(nodes_list) >= max_nodes:
                    result.is_truncated = True

                # Handle nodes
                for node in nodes_list:
                    node_id = str(id(node))  # Use internal node ID
                    if node_id not in seen_nodes:
                        result.nodes.append(
                            KnowledgeGraphNode(
                                id=node_id,
                                labels=[node.properties.get("entity_id", "")],
                                properties=dict(node.properties),
                            )
                        )
                        seen_nodes.add(node_id)

                # Handle relationships
                for rel in relationships_list:
                    edge_id = str(id(rel))  # Use internal relationship ID
                    if edge_id not in seen_edges:
                        # Get start and end node IDs
                        start_node_id = str(rel.src_node)
                        end_node_id = str(rel.dest_node)

                        result.edges.append(
                            KnowledgeGraphEdge(
                                id=edge_id,
                                type=rel.relation,
                                source=start_node_id,
                                target=end_node_id,
                                properties=dict(rel.properties),
                            )
                        )
                        seen_edges.add(edge_id)

                logger.info(
                    f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
                )

        except Exception as e:
            logger.error(f"Error in get_knowledge_graph: {str(e)}")
            # Return empty graph on error
            pass

        return result

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node labels in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list
        """
        workspace_label = self._get_workspace_label()
        query = f"""
        MATCH (n:`{workspace_label}`)
        WHERE n.entity_id IS NOT NULL
        RETURN DISTINCT n.entity_id AS label
        ORDER BY label
        """
        result = await self._run_query(query)
        labels = []

        if result.result_set:
            for record in result.result_set:
                labels.append(record[0])

        return labels

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((redis.exceptions.RedisError, Exception)),
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """
        try:
            workspace_label = self._get_workspace_label()
            query = f"""
            MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
            DETACH DELETE n
            """
            await self._run_query(query, {"entity_id": node_id})
            logger.debug(f"Deleted node with label '{node_id}'")
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((redis.exceptions.RedisError, Exception)),
    )
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        for node in nodes:
            await self.delete_node(node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((redis.exceptions.RedisError, Exception)),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(target:`{workspace_label}` {{entity_id: $target_entity_id}})
                DELETE r
                """
                await self._run_query(
                    query, {"source_entity_id": source, "target_entity_id": target}
                )
                logger.debug(f"Deleted edge from '{source}' to '{target}'")
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        workspace_label = self._get_workspace_label()
        query = f"""
        MATCH (n:`{workspace_label}`)
        RETURN n
        """
        result = await self._run_query(query)
        nodes = []

        if result.result_set:
            for record in result.result_set:
                node = record[0]
                node_dict = {key: value for key, value in node.properties.items()}
                # Add node id (entity_id) to the dictionary for easier access
                node_dict["id"] = node_dict.get("entity_id")
                nodes.append(node_dict)

        return nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        workspace_label = self._get_workspace_label()
        query = f"""
        MATCH (a:`{workspace_label}`)-[r]-(b:`{workspace_label}`)
        RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
        """
        result = await self._run_query(query)
        edges = []

        if result.result_set:
            for record in result.result_set:
                edge_properties = record[2]
                edge_properties["source"] = record[0]
                edge_properties["target"] = record[1]
                edges.append(edge_properties)

        return edges

    async def drop(self) -> dict[str, str]:
        """Drop all data from current workspace storage and clean up resources

        This method will delete all nodes and relationships in the current workspace only.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "workspace data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        workspace_label = self._get_workspace_label()
        try:
            # Delete all nodes and relationships in current workspace only
            query = f"MATCH (n:`{workspace_label}`) DETACH DELETE n"
            await self._run_query(query)

            logger.info(
                f"Process {os.getpid()} drop FalkorDB workspace '{workspace_label}'"
            )
            return {
                "status": "success",
                "message": f"workspace '{workspace_label}' data dropped",
            }
        except Exception as e:
            logger.error(f"Error dropping FalkorDB workspace '{workspace_label}': {e}")
            return {"status": "error", "message": str(e)}

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
            MATCH (n:`{workspace_label}`)
            WHERE n.entity_id IS NOT NULL
            OPTIONAL MATCH (n)-[r]-()
            WITH n.entity_id AS label, count(r) AS degree
            ORDER BY degree DESC, label ASC
            LIMIT {limit}
            RETURN label
            """
            result = await self._run_query(query)
            labels = []

            if result.result_set:
                for record in result.result_set:
                    labels.append(record[0])

            logger.debug(
                f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """
        workspace_label = self._get_workspace_label()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        try:
            # FalkorDB search using CONTAINS with relevance scoring
            cypher_query = f"""
            MATCH (n:`{workspace_label}`)
            WHERE n.entity_id IS NOT NULL
            WITH n.entity_id AS label, toLower(n.entity_id) AS label_lower
            WHERE label_lower CONTAINS $query_lower
            WITH label, label_lower,
                 CASE
                     WHEN label_lower = $query_lower THEN 1000
                     WHEN label_lower STARTS WITH $query_lower THEN 500
                     ELSE 100 - size(label)
                 END AS score
            ORDER BY score DESC, label ASC
            LIMIT {limit}
            RETURN label
            """

            result = await self._run_query(cypher_query, {"query_lower": query_lower})
            labels = []

            if result.result_set:
                for record in result.result_set:
                    labels.append(record[0])

            logger.debug(
                f"[{self.workspace}] Search query '{query}' returned {len(labels)} results (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error searching labels: {str(e)}")
            return []
