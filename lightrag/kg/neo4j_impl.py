import asyncio
import inspect
import os
import re
from dataclasses import dataclass
from typing import Any, final, Optional
import numpy as np
import configparser


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
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")

from neo4j import (  # type: ignore
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
    GraphDatabase,
)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

# Set neo4j logger level to ERROR to suppress warning logs
logging.getLogger("neo4j").setLevel(logging.ERROR)


@final
@dataclass
class Neo4JStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()

        URI = os.environ.get("NEO4J_URI", config.get("neo4j", "uri", fallback=None))
        USERNAME = os.environ.get(
            "NEO4J_USERNAME", config.get("neo4j", "username", fallback=None)
        )
        PASSWORD = os.environ.get(
            "NEO4J_PASSWORD", config.get("neo4j", "password", fallback=None)
        )
        MAX_CONNECTION_POOL_SIZE = int(
            os.environ.get(
                "NEO4J_MAX_CONNECTION_POOL_SIZE",
                config.get("neo4j", "connection_pool_size", fallback=50),
            )
        )
        CONNECTION_TIMEOUT = float(
            os.environ.get(
                "NEO4J_CONNECTION_TIMEOUT",
                config.get("neo4j", "connection_timeout", fallback=30.0),
            ),
        )
        CONNECTION_ACQUISITION_TIMEOUT = float(
            os.environ.get(
                "NEO4J_CONNECTION_ACQUISITION_TIMEOUT",
                config.get("neo4j", "connection_acquisition_timeout", fallback=30.0),
            ),
        )
        MAX_TRANSACTION_RETRY_TIME = float(
            os.environ.get(
                "NEO4J_MAX_TRANSACTION_RETRY_TIME",
                config.get("neo4j", "max_transaction_retry_time", fallback=30.0),
            ),
        )
        DATABASE = os.environ.get(
            "NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", namespace)
        )

        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
            max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
            connection_timeout=CONNECTION_TIMEOUT,
            connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
            max_transaction_retry_time=MAX_TRANSACTION_RETRY_TIME,
        )

        # Try to connect to the database
        with GraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
            max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
            connection_timeout=CONNECTION_TIMEOUT,
            connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
        ) as _sync_driver:
            for database in (DATABASE, None):
                self._DATABASE = database
                connected = False

                try:
                    with _sync_driver.session(database=database) as session:
                        try:
                            session.run("MATCH (n) RETURN n LIMIT 0")
                            logger.info(f"Connected to {database} at {URI}")
                            connected = True
                        except neo4jExceptions.ServiceUnavailable as e:
                            logger.error(
                                f"{database} at {URI} is not available".capitalize()
                            )
                            raise e
                except neo4jExceptions.AuthError as e:
                    logger.error(f"Authentication failed for {database} at {URI}")
                    raise e
                except neo4jExceptions.ClientError as e:
                    if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                        logger.info(
                            f"{database} at {URI} not found. Try to create specified database.".capitalize()
                        )
                        try:
                            with _sync_driver.session() as session:
                                session.run(
                                    f"CREATE DATABASE `{database}` IF NOT EXISTS"
                                )
                                logger.info(f"{database} at {URI} created".capitalize())
                                connected = True
                        except (
                            neo4jExceptions.ClientError,
                            neo4jExceptions.DatabaseError,
                        ) as e:
                            if (
                                e.code
                                == "Neo.ClientError.Statement.UnsupportedAdministrationCommand"
                            ) or (
                                e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                            ):
                                if database is not None:
                                    logger.warning(
                                        "This Neo4j instance does not support creating databases. Try to use Neo4j Desktop/Enterprise version or DozerDB instead. Fallback to use the default database."
                                    )
                            if database is None:
                                logger.error(f"Failed to create {database} at {URI}")
                                raise e

                if connected:
                    break

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        """Close the Neo4j driver and release all resources"""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure driver is closed when context manager exits"""
        await self.close()

    async def index_done_callback(self) -> None:
        # Noe4J handles persistence automatically
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
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = "MATCH (n:base {entity_id: $entity_id}) RETURN count(n) > 0 AS node_exists"
                result = await session.run(query, entity_id=node_id)
                single_result = await result.single()
                await result.consume()  # Ensure result is fully consumed
                return single_result["node_exists"]
            except Exception as e:
                logger.error(f"Error checking node existence for {node_id}: {str(e)}")
                await result.consume()  # Ensure results are consumed even on error
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
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = (
                    "MATCH (a:base {entity_id: $source_entity_id})-[r]-(b:base {entity_id: $target_entity_id}) "
                    "RETURN COUNT(r) > 0 AS edgeExists"
                )
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                single_result = await result.single()
                await result.consume()  # Ensure result is fully consumed
                return single_result["edgeExists"]
            except Exception as e:
                logger.error(
                    f"Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                await result.consume()  # Ensure results are consumed even on error
                raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier.

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = "MATCH (n:base {entity_id: $entity_id}) RETURN n"
                result = await session.run(query, entity_id=node_id)
                try:
                    records = await result.fetch(
                        2
                    )  # Get 2 records for duplication check

                    if len(records) > 1:
                        logger.warning(
                            f"Multiple nodes found with label '{node_id}'. Using first node."
                        )
                    if records:
                        node = records[0]["n"]
                        node_dict = dict(node)
                        # Remove base label from labels list if it exists
                        if "labels" in node_dict:
                            node_dict["labels"] = [
                                label
                                for label in node_dict["labels"]
                                if label != "base"
                            ]
                        logger.debug(f"Neo4j query node {query} return: {node_dict}")
                        return node_dict
                    return None
                finally:
                    await result.consume()  # Ensure result is fully consumed
            except Exception as e:
                logger.error(f"Error getting node for {node_id}: {str(e)}")
                raise

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
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = """
                    MATCH (n:base {entity_id: $entity_id})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN COUNT(r) AS degree
                """
                result = await session.run(query, entity_id=node_id)
                try:
                    record = await result.single()

                    if not record:
                        logger.warning(f"No node found with label '{node_id}'")
                        return 0

                    degree = record["degree"]
                    logger.debug(
                        "Neo4j query node degree for {node_id} return: {degree}"
                    )
                    return degree
                finally:
                    await result.consume()  # Ensure result is fully consumed
            except Exception as e:
                logger.error(f"Error getting node degree for {node_id}: {str(e)}")
                raise

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
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = """
                MATCH (start:base {entity_id: $source_entity_id})-[r]-(end:base {entity_id: $target_entity_id})
                RETURN properties(r) as edge_properties
                """
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                try:
                    records = await result.fetch(2)

                    if len(records) > 1:
                        logger.warning(
                            f"Multiple edges found between '{source_node_id}' and '{target_node_id}'. Using first edge."
                        )
                    if records:
                        try:
                            edge_result = dict(records[0]["edge_properties"])
                            logger.debug(f"Result: {edge_result}")
                            # Ensure required keys exist with defaults
                            required_keys = {
                                "weight": 0.0,
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

                            logger.debug(
                                f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_result}"
                            )
                            return edge_result
                        except (KeyError, TypeError, ValueError) as e:
                            logger.error(
                                f"Error processing edge properties between {source_node_id} "
                                f"and {target_node_id}: {str(e)}"
                            )
                            # Return default edge properties on error
                            return {
                                "weight": 0.0,
                                "source_id": None,
                                "description": None,
                                "keywords": None,
                            }

                    logger.debug(
                        f"{inspect.currentframe().f_code.co_name}: No edge found between {source_node_id} and {target_node_id}"
                    )
                    # Return default edge properties when no edge found
                    return {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
                finally:
                    await result.consume()  # Ensure result is fully consumed

        except Exception as e:
            logger.error(
                f"Error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
            )
            raise

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
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                try:
                    query = """MATCH (n:base {entity_id: $entity_id})
                            OPTIONAL MATCH (n)-[r]-(connected:base)
                            WHERE connected.entity_id IS NOT NULL
                            RETURN n, r, connected"""
                    results = await session.run(query, entity_id=source_node_id)

                    edges = []
                    async for record in results:
                        source_node = record["n"]
                        connected_node = record["connected"]

                        # Skip if either node is None
                        if not source_node or not connected_node:
                            continue

                        source_label = (
                            source_node.get("entity_id")
                            if source_node.get("entity_id")
                            else None
                        )
                        target_label = (
                            connected_node.get("entity_id")
                            if connected_node.get("entity_id")
                            else None
                        )

                        if source_label and target_label:
                            edges.append((source_label, target_label))

                    await results.consume()  # Ensure results are consumed
                    return edges
                except Exception as e:
                    logger.error(
                        f"Error getting edges for node {source_node_id}: {str(e)}"
                    )
                    await results.consume()  # Ensure results are consumed even on error
                    raise
        except Exception as e:
            logger.error(f"Error in get_node_edges for {source_node_id}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        properties = node_data
        entity_type = properties["entity_type"]
        entity_id = properties["entity_id"]
        if "entity_id" not in properties:
            raise ValueError("Neo4j: node properties must contain an 'entity_id' field")

        try:
            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    query = (
                        """
                    MERGE (n:base {entity_id: $properties.entity_id})
                    SET n += $properties
                    SET n:`%s`
                    """
                        % entity_type
                    )
                    result = await tx.run(query, properties=properties)
                    logger.debug(
                        f"Upserted node with entity_id '{entity_id}' and properties: {properties}"
                    )
                    await result.consume()  # Ensure result is fully consumed

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
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
            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    query = """
                    MATCH (source:base {entity_id: $source_entity_id})
                    WITH source
                    MATCH (target:base {entity_id: $target_entity_id})
                    MERGE (source)-[r:DIRECTED]-(target)
                    SET r += $properties
                    RETURN r, source, target
                    """
                    result = await tx.run(
                        query,
                        source_entity_id=source_node_id,
                        target_entity_id=target_node_id,
                        properties=edge_properties,
                    )
                    try:
                        records = await result.fetch(2)
                        if records:
                            logger.debug(
                                f"Upserted edge from '{source_node_id}' to '{target_node_id}'"
                                f"with properties: {edge_properties}"
                            )
                    finally:
                        await result.consume()  # Ensure result is consumed

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        min_degree: int = 0,
        inclusive: bool = False,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.
        Maximum number of nodes is constrained by the environment variable `MAX_GRAPH_NODES` (default: 1000).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. min_degree does not affect nodes directly connected to the matching nodes
            2. Label matching nodes take precedence
            3. Followed by nodes directly connected to the matching nodes
            4. Finally, the degree of the nodes

        Args:
            node_label: Label of the starting node
            max_depth: Maximum depth of the subgraph
            min_degree: Minimum degree of nodes to include. Defaults to 0
            inclusive: Do an inclusive search if true
        Returns:
            KnowledgeGraph: Complete connected subgraph for specified node
        """
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                if node_label == "*":
                    main_query = """
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COALESCE(count(r), 0) AS degree
                    WHERE degree >= $min_degree
                    ORDER BY degree DESC
                    LIMIT $max_nodes
                    WITH collect({node: n}) AS filtered_nodes
                    UNWIND filtered_nodes AS node_info
                    WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                    OPTIONAL MATCH (a)-[r]-(b)
                    WHERE a IN kept_nodes AND b IN kept_nodes
                    RETURN filtered_nodes AS node_info,
                           collect(DISTINCT r) AS relationships
                    """
                    result_set = await session.run(
                        main_query,
                        {"max_nodes": MAX_GRAPH_NODES, "min_degree": min_degree},
                    )

                else:
                    # Main query uses partial matching
                    main_query = """
                    MATCH (start)
                    WHERE
                        CASE
                            WHEN $inclusive THEN start.entity_id CONTAINS $entity_id
                            ELSE start.entity_id = $entity_id
                        END
                    WITH start
                    CALL apoc.path.subgraphAll(start, {
                        relationshipFilter: '',
                        minLevel: 0,
                        maxLevel: $max_depth,
                        bfs: true
                    })
                    YIELD nodes, relationships
                    WITH start, nodes, relationships
                    UNWIND nodes AS node
                    OPTIONAL MATCH (node)-[r]-()
                    WITH node, COALESCE(count(r), 0) AS degree, start, nodes, relationships
                    WHERE node = start OR EXISTS((start)--(node)) OR degree >= $min_degree
                    ORDER BY
                        CASE
                            WHEN node = start THEN 3
                            WHEN EXISTS((start)--(node)) THEN 2
                            ELSE 1
                        END DESC,
                        degree DESC
                    LIMIT $max_nodes
                    WITH collect({node: node}) AS filtered_nodes
                    UNWIND filtered_nodes AS node_info
                    WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                    OPTIONAL MATCH (a)-[r]-(b)
                    WHERE a IN kept_nodes AND b IN kept_nodes
                    RETURN filtered_nodes AS node_info,
                           collect(DISTINCT r) AS relationships
                    """
                    result_set = await session.run(
                        main_query,
                        {
                            "max_nodes": MAX_GRAPH_NODES,
                            "entity_id": node_label,
                            "inclusive": inclusive,
                            "max_depth": max_depth,
                            "min_degree": min_degree,
                        },
                    )

                try:
                    record = await result_set.single()

                    if record:
                        # Handle nodes (compatible with multi-label cases)
                        for node_info in record["node_info"]:
                            node = node_info["node"]
                            node_id = node.id
                            if node_id not in seen_nodes:
                                result.nodes.append(
                                    KnowledgeGraphNode(
                                        id=f"{node_id}",
                                        labels=[node.get("entity_id")],
                                        properties=dict(node),
                                    )
                                )
                                seen_nodes.add(node_id)

                        # Handle relationships (including direction information)
                        for rel in record["relationships"]:
                            edge_id = rel.id
                            if edge_id not in seen_edges:
                                start = rel.start_node
                                end = rel.end_node
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=f"{edge_id}",
                                        type=rel.type,
                                        source=f"{start.id}",
                                        target=f"{end.id}",
                                        properties=dict(rel),
                                    )
                                )
                                seen_edges.add(edge_id)

                        logger.info(
                            f"Process {os.getpid()} graph query return: {len(result.nodes)} nodes, {len(result.edges)} edges"
                        )
                finally:
                    await result_set.consume()  # Ensure result set is consumed

            except neo4jExceptions.ClientError as e:
                logger.warning(f"APOC plugin error: {str(e)}")
                if node_label != "*":
                    logger.warning(
                        "Neo4j: falling back to basic Cypher recursive search..."
                    )
                    if inclusive:
                        logger.warning(
                            "Neo4j: inclusive search mode is not supported in recursive query, using exact matching"
                        )
                    return await self._robust_fallback(
                        node_label, max_depth, min_degree
                    )

        return result

    async def _robust_fallback(
        self, node_label: str, max_depth: int, min_degree: int = 0
    ) -> KnowledgeGraph:
        """
        Fallback implementation when APOC plugin is not available or incompatible.
        This method implements the same functionality as get_knowledge_graph but uses
        only basic Cypher queries and recursive traversal instead of APOC procedures.
        """
        result = KnowledgeGraph()
        visited_nodes = set()
        visited_edges = set()

        async def traverse(
            node: KnowledgeGraphNode,
            edge: Optional[KnowledgeGraphEdge],
            current_depth: int,
        ):
            # Check traversal limits
            if current_depth > max_depth:
                logger.debug(f"Reached max depth: {max_depth}")
                return
            if len(visited_nodes) >= MAX_GRAPH_NODES:
                logger.debug(f"Reached max nodes limit: {MAX_GRAPH_NODES}")
                return

            # Check if node already visited
            if node.id in visited_nodes:
                return

            # Get all edges and target nodes
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = """
                MATCH (a:base {entity_id: $entity_id})-[r]-(b)
                WITH r, b, id(r) as edge_id, id(b) as target_id
                RETURN r, b, edge_id, target_id
                """
                results = await session.run(query, entity_id=node.id)

                # Get all records and release database connection
                records = await results.fetch(
                    1000
                )  # Max neighbour nodes we can handled
                await results.consume()  # Ensure results are consumed

                # Nodes not connected to start node need to check degree
                if current_depth > 1 and len(records) < min_degree:
                    return

                # Add current node to result
                result.nodes.append(node)
                visited_nodes.add(node.id)

                # Add edge to result if it exists and not already added
                if edge and edge.id not in visited_edges:
                    result.edges.append(edge)
                    visited_edges.add(edge.id)

                # Prepare nodes and edges for recursive processing
                nodes_to_process = []
                for record in records:
                    rel = record["r"]
                    edge_id = str(record["edge_id"])
                    if edge_id not in visited_edges:
                        b_node = record["b"]
                        target_id = b_node.get("entity_id")

                        if target_id:  # Only process if target node has entity_id
                            # Create KnowledgeGraphNode for target
                            target_node = KnowledgeGraphNode(
                                id=f"{target_id}",
                                labels=list(f"{target_id}"),
                                properties=dict(b_node.properties),
                            )

                            # Create KnowledgeGraphEdge
                            target_edge = KnowledgeGraphEdge(
                                id=f"{edge_id}",
                                type=rel.type,
                                source=f"{node.id}",
                                target=f"{target_id}",
                                properties=dict(rel),
                            )

                            nodes_to_process.append((target_node, target_edge))
                        else:
                            logger.warning(
                                f"Skipping edge {edge_id} due to missing labels on target node"
                            )

                # Process nodes after releasing database connection
                for target_node, target_edge in nodes_to_process:
                    await traverse(target_node, target_edge, current_depth + 1)

        # Get the starting node's data
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            MATCH (n:base {entity_id: $entity_id})
            RETURN id(n) as node_id, n
            """
            node_result = await session.run(query, entity_id=node_label)
            try:
                node_record = await node_result.single()
                if not node_record:
                    return result

                # Create initial KnowledgeGraphNode
                start_node = KnowledgeGraphNode(
                    id=f"{node_record['n'].get('entity_id')}",
                    labels=list(f"{node_record['n'].get('entity_id')}"),
                    properties=dict(node_record["n"].properties),
                )
            finally:
                await node_result.consume()  # Ensure results are consumed

            # Start traversal with the initial node
            await traverse(start_node, None, 0)

        return result

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node labels in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            # Method 1: Direct metadata query (Available for Neo4j 4.3+)
            # query = "CALL db.labels() YIELD label RETURN label"

            # Method 2: Query compatible with older versions
            query = """
            MATCH (n)
            WHERE n.entity_id IS NOT NULL
            RETURN DISTINCT n.entity_id AS label
            ORDER BY label
            """
            result = await session.run(query)
            labels = []
            try:
                async for record in result:
                    labels.append(record["label"])
            finally:
                await (
                    result.consume()
                )  # Ensure results are consumed even if processing fails
            return labels

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """

        async def _do_delete(tx: AsyncManagedTransaction):
            query = """
            MATCH (n:base {entity_id: $entity_id})
            DETACH DELETE n
            """
            result = await tx.run(query, entity_id=node_id)
            logger.debug(f"Deleted node with label '{node_id}'")
            await result.consume()  # Ensure result is fully consumed

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_delete)
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
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
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:

            async def _do_delete_edge(tx: AsyncManagedTransaction):
                query = """
                MATCH (source:base {entity_id: $source_entity_id})-[r]-(target:base {entity_id: $target_entity_id})
                DELETE r
                """
                result = await tx.run(
                    query, source_entity_id=source, target_entity_id=target
                )
                logger.debug(f"Deleted edge from '{source}' to '{target}'")
                await result.consume()  # Ensure result is fully consumed

            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    await session.execute_write(_do_delete_edge)
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        raise NotImplementedError
