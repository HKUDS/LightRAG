import os
import asyncio
import random
from dataclasses import dataclass
from typing import final
import configparser

from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..kg.shared_storage import get_data_init_lock
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")
from neo4j import (
    AsyncGraphDatabase,
    AsyncManagedTransaction,
)
from neo4j.exceptions import TransientError, ResultFailedError

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class MemgraphStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Priority: 1) MEMGRAPH_WORKSPACE env 2) user arg 3) default 'base'
        memgraph_workspace = os.environ.get("MEMGRAPH_WORKSPACE")
        if memgraph_workspace and memgraph_workspace.strip():
            workspace = memgraph_workspace

        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None

    def _get_workspace_label(self) -> str:
        """Return workspace label (guaranteed non-empty during initialization)"""
        return self.workspace

    async def initialize(self):
        async with get_data_init_lock():
            URI = os.environ.get(
                "MEMGRAPH_URI",
                config.get("memgraph", "uri", fallback="bolt://localhost:7687"),
            )
            USERNAME = os.environ.get(
                "MEMGRAPH_USERNAME", config.get("memgraph", "username", fallback="")
            )
            PASSWORD = os.environ.get(
                "MEMGRAPH_PASSWORD", config.get("memgraph", "password", fallback="")
            )
            DATABASE = os.environ.get(
                "MEMGRAPH_DATABASE",
                config.get("memgraph", "database", fallback="memgraph"),
            )

            self._driver = AsyncGraphDatabase.driver(
                URI,
                auth=(USERNAME, PASSWORD),
            )
            self._DATABASE = DATABASE
            try:
                async with self._driver.session(database=DATABASE) as session:
                    # Create index for base nodes on entity_id if it doesn't exist
                    try:
                        workspace_label = self._get_workspace_label()
                        await session.run(
                            f"""CREATE INDEX ON :{workspace_label}(entity_id)"""
                        )
                        logger.info(
                            f"[{self.workspace}] Created index on :{workspace_label}(entity_id) in Memgraph."
                        )
                    except Exception as e:
                        # Index may already exist, which is not an error
                        logger.warning(
                            f"[{self.workspace}] Index creation on :{workspace_label}(entity_id) may have failed or already exists: {e}"
                        )
                    await session.run("RETURN 1")
                    logger.info(f"[{self.workspace}] Connected to Memgraph at {URI}")
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to connect to Memgraph at {URI}: {e}"
                )
                raise

    async def finalize(self):
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self):
        # Memgraph handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_id: The ID of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.

        Raises:
            Exception: If there is an error checking the node existence.
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN count(n) > 0 AS node_exists"
                result = await session.run(query, entity_id=node_id)
                single_result = await result.single()
                await result.consume()  # Ensure result is fully consumed
                return (
                    single_result["node_exists"] if single_result is not None else False
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking node existence for {node_id}: {str(e)}"
                )
                if result is not None:
                    await (
                        result.consume()
                    )  # Ensure the result is consumed even on error
                raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes in the graph.

        Args:
            source_node_id: The ID of the source node.
            target_node_id: The ID of the target node.

        Returns:
            bool: True if the edge exists, False otherwise.

        Raises:
            Exception: If there is an error checking the edge existence.
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = (
                    f"MATCH (a:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(b:`{workspace_label}` {{entity_id: $target_entity_id}}) "
                    "RETURN COUNT(r) > 0 AS edgeExists"
                )
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )  # type: ignore
                single_result = await result.single()
                await result.consume()  # Ensure result is fully consumed
                return (
                    single_result["edgeExists"] if single_result is not None else False
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                if result is not None:
                    await (
                        result.consume()
                    )  # Ensure the result is consumed even on error
                raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found

        Raises:
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                workspace_label = self._get_workspace_label()
                query = (
                    f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN n"
                )
                result = await session.run(query, entity_id=node_id)
                try:
                    records = await result.fetch(
                        2
                    )  # Get 2 records for duplication check

                    if len(records) > 1:
                        logger.warning(
                            f"[{self.workspace}] Multiple nodes found with label '{node_id}'. Using first node."
                        )
                    if records:
                        node = records[0]["n"]
                        node_dict = dict(node)
                        # Remove workspace label from labels list if it exists
                        if "labels" in node_dict:
                            node_dict["labels"] = [
                                label
                                for label in node_dict["labels"]
                                if label != workspace_label
                            ]
                        return node_dict
                    return None
                finally:
                    await result.consume()  # Ensure result is fully consumed
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node for {node_id}: {str(e)}"
                )
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
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                    MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN COUNT(r) AS degree
                """
                result = await session.run(query, entity_id=node_id)
                try:
                    record = await result.single()

                    if not record:
                        logger.warning(
                            f"[{self.workspace}] No node found with label '{node_id}'"
                        )
                        return 0

                    degree = record["degree"]
                    return degree
                finally:
                    await result.consume()  # Ensure result is fully consumed
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node degree for {node_id}: {str(e)}"
                )
                raise

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node labels in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list

        Raises:
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (n:`{workspace_label}`)
                WHERE n.entity_id IS NOT NULL
                RETURN DISTINCT n.entity_id AS label
                ORDER BY label
                """
                result = await session.run(query)
                labels = []
                async for record in result:
                    labels.append(record["label"])
                await result.consume()
                return labels
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting all labels: {str(e)}")
                if result is not None:
                    await (
                        result.consume()
                    )  # Ensure the result is consumed even on error
                raise

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found

        Raises:
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                results = None
                try:
                    workspace_label = self._get_workspace_label()
                    query = f"""MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
                            OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`)
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
                        f"[{self.workspace}] Error getting edges for node {source_node_id}: {str(e)}"
                    )
                    if results is not None:
                        await (
                            results.consume()
                        )  # Ensure results are consumed even on error
                    raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in get_node_edges for {source_node_id}: {str(e)}"
            )
            raise

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
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (start:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(end:`{workspace_label}` {{entity_id: $target_entity_id}})
                RETURN properties(r) as edge_properties
                """
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                records = await result.fetch(2)
                await result.consume()
                if records:
                    edge_result = dict(records[0]["edge_properties"])
                    for key, default_value in {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        if key not in edge_result:
                            edge_result[key] = default_value
                            logger.warning(
                                f"[{self.workspace}] Edge between {source_node_id} and {target_node_id} is missing property: {key}. Using default value: {default_value}"
                            )
                    return edge_result
                return None
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting edge between {source_node_id} and {target_node_id}: {str(e)}"
                )
                if result is not None:
                    await (
                        result.consume()
                    )  # Ensure the result is consumed even on error
                raise

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Memgraph database with manual transaction-level retry logic for transient errors.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError(
                "Memgraph: node properties must contain an 'entity_id' field"
            )

        # Manual transaction-level retry following official Memgraph documentation
        max_retries = 100
        initial_wait_time = 0.2
        backoff_factor = 1.1
        jitter_factor = 0.1

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"[{self.workspace}] Attempting node upsert, attempt {attempt + 1}/{max_retries}"
                )
                async with self._driver.session(database=self._DATABASE) as session:
                    workspace_label = self._get_workspace_label()

                    async def execute_upsert(tx: AsyncManagedTransaction):
                        query = f"""
                        MERGE (n:`{workspace_label}` {{entity_id: $entity_id}})
                        SET n += $properties
                        SET n:`{entity_type}`
                        """
                        result = await tx.run(
                            query, entity_id=node_id, properties=properties
                        )
                        await result.consume()  # Ensure result is fully consumed

                    await session.execute_write(execute_upsert)
                    break  # Success - exit retry loop

            except (TransientError, ResultFailedError) as e:
                # Check if the root cause is a TransientError
                root_cause = e
                while hasattr(root_cause, "__cause__") and root_cause.__cause__:
                    root_cause = root_cause.__cause__

                # Check if this is a transient error that should be retried
                is_transient = (
                    isinstance(root_cause, TransientError)
                    or isinstance(e, TransientError)
                    or "TransientError" in str(e)
                    or "Cannot resolve conflicting transactions" in str(e)
                )

                if is_transient:
                    if attempt < max_retries - 1:
                        # Calculate wait time with exponential backoff and jitter
                        jitter = random.uniform(0, jitter_factor) * initial_wait_time
                        wait_time = (
                            initial_wait_time * (backoff_factor**attempt) + jitter
                        )
                        logger.warning(
                            f"[{self.workspace}] Node upsert failed. Attempt #{attempt + 1} retrying in {wait_time:.3f} seconds... Error: {str(e)}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"[{self.workspace}] Memgraph transient error during node upsert after {max_retries} retries: {str(e)}"
                        )
                        raise
                else:
                    # Non-transient error, don't retry
                    logger.error(
                        f"[{self.workspace}] Non-transient error during node upsert: {str(e)}"
                    )
                    raise
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Unexpected error during node upsert: {str(e)}"
                )
                raise

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels with manual transaction-level retry logic for transient errors.
        Ensures both source and target nodes exist and are unique before creating the edge.
        Uses entity_id property to uniquely identify nodes.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge

        Raises:
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )

        edge_properties = edge_data

        # Manual transaction-level retry following official Memgraph documentation
        max_retries = 100
        initial_wait_time = 0.2
        backoff_factor = 1.1
        jitter_factor = 0.1

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"[{self.workspace}] Attempting edge upsert, attempt {attempt + 1}/{max_retries}"
                )
                async with self._driver.session(database=self._DATABASE) as session:

                    async def execute_upsert(tx: AsyncManagedTransaction):
                        workspace_label = self._get_workspace_label()
                        query = f"""
                        MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})
                        WITH source
                        MATCH (target:`{workspace_label}` {{entity_id: $target_entity_id}})
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
                            await result.fetch(2)
                        finally:
                            await result.consume()  # Ensure result is consumed

                    await session.execute_write(execute_upsert)
                    break  # Success - exit retry loop

            except (TransientError, ResultFailedError) as e:
                # Check if the root cause is a TransientError
                root_cause = e
                while hasattr(root_cause, "__cause__") and root_cause.__cause__:
                    root_cause = root_cause.__cause__

                # Check if this is a transient error that should be retried
                is_transient = (
                    isinstance(root_cause, TransientError)
                    or isinstance(e, TransientError)
                    or "TransientError" in str(e)
                    or "Cannot resolve conflicting transactions" in str(e)
                )

                if is_transient:
                    if attempt < max_retries - 1:
                        # Calculate wait time with exponential backoff and jitter
                        jitter = random.uniform(0, jitter_factor) * initial_wait_time
                        wait_time = (
                            initial_wait_time * (backoff_factor**attempt) + jitter
                        )
                        logger.warning(
                            f"[{self.workspace}] Edge upsert failed. Attempt #{attempt + 1} retrying in {wait_time:.3f} seconds... Error: {str(e)}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"[{self.workspace}] Memgraph transient error during edge upsert after {max_retries} retries: {str(e)}"
                        )
                        raise
                else:
                    # Non-transient error, don't retry
                    logger.error(
                        f"[{self.workspace}] Non-transient error during edge upsert: {str(e)}"
                    )
                    raise
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Unexpected error during edge upsert: {str(e)}"
                )
                raise

    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete

        Raises:
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )

        async def _do_delete(tx: AsyncManagedTransaction):
            workspace_label = self._get_workspace_label()
            query = f"""
            MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
            DETACH DELETE n
            """
            result = await tx.run(query, entity_id=node_id)
            logger.debug(f"[{self.workspace}] Deleted node with label {node_id}")
            await result.consume()

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_delete)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during node deletion: {str(e)}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple

        Raises:
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        for source, target in edges:

            async def _do_delete_edge(tx: AsyncManagedTransaction):
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(target:`{workspace_label}` {{entity_id: $target_entity_id}})
                DELETE r
                """
                result = await tx.run(
                    query, source_entity_id=source, target_entity_id=target
                )
                logger.debug(
                    f"[{self.workspace}] Deleted edge from '{source}' to '{target}'"
                )
                await result.consume()  # Ensure result is fully consumed

            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    await session.execute_write(_do_delete_edge)
            except Exception as e:
                logger.error(f"[{self.workspace}] Error during edge deletion: {str(e)}")
                raise

    async def drop(self) -> dict[str, str]:
        """Drop all data from the current workspace and clean up resources

        This method will delete all nodes and relationships in the Memgraph database.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}

        Raises:
            Exception: If there is an error executing the query
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                workspace_label = self._get_workspace_label()
                query = f"MATCH (n:`{workspace_label}`) DETACH DELETE n"
                result = await session.run(query)
                await result.consume()
                logger.info(
                    f"[{self.workspace}] Dropped workspace {workspace_label} from Memgraph database {self._DATABASE}"
                )
                return {"status": "success", "message": "workspace data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping workspace {workspace_label} from Memgraph database {self._DATABASE}: {e}"
            )
            return {"status": "error", "message": str(e)}

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        return degrees

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
            max_nodes: Maximum nodes to return by BFS, Defaults to 1000
            min_degree: Minimum degree (connections) for nodes to be included. 0=all nodes
            include_orphans: Include orphan nodes (degree=0) even when min_degree > 0

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

        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                if node_label == "*":
                    # First check total node count to determine if graph is truncated
                    count_query = (
                        f"MATCH (n:`{workspace_label}`) RETURN count(n) as total"
                    )
                    count_result = None
                    try:
                        count_result = await session.run(count_query)
                        count_record = await count_result.single()

                        if count_record and count_record["total"] > max_nodes:
                            result.is_truncated = True
                            logger.info(
                                f"Graph truncated: {count_record['total']} nodes found, limited to {max_nodes}"
                            )
                    finally:
                        if count_result:
                            await count_result.consume()

                    # Run main query to get nodes with highest degree
                    main_query = f"""
                    MATCH (n:`{workspace_label}`)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COALESCE(count(r), 0) AS degree
                    ORDER BY degree DESC
                    LIMIT $max_nodes
                    WITH collect({{node: n}}) AS filtered_nodes
                    UNWIND filtered_nodes AS node_info
                    WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                    OPTIONAL MATCH (a)-[r]-(b)
                    WHERE a IN kept_nodes AND b IN kept_nodes
                    RETURN filtered_nodes AS node_info,
                        collect(DISTINCT r) AS relationships
                    """
                    result_set = None
                    try:
                        result_set = await session.run(
                            main_query,
                            {"max_nodes": max_nodes},
                        )
                        record = await result_set.single()
                    finally:
                        if result_set:
                            await result_set.consume()

                else:
                    # Run subgraph query for specific node_label
                    subgraph_query = f"""
                    MATCH (start:`{workspace_label}`)
                    WHERE start.entity_id = $entity_id

                    MATCH path = (start)-[*BFS 0..{max_depth}]-(end:`{workspace_label}`)
                    WHERE ALL(n IN nodes(path) WHERE '{workspace_label}' IN labels(n))
                    WITH collect(DISTINCT end) + start AS all_nodes_unlimited
                    WITH
                    CASE
                        WHEN size(all_nodes_unlimited) <= $max_nodes THEN all_nodes_unlimited
                        ELSE all_nodes_unlimited[0..$max_nodes]
                    END AS limited_nodes,
                    size(all_nodes_unlimited) > $max_nodes AS is_truncated

                    UNWIND limited_nodes AS n
                    MATCH (n)-[r]-(m)
                    WHERE m IN limited_nodes
                    WITH collect(DISTINCT n) AS limited_nodes, collect(DISTINCT r) AS relationships, is_truncated

                    RETURN
                    [node IN limited_nodes | {{node: node}}] AS node_info,
                    relationships,
                    is_truncated
                    """

                    result_set = None
                    try:
                        result_set = await session.run(
                            subgraph_query,
                            {
                                "entity_id": node_label,
                                "max_nodes": max_nodes,
                            },
                        )
                        record = await result_set.single()

                        # If no record found, return empty KnowledgeGraph
                        if not record:
                            logger.debug(
                                f"[{self.workspace}] No nodes found for entity_id: {node_label}"
                            )
                            return result

                        # Check if the result was truncated
                        if record.get("is_truncated"):
                            result.is_truncated = True
                            logger.info(
                                f"[{self.workspace}] Graph truncated: breadth-first search limited to {max_nodes} nodes"
                            )

                    finally:
                        if result_set:
                            await result_set.consume()

                if record:
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
                        f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
                    )

            except Exception as e:
                logger.warning(
                    f"[{self.workspace}] Memgraph error during subgraph query: {str(e)}"
                )

        return result

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (n:`{workspace_label}`)
            RETURN n
            """
            result = await session.run(query)
            nodes = []
            async for record in result:
                node = record["n"]
                node_dict = dict(node)
                # Add node id (entity_id) to the dictionary for easier access
                node_dict["id"] = node_dict.get("entity_id")
                nodes.append(node_dict)
            await result.consume()
            return nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (a:`{workspace_label}`)-[r]-(b:`{workspace_label}`)
            RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
            """
            result = await session.run(query)
            edges = []
            async for record in result:
                edge_properties = record["properties"]
                edge_properties["source"] = record["source"]
                edge_properties["target"] = record["target"]
                edges.append(edge_properties)
            await result.consume()
            return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )

        result = None
        try:
            workspace_label = self._get_workspace_label()
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = f"""
                MATCH (n:`{workspace_label}`)
                WHERE n.entity_id IS NOT NULL
                OPTIONAL MATCH (n)-[r]-()
                WITH n.entity_id AS label, count(r) AS degree
                ORDER BY degree DESC, label ASC
                LIMIT {limit}
                RETURN label
                """
                result = await session.run(query)
                labels = []
                async for record in result:
                    labels.append(record["label"])
                await result.consume()

                logger.debug(
                    f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
                )
                return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            if result is not None:
                await result.consume()
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )

        query_lower = query.lower().strip()

        if not query_lower:
            return []

        result = None
        try:
            workspace_label = self._get_workspace_label()
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
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

                result = await session.run(cypher_query, query_lower=query_lower)
                labels = []
                async for record in result:
                    labels.append(record["label"])
                await result.consume()

                logger.debug(
                    f"[{self.workspace}] Search query '{query}' returned {len(labels)} results (limit: {limit})"
                )
                return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error searching labels: {str(e)}")
            if result is not None:
                await result.consume()
            return []
