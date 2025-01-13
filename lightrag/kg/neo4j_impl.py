import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict

from neo4j import (
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
    GraphDatabase,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import logger
from ..base import BaseGraphStorage


@dataclass
class Neo4JStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with neo4j in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        URI = os.environ["NEO4J_URI"]
        USERNAME = os.environ["NEO4J_USERNAME"]
        PASSWORD = os.environ["NEO4J_PASSWORD"]
        MAX_CONNECTION_POOL_SIZE = os.environ.get("NEO4J_MAX_CONNECTION_POOL_SIZE", 800)
        DATABASE = os.environ.get(
            "NEO4J_DATABASE"
        )  # If this param is None, the home database will be used. If it is not None, the specified database will be used.
        self._DATABASE = DATABASE
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            URI, auth=(USERNAME, PASSWORD)
        )
        _database_name = "home database" if DATABASE is None else f"database {DATABASE}"
        with GraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
            max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
        ) as _sync_driver:
            try:
                with _sync_driver.session(database=DATABASE) as session:
                    try:
                        session.run("MATCH (n) RETURN n LIMIT 0")
                        logger.info(f"Connected to {DATABASE} at {URI}")
                    except neo4jExceptions.ServiceUnavailable as e:
                        logger.error(
                            f"{DATABASE} at {URI} is not available".capitalize()
                        )
                        raise e
            except neo4jExceptions.AuthError as e:
                logger.error(f"Authentication failed for {DATABASE} at {URI}")
                raise e
            except neo4jExceptions.ClientError as e:
                if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                    logger.info(
                        f"{DATABASE} at {URI} not found. Try to create specified database.".capitalize()
                    )
                try:
                    with _sync_driver.session() as session:
                        session.run(f"CREATE DATABASE `{DATABASE}` IF NOT EXISTS")
                        logger.info(f"{DATABASE} at {URI} created".capitalize())
                except neo4jExceptions.ClientError as e:
                    if (
                        e.code
                        == "Neo.ClientError.Statement.UnsupportedAdministrationCommand"
                    ):
                        logger.warning(
                            "This Neo4j instance does not support creating databases. Try to use Neo4j Desktop/Enterprise version or DozerDB instead."
                        )
                    logger.error(f"Failed to create {DATABASE} at {URI}")
                    raise e

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = node_id.strip('"')

        async with self._driver.session(database=self._DATABASE) as session:
            query = (
                f"MATCH (n:`{entity_name_label}`) RETURN count(n) > 0 AS node_exists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["node_exists"]}'
            )
            return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')

        async with self._driver.session(database=self._DATABASE) as session:
            query = (
                f"MATCH (a:`{entity_name_label_source}`)-[r]-(b:`{entity_name_label_target}`) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["edgeExists"]}'
            )
            return single_result["edgeExists"]

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self._driver.session(database=self._DATABASE) as session:
            entity_name_label = node_id.strip('"')
            query = f"MATCH (n:`{entity_name_label}`) RETURN n"
            result = await session.run(query)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = dict(node)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                )
                return node_dict
            return None

    async def node_degree(self, node_id: str) -> int:
        entity_name_label = node_id.strip('"')

        async with self._driver.session(database=self._DATABASE) as session:
            query = f"""
                MATCH (n:`{entity_name_label}`)
                RETURN COUNT{{ (n)--() }} AS totalEdgeCount
            """
            result = await session.run(query)
            record = await result.single()
            if record:
                edge_count = record["totalEdgeCount"]
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}"
                )
                return edge_count
            else:
                return None

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        entity_name_label_source = src_id.strip('"')
        entity_name_label_target = tgt_id.strip('"')
        src_degree = await self.node_degree(entity_name_label_source)
        trg_degree = await self.node_degree(entity_name_label_target)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}:query:src_Degree+trg_degree:result:{degrees}"
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        """
        Find all edges between nodes of two given labels

        Args:
            source_node_label (str): Label of the source nodes
            target_node_label (str): Label of the target nodes

        Returns:
            list: List of all relationships/edges found
        """
        async with self._driver.session(database=self._DATABASE) as session:
            query = f"""
            MATCH (start:`{entity_name_label_source}`)-[r]->(end:`{entity_name_label_target}`)
            RETURN properties(r) as edge_properties
            LIMIT 1
            """.format(
                entity_name_label_source=entity_name_label_source,
                entity_name_label_target=entity_name_label_target,
            )

            result = await session.run(query)
            record = await result.single()
            if record:
                result = dict(record["edge_properties"])
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{result}"
                )
                return result
            else:
                return None

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        node_label = source_node_id.strip('"')

        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        query = f"""MATCH (n:`{node_label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected"""
        async with self._driver.session(database=self._DATABASE) as session:
            results = await session.run(query)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]

                source_label = (
                    list(source_node.labels)[0] if source_node.labels else None
                )
                target_label = (
                    list(connected_node.labels)[0]
                    if connected_node and connected_node.labels
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label))

            return edges

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
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        label = node_id.strip('"')
        properties = node_data

        async def _do_upsert(tx: AsyncManagedTransaction):
            query = f"""
            MERGE (n:`{label}`)
            SET n += $properties
            """
            await tx.run(query, properties=properties)
            logger.debug(
                f"Upserted node with label '{label}' and properties: {properties}"
            )

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_upsert)
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
            )
        ),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        source_node_label = source_node_id.strip('"')
        target_node_label = target_node_id.strip('"')
        edge_properties = edge_data

        async def _do_upsert_edge(tx: AsyncManagedTransaction):
            query = f"""
            MATCH (source:`{source_node_label}`)
            WITH source
            MATCH (target:`{target_node_label}`)
            MERGE (source)-[r:DIRECTED]->(target)
            SET r += $properties
            RETURN r
            """
            await tx.run(query, properties=edge_properties)
            logger.debug(
                f"Upserted edge from '{source_node_label}' to '{target_node_label}' with properties: {edge_properties}"
            )

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")
