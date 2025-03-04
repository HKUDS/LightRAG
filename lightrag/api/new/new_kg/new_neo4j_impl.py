import asyncio
from dataclasses import dataclass
import inspect
import os
from typing import Any, Dict, List, Union

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

from lightrag.kg.neo4j_impl import Neo4JStorage as BaseGraphStorage


@dataclass
class NewNeo4JStorage(BaseGraphStorage):

    def __init__(self, namespace, global_config, embedding_func):
        logger.info("Initializing New Neo4JStorage")

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
        DATABASE = os.environ.get(
            "NEO4J_DATABASE"
        )  # If this param is None, the home database will be used. If it is not None, the specified database will be used.
        self._DATABASE = DATABASE
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            URI, auth=(USERNAME, PASSWORD)
        )

        _database_name = "home database" if DATABASE is None else f"database {DATABASE}"
        with GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD)) as _sync_driver:
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
        super().__post_init__()

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
    async def delete_all(self):
        # 删除全部实体和关系
        query = f"""
            MATCH (n)-[r]->(m)
            DELETE n, r, m
            RETURN n as source_node, r as relationship, m as target_node
            UNION ALL
            MATCH (n)
            WHERE NOT (n)--()
            DELETE n
            RETURN n as source_node, null as relationship, null as target_node
            """
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.run(query)
        except Exception as e:
            logger.error(f"Error during delete all: {str(e)}")
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
    async def query_all(self) -> List[Dict[str, Any]]:
        """
        Retrieves all nodes and their relationships in the Neo4j database.

        Returns:
            list: List of dictionaries containing node and relationship information
        """
        query = f"""
            MATCH (n)-[r]->(m)
            RETURN n AS source_node, r AS relationship, m AS target_node
            UNION
            MATCH (n)
            WHERE NOT (n)--()
            RETURN n AS source_node, null AS relationship, null AS target_node
            """
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                result = await session.run(query)
                entities = []
                async for record in result:
                    s_node = record["source_node"]
                    r_ship = record["relationship"]
                    t_node = record["target_node"]
                    # 获取源节点信息
                    source_node = {
                        "id": (
                            s_node.element_id
                            if s_node and hasattr(s_node, "element_id")
                            else None
                        ),
                        "labels": (
                            list(s_node.labels)
                            if s_node and hasattr(s_node, "labels")
                            else []
                        ),
                        "properties": dict(s_node.items()) if s_node else None,
                    }
                    # 获取关系信息
                    relationship = (
                        {
                            "id": (
                                r_ship.element_id
                                if r_ship and hasattr(r_ship, "element_id")
                                else None
                            ),
                            "type": (
                                r_ship.type
                                if r_ship and hasattr(r_ship, "type")
                                else None
                            ),
                            "properties": dict(r_ship.items()) if r_ship else None,
                        }
                        if r_ship
                        else None
                    )

                    # 获取目标节点信息
                    target_node = (
                        {
                            "id": (
                                t_node.element_id
                                if t_node and hasattr(t_node, "element_id")
                                else None
                            ),
                            "labels": (
                                list(t_node.labels)
                                if t_node and hasattr(t_node, "labels")
                                else []
                            ),
                            "properties": dict(t_node.items()) if t_node else None,
                        }
                        if t_node
                        else None
                    )

                    entities.append(
                        {
                            "source_node": source_node,
                            "relationship": relationship,
                            "target_node": target_node,
                        }
                    )

                return entities
        except Exception as e:
            logger.error(f"Error occurred while querying all nodes: {e}")

        async with self._driver.session(database=self._DATABASE) as session:
            entity_name_label = node_label.strip('"')
            query = f"""
                MATCH (n:`{entity_name_label}`) RETURN n
                """
            result = await session.run(query)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = {
                    "id": (
                        node.element_id
                        if node and hasattr(node, "element_id")
                        else None
                    ),
                    "labels": (
                        list(node.labels) if node and hasattr(node, "labels") else []
                    ),
                    "properties": dict(node.items()) if node else None,
                }
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                )
                return node_dict
            return None