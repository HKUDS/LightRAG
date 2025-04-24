import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, final

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..utils import logger
import configparser
from dotenv import load_dotenv

# Ensure kuzu is installed
import pipmaster as pm

if not pm.is_installed("kuzu"):
    pm.install("kuzu>=0.9.0")
if not pm.is_installed("numpy"):
    pm.install("numpy>=2.2.5")

import kuzu
from kuzu import AsyncConnection, Database

# --- Configuration Loading ---

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

# --- Kuzu Client Management ---


class KuzuClientManager:
    """Manages singleton Kuzu Database and AsyncConnection instances."""

    _instances: dict[str, Any] = {"db": None, "async_conn": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @staticmethod
    def get_config() -> dict[str, Any]:
        """Loads Kuzu configuration from environment variables and config.ini."""
        config = configparser.ConfigParser()
        # Assume config.ini is in the same directory or adjust path as needed
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        if os.path.exists(config_path):
            config.read(config_path, "utf-8")
        else:
            logger.warning(
                "config.ini not found, using environment variables or defaults."
            )

        kuzu_config = {
            "database_path": os.environ.get(
                "KUZU_DATABASE_PATH",
                config.get(
                    "kuzu", "database_path", fallback=":memory:"
                ),  # Default to in-memory
            ),
            "max_num_threads": int(
                os.environ.get(
                    "KUZU_MAX_NUM_THREADS",
                    config.get(
                        "kuzu", "max_num_threads", fallback=0
                    ),  # 0 means use system default
                )
            ),
            "max_concurrent_queries": int(
                os.environ.get(
                    "KUZU_MAX_CONCURRENT_QUERIES",
                    config.get(
                        "kuzu", "max_concurrent_queries", fallback=4
                    ),  # Default for AsyncConnection
                )
            ),
            "buffer_pool_size": int(
                os.environ.get(
                    "KUZU_BUFFER_POOL_SIZE",
                    config.get(
                        "kuzu", "buffer_pool_size", fallback=0
                    ),  # fallback to buffer_pool_size to 80% of system memory
                )
            ),
            # Add other Kuzu DB options if needed (buffer_pool_size, etc.)
        }
        logger.info(f"Kuzu Configuration: {kuzu_config}")
        return kuzu_config

    @classmethod
    async def get_client(cls) -> tuple[Database, AsyncConnection]:
        """Gets or creates the singleton Kuzu Database and AsyncConnection."""
        async with cls._lock:
            if cls._instances["db"] is None or cls._instances["async_conn"] is None:
                config = cls.get_config()
                db_path = config["database_path"]
                max_threads = config["max_num_threads"]
                max_concurrent = config["max_concurrent_queries"]

                # Ensure directory exists if not in-memory
                if db_path != ":memory:":
                    db_dir = os.path.dirname(db_path)
                    if db_dir and not os.path.exists(db_dir):
                        os.makedirs(db_dir, exist_ok=True)
                        logger.info(f"Created Kuzu database directory: {db_dir}")

                try:
                    # Initialize Kuzu Database
                    db = kuzu.Database(
                        database_path=db_path,
                        max_num_threads=max_threads,
                        # Add other config options here if needed
                    )
                    logger.info(f"Initialized Kuzu Database at: {db_path}")

                    # Initialize Kuzu AsyncConnection
                    async_conn = kuzu.AsyncConnection(
                        database=db,
                        max_concurrent_queries=max_concurrent,
                        max_threads_per_query=max_threads,  # Can be different from db max_threads
                    )
                    logger.info(
                        f"Initialized Kuzu AsyncConnection with max_concurrent_queries={max_concurrent}"
                    )

                    cls._instances["db"] = db
                    cls._instances["async_conn"] = async_conn
                    cls._instances["ref_count"] = 0  # Reset ref count on new creation
                except Exception as e:
                    logger.error(f"Failed to initialize Kuzu DB/Connection: {e}")
                    raise

            cls._instances["ref_count"] += 1
            logger.debug(
                f"Kuzu client reference count incremented to: {cls._instances['ref_count']}"
            )
            return cls._instances["db"], cls._instances["async_conn"]

    @classmethod
    async def release_client(cls, db: Database, async_conn: AsyncConnection):
        """Decrements reference count and closes connections if count reaches zero."""
        async with cls._lock:
            # Check if the provided instances match the managed singletons
            if (
                db is cls._instances["db"]
                and async_conn is cls._instances["async_conn"]
            ):
                cls._instances["ref_count"] -= 1
                logger.debug(
                    f"Kuzu client reference count decremented to: {cls._instances['ref_count']}"
                )
                if cls._instances["ref_count"] <= 0:
                    logger.info("Closing Kuzu AsyncConnection and Database...")
                    try:
                        if cls._instances["async_conn"]:
                            cls._instances[
                                "async_conn"
                            ].close()  # Closes pool and threads
                        # Kuzu Database itself doesn't have an explicit close in the same way,
                        # relies on garbage collection or __del__.
                        # We nullify references to allow GC.
                        cls._instances["async_conn"] = None
                        cls._instances["db"] = None
                        cls._instances["ref_count"] = 0  # Ensure it's 0
                        logger.info(
                            "Kuzu AsyncConnection closed and Database references released."
                        )
                    except Exception as e:
                        logger.error(f"Error closing Kuzu resources: {e}")
            else:
                # This case shouldn't happen if managed correctly, but log a warning
                logger.warning(
                    "Attempted to release Kuzu client instances that don't match the managed singletons."
                )


# --- Kuzu Graph Storage Implementation ---


@final
@dataclass
class KuzuGraphStorage(BaseGraphStorage):
    """
    Graph storage implementation using KuzuDB.

    Manages nodes and relationships within a Kuzu database, providing
    asynchronous operations compliant with BaseGraphStorage.
    """

    _db: Database | None = field(default=None, init=False)
    _async_conn: AsyncConnection | None = field(default=None, init=False)
    _node_table: str = field(default="base", init=False)
    _edge_table: str = field(default="DIRECTED", init=False)
    _entity_id_prop: str = field(
        default="entity_id", init=False
    )  # Property name for node identifier

    def __post_init__(self):
        """Additional initialization after dataclass setup."""
        # Namespace could potentially be used to prefix table names if needed,
        # but Kuzu manages data within the single DB file path.
        # We'll stick to 'base' and 'DIRECTED' for now, similar to other backends.
        logger.info(f"KuzuGraphStorage initialized for namespace: {self.namespace}")
        # MAX_GRAPH_NODES is loaded globally

    async def initialize(self):
        """Initializes the Kuzu database connection and ensures schema exists."""
        if self._db is None or self._async_conn is None:
            self._db, self._async_conn = await KuzuClientManager.get_client()
            await self._ensure_schema()

    async def finalize(self):
        """Releases the Kuzu database connection."""
        if self._db is not None and self._async_conn is not None:
            await KuzuClientManager.release_client(self._db, self._async_conn)
            self._db = None
            self._async_conn = None
            logger.info("KuzuGraphStorage finalized.")

    async def _ensure_schema(self):
        """Creates the necessary node and relationship tables and indexes if they don't exist."""
        if not self._async_conn:
            raise RuntimeError("Kuzu AsyncConnection not initialized.")

        try:
            # Create node table 'base' with 'entity_id' as primary key
            # Kuzu requires primary key to be INT64, FLOAT, or STRING. Using STRING.
            # Store other properties in a flexible MAP type.
            node_table_query = f"""
            CREATE NODE TABLE IF NOT EXISTS {self._node_table}(
                {self._entity_id_prop} STRING PRIMARY KEY,
                properties MAP(STRING, ANY)
            )
            """
            await self._async_conn.execute(node_table_query)
            logger.info(f"Ensured node table '{self._node_table}' exists.")

            # Create relationship table 'DIRECTED' connecting 'base' nodes
            # Store properties in a flexible MAP type.
            edge_table_query = f"""
            CREATE REL TABLE IF NOT EXISTS {self._edge_table}(
                FROM {self._node_table},
                TO {self._node_table},
                properties MAP(STRING, ANY)
            )
            """
            await self._async_conn.execute(edge_table_query)
            logger.info(f"Ensured relationship table '{self._edge_table}' exists.")

            # Note: Kuzu automatically creates indexes on primary keys.
            # Additional indexes on properties within the MAP might require specific syntax
            # or might not be directly supported depending on Kuzu version.
            # For now, rely on the PK index for entity_id lookups.
            # Example for indexing a specific property within the map (if needed and supported):
            # CREATE INDEX idx_node_name ON base(properties['name']);

        except Exception as e:
            logger.error(f"Error ensuring Kuzu schema: {e}")
            # If schema creation fails critically, maybe re-raise or handle appropriately
            raise

    # --- Helper Methods ---

    def _get_conn(self) -> AsyncConnection:
        """Ensures the async connection is available."""
        if not self._async_conn:
            # This might happen if initialize wasn't called or finalize was called prematurely.
            # Try to re-initialize. A better approach might involve state management.
            logger.warning(
                "Kuzu AsyncConnection accessed before initialization or after finalization. Attempting re-initialization."
            )
            # Note: Direct await in sync method is not possible.
            # This indicates a potential design issue if called outside an async context
            # after finalization. For simplicity here, we raise an error.
            raise RuntimeError(
                "Kuzu AsyncConnection not available. Ensure initialize() is called."
            )
        return self._async_conn

    def _prepare_cypher_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Prepares parameters for Kuzu Cypher execution (e.g., serializing complex types)."""
        prepared_params = {}
        for key, value in params.items():
            # Kuzu's Python driver handles basic types well.
            # We might need to serialize complex types like lists/dicts if storing them directly
            # in non-MAP properties, but here we use MAPs.
            # Ensure lists/dicts intended for MAP properties are passed as Python dicts.
            # For properties stored directly (like entity_id), ensure correct type.
            if isinstance(value, dict):
                # Convert nested dicts/lists within the properties dict to JSON strings
                # Kuzu's MAP(STRING, ANY) should handle basic Python types, but explicit is safer
                prepared_params[key] = {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list) and key.endswith(
                "_list"
            ):  # Convention for list params for UNWIND
                prepared_params[key] = value  # Pass lists directly for UNWIND
            else:
                prepared_params[key] = value
        return prepared_params

    # --- Core Graph Operations ---

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(
            (RuntimeError, TimeoutError)
        ),  # Add Kuzu specific transient errors if any
    )
    async def has_node(self, node_id: str) -> bool:
        """Checks if a node with the given entity_id exists."""
        conn = self._get_conn()
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            RETURN count(n) > 0 AS node_exists
        """
        params = {"node_id": node_id}
        try:
            result = await conn.execute(query, params)
            if await result.has_next():
                return (await result.get_next())[0]  # Get the boolean value
            return False  # Should not happen if query is correct
        except Exception as e:
            logger.error(f"Error checking node existence for '{node_id}': {e}")
            raise  # Re-raise after logging

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Checks if a 'DIRECTED' edge exists between two nodes."""
        conn = self._get_conn()
        # Check for edge in either direction since base class implies undirected check
        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            RETURN count(r) > 0 AS edge_exists
        """
        params = {"source_id": source_node_id, "target_id": target_node_id}
        try:
            result = await conn.execute(query, params)
            if await result.has_next():
                return (await result.get_next())[0]
            return False
        except Exception as e:
            logger.error(
                f"Error checking edge existence between '{source_node_id}' and '{target_node_id}': {e}"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Gets node properties by entity_id."""
        conn = self._get_conn()
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            RETURN n.properties AS properties
        """
        params = {"node_id": node_id}
        try:
            result = await conn.execute(query, params)
            if await result.has_next():
                # Kuzu returns properties directly (as dict if MAP type)
                node_props = (await result.get_next())[0]
                # Add entity_id back for consistency if needed by consumers
                if node_props is None:
                    node_props = {}  # Handle case where properties map is null/empty
                node_props[self._entity_id_prop] = node_id
                return node_props
            return None
        except Exception as e:
            logger.error(f"Error getting node for '{node_id}': {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def node_degree(self, node_id: str) -> int:
        """Gets the degree (number of relationships) of a node."""
        conn = self._get_conn()
        # Count relationships in either direction
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})-[r]-()
            RETURN count(r) AS degree
        """
        params = {"node_id": node_id}
        try:
            result = await conn.execute(query, params)
            if await result.has_next():
                degree_result = await result.get_next()
                # Check if the node was found, if not degree is 0
                return degree_result[0] if degree_result else 0
            return 0  # Node not found
        except Exception as e:
            logger.error(f"Error getting node degree for '{node_id}': {e}")
            # Check if the error is due to node not found vs other issues
            if "Node not found" in str(e):  # Adjust based on actual Kuzu error message
                return 0
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Calculates the combined degree of the source and target nodes."""
        # Reuse node_degree for efficiency
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        """Gets properties of the first edge found between two nodes."""
        conn = self._get_conn()
        # Find edge in either direction, return properties of the first one found
        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            RETURN r.properties AS properties
            LIMIT 1
        """
        params = {"source_id": source_node_id, "target_id": target_node_id}
        try:
            result = await conn.execute(query, params)
            if await result.has_next():
                edge_props = (await result.get_next())[0]
                return (
                    edge_props if edge_props else {}
                )  # Return empty dict if properties are null/empty
            return None
        except Exception as e:
            logger.error(
                f"Error getting edge between '{source_node_id}' and '{target_node_id}': {e}"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Gets all edges (source_id, target_id) connected to a node."""
        conn = self._get_conn()
        # Match relationships where the source node is either start or end
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
                  -[r:{self._edge_table}]-
                  (connected:{self._node_table})
            RETURN n.{self._entity_id_prop} AS source, connected.{self._entity_id_prop} AS target
        """
        params = {"node_id": source_node_id}
        edges = []
        try:
            result = await conn.execute(query, params)
            while await result.has_next():
                record = await result.get_next()
                # Determine correct source/target based on the match direction implicitly handled by '-'
                # Kuzu might return duplicates if a->b and b->a both exist, handle this if needed.
                # For now, assume the query returns pairs representing the connection.
                src, tgt = record[0], record[1]
                if src == source_node_id:
                    edges.append((src, tgt))
                else:
                    edges.append(
                        (tgt, src)
                    )  # Ensure source_node_id is first in the tuple

            # Deduplicate edges if necessary (e.g., if Kuzu returns both directions for undirected match)
            return list(set(edges)) if edges else None

        except Exception as e:
            # Handle node not found specifically
            # This check might need adjustment based on Kuzu's specific error for missing nodes in MATCH
            if "Node not found" in str(e):  # Placeholder check
                logger.warning(f"Node '{source_node_id}' not found for get_node_edges.")
                return None
            logger.error(f"Error getting edges for node '{source_node_id}': {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(
            (RuntimeError, TimeoutError)
        ),  # Add Kuzu write-related errors if any
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        """Creates a new node or updates an existing node's properties."""
        conn = self._get_conn()
        if self._entity_id_prop not in node_data:
            # If entity_id is only passed as node_id, add it to node_data
            node_data[self._entity_id_prop] = node_id
        elif node_data[self._entity_id_prop] != node_id:
            logger.warning(
                f"Mismatch between node_id ('{node_id}') and entity_id in properties ('{node_data[self._entity_id_prop]}'). Using node_id."
            )
            node_data[self._entity_id_prop] = node_id

        # Separate entity_id for matching, rest go into properties map
        # match_prop = {self._entity_id_prop: node_id}
        properties_map = {
            k: v for k, v in node_data.items() if k != self._entity_id_prop
        }

        # Use MERGE for upsert behavior
        # ON CREATE sets initial properties, ON MATCH updates existing properties
        # Kuzu's MAP update syntax might differ, using SET n.properties = $props for simplicity (overwrites map)
        # A more granular update (map_set or similar) might be possible depending on Kuzu version.
        query = f"""
            MERGE (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            ON CREATE SET n.properties = $properties
            ON MATCH SET n.properties = $properties
        """
        # Prepare parameters: node_id for matching, properties map for setting
        params = {"node_id": node_id, "properties": properties_map}
        prepared_params = self._prepare_cypher_params(params)

        try:
            await conn.execute(query, prepared_params)
            logger.debug(f"Upserted node '{node_id}' with properties: {properties_map}")
        except Exception as e:
            logger.error(f"Error upserting node '{node_id}': {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ) -> None:
        """Creates a new edge or updates an existing edge's properties."""
        conn = self._get_conn()

        # Use MERGE for upsert behavior based on nodes
        # Use ON CREATE/ON MATCH to set/update edge properties
        # Kuzu requires specifying direction in MERGE for relationships
        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
            MATCH (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            MERGE (a)-[r:{self._edge_table}]->(b)
            ON CREATE SET r.properties = $properties
            ON MATCH SET r.properties = $properties
        """
        params = {
            "source_id": source_node_id,
            "target_id": target_node_id,
            "properties": edge_data,
        }
        prepared_params = self._prepare_cypher_params(params)

        try:
            await conn.execute(query, prepared_params)
            logger.debug(
                f"Upserted edge '{source_node_id}'->'{target_node_id}' with properties: {edge_data}"
            )
        except Exception as e:
            logger.error(
                f"Error upserting edge '{source_node_id}'->'{target_node_id}': {e}"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def delete_node(self, node_id: str) -> None:
        """Deletes a node and its incident relationships."""
        conn = self._get_conn()
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            DETACH DELETE n
        """
        params = {"node_id": node_id}
        try:
            await conn.execute(query, params)
            logger.debug(f"Deleted node '{node_id}'.")
        except Exception as e:
            # Don't raise if node simply doesn't exist, but log warning.
            # Check Kuzu error message for "not found" indication.
            if "not found" in str(e).lower():  # Adjust based on actual Kuzu error
                logger.warning(f"Node '{node_id}' not found for deletion.")
            else:
                logger.error(f"Error deleting node '{node_id}': {e}")
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def remove_nodes(self, node_ids: list[str]) -> None:
        """Deletes multiple nodes and their incident relationships."""
        if not node_ids:
            return
        conn = self._get_conn()
        query = f"""
            UNWIND $node_id_list AS node_id
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: node_id }})
            DETACH DELETE n
        """
        # Kuzu expects list parameters directly for UNWIND
        params = {"node_id_list": node_ids}
        try:
            await conn.execute(query, params)
            logger.debug(f"Attempted deletion of nodes: {node_ids}")
        except Exception as e:
            logger.error(f"Error removing nodes batch: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Deletes multiple edges specified by (source_id, target_id) tuples."""
        if not edges:
            return
        conn = self._get_conn()
        # Prepare list of maps for UNWIND
        edge_pairs_list = [{"source": s, "target": t} for s, t in edges]

        # Match edges in either direction and delete
        query = f"""
            UNWIND $edge_pairs AS pair
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: pair.source }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: pair.target }})
            DELETE r
        """
        params = {"edge_pairs": edge_pairs_list}
        try:
            await conn.execute(query, params)
            logger.debug(f"Attempted deletion of edges: {edges}")
        except Exception as e:
            logger.error(f"Error removing edges batch: {e}")
            raise

    # --- Batch Operations ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieves multiple nodes by their entity_ids in a batch."""
        if not node_ids:
            return {}
        conn = self._get_conn()
        query = f"""
            UNWIND $node_id_list AS node_id
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: node_id }})
            RETURN n.{self._entity_id_prop} AS id, n.properties AS properties
        """
        params = {"node_id_list": node_ids}
        nodes_dict = {}
        try:
            result = await conn.execute(query, params)
            while await result.has_next():
                record = await result.get_next()
                node_id = record[0]
                node_props = record[1] if record[1] else {}
                node_props[self._entity_id_prop] = (
                    node_id  # Ensure entity_id is present
                )
                nodes_dict[node_id] = node_props
            return nodes_dict
        except Exception as e:
            logger.error(f"Error getting nodes batch: {e}")
            raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieves degrees for multiple nodes in a batch."""
        if not node_ids:
            return {}
        conn = self._get_conn()
        query = f"""
            UNWIND $node_id_list AS node_id
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: node_id }})
            OPTIONAL MATCH (n)-[r]-() // Match relationships optionally
            RETURN node_id, count(r) AS degree // Count relationships for each node
        """
        params = {"node_id_list": node_ids}
        degrees_dict = {node_id: 0 for node_id in node_ids}  # Initialize with 0
        try:
            result = await conn.execute(query, params)
            while await result.has_next():
                record = await result.get_next()
                node_id = record[0]
                degree = record[1]
                degrees_dict[node_id] = degree
            return degrees_dict
        except Exception as e:
            logger.error(f"Error getting node degrees batch: {e}")
            raise

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Calculates combined degrees for multiple edge pairs in a batch."""
        if not edge_pairs:
            return {}

        # Collect all unique node IDs involved
        all_node_ids = set()
        for src, tgt in edge_pairs:
            all_node_ids.add(src)
            all_node_ids.add(tgt)

        # Get degrees for all involved nodes in one batch call
        node_degrees = await self.node_degrees_batch(list(all_node_ids))

        # Calculate combined degrees for each pair
        edge_degrees_dict = {}
        for src, tgt in edge_pairs:
            src_degree = node_degrees.get(src, 0)
            tgt_degree = node_degrees.get(tgt, 0)
            edge_degrees_dict[(src, tgt)] = src_degree + tgt_degree

        return edge_degrees_dict

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Retrieves properties for multiple edges in a batch."""
        if not pairs:
            return {}
        conn = self._get_conn()
        # Prepare list of maps for UNWIND
        edge_pairs_list = [{"source": p["src"], "target": p["tgt"]} for p in pairs]

        query = f"""
            UNWIND $edge_pairs AS pair
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: pair.source }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: pair.target }})
            RETURN pair.source AS source_id, pair.target AS target_id, r.properties AS properties
            LIMIT 1 // Limit to 1 edge per pair if multiple exist (unlikely with directed merge)
        """
        params = {"edge_pairs": edge_pairs_list}
        edges_dict = {}
        try:
            result = await conn.execute(query, params)
            while await result.has_next():
                record = await result.get_next()
                source_id = record[0]
                target_id = record[1]
                edge_props = record[2] if record[2] else {}
                # Store with the original (src, tgt) tuple order from input
                # Need to find the original pair that matches source_id, target_id
                # This assumes the MATCH finds edges in the direction specified by the pair,
                # or the undirected match '-' finds one representation.
                # A safer approach might be to return both directions if undirected match is used.
                # For now, assume the key corresponds to the found edge nodes.
                # We store it based on the returned order first.
                edges_dict[(source_id, target_id)] = edge_props
                # If undirected match was used, potentially add the reverse tuple too if needed
                # edges_dict[(target_id, source_id)] = edge_props

            # Re-key the dictionary based on the original input pairs for consistency
            final_edges_dict = {}
            for pair_input in pairs:
                src, tgt = pair_input["src"], pair_input["tgt"]
                if (src, tgt) in edges_dict:
                    final_edges_dict[(src, tgt)] = edges_dict[(src, tgt)]
                elif (
                    tgt,
                    src,
                ) in edges_dict:  # Check reverse if undirected match was used
                    final_edges_dict[(src, tgt)] = edges_dict[(tgt, src)]
                # else: edge not found for this pair

            return final_edges_dict
        except Exception as e:
            logger.error(f"Error getting edges batch: {e}")
            raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Retrieves all connected edges for multiple nodes in a batch."""
        if not node_ids:
            return {}
        conn = self._get_conn()
        query = f"""
            UNWIND $node_id_list AS node_id
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: node_id }})
                  -[r:{self._edge_table}]-
                  (connected:{self._node_table})
            RETURN n.{self._entity_id_prop} AS source_node, connected.{self._entity_id_prop} AS connected_node
        """
        params = {"node_id_list": node_ids}
        # Initialize result dict with empty lists
        nodes_edges_dict = {node_id: [] for node_id in node_ids}
        processed_edges = set()  # To avoid duplicate edges from undirected match

        try:
            result = await conn.execute(query, params)
            while await result.has_next():
                record = await result.get_next()
                src_node = record[0]
                connected_node = record[1]

                # Determine which node was the one queried in the UNWIND list
                queried_node_id = src_node if src_node in node_ids else connected_node

                # Create canonical edge tuple (smaller id first) to handle undirected match
                edge_tuple = tuple(sorted((src_node, connected_node)))

                if edge_tuple not in processed_edges:
                    # Add edge tuple (queried_node, other_node) to the list for the queried node
                    if src_node == queried_node_id:
                        nodes_edges_dict[queried_node_id].append(
                            (src_node, connected_node)
                        )
                    else:
                        nodes_edges_dict[queried_node_id].append(
                            (connected_node, src_node)
                        )  # Store as (other, queried) to match convention? Or always (queried, other)? Let's stick to (queried, other)
                        nodes_edges_dict[queried_node_id][-1] = (
                            queried_node_id,
                            src_node,
                        )  # Corrected: always (queried, other)

                    processed_edges.add(edge_tuple)

            return nodes_edges_dict
        except Exception as e:
            logger.error(f"Error getting nodes edges batch: {e}")
            raise

    # --- Other Methods ---

    async def get_all_labels(self) -> list[str]:
        """Gets all distinct entity_ids from the 'base' node table."""
        conn = self._get_conn()
        query = f"""
            MATCH (n:{self._node_table})
            RETURN DISTINCT n.{self._entity_id_prop} AS label
            ORDER BY label
        """
        labels = []
        try:
            result = await conn.execute(query)
            while await result.has_next():
                labels.append((await result.get_next())[0])
            return labels
        except Exception as e:
            logger.error(f"Error getting all labels: {e}")
            raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """
        Retrieves a connected subgraph starting from node_label.

        Uses variable-length path matching in Kuzu. Handles '*' wildcard
        by fetching top nodes by degree.
        """
        conn = self._get_conn()
        kg = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()  # Use edge ID (internal Kuzu ID)

        if node_label == "*":
            # 1. Get total node count for truncation check
            count_query = f"MATCH (n:{self._node_table}) RETURN count(n) AS total"
            try:
                count_res = await conn.execute(count_query)
                total_nodes = (
                    (await count_res.get_next())[0] if await count_res.has_next() else 0
                )
                if total_nodes > max_nodes:
                    kg.is_truncated = True
                    logger.info(
                        f"Graph truncated: {total_nodes} nodes found, limiting to top {max_nodes} by degree."
                    )
            except Exception as e:
                logger.error(f"Error counting nodes for '*': {e}")
                total_nodes = 0  # Assume potentially truncated if count fails

            # 2. Get top N nodes by degree and their connecting edges
            # Note: Getting edges between *only* the top N nodes can be complex/slow.
            # A simpler approach is to get top N nodes, then get their direct neighbors (1-hop).
            # Or, fetch top N nodes and *all* edges, then filter in Python (might fetch too much data).
            # Let's fetch top N nodes and the edges *between* them.
            query = f"""
                MATCH (n:{self._node_table})
                WITH n ORDER BY size((n)-->()) DESC LIMIT {max_nodes} // Order by out-degree, adjust if needed
                WITH collect(n) as top_nodes
                UNWIND top_nodes as n1
                UNWIND top_nodes as n2 // Create pairs of top nodes
                MATCH (n1)-[r:{self._edge_table}]-(n2) // Find edges *between* top nodes
                RETURN collect(distinct n1) + collect(distinct n2) as nodes, collect(distinct r) as edges // Collect distinct nodes and edges
            """
            # Alternative: Get top N nodes, then separately get edges involving them (might be more edges than needed)
            # query = f"""
            #     MATCH (n:{self._node_table})
            #     WITH n ORDER BY size((n)-->()) DESC LIMIT {max_nodes}
            #     OPTIONAL MATCH (n)-[r]-(m) // Get edges connected to top nodes
            #     RETURN collect(distinct n) as nodes, collect(distinct r) as edges
            # """
            try:
                result = await conn.execute(query)
                if await result.has_next():
                    record = await result.get_next()
                    nodes_data = record[0] if record[0] else []
                    edges_data = record[1] if record[1] else []

                    # Process nodes
                    for node_kuzu in nodes_data:
                        # Kuzu node structure: {'_id': {'offset': 0, 'table': 0}, '_label': 'base', 'entity_id': 'id1', 'properties': {'prop': 'val'}}
                        node_id_internal = node_kuzu[
                            "_id"
                        ]  # Use internal ID for edge matching
                        node_id_str = (
                            str(node_id_internal["table"])
                            + "_"
                            + str(node_id_internal["offset"])
                        )  # Create unique string ID
                        if node_id_str not in seen_nodes:
                            props = node_kuzu.get("properties", {})
                            entity_id = node_kuzu.get(
                                self._entity_id_prop, node_id_str
                            )  # Fallback if prop missing
                            props[self._entity_id_prop] = (
                                entity_id  # Ensure prop exists
                            )

                            kg.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id_str,  # Use string internal ID
                                    labels=[entity_id],  # Use entity_id as label
                                    properties=props,
                                )
                            )
                            seen_nodes.add(node_id_str)

                    # Process edges
                    for edge_kuzu in edges_data:
                        # Kuzu edge structure: {'_src': {'offset': 0, 'table': 0}, '_dst': {'offset': 1, 'table': 0}, '_label': 'DIRECTED', '_id': {'offset': 0, 'rel_table': 0}, 'properties': {'weight': 1.0}}
                        edge_id_internal = edge_kuzu["_id"]
                        edge_id_str = (
                            str(edge_id_internal["rel_table"])
                            + "_"
                            + str(edge_id_internal["offset"])
                        )

                        if edge_id_str not in seen_edges:
                            src_id_internal = edge_kuzu["_src"]
                            dst_id_internal = edge_kuzu["_dst"]
                            src_id_str = (
                                str(src_id_internal["table"])
                                + "_"
                                + str(src_id_internal["offset"])
                            )
                            dst_id_str = (
                                str(dst_id_internal["table"])
                                + "_"
                                + str(dst_id_internal["offset"])
                            )

                            # Ensure source and target nodes are in our seen_nodes set (due to LIMIT)
                            if src_id_str in seen_nodes and dst_id_str in seen_nodes:
                                kg.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id_str,
                                        type=edge_kuzu["_label"],
                                        source=src_id_str,
                                        target=dst_id_str,
                                        properties=edge_kuzu.get("properties", {}),
                                    )
                                )
                                seen_edges.add(edge_id_str)

            except Exception as e:
                logger.error(f"Error getting knowledge graph for '*': {e}")

        else:  # Specific node_label
            # Use variable length path query
            # Kuzu's path finding might return paths, need to extract distinct nodes/edges.
            # `*1..{max_depth}` finds paths up to max_depth.
            # Add LIMIT clause to restrict results if needed, but path finding limits are complex.
            # A BFS approach might be more controllable for max_nodes. Let's try path first.
            # We need to collect distinct nodes and edges from the paths found.

            # Check if start node exists first
            if not await self.has_node(node_label):
                logger.warning(
                    f"Start node '{node_label}' not found for get_knowledge_graph."
                )
                return kg  # Return empty graph

            query = f"""
                MATCH path = (startNode:{self._node_table} {{ {self._entity_id_prop}: $start_node_id }})
                            -[*1..{max_depth}]- // Undirected path up to max_depth
                            (neighborNode:{self._node_table})
                WITH nodes(path) as path_nodes, relationships(path) as path_rels
                UNWIND path_nodes as n // Unwind nodes from all paths
                UNWIND path_rels as r // Unwind relationships from all paths
                RETURN collect(distinct n) as nodes, collect(distinct r) as edges
                // LIMIT {max_nodes} // Applying limit here might be incorrect for BFS-like result
            """
            # Note: Kuzu's path finding might be exhaustive and slow for large graphs/depths.
            # Applying a node limit during traversal isn't standard Cypher.
            # A manual BFS implementation might be necessary for strict max_nodes control.
            # Let's proceed with this query and handle potential truncation based on results.

            params = {"start_node_id": node_label}
            try:
                result = await conn.execute(query, params)
                if await result.has_next():
                    record = await result.get_next()
                    nodes_data = record[0] if record[0] else []
                    edges_data = record[1] if record[1] else []

                    node_count = 0
                    # Process nodes
                    for node_kuzu in nodes_data:
                        if node_count >= max_nodes:
                            kg.is_truncated = True
                            logger.info(
                                f"Graph truncated: Node limit {max_nodes} reached during processing."
                            )
                            break

                        node_id_internal = node_kuzu["_id"]
                        node_id_str = (
                            str(node_id_internal["table"])
                            + "_"
                            + str(node_id_internal["offset"])
                        )
                        if node_id_str not in seen_nodes:
                            props = node_kuzu.get("properties", {})
                            entity_id = node_kuzu.get(self._entity_id_prop, node_id_str)
                            props[self._entity_id_prop] = entity_id

                            kg.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id_str, labels=[entity_id], properties=props
                                )
                            )
                            seen_nodes.add(node_id_str)
                            node_count += 1

                    # Process edges, ensuring both endpoints are within the collected nodes
                    for edge_kuzu in edges_data:
                        edge_id_internal = edge_kuzu["_id"]
                        edge_id_str = (
                            str(edge_id_internal["rel_table"])
                            + "_"
                            + str(edge_id_internal["offset"])
                        )

                        if edge_id_str not in seen_edges:
                            src_id_internal = edge_kuzu["_src"]
                            dst_id_internal = edge_kuzu["_dst"]
                            src_id_str = (
                                str(src_id_internal["table"])
                                + "_"
                                + str(src_id_internal["offset"])
                            )
                            dst_id_str = (
                                str(dst_id_internal["table"])
                                + "_"
                                + str(dst_id_internal["offset"])
                            )

                            # Only add edge if both source and target nodes were included (due to potential truncation)
                            if src_id_str in seen_nodes and dst_id_str in seen_nodes:
                                kg.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id_str,
                                        type=edge_kuzu["_label"],
                                        source=src_id_str,
                                        target=dst_id_str,
                                        properties=edge_kuzu.get("properties", {}),
                                    )
                                )
                                seen_edges.add(edge_id_str)

                    # If we processed all nodes from the query and didn't hit max_nodes,
                    # but the query itself might have implicitly limited due to path finding limits,
                    # it's hard to be certain about truncation without a full graph count.
                    # We set truncation based on hitting the limit during processing.

            except Exception as e:
                logger.error(f"Error getting knowledge graph for '{node_label}': {e}")

        logger.info(
            f"Subgraph query successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)} | Truncated: {kg.is_truncated}"
        )
        return kg

    async def index_done_callback(self) -> None:
        """Handles persistence after indexing. Kuzu checkpoints automatically or manually."""
        # Kuzu handles persistence via its write-ahead log (WAL) and checkpointing.
        # If using a file-based DB, changes are generally persisted.
        # Explicit checkpointing can be forced if needed, but usually not required here.
        # For :memory: databases, data is lost on close unless explicitly saved.
        # This callback might trigger a manual checkpoint if desired.
        conn = self._get_conn()
        try:
            # Optional: Trigger manual checkpoint if needed, e.g., for :memory: backup
            await conn.execute("CHECKPOINT;")
            logger.info("Kuzu manual checkpoint triggered by index_done_callback.")
            pass  # Kuzu's default WAL mechanism usually suffices
        except Exception as e:
            logger.error(f"Error during Kuzu index_done_callback (checkpoint): {e}")
            # Decide if this should raise an error or just log

    async def drop(self) -> dict[str, str]:
        """Drops all nodes and relationships from the Kuzu graph."""
        conn = self._get_conn()
        # Kuzu doesn't have a direct equivalent to DETACH DELETE for all nodes easily.
        # Option 1: Delete relationships first, then nodes.
        # Option 2: Recreate tables (might be faster for full clear).
        # Option 3: If DB path is known and not :memory:, delete the DB files. (Risky if other processes use it)

        # Let's try Option 1: Delete all relationships, then all nodes.
        try:
            logger.warning(
                f"Dropping all data from Kuzu graph (namespace: {self.namespace})..."
            )
            # Delete all relationships
            rel_query = "MATCH ()-[r]-() DELETE r"
            await conn.execute(rel_query)
            logger.debug("Deleted all relationships.")
            # Delete all nodes
            node_query = "MATCH (n) DELETE n"
            await conn.execute(node_query)
            logger.debug("Deleted all nodes.")

            # Alternatively, drop and recreate tables (might require exclusive access)
            # await conn.execute(f"DROP REL TABLE {self._edge_table}")
            # await conn.execute(f"DROP NODE TABLE {self._node_table}")
            # await self._ensure_schema() # Recreate schema

            logger.info(
                f"Successfully dropped all data from Kuzu graph (namespace: {self.namespace})."
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping Kuzu graph data: {e}")
            return {"status": "error", "message": str(e)}
