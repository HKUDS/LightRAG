import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, final

# Ensure kuzu and numpy are installed with specific versions
import pipmaster as pm

if not pm.is_installed("kuzu"):
    pm.install("kuzu>=0.9.0")
if not pm.is_installed("numpy"):
    # Using >=1.22.5 as a generally compatible baseline.
    # Pip will resolve the latest available version satisfying this if 2.2.5+ isn't directly available/compatible.
    pm.install("numpy>=1.22.5")

try:
    import kuzu
    from kuzu import (
        AsyncConnection,
        Database,
        QueryResult,
    )  # Import QueryResult for type hints if needed
    import numpy as np  # noqa: F401  # Import numpy after ensuring installation
except ImportError as e:
    print(f"Error importing kuzu or numpy after installation attempt: {e}")
    # Handle the error appropriately, maybe raise it or exit
    raise


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Assuming these base classes and types are in parent directories or installed package
try:
    from ..base import BaseGraphStorage
    from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
    from ..utils import logger  # Assuming logger is configured elsewhere
except ImportError:
    # Provide dummy classes/functions if running standalone for testing
    import logging  # Import standard logging here

    logger = logging.getLogger("KuzuGraphStorage")
    logger.setLevel(logging.INFO)
    # Avoid adding handlers if logger might already exist from parent modules
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    class BaseGraphStorage:
        def __init__(self, namespace, global_config, embedding_func):
            self.namespace = namespace
            self.global_config = global_config
            self.embedding_func = embedding_func

        async def initialize(self):
            pass

        async def finalize(self):
            pass

        async def index_done_callback(self):
            pass

        async def drop(self):
            return {"status": "success", "message": "data dropped"}

    @dataclass
    class KnowledgeGraphNode:
        id: str
        labels: list[str]
        properties: dict[str, Any]

    @dataclass
    class KnowledgeGraphEdge:
        id: str
        type: str
        source: str
        target: str
        properties: dict[str, Any]

    @dataclass
    class KnowledgeGraph:
        nodes: list[KnowledgeGraphNode] = field(default_factory=list)
        edges: list[KnowledgeGraphEdge] = field(default_factory=list)
        is_truncated: bool = False


import configparser

# logging is imported above if needed
from dotenv import load_dotenv

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
        # Use absolute path based on this file's location if needed
        # Correctly get the directory of the *current* file
        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # __file__ might not be defined in some environments (e.g., interactive)
            current_file_dir = os.getcwd()  # Fallback to current working directory
        config_path = os.path.join(current_file_dir, "config.ini")

        if os.path.exists(config_path):
            config.read(config_path, "utf-8")
            logger.info(f"Loaded Kuzu config from: {config_path}")
        else:
            logger.warning(
                f"config.ini not found at {config_path}, using environment variables or defaults."
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
            # Add other Kuzu DB options if needed (buffer_pool_size, etc.)
        }
        logger.info(f"Kuzu Configuration resolved to: {kuzu_config}")
        return kuzu_config

    @classmethod
    async def get_client(cls) -> tuple[Database, AsyncConnection]:
        """Gets or creates the singleton Kuzu Database and AsyncConnection."""
        async with cls._lock:
            # Check if either db or async_conn is None, or if db is closed (might happen externally)
            # Kuzu DB doesn't have an explicit is_closed, rely on async_conn status or try/except
            if cls._instances["db"] is None or cls._instances["async_conn"] is None:
                logger.info("Creating new Kuzu DB and AsyncConnection instances.")
                config = cls.get_config()
                db_path = config["database_path"]
                # Use max_num_threads for both DB and AsyncConnection thread pool for simplicity
                # Can be configured separately if needed
                max_threads = config["max_num_threads"]
                max_concurrent = config["max_concurrent_queries"]

                # Ensure directory exists if not in-memory
                if db_path != ":memory:":
                    db_dir = os.path.dirname(db_path)
                    if db_dir and not os.path.exists(db_dir):
                        try:
                            os.makedirs(db_dir, exist_ok=True)
                            logger.info(f"Created Kuzu database directory: {db_dir}")
                        except OSError as e:
                            logger.error(
                                f"Failed to create Kuzu database directory {db_dir}: {e}"
                            )
                            raise  # Re-raise directory creation error

                try:
                    # Initialize Kuzu Database
                    db = kuzu.Database(
                        database_path=db_path,
                        # max_num_threads=max_threads # Set max_threads on connection instead if preferred
                    )
                    logger.info(f"Initialized Kuzu Database at: {db_path}")

                    # Initialize Kuzu AsyncConnection
                    async_conn = kuzu.AsyncConnection(
                        database=db,
                        max_concurrent_queries=max_concurrent,
                        max_threads_per_query=max_threads,  # Controls threads per query in the pool
                    )
                    logger.info(
                        f"Initialized Kuzu AsyncConnection with max_concurrent_queries={max_concurrent}, max_threads_per_query={max_threads}"
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
            # Return the potentially newly created or existing instances
            return cls._instances["db"], cls._instances["async_conn"]

    @classmethod
    async def release_client(cls, db: Database, async_conn: AsyncConnection):
        """Decrements reference count and closes connections if count reaches zero."""
        async with cls._lock:
            # Check if the provided instances match the managed singletons AND they exist
            if (
                cls._instances["db"] is not None
                and db is cls._instances["db"]
                and cls._instances["async_conn"] is not None
                and async_conn is cls._instances["async_conn"]
            ):
                cls._instances["ref_count"] -= 1
                logger.debug(
                    f"Kuzu client reference count decremented to: {cls._instances['ref_count']}"
                )

                if cls._instances["ref_count"] <= 0:
                    logger.info(
                        "Reference count reached zero. Closing Kuzu AsyncConnection and Database..."
                    )
                    try:
                        if cls._instances["async_conn"]:
                            cls._instances[
                                "async_conn"
                            ].close()  # Closes pool and threads
                        # Kuzu Database itself doesn't have an explicit close in the same way,
                        # relies on garbage collection or __del__.
                        # Nullify references to allow GC.
                        cls._instances["async_conn"] = None
                        cls._instances["db"] = None
                        cls._instances["ref_count"] = 0  # Ensure it's 0
                        logger.info(
                            "Kuzu AsyncConnection closed and Database references released."
                        )
                    except Exception as e:
                        logger.error(f"Error closing Kuzu resources: {e}")
                        # Reset instances even if close fails to prevent reuse
                        cls._instances["async_conn"] = None
                        cls._instances["db"] = None
                        cls._instances["ref_count"] = 0
            elif cls._instances["ref_count"] > 0:
                logger.debug(
                    f"Kuzu client release called, but ref count is still {cls._instances['ref_count']}. Not closing."
                )
            else:
                # This case might happen if release is called multiple times after ref count hits zero
                logger.warning(
                    "Attempted to release Kuzu client instances that don't match the managed singletons or were already released."
                )


# --- Kuzu Graph Storage Implementation ---


@final
@dataclass
class KuzuGraphStorage(BaseGraphStorage):
    """
    Graph storage implementation using KuzuDB.

    Manages nodes and relationships within a Kuzu database, providing
    asynchronous operations compliant with BaseGraphStorage. Stores properties
    as JSON strings for flexibility.
    """

    _db: Database | None = field(default=None, init=False, repr=False)
    _async_conn: AsyncConnection | None = field(default=None, init=False, repr=False)
    _node_table: str = field(default="base", init=False)
    _edge_table: str = field(default="DIRECTED", init=False)
    _entity_id_prop: str = field(
        default="entity_id", init=False
    )  # Property name for node identifier
    _properties_col: str = field(
        default="properties_json", init=False
    )  # Column name for JSON string properties

    def __post_init__(self):
        """Additional initialization after dataclass setup."""
        logger.info(f"KuzuGraphStorage initialized for namespace: {self.namespace}")

    async def initialize(self):
        """Initializes the Kuzu database connection and ensures schema exists."""
        if self._db is None or self._async_conn is None:
            logger.info("KuzuGraphStorage initializing client...")
            self._db, self._async_conn = await KuzuClientManager.get_client()
            await self._ensure_schema()
        else:
            logger.info("KuzuGraphStorage client already initialized.")

    async def finalize(self):
        """Releases the Kuzu database connection."""
        if self._db is not None and self._async_conn is not None:
            logger.info("KuzuGraphStorage finalizing client...")
            await KuzuClientManager.release_client(self._db, self._async_conn)
            self._db = None
            self._async_conn = None
            logger.info("KuzuGraphStorage finalized.")
        else:
            logger.info("KuzuGraphStorage already finalized or not initialized.")

    async def _ensure_schema(self):
        """Creates the necessary node and relationship tables using STRING for properties."""
        conn = self._get_conn()
        try:
            # Create node table 'base' with 'entity_id' PK and properties as JSON STRING
            node_table_query = f"""
            CREATE NODE TABLE IF NOT EXISTS {self._node_table}(
                {self._entity_id_prop} STRING PRIMARY KEY,
                {self._properties_col} STRING
            )
            """
            await conn.execute(node_table_query)
            logger.info(
                f"Ensured node table '{self._node_table}' exists with properties column '{self._properties_col}'."
            )

            # Create relationship table 'DIRECTED' connecting 'base' nodes, properties as JSON STRING
            edge_table_query = f"""
            CREATE REL TABLE IF NOT EXISTS {self._edge_table}(
                FROM {self._node_table}
                TO {self._node_table},
                {self._properties_col} STRING
            )
            """
            await conn.execute(edge_table_query)
            logger.info(
                f"Ensured relationship table '{self._edge_table}' exists with properties column '{self._properties_col}'."
            )

        except Exception as e:
            logger.error(f"Error ensuring Kuzu schema: {e}", exc_info=True)
            raise

    # --- Helper Methods ---

    def _get_conn(self) -> AsyncConnection:
        """Ensures the async connection is available and raises error if not."""
        if not self._async_conn:
            raise RuntimeError(
                "Kuzu AsyncConnection not available. Ensure initialize() is called before use."
            )
        return self._async_conn

    def _properties_to_json(self, props: dict[str, Any]) -> str:
        """Serializes a properties dictionary to a JSON string."""
        try:
            # Ensure entity_id is not included in the JSON string itself if it's the PK
            props_to_serialize = {
                k: v for k, v in props.items() if k != self._entity_id_prop
            }
            return json.dumps(props_to_serialize)
        except TypeError as e:
            logger.error(f"Error serializing properties to JSON: {props} - {e}")
            return "{}"  # Return empty JSON object on error

    def _json_to_properties(
        self, json_str: str | None, entity_id: str | None = None
    ) -> dict[str, Any]:
        """Deserializes a JSON string into a properties dictionary."""
        props = {}
        if json_str:
            try:
                props = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON properties: {json_str} - {e}")
                props = {"_raw_properties": json_str, "_error": "JSONDecodeError"}
        # Ensure entity_id is present in the final dict if provided
        if entity_id:
            props[self._entity_id_prop] = entity_id
        return props

    # --- Core Graph Operations ---

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
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
            result: QueryResult = await conn.execute(query, params)
            has_data = result.has_next()
            if has_data:
                return result.get_next()[0]
            else:
                logger.warning(
                    f"has_node query for '{node_id}' returned no rows, assuming node does not exist."
                )
                return False
        except Exception as e:
            logger.error(
                f"Error checking node existence for '{node_id}': {e}", exc_info=True
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Checks if a 'DIRECTED' edge exists between two nodes (in either direction)."""
        conn = self._get_conn()
        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            RETURN count(r) > 0 AS edge_exists
        """
        params = {"source_id": source_node_id, "target_id": target_node_id}
        try:
            result: QueryResult = await conn.execute(query, params)
            has_data = result.has_next()
            if has_data:
                return result.get_next()[0]
            else:
                logger.warning(
                    f"has_edge query between '{source_node_id}' and '{target_node_id}' returned no rows, assuming edge does not exist."
                )
                return False
        except Exception as e:
            logger.error(
                f"Error checking edge existence between '{source_node_id}' and '{target_node_id}': {e}",
                exc_info=True,
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Gets node properties (deserialized from JSON) by entity_id."""
        conn = self._get_conn()
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            RETURN n.{self._properties_col} AS props_json
        """
        params = {"node_id": node_id}
        try:
            result: QueryResult = await conn.execute(query, params)
            if result.has_next():
                props_json = result.get_next()[0]
                return self._json_to_properties(props_json, entity_id=node_id)
            logger.debug(f"Node '{node_id}' not found in get_node.")
            return None
        except Exception as e:
            logger.error(f"Error getting node for '{node_id}': {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def node_degree(self, node_id: str) -> int:
        """Gets the degree (number of relationships) of a node."""
        conn = self._get_conn()
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            OPTIONAL MATCH (n)-[r]-()
            RETURN count(r) AS degree
        """
        params = {"node_id": node_id}
        try:
            result: QueryResult = await conn.execute(query, params)
            if result.has_next():
                degree_result = result.get_next()
                return degree_result[0] if degree_result else 0
            logger.error(
                f"Unexpected empty result for node_degree query for '{node_id}'"
            )
            return 0
        except Exception as e:
            logger.error(
                f"Error getting node degree for '{node_id}': {e}", exc_info=True
            )
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Calculates the combined degree of the source and target nodes."""
        try:
            src_degree = await self.node_degree(src_id)
            tgt_degree = await self.node_degree(tgt_id)
            return src_degree + tgt_degree
        except Exception:
            logger.warning(
                f"Could not calculate edge_degree for '{src_id}'-'{tgt_id}' due to error in node_degree.",
                exc_info=False,
            )  # Less verbose logging
            return 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        """Gets properties (deserialized from JSON) of the first edge found between two nodes."""
        conn = self._get_conn()
        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            RETURN r.{self._properties_col} AS props_json
            LIMIT 1
        """
        params = {"source_id": source_node_id, "target_id": target_node_id}
        try:
            result: QueryResult = await conn.execute(query, params)
            if result.has_next():
                props_json = result.get_next()[0]
                return self._json_to_properties(
                    props_json
                )  # No entity_id needed for edge props
            logger.debug(
                f"Edge between '{source_node_id}' and '{target_node_id}' not found."
            )
            return None
        except Exception as e:
            logger.error(
                f"Error getting edge between '{source_node_id}' and '{target_node_id}': {e}",
                exc_info=True,
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
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
                  -[r:{self._edge_table}]-
                  (connected:{self._node_table})
            RETURN n.{self._entity_id_prop} AS node1, connected.{self._entity_id_prop} AS node2
        """
        params = {"node_id": source_node_id}
        edges = []
        processed_pairs = set()
        try:
            result: QueryResult = await conn.execute(query, params)
            while result.has_next():
                record = result.get_next()
                node1, node2 = record[0], record[1]
                pair = tuple(sorted((node1, node2)))
                if pair not in processed_pairs:
                    if node1 == source_node_id:
                        edges.append((node1, node2))
                    else:
                        edges.append((node2, node1))
                    processed_pairs.add(pair)
            return edges if edges else None
        except Exception as e:
            # Updated error check based on potential Kuzu error messages
            if (
                "Binder exception:" in str(e)
                and f"Node {self._node_table} with primary key = {source_node_id} does not exist"
                in str(e)
            ):
                logger.warning(f"Node '{source_node_id}' not found for get_node_edges.")
                return None
            logger.error(
                f"Error getting edges for node '{source_node_id}': {e}", exc_info=True
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        """Creates/updates a node, storing properties as a JSON string."""
        conn = self._get_conn()
        if self._entity_id_prop not in node_data:
            node_data[self._entity_id_prop] = node_id
        elif node_data[self._entity_id_prop] != node_id:
            logger.warning(
                f"Mismatch between node_id ('{node_id}') and '{self._entity_id_prop}' in properties ('{node_data[self._entity_id_prop]}'). Using node_id ('{node_id}')."
            )
            node_data[self._entity_id_prop] = node_id

        properties_json = self._properties_to_json(node_data)

        query = f"""
            MERGE (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            ON CREATE SET n.{self._properties_col} = $properties_json
            ON MATCH SET n.{self._properties_col} = $properties_json
        """
        params = {"node_id": node_id, "properties_json": properties_json}
        try:
            await conn.execute(query, params)
            logger.debug(
                f"Upserted node '{node_id}' with JSON properties: {properties_json}"
            )
        except Exception as e:
            logger.error(f"Error upserting node '{node_id}': {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ) -> None:
        """Creates/updates an edge, storing properties as a JSON string."""
        conn = self._get_conn()
        properties_json = self._properties_to_json(edge_data)

        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
            MATCH (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            MERGE (a)-[r:{self._edge_table}]->(b)
            ON CREATE SET r.{self._properties_col} = $properties_json
            ON MATCH SET r.{self._properties_col} = $properties_json
        """
        params = {
            "source_id": source_node_id,
            "target_id": target_node_id,
            "properties_json": properties_json,
        }
        try:
            await conn.execute(query, params)
            logger.debug(
                f"Upserted edge '{source_node_id}'->'{target_node_id}' with JSON properties: {properties_json}"
            )
        except Exception as e:
            logger.error(
                f"Error upserting edge '{source_node_id}'->'{target_node_id}': {e}",
                exc_info=True,
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
            logger.debug(f"Attempted deletion of node '{node_id}'.")
        except Exception as e:
            logger.error(f"Error deleting node '{node_id}': {e}", exc_info=True)
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
            WITH $node_id_list AS ids_to_delete
            MATCH (n:{self._node_table}) WHERE n.{self._entity_id_prop} IN ids_to_delete
            DETACH DELETE n
        """
        params = {"node_id_list": node_ids}
        try:
            await conn.execute(query, params)
            logger.debug(f"Attempted batch deletion of nodes: {node_ids}")
        except Exception as e:
            logger.error(f"Error removing nodes batch: {e}", exc_info=True)
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
        edge_pairs_list = [{"source": s, "target": t} for s, t in edges]
        # Match relationship in the specified direction (->) as created by upsert_edge
        query = f"""
            UNWIND $edge_pairs AS pair
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: pair.source }})
                  -[r:{self._edge_table}]->
                  (b:{self._node_table} {{ {self._entity_id_prop}: pair.target }})
            DELETE r
        """
        params = {"edge_pairs": edge_pairs_list}
        try:
            await conn.execute(query, params)
            logger.debug(f"Attempted batch deletion of edges: {edges}")
        except Exception as e:
            logger.error(f"Error removing edges batch: {e}", exc_info=True)
            raise

    # --- Batch Operations ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieves multiple nodes (properties deserialized from JSON) by their entity_ids."""
        if not node_ids:
            return {}
        conn = self._get_conn()
        query = f"""
            WITH $node_id_list AS ids_to_get
            MATCH (n:{self._node_table}) WHERE n.{self._entity_id_prop} IN ids_to_get
            RETURN n.{self._entity_id_prop} AS id, n.{self._properties_col} AS props_json
        """
        params = {"node_id_list": node_ids}
        nodes_dict = {}
        try:
            result: QueryResult = await conn.execute(query, params)
            while result.has_next():
                record = result.get_next()
                node_id = record[0]
                props_json = record[1]
                nodes_dict[node_id] = self._json_to_properties(
                    props_json, entity_id=node_id
                )
            for req_id in node_ids:
                if req_id not in nodes_dict:
                    logger.debug(f"Node '{req_id}' not found in get_nodes_batch.")
            return nodes_dict
        except Exception as e:
            logger.error(f"Error getting nodes batch: {e}", exc_info=True)
            raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieves degrees for multiple nodes in a batch."""
        if not node_ids:
            return {}
        conn = self._get_conn()
        query = f"""
            WITH $node_id_list AS ids_to_check
            MATCH (n:{self._node_table}) WHERE n.{self._entity_id_prop} IN ids_to_check
            OPTIONAL MATCH (n)-[r]-()
            RETURN n.{self._entity_id_prop} AS node_id, count(r) AS degree
        """
        params = {"node_id_list": node_ids}
        degrees_dict = {node_id: 0 for node_id in node_ids}
        try:
            result: QueryResult = await conn.execute(query, params)
            while result.has_next():
                record = result.get_next()
                degrees_dict[record[0]] = record[1]
            return degrees_dict
        except Exception as e:
            logger.error(f"Error getting node degrees batch: {e}", exc_info=True)
            raise

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Calculates combined degrees for multiple edge pairs in a batch."""
        if not edge_pairs:
            return {}
        all_node_ids = set(src for src, tgt in edge_pairs) | set(
            tgt for src, tgt in edge_pairs
        )
        node_degrees = await self.node_degrees_batch(list(all_node_ids))
        return {
            (src, tgt): node_degrees.get(src, 0) + node_degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Retrieves properties (deserialized from JSON) for multiple edges."""
        if not pairs:
            return {}
        conn = self._get_conn()
        edge_pairs_list = [{"source": p["src"], "target": p["tgt"]} for p in pairs]
        # Match directed edge as created by upsert_edge
        query = f"""
            UNWIND $edge_pairs AS pair
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: pair.source }})
                  -[r:{self._edge_table}]->
                  (b:{self._node_table} {{ {self._entity_id_prop}: pair.target }})
            RETURN pair.source AS req_source_id, pair.target AS req_target_id,
                   r.{self._properties_col} AS props_json
        """
        params = {"edge_pairs": edge_pairs_list}
        final_edges_dict = {(p["src"], p["tgt"]): None for p in pairs}
        try:
            result: QueryResult = await conn.execute(query, params)
            while result.has_next():
                record = result.get_next()
                req_src, req_tgt, props_json = record
                original_pair = (req_src, req_tgt)
                if final_edges_dict.get(original_pair, None) is None:
                    final_edges_dict[original_pair] = self._json_to_properties(
                        props_json
                    )
            # Filter out pairs where no edge was found
            return {k: v for k, v in final_edges_dict.items() if v is not None}
        except Exception as e:
            logger.error(f"Error getting edges batch: {e}", exc_info=True)
            raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Retrieves all connected edges for multiple nodes in a batch."""
        if not node_ids:
            return {}
        conn = self._get_conn()
        query = f"""
            WITH $node_id_list AS ids_to_check
            MATCH (n:{self._node_table}) WHERE n.{self._entity_id_prop} IN ids_to_check
            OPTIONAL MATCH (n)-[r:{self._edge_table}]-(connected:{self._node_table})
            RETURN n.{self._entity_id_prop} AS source_node, connected.{self._entity_id_prop} AS connected_node
        """
        params = {"node_id_list": node_ids}
        nodes_edges_dict = {node_id: [] for node_id in node_ids}
        processed_undirected_edges = set()
        try:
            result: QueryResult = await conn.execute(query, params)
            while result.has_next():
                record = result.get_next()
                src_node, connected_node = record[0], record[1]
                if src_node and connected_node:
                    canonical_pair = tuple(sorted((src_node, connected_node)))
                    if canonical_pair not in processed_undirected_edges:
                        if src_node in nodes_edges_dict:
                            nodes_edges_dict[src_node].append(
                                (src_node, connected_node)
                            )
                        if connected_node in nodes_edges_dict:
                            nodes_edges_dict[connected_node].append(
                                (connected_node, src_node)
                            )
                        processed_undirected_edges.add(canonical_pair)
            return nodes_edges_dict
        except Exception as e:
            logger.error(f"Error getting nodes edges batch: {e}", exc_info=True)
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
            result: QueryResult = await conn.execute(query)
            while result.has_next():
                labels.append(result.get_next()[0])
            return labels
        except Exception as e:
            logger.error(f"Error getting all labels: {e}", exc_info=True)
            raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """
        Retrieves a connected subgraph starting from node_label.
        Handles '*' wildcard. Returns nodes/edges with deserialized properties.
        Uses internal Kuzu IDs for processing but returns KnowledgeGraph with entity_ids.
        """
        conn = self._get_conn()
        kg = KnowledgeGraph()
        seen_node_entity_ids = set()  # Track processed entity IDs for final KG nodes
        seen_edge_internal_ids = set()  # Track processed internal edge IDs

        # Helper to convert Kuzu internal ID dict to string
        def kuzu_internal_id_to_str(internal_id_dict):
            if (
                isinstance(internal_id_dict, dict)
                and "table" in internal_id_dict
                and "offset" in internal_id_dict
            ):
                return f"{internal_id_dict['table']}_{internal_id_dict['offset']}"
            logger.warning(
                f"Unexpected format for Kuzu internal ID: {internal_id_dict}"
            )
            return str(internal_id_dict)

        # Helper to convert Kuzu edge internal ID dict to string
        def kuzu_rel_internal_id_to_str(internal_id_dict):
            if (
                isinstance(internal_id_dict, dict)
                and "rel_table" in internal_id_dict
                and "offset" in internal_id_dict
            ):
                return f"{internal_id_dict['rel_table']}_{internal_id_dict['offset']}"
            logger.warning(
                f"Unexpected format for Kuzu edge internal ID: {internal_id_dict}"
            )
            return str(internal_id_dict)

        if node_label == "*":
            # 1. Count for truncation check
            count_query = f"MATCH (n:{self._node_table}) RETURN count(n) AS total"
            try:
                count_res: QueryResult = await conn.execute(count_query)
                total_nodes = count_res.get_next()[0] if count_res.has_next() else 0
                if total_nodes > max_nodes:
                    kg.is_truncated = True
                    logger.info(
                        f"Graph truncated: {total_nodes} nodes found, limiting to top {max_nodes} by degree."
                    )
            except Exception as e:
                logger.error(f"Error counting nodes for '*': {e}", exc_info=True)

            # 2. Get top N nodes by degree and edges between them
            # CORRECTED query structure
            query = f"""
                MATCH (n:{self._node_table})
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS degree
                ORDER BY degree DESC LIMIT {max_nodes}
                WITH collect(n) as top_nodes_list // Collect the node objects
                // Now match edges between nodes *within* this collected list
                MATCH (n1:{self._node_table})-[rel:{self._edge_table}]-(n2:{self._node_table})
                WHERE n1 IN top_nodes_list AND n2 IN top_nodes_list AND id(n1) < id(n2) // Ensure both ends are top nodes and avoid duplicates
                // Return the nodes and the edges found between them
                RETURN top_nodes_list as nodes, collect(distinct rel) as edges
            """
            try:
                result: QueryResult = await conn.execute(query)
                if result.has_next():
                    record = result.get_next()
                    # The first element 'nodes' is the original top_nodes_list
                    nodes_data = record[0] if record[0] else []
                    # The second element 'edges' contains the relationships *between* those top nodes
                    edges_data = record[1] if record[1] else []

                    # Process nodes
                    node_internal_to_entity_id = {}  # Map internal ID to entity_id for edges
                    for node_kuzu in nodes_data:
                        node_id_internal = node_kuzu["_id"]
                        entity_id = node_kuzu.get(self._entity_id_prop)
                        if not entity_id:
                            continue

                        # Map internal ID string to entity_id
                        node_internal_to_entity_id[
                            kuzu_internal_id_to_str(node_id_internal)
                        ] = entity_id

                        if entity_id not in seen_node_entity_ids:
                            props_json = node_kuzu.get(self._properties_col)
                            props = self._json_to_properties(
                                props_json, entity_id=entity_id
                            )

                            kg.nodes.append(
                                KnowledgeGraphNode(
                                    id=entity_id, labels=[entity_id], properties=props
                                )
                            )
                            seen_node_entity_ids.add(entity_id)

                    # Process edges
                    for edge_kuzu in edges_data:
                        edge_id_internal = edge_kuzu["_id"]
                        edge_id_str = kuzu_rel_internal_id_to_str(edge_id_internal)

                        if edge_id_str not in seen_edge_internal_ids:
                            src_id_internal_str = kuzu_internal_id_to_str(
                                edge_kuzu["_src"]
                            )
                            dst_id_internal_str = kuzu_internal_id_to_str(
                                edge_kuzu["_dst"]
                            )

                            src_entity_id = node_internal_to_entity_id.get(
                                src_id_internal_str
                            )
                            dst_entity_id = node_internal_to_entity_id.get(
                                dst_id_internal_str
                            )

                            # Ensure source and target nodes (by entity_id) were included
                            if (
                                src_entity_id in seen_node_entity_ids
                                and dst_entity_id in seen_node_entity_ids
                            ):
                                props_json = edge_kuzu.get(self._properties_col)
                                props = self._json_to_properties(props_json)

                                kg.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id_str,
                                        type=edge_kuzu["_label"],
                                        source=src_entity_id,
                                        target=dst_entity_id,
                                        properties=props,
                                    )
                                )
                                seen_edge_internal_ids.add(edge_id_str)

            except Exception as e:
                logger.error(
                    f"Error getting knowledge graph for '*': {e}", exc_info=True
                )

        else:  # Specific node_label
            if not await self.has_node(node_label):
                logger.warning(
                    f"Start node '{node_label}' not found for get_knowledge_graph."
                )
                return kg

            query = f"""
                MATCH path = (startNode:{self._node_table} {{ {self._entity_id_prop}: $start_node_id }})
                            -[*1..{max_depth}]-
                            (neighborNode:{self._node_table})
                WITH nodes(path) as path_nodes, relationships(path) as path_rels
                UNWIND path_nodes as n
                UNWIND path_rels as r
                RETURN collect(distinct n) as nodes, collect(distinct r) as edges
            """
            params = {"start_node_id": node_label}
            try:
                result: QueryResult = await conn.execute(query, params)
                if result.has_next():
                    record = result.get_next()
                    nodes_data = record[0] if record[0] else []
                    edges_data = record[1] if record[1] else []

                    node_count = 0
                    node_internal_to_entity_id = {}  # Map internal ID to entity_id

                    # Process nodes, respecting max_nodes limit
                    for node_kuzu in nodes_data:
                        if node_count >= max_nodes:
                            kg.is_truncated = True
                            logger.info(
                                f"Graph truncated: Node limit {max_nodes} reached during processing."
                            )
                            break

                        node_id_internal = node_kuzu["_id"]
                        entity_id = node_kuzu.get(self._entity_id_prop)
                        if not entity_id:
                            continue

                        node_internal_to_entity_id[
                            kuzu_internal_id_to_str(node_id_internal)
                        ] = entity_id

                        if entity_id not in seen_node_entity_ids:
                            props_json = node_kuzu.get(self._properties_col)
                            props = self._json_to_properties(
                                props_json, entity_id=entity_id
                            )

                            kg.nodes.append(
                                KnowledgeGraphNode(
                                    id=entity_id, labels=[entity_id], properties=props
                                )
                            )
                            seen_node_entity_ids.add(entity_id)
                            node_count += 1

                    # Process edges, ensuring both endpoints are within the collected nodes
                    for edge_kuzu in edges_data:
                        edge_id_internal = edge_kuzu["_id"]
                        edge_id_str = kuzu_rel_internal_id_to_str(edge_id_internal)

                        if edge_id_str not in seen_edge_internal_ids:
                            src_id_internal_str = kuzu_internal_id_to_str(
                                edge_kuzu["_src"]
                            )
                            dst_id_internal_str = kuzu_internal_id_to_str(
                                edge_kuzu["_dst"]
                            )

                            src_entity_id = node_internal_to_entity_id.get(
                                src_id_internal_str
                            )
                            dst_entity_id = node_internal_to_entity_id.get(
                                dst_id_internal_str
                            )

                            if (
                                src_entity_id in seen_node_entity_ids
                                and dst_entity_id in seen_node_entity_ids
                            ):
                                props_json = edge_kuzu.get(self._properties_col)
                                props = self._json_to_properties(props_json)

                                kg.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id_str,
                                        type=edge_kuzu["_label"],
                                        source=src_entity_id,
                                        target=dst_entity_id,
                                        properties=props,
                                    )
                                )
                                seen_edge_internal_ids.add(edge_id_str)

                    if not kg.is_truncated and len(nodes_data) >= max_nodes:
                        kg.is_truncated = True
                        logger.info(
                            f"Graph potentially truncated by query limits or processing node limit {max_nodes}."
                        )

            except Exception as e:
                logger.error(
                    f"Error getting knowledge graph for '{node_label}': {e}",
                    exc_info=True,
                )

        logger.info(
            f"Subgraph query successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)} | Truncated: {kg.is_truncated}"
        )
        return kg

    async def index_done_callback(self) -> None:
        """Handles persistence after indexing. Kuzu checkpoints automatically or manually."""
        logger.debug(
            "index_done_callback called for KuzuGraphStorage. No explicit action taken (relying on Kuzu persistence)."
        )
        pass

    async def drop(self) -> dict[str, str]:
        """Drops all nodes and relationships by deleting all data."""
        conn = self._get_conn()
        try:
            logger.warning(
                f"Dropping all data from Kuzu graph (namespace: {self.namespace})..."
            )
            # Delete all directed relationships first
            # Using directed match '->' as created by upsert_edge
            rel_query = f"MATCH ()-[r:{self._edge_table}]->() DELETE r"
            await conn.execute(rel_query)
            logger.debug(f"Deleted all relationships from table '{self._edge_table}'.")
            # Delete all nodes
            node_query = f"MATCH (n:{self._node_table}) DELETE n"
            await conn.execute(node_query)
            logger.debug(f"Deleted all nodes from table '{self._node_table}'.")

            logger.info(
                f"Successfully dropped all data from Kuzu graph tables '{self._node_table}' and '{self._edge_table}'."
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping Kuzu graph data: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
