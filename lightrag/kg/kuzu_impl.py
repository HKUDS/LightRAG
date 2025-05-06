import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, final, Dict, Tuple, Optional

# Ensure kuzu and httpx are installed
import pipmaster as pm

if not pm.is_installed("kuzu"):
    pm.install("kuzu>=0.9.0")
if not pm.is_installed("httpx"):
    pm.install("httpx>=0.28.1")

try:
    import kuzu
    from kuzu import AsyncConnection, Database, QueryResult
    import httpx  # For Kuzu REST API
except ImportError as e:
    print(f"Error importing required libraries: {e}. Please ensure installation.")
    raise


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

# --- Configuration Loading ---

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

# --- Kuzu Client Management (Primarily for Embedded Mode) ---


class KuzuClientManager:
    """
    Manages singleton Kuzu Database and AsyncConnection instances for EMBEDDED mode.
    For REST mode, it primarily provides configuration.
    """

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
            current_file_dir = os.getcwd()
        config_path = os.path.join(current_file_dir, "config.ini")

        if os.path.exists(config_path):
            config.read(config_path, "utf-8")
            logger.info(f"Loaded Kuzu config from: {config_path}")
        else:
            logger.warning(
                f"config.ini not found at {config_path}, using environment variables or defaults."
            )

        kuzu_config = {
            "connection_mode": os.environ.get(
                "KUZU_CONNECTION_MODE",
                config.get(
                    "kuzu", "connection_mode", fallback="embedded"
                ),  # embedded or rest
            ).lower(),
            "database_path": os.environ.get(
                "KUZU_DATABASE_PATH",
                config.get("kuzu", "database_path", fallback=":memory:"),
            ),
            "api_url": os.environ.get(
                "KUZU_API_URL",
                config.get("kuzu", "api_url", fallback="http://localhost:8000"),
            ),
            "max_num_threads": int(
                os.environ.get(
                    "KUZU_MAX_NUM_THREADS",
                    config.get("kuzu", "max_num_threads", fallback=0),
                )
            ),
            "max_concurrent_queries": int(
                os.environ.get(
                    "KUZU_MAX_CONCURRENT_QUERIES",
                    config.get("kuzu", "max_concurrent_queries", fallback=4),
                )
            ),
        }
        if kuzu_config["connection_mode"] == "rest" and not kuzu_config["api_url"]:
            raise ValueError("KUZU_API_URL must be set for 'rest' connection mode.")

        logger.info(f"Kuzu Configuration resolved to: {kuzu_config}")
        return kuzu_config

    @classmethod
    async def get_embedded_client(cls) -> Tuple[Database, AsyncConnection]:
        """Gets or creates the singleton Kuzu Database and AsyncConnection for EMBEDDED mode."""
        async with cls._lock:
            if cls._instances["db"] is None or cls._instances["async_conn"] is None:
                logger.info(
                    "Creating new Kuzu EMBEDDED DB and AsyncConnection instances."
                )
                full_config = cls.get_config()
                db_path = full_config["database_path"]
                max_threads = full_config["max_num_threads"]
                max_concurrent = full_config["max_concurrent_queries"]

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
                            raise

                try:
                    db = kuzu.Database(database_path=db_path)
                    logger.info(f"Initialized Kuzu EMBEDDED Database at: {db_path}")
                    async_conn = kuzu.AsyncConnection(
                        database=db,
                        max_concurrent_queries=max_concurrent,
                        max_threads_per_query=max_threads,
                    )
                    logger.info("Initialized Kuzu EMBEDDED AsyncConnection.")
                    cls._instances["db"] = db
                    cls._instances["async_conn"] = async_conn
                    cls._instances["ref_count"] = 0
                except Exception as e:
                    logger.error(
                        f"Failed to initialize Kuzu EMBEDDED DB/Connection: {e}"
                    )
                    raise

            cls._instances["ref_count"] += 1
            logger.debug(
                f"Kuzu EMBEDDED client reference count incremented to: {cls._instances['ref_count']}"
            )
            return cls._instances["db"], cls._instances["async_conn"]

    @classmethod
    async def release_embedded_client(cls, db: Database, async_conn: AsyncConnection):
        """Decrements reference count for EMBEDDED client and closes if count reaches zero."""
        async with cls._lock:
            if (
                cls._instances["db"] is not None
                and db is cls._instances["db"]
                and cls._instances["async_conn"] is not None
                and async_conn is cls._instances["async_conn"]
            ):
                cls._instances["ref_count"] -= 1
                logger.debug(
                    f"Kuzu EMBEDDED client reference count decremented to: {cls._instances['ref_count']}"
                )
                if cls._instances["ref_count"] <= 0:
                    logger.info(
                        "Reference count reached zero. Closing Kuzu EMBEDDED AsyncConnection and Database..."
                    )
                    try:
                        if cls._instances["async_conn"]:
                            cls._instances["async_conn"].close()
                        cls._instances = {
                            "db": None,
                            "async_conn": None,
                            "ref_count": 0,
                        }
                        logger.info(
                            "Kuzu EMBEDDED AsyncConnection closed and Database references released."
                        )
                    except Exception as e:
                        logger.error(f"Error closing Kuzu EMBEDDED resources: {e}")
                        cls._instances = {
                            "db": None,
                            "async_conn": None,
                            "ref_count": 0,
                        }
            elif cls._instances["ref_count"] > 0:
                logger.debug(
                    f"Kuzu EMBEDDED client release called, but ref count is still {cls._instances['ref_count']}. Not closing."
                )
            else:
                logger.warning(
                    "Attempted to release Kuzu EMBEDDED client instances that don't match or were already released."
                )


# --- Kuzu Graph Storage Implementation ---


@final
@dataclass
class KuzuGraphStorage(BaseGraphStorage):
    """
    Graph storage implementation using KuzuDB, supporting both embedded and REST API modes.
    """

    _connection_mode: str = field(default="embedded", init=False)
    _api_url: Optional[str] = field(default=None, init=False)
    _http_client: Optional[httpx.AsyncClient] = field(
        default=None, init=False, repr=False
    )

    _db: Database | None = field(
        default=None, init=False, repr=False
    )  # For embedded mode
    _async_conn: AsyncConnection | None = field(
        default=None, init=False, repr=False
    )  # For embedded mode

    _node_table: str = field(default="base", init=False)
    _edge_table: str = field(default="DIRECTED", init=False)
    _entity_id_prop: str = field(default="entity_id", init=False)
    _properties_col: str = field(default="properties_json", init=False)

    def __post_init__(self):
        """Additional initialization after dataclass setup."""
        logger.info(f"KuzuGraphStorage initialized for namespace: {self.namespace}")
        self._config = KuzuClientManager.get_config()  # Store config for later use
        self._connection_mode = self._config["connection_mode"]
        if self._connection_mode == "rest":
            self._api_url = self._config["api_url"]

    async def initialize(self):
        """Initializes Kuzu connection based on mode (embedded or REST API)."""
        logger.info(
            f"KuzuGraphStorage initializing in '{self._connection_mode}' mode..."
        )
        if self._connection_mode == "embedded":
            if self._db is None or self._async_conn is None:
                (
                    self._db,
                    self._async_conn,
                ) = await KuzuClientManager.get_embedded_client()
                await self._ensure_schema()
        elif self._connection_mode == "rest":
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    base_url=self._api_url, timeout=30.0
                )
                await self._check_api_server_status()  # Check server status
                await self._ensure_schema()
        else:
            raise ValueError(
                f"Unsupported Kuzu connection mode: {self._connection_mode}"
            )

    async def finalize(self):
        """Closes the Kuzu connection or HTTP client."""
        logger.info(f"KuzuGraphStorage finalizing in '{self._connection_mode}' mode...")
        if self._connection_mode == "embedded":
            if self._db is not None and self._async_conn is not None:
                await KuzuClientManager.release_embedded_client(
                    self._db, self._async_conn
                )
                self._db = None
                self._async_conn = None
        elif self._connection_mode == "rest":
            if self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None
        logger.info("KuzuGraphStorage finalized.")

    async def _check_api_server_status(self):
        """Checks the status of the Kuzu API server."""
        if not self._http_client:
            return
        try:
            response = await self._http_client.get("/")
            response.raise_for_status()
            status_data = response.json()
            logger.info(f"Kuzu API Server status: {status_data}")
            if status_data.get("status") != "ok":
                raise RuntimeError(f"Kuzu API Server not healthy: {status_data}")
        except httpx.RequestError as e:
            logger.error(f"Error connecting to Kuzu API Server at {self._api_url}: {e}")
            raise RuntimeError(f"Could not connect to Kuzu API Server: {e}") from e
        except Exception as e:
            logger.error(f"Error checking Kuzu API server status: {e}")
            raise

    async def _ensure_schema(self):
        """Ensures schema for both EMBEDDED and REST modes."""
        node_table_query = f"""
        CREATE NODE TABLE IF NOT EXISTS {self._node_table}(
            {self._entity_id_prop} STRING PRIMARY KEY,
            {self._properties_col} STRING
        )
        """
        edge_table_query = f"""
        CREATE REL TABLE IF NOT EXISTS {self._edge_table}(
            FROM {self._node_table}
            TO {self._node_table},
            {self._properties_col} STRING
        )
        """
        try:
            if self._connection_mode == "embedded":
                conn = self._get_embedded_conn()
                await conn.execute(node_table_query)
                logger.info(f"Ensured EMBEDDED node table '{self._node_table}'.")
                await conn.execute(edge_table_query)
                logger.info(
                    f"Ensured EMBEDDED relationship table '{self._edge_table}'."
                )
            elif self._connection_mode == "rest":
                # For REST, send DDL via /cypher endpoint
                # The API server should handle "IF NOT EXISTS" gracefully
                await self._execute_cypher_rest(node_table_query)
                logger.info(
                    f"Sent CREATE NODE TABLE IF NOT EXISTS to REST API for '{self._node_table}'."
                )
                await self._execute_cypher_rest(edge_table_query)
                logger.info(
                    f"Sent CREATE REL TABLE IF NOT EXISTS to REST API for '{self._edge_table}'."
                )
            else:
                raise ValueError(
                    f"Unknown connection mode for schema creation: {self._connection_mode}"
                )
        except Exception as e:
            # The Kuzu API server might return an error if the table already exists
            # even with "IF NOT EXISTS", depending on its specific error handling.
            # Or, the error could be a connection issue.
            # If the error message indicates "already exists", we can often ignore it.
            if (
                "already exists" in str(e).lower()
                or (
                    "Catalog exception: Node table" in str(e)
                    and "already exists" in str(e)
                )
                or (
                    "Catalog exception: Rel table" in str(e)
                    and "already exists" in str(e)
                )
                or ("Binder exception: Table" in str(e) and "already exists!" in str(e))
            ):  # Added for REST API error
                logger.debug(
                    f"Schema element likely already exists (error: {e}). Continuing."
                )
            else:
                logger.error(
                    f"Error ensuring Kuzu schema in '{self._connection_mode}' mode: {e}",
                    exc_info=True,
                )
                raise

    # --- Helper Methods ---

    def _get_embedded_conn(self) -> AsyncConnection:
        if not self._async_conn:
            raise RuntimeError("Kuzu EMBEDDED AsyncConnection not available.")
        return self._async_conn

    def _get_http_client(self) -> httpx.AsyncClient:
        if not self._http_client:
            raise RuntimeError("Kuzu HTTP client not available for REST mode.")
        return self._http_client

    def _properties_to_json(self, props: dict[str, Any]) -> str:
        props_to_serialize = {
            k: v for k, v in props.items() if k != self._entity_id_prop
        }
        return json.dumps(props_to_serialize)

    def _json_to_properties(
        self, json_data: Any, entity_id: str | None = None
    ) -> dict[str, Any]:
        """Deserializes a JSON string into a properties dictionary."""
        props = {}
        if isinstance(json_data, str):  # If it's a string, try to parse
            try:
                props = json.loads(json_data)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON properties string: {json_data}")
                return {"_raw_properties": json_data, "_error": "JSONDecodeError"}
        elif isinstance(json_data, dict):  # If it's already a dict (from Kuzu API)
            props = json_data
        else:  # Unexpected type
            logger.warning(
                f"Unexpected type for JSON properties: {type(json_data)}, value: {json_data}"
            )
            return {"_raw_data": str(json_data), "_error": "UnexpectedDataType"}

        if entity_id:
            props[self._entity_id_prop] = entity_id
        return props

    async def _execute_cypher_rest(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Executes a Cypher query against the Kuzu REST API."""
        client = self._get_http_client()
        payload = {"query": query}
        if params:
            payload["params"] = params
        try:
            response = await client.post("/cypher", json=payload)
            response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
            return response.json()  # Kuzu API returns JSON
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Kuzu API error for query '{query[:100]}...': {e.response.status_code} - {e.response.text}"
            )
            # Map to a more generic error or re-raise specific Kuzu API error if identifiable
            raise RuntimeError(
                f"Kuzu API query failed: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Kuzu API request error for query '{query[:100]}...': {e}")
            raise RuntimeError(f"Kuzu API request failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode Kuzu API JSON response for query '{query[:100]}...': {e.msg} - Response text: {response.text if 'response' in locals() else 'N/A'}"
            )
            raise RuntimeError(f"Kuzu API JSON decode error: {e.msg}") from e

    # --- Core Graph Operations (Branching for Embedded vs REST) ---

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def has_node(self, node_id: str) -> bool:
        """Checks if a node with the given entity_id exists."""
        query = f"MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }}) RETURN count(n) > 0 AS node_exists"  # Added alias
        params = {"node_id": node_id}

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                return result.get_next()[0] if result.has_next() else False
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error checking node '{node_id}': {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                return (
                    response_data.get("rows", [{}])[0].get("node_exists", False)
                    if response_data.get("rows") and response_data["rows"][0]
                    else False
                )
            except Exception as e:
                logger.error(
                    f"REST Kuzu error checking node '{node_id}': {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Checks if a 'DIRECTED' edge exists between two nodes (in either direction)."""
        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            RETURN count(r) > 0 AS edge_exists
        """
        params = {"source_id": source_node_id, "target_id": target_node_id}
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                return result.get_next()[0] if result.has_next() else False
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error checking edge '{source_node_id}'-'{target_node_id}': {e}",
                    exc_info=True,
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                return (
                    response_data.get("rows", [{}])[0].get("edge_exists", False)
                    if response_data.get("rows") and response_data["rows"][0]
                    else False
                )
            except Exception as e:
                logger.error(
                    f"REST Kuzu error checking edge '{source_node_id}'-'{target_node_id}': {e}",
                    exc_info=True,
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Gets node properties (deserialized from JSON) by entity_id."""
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            RETURN n.{self._properties_col} AS props_json
        """
        params = {"node_id": node_id}

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                if result.has_next():
                    props_json = result.get_next()[0]
                    return self._json_to_properties(props_json, entity_id=node_id)
                return None
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting node '{node_id}': {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                if response_data.get("rows") and response_data["rows"][0]:
                    # For single aliased column, row is a dict: {"props_json": "value"}
                    props_json = response_data["rows"][0].get("props_json")
                    return self._json_to_properties(props_json, entity_id=node_id)
                return None
            except Exception as e:
                logger.error(
                    f"REST Kuzu error getting node '{node_id}': {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def node_degree(self, node_id: str) -> int:
        """Gets the degree (number of relationships) of a node."""
        query = f"MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }}) OPTIONAL MATCH (n)-[r]-() RETURN count(r) AS degree"
        params = {"node_id": node_id}
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                return result.get_next()[0] if result.has_next() else 0
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting degree for node '{node_id}': {e}",
                    exc_info=True,
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                return (
                    response_data.get("rows", [{}])[0].get("degree", 0)
                    if response_data.get("rows") and response_data["rows"][0]
                    else 0
                )
            except Exception as e:
                logger.error(
                    f"REST Kuzu error getting degree for node '{node_id}': {e}",
                    exc_info=True,
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Calculates the combined degree of the source and target nodes."""
        # This method relies on node_degree, which is already mode-aware
        try:
            src_degree = await self.node_degree(src_id)
            tgt_degree = await self.node_degree(tgt_id)
            return src_degree + tgt_degree
        except Exception as e:
            logger.warning(
                f"Could not calculate edge_degree for '{src_id}'-'{tgt_id}': {e}",
                exc_info=False,
            )
            return 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        """Gets properties (deserialized from JSON) of the first edge found between two nodes."""
        query = f"""
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: $source_id }})
                  -[r:{self._edge_table}]-
                  (b:{self._node_table} {{ {self._entity_id_prop}: $target_id }})
            RETURN r.{self._properties_col} AS props_json
            LIMIT 1
        """
        params = {"source_id": source_node_id, "target_id": target_node_id}
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                if result.has_next():
                    props_json = result.get_next()[0]
                    return self._json_to_properties(props_json)
                return None
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting edge '{source_node_id}'-'{target_node_id}': {e}",
                    exc_info=True,
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                if response_data.get("rows") and response_data["rows"][0]:
                    props_json = response_data["rows"][0].get("props_json")
                    return self._json_to_properties(props_json)
                return None
            except Exception as e:
                logger.error(
                    f"REST Kuzu error getting edge '{source_node_id}'-'{target_node_id}': {e}",
                    exc_info=True,
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Gets all edges (source_id, target_id) connected to a node."""
        query = f"""
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
                  -[r:{self._edge_table}]-
                  (connected:{self._node_table})
            RETURN n.{self._entity_id_prop} AS node1, connected.{self._entity_id_prop} AS node2
        """
        params = {"node_id": source_node_id}
        edges = []
        processed_pairs = set()

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                while result.has_next():
                    record = result.get_next()
                    node1, node2 = record[0], record[1]
                    pair = tuple(sorted((node1, node2)))
                    if pair not in processed_pairs:
                        edges.append(
                            (node1, node2)
                            if node1 == source_node_id
                            else (node2, node1)
                        )
                        processed_pairs.add(pair)
                return edges if edges else None
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting edges for node '{source_node_id}': {e}",
                    exc_info=True,
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                # Expected: {"rows": [{"node1": "id1", "node2": "id2"}, ...], ...}
                for row_dict in response_data.get("rows", []):
                    node1 = row_dict.get("node1")
                    node2 = row_dict.get("node2")
                    if node1 and node2:  # Ensure both keys exist
                        pair = tuple(sorted((node1, node2)))
                        if pair not in processed_pairs:
                            edges.append(
                                (node1, node2)
                                if node1 == source_node_id
                                else (node2, node1)
                            )
                            processed_pairs.add(pair)
                return edges if edges else None
            except Exception as e:
                logger.error(
                    f"REST Kuzu error getting edges for node '{source_node_id}': {e}",
                    exc_info=True,
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        """Creates/updates a node, storing properties as a JSON string."""
        if self._entity_id_prop not in node_data:
            node_data[self._entity_id_prop] = node_id
        elif node_data[self._entity_id_prop] != node_id:
            logger.warning(
                f"Mismatch: node_id='{node_id}', props['{self._entity_id_prop}']='{node_data[self._entity_id_prop]}'. Using node_id."
            )
            node_data[self._entity_id_prop] = node_id

        properties_json = self._properties_to_json(node_data)

        query = f"""
            MERGE (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }})
            ON CREATE SET n.{self._properties_col} = $properties_json
            ON MATCH SET n.{self._properties_col} = $properties_json
        """
        params = {"node_id": node_id, "properties_json": properties_json}

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                await conn.execute(query, params)
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error upserting node '{node_id}': {e}",
                    exc_info=True,
                )
                raise
        elif self._connection_mode == "rest":
            try:
                await self._execute_cypher_rest(query, params)
            except Exception as e:
                logger.error(
                    f"REST Kuzu error upserting node '{node_id}': {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        logger.debug(f"Upserted node '{node_id}'")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ) -> None:
        """Creates/updates an edge, storing properties as a JSON string."""
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
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                await conn.execute(query, params)
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error upserting edge '{source_node_id}'-'{target_node_id}': {e}",
                    exc_info=True,
                )
                raise
        elif self._connection_mode == "rest":
            try:
                await self._execute_cypher_rest(query, params)
            except Exception as e:
                logger.error(
                    f"REST Kuzu error upserting edge '{source_node_id}'-'{target_node_id}': {e}",
                    exc_info=True,
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        logger.debug(f"Upserted edge '{source_node_id}'->'{target_node_id}'")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def delete_node(self, node_id: str) -> None:
        query = f"MATCH (n:{self._node_table} {{ {self._entity_id_prop}: $node_id }}) DETACH DELETE n"
        params = {"node_id": node_id}
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                await conn.execute(query, params)
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error deleting node '{node_id}': {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                await self._execute_cypher_rest(query, params)
            except Exception as e:
                logger.error(
                    f"REST Kuzu error deleting node '{node_id}': {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        logger.debug(f"Attempted deletion of node '{node_id}'.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def remove_nodes(self, node_ids: list[str]) -> None:
        """Deletes multiple nodes and their incident relationships."""
        if not node_ids:
            return
        query = f"UNWIND $node_id_list AS id_to_delete MATCH (n:{self._node_table} {{ {self._entity_id_prop}: id_to_delete }}) DETACH DELETE n"
        params = {"node_id_list": node_ids}
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                await conn.execute(query, params)
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error removing nodes batch: {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                await self._execute_cypher_rest(query, params)
            except Exception as e:
                logger.error(
                    f"REST Kuzu error removing nodes batch: {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        logger.debug(f"Attempted batch deletion of nodes: {node_ids}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RuntimeError, TimeoutError, httpx.RequestError)),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Deletes multiple edges between specified nodes."""
        if not edges:
            return

        edge_pairs_list = [{"source": s, "target": t} for s, t in edges]

        query = f"""
            UNWIND $edge_pairs AS pair
            MATCH (a:{self._node_table} {{ {self._entity_id_prop}: pair.source }})
                  -[r:{self._edge_table}]->
                  (b:{self._node_table} {{ {self._entity_id_prop}: pair.target }})
            DELETE r
        """
        params = {"edge_pairs": edge_pairs_list}
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                await conn.execute(query, params)
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error removing edges batch: {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                await self._execute_cypher_rest(query, params)
            except Exception as e:
                logger.error(
                    f"REST Kuzu error removing edges batch: {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        logger.debug(f"Attempted batch deletion of edges: {edges}")

    # --- Batch Operations ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        query = f"UNWIND $node_id_list AS id_to_get MATCH (n:{self._node_table} {{ {self._entity_id_prop}: id_to_get }}) RETURN n.{self._entity_id_prop} AS id, n.{self._properties_col} AS props_json"
        params = {"node_id_list": node_ids}
        nodes_dict = {}

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                while result.has_next():
                    record = result.get_next()
                    nodes_dict[record[0]] = self._json_to_properties(
                        record[1], entity_id=record[0]
                    )
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting nodes batch: {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                for row_dict in response_data.get(
                    "rows", []
                ):  # REST API returns list of dicts for multiple columns
                    nodes_dict[row_dict.get("id")] = self._json_to_properties(
                        row_dict.get("props_json"), entity_id=row_dict.get("id")
                    )
            except Exception as e:
                logger.error(f"REST Kuzu error getting nodes batch: {e}", exc_info=True)
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")

        for req_id in node_ids:
            if req_id not in nodes_dict:
                logger.debug(f"Node '{req_id}' not found in get_nodes_batch.")
        return nodes_dict

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        query = f"UNWIND $node_id_list AS id_to_check MATCH (n:{self._node_table} {{ {self._entity_id_prop}: id_to_check }}) OPTIONAL MATCH (n)-[r]-() RETURN n.{self._entity_id_prop} AS node_id, count(r) AS degree"
        params = {"node_id_list": node_ids}
        degrees_dict = {node_id: 0 for node_id in node_ids}

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                while result.has_next():
                    record = result.get_next()
                    degrees_dict[record[0]] = record[1]
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting node degrees batch: {e}",
                    exc_info=True,
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                for row_dict in response_data.get(
                    "rows", []
                ):  # REST API returns list of dicts for multiple columns
                    degrees_dict[row_dict.get("node_id")] = row_dict.get("degree", 0)
            except Exception as e:
                logger.error(
                    f"REST Kuzu error getting node degrees batch: {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        return degrees_dict

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if not edge_pairs:
            return {}
        all_node_ids = set(src for src, tgt in edge_pairs) | set(
            tgt for src, tgt in edge_pairs
        )
        node_degrees = await self.node_degrees_batch(
            list(all_node_ids)
        )  # This is mode-aware
        return {
            (src, tgt): node_degrees.get(src, 0) + node_degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}
        edge_pairs_list = [{"source": p["src"], "target": p["tgt"]} for p in pairs]
        # Use an explicit query that checks both directions for each pair to ensure
        # behavior consistent with singular get_edge (undirected find)
        query = f"""
            UNWIND $edge_pairs AS pair
            MATCH (n1:{self._node_table} {{ {self._entity_id_prop}: pair.source }})
            MATCH (n2:{self._node_table} {{ {self._entity_id_prop}: pair.target }})
            OPTIONAL MATCH (n1)-[r1:{self._edge_table}]->(n2)
            OPTIONAL MATCH (n2)-[r2:{self._edge_table}]->(n1) // Check reverse direction
            WITH pair, r1, r2
            WHERE r1 IS NOT NULL OR r2 IS NOT NULL // Ensure an edge was found
            RETURN pair.source AS req_source_id, pair.target AS req_target_id,
                   COALESCE(r1.{self._properties_col}, r2.{self._properties_col}) AS props_json
        """
        params = {"edge_pairs": edge_pairs_list}
        final_edges_dict = {(p["src"], p["tgt"]): None for p in pairs}

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                while result.has_next():
                    record = result.get_next()
                    original_pair = (record[0], record[1])
                    if (
                        final_edges_dict.get(original_pair, None) is None
                    ):  # Store first found
                        final_edges_dict[original_pair] = self._json_to_properties(
                            record[2]
                        )
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting edges batch: {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                for row_dict in response_data.get(
                    "rows", []
                ):  # REST API returns list of dicts
                    original_pair = (
                        row_dict.get("req_source_id"),
                        row_dict.get("req_target_id"),
                    )
                    if final_edges_dict.get(original_pair, None) is None:
                        final_edges_dict[original_pair] = self._json_to_properties(
                            row_dict.get("props_json")
                        )
            except Exception as e:
                logger.error(f"REST Kuzu error getting edges batch: {e}", exc_info=True)
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        return {k: v for k, v in final_edges_dict.items() if v is not None}

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}
        query = f"""
            UNWIND $node_id_list AS id_to_check
            MATCH (n:{self._node_table} {{ {self._entity_id_prop}: id_to_check }})
            OPTIONAL MATCH (n)-[r:{self._edge_table}]-(connected:{self._node_table})
            RETURN n.{self._entity_id_prop} AS source_node, connected.{self._entity_id_prop} AS connected_node
        """
        params = {"node_id_list": node_ids}
        nodes_edges_dict = {node_id: [] for node_id in node_ids}
        processed_undirected_edges = set()

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query, params)
                while result.has_next():
                    record = result.get_next()
                    src_node, connected_node = record[0], record[1]
                    if src_node and connected_node:
                        pair = tuple(sorted((src_node, connected_node)))
                        if pair not in processed_undirected_edges:
                            if src_node in nodes_edges_dict:
                                nodes_edges_dict[src_node].append(
                                    (src_node, connected_node)
                                )
                            if connected_node in nodes_edges_dict:
                                nodes_edges_dict[connected_node].append(
                                    (connected_node, src_node)
                                )
                            processed_undirected_edges.add(pair)
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error in get_nodes_edges_batch: {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query, params)
                for row_dict in response_data.get(
                    "rows", []
                ):  # REST API returns list of dicts
                    src_node, connected_node = (
                        row_dict.get("source_node"),
                        row_dict.get("connected_node"),
                    )
                    if src_node and connected_node:
                        pair = tuple(sorted((src_node, connected_node)))
                        if pair not in processed_undirected_edges:
                            if src_node in nodes_edges_dict:
                                nodes_edges_dict[src_node].append(
                                    (src_node, connected_node)
                                )
                            if connected_node in nodes_edges_dict:
                                nodes_edges_dict[connected_node].append(
                                    (connected_node, src_node)
                                )
                            processed_undirected_edges.add(pair)
            except Exception as e:
                logger.error(
                    f"REST Kuzu error in get_nodes_edges_batch: {e}", exc_info=True
                )
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        return nodes_edges_dict

    # --- Other Methods ---

    async def get_all_labels(self) -> list[str]:
        query = f"MATCH (n:{self._node_table}) RETURN DISTINCT n.{self._entity_id_prop} AS label ORDER BY label"
        labels = []
        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                result: QueryResult = await conn.execute(query)
                while result.has_next():
                    labels.append(result.get_next()[0])
            except Exception as e:
                logger.error(
                    f"Embedded Kuzu error getting all labels: {e}", exc_info=True
                )
                raise
        elif self._connection_mode == "rest":
            try:
                response_data = await self._execute_cypher_rest(query)
                for row_dict in response_data.get(
                    "rows", []
                ):  # REST API returns list of dicts for single column
                    labels.append(row_dict.get("label"))
            except Exception as e:
                logger.error(f"REST Kuzu error getting all labels: {e}", exc_info=True)
                raise
        else:
            raise ValueError(f"Unknown connection mode: {self._connection_mode}")
        return labels

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        seen_node_entity_ids = set()
        seen_edge_internal_ids = set()

        # Helper to convert Kuzu internal ID dict to string
        def kuzu_internal_id_to_str(internal_id_dict):
            if (
                isinstance(internal_id_dict, dict)
                and "table" in internal_id_dict
                and "offset" in internal_id_dict
            ):
                return f"{internal_id_dict['table']}_{internal_id_dict['offset']}"
            return str(internal_id_dict)

        # Helper to convert Kuzu edge internal ID dict to string
        def kuzu_rel_internal_id_to_str(internal_id_dict):
            if (
                isinstance(internal_id_dict, dict)
                and "table" in internal_id_dict
                and "offset" in internal_id_dict
            ):
                return f"{internal_id_dict['table']}_{internal_id_dict['offset']}"
            return str(internal_id_dict)

        # This helper expects a dictionary (as returned by REST API or from embedded collect())
        def process_node_data(
            node_data_dict: Dict, entity_id_key: str, props_json_key: str
        ) -> Optional[KnowledgeGraphNode]:
            entity_id = node_data_dict.get(entity_id_key)
            if not entity_id or entity_id in seen_node_entity_ids:
                return None

            props_json_str = node_data_dict.get(props_json_key)
            props = self._json_to_properties(props_json_str, entity_id=entity_id)

            seen_node_entity_ids.add(entity_id)
            return KnowledgeGraphNode(
                id=entity_id, labels=[entity_id], properties=props
            )

        # This helper expects a dictionary
        def process_edge_data(
            edge_data_dict: Dict, props_json_key: str, node_map: Dict[str, str]
        ) -> Optional[KnowledgeGraphEdge]:
            src_internal_id_str = kuzu_internal_id_to_str(edge_data_dict["_src"])
            dst_internal_id_str = kuzu_internal_id_to_str(edge_data_dict["_dst"])
            edge_internal_id_str = kuzu_rel_internal_id_to_str(edge_data_dict["_id"])

            if edge_internal_id_str in seen_edge_internal_ids:
                return None

            src_entity_id = node_map.get(src_internal_id_str)
            dst_entity_id = node_map.get(dst_internal_id_str)

            if (
                not src_entity_id
                or not dst_entity_id
                or src_entity_id not in seen_node_entity_ids
                or dst_entity_id not in seen_node_entity_ids
            ):
                return None

            props_json_str = edge_data_dict.get(props_json_key)
            props = self._json_to_properties(props_json_str)

            seen_edge_internal_ids.add(edge_internal_id_str)
            return KnowledgeGraphEdge(
                id=edge_internal_id_str,
                type=edge_data_dict["_label"],
                source=src_entity_id,
                target=dst_entity_id,
                properties=props,
            )

        if node_label == "*":
            count_query = f"MATCH (n:{self._node_table}) RETURN count(n) AS total"
            total_nodes = 0
            if self._connection_mode == "embedded":
                conn_emb = self._get_embedded_conn()
                count_res: QueryResult = await conn_emb.execute(count_query)
                total_nodes = count_res.get_next()[0] if count_res.has_next() else 0
            elif self._connection_mode == "rest":
                count_data = await self._execute_cypher_rest(count_query)
                total_nodes = (
                    count_data.get("rows", [{}])[0].get("total", 0)
                    if count_data.get("rows") and count_data.get("rows")[0]
                    else 0
                )

            if total_nodes > max_nodes:
                kg.is_truncated = True
                logger.info(
                    f"Graph truncated: {total_nodes} nodes found, limiting to top {max_nodes} by degree."
                )

            query = f"""
                MATCH (n:{self._node_table})
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS degree
                ORDER BY degree DESC LIMIT {max_nodes}
                WITH collect(n) as top_nodes_list
                MATCH (n1:{self._node_table})-[rel:{self._edge_table}]-(n2:{self._node_table})
                WHERE n1 IN top_nodes_list AND n2 IN top_nodes_list AND id(n1) < id(n2)
                RETURN top_nodes_list as nodes, collect(distinct rel) as edges
            """
            nodes_data_list = []
            edges_data_list = []

            if self._connection_mode == "embedded":
                conn_emb = self._get_embedded_conn()
                result_emb: QueryResult = await conn_emb.execute(query)
                if result_emb.has_next():
                    record = result_emb.get_next()
                    nodes_data_list = record[0] if record[0] else []
                    edges_data_list = record[1] if record[1] else []
            elif self._connection_mode == "rest":
                response_data = await self._execute_cypher_rest(query)
                if response_data.get("rows") and response_data["rows"][0]:
                    nodes_data_list = response_data["rows"][0].get("nodes", [])
                    edges_data_list = response_data["rows"][0].get("edges", [])

            node_internal_to_entity_id_map = {}
            for (
                node_data_item_dict
            ) in nodes_data_list:  # Data is already a list of dicts
                node_kg = process_node_data(
                    node_data_item_dict, self._entity_id_prop, self._properties_col
                )
                if node_kg:
                    kg.nodes.append(node_kg)
                    if "_id" in node_data_item_dict:
                        node_internal_to_entity_id_map[
                            kuzu_internal_id_to_str(node_data_item_dict["_id"])
                        ] = node_kg.id

            for (
                edge_data_item_dict
            ) in edges_data_list:  # Data is already a list of dicts
                edge_kg = process_edge_data(
                    edge_data_item_dict,
                    self._properties_col,
                    node_internal_to_entity_id_map,
                )
                if edge_kg:
                    kg.edges.append(edge_kg)

        else:  # Specific start node_label
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
            nodes_data_list = []
            edges_data_list = []

            if self._connection_mode == "embedded":
                conn_emb = self._get_embedded_conn()
                result_emb: QueryResult = await conn_emb.execute(query, params)
                if result_emb.has_next():
                    record = result_emb.get_next()
                    nodes_data_list = record[0] if record[0] else []
                    edges_data_list = record[1] if record[1] else []
            elif self._connection_mode == "rest":
                response_data = await self._execute_cypher_rest(query, params)
                if response_data.get("rows") and response_data["rows"][0]:
                    nodes_data_list = response_data["rows"][0].get("nodes", [])
                    edges_data_list = response_data["rows"][0].get("edges", [])

            node_count = 0
            node_internal_to_entity_id_map = {}
            for node_data_item_dict in nodes_data_list:
                if node_count >= max_nodes:
                    kg.is_truncated = True
                    break

                node_kg = process_node_data(
                    node_data_item_dict, self._entity_id_prop, self._properties_col
                )
                if node_kg:
                    kg.nodes.append(node_kg)
                    node_count += 1
                    if "_id" in node_data_item_dict:
                        node_internal_to_entity_id_map[
                            kuzu_internal_id_to_str(node_data_item_dict["_id"])
                        ] = node_kg.id

            for edge_data_item_dict in edges_data_list:
                edge_kg = process_edge_data(
                    edge_data_item_dict,
                    self._properties_col,
                    node_internal_to_entity_id_map,
                )
                if edge_kg:
                    kg.edges.append(edge_kg)

            if (
                not kg.is_truncated
                and len(nodes_data_list) >= max_nodes
                and node_count >= max_nodes
            ):
                kg.is_truncated = True

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
        logger.warning(
            f"Dropping all data from Kuzu graph (namespace: {self.namespace}, mode: {self._connection_mode})..."
        )
        # Delete all directed relationships first
        rel_query = f"MATCH ()-[r:{self._edge_table}]->() DELETE r"
        node_query = f"MATCH (n:{self._node_table}) DELETE n"

        if self._connection_mode == "embedded":
            conn = self._get_embedded_conn()
            try:
                await conn.execute(rel_query)
                logger.debug(
                    f"EMBEDDED: Deleted all relationships from table '{self._edge_table}'."
                )
                await conn.execute(node_query)
                logger.debug(
                    f"EMBEDDED: Deleted all nodes from table '{self._node_table}'."
                )
            except Exception as e:
                logger.error(
                    f"Error dropping Kuzu EMBEDDED graph data: {e}", exc_info=True
                )
                return {"status": "error", "message": str(e)}
        elif self._connection_mode == "rest":
            try:
                await self._execute_cypher_rest(rel_query)  # Send as separate queries
                logger.debug(
                    f"REST: Sent delete relationships command for table '{self._edge_table}'."
                )
                await self._execute_cypher_rest(node_query)
                logger.debug(
                    f"REST: Sent delete nodes command for table '{self._node_table}'."
                )
            except Exception as e:
                logger.error(f"Error dropping Kuzu REST graph data: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
        else:
            return {
                "status": "error",
                "message": f"Unknown connection mode: {self._connection_mode}",
            }

        logger.info(
            f"Successfully dropped all data from Kuzu graph tables '{self._node_table}' and '{self._edge_table}'."
        )
        return {"status": "success", "message": "data dropped"}
