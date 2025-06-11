import os
import re
import asyncio
import asyncpg
from threading import Lock
from dataclasses import dataclass
from datetime import datetime  # Added missing import
from typing import final, Dict, List, Optional, Any, Union
import configparser
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
import numpy as np
from .. import utils
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..validation import (
    DatabaseValidator,
    log_validation_errors,
)  # Add validation import
from .utils.semantic_utils import (
    process_relationship_weight,
    calculate_semantic_weight,
    set_default_threshold_manager,
)
from .utils.threshold_manager import ThresholdManager
from .utils.relationship_registry import (
    RelationshipTypeRegistry,
)
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")

from neo4j import (  # type: ignore
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


# Set neo4j logger level to ERROR to suppress warning logs
logging.getLogger("neo4j").setLevel(logging.ERROR)


class ConnectionHealthMonitor:
    """Monitor database connection health and manage reconnection"""

    def __init__(self, max_failures: int = 5, failure_window: int = 300):
        self.max_failures = max_failures
        self.failure_window = failure_window  # seconds
        self.failure_count = 0
        self.failure_timestamps = []
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        self.is_circuit_open = False
        self.circuit_open_time = 0
        self.circuit_timeout = 60  # seconds before attempting to close circuit

    def record_failure(self):
        """Record a connection failure"""
        current_time = time.time()
        self.failure_timestamps.append(current_time)

        # Remove old failures outside the window
        self.failure_timestamps = [
            ts
            for ts in self.failure_timestamps
            if current_time - ts < self.failure_window
        ]

        self.failure_count = len(self.failure_timestamps)

        # Open circuit if too many failures
        if self.failure_count >= self.max_failures and not self.is_circuit_open:
            self.is_circuit_open = True
            self.circuit_open_time = current_time
            utils.logger.error(
                f"Circuit breaker opened after {self.failure_count} failures in {self.failure_window}s"
            )

    def record_success(self):
        """Record a successful connection"""
        if self.is_circuit_open:
            self.is_circuit_open = False
            self.circuit_open_time = 0
            utils.logger.info("Circuit breaker closed after successful operation")

        # Reset failure count on success
        self.failure_count = max(0, self.failure_count - 1)
        if self.failure_count == 0:
            self.failure_timestamps.clear()

    def should_allow_request(self) -> bool:
        """Check if requests should be allowed (circuit breaker pattern)"""
        if not self.is_circuit_open:
            return True

        # Check if circuit timeout has passed
        current_time = time.time()
        if current_time - self.circuit_open_time > self.circuit_timeout:
            utils.logger.info(
                "Circuit breaker attempting to close - allowing test request"
            )
            return True

        return False

    def need_health_check(self) -> bool:
        """Check if a health check is needed"""
        current_time = time.time()
        return current_time - self.last_health_check > self.health_check_interval

    def update_health_check_time(self):
        """Update the last health check timestamp"""
        self.last_health_check = time.time()


class EmbeddingCache:
    """Simple cache for entity embeddings to improve performance"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, entity_id):
        """Get embedding from cache if available"""
        result = self.cache.get(entity_id)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def set(self, entity_id, embedding):
        """Store embedding in cache, implementing simple LRU if full"""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove a random item if cache is full
            self.cache.pop(next(iter(self.cache)))
        self.cache[entity_id] = embedding

    def stats(self):
        """Return cache hit/miss statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate,
        }

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


@final
@dataclass
class Neo4JStorage(BaseGraphStorage):
    def __init__(
        self,
        namespace: str = "lightrag",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        ssl: bool = False,
        max_retrieval_nodes: int = 50000,
        use_dynamic_thresholds: bool = True,
        min_threshold: float = 0.2,
        **kwargs,
    ):
        """
        Initialize Neo4J storage with dynamic threshold support and enhanced connection resilience.

        Args:
            namespace: Namespace for entities in the graph
            neo4j_uri: Neo4j server URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            ssl: Whether to use SSL for Neo4j connection
            max_retrieval_nodes: Maximum number of nodes to retrieve in graph queries
            use_dynamic_thresholds: Whether to use dynamic thresholds for relationship weights
            min_threshold: Minimum threshold for relationship weights
        """
        self.namespace = namespace
        # Initialize the global_config required by parent class
        self.global_config = kwargs.get("global_config", {})

        self.relationship_types = set()
        self.lock = Lock()
        self.driver = None
        self.max_retrieval_nodes = max_retrieval_nodes

        # Get Neo4j connection details from environment if not provided
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "neo4j")
        self.ssl = ssl or os.getenv("NEO4J_SSL", "false").lower() in [
            "true",
            "1",
            "yes",
        ]

        # PostgreSQL connection parameters for retrieving embeddings
        self.pg_connection_params = None
        self.embedding_dim = int(kwargs.get("embedding_dim", 1536))

        # Initialize threshold manager
        self.threshold_manager = ThresholdManager(
            default_threshold=min_threshold,
            use_dynamic_thresholds=use_dynamic_thresholds,
        )

        # Set as default threshold manager for consistency
        set_default_threshold_manager(self.threshold_manager)

        # Initialize relationship type registry
        self.rel_registry = RelationshipTypeRegistry()

        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache()

        # Initialize connection health monitoring
        self.health_monitor = ConnectionHealthMonitor(
            max_failures=kwargs.get("max_connection_failures", 5),
            failure_window=kwargs.get("failure_window", 300),
        )

        # Connection state tracking
        self._is_connected = False
        self._last_connection_attempt = 0
        self._connection_retry_delay = 1  # Start with 1 second delay
        self._max_retry_delay = 60  # Maximum retry delay in seconds

        # Enhanced error handling settings
        self.enable_graceful_degradation = kwargs.get(
            "enable_graceful_degradation", True
        )
        self.operation_timeout = kwargs.get("operation_timeout", 30)  # seconds

    async def initialize(self):
        # Get Neo4j connection details
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
            "NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace)
        )

        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
            max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
            connection_timeout=CONNECTION_TIMEOUT,
            connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
            max_transaction_retry_time=MAX_TRANSACTION_RETRY_TIME,
        )

        # Initialize PostgreSQL connection parameters for embedding retrieval
        self.pg_connection_params = {
            "host": os.environ.get(
                "POSTGRES_HOST", config.get("postgres", "host", fallback="localhost")
            ),
            "port": int(
                os.environ.get(
                    "POSTGRES_PORT", config.get("postgres", "port", fallback="5432")
                )
            ),
            "user": os.environ.get(
                "POSTGRES_USER", config.get("postgres", "user", fallback="postgres")
            ),
            "password": os.environ.get(
                "POSTGRES_PASSWORD", config.get("postgres", "password", fallback="")
            ),
            "database": os.environ.get(
                "POSTGRES_DATABASE",
                config.get("postgres", "database", fallback="postgres"),
            ),
        }

        # Try to connect to the database and create it if it doesn't exist
        for database in (DATABASE, None):
            self._DATABASE = database
            connected = False

            try:
                async with self._driver.session(database=database) as session:
                    try:
                        result = await session.run("MATCH (n) RETURN n LIMIT 0")
                        await result.consume()  # Ensure result is consumed
                        utils.logger.info(f"Connected to {database} at {URI}")
                        connected = True
                    except neo4jExceptions.ServiceUnavailable as e:
                        utils.logger.error(
                            f"{database} at {URI} is not available".capitalize()
                        )
                        raise e
            except neo4jExceptions.AuthError as e:
                utils.logger.error(f"Authentication failed for {database} at {URI}")
                raise e
            except neo4jExceptions.ClientError as e:
                if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                    utils.logger.info(
                        f"{database} at {URI} not found. Try to create specified database.".capitalize()
                    )
                    try:
                        async with self._driver.session() as session:
                            result = await session.run(
                                f"CREATE DATABASE `{database}` IF NOT EXISTS"
                            )
                            await result.consume()  # Ensure result is consumed
                            utils.logger.info(
                                f"{database} at {URI} created".capitalize()
                            )
                            connected = True
                    except (
                        neo4jExceptions.ClientError,
                        neo4jExceptions.DatabaseError,
                    ) as e:
                        if (
                            e.code
                            == "Neo.ClientError.Statement.UnsupportedAdministrationCommand"
                        ) or (e.code == "Neo.DatabaseError.Statement.ExecutionFailed"):
                            if database is not None:
                                utils.logger.warning(
                                    "This Neo4j instance does not support creating databases. Try to use Neo4j Desktop/Enterprise version or DozerDB instead. Fallback to use the default database."
                                )
                        if database is None:
                            utils.logger.error(f"Failed to create {database} at {URI}")
                            raise e

            if connected:
                # Create index for base nodes on entity_id if it doesn't exist
                try:
                    async with self._driver.session(database=database) as session:
                        # Check if index exists first
                        check_query = """
                        CALL db.indexes() YIELD name, labelsOrTypes, properties
                        WHERE labelsOrTypes = ['base'] AND properties = ['entity_id']
                        RETURN count(*) > 0 AS exists
                        """
                        try:
                            check_result = await session.run(check_query)
                            record = await check_result.single()
                            await check_result.consume()

                            index_exists = record and record.get("exists", False)

                            if not index_exists:
                                # Create index only if it doesn't exist
                                result = await session.run(
                                    "CREATE INDEX FOR (n:base) ON (n.entity_id)"
                                )
                                await result.consume()
                                utils.logger.info(
                                    f"Created index for base nodes on entity_id in {database}"
                                )
                        except Exception:
                            # Fallback if db.indexes() is not supported in this Neo4j version
                            result = await session.run(
                                "CREATE INDEX IF NOT EXISTS FOR (n:base) ON (n.entity_id)"
                            )
                            await result.consume()
                except Exception as e:
                    utils.logger.warning(f"Failed to create index: {str(e)}")
                break

        # Relationship type system is already initialized in __init__ method
        utils.logger.info(
            f"Neo4j storage initialized with relationship registry containing {len(self.rel_registry.registry)} types"
        )

    async def finalize(self):
        """Close the Neo4j driver and release all resources"""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._is_connected = False

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure driver is closed when context manager exits"""
        await self.finalize()

    async def _check_connection_health(self) -> bool:
        """
        Check if the database connection is healthy

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        if not self._driver:
            return False

        try:
            # Update health check timestamp
            self.health_monitor.update_health_check_time()

            # Simple health check query with timeout
            async with asyncio.wait_for(
                self._driver.session(database=self._DATABASE).__aenter__(), timeout=5.0
            ) as session:
                result = await asyncio.wait_for(
                    session.run("RETURN 1 as health_check"), timeout=5.0
                )
                record = await result.single()
                await result.consume()

                if record and record.get("health_check") == 1:
                    self.health_monitor.record_success()
                    self._is_connected = True
                    return True

        except Exception as e:
            utils.logger.warning(f"Database health check failed: {str(e)}")
            self.health_monitor.record_failure()
            self._is_connected = False

        return False

    async def _ensure_connection(self) -> bool:
        """
        Ensure database connection is available, reconnect if necessary

        Returns:
            bool: True if connection is available, False otherwise
        """
        # Check circuit breaker
        if not self.health_monitor.should_allow_request():
            utils.logger.warning("Circuit breaker is open, rejecting request")
            return False

        # Check if health check is needed
        if self.health_monitor.need_health_check() or not self._is_connected:
            is_healthy = await self._check_connection_health()
            if is_healthy:
                return True

        # If connection is already healthy, return True
        if self._is_connected:
            return True

        # Attempt reconnection with exponential backoff
        current_time = time.time()
        if current_time - self._last_connection_attempt < self._connection_retry_delay:
            utils.logger.debug(f"Waiting {self._connection_retry_delay}s before retry")
            return False

        self._last_connection_attempt = current_time

        try:
            utils.logger.info("Attempting to reconnect to Neo4j database")

            # Re-initialize the driver
            if self._driver:
                await self._driver.close()

            # Get connection parameters (reuse from initialize method)
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
                )
            )
            CONNECTION_ACQUISITION_TIMEOUT = float(
                os.environ.get(
                    "NEO4J_CONNECTION_ACQUISITION_TIMEOUT",
                    config.get(
                        "neo4j", "connection_acquisition_timeout", fallback=30.0
                    ),
                )
            )
            MAX_TRANSACTION_RETRY_TIME = float(
                os.environ.get(
                    "NEO4J_MAX_TRANSACTION_RETRY_TIME",
                    config.get("neo4j", "max_transaction_retry_time", fallback=30.0),
                )
            )

            self._driver = AsyncGraphDatabase.driver(
                URI,
                auth=(USERNAME, PASSWORD),
                max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
                connection_timeout=CONNECTION_TIMEOUT,
                connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
                max_transaction_retry_time=MAX_TRANSACTION_RETRY_TIME,
            )

            # Test the connection
            is_healthy = await self._check_connection_health()
            if is_healthy:
                utils.logger.info("Successfully reconnected to Neo4j database")
                self._connection_retry_delay = 1  # Reset delay on success
                return True
            else:
                utils.logger.error("Reconnection failed - health check unsuccessful")

        except Exception as e:
            utils.logger.error(f"Failed to reconnect to Neo4j: {str(e)}")
            self.health_monitor.record_failure()

        # Exponential backoff for retry delay
        self._connection_retry_delay = min(
            self._connection_retry_delay * 2, self._max_retry_delay
        )
        return False

    async def _execute_with_retry(self, operation_func, *args, **kwargs):
        """
        Execute a database operation with automatic retry and error handling

        Args:
            operation_func: The database operation function to execute
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            The result of the operation or None if all retries failed
        """
        max_retries = 3
        retry_count = 0
        last_exception = None

        while retry_count < max_retries:
            try:
                # Ensure connection is available
                if not await self._ensure_connection():
                    if self.enable_graceful_degradation:
                        utils.logger.warning(
                            "Database unavailable, but graceful degradation is enabled"
                        )
                        return None
                    else:
                        raise Exception("Database connection unavailable")

                # Execute the operation with timeout
                result = await asyncio.wait_for(
                    operation_func(*args, **kwargs), timeout=self.operation_timeout
                )

                # Record success and return result
                self.health_monitor.record_success()
                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                utils.logger.warning(
                    f"Database operation timed out (attempt {retry_count + 1}/{max_retries})"
                )
                self.health_monitor.record_failure()

            except (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.SessionExpired,
                neo4jExceptions.TransientError,
            ) as e:
                last_exception = e
                utils.logger.warning(
                    f"Transient database error (attempt {retry_count + 1}/{max_retries}): {str(e)}"
                )
                self.health_monitor.record_failure()
                self._is_connected = False

            except Exception as e:
                last_exception = e
                utils.logger.error(
                    f"Database operation failed (attempt {retry_count + 1}/{max_retries}): {str(e)}"
                )
                self.health_monitor.record_failure()

                # Don't retry for certain types of errors
                if isinstance(
                    e, (neo4jExceptions.AuthError, neo4jExceptions.ClientError)
                ):
                    break

            retry_count += 1
            if retry_count < max_retries:
                # Wait before retry with exponential backoff
                wait_time = min(2**retry_count, 10)
                utils.logger.debug(f"Waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)

        # All retries failed
        if self.enable_graceful_degradation:
            utils.logger.warning(
                f"Database operation failed after {max_retries} retries, graceful degradation enabled"
            )
            return None
        else:
            utils.logger.error(f"Database operation failed after {max_retries} retries")
            raise last_exception or Exception("Database operation failed")

    async def index_done_callback(self) -> None:
        # Noe4J handles persistence automatically
        pass

    async def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """
        Retrieve entity embedding from PostgreSQL vector storage
        Uses caching to avoid repeated database queries

        Args:
            entity_id: The entity ID to retrieve embedding for

        Returns:
            np.ndarray or None: The embedding vector if found, None otherwise
        """
        # Check cache first
        cached_embedding = self.embedding_cache.get(entity_id)
        if cached_embedding is not None:
            return cached_embedding

        # If not in cache, try to retrieve from PostgreSQL
        try:
            import asyncpg

            # Connect to PostgreSQL
            conn = await asyncpg.connect(**self.pg_connection_params)

            # Query for entity embedding - Using the correct table
            table_name = "lightrag_vdb_entity"
            query = f"""
            SELECT content_vector FROM {table_name}
            WHERE id = $1 AND workspace = $2
            LIMIT 1
            """

            row = await conn.fetchrow(query, entity_id, self.namespace)
            await conn.close()

            if row:
                # Convert the embedding to numpy array
                embedding = np.array(row["content_vector"])

                # Cache the result
                self.embedding_cache.set(entity_id, embedding)

                return embedding

            utils.logger.debug(
                f"No embedding found for entity {entity_id} in workspace {self.namespace}"
            )
            return None

        except Exception as e:
            utils.logger.warning(
                f"Error retrieving embedding for {entity_id}: {str(e)}"
            )
            return None

    async def batch_get_entity_embeddings(
        self, entity_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve embeddings for multiple entities in a single query

        Args:
            entity_ids: List of entity IDs to retrieve embeddings for

        Returns:
            dict: Dictionary mapping entity IDs to embedding vectors
        """
        # Check which entities we need to fetch (not in cache)
        to_fetch = []
        results = {}

        for entity_id in entity_ids:
            cached = self.embedding_cache.get(entity_id)
            if cached is not None:
                results[entity_id] = cached
            else:
                to_fetch.append(entity_id)

        # If everything was in cache, return early
        if not to_fetch:
            return results

        # Fetch remaining embeddings from PostgreSQL
        try:
            import asyncpg

            # Connect to PostgreSQL
            conn = await asyncpg.connect(**self.pg_connection_params)

            # Query for multiple embeddings at once - Using the correct table
            table_name = "lightrag_vdb_entity"
            query = f"""
            SELECT id, content_vector FROM {table_name}
            WHERE id = ANY($1) AND workspace = $2
            """

            rows = await conn.fetch(query, to_fetch, self.namespace)
            await conn.close()

            # Process results
            for row in rows:
                entity_id = row["id"]
                embedding = np.array(row["content_vector"])

                # Add to results and cache
                results[entity_id] = embedding
                self.embedding_cache.set(entity_id, embedding)

                # Remove from to_fetch list
                if entity_id in to_fetch:
                    to_fetch.remove(entity_id)

            # Log any entities we couldn't find
            if to_fetch:
                utils.logger.debug(
                    f"Could not find embeddings for {len(to_fetch)} entities in workspace {self.namespace}"
                )

            return results

        except Exception as e:
            utils.logger.warning(f"Error batch retrieving embeddings: {str(e)}")
            return results  # Return whatever we got from cache

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given label exists in the database with flexible matching.

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
                # First try exact match by entity_id (most efficient)
                query = "MATCH (n:base {entity_id: $entity_id}) RETURN count(n) > 0 AS node_exists"
                utils.logger.debug(
                    f"Executing Cypher query in has_node (exact): {query} with params: {{'entity_id': '{node_id}'}}"
                )
                result = await session.run(query, entity_id=node_id)
                single_result = await result.single()
                await result.consume()
                if single_result and single_result["node_exists"]:
                    return True

                # If exact match fails, try flexible matching
                flexible_query = """
                MATCH (n:base) 
                WHERE toLower(n.entity_id) CONTAINS toLower($term)
                   OR toLower(n.description) CONTAINS toLower($term)
                RETURN count(n) > 0 AS node_exists
                """
                utils.logger.debug(
                    f"Executing Cypher query in has_node (flexible): {flexible_query} with params: {{'term': '{node_id}'}}"
                )
                result = await session.run(flexible_query, term=node_id)
                single_result = await result.single()
                await result.consume()
                return single_result and single_result["node_exists"]

            except Exception as e:
                utils.logger.error(
                    f"Error checking node existence for {node_id}: {str(e)}"
                )
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
                utils.logger.debug(
                    f"Executing Cypher query in has_edge: {query} with params: {{'source_entity_id': '{source_node_id}', 'target_entity_id': '{target_node_id}'}}"
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
                utils.logger.error(
                    f"Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                await result.consume()  # Ensure results are consumed even on error
                raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier with flexible matching.

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
                # First try exact match by entity_id
                query = "MATCH (n:base {entity_id: $entity_id}) RETURN n"
                utils.logger.debug(
                    f"Executing Cypher query in get_node (exact): {query} with params: {{'entity_id': '{node_id}'}}"
                )
                result = await session.run(query, entity_id=node_id)
                records = await result.fetch(2)
                await result.consume()

                if records:
                    node = records[0]["n"]
                    node_dict = dict(node)
                    if "labels" in node_dict:
                        node_dict["labels"] = [
                            label for label in node_dict["labels"] if label != "base"
                        ]
                    else:
                        # Ensure labels field always exists, using entity_type as fallback if available
                        node_dict["labels"] = [node_dict.get("entity_type", "unknown")]
                    return node_dict

                # If exact match fails, try flexible matching
                flexible_query = """
                MATCH (n:base) 
                WHERE toLower(n.entity_id) CONTAINS toLower($term)
                   OR toLower(n.description) CONTAINS toLower($term)
                RETURN n LIMIT 1
                """
                utils.logger.debug(
                    f"Executing Cypher query in get_node (flexible): {flexible_query} with params: {{'term': '{node_id}'}}"
                )
                result = await session.run(flexible_query, term=node_id)
                records = await result.fetch(2)
                await result.consume()

                if records:
                    node = records[0]["n"]
                    node_dict = dict(node)
                    if "labels" in node_dict:
                        node_dict["labels"] = [
                            label for label in node_dict["labels"] if label != "base"
                        ]
                    else:
                        # Ensure labels field always exists, using entity_type as fallback if available
                        node_dict["labels"] = [node_dict.get("entity_type", "unknown")]
                    utils.logger.info(
                        f"Found node via flexible match for term: {node_id}"
                    )
                    return node_dict

                return None
            except Exception as e:
                utils.logger.error(f"Error getting node for {node_id}: {str(e)}")
                raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            UNWIND $node_ids AS id
            MATCH (n:base {entity_id: id})
            RETURN n.entity_id AS entity_id, n
            """
            utils.logger.debug(
                f"Executing Cypher query in get_nodes_batch: {query} with params: {{'node_ids': {node_ids}}}"
            )
            result = await session.run(query, node_ids=node_ids)
            nodes = {}
            async for record in result:
                entity_id = record["entity_id"]
                node = record["n"]
                node_dict = dict(node)
                # Remove the 'base' label if present in a 'labels' property
                if "labels" in node_dict:
                    node_dict["labels"] = [
                        label for label in node_dict["labels"] if label != "base"
                    ]
                else:
                    # Ensure labels field always exists, using entity_type as fallback if available
                    node_dict["labels"] = [node_dict.get("entity_type", "unknown")]
                nodes[entity_id] = node_dict
            await result.consume()  # Make sure to consume the result fully
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
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = """
                    MATCH (n:base {entity_id: $entity_id})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN COUNT(r) AS degree
                """
                utils.logger.debug(
                    f"Executing Cypher query in node_degree: {query} with params: {{'entity_id': '{node_id}'}}"
                )
                result = await session.run(query, entity_id=node_id)
                try:
                    record = await result.single()

                    if not record:
                        utils.logger.warning(f"No node found with label '{node_id}'")
                        return 0

                    degree = record["degree"]
                    utils.logger.debug(
                        f"Neo4j query node degree for {node_id} return: {degree}"
                    )
                    return degree
                finally:
                    await result.consume()  # Ensure result is fully consumed
            except Exception as e:
                utils.logger.error(f"Error getting node degree for {node_id}: {str(e)}")
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
        if not node_ids:
            return {}

        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                # Use this format to count relationships - same pattern as node_degree()
                query = """
                    UNWIND $node_ids AS id
                    MATCH (n:base {entity_id: id})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN id AS entity_id, COUNT(r) AS degree
                """
                utils.logger.debug(
                    f"Executing Cypher query in node_degrees_batch: {query} with params: {{'node_ids': {node_ids}}}"
                )
                result = await session.run(query, node_ids=node_ids)

                degrees = {}
                async for record in result:
                    entity_id = record["entity_id"]
                    degrees[entity_id] = record["degree"]

                await result.consume()  # Ensure result is fully consumed

                # For any node_id that did not return a record, set degree to 0.
                for nid in node_ids:
                    if nid not in degrees:
                        utils.logger.warning(f"No node found with label '{nid}'")
                        degrees[nid] = 0

                utils.logger.debug(f"Neo4j batch node degree query returned: {degrees}")
                return degrees

            except Exception as e:
                utils.logger.error(f"Error in node_degrees_batch: {str(e)}")
                # Fallback to calling node_degree individually for each node
                utils.logger.info("Falling back to individual node_degree calls")
                degrees = {}
                for nid in node_ids:
                    try:
                        degrees[nid] = await self.node_degree(nid)
                    except Exception as inner_e:
                        utils.logger.error(
                            f"Failed to get degree for node '{nid}': {str(inner_e)}"
                        )
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

    # Legacy get_edge function - replaced by updated version below  # noqa: F811
    # async def get_edge(
    #     self, source_id: str, target_id: str, rel_type: str = "related"
    # ) -> Dict[str, Any]:
    #     """
    #     Get properties of an edge between two nodes.
    #
    #     Args:
    #         source_id: Source entity ID
    #         target_id: Target entity ID
    #         rel_type: Relationship type
    #
    #     Returns:
    #         Dictionary with edge properties
    #     """
    #     query = """
    #     MATCH (src:base {entity_id: $source_id})-[r:related {rel_type: $rel_type}]->(tgt:base {entity_id: $target_id})
    #     RETURN properties(r) as properties
    #     """
    #
    #     try:
    #         async with self._driver.session(database=self._DATABASE) as session:
    #             utils.logger.debug(
    #                 f"Executing Cypher query in get_edge: {query} with params: {{'source_id': '{source_id}', 'target_id': '{target_id}', 'rel_type': '{rel_type}'}}"
    #             )
    #             result = await session.run(
    #                 query, source_id=source_id, target_id=target_id, rel_type=rel_type
    #             )
    #             record = await result.single()
    #             await result.consume()
    #
    #             # If no edge is found, return default properties using threshold manager
    #             if not record or not record.get("properties"):
    #                 threshold = self.threshold_manager.get_threshold(rel_type)
    #                 return {
    #                     "weight": threshold,
    #                     "source_id": None,
    #                     "description": None,
    #                     "keywords": None,
    #                 }
    #
    #             # Extract properties
    #             props = record["properties"]
    #
    #             # Ensure weight is a float
    #             if "weight" in props:
    #                 props["weight"] = float(props["weight"])
    #             else:
    #                 # Use dynamic threshold if weight is missing
    #                 props["weight"] = self.threshold_manager.get_threshold(rel_type)
    #
    #             return props
    #
    #     except Exception as e:
    #         utils.logger.error(f"Error getting edge: {str(e)}")
    #         threshold = self.threshold_manager.get_threshold(rel_type)
    #         return {
    #             "weight": threshold,
    #             "source_id": None,
    #             "description": None,
    #             "keywords": None,
    #         }

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
        if not pairs:
            return {}

        # Convert pairs to a list of parameter dictionaries for the Cypher query
        params = []
        for i, pair in enumerate(pairs):
            if "src" not in pair or "tgt" not in pair:
                utils.logger.warning(
                    f"Invalid edge pair: {pair}, missing 'src' or 'tgt' key"
                )
                continue

            src_id = pair["src"]
            tgt_id = pair["tgt"]
            rel_type = pair.get("rel_type")  # Optional relationship type filter

            param_dict = {
                "index": i,
                "src": src_id,
                "tgt": tgt_id,
            }

            if rel_type:
                param_dict["rel_type"] = rel_type

            params.append(param_dict)

        if not params:
            return {}

        # Build the UNWIND query that works with native relationship types
        query = """
        UNWIND $pairs AS pair
        MATCH (a:base {entity_id: pair.src})-[r]-(b:base {entity_id: pair.tgt})
        RETURN pair.src AS source, pair.tgt AS target, properties(r) AS properties, 
               type(r) AS neo4j_type, r.original_type AS original_type, r.rel_type AS rel_type
        """

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                utils.logger.debug(
                    f"Executing Cypher query in get_edges_batch: {query} with params: {{'pairs': {params}}}"
                )
                result = await session.run(query, pairs=params)

                # Process the results
                edges_dict = {}
                async for record in result:
                    if all(k in record for k in ["source", "target", "properties"]):
                        source = record["source"]
                        target = record["target"]
                        props = dict(record["properties"])

                        # Handle relationship type properties
                        neo4j_type = record["neo4j_type"]
                        original_type = record["original_type"]
                        rel_type = record["rel_type"]

                        # Prioritize types in this order: original_type, rel_type, neo4j_type
                        if original_type:
                            props["original_type"] = original_type
                        elif rel_type:
                            props["original_type"] = rel_type
                        else:
                            props["original_type"] = neo4j_type

                        # Ensure backward compatibility with rel_type
                        if not rel_type:
                            props["rel_type"] = props.get("original_type", neo4j_type)

                        # Store Neo4j type for reference
                        props["neo4j_type"] = neo4j_type

                        # Ensure weight is a float
                        if "weight" in props:
                            props["weight"] = float(props["weight"])
                        else:
                            # Use dynamic threshold if weight is missing
                            rel_type_for_threshold = props.get("rel_type", "related")
                            props["weight"] = self.threshold_manager.get_threshold(
                                rel_type_for_threshold
                            )

                        # Add standard timestamp if missing
                        if "extraction_timestamp" not in props:
                            props["extraction_timestamp"] = props.get(
                                "timestamp", datetime.now().isoformat()
                            )

                        # Ensure description exists
                        if "description" not in props or not props["description"]:
                            props["description"] = (
                                f"Relationship of type {props.get('original_type', neo4j_type)} between {source} and {target}"
                            )

                        # Default confidence if not present
                        if "confidence" not in props:
                            props["confidence"] = props.get(
                                "weight", 0.5
                            )  # Default to weight

                        # Convert keywords to proper format if needed
                        if "keywords" in props and isinstance(props["keywords"], str):
                            props["keywords"] = [
                                k.strip() for k in props["keywords"].split(";")
                            ]
                        elif "keywords" not in props:
                            props["keywords"] = [
                                source,
                                target,
                                props.get("original_type", "related"),
                            ]

                        # Set extraction source if missing
                        if "extraction_source" not in props:
                            props["extraction_source"] = "system"

                        edges_dict[(source, target)] = props

                await result.consume()

                # For any pairs not found, add default properties from threshold manager
                missing_pairs = []
                for pair in pairs:
                    if (pair["src"], pair["tgt"]) not in edges_dict:
                        missing_pairs.append((pair["src"], pair["tgt"]))

                # Get current timestamp for consistency
                current_time = datetime.now().isoformat()

                for src, tgt in missing_pairs:
                    # Get relationship type from pair if specified
                    rel_type = next(
                        (
                            p.get("rel_type", "related")
                            for p in params
                            if p["src"] == src and p["tgt"] == tgt
                        ),
                        "related",
                    )

                    # Get standardized Neo4j type
                    neo4j_type = self.rel_registry.get_neo4j_type(rel_type)

                    # Get threshold based on relationship type
                    threshold = self.threshold_manager.get_threshold(rel_type)

                    # Create comprehensive default properties following schema
                    default_props = {
                        "original_type": rel_type,
                        "rel_type": rel_type,  # For backward compatibility
                        "neo4j_type": neo4j_type,
                        "weight": threshold,
                        "confidence": threshold,
                        "description": f"Relationship of type {rel_type} between {src} and {tgt}",
                        "keywords": [src, tgt, rel_type],
                        "extraction_source": "system",
                        "extraction_timestamp": current_time,
                        "source_id": None,
                    }
                    edges_dict[(src, tgt)] = default_props

                return edges_dict

        except Exception as e:
            utils.logger.error(f"Error retrieving edge properties batch: {e}")
            # Return comprehensive default properties for all edges if there's an error
            current_time = datetime.now().isoformat()
            return {
                (pair["src"], pair["tgt"]): {
                    "original_type": pair.get("rel_type", "related"),
                    "rel_type": pair.get("rel_type", "related"),
                    "neo4j_type": self.rel_registry.get_neo4j_type(
                        pair.get("rel_type", "related")
                    ),
                    "weight": self.threshold_manager.get_threshold(
                        pair.get("rel_type", "related")
                    ),
                    "confidence": self.threshold_manager.get_threshold(
                        pair.get("rel_type", "related")
                    ),
                    "description": f"Relationship between {pair['src']} and {pair['tgt']}",
                    "keywords": [
                        pair["src"],
                        pair["tgt"],
                        pair.get("rel_type", "related"),
                    ],
                    "extraction_source": "system",
                    "extraction_timestamp": current_time,
                    "source_id": None,
                }
                for pair in pairs
            }

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
                    utils.logger.debug(
                        f"Executing Cypher query in get_node_edges: {query} with params: {{'entity_id': '{source_node_id}'}}"
                    )
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
                    utils.logger.error(
                        f"Error getting edges for node {source_node_id}: {str(e)}"
                    )
                    await results.consume()  # Ensure results are consumed even on error
                    raise
        except Exception as e:
            utils.logger.error(
                f"Error in get_node_edges for {source_node_id}: {str(e)}"
            )
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
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            # Query to get both outgoing and incoming edges
            query = """
                UNWIND $node_ids AS id
                MATCH (n:base {entity_id: id})
                OPTIONAL MATCH (n)-[r]-(connected:base)
                RETURN id AS queried_id, n.entity_id AS node_entity_id,
                       connected.entity_id AS connected_entity_id,
                       startNode(r).entity_id AS start_entity_id
            """
            utils.logger.debug(
                f"Executing Cypher query in get_nodes_edges_batch: {query} with params: {{'node_ids': {node_ids}}}"
            )
            result = await session.run(query, node_ids=node_ids)

            # Initialize the dictionary with empty lists for each node ID
            edges_dict = {node_id: [] for node_id in node_ids}

            # Process results to include both outgoing and incoming edges
            async for record in result:
                queried_id = record["queried_id"]
                node_entity_id = record["node_entity_id"]
                connected_entity_id = record["connected_entity_id"]
                start_entity_id = record["start_entity_id"]

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

            await result.consume()  # Ensure results are fully consumed
            return edges_dict

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
        # Validate node data before database operation
        validation_result = DatabaseValidator.validate_node_data(node_data)
        if validation_result.has_errors():
            error_messages = [e.message for e in validation_result.errors]
            utils.logger.error(
                f"Node validation failed for '{node_id}': {error_messages}"
            )
            log_validation_errors(validation_result.errors, f"upsert_node({node_id})")
            raise ValueError(
                f"Node data validation failed for '{node_id}': {error_messages}"
            )

        if validation_result.has_warnings():
            warning_messages = [w.message for w in validation_result.warnings]
            utils.logger.warning(
                f"Node validation warnings for '{node_id}': {warning_messages}"
            )
            log_validation_errors(validation_result.warnings, f"upsert_node({node_id})")

        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError("Neo4j: node properties must contain an 'entity_id' field")

        try:
            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    query = (
                        """
                    MERGE (n:base {entity_id: $entity_id})
                    SET n += $properties
                    SET n:`%s`
                    """
                        % entity_type
                    )
                    utils.logger.debug(
                        f"Executing Cypher query in upsert_node: {query} with params: {{'entity_id': '{node_id}', 'properties': {properties}}}"
                    )
                    result = await tx.run(
                        query, entity_id=node_id, properties=properties
                    )
                    utils.logger.debug(
                        f"Upserted node with entity_id '{node_id}' and properties: {properties}"
                    )
                    await result.consume()  # Ensure result is fully consumed

                await session.execute_write(execute_upsert)
        except Exception as e:
            utils.logger.error(f"Error during upsert: {str(e)}")
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
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ) -> None:
        """
        Base class compatible upsert_edge method.
        This method follows the BaseGraphStorage interface and delegates to the
        more complex upsert_edge_detailed method.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            edge_data: A dictionary of edge properties
        """
        # edge_data comes from _merge_edges_then_upsert and should contain:
        # "original_type", "rel_type" (human-readable std), "neo4j_type" (Neo4j label)
        # "weight", "description", "keywords" (list of strings), "source_id" (string), "file_path" (string)

        rel_type_param_for_detailed = edge_data.get(
            "rel_type", "related"
        )  # Human-readable std
        weight_param_for_detailed = float(edge_data.get("weight", 0.5))
        desc_param_for_detailed = edge_data.get("description", "")
        keywords_param_for_detailed = edge_data.get(
            "keywords", []
        )  # Should be list of str

        # Ensure keywords is a list of strings
        if isinstance(keywords_param_for_detailed, str):
            keywords_param_for_detailed = [
                kw.strip()
                for kw in keywords_param_for_detailed.split(",")
                if kw.strip()
            ]
        elif not isinstance(keywords_param_for_detailed, list):
            keywords_param_for_detailed = [str(keywords_param_for_detailed)]

        await self.upsert_edge_detailed(
            source_id=source_node_id,
            target_id=target_node_id,
            rel_type=rel_type_param_for_detailed,  # Pass human-readable std type
            weight=weight_param_for_detailed,
            properties=edge_data.copy(),  # Pass the whole merged dict
            description=desc_param_for_detailed,
            keywords=keywords_param_for_detailed,
            source_ids=edge_data.get("source_id"),  # These are strings from merge step
            file_paths=edge_data.get("file_path"),  # These are strings from merge step
        )

    async def upsert_edge_detailed(
        self,
        source_id: str,
        target_id: str,
        rel_type: str = "related",
        weight: float = 0.2,
        merge_strategy: str = "max",
        properties: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        source_ids: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[Union[str, List[str]]] = None,
        keywords: Optional[List[str]] = None,
    ) -> bool:
        """
        Upsert an edge between two nodes.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            weight: Edge weight
            merge_strategy: Strategy for merging properties if edge exists
            properties: Additional edge properties
            description: Edge description
            source_ids: Source IDs for content provenance
            file_paths: File paths for content provenance
            keywords: Keywords associated with the edge

        Returns:
            True if the edge was successfully upserted, False otherwise
        """
        # Ensure both nodes exist but don't overwrite existing entity_type
        # Only create nodes if they don't exist, without setting entity_type to UNKNOWN
        async with self._driver.session(database=self._DATABASE) as session:
            # Check and create source node if it doesn't exist
            source_check_query = """
            MERGE (n:base {entity_id: $entity_id})
            ON CREATE SET n.entity_type = COALESCE(n.entity_type, "UNKNOWN")
            """
            await session.run(source_check_query, entity_id=source_id)

            # Check and create target node if it doesn't exist
            target_check_query = """
            MERGE (n:base {entity_id: $entity_id})
            ON CREATE SET n.entity_type = COALESCE(n.entity_type, "UNKNOWN")
            """
            await session.run(target_check_query, entity_id=target_id)

        # Normalize properties input
        if properties is None:
            properties = {}

        # Start with a fresh dictionary for properties to be stored in Neo4j
        final_properties_for_db = {}

        # 1. Handle direct parameters first and store them stringified if necessary
        if description:
            final_properties_for_db["description"] = description

        # Ensure keywords is a list of strings, then join to a string for DB
        if keywords:  # keywords is a list of strings from the calling function
            if isinstance(keywords, list) and all(
                isinstance(kw, str) for kw in keywords
            ):
                final_properties_for_db["keywords"] = ";".join(
                    keywords
                )  # Join list into a string
            elif isinstance(keywords, str):  # If it's already a string, use it
                final_properties_for_db["keywords"] = keywords
            else:
                utils.logger.warning(
                    f"Keywords for {source_id}->{target_id} are not a list of strings or a string: {keywords}. Storing as string."
                )
                final_properties_for_db["keywords"] = str(keywords)

        if source_ids:
            final_properties_for_db["source_id"] = (
                ";".join(source_ids)
                if isinstance(source_ids, list)
                else str(source_ids)
            )

        if file_paths:
            final_properties_for_db["file_path"] = (
                ";".join(file_paths)
                if isinstance(file_paths, list)
                else str(file_paths)
            )

        # 2. Add custom/additional properties, ensuring they don't overwrite the critical ones handled above
        # unless explicitly intended (e.g. if 'properties' dict has 'description', it will overwrite)
        if (
            properties
        ):  # 'properties' is the dict passed as a parameter to upsert_edge_detailed
            for key, value in properties.items():
                if key not in [
                    "description",
                    "keywords",
                    "source_id",
                    "file_path",
                    "weight",
                    "rel_type",
                    "original_type",
                    "neo4j_type",
                ]:
                    final_properties_for_db[key] = value  # Add other custom properties
                elif (
                    key not in final_properties_for_db
                ):  # Only add if not set by direct params
                    final_properties_for_db[key] = value

        # 3. Handle relationship type fields consistently
        # rel_type parameter should be the human-readable standardized type or LLM raw if advanced_operate not used
        original_type_from_param = (
            rel_type  # This is the input 'rel_type' to this function
        )

        # **CRITICAL FIX**: Prefer neo4j_type from input 'properties' if available (set by advanced_operate)
        # The properties dict comes from _merge_edges_then_upsert which preserves the type information
        neo4j_label_to_use = properties.get("neo4j_type") if properties else None

        if not neo4j_label_to_use or neo4j_label_to_use == "RELATED":
            # Only fall back to registry if no specific type was provided
            utils.logger.warning(
                f"neo4j_type missing or generic in input properties for edge {source_id}->{target_id} using rel_type='{original_type_from_param}'. Standardizing with registry."
            )
            neo4j_label_to_use = self.rel_registry.get_neo4j_type(
                original_type_from_param
            )
        else:
            # We have a specific neo4j_type from advanced processing - use it directly
            utils.logger.debug(
                f"Using neo4j_type '{neo4j_label_to_use}' from properties for edge {source_id}->{target_id}"
            )

        # Validate and sanitize the Neo4j label
        if not neo4j_label_to_use or not isinstance(neo4j_label_to_use, str):
            utils.logger.error(
                f"Invalid Neo4j relationship type '{neo4j_label_to_use}' for {source_id}->{target_id} (from original '{original_type_from_param}'). Defaulting to RELATED_ERROR."
            )
            neo4j_label_to_use = "RELATED_ERROR"  # Use a distinct error type
        elif not re.match(r"^[A-Z0-9_]+$", neo4j_label_to_use):
            # If it doesn't match Neo4j format, try to fix it instead of throwing error
            utils.logger.warning(
                f"Neo4j relationship type '{neo4j_label_to_use}' has invalid format. Attempting to fix."
            )
            # Convert to proper Neo4j format
            neo4j_label_to_use = re.sub(r"[^A-Z0-9_]", "_", neo4j_label_to_use.upper())
            if not neo4j_label_to_use:
                neo4j_label_to_use = "RELATED_ERROR"
            utils.logger.info(
                f"Fixed Neo4j relationship type to '{neo4j_label_to_use}' for edge {source_id}->{target_id}"
            )

        final_properties_for_db["neo4j_type"] = neo4j_label_to_use
        final_properties_for_db["original_type"] = (
            properties.get("original_type", original_type_from_param)
            if properties
            else original_type_from_param
        )
        final_properties_for_db["rel_type"] = (
            properties.get(
                "rel_type",
                self.rel_registry.get_relationship_metadata(original_type_from_param)
                .get("neo4j_type", "RELATED")
                .lower()
                .replace("_", " "),
            )
            if properties
            else original_type_from_param
        )

        # 4. Handle weight (ensure it's float)
        # 'weight' param is the initial weight, properties might contain an override
        final_weight = properties.get(
            "weight", weight
        )  # Prioritize weight from properties if it exists
        try:
            final_weight_float = float(final_weight)
        except (ValueError, TypeError):
            utils.logger.warning(
                f"Invalid weight '{final_weight}' for edge {source_id}->{target_id}. Defaulting to 0.5."
            )
            final_weight_float = 0.5

        final_properties_for_db["weight"] = process_relationship_weight(
            final_weight_float,
            relationship_type=final_properties_for_db[
                "rel_type"
            ],  # Use human-readable std type
            threshold_manager=self.threshold_manager,
        )
        if not isinstance(final_properties_for_db["weight"], float):  # Final check
            utils.logger.error(
                f"Weight for {source_id}->{target_id} not float after processing: {final_properties_for_db['weight']}. Defaulting to 0.2."
            )
            final_properties_for_db["weight"] = 0.2

        # Log what's being sent to Neo4j
        utils.logger.info(
            f"Neo4j Upsert: {source_id}-[{neo4j_label_to_use}]->{target_id} with properties: {final_properties_for_db}"
        )

        # Validate edge data before database operation
        validation_result = DatabaseValidator.validate_edge_data(
            final_properties_for_db
        )
        if validation_result.has_errors():
            error_messages = [e.message for e in validation_result.errors]
            utils.logger.error(
                f"Edge validation failed for '{source_id}' -> '{target_id}': {error_messages}"
            )
            log_validation_errors(
                validation_result.errors, f"upsert_edge({source_id}->{target_id})"
            )
            return False

        if validation_result.has_warnings():
            warning_messages = [w.message for w in validation_result.warnings]
            utils.logger.warning(
                f"Edge validation warnings for '{source_id}' -> '{target_id}': {warning_messages}"
            )
            log_validation_errors(
                validation_result.warnings, f"upsert_edge({source_id}->{target_id})"
            )

        # Create Cypher query for upserting edge - Use the standardized relationship type
        query = f"""
        MATCH (src:base {{entity_id: $source_id}}), (tgt:base {{entity_id: $target_id}})
        MERGE (src)-[r:{neo4j_label_to_use}]->(tgt)
        ON CREATE SET r = $properties_for_db 
        ON MATCH SET r += $properties_for_db 
        RETURN r
        """

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                utils.logger.debug(
                    f"Executing Cypher query in upsert_edge (main upsert): {query} with params: {{'source_id': '{source_id}', 'target_id': '{target_id}', 'properties_for_db': {final_properties_for_db}}}"
                )
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    properties_for_db=final_properties_for_db,  # Pass the cleaned properties
                )
                record = await result.single()
                await result.consume()
                return record is not None
        except Exception as e:
            utils.logger.error(f"Error upserting edge: {str(e)}")
            return False

    async def enhance_edge_weight_with_embeddings(
        self,
        source_id: str,
        target_id: str,
        workspace: str = None,
        rel_type: str = "related",
    ) -> Optional[float]:
        """
        Enhance edge weight using embeddings for semantic similarity.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            workspace: Optional workspace filter
            rel_type: Relationship type

        Returns:
            Enhanced weight based on semantic similarity, or None if enhancement fails
        """
        if not self.pg_connection_params:
            return None

        try:
            # Connect to PostgreSQL
            async with await asyncpg.connect(**self.pg_connection_params) as conn:
                # Get source and target embeddings
                src_embedding = await self._get_entity_embedding(
                    conn, source_id, workspace
                )
                tgt_embedding = await self._get_entity_embedding(
                    conn, target_id, workspace
                )

                if src_embedding is None or tgt_embedding is None:
                    utils.logger.warning(
                        f"Missing embedding for edge {source_id}->{target_id}"
                    )
                    return None

                # Convert to numpy arrays
                src_embedding = np.array(src_embedding)
                tgt_embedding = np.array(tgt_embedding)

                # Calculate semantic weight with threshold manager
                new_weight = calculate_semantic_weight(
                    src_embedding,
                    tgt_embedding,
                    relationship_type=rel_type,
                    threshold_manager=self.threshold_manager,
                )

                # Update edge weight in Neo4j with proper relationship type
                await self._update_edge_weight(
                    source_id, target_id, rel_type, new_weight
                )

                return new_weight

        except Exception as e:
            utils.logger.error(f"Error enhancing edge weight: {str(e)}")
            return None

    async def get_edge(
        self, source_id: str, target_id: str, rel_type: str = "related"
    ) -> Dict[str, Any]:
        """
        Get properties of an edge between two nodes.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type

        Returns:
            Dictionary with edge properties
        """
        # Use actual relationship type in query instead of generic "related"
        query = """
        MATCH (src:base {entity_id: $source_id})-[r:%s]->(tgt:base {entity_id: $target_id})
        RETURN properties(r) as properties
        """ % rel_type.upper()  # Note: rel_type is directly embedded here.

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                utils.logger.debug(
                    f"Executing Cypher query in get_edge (typed): {query} with params: {{'source_id': '{source_id}', 'target_id': '{target_id}'}}"
                )
                result = await session.run(
                    query, source_id=source_id, target_id=target_id
                )
                record = await result.single()
                await result.consume()

                # If no edge is found, return default properties using threshold manager
                if not record or not record.get("properties"):
                    threshold = self.threshold_manager.get_threshold(rel_type)
                    return {
                        "weight": threshold,
                        "rel_type": rel_type,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }

                # Extract properties
                props = record["properties"]

                # Ensure weight is a float
                if "weight" in props:
                    props["weight"] = float(props["weight"])
                else:
                    # Use dynamic threshold if weight is missing
                    props["weight"] = self.threshold_manager.get_threshold(rel_type)

                # Ensure rel_type is present
                if "rel_type" not in props:
                    props["rel_type"] = rel_type

                return props

        except Exception as e:
            utils.logger.error(f"Error getting edge: {str(e)}")
            threshold = self.threshold_manager.get_threshold(rel_type)
            return {
                "weight": threshold,
                "rel_type": rel_type,
                "source_id": None,
                "description": None,
                "keywords": None,
            }

    async def query_graph(
        self,
        query_str: str,
        max_results: int = 50,
        seed_entities: list[str] = None,
        filter_entity_types: list[str] = None,
        filter_relationship_types: list[str] = None,
        min_weight: float = 0.2,
        return_incomplete: bool = True,
        use_embeddings: bool = False,
        **kwargs,
    ) -> KnowledgeGraph:
        """
        Query the Neo4j knowledge graph using a Cypher query.

        Args:
            query_str: Cypher query or semantic search query
            max_results: Maximum number of results to return
            seed_entities: List of entity IDs to start from
            filter_entity_types: List of entity types to filter by
            filter_relationship_types: List of relationship types to filter by
            min_weight: Minimum weight for relationships
            return_incomplete: Return incomplete results if query fails
            use_embeddings: Use embeddings for semantic search

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        result = KnowledgeGraph()

        try:
            # If seed_entities is provided, use them as starting point
            if seed_entities and len(seed_entities) > 0:
                return await self.expand_graph_from_seeds(
                    seed_entities=seed_entities,
                    max_nodes=max_results,
                    filter_entity_types=filter_entity_types,
                    filter_relationship_types=filter_relationship_types,
                    min_weight=min_weight,
                )

            # If it's a Cypher query, run it directly
            if query_str.strip().upper().startswith(
                "MATCH"
            ) or query_str.strip().upper().startswith("CALL"):
                return await self.run_cypher_query(query_str, max_results=max_results)

            # Otherwise use semantic search to find relevant nodes
            nodes = await self.search_entities(
                query_str, limit=min(20, max_results), entity_types=filter_entity_types
            )

            if not nodes:
                return result

            # Get the IDs of the found nodes
            seed_entities = [node.id for node in nodes]

            # Add the found nodes to the result
            for node in nodes:
                result.nodes.append(node)

            # Expand the graph from the seed entities
            expanded = await self.expand_graph_from_seeds(
                seed_entities=seed_entities,
                max_nodes=max_results,
                filter_entity_types=filter_entity_types,
                filter_relationship_types=filter_relationship_types,
                min_weight=min_weight,
            )

            # Merge the expanded graph with the initial result
            for node in expanded.nodes:
                if node.id not in [n.id for n in result.nodes]:
                    result.nodes.append(node)

            result.edges = expanded.edges

            return result

        except Exception as e:
            utils.logger.error(f"Error querying graph: {str(e)}")
            if return_incomplete:
                return result
            raise

    async def expand_graph_from_seeds(
        self,
        seed_entities: list[str],
        max_nodes: int = 50,
        filter_entity_types: list[str] = None,
        filter_relationship_types: list[str] = None,
        min_weight: float = 0.2,
        max_hops: int = 2,
    ) -> KnowledgeGraph:
        """
        Expand graph from seed entities, respecting filters and max nodes.

        Args:
            seed_entities: List of entity IDs to start from
            max_nodes: Maximum number of nodes to return
            filter_entity_types: List of entity types to filter by
            filter_relationship_types: List of relationship types to filter by
            min_weight: Minimum weight for relationships
            max_hops: Maximum number of hops from seed entities

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        result = KnowledgeGraph()

        if not seed_entities:
            return result

        # Construct relationship type filter
        rel_type_filter = ""
        if filter_relationship_types and len(filter_relationship_types) > 0:
            # Format the relationship types for Neo4j
            neo4j_rel_types = [
                rel_type.upper().replace(" ", "_").replace("-", "_")
                for rel_type in filter_relationship_types
            ]
            rel_type_list = "|".join(neo4j_rel_types)
            rel_type_filter = f"type(r) =~ '{rel_type_list}'"

        # Construct node type filter
        node_type_filter = ""
        if filter_entity_types and len(filter_entity_types) > 0:
            entity_types_str = "', '".join(filter_entity_types)
            node_type_filter = f"target.entity_type IN ['{entity_types_str}']"

        # Build combined filter
        combined_filters = []
        if rel_type_filter:
            combined_filters.append(rel_type_filter)
        if node_type_filter:
            combined_filters.append(node_type_filter)
        if min_weight > 0:
            combined_filters.append(f"r.weight >= {min_weight}")

        filter_clause = ""
        if combined_filters:
            filter_clause = "WHERE " + " AND ".join(combined_filters)

        # Construct the query to expand from seeds with typed relationships
        query = f"""
        MATCH (source:base)
        WHERE source.entity_id IN $seed_entities
        MATCH path = (source)-[r*1..{max_hops}]-(target:base)
        {filter_clause}
        UNWIND relationships(path) as rel
        RETURN nodes(path) as path_nodes, startNode(rel) as start_node, endNode(rel) as end_node, rel, type(rel) as rel_type
        LIMIT $max_nodes
        """

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                result = await session.run(
                    query, seed_entities=seed_entities, max_nodes=max_nodes
                )

                seen_nodes = set()
                seen_edges = set()

                async for record in result:
                    if "path_nodes" in record:
                        path_nodes = record["path_nodes"]

                        # Process nodes
                        for node in path_nodes:
                            node_id = node.get("entity_id")
                            if node_id and node_id not in seen_nodes:
                                seen_nodes.add(node_id)

                                # Convert Neo4j node to KnowledgeGraphNode
                                node_dict = dict(node)
                                result.nodes.append(
                                    KnowledgeGraphNode(
                                        id=node_id,
                                        label=node.get("entity_type", "unknown"),
                                        labels=[node.get("entity_type", "unknown")],
                                        properties={
                                            "name": node_id,
                                            "description": node.get("description", ""),
                                            "entity_type": node.get(
                                                "entity_type", "unknown"
                                            ),
                                            **{
                                                k: v
                                                for k, v in node_dict.items()
                                                if k
                                                not in [
                                                    "entity_id",
                                                    "entity_type",
                                                    "description",
                                                ]
                                            },
                                        },
                                    )
                                )

                    # Process relationships using the explicit start_node and end_node
                    if (
                        "start_node" in record
                        and "end_node" in record
                        and "rel" in record
                    ):
                        start_node = record["start_node"]
                        end_node = record["end_node"]
                        rel = record["rel"]
                        rel_type = record["rel_type"]

                        if start_node and end_node:
                            source_id = start_node.get("entity_id")
                            target_id = end_node.get("entity_id")

                            if source_id and target_id:
                                # Create unique edge ID to avoid duplicates
                                edge_key = tuple(
                                    sorted([source_id, target_id, rel_type])
                                )

                                if edge_key not in seen_edges:
                                    seen_edges.add(edge_key)

                                    # Convert Neo4j relationship to KnowledgeGraphEdge
                                    rel_dict = dict(rel)

                                    # Ensure numeric edge weight
                                    edge_weight = 1.0  # Default if not found or invalid
                                    try:
                                        raw_weight = rel_dict.get("weight")
                                        if raw_weight is not None:
                                            edge_weight = float(raw_weight)
                                            # Ensure weight is not NaN or Inf, which can also cause issues
                                            if not (
                                                isinstance(edge_weight, (int, float))
                                                and edge_weight == edge_weight
                                            ):  # Checks for NaN
                                                utils.logger.warning(
                                                    f"Edge {source_id}->{target_id} (type: {rel_type}) has non-finite DB weight '{raw_weight}'. Defaulting to 1.0."
                                                )
                                                edge_weight = 1.0
                                    except (ValueError, TypeError):
                                        utils.logger.warning(
                                            f"Edge {source_id}->{target_id} (type: {rel_type}) has invalid DB weight '{raw_weight}'. Defaulting to 1.0."
                                        )

                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            source=source_id,
                                            target=target_id,
                                            id=f"{source_id}_{target_id}_{rel_type}",
                                            type=rel_type,
                                            properties={
                                                "relationship_type": rel_type,
                                                "weight": edge_weight,  # Ensures it's always a valid float
                                                "description": rel_dict.get(
                                                    "description",
                                                    f"Relationship between {source_id} and {target_id}",
                                                ),
                                                **{
                                                    k: v
                                                    for k, v in rel_dict.items()
                                                    if k
                                                    not in ["weight", "description"]
                                                },
                                            },
                                        )
                                    )

                await result.consume()

            return result

        except Exception as e:
            utils.logger.error(f"Error expanding graph from seeds: {str(e)}")
            return result

    async def search_entities(
        self, query_str: str, limit: int = 20, entity_types: list[str] = None
    ) -> list[KnowledgeGraphNode]:
        """
        Search for entities using text matching and optional type filtering.

        Args:
            query_str: Search query string
            limit: Maximum number of results
            entity_types: List of entity types to filter by

        Returns:
            List of KnowledgeGraphNode objects
        """
        # Build the query with optional entity type filtering
        type_filter = ""
        if entity_types and len(entity_types) > 0:
            entity_types_str = "', '".join(entity_types)
            type_filter = f"AND n.entity_type IN ['{entity_types_str}']"

        query = f"""
        MATCH (n:base)
        WHERE (toLower(n.entity_id) CONTAINS toLower($query_str)
               OR toLower(n.description) CONTAINS toLower($query_str))
              {type_filter}
        RETURN n
        LIMIT $limit
        """

        result_nodes = []

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                result = await session.run(query, query_str=query_str, limit=limit)

                async for record in result:
                    node = record["n"]
                    node_dict = dict(node)

                    result_nodes.append(
                        KnowledgeGraphNode(
                            id=node.get("entity_id", ""),
                            label=node.get("entity_type", "unknown"),
                            labels=[node.get("entity_type", "unknown")],
                            properties={
                                "name": node.get("entity_id", ""),
                                "description": node.get("description", ""),
                                "entity_type": node.get("entity_type", "unknown"),
                                **{
                                    k: v
                                    for k, v in node_dict.items()
                                    if k
                                    not in ["entity_id", "entity_type", "description"]
                                },
                            },
                        )
                    )

                await result.consume()

        except Exception as e:
            utils.logger.error(f"Error searching entities: {str(e)}")

        return result_nodes

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node and all its relationships.

        Args:
            node_id: The entity ID of the node to delete
        """
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Delete node and all its relationships
                query = """
                MATCH (n:base {entity_id: $entity_id})
                DETACH DELETE n
                """
                utils.logger.debug(
                    f"Executing Cypher query in delete_node: {query} with params: {{'entity_id': '{node_id}'}}"
                )
                result = await session.run(query, entity_id=node_id)
                await result.consume()
                utils.logger.debug(f"Deleted node with entity_id: {node_id}")
        except Exception as e:
            utils.logger.error(f"Error deleting node {node_id}: {str(e)}")
            raise

    async def drop(self) -> None:
        """
        Drop the entire graph database (delete all nodes and relationships).
        """
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Delete all nodes and relationships
                query = "MATCH (n) DETACH DELETE n"
                utils.logger.debug(f"Executing Cypher query in drop: {query}")
                result = await session.run(query)
                await result.consume()
                utils.logger.info("Dropped all nodes and relationships from the graph")
        except Exception as e:
            utils.logger.error(f"Error dropping graph: {str(e)}")
            raise

    async def get_all_labels(self) -> list[str]:
        """
        Get all unique entity types/labels in the graph.

        Returns:
            List of unique entity types
        """
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Get all unique entity types
                query = """
                MATCH (n:base)
                WHERE n.entity_type IS NOT NULL
                RETURN DISTINCT n.entity_type AS entity_type
                ORDER BY entity_type
                """
                utils.logger.debug(f"Executing Cypher query in get_all_labels: {query}")
                result = await session.run(query)

                labels = []
                async for record in result:
                    if record["entity_type"]:
                        labels.append(record["entity_type"])

                await result.consume()
                return labels
        except Exception as e:
            utils.logger.error(f"Error getting all labels: {str(e)}")
            return []

    async def get_knowledge_graph(
        self, node_label: str = "*", max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Get the knowledge graph filtered by node label with depth and node limits.

        Args:
            node_label: Label of the starting node, "*" means all nodes
            max_depth: Maximum depth of the subgraph, defaults to 3
            max_nodes: Maximum number of nodes to return, defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        result = KnowledgeGraph()

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Build query based on node_label filter
                if node_label == "*":
                    # Get all nodes without label filtering
                    node_query = """
                    MATCH (n:base)
                    RETURN n
                    LIMIT $max_nodes
                    """
                    node_params = {"max_nodes": max_nodes}
                else:
                    # Filter by specific entity type
                    node_query = """
                    MATCH (n:base)
                    WHERE n.entity_type = $node_label
                    RETURN n
                    LIMIT $max_nodes
                    """
                    node_params = {"node_label": node_label, "max_nodes": max_nodes}

                utils.logger.debug(
                    f"Executing Cypher query in get_knowledge_graph (nodes): {node_query} with params: {node_params}"
                )
                node_result = await session.run(node_query, **node_params)

                seen_nodes = set()
                node_ids = []

                async for record in node_result:
                    node = record["n"]
                    node_id = node.get("entity_id")

                    if node_id and node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        node_ids.append(node_id)
                        node_dict = dict(node)

                        result.nodes.append(
                            KnowledgeGraphNode(
                                id=node_id,
                                label=node.get("entity_type", "unknown"),
                                labels=[node.get("entity_type", "unknown")],
                                properties={
                                    "name": node_id,
                                    "description": node.get("description", ""),
                                    "entity_type": node.get("entity_type", "unknown"),
                                    **{
                                        k: v
                                        for k, v in node_dict.items()
                                        if k
                                        not in [
                                            "entity_id",
                                            "entity_type",
                                            "description",
                                        ]
                                    },
                                },
                            )
                        )

                await node_result.consume()

                # If we have nodes, get their relationships within max_depth
                if node_ids and max_depth > 0:
                    # Modified query to return nodes and relationships separately
                    edge_query = f"""
                    MATCH (source:base)-[rel]->(target:base)
                    WHERE source.entity_id IN $node_ids AND target.entity_id IN $node_ids
                    RETURN source as start_node, target as end_node, rel, type(rel) as rel_type, elementId(rel) as rel_id
                    LIMIT {max_nodes * 5}
                    """

                    utils.logger.debug(
                        f"Executing Cypher query in get_knowledge_graph (edges): {edge_query} with params: {{'node_ids': {node_ids}}}"
                    )
                    edge_result = await session.run(edge_query, node_ids=node_ids)

                    seen_edges = set()
                    async for record in edge_result:
                        start_node = record["start_node"]
                        end_node = record["end_node"]
                        rel = record["rel"]
                        rel_type = record["rel_type"]
                        rel_id = record["rel_id"]  # Neo4j internal relationship ID

                        if start_node and end_node and rel_id:
                            rel_source = start_node.get("entity_id")
                            rel_target = end_node.get("entity_id")

                            if rel_source and rel_target:
                                # Use Neo4j internal relationship ID to ensure true uniqueness
                                if rel_id not in seen_edges:
                                    seen_edges.add(rel_id)

                                    # Get relationship properties
                                    props = dict(rel) if rel else {}

                                    # Ensure numeric edge weight
                                    edge_weight = 1.0
                                    try:
                                        raw_weight = props.get("weight")
                                        if raw_weight is not None:
                                            edge_weight = float(raw_weight)
                                            # Ensure weight is not NaN or Inf, which can also cause issues
                                            if not (
                                                isinstance(edge_weight, (int, float))
                                                and edge_weight == edge_weight
                                            ):  # Checks for NaN
                                                utils.logger.warning(
                                                    f"Edge {rel_source}->{rel_target} (type: {rel_type}) has non-finite DB weight '{raw_weight}'. Defaulting to 1.0."
                                                )
                                                edge_weight = 1.0
                                    except (ValueError, TypeError):
                                        utils.logger.warning(
                                            f"Edge {rel_source}->{rel_target} (type: {rel_type}) has invalid DB weight '{raw_weight}'. Defaulting to 1.0."
                                        )

                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            source=rel_source,
                                            target=rel_target,
                                            id=rel_id,  # Use Neo4j's unique relationship ID
                                            type=rel_type,
                                            properties={
                                                "relationship_type": rel_type,
                                                "weight": edge_weight,  # Ensures it's always a valid float
                                                "description": props.get(
                                                    "description",
                                                    f"Relationship between {rel_source} and {rel_target}",
                                                ),
                                                "neo4j_id": rel_id,  # Keep the Neo4j ID for reference
                                                **{
                                                    k: v
                                                    for k, v in props.items()
                                                    if k
                                                    not in ["weight", "description"]
                                                },
                                            },
                                        )
                                    )

                    await edge_result.consume()

        except Exception as e:
            utils.logger.error(f"Error getting knowledge graph: {str(e)}")

        return result

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """
        Remove multiple edges from the graph.

        Args:
            edges: List of (source_id, target_id) tuples representing edges to remove
        """
        if not edges:
            return

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Use UNWIND for batch deletion
                query = """
                UNWIND $edges AS edge
                MATCH (source:base {entity_id: edge.source})-[r]-(target:base {entity_id: edge.target})
                DELETE r
                """

                # Convert edges to the format expected by the query
                edge_params = [
                    {"source": source, "target": target} for source, target in edges
                ]

                utils.logger.debug(
                    f"Executing Cypher query in remove_edges: {query} with params: {{'edges': {edge_params}}}"
                )
                result = await session.run(query, edges=edge_params)
                await result.consume()
                utils.logger.debug(f"Removed {len(edges)} edges from the graph")
        except Exception as e:
            utils.logger.error(f"Error removing edges: {str(e)}")
            raise

    async def remove_nodes(self, node_ids: list[str]) -> None:
        """
        Remove multiple nodes and their relationships from the graph.

        Args:
            node_ids: List of entity IDs to remove
        """
        if not node_ids:
            return

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Use UNWIND for batch deletion
                query = """
                UNWIND $node_ids AS node_id
                MATCH (n:base {entity_id: node_id})
                DETACH DELETE n
                """

                utils.logger.debug(
                    f"Executing Cypher query in remove_nodes: {query} with params: {{'node_ids': {node_ids}}}"
                )
                result = await session.run(query, node_ids=node_ids)
                await result.consume()
                utils.logger.debug(f"Removed {len(node_ids)} nodes from the graph")
        except Exception as e:
            utils.logger.error(f"Error removing nodes: {str(e)}")
            raise

    async def run_cypher_query(
        self, query: str, max_results: int = 50
    ) -> KnowledgeGraph:
        """
        Run a raw Cypher query and return results as a KnowledgeGraph.

        Args:
            query: Cypher query string
            max_results: Maximum number of results to process

        Returns:
            KnowledgeGraph object containing query results
        """
        result = KnowledgeGraph()

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                utils.logger.info(f"Executing raw Cypher query: {query}")
                cypher_result = await session.run(query)

                records_processed = 0
                async for record in cypher_result:
                    if records_processed >= max_results:
                        break

                    # Try to extract nodes and relationships from the record
                    for key, value in record.items():
                        if hasattr(value, "labels"):  # It's a node
                            node_id = value.get("entity_id")
                            if node_id:
                                node_dict = dict(value)
                                result.nodes.append(
                                    KnowledgeGraphNode(
                                        id=node_id,
                                        label=value.get("entity_type", "unknown"),
                                        labels=[value.get("entity_type", "unknown")],
                                        properties={
                                            "name": node_id,
                                            "description": value.get("description", ""),
                                            "entity_type": value.get(
                                                "entity_type", "unknown"
                                            ),
                                            **{
                                                k: v
                                                for k, v in node_dict.items()
                                                if k
                                                not in [
                                                    "entity_id",
                                                    "entity_type",
                                                    "description",
                                                ]
                                            },
                                        },
                                    )
                                )
                        elif hasattr(value, "type"):  # It's a relationship
                            if hasattr(value, "start_node") and hasattr(
                                value, "end_node"
                            ):
                                source_id = value.start_node.get("entity_id")
                                target_id = value.end_node.get("entity_id")
                                rel_type = value.type

                                if source_id and target_id:
                                    rel_dict = dict(value)

                                    # Ensure numeric edge weight (PRD 4.2.2)
                                    edge_weight = 1.0  # Default if not found or invalid
                                    try:
                                        raw_weight = rel_dict.get("weight")
                                        if raw_weight is not None:
                                            edge_weight = float(raw_weight)
                                    except (ValueError, TypeError):
                                        utils.logger.warning(
                                            f"Edge {source_id}->{target_id} has invalid DB weight '{raw_weight}'. Defaulting to 1.0."
                                        )

                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            source=source_id,
                                            target=target_id,
                                            id=f"{source_id}_{target_id}_{rel_type}",
                                            type=rel_type,
                                            properties={
                                                "relationship_type": rel_type,
                                                "weight": edge_weight,  # Ensures float
                                                "description": rel_dict.get(
                                                    "description",
                                                    f"Relationship between {source_id} and {target_id}",
                                                ),
                                                **{
                                                    k: v
                                                    for k, v in rel_dict.items()
                                                    if k
                                                    not in ["weight", "description"]
                                                },
                                            },
                                        )
                                    )

                    records_processed += 1

                await cypher_result.consume()

        except Exception as e:
            utils.logger.error(f"Error running Cypher query: {str(e)}")

        return result

    async def _get_entity_embedding(
        self, conn, entity_id: str, workspace: str = None
    ) -> Optional[list]:
        """
        Helper method to get entity embedding from PostgreSQL.

        Args:
            conn: PostgreSQL connection
            entity_id: Entity ID to get embedding for
            workspace: Optional workspace filter

        Returns:
            Embedding vector as list or None if not found
        """
        try:
            table_name = "lightrag_vdb_entity"
            workspace_filter = ""
            params = [entity_id]

            if workspace:
                workspace_filter = " AND workspace = $2"
                params.append(workspace)

            query = f"""
            SELECT content_vector FROM {table_name}
            WHERE id = $1{workspace_filter}
            LIMIT 1
            """

            row = await conn.fetchrow(query, *params)

            if row:
                return row["content_vector"]
            return None

        except Exception as e:
            utils.logger.warning(
                f"Error getting entity embedding for {entity_id}: {str(e)}"
            )
            return None

    async def _update_edge_weight(
        self, source_id: str, target_id: str, rel_type: str, weight: float
    ) -> None:
        """
        Helper method to update edge weight in Neo4j.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            weight: New weight value
        """
        try:
            # Convert relationship type for Neo4j
            neo4j_rel_type = rel_type.upper().replace(" ", "_").replace("-", "_")

            query = f"""
            MATCH (src:base {{entity_id: $source_id}})-[r:{neo4j_rel_type}]->(tgt:base {{entity_id: $target_id}})
            SET r.weight = $weight
            RETURN r
            """

            async with self._driver.session(database=self._DATABASE) as session:
                utils.logger.debug(
                    f"Executing Cypher query in _update_edge_weight: {query} with params: {{'source_id': '{source_id}', 'target_id': '{target_id}', 'weight': {weight}}}"
                )
                result = await session.run(
                    query, source_id=source_id, target_id=target_id, weight=weight
                )
                await result.consume()

        except Exception as e:
            utils.logger.error(f"Error updating edge weight: {str(e)}")
