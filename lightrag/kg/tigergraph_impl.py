import os
import re
import asyncio
from dataclasses import dataclass
from typing import final
from enum import StrEnum
import configparser
from urllib.parse import urlparse

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
from ..kg.shared_storage import get_data_init_lock, get_graph_db_lock
import pipmaster as pm

if not pm.is_installed("pyTigerGraph"):
    pm.install("pyTigerGraph")

from pyTigerGraph import TigerGraphConnection  # type: ignore

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


# Set pyTigerGraph logger level to ERROR to suppress warning logs
logging.getLogger("pyTigerGraph").setLevel(logging.ERROR)


class VertexType(StrEnum):
    """Vertex types used in TigerGraph schema."""

    ENTITY = "LR_Entity"


class EdgeType(StrEnum):
    """Edge types used in TigerGraph schema."""

    RELATES = "LR_Relates"


@dataclass
class EdgeDefinition:
    """Definition of an edge type and the vertex types it connects."""

    edge_type: EdgeType
    from_vertex: VertexType
    to_vertex: VertexType


@final
@dataclass
class TigerGraphStorage(BaseGraphStorage):
    # Schema definition: list of edges with their vertex connections
    # This allows for extensibility - add more edge/vertex types here
    _EDGE_DEFINITIONS: list[EdgeDefinition] | None = None

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Initialize edge definitions if not already set
        if TigerGraphStorage._EDGE_DEFINITIONS is None:
            TigerGraphStorage._EDGE_DEFINITIONS = [
                EdgeDefinition(
                    edge_type=EdgeType.RELATES,
                    from_vertex=VertexType.ENTITY,
                    to_vertex=VertexType.ENTITY,
                )
            ]

        # Read env and override the arg if present
        tigergraph_workspace = os.environ.get("TIGERGRAPH_WORKSPACE")
        if tigergraph_workspace and tigergraph_workspace.strip():
            workspace = tigergraph_workspace

        # Default to 'base' when both arg and env are empty
        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._conn = None
        self._graph_name = None

    def _get_workspace_label(self) -> str:
        """Return workspace label (guaranteed non-empty during initialization)"""
        return self.workspace

    def _is_chinese_text(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
        return bool(chinese_pattern.search(text))

    def _parse_uri(self, uri: str) -> tuple[str, int]:
        """Parse URI to extract host and port for TigerGraphConnection.

        Returns:
            tuple: (hostname, port) where hostname is just the hostname (no port, no scheme)
                   as pyTigerGraph expects host and port separately or hostname:port format
        """
        parsed = urlparse(uri)
        hostname = parsed.hostname or "localhost"

        # Extract port from URI
        if parsed.port:
            port = parsed.port
        elif ":" in parsed.netloc:
            # Port might be in netloc but not parsed by urlparse (e.g., if scheme is missing)
            netloc_parts = parsed.netloc.split(":")
            if len(netloc_parts) >= 2:
                try:
                    port = int(netloc_parts[-1])  # Take last part as port
                except ValueError:
                    # Not a valid port, use default
                    port = 9000 if parsed.scheme != "https" else 443
            else:
                port = 9000 if parsed.scheme != "https" else 443
        else:
            # No port in URI, use default
            port = 9000 if parsed.scheme != "https" else 443

        # Return just hostname (pyTigerGraph will handle port separately or via hostname:port format)
        # But to avoid double port issues, return hostname:port as a single string
        # This way pyTigerGraph gets exactly what we parsed
        host_string = f"{hostname}:{port}"

        return host_string, port

    def _escape_filter_value(self, value: str) -> str:
        """Escape special characters in filter values for TigerGraph."""
        # Escape single quotes by doubling them
        return value.replace("'", "''")

    async def initialize(self):
        """
        Initialize TigerGraph connection and ensure graph and schema exist.

        Note on GRAPH_NAME vs WORKSPACE:
        - GRAPH_NAME: The TigerGraph graph instance (like a database in Neo4j).
          Defaults to sanitized namespace, can be overridden via TIGERGRAPH_GRAPH_NAME env var.
          Multiple workspaces can share the same graph.
        - WORKSPACE: Used as a label in the labels SET<STRING> attribute for data isolation.
          All vertices use the VertexType.ENTITY.value vertex type, with workspace and entity_type stored
          as labels in the labels SET attribute (similar to multi-label support in Neo4j/Memgraph).
        """
        async with get_data_init_lock():
            URI = os.environ.get(
                "TIGERGRAPH_URI", config.get("tigergraph", "uri", fallback=None)
            )
            USERNAME = os.environ.get(
                "TIGERGRAPH_USERNAME",
                config.get("tigergraph", "username", fallback="tigergraph"),
            )
            PASSWORD = os.environ.get(
                "TIGERGRAPH_PASSWORD",
                config.get("tigergraph", "password", fallback=""),
            )
            GRAPH_NAME = os.environ.get(
                "TIGERGRAPH_GRAPH_NAME",
                config.get(
                    "tigergraph",
                    "graph_name",
                    fallback=re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace),
                ),
            )
            # GSQL port (default 14240)
            GS_PORT = os.environ.get(
                "TIGERGRAPH_GS_PORT",
                config.get("tigergraph", "gs_port", fallback="14240"),
            )
            # Convert to int if it's a string
            try:
                GS_PORT = int(GS_PORT) if isinstance(GS_PORT, str) else GS_PORT
            except ValueError:
                logger.warning(
                    f"[{self.workspace}] Invalid TIGERGRAPH_GS_PORT value '{GS_PORT}', using default 14240"
                )
                GS_PORT = 14240

            if not URI:
                raise ValueError(
                    "TIGERGRAPH_URI is required. Please set it in environment variables or config.ini"
                )

            # Parse URI to get host and port
            host, port = self._parse_uri(URI)
            self._graph_name = GRAPH_NAME

            # Initialize TigerGraph connection (synchronous)
            # First connect without graphname to check/create graph
            def _init_connection():
                # Parse the original URI to get scheme and hostname
                parsed = urlparse(URI)
                scheme = parsed.scheme or "http"
                hostname = parsed.hostname or "localhost"
                # pyTigerGraph expects host as "http://hostname" (with protocol, without port)
                # and restppPort as a separate parameter
                host_url = f"{scheme}://{hostname}"

                # Connect without graphname first to check/create graph
                conn = TigerGraphConnection(
                    host=host_url,  # Full URL with protocol, no port
                    restppPort=port,  # REST++ port
                    gsPort=GS_PORT,  # GSQL port
                    username=USERNAME,
                    password=PASSWORD,
                )

                # Check if graph exists using GSQL
                try:
                    # Use GSQL to list graphs
                    result = conn.gsql("LS")
                    # Parse the result - GSQL LS command returns a string listing graphs
                    graph_exists = GRAPH_NAME in str(result)

                    if not graph_exists:
                        # Note: In TigerGraph, the graph is created implicitly when the first
                        # schema element (vertex or edge type) is created in _ensure_graph_and_schema
                        logger.info(
                            f"[{self.workspace}] Graph '{GRAPH_NAME}' does not exist. "
                            "It will be created when schema is defined."
                        )
                    else:
                        logger.debug(
                            f"[{self.workspace}] Graph '{GRAPH_NAME}' already exists."
                        )
                except Exception as e:
                    # If GSQL LS fails, try to continue - graph might be created during schema definition
                    logger.debug(
                        f"[{self.workspace}] Could not check graph existence: {str(e)}. "
                        "Will attempt to create during schema definition."
                    )

                # Now connect with graphname for operations
                # Use the same host_url and port as above
                conn = TigerGraphConnection(
                    host=host_url,  # Full URL with protocol, no port
                    restppPort=port,  # REST++ port
                    gsPort=GS_PORT,  # GSQL port
                    username=USERNAME,
                    password=PASSWORD,
                    graphname=GRAPH_NAME,
                )
                return conn

            # Run in thread pool to avoid blocking
            self._conn = await asyncio.to_thread(_init_connection)
            logger.info(
                f"[{self.workspace}] Connected to TigerGraph at {host} (graph: {GRAPH_NAME}, workspace: {self.workspace})"
            )

            # Ensure graph and schema exist
            await self._ensure_graph_and_schema()

    async def _ensure_graph_and_schema(self):
        """
        Ensure the graph exists and schema is defined with required vertex and edge types.

        In TigerGraph, the graph is created implicitly when the first schema element
        (vertex or edge type) is created. We'll create the schema which will create
        the graph if it doesn't exist.

        Uses VertexType.ENTITY.value as the single vertex type with multi-label support via labels SET<STRING>.
        """
        workspace_label = self._get_workspace_label()

        def _graph_exists(graph_name: str) -> bool:
            """Check if a graph exists using USE GRAPH command."""
            try:
                result = self._conn.gsql(f"USE GRAPH {graph_name}")
                result_str = str(result).lower()
                # If the graph doesn't exist, USE GRAPH returns an error message
                error_patterns = [
                    "does not exist",
                    "doesn't exist",
                    "doesn't exist!",
                    f"graph '{graph_name.lower()}' does not exist",
                ]
                for pattern in error_patterns:
                    if pattern in result_str:
                        return False
                return True
            except Exception as e:
                error_str = str(e).lower()
                if "does not exist" in error_str or "doesn't exist" in error_str:
                    return False
                # If exception doesn't indicate "doesn't exist", assume it exists
                return True

        def _create_graph_and_schema():
            """
            Create vertex and edge types globally, then create graph with those types.

            According to TigerGraph docs:
            CREATE GRAPH Graph_Name (Vertex_Or_Edge_Type, Vertex_Or_Edge_Type...)

            This creates the graph with the specified types in one command.

            Uses the generic edge definitions list to create all vertex and edge types.
            """
            # Collect all unique vertex types from edge definitions
            vertex_types_to_create = set()
            for edge_def in TigerGraphStorage._EDGE_DEFINITIONS:
                vertex_types_to_create.add(edge_def.from_vertex)
                vertex_types_to_create.add(edge_def.to_vertex)

            # Step 1: Create all vertex types globally (must exist before CREATE GRAPH)
            for vertex_type in vertex_types_to_create:
                # For now, all vertex types use the same schema
                # In the future, this could be made configurable per vertex type
                gsql_create_vertex = f"""CREATE VERTEX {vertex_type.value} (
    PRIMARY_ID entity_id STRING,
    labels SET<STRING>,
    entity_type STRING,
    description STRING,
    keywords STRING,
    source_id STRING,
    file_path STRING,
    created_at INT,
    truncate STRING
) WITH primary_id_as_attribute="true"
"""
                try:
                    self._conn.gsql(gsql_create_vertex)
                    logger.info(
                        f"[{self.workspace}] Created vertex type '{vertex_type.value}'"
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if (
                        "used by another object" in error_str
                        or "already exists" in error_str
                    ):
                        logger.debug(
                            f"[{self.workspace}] Vertex type '{vertex_type.value}' already exists"
                        )
                    else:
                        logger.error(
                            f"[{self.workspace}] Failed to create vertex type: {e}"
                        )
                        raise

            # Step 2: Create all edge types globally (must exist before CREATE GRAPH)
            # Each edge explicitly references the vertex types it connects
            for edge_def in TigerGraphStorage._EDGE_DEFINITIONS:
                gsql_create_edge = f"""CREATE UNDIRECTED EDGE {edge_def.edge_type.value} (
    FROM {edge_def.from_vertex.value},
    TO {edge_def.to_vertex.value},
    weight FLOAT DEFAULT 1.0,
    description STRING,
    keywords STRING,
    source_id STRING,
    file_path STRING,
    created_at INT,
    truncate STRING
)
"""
                try:
                    self._conn.gsql(gsql_create_edge)
                    logger.info(
                        f"[{self.workspace}] Created edge type '{edge_def.edge_type.value}' "
                        f"(FROM {edge_def.from_vertex.value} TO {edge_def.to_vertex.value})"
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if (
                        "used by another object" in error_str
                        or "already exists" in error_str
                    ):
                        logger.debug(
                            f"[{self.workspace}] Edge type '{edge_def.edge_type.value}' already exists"
                        )
                    else:
                        logger.error(
                            f"[{self.workspace}] Failed to create edge type: {e}"
                        )
                        raise

            # Step 3: Create graph with all types (or ensure types are in existing graph)
            graph_exists = _graph_exists(self._graph_name)

            # Build list of all types for CREATE GRAPH command
            all_types = [vt.value for vt in vertex_types_to_create]
            all_types.extend(
                [ed.edge_type.value for ed in TigerGraphStorage._EDGE_DEFINITIONS]
            )
            types_str = ", ".join(all_types)

            if not graph_exists:
                # Create graph with all types in one command
                logger.info(
                    f"[{self.workspace}] Creating graph '{self._graph_name}' with types: {types_str}"
                )
                gsql_create_graph = f"CREATE GRAPH {self._graph_name} ({types_str})"
                try:
                    self._conn.gsql(gsql_create_graph)
                    logger.info(
                        f"[{self.workspace}] Created graph '{self._graph_name}' with types"
                    )
                except Exception as e:
                    logger.error(f"[{self.workspace}] Failed to create graph: {e}")
                    raise
            else:
                # Graph exists - check if types are in it, add if missing using schema change job
                logger.info(
                    f"[{self.workspace}] Graph '{self._graph_name}' exists. Checking if types are associated..."
                )

                # Check current schema
                try:
                    schema = self._conn.getSchema(force=True)
                    vertex_types = [vt["Name"] for vt in schema.get("VertexTypes", [])]
                    edge_types = [et["Name"] for et in schema.get("EdgeTypes", [])]

                    # Build list of types to add
                    types_to_add = []
                    for vertex_type in vertex_types_to_create:
                        if vertex_type.value not in vertex_types:
                            types_to_add.append(("VERTEX", vertex_type.value))

                    for edge_def in TigerGraphStorage._EDGE_DEFINITIONS:
                        if edge_def.edge_type.value not in edge_types:
                            types_to_add.append(("EDGE", edge_def.edge_type.value))

                    if types_to_add:
                        # Use schema change job to add types to existing graph
                        job_name = f"add_types_to_{self._graph_name}"

                        # Build ADD statements with correct syntax: ADD VERTEX/EDGE ... to graph ...
                        add_statements = []
                        for type_kind, type_name in types_to_add:
                            add_statements.append(
                                f"  ADD {type_kind} {type_name} to graph {self._graph_name};"
                            )

                        gsql_schema_change = f"""CREATE GLOBAL SCHEMA_CHANGE JOB {job_name} {{{chr(10).join(add_statements)}}} RUN GLOBAL SCHEMA_CHANGE JOB {job_name}"""
                        try:
                            # Drop job if it exists (cleanup from previous runs)
                            try:
                                self._conn.gsql(f"DROP JOB {job_name}")
                            except Exception:
                                pass  # Job doesn't exist, which is fine

                            # Create and run the schema change job
                            self._conn.gsql(gsql_schema_change)
                            logger.info(
                                f"[{self.workspace}] Added {len(types_to_add)} type(s) to graph '{self._graph_name}' "
                                f"using schema change job: {[t[1] for t in types_to_add]}"
                            )
                        except Exception as e:
                            error_str = str(e).lower()
                            if (
                                "already" in error_str
                                or "exist" in error_str
                                or "added" in error_str
                            ):
                                logger.debug(
                                    f"[{self.workspace}] Types may already be in graph: {[t[1] for t in types_to_add]}"
                                )
                            else:
                                logger.warning(
                                    f"[{self.workspace}] Could not add types to graph using schema change job: {e}"
                                )
                                # Fallback: try ALTER GRAPH for each type
                                for type_kind, type_name in types_to_add:
                                    try:
                                        if type_kind == "VERTEX":
                                            gsql_alter = (
                                                f"USE GRAPH {self._graph_name}\n"
                                                f"ALTER GRAPH {self._graph_name} ADD VERTEX {type_name}"
                                            )
                                        else:  # EDGE
                                            gsql_alter = (
                                                f"USE GRAPH {self._graph_name}\n"
                                                f"ALTER GRAPH {self._graph_name} ADD UNDIRECTED EDGE {type_name}"
                                            )
                                        self._conn.gsql(gsql_alter)
                                        logger.info(
                                            f"[{self.workspace}] Added {type_kind} '{type_name}' to graph (fallback method)"
                                        )
                                    except Exception as fallback_error:
                                        logger.warning(
                                            f"[{self.workspace}] Could not add {type_kind} '{type_name}' to graph: {fallback_error}"
                                        )
                    else:
                        logger.debug(
                            f"[{self.workspace}] All types already in graph: "
                            f"vertices={[vt.value for vt in vertex_types_to_create]}, "
                            f"edges={[ed.edge_type.value for ed in TigerGraphStorage._EDGE_DEFINITIONS]}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[{self.workspace}] Could not check/add types to graph: {e}"
                    )

            # Install GSQL queries for efficient operations
            self._install_queries(workspace_label)

        await asyncio.to_thread(_create_graph_and_schema)

    def _install_queries(self, workspace_label: str):
        """Install GSQL queries for efficient graph operations."""
        try:
            vertex_type = VertexType.ENTITY.value
            edge_type = EdgeType.RELATES.value

            # Query to get popular labels by degree
            # This query counts edges per vertex and returns sorted by degree
            # Filters by workspace label in the labels SET
            popular_labels_query = f"""
            CREATE QUERY get_popular_labels_{workspace_label}(INT limit) FOR GRAPH {self._graph_name} {{
                MapAccum<STRING, INT> @@degree_map;
                HeapAccum<Tuple2<INT, STRING>>(limit, f0 DESC, f1 ASC) @@top_labels;

                # Initialize all vertices with degree 0, filtered by workspace label
                Start = {{{vertex_type}.*}};
                Start = SELECT v FROM Start:v
                    WHERE v.entity_id != "" AND "{workspace_label}" IN v.labels
                    ACCUM @@degree_map += (v.entity_id -> 0);

                # Count edges (both directions for undirected graph)
                Start = SELECT v FROM Start:v - ({edge_type}:e) - {vertex_type}:t
                    WHERE v.entity_id != "" AND t.entity_id != ""
                        AND "{workspace_label}" IN v.labels
                        AND "{workspace_label}" IN t.labels
                    ACCUM @@degree_map += (v.entity_id -> 1);

                # Build heap with degree and label, sorted by degree DESC, label ASC
                Start = SELECT v FROM Start:v
                    WHERE v.entity_id != "" AND "{workspace_label}" IN v.labels
                    POST-ACCUM
                        INT degree = @@degree_map.get(v.entity_id),
                        @@top_labels += Tuple2(degree, v.entity_id);

                # Extract labels from heap (already sorted)
                ListAccum<STRING> @@result;
                FOREACH item IN @@top_labels DO
                    @@result += item.f1;
                END;

                PRINT @@result;
            }}
            """

            # Query to search labels with fuzzy matching
            # This query filters vertices by entity_id containing the search query
            # Filters by workspace label in the labels SET
            search_labels_query = f"""
            CREATE QUERY search_labels_{workspace_label}(STRING search_query, INT limit) FOR GRAPH {self._graph_name} {{
                ListAccum<STRING> @@matches;
                STRING query_lower = lower(search_query);

                Start = {{{vertex_type}.*}};
                Start = SELECT v FROM Start:v
                    WHERE v.entity_id != ""
                        AND str_contains(lower(v.entity_id), query_lower)
                        AND "{workspace_label}" IN v.labels
                    ACCUM @@matches += v.entity_id;

                PRINT @@matches;
            }}
            """

            # Try to install queries (drop first if they exist)
            try:
                # Drop existing queries if they exist
                try:
                    self._conn.gsql(f"DROP QUERY get_popular_labels_{workspace_label}")
                except Exception:
                    pass  # Query doesn't exist, which is fine

                try:
                    self._conn.gsql(f"DROP QUERY search_labels_{workspace_label}")
                except Exception:
                    pass  # Query doesn't exist, which is fine

                # Install new queries
                self._conn.gsql(popular_labels_query)
                self._conn.gsql(search_labels_query)
                logger.info(
                    f"[{self.workspace}] Installed GSQL queries for workspace '{workspace_label}'"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.workspace}] Could not install GSQL queries: {str(e)}. "
                    "Will fall back to traversal-based methods."
                )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Error installing GSQL queries: {str(e)}. "
                "Will fall back to traversal-based methods."
            )

    async def finalize(self):
        """Close the TigerGraph connection and release all resources"""
        async with get_graph_db_lock():
            if self._conn:
                # TigerGraph connection doesn't have explicit close, but we can clear reference
                self._conn = None

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure connection is closed when context manager exits"""
        await self.finalize()

    async def index_done_callback(self) -> None:
        # TigerGraph handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        workspace_label = self._get_workspace_label()

        def _check_node():
            try:
                # Use getVerticesById since entity_id is the PRIMARY_ID
                try:
                    result = self._conn.getVerticesById(
                        VertexType.ENTITY.value, node_id
                    )
                    if isinstance(result, dict) and node_id in result:
                        attrs = result[node_id].get("attributes", {})
                        labels = attrs.get("labels", set())
                        if isinstance(labels, set) and workspace_label in labels:
                            return True
                    return False
                except Exception:
                    # Fallback: try with filter using double quotes
                    escaped_node_id = self._escape_filter_value(node_id)
                    result = self._conn.getVertices(
                        VertexType.ENTITY.value,
                        where=f'entity_id=="{escaped_node_id}"',
                        limit=10,
                    )
                    # Filter by workspace label in labels SET
                    for vertex in result:
                        labels = vertex.get("attributes", {}).get("labels", set())
                        if isinstance(labels, set) and workspace_label in labels:
                            return True
                    return False
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking node existence for {node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_check_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""
        # workspace_label = self._get_workspace_label()

        def _check_edge():
            try:
                # Check both directions for undirected graph
                try:
                    result1 = self._conn.getEdges(
                        VertexType.ENTITY.value,
                        source_node_id,
                        EdgeType.RELATES.value,
                        VertexType.ENTITY.value,
                        target_node_id,
                        limit=1,
                    )
                    if result1 and len(result1) > 0:
                        return True
                except Exception as e1:
                    # Error code 602 means edge doesn't exist, which is fine
                    error_str = str(e1).lower()
                    if (
                        "602" not in str(e1)
                        and "does not have an edge" not in error_str
                    ):
                        logger.debug(
                            f"[{self.workspace}] Error checking edge from {source_node_id} to {target_node_id}: {str(e1)}"
                        )

                try:
                    result2 = self._conn.getEdges(
                        VertexType.ENTITY.value,
                        target_node_id,
                        EdgeType.RELATES.value,
                        VertexType.ENTITY.value,
                        source_node_id,
                        limit=1,
                    )
                    if result2 and len(result2) > 0:
                        return True
                except Exception as e2:
                    # Error code 602 means edge doesn't exist, which is fine
                    error_str = str(e2).lower()
                    if (
                        "602" not in str(e2)
                        and "does not have an edge" not in error_str
                    ):
                        logger.debug(
                            f"[{self.workspace}] Error checking edge from {target_node_id} to {source_node_id}: {str(e2)}"
                        )

                # No edge found in either direction
                return False
            except Exception as e:
                # For any other unexpected error, log and return False
                logger.debug(
                    f"[{self.workspace}] Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                return False

        return await asyncio.to_thread(_check_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its entity_id, return only node properties."""
        workspace_label = self._get_workspace_label()

        def _get_node():
            try:
                # Use getVerticesById since entity_id is the PRIMARY_ID
                # This avoids filter syntax issues and is more efficient
                try:
                    result = self._conn.getVerticesById(
                        VertexType.ENTITY.value, node_id
                    )
                except Exception:
                    # If getVerticesById fails, try with filter using double quotes
                    escaped_node_id = self._escape_filter_value(node_id)
                    result = self._conn.getVertices(
                        VertexType.ENTITY.value,
                        where=f'entity_id=="{escaped_node_id}"',
                        limit=10,
                    )

                # Filter by workspace label in labels SET
                # Note: TigerGraph returns labels as a list in JSON, not a set
                matching_vertices = []
                if isinstance(result, dict):
                    # getVerticesById returns a dict {vertex_id: {attributes: {...}}}
                    for vertex_id, vertex_data in result.items():
                        if vertex_id == node_id:
                            attrs = vertex_data.get("attributes", {})
                            labels = attrs.get("labels", [])
                            # Handle both list and set (list from JSON, set from Python)
                            if (
                                isinstance(labels, (list, set, tuple))
                                and workspace_label in labels
                            ):
                                matching_vertices.append({"attributes": attrs})
                elif isinstance(result, list):
                    # getVertices returns a list of vertex dicts
                    for vertex in result:
                        labels = vertex.get("attributes", {}).get("labels", [])
                        # Handle both list and set (list from JSON, set from Python)
                        if (
                            isinstance(labels, (list, set, tuple))
                            and workspace_label in labels
                        ):
                            matching_vertices.append(vertex)

                if len(matching_vertices) > 1:
                    logger.warning(
                        f"[{self.workspace}] Multiple nodes found with entity_id '{node_id}'. Using first node."
                    )
                if matching_vertices:
                    node_data = matching_vertices[0]["attributes"].copy()
                    # Convert labels to list if needed, and filter out workspace label, entity_type, and "UNKNOWN"
                    # Labels should only be used for workspace filtering, not for storing entity_type
                    if "labels" in node_data:
                        labels = node_data["labels"]
                        if isinstance(labels, (set, tuple)):
                            labels_list = list(labels)
                        else:
                            labels_list = (
                                labels.copy() if isinstance(labels, list) else []
                            )
                        # Remove workspace label, entity_type, and "UNKNOWN" from labels list
                        # Only workspace should be in labels for filtering - everything else should be filtered out
                        entity_type = node_data.get("entity_type")
                        labels_list = [
                            label
                            for label in labels_list
                            if label != workspace_label
                            and label != entity_type
                            and label != "UNKNOWN"
                        ]
                        node_data["labels"] = labels_list
                    # Keep entity_id in the dict
                    if "entity_id" not in node_data:
                        node_data["entity_id"] = node_id
                    return node_data
                return None
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node for {node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_get_node)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in batch."""
        workspace_label = self._get_workspace_label()

        def _get_nodes_batch():
            nodes = {}
            try:
                # TigerGraph doesn't have native batch query, so we query in parallel
                # For now, iterate through node_ids
                for node_id in node_ids:
                    try:
                        # Use getVerticesById for primary key lookup
                        result = []
                        try:
                            vertex_result = self._conn.getVerticesById(
                                VertexType.ENTITY.value, node_id
                            )
                            # getVerticesById returns {vertex_id: {attributes: {...}}}
                            # The key might be node_id or might be formatted differently
                            if isinstance(vertex_result, dict) and vertex_result:
                                # Try to find the vertex by checking all keys
                                for vid, vdata in vertex_result.items():
                                    attrs = vdata.get("attributes", {})
                                    # Verify this is the node we're looking for by checking entity_id
                                    if (
                                        attrs.get("entity_id") == node_id
                                        or vid == node_id
                                    ):
                                        result.append({"attributes": attrs})
                                        break

                            # If getVerticesById returned empty dict or no match found, try filter
                            if not result:
                                escaped_node_id = self._escape_filter_value(node_id)
                                result = self._conn.getVertices(
                                    VertexType.ENTITY.value,
                                    where=f'entity_id=="{escaped_node_id}"',
                                    limit=10,
                                )
                        except Exception as e:
                            # Fallback to filter with double quotes if getVerticesById raises exception
                            logger.debug(
                                f"[{self.workspace}] getVerticesById failed for {node_id}, trying filter: {e}"
                            )
                            try:
                                escaped_node_id = self._escape_filter_value(node_id)
                                result = self._conn.getVertices(
                                    VertexType.ENTITY.value,
                                    where=f'entity_id=="{escaped_node_id}"',
                                    limit=10,
                                )
                            except Exception as e2:
                                logger.debug(
                                    f"[{self.workspace}] Filter also failed for {node_id}: {e2}"
                                )
                                result = []
                        # Filter by workspace label in labels SET
                        # Note: TigerGraph returns labels as a list in JSON, not a set
                        if not result:
                            logger.debug(
                                f"[{self.workspace}] No vertex found for node_id '{node_id}'"
                            )
                        else:
                            for vertex in result:
                                attrs = vertex.get("attributes", {})
                                labels = attrs.get("labels", [])
                                # Handle both list and set (list from JSON, set from Python)
                                if isinstance(labels, (list, set, tuple)):
                                    # Check if workspace label is in labels
                                    if workspace_label in labels:
                                        node_data = attrs.copy()
                                        # Convert labels to list and filter out workspace label, entity_type, and "UNKNOWN"
                                        # Labels should only be used for workspace filtering, not for storing entity_type
                                        if isinstance(labels, (set, tuple)):
                                            labels_list = list(labels)
                                        else:
                                            labels_list = labels.copy()

                                        entity_type = node_data.get("entity_type")
                                        labels_list = [
                                            label
                                            for label in labels_list
                                            if label != workspace_label
                                            and label != entity_type
                                            and label != "UNKNOWN"
                                        ]
                                        node_data["labels"] = labels_list
                                        # Ensure entity_id is in the dict
                                        if "entity_id" not in node_data:
                                            node_data["entity_id"] = node_id
                                        nodes[node_id] = node_data
                                        break  # Found matching node, move to next
                                    else:
                                        # Debug: log when workspace label doesn't match
                                        logger.debug(
                                            f"[{self.workspace}] Node '{node_id}' found but workspace label '{workspace_label}' not in labels: {labels}"
                                        )
                                else:
                                    logger.debug(
                                        f"[{self.workspace}] Node '{node_id}' has invalid labels format: {type(labels)}, value: {labels}"
                                    )
                    except Exception as e:
                        logger.warning(
                            f"[{self.workspace}] Error getting node {node_id}: {str(e)}"
                        )
                return nodes
            except Exception as e:
                logger.error(f"[{self.workspace}] Error in batch get nodes: {str(e)}")
                raise

        return await asyncio.to_thread(_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node."""
        workspace_label = self._get_workspace_label()

        def _get_degree():
            try:
                # TigerGraph's getEdges doesn't support '*' as wildcard
                # Instead, we use getEdges without target vertex ID to get all outgoing edges
                # and then get all incoming edges by querying from other vertices
                # However, a simpler approach is to use getNodeNeighbors or get all edges

                # Method 1: Use getEdges with empty string for target (gets all outgoing edges)
                # Then we need to get incoming edges separately
                try:
                    # Get outgoing edges (from this node to any target)
                    # pyTigerGraph's getEdges signature: getEdges(sourceVertexType, sourceVertexId, edgeType, targetVertexType, targetVertexId)
                    # When targetVertexId is not provided or is empty, it should return all edges
                    # But let's use getNodeNeighbors if available, or fetch all edges from get_node_edges

                    # Use get_node_edges which already handles both directions
                    edges = self._get_node_edges_sync(node_id, workspace_label)
                    if edges:
                        # Count unique edges (avoid double counting for undirected graph)
                        edge_pairs = set()
                        for source, target in edges:
                            # Normalize edge direction for undirected graph
                            if source < target:
                                edge_pairs.add((source, target))
                            else:
                                edge_pairs.add((target, source))
                        return len(edge_pairs)
                    return 0
                except Exception as e:
                    # Fallback: try to get edges using a different method
                    logger.debug(
                        f"[{self.workspace}] Error getting node degree via get_node_edges for {node_id}: {str(e)}"
                    )
                    # Try direct getEdges call - but we need to know all possible targets
                    # This is inefficient, so we'll return 0 as fallback
                    return 0
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node degree for {node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_get_degree)

    def _get_node_edges_sync(
        self, source_node_id: str, workspace_label: str
    ) -> list[tuple[str, str]]:
        """Synchronous helper to get node edges (used by node_degree)."""
        try:
            # TigerGraph's getEdges doesn't support '*' as wildcard
            # Use getEdges with only source vertex (omitting target parameters) to get all outgoing edges
            # pyTigerGraph's getEdges signature allows omitting target parameters

            edges = []
            edge_pairs = set()  # To avoid duplicates for undirected graph

            # Get outgoing edges: call getEdges with only source vertex type and ID
            # When target parameters are omitted, it should return all edges from this vertex
            try:
                # Try calling getEdges with minimal parameters (source only)
                result1 = self._conn.getEdges(
                    VertexType.ENTITY.value,
                    source_node_id,
                    EdgeType.RELATES.value,
                )

                if isinstance(result1, list):
                    for edge in result1:
                        # Extract target ID from edge
                        target_id = None
                        if isinstance(edge, dict):
                            target_id = edge.get("to_id") or edge.get("to")
                            if isinstance(target_id, dict):
                                target_id = target_id.get("v_id") or target_id.get("id")

                        if target_id:
                            # Normalize for undirected graph
                            pair = (
                                (source_node_id, target_id)
                                if source_node_id < target_id
                                else (target_id, source_node_id)
                            )
                            if pair not in edge_pairs:
                                edge_pairs.add(pair)
                                edges.append((source_node_id, target_id))
            except Exception as e1:
                logger.debug(
                    f"[{self.workspace}] Error getting outgoing edges for {source_node_id}: {str(e1)}"
                )

            # For incoming edges, we need a different approach
            # Since we can't query with '*' as source, we'll use getNeighbors if available
            # or we'll need to query all vertices (inefficient but necessary)
            try:
                # Try getNeighbors method if available
                if hasattr(self._conn, "getNeighbors"):
                    neighbors = self._conn.getNeighbors(
                        VertexType.ENTITY.value, source_node_id, EdgeType.RELATES.value
                    )
                    if isinstance(neighbors, list):
                        for neighbor in neighbors:
                            target_id = None
                            if isinstance(neighbor, dict):
                                target_id = neighbor.get("v_id") or neighbor.get("id")
                            if target_id and target_id != source_node_id:
                                # Normalize for undirected graph
                                pair = (
                                    (source_node_id, target_id)
                                    if source_node_id < target_id
                                    else (target_id, source_node_id)
                                )
                                if pair not in edge_pairs:
                                    edge_pairs.add(pair)
                                    edges.append((source_node_id, target_id))
            except Exception as e2:
                logger.debug(
                    f"[{self.workspace}] Error getting neighbors for {source_node_id}: {str(e2)}"
                )

            return edges
        except Exception as e:
            logger.debug(
                f"[{self.workspace}] Error in _get_node_edges_sync for {source_node_id}: {str(e)}"
            )
            return []

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve the degree for multiple nodes in batch."""
        degrees = {}
        for node_id in node_ids:
            degree = await self.node_degree(node_id)
            degrees[node_id] = degree
        return degrees

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes."""
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(trg_degree)

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Calculate the combined degree for each edge in batch."""
        # Collect unique node IDs
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        # Get degrees for all nodes
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Sum up degrees for each edge pair
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes."""
        # workspace_label = self._get_workspace_label()

        def _get_edge():
            try:
                # Check both directions for undirected graph
                result1 = self._conn.getEdges(
                    VertexType.ENTITY.value,
                    source_node_id,
                    EdgeType.RELATES.value,
                    VertexType.ENTITY.value,
                    target_node_id,
                    limit=2,
                )
                result2 = self._conn.getEdges(
                    VertexType.ENTITY.value,
                    target_node_id,
                    EdgeType.RELATES.value,
                    VertexType.ENTITY.value,
                    source_node_id,
                    limit=2,
                )

                if len(result1) > 1 or len(result2) > 1:
                    logger.warning(
                        f"[{self.workspace}] Multiple edges found between '{source_node_id}' and '{target_node_id}'. Using first edge."
                    )

                if result1:
                    edge_attrs = result1[0].get("attributes", {})
                    # Ensure required keys exist with defaults
                    required_keys = {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                        "file_path": None,
                        "created_at": None,
                        "truncate": None,
                    }
                    for key, default_value in required_keys.items():
                        if key not in edge_attrs:
                            edge_attrs[key] = default_value
                    return edge_attrs
                elif result2:
                    edge_attrs = result2[0].get("attributes", {})
                    # Ensure required keys exist with defaults
                    required_keys = {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                        "file_path": None,
                        "created_at": None,
                        "truncate": None,
                    }
                    for key, default_value in required_keys.items():
                        if key not in edge_attrs:
                            edge_attrs[key] = default_value
                    return edge_attrs
                return None
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_get_edge)

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple (src, tgt) pairs."""
        edges_dict = {}
        for pair in pairs:
            src = pair["src"]
            tgt = pair["tgt"]
            edge = await self.get_edge(src, tgt)
            if edge is not None:
                edges_dict[(src, tgt)] = edge
            else:
                # Set default edge properties
                edges_dict[(src, tgt)] = {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                    "file_path": None,
                    "created_at": None,
                    "truncate": None,
                }
        return edges_dict

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node."""
        workspace_label = self._get_workspace_label()

        def _get_node_edges():
            try:
                # Use the same helper method as node_degree to avoid '*' wildcard issue
                edges = self._get_node_edges_sync(source_node_id, workspace_label)
                return edges if edges else None
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting edges for node {source_node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_get_node_edges)

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes."""
        edges_dict = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            edges_dict[node_id] = edges if edges is not None else []
        return edges_dict

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, OSError, Exception)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the TigerGraph database."""
        workspace_label = self._get_workspace_label()

        def _upsert_node():
            try:
                # Make a copy to avoid modifying the original
                node_data_copy = node_data.copy()

                # Ensure entity_id is in node_data
                if "entity_id" not in node_data_copy:
                    node_data_copy["entity_id"] = node_id

                # Ensure labels SET includes ONLY workspace (for filtering/isolation)
                # entity_type should NOT be in labels - it's stored in entity_type property
                # Always set labels to contain only workspace, regardless of what's in node_data
                # This ensures entity_type never seeps into labels, even if it was there before
                # Explicitly remove labels key first, then set it fresh to avoid any merge behavior
                if "labels" in node_data_copy:
                    del node_data_copy["labels"]

                # Set labels to only contain workspace_label, explicitly filtering out "UNKNOWN"
                # Even though workspace_label should never be "UNKNOWN", we filter to be safe
                # and to prevent any accidental inclusion of "UNKNOWN" from old data
                labels_to_set = (
                    [workspace_label]
                    if workspace_label and workspace_label != "UNKNOWN"
                    else []
                )
                node_data_copy["labels"] = labels_to_set

                # Upsert vertex
                self._conn.upsertVertex(
                    VertexType.ENTITY.value, node_id, node_data_copy
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error during node upsert for {node_id}: {str(e)}"
                )
                raise

        await asyncio.to_thread(_upsert_node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, OSError, Exception)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Upsert an edge and its properties between two nodes."""

        # Ensure both nodes exist first - use upsert_node to ensure clean labels
        # This ensures all node creation goes through the same label-cleaning logic
        source_exists = await self.has_node(source_node_id)
        if not source_exists:
            # Create source node with minimal data - upsert_node will ensure clean labels
            await self.upsert_node(
                source_node_id,
                {
                    "entity_id": source_node_id,
                    "entity_type": "UNKNOWN",
                    "description": "",
                    "source_id": "",
                    "file_path": "",
                    "created_at": 0,
                },
            )

        target_exists = await self.has_node(target_node_id)
        if not target_exists:
            # Create target node with minimal data - upsert_node will ensure clean labels
            await self.upsert_node(
                target_node_id,
                {
                    "entity_id": target_node_id,
                    "entity_type": "UNKNOWN",
                    "description": "",
                    "source_id": "",
                    "file_path": "",
                    "created_at": 0,
                },
            )

        # Upsert edge (undirected, so direction doesn't matter)
        def _upsert_edge():
            try:
                self._conn.upsertEdge(
                    VertexType.ENTITY.value,
                    source_node_id,
                    EdgeType.RELATES.value,
                    VertexType.ENTITY.value,
                    target_node_id,
                    edge_data,
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error during edge upsert between {source_node_id} and {target_node_id}: {str(e)}"
                )
                raise

        await asyncio.to_thread(_upsert_edge)

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.
        """
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        workspace_label = self._get_workspace_label()
        result = KnowledgeGraph()

        def _get_knowledge_graph():
            try:
                if node_label == "*":
                    # Get all nodes sorted by degree, filtered by workspace label
                    # TigerGraph REST API doesn't support IN operator for SET attributes
                    # So we fetch all and filter in Python
                    all_vertices_raw = self._conn.getVertices(
                        VertexType.ENTITY.value,
                        limit=max_nodes * 2,  # Fetch more to account for filtering
                    )
                    # Filter by workspace label in labels SET
                    all_vertices = []
                    for vertex in all_vertices_raw:
                        labels = vertex.get("attributes", {}).get("labels", set())
                        if isinstance(labels, set) and workspace_label in labels:
                            all_vertices.append(vertex)
                            if len(all_vertices) >= max_nodes:
                                break
                    # For simplicity, take first max_nodes vertices
                    # In a real implementation, you'd want to sort by degree
                    vertices = all_vertices[:max_nodes]
                    if len(all_vertices) > max_nodes:
                        result.is_truncated = True

                    # Build node and edge sets
                    node_ids = [v["attributes"].get("entity_id") for v in vertices]
                    node_ids = [nid for nid in node_ids if nid]

                    # Get all edges between these nodes
                    edges_data = []
                    for node_id in node_ids:
                        try:
                            node_edges = self._conn.getEdges(
                                VertexType.ENTITY.value,
                                node_id,
                                EdgeType.RELATES.value,
                                VertexType.ENTITY.value,
                                "*",
                                limit=10000,
                            )
                            for edge in node_edges:
                                target_id = edge.get("to_id")
                                if target_id in node_ids:
                                    edges_data.append(edge)
                        except Exception:
                            continue

                    # Build result
                    for vertex in vertices:
                        attrs = vertex.get("attributes", {})
                        entity_id = attrs.get("entity_id")
                        if entity_id:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=entity_id,
                                    labels=[entity_id],
                                    properties=attrs,
                                )
                            )

                    edge_ids_seen = set()
                    for edge in edges_data:
                        source_id = edge.get("from_id")
                        target_id = edge.get("to_id")
                        if source_id and target_id:
                            edge_tuple = tuple(sorted([source_id, target_id]))
                            if edge_tuple not in edge_ids_seen:
                                edge_attrs = edge.get("attributes", {})
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=f"{source_id}-{target_id}",
                                        type=EdgeType.RELATES.value,
                                        source=source_id,
                                        target=target_id,
                                        properties=edge_attrs,
                                    )
                                )
                                edge_ids_seen.add(edge_tuple)
                else:
                    # BFS traversal starting from node_label
                    from collections import deque

                    visited_nodes = set()
                    visited_edges = set()
                    queue = deque([(node_label, 0)])

                    while queue and len(visited_nodes) < max_nodes:
                        current_id, depth = queue.popleft()

                        if current_id in visited_nodes or depth > max_depth:
                            continue

                        # Get node (filter by workspace label in Python)
                        try:
                            # Use getVerticesById for primary key lookup
                            try:
                                vertex_result = self._conn.getVerticesById(
                                    VertexType.ENTITY.value, current_id
                                )
                                vertices_raw = []
                                if (
                                    isinstance(vertex_result, dict)
                                    and current_id in vertex_result
                                ):
                                    vertices_raw.append(
                                        {
                                            "attributes": vertex_result[current_id].get(
                                                "attributes", {}
                                            )
                                        }
                                    )
                            except Exception:
                                # Fallback to filter with double quotes
                                escaped_current_id = self._escape_filter_value(
                                    current_id
                                )
                                vertices_raw = self._conn.getVertices(
                                    VertexType.ENTITY.value,
                                    where=f'entity_id=="{escaped_current_id}"',
                                    limit=10,
                                )
                            # Filter by workspace label in labels SET
                            vertices = []
                            for v in vertices_raw:
                                labels = v.get("attributes", {}).get("labels", set())
                                if (
                                    isinstance(labels, set)
                                    and workspace_label in labels
                                ):
                                    vertices.append(v)

                            if not vertices:
                                continue

                            vertex = vertices[0]
                            attrs = vertex.get("attributes", {})
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=current_id,
                                    labels=[current_id],
                                    properties=attrs,
                                )
                            )
                            visited_nodes.add(current_id)

                            if depth < max_depth:
                                # Get neighbors
                                edges = self._conn.getEdges(
                                    VertexType.ENTITY.value,
                                    current_id,
                                    EdgeType.RELATES.value,
                                    VertexType.ENTITY.value,
                                    "*",
                                    limit=10000,
                                )
                                for edge in edges:
                                    target_id = edge.get("to_id")
                                    if target_id and target_id not in visited_nodes:
                                        edge_tuple = tuple(
                                            sorted([current_id, target_id])
                                        )
                                        if edge_tuple not in visited_edges:
                                            edge_attrs = edge.get("attributes", {})
                                            result.edges.append(
                                                KnowledgeGraphEdge(
                                                    id=f"{current_id}-{target_id}",
                                                    type=EdgeType.RELATES.value,
                                                    source=current_id,
                                                    target=target_id,
                                                    properties=edge_attrs,
                                                )
                                            )
                                            visited_edges.add(edge_tuple)
                                        queue.append((target_id, depth + 1))
                        except Exception as e:
                            logger.warning(
                                f"[{self.workspace}] Error in BFS traversal for {current_id}: {str(e)}"
                            )
                            continue

                    if len(visited_nodes) >= max_nodes:
                        result.is_truncated = True

                return result
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error in get_knowledge_graph: {str(e)}"
                )
                raise

        result = await asyncio.to_thread(_get_knowledge_graph)
        logger.info(
            f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def get_all_labels(self) -> list[str]:
        """Get all existing node labels in the database."""
        workspace_label = self._get_workspace_label()

        def _get_all_labels():
            try:
                # TigerGraph REST API doesn't support IN operator for SET attributes
                # So we fetch all and filter in Python
                vertices_raw = self._conn.getVertices(
                    VertexType.ENTITY.value, limit=100000
                )
                # Filter by workspace label in labels SET
                vertices = []
                for vertex in vertices_raw:
                    labels = vertex.get("attributes", {}).get("labels", set())
                    if isinstance(labels, set) and workspace_label in labels:
                        vertices.append(vertex)
                labels = set()
                for vertex in vertices:
                    entity_id = vertex.get("attributes", {}).get("entity_id")
                    if entity_id:
                        labels.add(entity_id)
                return sorted(list(labels))
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting all labels: {str(e)}")
                raise

        return await asyncio.to_thread(_get_all_labels)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, OSError, Exception)),
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified entity_id."""
        workspace_label = self._get_workspace_label()

        def _delete_node():
            try:
                # Check if node exists with workspace label first, then delete
                try:
                    result = self._conn.getVerticesById(
                        VertexType.ENTITY.value, node_id
                    )
                    if isinstance(result, dict) and node_id in result:
                        attrs = result[node_id].get("attributes", {})
                        labels = attrs.get("labels", set())
                        if isinstance(labels, set) and workspace_label in labels:
                            # Delete this specific vertex
                            escaped_node_id = self._escape_filter_value(node_id)
                            self._conn.delVertices(
                                VertexType.ENTITY.value,
                                where=f'entity_id=="{escaped_node_id}"',
                            )
                except Exception:
                    # Node doesn't exist or error occurred
                    pass
                logger.debug(
                    f"[{self.workspace}] Deleted node with entity_id '{node_id}'"
                )
            except Exception as e:
                logger.error(f"[{self.workspace}] Error during node deletion: {str(e)}")
                raise

        await asyncio.to_thread(_delete_node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, OSError, Exception)),
    )
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, OSError, Exception)),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges."""
        # workspace_label = self._get_workspace_label()

        def _delete_edge(source, target):
            try:
                # Delete edge in both directions
                self._conn.delEdges(
                    VertexType.ENTITY.value,
                    source,
                    EdgeType.RELATES.value,
                    VertexType.ENTITY.value,
                    target,
                )
            except Exception as e:
                logger.warning(
                    f"[{self.workspace}] Error deleting edge from '{source}' to '{target}': {str(e)}"
                )

        for source, target in edges:
            await asyncio.to_thread(_delete_edge, source, target)

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph."""
        workspace_label = self._get_workspace_label()

        def _get_all_nodes():
            try:
                # TigerGraph REST API doesn't support IN operator for SET attributes
                # So we fetch all and filter in Python
                vertices_raw = self._conn.getVertices(
                    VertexType.ENTITY.value, limit=100000
                )
                # Filter by workspace label in labels SET
                vertices = []
                for vertex in vertices_raw:
                    labels = vertex.get("attributes", {}).get("labels", set())
                    if isinstance(labels, set) and workspace_label in labels:
                        vertices.append(vertex)
                nodes = []
                for vertex in vertices:
                    attrs = vertex.get("attributes", {})
                    attrs["id"] = attrs.get("entity_id")
                    # Convert labels SET to list and filter out workspace label, entity_type, and "UNKNOWN"
                    # Labels should only be used for workspace filtering, not for storing entity_type
                    if "labels" in attrs and isinstance(attrs["labels"], set):
                        entity_type = attrs.get("entity_type")
                        attrs["labels"] = [
                            label
                            for label in attrs["labels"]
                            if label != workspace_label
                            and label != entity_type
                            and label != "UNKNOWN"
                        ]
                    nodes.append(attrs)
                return nodes
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting all nodes: {str(e)}")
                raise

        return await asyncio.to_thread(_get_all_nodes)

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph."""
        workspace_label = self._get_workspace_label()

        def _get_all_edges():
            try:
                # TigerGraph REST API doesn't support IN operator for SET attributes
                # So we fetch all and filter in Python
                vertices_raw = self._conn.getVertices(
                    VertexType.ENTITY.value, limit=100000
                )
                # Filter by workspace label in labels SET
                vertices = []
                for vertex in vertices_raw:
                    labels = vertex.get("attributes", {}).get("labels", set())
                    if isinstance(labels, set) and workspace_label in labels:
                        vertices.append(vertex)
                edges = []
                processed_edges = set()

                for vertex in vertices:
                    source_id = vertex.get("attributes", {}).get("entity_id")
                    if not source_id:
                        continue

                    try:
                        vertex_edges = self._conn.getEdges(
                            VertexType.ENTITY.value,
                            source_id,
                            EdgeType.RELATES.value,
                            VertexType.ENTITY.value,
                            "*",
                            limit=10000,
                        )
                        for edge in vertex_edges:
                            target_id = edge.get("to_id")
                            edge_tuple = tuple(sorted([source_id, target_id]))
                            if edge_tuple not in processed_edges:
                                edge_attrs = edge.get("attributes", {})
                                edge_attrs["source"] = source_id
                                edge_attrs["target"] = target_id
                                edges.append(edge_attrs)
                                processed_edges.add(edge_tuple)
                    except Exception:
                        continue

                return edges
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting all edges: {str(e)}")
                raise

        return await asyncio.to_thread(_get_all_edges)

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities)."""
        workspace_label = self._get_workspace_label()

        def _get_popular_labels():
            try:
                # Try to use installed GSQL query first
                query_name = f"get_popular_labels_{workspace_label}"
                try:
                    result = self._conn.runInstalledQuery(
                        query_name, params={"limit": limit}
                    )
                    if result and len(result) > 0:
                        # Extract labels from query result
                        # Result format: [{"@@result": ["label1", "label2", ...]}]
                        labels = []
                        for record in result:
                            if "@@result" in record:
                                labels.extend(record["@@result"])

                        # GSQL query already returns sorted labels (by degree DESC, label ASC)
                        # Just return the limited results
                        if labels:
                            return labels[:limit]
                except Exception as query_error:
                    logger.debug(
                        f"[{self.workspace}] GSQL query '{query_name}' not available or failed: {str(query_error)}. "
                        "Falling back to traversal method."
                    )

                # Fallback to traversal method if GSQL query fails
                # TigerGraph REST API doesn't support IN operator for SET attributes
                # So we fetch all and filter in Python
                vertices_raw = self._conn.getVertices(
                    VertexType.ENTITY.value, limit=100000
                )
                # Filter by workspace label in labels SET
                vertices = []
                for vertex in vertices_raw:
                    labels = vertex.get("attributes", {}).get("labels", set())
                    if isinstance(labels, set) and workspace_label in labels:
                        vertices.append(vertex)
                node_degrees = {}

                for vertex in vertices:
                    entity_id = vertex.get("attributes", {}).get("entity_id")
                    if not entity_id:
                        continue

                    # Calculate degree
                    try:
                        edges = self._conn.getEdges(
                            VertexType.ENTITY.value,
                            entity_id,
                            EdgeType.RELATES.value,
                            VertexType.ENTITY.value,
                            "*",
                            limit=10000,
                        )
                        # Count unique neighbors
                        neighbors = set()
                        for edge in edges:
                            target_id = edge.get("to_id")
                            if target_id:
                                neighbors.add(target_id)
                        node_degrees[entity_id] = len(neighbors)
                    except Exception:
                        node_degrees[entity_id] = 0

                # Sort by degree descending, then by label ascending
                sorted_labels = sorted(
                    node_degrees.items(),
                    key=lambda x: (-x[1], x[0]),
                )[:limit]

                return [label for label, _ in sorted_labels]
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting popular labels: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_get_popular_labels)

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching."""
        workspace_label = self._get_workspace_label()
        query_strip = query.strip()
        if not query_strip:
            return []

        query_lower = query_strip.lower()
        is_chinese = self._is_chinese_text(query_strip)

        def _search_labels():
            try:
                # Try to use installed GSQL query first
                query_name = f"search_labels_{workspace_label}"
                try:
                    result = self._conn.runInstalledQuery(
                        query_name, params={"search_query": query_strip, "limit": limit}
                    )
                    if result and len(result) > 0:
                        # Extract labels from query result
                        labels = []
                        for record in result:
                            if "@@matches" in record:
                                labels.extend(record["@@matches"])

                        if labels:
                            # GSQL query does basic filtering, we still need to score and sort
                            # Score the results (exact match, prefix match, contains match)
                            matches = []
                            for entity_id_str in labels:
                                if is_chinese:
                                    # For Chinese, use direct contains
                                    if query_strip not in entity_id_str:
                                        continue
                                    # Calculate relevance score
                                    if entity_id_str == query_strip:
                                        score = 1000
                                    elif entity_id_str.startswith(query_strip):
                                        score = 500
                                    else:
                                        score = 100 - len(entity_id_str)
                                else:
                                    # For non-Chinese, use case-insensitive contains
                                    entity_id_lower = entity_id_str.lower()
                                    if query_lower not in entity_id_lower:
                                        continue
                                    # Calculate relevance score
                                    if entity_id_lower == query_lower:
                                        score = 1000
                                    elif entity_id_lower.startswith(query_lower):
                                        score = 500
                                    else:
                                        score = 100 - len(entity_id_str)
                                        # Bonus for word boundary matches
                                        if (
                                            f" {query_lower}" in entity_id_lower
                                            or f"_{query_lower}" in entity_id_lower
                                        ):
                                            score += 50

                                matches.append((entity_id_str, score))

                            # Sort by relevance score (desc) then alphabetically
                            matches.sort(key=lambda x: (-x[1], x[0]))

                            # Return top matches
                            return [match[0] for match in matches[:limit]]
                except Exception as query_error:
                    logger.debug(
                        f"[{self.workspace}] GSQL query '{query_name}' not available or failed: {str(query_error)}. "
                        "Falling back to traversal method."
                    )

                # Fallback to traversal method if GSQL query fails
                # TigerGraph REST API doesn't support IN operator for SET attributes
                # So we fetch all and filter in Python
                vertices_raw = self._conn.getVertices(
                    VertexType.ENTITY.value, limit=100000
                )
                # Filter by workspace label in labels SET
                vertices = []
                for vertex in vertices_raw:
                    labels = vertex.get("attributes", {}).get("labels", set())
                    if isinstance(labels, set) and workspace_label in labels:
                        vertices.append(vertex)
                matches = []

                for vertex in vertices:
                    entity_id = vertex.get("attributes", {}).get("entity_id")
                    if not entity_id:
                        continue

                    entity_id_str = str(entity_id)
                    if is_chinese:
                        # For Chinese, use direct contains
                        if query_strip not in entity_id_str:
                            continue
                        # Calculate relevance score
                        if entity_id_str == query_strip:
                            score = 1000
                        elif entity_id_str.startswith(query_strip):
                            score = 500
                        else:
                            score = 100 - len(entity_id_str)
                    else:
                        # For non-Chinese, use case-insensitive contains
                        entity_id_lower = entity_id_str.lower()
                        if query_lower not in entity_id_lower:
                            continue
                        # Calculate relevance score
                        if entity_id_lower == query_lower:
                            score = 1000
                        elif entity_id_lower.startswith(query_lower):
                            score = 500
                        else:
                            score = 100 - len(entity_id_str)
                            # Bonus for word boundary matches
                            if (
                                f" {query_lower}" in entity_id_lower
                                or f"_{query_lower}" in entity_id_lower
                            ):
                                score += 50

                    matches.append((entity_id_str, score))

                # Sort by relevance score (desc) then alphabetically
                matches.sort(key=lambda x: (-x[1], x[0]))

                # Return top matches
                return [match[0] for match in matches[:limit]]
            except Exception as e:
                logger.error(f"[{self.workspace}] Error searching labels: {str(e)}")
                raise

        return await asyncio.to_thread(_search_labels)

    async def drop(self) -> dict[str, str]:
        """Drop all data from current workspace storage and clean up resources."""
        async with get_graph_db_lock():
            workspace_label = self._get_workspace_label()
            try:

                def _drop():
                    # TigerGraph REST API doesn't support IN operator for SET attributes
                    # So we fetch all vertices, filter by workspace label, and delete them
                    vertices_raw = self._conn.getVertices(
                        VertexType.ENTITY.value, limit=100000
                    )
                    # Filter by workspace label and collect entity_ids to delete
                    entity_ids_to_delete = []
                    for vertex in vertices_raw:
                        labels = vertex.get("attributes", {}).get("labels", set())
                        if isinstance(labels, set) and workspace_label in labels:
                            entity_id = vertex.get("attributes", {}).get("entity_id")
                            if entity_id:
                                entity_ids_to_delete.append(entity_id)

                    # Delete vertices by entity_id
                    for entity_id in entity_ids_to_delete:
                        try:
                            escaped_entity_id = self._escape_filter_value(entity_id)
                            self._conn.delVertices(
                                VertexType.ENTITY.value,
                                where=f'entity_id=="{escaped_entity_id}"',
                            )
                        except Exception as e:
                            logger.warning(f"Could not delete vertex {entity_id}: {e}")

                await asyncio.to_thread(_drop)
                return {
                    "status": "success",
                    "message": f"workspace '{workspace_label}' data dropped",
                }
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error dropping TigerGraph workspace '{workspace_label}': {e}"
                )
                return {"status": "error", "message": str(e)}
