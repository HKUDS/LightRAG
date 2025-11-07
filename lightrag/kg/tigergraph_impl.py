import os
import re
import asyncio
from dataclasses import dataclass
from typing import final
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


@final
@dataclass
class TigerGraphStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
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
        """Parse URI to extract host and port."""
        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # Construct full URL with scheme
        if not parsed.scheme:
            scheme = "http"
        else:
            scheme = parsed.scheme
        full_host = f"{scheme}://{host}:{port}"
        return full_host, port

    async def initialize(self):
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

            if not URI:
                raise ValueError(
                    "TIGERGRAPH_URI is required. Please set it in environment variables or config.ini"
                )

            # Parse URI to get host and port
            host, port = self._parse_uri(URI)
            self._graph_name = GRAPH_NAME

            # Initialize TigerGraph connection (synchronous)
            def _init_connection():
                conn = TigerGraphConnection(
                    host=host,
                    username=USERNAME,
                    password=PASSWORD,
                    graphname=GRAPH_NAME,
                )
                # Test connection
                try:
                    conn.getVertices("Entity", limit=1)
                except Exception as e:
                    # If graph doesn't exist, we'll create schema in _ensure_schema
                    logger.debug(
                        f"[{self.workspace}] Graph may not exist yet: {str(e)}"
                    )
                return conn

            # Run in thread pool to avoid blocking
            self._conn = await asyncio.to_thread(_init_connection)
            logger.info(
                f"[{self.workspace}] Connected to TigerGraph at {host} (graph: {GRAPH_NAME})"
            )

            # Ensure schema exists
            await self._ensure_schema()

    async def _ensure_schema(self):
        """Ensure the graph schema exists with required vertex and edge types."""
        workspace_label = self._get_workspace_label()

        def _create_schema():
            # Create vertex type for entities (similar to Neo4j workspace label)
            # Use workspace label as vertex type name
            vertex_type = workspace_label

            # Check if vertex type exists
            try:
                schema = self._conn.getSchema(force=True)
                vertex_types = [vt["Name"] for vt in schema["VertexTypes"]]
                if vertex_type not in vertex_types:
                    # Create vertex type with entity_id as primary key
                    # All properties will be stored as attributes
                    gsql = f"""
                    CREATE VERTEX {vertex_type} (
                        PRIMARY_ID entity_id STRING,
                        entity_type STRING,
                        description STRING,
                        keywords STRING,
                        source_id STRING
                    ) WITH primary_id_as_attribute="true"
                    """
                    self._conn.gsql(gsql)
                    logger.info(
                        f"[{self.workspace}] Created vertex type '{vertex_type}'"
                    )
            except Exception as e:
                # If vertex type creation fails, try to continue
                logger.warning(
                    f"[{self.workspace}] Could not create vertex type '{vertex_type}': {str(e)}"
                )

            # Create edge type for relationships (undirected, similar to Neo4j)
            edge_type = "DIRECTED"
            try:
                schema = self._conn.getSchema(force=True)
                edge_types = [et["Name"] for et in schema["EdgeTypes"]]
                if edge_type not in edge_types:
                    # Create undirected edge type
                    gsql = f"""
                    CREATE UNDIRECTED EDGE {edge_type} (
                        FROM {vertex_type},
                        TO {vertex_type},
                        weight FLOAT DEFAULT 1.0,
                        description STRING,
                        keywords STRING,
                        source_id STRING
                    )
                    """
                    self._conn.gsql(gsql)
                    logger.info(f"[{self.workspace}] Created edge type '{edge_type}'")
            except Exception as e:
                logger.warning(
                    f"[{self.workspace}] Could not create edge type '{edge_type}': {str(e)}"
                )

            # Install GSQL queries for efficient operations
            self._install_queries(workspace_label)

        await asyncio.to_thread(_create_schema)

    def _install_queries(self, workspace_label: str):
        """Install GSQL queries for efficient graph operations."""
        try:
            # Query to get popular labels by degree
            # This query counts edges per vertex and returns sorted by degree
            popular_labels_query = f"""
            CREATE QUERY get_popular_labels_{workspace_label}(INT limit) FOR GRAPH {self._graph_name} {{
                MapAccum<STRING, INT> @@degree_map;
                HeapAccum<Tuple2<INT, STRING>>(limit, f0 DESC, f1 ASC) @@top_labels;

                # Initialize all vertices with degree 0
                Start = {{{workspace_label}}};
                Start = SELECT v FROM Start:v
                    WHERE v.entity_id != ""
                    ACCUM @@degree_map += (v.entity_id -> 0);

                # Count edges (both directions for undirected graph)
                Start = SELECT v FROM Start:v - (DIRECTED:e) - {workspace_label}:t
                    WHERE v.entity_id != "" AND t.entity_id != ""
                    ACCUM @@degree_map += (v.entity_id -> 1);

                # Build heap with degree and label, sorted by degree DESC, label ASC
                Start = SELECT v FROM Start:v
                    WHERE v.entity_id != ""
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
            search_labels_query = f"""
            CREATE QUERY search_labels_{workspace_label}(STRING search_query, INT limit) FOR GRAPH {self._graph_name} {{
                ListAccum<STRING> @@matches;
                STRING query_lower = lower(search_query);

                Start = {{{workspace_label}}};
                Start = SELECT v FROM Start:v
                    WHERE v.entity_id != "" AND str_contains(lower(v.entity_id), query_lower)
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
                result = self._conn.getVertices(
                    workspace_label, where=f'entity_id=="{node_id}"', limit=1
                )
                return len(result) > 0
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking node existence for {node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_check_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""
        workspace_label = self._get_workspace_label()

        def _check_edge():
            try:
                # Check both directions for undirected graph
                result1 = self._conn.getEdges(
                    workspace_label,
                    source_node_id,
                    "DIRECTED",
                    workspace_label,
                    target_node_id,
                    limit=1,
                )
                result2 = self._conn.getEdges(
                    workspace_label,
                    target_node_id,
                    "DIRECTED",
                    workspace_label,
                    source_node_id,
                    limit=1,
                )
                return len(result1) > 0 or len(result2) > 0
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_check_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its entity_id, return only node properties."""
        workspace_label = self._get_workspace_label()

        def _get_node():
            try:
                result = self._conn.getVertices(
                    workspace_label, where=f'entity_id=="{node_id}"', limit=2
                )
                if len(result) > 1:
                    logger.warning(
                        f"[{self.workspace}] Multiple nodes found with entity_id '{node_id}'. Using first node."
                    )
                if result:
                    node_data = result[0]["attributes"]
                    # Remove entity_id from attributes if it's duplicated (it's the primary key)
                    if "entity_id" in node_data:
                        # Keep entity_id in the dict
                        pass
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
                        result = self._conn.getVertices(
                            workspace_label,
                            where=f'entity_id=="{node_id}"',
                            limit=1,
                        )
                        if result:
                            node_data = result[0]["attributes"]
                            nodes[node_id] = node_data
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
                # Get edges from this node (both directions for undirected graph)
                result1 = self._conn.getEdges(
                    workspace_label,
                    node_id,
                    "DIRECTED",
                    workspace_label,
                    "*",
                    limit=10000,
                )
                result2 = self._conn.getEdges(
                    workspace_label,
                    "*",
                    "DIRECTED",
                    workspace_label,
                    node_id,
                    limit=10000,
                )
                # Count unique edges (avoid double counting)
                edge_ids = set()
                for edge in result1:
                    edge_id = edge.get("to_id", "")
                    edge_ids.add((node_id, edge_id))
                for edge in result2:
                    edge_id = edge.get("from_id", "")
                    edge_ids.add((edge_id, node_id))
                return len(edge_ids)
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node degree for {node_id}: {str(e)}"
                )
                raise

        return await asyncio.to_thread(_get_degree)

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
        workspace_label = self._get_workspace_label()

        def _get_edge():
            try:
                # Check both directions for undirected graph
                result1 = self._conn.getEdges(
                    workspace_label,
                    source_node_id,
                    "DIRECTED",
                    workspace_label,
                    target_node_id,
                    limit=2,
                )
                result2 = self._conn.getEdges(
                    workspace_label,
                    target_node_id,
                    "DIRECTED",
                    workspace_label,
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
                }
        return edges_dict

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node."""
        workspace_label = self._get_workspace_label()

        def _get_node_edges():
            try:
                # Get edges from this node (both directions for undirected graph)
                result1 = self._conn.getEdges(
                    workspace_label,
                    source_node_id,
                    "DIRECTED",
                    workspace_label,
                    "*",
                    limit=10000,
                )
                result2 = self._conn.getEdges(
                    workspace_label,
                    "*",
                    "DIRECTED",
                    workspace_label,
                    source_node_id,
                    limit=10000,
                )

                edges = []
                edge_pairs = set()  # To avoid duplicates

                # Process outgoing edges
                for edge in result1:
                    target_id = edge.get("to_id")
                    if target_id:
                        pair = tuple(sorted([source_node_id, target_id]))
                        if pair not in edge_pairs:
                            edges.append((source_node_id, target_id))
                            edge_pairs.add(pair)

                # Process incoming edges
                for edge in result2:
                    source_id = edge.get("from_id")
                    if source_id:
                        pair = tuple(sorted([source_node_id, source_id]))
                        if pair not in edge_pairs:
                            edges.append((source_id, source_node_id))
                            edge_pairs.add(pair)

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
                # Ensure entity_id is in node_data
                if "entity_id" not in node_data:
                    node_data["entity_id"] = node_id

                # Upsert vertex using upsertVertex
                self._conn.upsertVertex(workspace_label, node_id, node_data)
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
        workspace_label = self._get_workspace_label()

        def _upsert_edge():
            try:
                # Ensure both nodes exist first
                # Check if source node exists
                source_exists = self._conn.getVertices(
                    workspace_label, where=f'entity_id=="{source_node_id}"', limit=1
                )
                if not source_exists:
                    # Create source node with minimal data
                    self._conn.upsertVertex(
                        workspace_label, source_node_id, {"entity_id": source_node_id}
                    )

                # Check if target node exists
                target_exists = self._conn.getVertices(
                    workspace_label, where=f'entity_id=="{target_node_id}"', limit=1
                )
                if not target_exists:
                    # Create target node with minimal data
                    self._conn.upsertVertex(
                        workspace_label, target_node_id, {"entity_id": target_node_id}
                    )

                # Upsert edge (undirected, so direction doesn't matter)
                self._conn.upsertEdge(
                    workspace_label,
                    source_node_id,
                    "DIRECTED",
                    workspace_label,
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
                    # Get all nodes sorted by degree
                    all_vertices = self._conn.getVertices(
                        workspace_label, limit=max_nodes
                    )
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
                                workspace_label,
                                node_id,
                                "DIRECTED",
                                workspace_label,
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
                                        type="DIRECTED",
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

                        # Get node
                        try:
                            vertices = self._conn.getVertices(
                                workspace_label,
                                where=f'entity_id=="{current_id}"',
                                limit=1,
                            )
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
                                    workspace_label,
                                    current_id,
                                    "DIRECTED",
                                    workspace_label,
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
                                                    type="DIRECTED",
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
                vertices = self._conn.getVertices(workspace_label, limit=100000)
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
                self._conn.delVertices(workspace_label, where=f'entity_id=="{node_id}"')
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
        workspace_label = self._get_workspace_label()

        def _delete_edge(source, target):
            try:
                # Delete edge in both directions
                self._conn.delEdges(
                    workspace_label,
                    source,
                    "DIRECTED",
                    workspace_label,
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
                vertices = self._conn.getVertices(workspace_label, limit=100000)
                nodes = []
                for vertex in vertices:
                    attrs = vertex.get("attributes", {})
                    attrs["id"] = attrs.get("entity_id")
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
                # Get all vertices first
                vertices = self._conn.getVertices(workspace_label, limit=100000)
                edges = []
                processed_edges = set()

                for vertex in vertices:
                    source_id = vertex.get("attributes", {}).get("entity_id")
                    if not source_id:
                        continue

                    try:
                        vertex_edges = self._conn.getEdges(
                            workspace_label,
                            source_id,
                            "DIRECTED",
                            workspace_label,
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
                # Get all vertices and calculate degrees
                vertices = self._conn.getVertices(workspace_label, limit=100000)
                node_degrees = {}

                for vertex in vertices:
                    entity_id = vertex.get("attributes", {}).get("entity_id")
                    if not entity_id:
                        continue

                    # Calculate degree
                    try:
                        edges = self._conn.getEdges(
                            workspace_label,
                            entity_id,
                            "DIRECTED",
                            workspace_label,
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
                # Get all vertices and filter
                vertices = self._conn.getVertices(workspace_label, limit=100000)
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
                    # Delete all vertices with this workspace label
                    self._conn.delVertices(workspace_label, where="")

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
