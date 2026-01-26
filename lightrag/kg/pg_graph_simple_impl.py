"""
PGGraphStorageSimple - Simple PostgreSQL Graph Storage without Apache AGE.

This module provides a lightweight graph storage implementation using standard
PostgreSQL tables (JSONB) instead of Apache AGE. It's designed for environments
where AGE is not available (e.g., Supabase, standard PostgreSQL) and enables
stateless deployments for autoscaling.

Tables:
- LIGHTRAG_GRAPH_NODES: Stores graph nodes with JSONB properties
- LIGHTRAG_GRAPH_EDGES: Stores graph edges with JSONB properties
"""

import json
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import BaseGraphStorage
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import logger

from .postgres_impl import PostgreSQLDB, ClientManager, get_data_init_lock


# Table DDL for graph storage
GRAPH_SIMPLE_TABLES = {
    "LIGHTRAG_GRAPH_NODES": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_GRAPH_NODES (
            workspace VARCHAR(255) NOT NULL,
            node_id VARCHAR(512) NOT NULL,
            properties JSONB NOT NULL DEFAULT '{}',
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT LIGHTRAG_GRAPH_NODES_PK PRIMARY KEY (workspace, node_id)
        )""",
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_graph_nodes_workspace ON LIGHTRAG_GRAPH_NODES(workspace)",
        ],
    },
    "LIGHTRAG_GRAPH_EDGES": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_GRAPH_EDGES (
            workspace VARCHAR(255) NOT NULL,
            source_id VARCHAR(512) NOT NULL,
            target_id VARCHAR(512) NOT NULL,
            properties JSONB NOT NULL DEFAULT '{}',
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT LIGHTRAG_GRAPH_EDGES_PK PRIMARY KEY (workspace, source_id, target_id)
        )""",
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_workspace ON LIGHTRAG_GRAPH_EDGES(workspace)",
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON LIGHTRAG_GRAPH_EDGES(workspace, source_id)",
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON LIGHTRAG_GRAPH_EDGES(workspace, target_id)",
        ],
    },
}


@final
@dataclass
class PGGraphStorageSimple(BaseGraphStorage):
    """
    Simple PostgreSQL graph storage without Apache AGE.

    Uses standard PostgreSQL tables with JSONB for storing graph nodes and edges.
    Suitable for environments where AGE extension is not available (Supabase, etc.).
    """

    def __post_init__(self):
        self.db: PostgreSQLDB | None = None
        self._node_count_cache: int | None = None

    async def initialize(self):
        """Initialize the storage and create tables if needed."""
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Set workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                self.workspace = self.db.workspace
            elif not hasattr(self, "workspace") or not self.workspace:
                self.workspace = "default"

            logger.info(
                f"[{self.workspace}] PGGraphStorageSimple initialized"
            )

            # Create tables and indexes
            async with self.db.pool.acquire() as conn:
                for table_name, table_def in GRAPH_SIMPLE_TABLES.items():
                    try:
                        await conn.execute(table_def["ddl"])
                        for index_sql in table_def.get("indexes", []):
                            try:
                                await conn.execute(index_sql)
                            except Exception as idx_err:
                                # Index might already exist
                                logger.debug(f"Index creation note: {idx_err}")
                    except Exception as e:
                        logger.error(f"Error creating table {table_name}: {e}")
                        raise

    async def finalize(self):
        """Release database connection."""
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def index_done_callback(self) -> None:
        """Called when indexing is complete. PostgreSQL handles persistence automatically."""
        # Invalidate node count cache
        self._node_count_cache = None

    # ========== Node Operations ==========

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        query = """
            SELECT 1 FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = $1 AND node_id = $2
            LIMIT 1
        """
        async with self.db.pool.acquire() as conn:
            result = await conn.fetchval(query, self.workspace, node_id)
            return result is not None

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node properties by ID."""
        query = """
            SELECT properties FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = $1 AND node_id = $2
        """
        async with self.db.pool.acquire() as conn:
            row = await conn.fetchrow(query, self.workspace, node_id)
            if row:
                props = row["properties"]
                if isinstance(props, str):
                    return json.loads(props)
                return dict(props) if props else {}
            return None

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node."""
        query = """
            SELECT COUNT(*) FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace = $1 AND (source_id = $2 OR target_id = $2)
        """
        async with self.db.pool.acquire() as conn:
            count = await conn.fetchval(query, self.workspace, node_id)
            return count or 0

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert or update a node."""
        # Use batch method for single node to maintain consistency
        await self.upsert_nodes_batch({node_id: node_data})

    async def upsert_nodes_batch(self, nodes: dict[str, dict[str, str]]) -> None:
        """Insert or update multiple nodes in a single transaction.

        This is much more efficient than individual upserts for bulk operations.

        Args:
            nodes: Dictionary mapping node_id to node properties
        """
        if not nodes:
            return

        query = """
            INSERT INTO LIGHTRAG_GRAPH_NODES (workspace, node_id, properties, updated_at)
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (workspace, node_id)
            DO UPDATE SET properties = $3, updated_at = CURRENT_TIMESTAMP
        """

        async with self.db.pool.acquire() as conn:
            # Use executemany for batch insert
            records = [
                (self.workspace, node_id, json.dumps(node_data, ensure_ascii=False))
                for node_id, node_data in nodes.items()
            ]
            await conn.executemany(query, records)

        # Invalidate cache
        self._node_count_cache = None
        logger.debug(f"[{self.workspace}] Batch upserted {len(nodes)} nodes")

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its connected edges."""
        async with self.db.pool.acquire() as conn:
            async with conn.transaction():
                # Delete edges first
                await conn.execute(
                    """
                    DELETE FROM LIGHTRAG_GRAPH_EDGES
                    WHERE workspace = $1 AND (source_id = $2 OR target_id = $2)
                    """,
                    self.workspace,
                    node_id,
                )
                # Delete node
                await conn.execute(
                    """
                    DELETE FROM LIGHTRAG_GRAPH_NODES
                    WHERE workspace = $1 AND node_id = $2
                    """,
                    self.workspace,
                    node_id,
                )
        self._node_count_cache = None
        logger.debug(f"[{self.workspace}] Deleted node: {node_id}")

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Delete multiple nodes and their edges."""
        if not nodes:
            return
        async with self.db.pool.acquire() as conn:
            async with conn.transaction():
                # Delete edges
                await conn.execute(
                    """
                    DELETE FROM LIGHTRAG_GRAPH_EDGES
                    WHERE workspace = $1 AND (source_id = ANY($2) OR target_id = ANY($2))
                    """,
                    self.workspace,
                    nodes,
                )
                # Delete nodes
                await conn.execute(
                    """
                    DELETE FROM LIGHTRAG_GRAPH_NODES
                    WHERE workspace = $1 AND node_id = ANY($2)
                    """,
                    self.workspace,
                    nodes,
                )
        self._node_count_cache = None
        logger.debug(f"[{self.workspace}] Deleted {len(nodes)} nodes")

    # ========== Edge Operations ==========

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists (undirected)."""
        query = """
            SELECT 1 FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace = $1
              AND ((source_id = $2 AND target_id = $3)
                   OR (source_id = $3 AND target_id = $2))
            LIMIT 1
        """
        async with self.db.pool.acquire() as conn:
            result = await conn.fetchval(
                query, self.workspace, source_node_id, target_node_id
            )
            return result is not None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties (undirected)."""
        query = """
            SELECT properties FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace = $1
              AND ((source_id = $2 AND target_id = $3)
                   OR (source_id = $3 AND target_id = $2))
            LIMIT 1
        """
        async with self.db.pool.acquire() as conn:
            row = await conn.fetchrow(
                query, self.workspace, source_node_id, target_node_id
            )
            if row:
                props = row["properties"]
                if isinstance(props, str):
                    return json.loads(props)
                return dict(props) if props else {}
            return None

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get total degree of an edge (sum of source and target node degrees)."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node."""
        if not await self.has_node(source_node_id):
            return None

        query = """
            SELECT source_id, target_id FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace = $1 AND (source_id = $2 OR target_id = $2)
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace, source_node_id)
            return [(row["source_id"], row["target_id"]) for row in rows]

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert or update an edge."""
        # Use batch method for single edge to maintain consistency
        await self.upsert_edges_batch([(source_node_id, target_node_id, edge_data)])

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Insert or update multiple edges in a single transaction.

        This is much more efficient than individual upserts for bulk operations.

        Args:
            edges: List of (source_id, target_id, edge_data) tuples
        """
        if not edges:
            return

        query = """
            INSERT INTO LIGHTRAG_GRAPH_EDGES (workspace, source_id, target_id, properties, updated_at)
            VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            ON CONFLICT (workspace, source_id, target_id)
            DO UPDATE SET properties = $4, updated_at = CURRENT_TIMESTAMP
        """

        async with self.db.pool.acquire() as conn:
            # Normalize edge direction and prepare records
            records = []
            for source_id, target_id, edge_data in edges:
                # Normalize direction (alphabetically) for consistency
                if source_id > target_id:
                    source_id, target_id = target_id, source_id
                records.append(
                    (self.workspace, source_id, target_id, json.dumps(edge_data, ensure_ascii=False))
                )
            await conn.executemany(query, records)

        logger.debug(f"[{self.workspace}] Batch upserted {len(edges)} edges")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Delete multiple edges."""
        if not edges:
            return

        async with self.db.pool.acquire() as conn:
            for source_id, target_id in edges:
                # Try both directions since graph is undirected
                await conn.execute(
                    """
                    DELETE FROM LIGHTRAG_GRAPH_EDGES
                    WHERE workspace = $1
                      AND ((source_id = $2 AND target_id = $3)
                           OR (source_id = $3 AND target_id = $2))
                    """,
                    self.workspace,
                    source_id,
                    target_id,
                )
        logger.debug(f"[{self.workspace}] Deleted {len(edges)} edges")

    # ========== Batch Operations ==========

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get multiple nodes in one query."""
        if not node_ids:
            return {}

        query = """
            SELECT node_id, properties FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = $1 AND node_id = ANY($2)
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace, node_ids)
            result = {}
            for row in rows:
                props = row["properties"]
                if isinstance(props, str):
                    result[row["node_id"]] = json.loads(props)
                else:
                    result[row["node_id"]] = dict(props) if props else {}
            return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Get degrees for multiple nodes."""
        if not node_ids:
            return {}

        query = """
            SELECT node_id, COUNT(*) as degree
            FROM (
                SELECT source_id as node_id FROM LIGHTRAG_GRAPH_EDGES
                WHERE workspace = $1 AND source_id = ANY($2)
                UNION ALL
                SELECT target_id as node_id FROM LIGHTRAG_GRAPH_EDGES
                WHERE workspace = $1 AND target_id = ANY($2)
            ) edges
            GROUP BY node_id
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace, node_ids)
            result = {nid: 0 for nid in node_ids}
            for row in rows:
                result[row["node_id"]] = row["degree"]
            return result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Get multiple edges in one query."""
        if not pairs:
            return {}

        result = {}
        async with self.db.pool.acquire() as conn:
            for pair in pairs:
                src_id = pair["src"]
                tgt_id = pair["tgt"]
                edge = await self.get_edge(src_id, tgt_id)
                if edge is not None:
                    result[(src_id, tgt_id)] = edge
        return result

    # ========== Query Operations ==========

    async def get_all_labels(self) -> list[str]:
        """Get all unique node labels (entity_id field)."""
        query = """
            SELECT DISTINCT node_id FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = $1
            ORDER BY node_id
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace)
            return [row["node_id"] for row in rows]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get labels sorted by node degree (most connected first)."""
        query = """
            WITH node_degrees AS (
                SELECT node_id, COUNT(*) as degree
                FROM (
                    SELECT source_id as node_id FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = $1
                    UNION ALL
                    SELECT target_id as node_id FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = $1
                ) edges
                GROUP BY node_id
            )
            SELECT n.node_id
            FROM LIGHTRAG_GRAPH_NODES n
            LEFT JOIN node_degrees d ON n.node_id = d.node_id
            WHERE n.workspace = $1
            ORDER BY COALESCE(d.degree, 0) DESC
            LIMIT $2
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace, limit)
            return [row["node_id"] for row in rows]

    async def search_labels(self, query_str: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching."""
        query = """
            SELECT node_id FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = $1 AND node_id ILIKE $2
            ORDER BY node_id
            LIMIT $3
        """
        pattern = f"%{query_str}%"
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace, pattern, limit)
            return [row["node_id"] for row in rows]

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph."""
        query = """
            SELECT node_id, properties FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = $1
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace)
            result = []
            for row in rows:
                props = row["properties"]
                if isinstance(props, str):
                    node_data = json.loads(props)
                else:
                    node_data = dict(props) if props else {}
                node_data["id"] = row["node_id"]
                result.append(node_data)
            return result

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph."""
        query = """
            SELECT source_id, target_id, properties FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace = $1
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(query, self.workspace)
            result = []
            for row in rows:
                props = row["properties"]
                if isinstance(props, str):
                    edge_data = json.loads(props)
                else:
                    edge_data = dict(props) if props else {}
                edge_data["source"] = row["source_id"]
                edge_data["target"] = row["target_id"]
                result.append(edge_data)
            return result

    async def get_node_count(self) -> int:
        """Get total count of nodes (O(1) with cache)."""
        if self._node_count_cache is not None:
            return self._node_count_cache

        query = """
            SELECT COUNT(*) FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = $1
        """
        async with self.db.pool.acquire() as conn:
            count = await conn.fetchval(query, self.workspace)
            self._node_count_cache = count or 0
            return self._node_count_cache

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph starting from nodes matching the label.

        This method first attempts to use the optimized stored procedure
        `lightrag_get_knowledge_graph` which performs BFS server-side in a single
        round-trip. If the stored procedure is not available, it falls back to
        the Python-based BFS implementation.

        Args:
            node_label: Label to match nodes (use "*" for all nodes)
            max_depth: Maximum BFS depth (default: 3)
            max_nodes: Maximum nodes to return (default: 1000)

        Returns:
            KnowledgeGraph with nodes, edges, and is_truncated flag
        """
        # Try stored procedure first (single round-trip, much faster)
        try:
            return await self._get_knowledge_graph_stored_proc(
                node_label, max_depth, max_nodes
            )
        except Exception as e:
            error_msg = str(e)
            if "lightrag_get_knowledge_graph" in error_msg and "does not exist" in error_msg:
                logger.debug(
                    f"[{self.workspace}] Stored procedure not available, using BFS fallback"
                )
            else:
                logger.warning(
                    f"[{self.workspace}] Stored procedure failed ({e}), using BFS fallback"
                )

        # Fallback to Python BFS implementation
        return await self._get_knowledge_graph_bfs(node_label, max_depth, max_nodes)

    async def _get_knowledge_graph_stored_proc(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """Retrieve knowledge graph using the optimized stored procedure.

        The stored procedure performs BFS server-side, eliminating N+1 queries.
        """
        async with self.db.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT lightrag_get_knowledge_graph($1, $2, $3, $4)",
                self.workspace,
                node_label,
                max_depth,
                max_nodes,
            )

            if result is None:
                return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

            # Parse JSON result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = dict(result)

            # Convert to Pydantic models with robust field mapping
            nodes = []
            for n in data.get("nodes", []):
                try:
                    nodes.append(KnowledgeGraphNode(
                        id=n.get("id", n.get("node_id", "")),
                        labels=n.get("labels", [n.get("entity_type", "entity")]),
                        properties=n.get("properties", {}),
                    ))
                except Exception as node_err:
                    logger.warning(f"[{self.workspace}] Skipping invalid node: {node_err}")

            edges = []
            for e in data.get("edges", []):
                try:
                    source = e.get("source", e.get("source_id", ""))
                    target = e.get("target", e.get("target_id", ""))
                    edges.append(KnowledgeGraphEdge(
                        id=e.get("id", f"{source}-{target}"),
                        type=e.get("type", e.get("relationship", "related_to")),
                        source=source,
                        target=target,
                        properties=e.get("properties", {}),
                    ))
                except Exception as edge_err:
                    logger.warning(f"[{self.workspace}] Skipping invalid edge: {edge_err}")

            logger.debug(
                f"[{self.workspace}] Stored proc returned {len(nodes)} nodes, "
                f"{len(edges)} edges (truncated: {data.get('is_truncated', False)})"
            )

            return KnowledgeGraph(
                nodes=nodes,
                edges=edges,
                is_truncated=data.get("is_truncated", False),
            )

    async def _get_knowledge_graph_bfs(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """Retrieve knowledge graph using Python-based BFS (fallback method).

        This is slower due to multiple round-trips but works without the stored procedure.
        """
        nodes_dict: dict[str, KnowledgeGraphNode] = {}
        edges_dict: dict[str, KnowledgeGraphEdge] = {}  # Use dict to deduplicate edges
        is_truncated = False

        async with self.db.pool.acquire() as conn:
            # Find starting nodes
            if node_label == "*":
                start_query = """
                    SELECT node_id, properties FROM LIGHTRAG_GRAPH_NODES
                    WHERE workspace = $1
                    LIMIT $2
                """
                start_rows = await conn.fetch(start_query, self.workspace, max_nodes)
            else:
                start_query = """
                    SELECT node_id, properties FROM LIGHTRAG_GRAPH_NODES
                    WHERE workspace = $1 AND node_id ILIKE $2
                    LIMIT $3
                """
                pattern = f"%{node_label}%"
                start_rows = await conn.fetch(
                    start_query, self.workspace, pattern, max_nodes
                )

            # Add starting nodes
            visited = set()
            frontier = []
            for row in start_rows:
                node_id = row["node_id"]
                if node_id not in visited:
                    visited.add(node_id)
                    frontier.append(node_id)
                    props = row["properties"]
                    if isinstance(props, str):
                        props = json.loads(props)
                    # Extract entity_type for labels, default to "entity"
                    entity_type = (props or {}).get("entity_type", "entity")
                    nodes_dict[node_id] = KnowledgeGraphNode(
                        id=node_id,
                        labels=[entity_type] if entity_type else ["entity"],
                        properties=props or {},
                    )

            # BFS traversal
            for depth in range(max_depth):
                if not frontier or len(nodes_dict) >= max_nodes:
                    if len(nodes_dict) >= max_nodes:
                        is_truncated = True
                    break

                next_frontier = []
                for node_id in frontier:
                    if len(nodes_dict) >= max_nodes:
                        is_truncated = True
                        break

                    # Get edges for current node
                    edges_query = """
                        SELECT source_id, target_id, properties FROM LIGHTRAG_GRAPH_EDGES
                        WHERE workspace = $1 AND (source_id = $2 OR target_id = $2)
                    """
                    edge_rows = await conn.fetch(edges_query, self.workspace, node_id)

                    for edge_row in edge_rows:
                        source_id = edge_row["source_id"]
                        target_id = edge_row["target_id"]
                        edge_props = edge_row["properties"]
                        if isinstance(edge_props, str):
                            edge_props = json.loads(edge_props)

                        # Add edge (use normalized key for deduplication)
                        edge_type = (edge_props or {}).get("relationship", "related_to")
                        # Normalize edge key (alphabetically) for deduplication
                        edge_key = f"{min(source_id, target_id)}-{max(source_id, target_id)}"
                        if edge_key not in edges_dict:
                            edges_dict[edge_key] = KnowledgeGraphEdge(
                                id=f"{source_id}-{target_id}",
                                type=edge_type,
                                source=source_id,
                                target=target_id,
                                properties=edge_props or {},
                            )

                        # Add neighbor node if not visited
                        neighbor_id = target_id if source_id == node_id else source_id
                        if neighbor_id not in visited:
                            if len(nodes_dict) >= max_nodes:
                                is_truncated = True
                                break
                            visited.add(neighbor_id)
                            next_frontier.append(neighbor_id)

                            # Fetch neighbor properties
                            neighbor_query = """
                                SELECT properties FROM LIGHTRAG_GRAPH_NODES
                                WHERE workspace = $1 AND node_id = $2
                            """
                            neighbor_row = await conn.fetchrow(
                                neighbor_query, self.workspace, neighbor_id
                            )
                            if neighbor_row:
                                n_props = neighbor_row["properties"]
                                if isinstance(n_props, str):
                                    n_props = json.loads(n_props)
                                # Extract entity_type for labels, default to "entity"
                                n_entity_type = (n_props or {}).get("entity_type", "entity")
                                nodes_dict[neighbor_id] = KnowledgeGraphNode(
                                    id=neighbor_id,
                                    labels=[n_entity_type] if n_entity_type else ["entity"],
                                    properties=n_props or {},
                                )

                frontier = next_frontier

        logger.debug(
            f"[{self.workspace}] BFS fallback returned {len(nodes_dict)} nodes, "
            f"{len(edges_dict)} edges (truncated: {is_truncated})"
        )

        return KnowledgeGraph(
            nodes=list(nodes_dict.values()),
            edges=list(edges_dict.values()),
            is_truncated=is_truncated,
        )

    async def drop(self) -> dict[str, str]:
        """Drop all graph data for this workspace."""
        try:
            async with self.db.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "DELETE FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = $1",
                        self.workspace,
                    )
                    await conn.execute(
                        "DELETE FROM LIGHTRAG_GRAPH_NODES WHERE workspace = $1",
                        self.workspace,
                    )
            self._node_count_cache = None
            return {
                "status": "success",
                "message": f"Workspace '{self.workspace}' graph data dropped",
            }
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping graph: {e}")
            return {"status": "error", "message": str(e)}

    # ========== Entity Consolidation (Stored Procedure) ==========

    async def consolidate_entity(
        self, old_name: str, canonical_name: str
    ) -> dict[str, Any]:
        """
        Consolidate two entities using the database stored procedure.

        This is much more efficient than doing multiple round-trips:
        - Merges descriptions from old_name into canonical_name
        - Redirects all edges from old_name to canonical_name
        - Deletes the old node

        Args:
            old_name: The entity name to be merged (will be deleted)
            canonical_name: The target canonical name (will be kept)

        Returns:
            dict with status and details:
            - {"status": "skipped", "reason": "old_node_not_found"}
            - {"status": "renamed", "old_name": ..., "new_name": ...}
            - {"status": "merged", "old_name": ..., "canonical_name": ...,
               "edges_redirected": N, "edges_deleted": N}
            - {"status": "error", "message": ...} if stored proc not available
        """
        try:
            async with self.db.pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT lightrag_consolidate_entity($1, $2, $3)",
                    self.workspace,
                    old_name,
                    canonical_name,
                )
                # Result is JSONB, asyncpg returns it as dict or str
                if isinstance(result, str):
                    return json.loads(result)
                return dict(result) if result else {"status": "error", "message": "No result"}
        except Exception as e:
            # Stored procedure might not exist - fall back gracefully
            error_msg = str(e)
            if "lightrag_consolidate_entity" in error_msg and "does not exist" in error_msg:
                logger.warning(
                    f"[{self.workspace}] Stored procedure not available, "
                    "consolidation will use fallback method"
                )
                return {"status": "error", "message": "stored_procedure_not_available"}
            logger.error(f"[{self.workspace}] Error consolidating entity: {e}")
            return {"status": "error", "message": error_msg}

    async def consolidate_entities_batch(
        self, consolidation_map: dict[str, str]
    ) -> dict[str, dict[str, Any]]:
        """
        Consolidate multiple entities using the batch stored procedure.

        This method uses a single database call to consolidate all entities,
        which is much more efficient than individual calls (~10-20ms total
        vs 100+ round-trips for large batches).

        Args:
            consolidation_map: Dict mapping old_name -> canonical_name

        Returns:
            Dict mapping old_name -> consolidation result
        """
        if not consolidation_map:
            return {}

        # Build JSONB array for batch procedure
        consolidations = [
            {"old": old_name, "canonical": canonical_name}
            for old_name, canonical_name in consolidation_map.items()
        ]

        try:
            async with self.db.pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT lightrag_consolidate_entities_batch($1, $2::jsonb)",
                    self.workspace,
                    json.dumps(consolidations),
                )

                # Parse result
                if isinstance(result, str):
                    batch_result = json.loads(result)
                else:
                    batch_result = dict(result) if result else {}

                # Check for batch-level error
                if batch_result.get("status") == "error":
                    error_msg = batch_result.get("message", "Unknown error")
                    if "lightrag_consolidate_entities_batch" in error_msg and "does not exist" in error_msg:
                        logger.warning(
                            f"[{self.workspace}] Batch stored procedure not available, "
                            "falling back to individual calls"
                        )
                        return await self._consolidate_entities_fallback(consolidation_map)
                    logger.error(f"[{self.workspace}] Batch consolidation error: {error_msg}")
                    return {old: {"status": "error", "message": error_msg} for old in consolidation_map}

                # Convert results array to dict keyed by old_name
                results = {}
                for item_result in batch_result.get("results", []):
                    old_name = item_result.get("old_name")
                    if old_name:
                        results[old_name] = item_result

                logger.info(
                    f"[{self.workspace}] Batch consolidated {batch_result.get('processed', 0)} entities "
                    f"in single DB call"
                )

                self._node_count_cache = None  # Invalidate cache
                return results

        except Exception as e:
            error_msg = str(e)
            if "lightrag_consolidate_entities_batch" in error_msg and "does not exist" in error_msg:
                logger.warning(
                    f"[{self.workspace}] Batch stored procedure not available, "
                    "falling back to individual calls"
                )
                return await self._consolidate_entities_fallback(consolidation_map)
            logger.error(f"[{self.workspace}] Batch consolidation error: {e}")
            return {old: {"status": "error", "message": error_msg} for old in consolidation_map}

    async def _consolidate_entities_fallback(
        self, consolidation_map: dict[str, str]
    ) -> dict[str, dict[str, Any]]:
        """
        Fallback method: consolidate entities one by one.

        Used when batch stored procedure is not available.
        """
        results = {}
        for old_name, canonical_name in consolidation_map.items():
            results[old_name] = await self.consolidate_entity(old_name, canonical_name)
        self._node_count_cache = None
        return results
