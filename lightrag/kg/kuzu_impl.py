import os
from dataclasses import dataclass
from typing import final, cast
from collections import deque

from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

import kuzu


@final
@dataclass
class KuzuDBStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Check KUZU_WORKSPACE environment variable and override workspace if set
        kuzu_workspace = os.environ.get("KUZU_WORKSPACE")
        if kuzu_workspace and kuzu_workspace.strip():
            workspace = kuzu_workspace

        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._db = None
        self._conn = None

    def _get_workspace_label(self) -> str:
        """Get workspace label, return 'base' for compatibility when workspace is empty"""
        workspace = getattr(self, "workspace", None)
        return workspace if workspace else "base"

    @property
    def connection(self) -> kuzu.Connection:
        """Get connection with type safety guarantee"""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is not initialized. Call initialize() first."
            )
        return cast(kuzu.Connection, self._conn)

    def get_first(self, result) -> kuzu.QueryResult:
        """Normalize query result to handle both single QueryResult and list[QueryResult] cases"""
        if isinstance(result, list):
            if len(result) == 0:
                raise RuntimeError("Query returned empty result list")
            return result[0]  # Take the first result
        return result

    def get_all(self, result) -> list[kuzu.QueryResult]:
        """Get all query results, handling both single QueryResult and list[QueryResult] cases"""
        if isinstance(result, list):
            return result
        return [result]

    async def initialize(self):
        if self._conn is not None:
            # Already initialized
            return

        db_path = os.environ.get("KUZU_DB_PATH", f"kuzu_db_{self.namespace}")

        # Initialize the KuzuDB instance
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)

        # Create node and relationship tables if they don't exist
        workspace_label = self._get_workspace_label()

        try:
            # Create node table with flexible schema
            self._conn.execute(
                f"""
                CREATE NODE TABLE IF NOT EXISTS {workspace_label}(
                    entity_id STRING,
                    entity_type STRING,
                    description STRING,
                    keywords STRING,
                    source_id STRING,
                    PRIMARY KEY(entity_id)
                )
            """
            )

            # Create relationship table
            self._conn.execute(
                f"""
                CREATE REL TABLE IF NOT EXISTS DIRECTED(
                    FROM {workspace_label} TO {workspace_label},
                    relationship STRING,
                    weight DOUBLE,
                    description STRING,
                    keywords STRING,
                    source_id STRING
                )
            """
            )

            logger.info(f"KuzuDB initialized at {db_path}")

        except Exception as e:
            logger.error(f"Error initializing KuzuDB: {str(e)}")
            raise

    async def finalize(self):
        """Close the KuzuDB connection and release all resources"""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._db = None

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure connection is closed when context manager exits"""
        await self.finalize()

    async def index_done_callback(self) -> None:
        # KuzuDB handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        """Check if a node with the given entity_id exists in the database"""
        workspace_label = self._get_workspace_label()
        try:
            query = (
                f"MATCH (n:{workspace_label}) WHERE n.entity_id = $entity_id RETURN n"
            )
            result = self.get_first(
                self.connection.execute(query, {"entity_id": node_id})
            )
            return result.has_next()
        except Exception as e:
            logger.error(f"Error checking node existence for {node_id}: {str(e)}")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes"""
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
                MATCH (a:{workspace_label})-[r:DIRECTED]-(b:{workspace_label})
                WHERE a.entity_id = $source_id AND b.entity_id = $target_id
                RETURN r
            """
            result = self.get_first(
                self.connection.execute(
                    query, {"source_id": source_node_id, "target_id": target_node_id}
                )
            )
            return result.has_next()
        except Exception as e:
            logger.error(f"Error checking edge existence: {str(e)}")
            return False

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its entity_id identifier, return only node properties"""
        workspace_label = self._get_workspace_label()
        try:
            query = (
                f"MATCH (n:{workspace_label}) WHERE n.entity_id = $entity_id RETURN n.*"
            )
            result = self.get_first(
                self.connection.execute(query, {"entity_id": node_id})
            )

            if result.has_next():
                row = result.get_next()
                node_dict = {
                    "entity_id": row[0],
                    "entity_type": row[1],
                    "description": row[2],
                    "keywords": row[3],
                    "source_id": row[4],
                }
                return node_dict
            return None
        except Exception as e:
            logger.error(f"Error getting node for {node_id}: {str(e)}")
            return None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in one query"""
        # workspace_label = self._get_workspace_label()
        nodes = {}

        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node is not None:
                nodes[node_id] = node

        return nodes

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node"""
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
                MATCH (n:{workspace_label})
                WHERE n.entity_id = $entity_id
                OPTIONAL MATCH (n)-[r:DIRECTED]-()
                RETURN COUNT(r) AS degree
            """
            result = self.get_first(
                self.connection.execute(query, {"entity_id": node_id})
            )

            if result.has_next():
                row = result.get_next()
                # Since we create bidirectional edges, divide by 2 to get the actual degree
                degree = row[0] if row[0] is not None else 0
                return degree // 2
            return 0
        except Exception as e:
            logger.error(f"Error getting node degree for {node_id}: {str(e)}")
            return 0

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve the degree for multiple nodes"""
        degrees = {}
        for node_id in node_ids:
            degrees[node_id] = await self.node_degree(node_id)
        return degrees

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes"""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Calculate the combined degree for each edge"""
        unique_node_ids = set()
        for src, tgt in edge_pairs:
            unique_node_ids.add(src)
            unique_node_ids.add(tgt)

        degrees = await self.node_degrees_batch(list(unique_node_ids))

        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str | float | None] | None:
        """Get edge properties between two nodes"""
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
                MATCH (a:{workspace_label})-[r:DIRECTED]-(b:{workspace_label})
                WHERE a.entity_id = $source_id AND b.entity_id = $target_id
                RETURN r.*
            """
            result = self.get_first(
                self.connection.execute(
                    query, {"source_id": source_node_id, "target_id": target_node_id}
                )
            )

            if result.has_next():
                row = result.get_next()
                column_names = result.get_column_names()
                edge_dict = dict(zip(column_names, row))

                # Remove the 'r.' prefix from column names for cleaner keys
                clean_edge_dict = {}
                for key, value in edge_dict.items():
                    clean_key = key.replace("r.", "") if key.startswith("r.") else key
                    clean_edge_dict[clean_key] = value

                # Ensure required keys exist with defaults
                required_keys = {
                    "relationship": None,
                    "weight": 0.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }
                for key, default_value in required_keys.items():
                    if key not in clean_edge_dict:
                        clean_edge_dict[key] = default_value

                return clean_edge_dict
            return None
        except Exception as e:
            logger.error(
                f"Error getting edge between {source_node_id} and {target_node_id}: {str(e)}"
            )
            return None

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple (src, tgt) pairs"""
        edges_dict = {}
        for pair in pairs:
            src_id = pair["src"]
            tgt_id = pair["tgt"]
            edge = await self.get_edge(src_id, tgt_id)
            if edge is not None:
                edges_dict[(src_id, tgt_id)] = edge
            else:
                edges_dict[(src_id, tgt_id)] = {
                    "relationship": None,
                    "weight": 0.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }
        return edges_dict

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges for a particular node"""
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
                    MATCH (n:{workspace_label})-[r:DIRECTED]-(connected:{workspace_label})
                WHERE n.entity_id = $entity_id
                RETURN n.entity_id, connected.entity_id
            """
            result = self.get_all(
                self.connection.execute(query, {"entity_id": source_node_id})
            )

            edges = []
            seen_pairs = set()
            for result in result:
                if not result.has_next():
                    continue
                while result.has_next():
                    row = result.get_next()
                    # Create a normalized edge pair to avoid duplicates
                    node1, node2 = row[0], row[1]
                    # Sort the pair to ensure consistent ordering
                    edge_pair = tuple(sorted([node1, node2]))
                    if edge_pair not in seen_pairs:
                        edges.append((node1, node2))
                        seen_pairs.add(edge_pair)

            return edges if edges else None
        except Exception as e:
            logger.error(f"Error getting edges for node {source_node_id}: {str(e)}")
            return None

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes"""
        result = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges is not None else []
        return result

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all nodes that are associated with the given chunk_ids"""
        workspace_label = self._get_workspace_label()
        nodes = []
        seen_nodes = set()  # Track seen entity_ids to avoid duplicates

        try:
            for chunk_id in chunk_ids:
                query = f"""
                    MATCH (n:{workspace_label})
                    WHERE n.source_id CONTAINS $chunk_id
                    RETURN n.*
                """
                result = self.get_all(
                    self.connection.execute(query, {"chunk_id": chunk_id})
                )

                for result in result:
                    if not result.has_next():
                        continue
                    while result.has_next():
                        row = result.get_next()
                        column_names = result.get_column_names()
                        node_dict = dict(zip(column_names, row))

                        # Clean up column names by removing prefixes
                        clean_node_dict = {}
                        for key, value in node_dict.items():
                            clean_key = (
                                key.replace("n.", "") if key.startswith("n.") else key
                            )
                            clean_node_dict[clean_key] = value

                        clean_node_dict["id"] = clean_node_dict.get("entity_id")

                        # Only add the node if we haven't seen it before
                        entity_id = clean_node_dict.get("entity_id")
                        if entity_id and entity_id not in seen_nodes:
                            nodes.append(clean_node_dict)
                            seen_nodes.add(entity_id)
        except Exception as e:
            logger.error(f"Error getting nodes by chunk ids: {str(e)}")

        return nodes

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all edges that are associated with the given chunk_ids"""
        workspace_label = self._get_workspace_label()
        edges = []
        seen_edges = set()  # Track seen edge pairs to avoid duplicates

        try:
            for chunk_id in chunk_ids:
                query = f"""
                            MATCH (a:{workspace_label})-[r:DIRECTED]-(b:{workspace_label})
                            WHERE r.source_id CONTAINS $chunk_id
                            RETURN a.entity_id, b.entity_id, r.*
                """
                result = self.get_all(
                    self.connection.execute(query, {"chunk_id": chunk_id})
                )
                for result in result:
                    if not result.has_next():
                        continue
                    # Process each result
                    # Note: KuzuDB returns results in a QueryResult object
                    # We need to iterate through it to get the actual rows
                    while result.has_next():
                        row = result.get_next()
                        column_names = result.get_column_names()
                        edge_dict = dict(
                            zip(column_names[2:], row[2:])
                        )  # Skip source and target

                        # Clean up column names by removing prefixes
                        clean_edge_dict = {}
                        for key, value in edge_dict.items():
                            clean_key = (
                                key.replace("r.", "") if key.startswith("r.") else key
                            )
                            clean_edge_dict[clean_key] = value

                        source, target = row[0], row[1]
                        clean_edge_dict["source"] = source
                        clean_edge_dict["target"] = target

                        # Create normalized edge pair to avoid bidirectional duplicates
                        edge_pair = tuple(sorted([source, target]))
                        if edge_pair not in seen_edges:
                            edges.append(clean_edge_dict)
                            seen_edges.add(edge_pair)
        except Exception as e:
            logger.error(f"Error getting edges by chunk ids: {str(e)}")

        return edges

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the KuzuDB database"""
        workspace_label = self._get_workspace_label()

        if "entity_id" not in node_data:
            raise ValueError(
                "KuzuDB: node properties must contain an 'entity_id' field"
            )

        try:
            # Build the properties for the MERGE statement
            params = {"entity_id": node_id}
            set_props = []

            for key, value in node_data.items():
                if key == "entity_id":
                    # entity_id is used in MERGE condition, skip it
                    continue
                else:
                    param_name = f"prop_{key}"
                    params[param_name] = value
                    set_props.append(f"n.{key} = ${param_name}")

            set_clause = ", ".join(set_props) if set_props else ""

            if set_clause:
                query = f"""
                    MERGE (n:{workspace_label} {{entity_id: $entity_id}})
                    SET {set_clause}
                """
            else:
                query = f"""
                    MERGE (n:{workspace_label} {{entity_id: $entity_id}})
                """

            self.connection.execute(query, params)

        except Exception as e:
            logger.error(f"Error during node upsert: {str(e)}")
            raise

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Upsert an edge and its properties between two nodes"""
        workspace_label = self._get_workspace_label()

        try:
            # Build the properties dict
            params = {"source_id": source_node_id, "target_id": target_node_id}
            set_props = []

            for key, value in edge_data.items():
                param_name = f"prop_{key}"
                params[param_name] = value
                set_props.append(f"r.{key} = ${param_name}")

            set_clause = f"SET {', '.join(set_props)}" if set_props else ""

            # Create bidirectional edges to simulate undirected behavior
            # Edge 1: source -> target
            query1 = f"""
                MATCH (source:{workspace_label} {{entity_id: $source_id}})
                MATCH (target:{workspace_label} {{entity_id: $target_id}})
                MERGE (source)-[r:DIRECTED]->(target)
                {set_clause}
            """

            # Edge 2: target -> source
            query2 = f"""
                MATCH (source:{workspace_label} {{entity_id: $source_id}})
                MATCH (target:{workspace_label} {{entity_id: $target_id}})
                MERGE (target)-[r:DIRECTED]->(source)
                {set_clause}
            """

            self.connection.execute(query1, params)
            self.connection.execute(query2, params)

        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int | None = None
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph of nodes"""
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        workspace_label = self._get_workspace_label()
        result = KnowledgeGraph()

        if node_label == "*":
            # Get all nodes with highest degree
            try:
                query = f"""
                    MATCH (n:{workspace_label})
                    OPTIONAL MATCH (n)-[r:DIRECTED]-()
                    RETURN n.*, COUNT(r) AS degree
                    ORDER BY degree DESC
                    LIMIT {max_nodes}
                """

                node_result = self.get_all(self.connection.execute(query))
                seen_nodes = set()

                for node_query_result in node_result:
                    if not node_query_result.has_next():
                        continue
                    while node_query_result.has_next():
                        row = node_query_result.get_next()
                        column_names = node_query_result.get_column_names()
                        node_data = dict(
                            zip(column_names[:-1], row[:-1])
                        )  # Exclude degree

                        # Clean up column names by removing prefixes
                        clean_node_data = {}
                        for key, value in node_data.items():
                            clean_key = (
                                key.replace("n.", "") if key.startswith("n.") else key
                            )
                            clean_node_data[clean_key] = value

                        entity_id = clean_node_data.get("entity_id")
                        if entity_id and entity_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=entity_id,
                                    labels=[entity_id],
                                    properties=clean_node_data,
                                )
                            )
                            seen_nodes.add(entity_id)

                # Get edges between these nodes
                if seen_nodes:
                    entity_ids = list(seen_nodes)
                    edges_query = f"""
                        MATCH (a:{workspace_label})-[r:DIRECTED]-(b:{workspace_label})
                        WHERE a.entity_id IN $entity_ids AND b.entity_id IN $entity_ids
                        RETURN a.entity_id, b.entity_id, r.*
                    """

                    edge_result = self.get_all(
                        self.connection.execute(edges_query, {"entity_ids": entity_ids})
                    )
                    seen_edges = set()

                    for edge_query_result in edge_result:
                        if not edge_query_result.has_next():
                            continue
                        while edge_query_result.has_next():
                            row = edge_query_result.get_next()
                            column_names = edge_query_result.get_column_names()
                            edge_data = dict(zip(column_names[2:], row[2:]))

                            # Create normalized edge_id to avoid bidirectional duplicates
                            source, target = row[0], row[1]
                            edge_pair = tuple(sorted([source, target]))
                            edge_id = f"{edge_pair[0]}-{edge_pair[1]}"

                            if edge_id not in seen_edges:
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id,
                                        type="UNDIRECTED",
                                        source=edge_pair[0],
                                        target=edge_pair[1],
                                        properties=edge_data,
                                    )
                                )
                                seen_edges.add(edge_id)

            except Exception as e:
                logger.error(f"Error getting knowledge graph: {str(e)}")
        else:
            # BFS traversal for specific node
            # Ensure max_nodes is not None before passing to _bfs_subgraph
            assert max_nodes is not None
            return await self._bfs_subgraph(node_label, max_depth, max_nodes)

        return result

    async def _bfs_subgraph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """BFS implementation for subgraph traversal"""
        # workspace_label = self._get_workspace_label()
        result = KnowledgeGraph()
        visited_nodes = set()
        visited_edges = set()

        # Get starting node
        start_node = await self.get_node(node_label)
        if not start_node:
            return result

        # Initialize BFS
        queue = deque([(node_label, 0)])

        while queue and len(visited_nodes) < max_nodes:
            current_node_id, current_depth = queue.popleft()

            if current_node_id in visited_nodes or current_depth > max_depth:
                continue

            # Add current node
            node_data = await self.get_node(current_node_id)
            if node_data:
                result.nodes.append(
                    KnowledgeGraphNode(
                        id=current_node_id,
                        labels=[current_node_id],
                        properties=node_data,
                    )
                )
                visited_nodes.add(current_node_id)

                # Get neighbors if not at max depth
                if current_depth < max_depth:
                    edges = await self.get_node_edges(current_node_id)
                    if edges:
                        for source_id, target_id in edges:
                            edge_id = f"{source_id}-{target_id}"
                            if edge_id not in visited_edges:
                                edge_data = await self.get_edge(source_id, target_id)
                                if edge_data:
                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            id=edge_id,
                                            type="UNDIRECTED",
                                            source=source_id,
                                            target=target_id,
                                            properties=edge_data,
                                        )
                                    )
                                    visited_edges.add(edge_id)

                            # Add unvisited neighbors to queue
                            neighbor_id = (
                                target_id if source_id == current_node_id else source_id
                            )
                            if neighbor_id not in visited_nodes:
                                queue.append((neighbor_id, current_depth + 1))

        if len(visited_nodes) >= max_nodes:
            result.is_truncated = True

        return result

    async def get_all_labels(self) -> list[str]:
        """Get all existing node labels in the database"""
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
                MATCH (n:{workspace_label})
                WHERE n.entity_id IS NOT NULL
                RETURN DISTINCT n.entity_id
                ORDER BY n.entity_id
            """
            result = self.get_all(self.connection.execute(query))

            labels = []
            for query_result in result:
                if not query_result.has_next():
                    continue
                while query_result.has_next():
                    row = query_result.get_next()
                    labels.append(row[0])

            return labels
        except Exception as e:
            logger.error(f"Error getting all labels: {str(e)}")
            return []

    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified entity_id"""
        workspace_label = self._get_workspace_label()
        try:
            query = f"""
                MATCH (n:{workspace_label} {{entity_id: $entity_id}})
                DETACH DELETE n
            """
            self.connection.execute(query, {"entity_id": node_id})
            logger.debug(f"Deleted node with entity_id '{node_id}'")
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes"""
        for node_id in nodes:
            await self.delete_node(node_id)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges"""
        workspace_label = self._get_workspace_label()
        for source, target in edges:
            try:
                # Since we create bidirectional edges, we need to delete both directions
                # Delete source -> target
                query1 = f"""
                    MATCH (source:{workspace_label} {{entity_id: $source_id}})-[r:DIRECTED]->(target:{workspace_label} {{entity_id: $target_id}})
                    DELETE r
                """
                self.connection.execute(
                    query1, {"source_id": source, "target_id": target}
                )

                # Delete target -> source
                query2 = f"""
                    MATCH (target:{workspace_label} {{entity_id: $target_id}})-[r:DIRECTED]->(source:{workspace_label} {{entity_id: $source_id}})
                    DELETE r
                """
                self.connection.execute(
                    query2, {"source_id": source, "target_id": target}
                )

                logger.debug(
                    f"Deleted bidirectional edge between '{source}' and '{target}'"
                )
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def drop(self) -> dict[str, str]:
        """Drop all data from current workspace storage and clean up resources"""
        workspace_label = self._get_workspace_label()
        try:
            query = f"MATCH (n:{workspace_label}) DETACH DELETE n"
            self.connection.execute(query)

            logger.info(f"Dropped KuzuDB workspace '{workspace_label}'")
            return {
                "status": "success",
                "message": f"workspace '{workspace_label}' data dropped",
            }
        except Exception as e:
            logger.error(f"Error dropping KuzuDB workspace '{workspace_label}': {e}")
            return {"status": "error", "message": str(e)}
