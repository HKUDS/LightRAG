import os
from dataclasses import dataclass
from typing import final

from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import logger
from lightrag.base import BaseGraphStorage
from lightrag.constants import GRAPH_FIELD_SEP
import networkx as nx
from .shared_storage import (
    get_storage_lock,
    get_update_flag,
    set_all_update_flags,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


@final
@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name, workspace="_"):
        logger.info(
            f"[{workspace}] Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        
        # Get composite workspace (supports multi-tenant isolation)
        composite_workspace = self._get_composite_workspace()
        
        if composite_workspace and composite_workspace != "_":
            # Include composite workspace in the file path for data isolation
            # For multi-tenant: tenant_id:kb_id:workspace
            # For single-tenant: just workspace
            workspace_dir = os.path.join(working_dir, composite_workspace)
            self.final_namespace = f"{composite_workspace}_{self.namespace}"
        else:
            # Default behavior when workspace is empty
            self.final_namespace = self.namespace
            workspace_dir = working_dir
            self.workspace = "_"

        os.makedirs(workspace_dir, exist_ok=True)
        self._graphml_xml_file = os.path.join(
            workspace_dir, f"graph_{self.namespace}.graphml"
        )
        self._storage_lock = None
        self.storage_updated = None
        self._graph = None

        # Load initial graph
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"[{self.workspace}] Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        else:
            logger.info(
                f"[{self.workspace}] Created new empty graph file: {self._graphml_xml_file}"
            )
        self._graph = preloaded_graph or nx.Graph()

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(self.final_namespace)
        # Get the storage lock for use in other methods
        self._storage_lock = get_storage_lock()

    async def _get_graph(self):
        """Check if the storage should be reloaded"""
        # Acquire lock to prevent concurrent read and write
        async with self._storage_lock:
            # Check if data needs to be reloaded
            if self.storage_updated.value:
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} reloading graph {self._graphml_xml_file} due to modifications by another process"
                )
                # Reload data
                self._graph = (
                    NetworkXStorage.load_nx_graph(self._graphml_xml_file) or nx.Graph()
                )
                # Reset update flag
                self.storage_updated.value = False

            return self._graph

    async def has_node(self, node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        graph = await self._get_graph()
        return graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        graph = await self._get_graph()
        return graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        graph = await self._get_graph()
        src_degree = graph.degree(src_id) if graph.has_node(src_id) else 0
        tgt_degree = graph.degree(tgt_id) if graph.has_node(tgt_id) else 0
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        graph = await self._get_graph()
        return graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        graph = await self._get_graph()
        if graph.has_node(source_node_id):
            return list(graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        graph = await self._get_graph()
        graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        graph = await self._get_graph()
        graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        graph = await self._get_graph()
        if graph.has_node(node_id):
            graph.remove_node(node_id)
            logger.debug(f"[{self.workspace}] Node {node_id} deleted from the graph")
        else:
            logger.warning(
                f"[{self.workspace}] Node {node_id} not found in the graph for deletion"
            )

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            nodes: List of node IDs to be deleted
        """
        graph = await self._get_graph()
        for node in nodes:
            if graph.has_node(node):
                graph.remove_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        graph = await self._get_graph()
        for source, target in edges:
            if graph.has_edge(source, target):
                graph.remove_edge(source, target)

    async def get_all_labels(self) -> list[str]:
        """
        Get all node labels in the graph
        Returns:
            [label1, label2, ...]  # Alphabetically sorted label list
        """
        graph = await self._get_graph()
        labels = set()
        for node in graph.nodes():
            labels.add(str(node))  # Add node id as a label

        # Return sorted list
        return sorted(list(labels))

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """
        Get popular labels by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """
        graph = await self._get_graph()

        # Get degrees of all nodes and sort by degree descending
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        # Return top labels limited by the specified limit
        popular_labels = [str(node) for node, _ in sorted_nodes[:limit]]

        logger.debug(
            f"[{self.workspace}] Retrieved {len(popular_labels)} popular labels (limit: {limit})"
        )

        return popular_labels

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """
        Search labels with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """
        graph = await self._get_graph()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        # Collect matching nodes with relevance scores
        matches = []
        for node in graph.nodes():
            node_str = str(node)
            node_lower = node_str.lower()

            # Skip if no match
            if query_lower not in node_lower:
                continue

            # Calculate relevance score
            # Exact match gets highest score
            if node_lower == query_lower:
                score = 1000
            # Prefix match gets high score
            elif node_lower.startswith(query_lower):
                score = 500
            # Contains match gets base score, with bonus for shorter strings
            else:
                # Shorter strings with matches are more relevant
                score = 100 - len(node_str)
                # Bonus for word boundary matches
                if f" {query_lower}" in node_lower or f"_{query_lower}" in node_lower:
                    score += 50

            matches.append((node_str, score))

        # Sort by relevance score (desc) then alphabetically
        matches.sort(key=lambda x: (-x[1], x[0]))

        # Return top matches limited by the specified limit
        search_results = [match[0] for match in matches[:limit]]

        logger.debug(
            f"[{self.workspace}] Search query '{query}' returned {len(search_results)} results (limit: {limit})"
        )

        return search_results

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        graph = await self._get_graph()

        result = KnowledgeGraph()

        # Handle special case for "*" label
        if node_label == "*":
            # Get degrees of all nodes
            degrees = dict(graph.degree())
            # Sort nodes by degree in descending order and take top max_nodes
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

            # Check if graph is truncated
            if len(sorted_nodes) > max_nodes:
                result.is_truncated = True
                logger.info(
                    f"[{self.workspace}] Graph truncated: {len(sorted_nodes)} nodes found, limited to {max_nodes}"
                )

            limited_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
            # Create subgraph with the highest degree nodes
            subgraph = graph.subgraph(limited_nodes)
        else:
            # Check if node exists
            if node_label not in graph:
                logger.warning(
                    f"[{self.workspace}] Node {node_label} not found in the graph"
                )
                return KnowledgeGraph()  # Return empty graph

            # Use modified BFS to get nodes, prioritizing high-degree nodes at the same depth
            bfs_nodes = []
            visited = set()
            # Store (node, depth, degree) in the queue
            queue = [(node_label, 0, graph.degree(node_label))]

            # Flag to track if there are unexplored neighbors due to depth limit
            has_unexplored_neighbors = False

            # Modified breadth-first search with degree-based prioritization
            while queue and len(bfs_nodes) < max_nodes:
                # Get the current depth from the first node in queue
                current_depth = queue[0][1]

                # Collect all nodes at the current depth
                current_level_nodes = []
                while queue and queue[0][1] == current_depth:
                    current_level_nodes.append(queue.pop(0))

                # Sort nodes at current depth by degree (highest first)
                current_level_nodes.sort(key=lambda x: x[2], reverse=True)

                # Process all nodes at current depth in order of degree
                for current_node, depth, degree in current_level_nodes:
                    if current_node not in visited:
                        visited.add(current_node)
                        bfs_nodes.append(current_node)

                        # Only explore neighbors if we haven't reached max_depth
                        if depth < max_depth:
                            # Add neighbor nodes to queue with incremented depth
                            neighbors = list(graph.neighbors(current_node))
                            # Filter out already visited neighbors
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            # Add neighbors to the queue with their degrees
                            for neighbor in unvisited_neighbors:
                                neighbor_degree = graph.degree(neighbor)
                                queue.append((neighbor, depth + 1, neighbor_degree))
                        else:
                            # Check if there are unexplored neighbors (skipped due to depth limit)
                            neighbors = list(graph.neighbors(current_node))
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            if unvisited_neighbors:
                                has_unexplored_neighbors = True

                    # Check if we've reached max_nodes
                    if len(bfs_nodes) >= max_nodes:
                        break

            # Check if graph is truncated - either due to max_nodes limit or depth limit
            if (queue and len(bfs_nodes) >= max_nodes) or has_unexplored_neighbors:
                if len(bfs_nodes) >= max_nodes:
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Graph truncated: max_nodes limit {max_nodes} reached"
                    )
                else:
                    logger.info(
                        f"[{self.workspace}] Graph truncated: found {len(bfs_nodes)} nodes within max_depth {max_depth}"
                    )

            # Create subgraph with BFS discovered nodes
            subgraph = graph.subgraph(bfs_nodes)

        # Add nodes to result
        seen_nodes = set()
        seen_edges = set()
        for node in subgraph.nodes():
            if str(node) in seen_nodes:
                continue

            node_data = dict(subgraph.nodes[node])
            # Get entity_type as labels
            labels = []
            if "entity_type" in node_data:
                if isinstance(node_data["entity_type"], list):
                    labels.extend(node_data["entity_type"])
                else:
                    labels.append(node_data["entity_type"])

            # Create node with properties
            node_properties = {k: v for k, v in node_data.items()}

            result.nodes.append(
                KnowledgeGraphNode(
                    id=str(node), labels=[str(node)], properties=node_properties
                )
            )
            seen_nodes.add(str(node))

        # Add edges to result
        for edge in subgraph.edges():
            source, target = edge
            # Esure unique edge_id for undirect graph
            if str(source) > str(target):
                source, target = target, source
            edge_id = f"{source}-{target}"
            if edge_id in seen_edges:
                continue

            edge_data = dict(subgraph.edges[edge])

            # Create edge with complete information
            result.edges.append(
                KnowledgeGraphEdge(
                    id=edge_id,
                    type="DIRECTED",
                    source=str(source),
                    target=str(target),
                    properties=edge_data,
                )
            )
            seen_edges.add(edge_id)

        logger.info(
            f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        chunk_ids_set = set(chunk_ids)
        graph = await self._get_graph()
        matching_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if "source_id" in node_data:
                node_source_ids = set(node_data["source_id"].split(GRAPH_FIELD_SEP))
                if not node_source_ids.isdisjoint(chunk_ids_set):
                    node_data_with_id = node_data.copy()
                    node_data_with_id["id"] = node_id
                    matching_nodes.append(node_data_with_id)
        return matching_nodes

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        chunk_ids_set = set(chunk_ids)
        graph = await self._get_graph()
        matching_edges = []
        for u, v, edge_data in graph.edges(data=True):
            if "source_id" in edge_data:
                edge_source_ids = set(edge_data["source_id"].split(GRAPH_FIELD_SEP))
                if not edge_source_ids.isdisjoint(chunk_ids_set):
                    edge_data_with_nodes = edge_data.copy()
                    edge_data_with_nodes["source"] = u
                    edge_data_with_nodes["target"] = v
                    matching_edges.append(edge_data_with_nodes)
        return matching_edges

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        graph = await self._get_graph()
        all_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_data_with_id = node_data.copy()
            node_data_with_id["id"] = node_id
            all_nodes.append(node_data_with_id)
        return all_nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        graph = await self._get_graph()
        all_edges = []
        for u, v, edge_data in graph.edges(data=True):
            edge_data_with_nodes = edge_data.copy()
            edge_data_with_nodes["source"] = u
            edge_data_with_nodes["target"] = v
            all_edges.append(edge_data_with_nodes)
        return all_edges

    async def index_done_callback(self) -> bool:
        """Save data to disk"""
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                # Storage was updated by another process, reload data instead of saving
                logger.info(
                    f"[{self.workspace}] Graph was updated by another process, reloading..."
                )
                self._graph = (
                    NetworkXStorage.load_nx_graph(self._graphml_xml_file) or nx.Graph()
                )
                # Reset update flag
                self.storage_updated.value = False
                return False  # Return error

        # Acquire lock and perform persistence
        async with self._storage_lock:
            try:
                # Save data to disk
                NetworkXStorage.write_nx_graph(
                    self._graph, self._graphml_xml_file, self.workspace
                )
                # Notify other processes that data has been updated
                await set_all_update_flags(self.final_namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                return True  # Return success
            except Exception as e:
                logger.error(f"[{self.workspace}] Error saving graph: {e}")
                return False  # Return error

        return True

    async def drop(self) -> dict[str, str]:
        """Drop all graph data from storage and clean up resources

        This method will:
        1. Remove the graph storage file if it exists
        2. Reset the graph to an empty state
        3. Update flags to notify other processes
        4. Changes is persisted to disk immediately

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                # delete _client_file_name
                if os.path.exists(self._graphml_xml_file):
                    os.remove(self._graphml_xml_file)
                self._graph = nx.Graph()
                # Notify other processes that data has been updated
                await set_all_update_flags(self.final_namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop graph file:{self._graphml_xml_file}"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping graph file:{self._graphml_xml_file}: {e}"
            )
            return {"status": "error", "message": str(e)}
