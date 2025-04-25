import os
from dataclasses import dataclass
from typing import final

from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import logger
from lightrag.base import BaseGraphStorage

import pipmaster as pm

if not pm.is_installed("networkx"):
    pm.install("networkx")

if not pm.is_installed("graspologic"):
    pm.install("graspologic")

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

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))


@final
@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        self._storage_lock = None
        self.storage_updated = None
        self._graph = None

        # Load initial graph
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        else:
            logger.info("Created new empty graph")
        self._graph = preloaded_graph or nx.Graph()

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(self.namespace)
        # Get the storage lock for use in other methods
        self._storage_lock = get_storage_lock()

    async def _get_graph(self):
        """Check if the storage should be reloaded"""
        # Acquire lock to prevent concurrent read and write
        async with self._storage_lock:
            # Check if data needs to be reloaded
            if self.storage_updated.value:
                logger.info(
                    f"Process {os.getpid()} reloading graph {self.namespace} due to update by another process"
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
        return graph.degree(src_id) + graph.degree(tgt_id)

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
            logger.debug(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

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

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
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
                    f"Graph truncated: {len(sorted_nodes)} nodes found, limited to {max_nodes}"
                )

            limited_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
            # Create subgraph with the highest degree nodes
            subgraph = graph.subgraph(limited_nodes)
        else:
            # Check if node exists
            if node_label not in graph:
                logger.warning(f"Node {node_label} not found in the graph")
                return KnowledgeGraph()  # Return empty graph

            # Use modified BFS to get nodes, prioritizing high-degree nodes at the same depth
            bfs_nodes = []
            visited = set()
            # Store (node, depth, degree) in the queue
            queue = [(node_label, 0, graph.degree(node_label))]

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

                    # Check if we've reached max_nodes
                    if len(bfs_nodes) >= max_nodes:
                        break

            # Check if graph is truncated - if we still have nodes in the queue
            # and we've reached max_nodes, then the graph is truncated
            if queue and len(bfs_nodes) >= max_nodes:
                result.is_truncated = True
                logger.info(
                    f"Graph truncated: breadth-first search limited to {max_nodes} nodes"
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
            f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def index_done_callback(self) -> bool:
        """Save data to disk"""
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                # Storage was updated by another process, reload data instead of saving
                logger.info(
                    f"Graph for {self.namespace} was updated by another process, reloading..."
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
                NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                return True  # Return success
            except Exception as e:
                logger.error(f"Error saving graph for {self.namespace}: {e}")
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
                await set_all_update_flags(self.namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                logger.info(
                    f"Process {os.getpid()} drop graph {self.namespace} (file:{self._graphml_xml_file})"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping graph {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
