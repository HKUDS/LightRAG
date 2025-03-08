import os
from dataclasses import dataclass
from typing import Any, final
import numpy as np

from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import logger
from lightrag.base import BaseGraphStorage

import pipmaster as pm

if not pm.is_installed("networkx"):
    pm.install("networkx")

if not pm.is_installed("graspologic"):
    pm.install("graspologic")

import networkx as nx
from graspologic import embed
from .shared_storage import (
    get_storage_lock,
    get_update_flag,
    set_all_update_flags,
    is_multiprocess,
)

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

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

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

        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

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
            if (is_multiprocess and self.storage_updated.value) or (
                not is_multiprocess and self.storage_updated
            ):
                logger.info(
                    f"Process {os.getpid()} reloading graph {self.namespace} due to update by another process"
                )
                # Reload data
                self._graph = (
                    NetworkXStorage.load_nx_graph(self._graphml_xml_file) or nx.Graph()
                )
                # Reset update flag
                if is_multiprocess:
                    self.storage_updated.value = False
                else:
                    self.storage_updated = False

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
        graph = await self._get_graph()
        graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        graph = await self._get_graph()
        graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        graph = await self._get_graph()
        if graph.has_node(node_id):
            graph.remove_node(node_id)
            logger.debug(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # TODO: NOT USED
    async def _node2vec_embed(self):
        graph = await self._get_graph()
        embeddings, nodes = embed.node2vec_embed(
            graph,
            **self.global_config["node2vec_params"],
        )
        nodes_ids = [graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """
        graph = await self._get_graph()
        for node in nodes:
            if graph.has_node(node):
                graph.remove_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

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
        min_degree: int = 0,
        inclusive: bool = False,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.
        Maximum number of nodes is constrained by the environment variable `MAX_GRAPH_NODES` (default: 1000).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. min_degree does not affect nodes directly connected to the matching nodes
            2. Label matching nodes take precedence
            3. Followed by nodes directly connected to the matching nodes
            4. Finally, the degree of the nodes

        Args:
            node_label: Label of the starting node
            max_depth: Maximum depth of the subgraph
            min_degree: Minimum degree of nodes to include. Defaults to 0
            inclusive: Do an inclusive search if true

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        graph = await self._get_graph()

        # Initialize sets for start nodes and direct connected nodes
        start_nodes = set()
        direct_connected_nodes = set()

        # Handle special case for "*" label
        if node_label == "*":
            # For "*", return the entire graph including all nodes and edges
            subgraph = (
                graph.copy()
            )  # Create a copy to avoid modifying the original graph
        else:
            # Find nodes with matching node id based on search_mode
            nodes_to_explore = []
            for n, attr in graph.nodes(data=True):
                node_str = str(n)
                if not inclusive:
                    if node_label == node_str:  # Use exact matching
                        nodes_to_explore.append(n)
                else:  # inclusive mode
                    if node_label in node_str:  # Use partial matching
                        nodes_to_explore.append(n)

            if not nodes_to_explore:
                logger.warning(f"No nodes found with label {node_label}")
                return result

            # Get subgraph using ego_graph from all matching nodes
            combined_subgraph = nx.Graph()
            for start_node in nodes_to_explore:
                node_subgraph = nx.ego_graph(graph, start_node, radius=max_depth)
                combined_subgraph = nx.compose(combined_subgraph, node_subgraph)

            # Get start nodes and direct connected nodes
            if nodes_to_explore:
                start_nodes = set(nodes_to_explore)
                # Get nodes directly connected to all start nodes
                for start_node in start_nodes:
                    direct_connected_nodes.update(
                        combined_subgraph.neighbors(start_node)
                    )

                # Remove start nodes from directly connected nodes (avoid duplicates)
                direct_connected_nodes -= start_nodes

            subgraph = combined_subgraph

        # Filter nodes based on min_degree, but keep start nodes and direct connected nodes
        if min_degree > 0:
            nodes_to_keep = [
                node
                for node, degree in subgraph.degree()
                if node in start_nodes
                or node in direct_connected_nodes
                or degree >= min_degree
            ]
            subgraph = subgraph.subgraph(nodes_to_keep)

        # Check if number of nodes exceeds max_graph_nodes
        if len(subgraph.nodes()) > MAX_GRAPH_NODES:
            origin_nodes = len(subgraph.nodes())
            node_degrees = dict(subgraph.degree())

            def priority_key(node_item):
                node, degree = node_item
                # Priority order: start(2) > directly connected(1) > other nodes(0)
                if node in start_nodes:
                    priority = 2
                elif node in direct_connected_nodes:
                    priority = 1
                else:
                    priority = 0
                return (priority, degree)

            # Sort by priority and degree and select top MAX_GRAPH_NODES nodes
            top_nodes = sorted(node_degrees.items(), key=priority_key, reverse=True)[
                :MAX_GRAPH_NODES
            ]
            top_node_ids = [node[0] for node in top_nodes]
            # Create new subgraph and keep nodes only with most degree
            subgraph = subgraph.subgraph(top_node_ids)
            logger.info(
                f"Reduced graph from {origin_nodes} nodes to {MAX_GRAPH_NODES} nodes (depth={max_depth})"
            )

        # Add nodes to result
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
        # Check if storage was updated by another process
        if is_multiprocess and self.storage_updated.value:
            # Storage was updated by another process, reload data instead of saving
            logger.warning(
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
                if is_multiprocess:
                    self.storage_updated.value = False
                else:
                    self.storage_updated = False
                return True  # Return success
            except Exception as e:
                logger.error(f"Error saving graph for {self.namespace}: {e}")
                return False  # Return error

        return True
