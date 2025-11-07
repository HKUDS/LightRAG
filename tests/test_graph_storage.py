#!/usr/bin/env python
"""
General-purpose graph storage test program.

This program selects the graph storage type to use based on the LIGHTRAG_GRAPH_STORAGE configuration in .env,
and tests its basic and advanced operations.

Supported graph storage types include:
- NetworkXStorage
- Neo4JStorage
- MongoDBStorage
- PGGraphStorage
- MemgraphStorage
- TigerGraphStorage
"""

import asyncio
import os
import sys
import importlib
import numpy as np
from dotenv import load_dotenv
from ascii_colors import ASCIIColors

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.types import KnowledgeGraph
from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.constants import GRAPH_FIELD_SEP


# Mock embedding function that returns random vectors
async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10)  # Return 10-dimensional random vectors


def check_env_file():
    """
    Check if the .env file exists and issue a warning if it does not.
    Returns True to continue execution, False to exit.
    """
    if not os.path.exists(".env"):
        warning_msg = "Warning: .env file not found in the current directory. This may affect storage configuration loading."
        ASCIIColors.yellow(warning_msg)

        # Check if running in an interactive terminal
        if sys.stdin.isatty():
            response = input("Do you want to continue? (yes/no): ")
            if response.lower() != "yes":
                ASCIIColors.red("Test program cancelled.")
                return False
    return True


async def initialize_graph_storage():
    """
    Initialize the corresponding graph storage instance based on environment variables.
    Returns the initialized storage instance.
    """
    # Get the graph storage type from environment variables
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")

    # Validate the storage type
    try:
        verify_storage_implementation("GRAPH_STORAGE", graph_storage_type)
    except ValueError as e:
        ASCIIColors.red(f"Error: {str(e)}")
        ASCIIColors.yellow(
            f"Supported graph storage types: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
        )
        return None

    # Check for required environment variables
    required_env_vars = STORAGE_ENV_REQUIREMENTS.get(graph_storage_type, [])
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_env_vars:
        ASCIIColors.red(
            f"Error: {graph_storage_type} requires the following environment variables, but they are not set: {', '.join(missing_env_vars)}"
        )
        return None

    # Dynamically import the corresponding module
    module_path = STORAGES.get(graph_storage_type)
    if not module_path:
        ASCIIColors.red(f"Error: Module path for {graph_storage_type} not found.")
        return None

    try:
        module = importlib.import_module(module_path, package="lightrag")
        storage_class = getattr(module, graph_storage_type)
    except (ImportError, AttributeError) as e:
        ASCIIColors.red(f"Error: Failed to import {graph_storage_type}: {str(e)}")
        return None

    # Initialize the storage instance
    global_config = {
        "embedding_batch_num": 10,  # Batch size
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5  # Cosine similarity threshold
        },
        "working_dir": os.environ.get(
            "WORKING_DIR", "./rag_storage"
        ),  # Working directory
    }

    # Initialize shared_storage for all storage types (required for locks)
    # All graph storage implementations use locks like get_data_init_lock() and get_graph_db_lock()
    initialize_share_data()  # Use single-process mode (workers=1)

    try:
        storage = storage_class(
            namespace="test_graph",
            workspace="test_workspace",
            global_config=global_config,
            embedding_func=mock_embedding_func,
        )

        # Initialize the connection
        await storage.initialize()
        return storage
    except Exception as e:
        ASCIIColors.red(f"Error: Failed to initialize {graph_storage_type}: {str(e)}")
        return None


async def test_graph_basic(storage):
    """
    Test basic graph database operations:
    1. Use upsert_node to insert two nodes.
    2. Use upsert_edge to insert an edge connecting the two nodes.
    3. Use get_node to read a node.
    4. Use get_edge to read an edge.
    """
    try:
        # 1. Insert the first node
        node1_id = "Artificial Intelligence"
        node1_data = {
            "entity_id": node1_id,
            "description": "Artificial intelligence is a branch of computer science that aims to understand the essence of intelligence and produce a new kind of intelligent machine that can react in a manner similar to human intelligence.",
            "keywords": "AI,Machine Learning,Deep Learning",
            "entity_type": "Technology Field",
        }
        print(f"Inserting node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. Insert the second node
        node2_id = "Machine Learning"
        node2_data = {
            "entity_id": node2_id,
            "description": "Machine learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
            "keywords": "Supervised Learning,Unsupervised Learning,Reinforcement Learning",
            "entity_type": "Technology Field",
        }
        print(f"Inserting node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. Insert the connecting edge
        edge_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of machine learning.",
        }
        print(f"Inserting edge: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge_data)

        # 4. Read node properties
        print(f"Reading node properties: {node1_id}")
        node1_props = await storage.get_node(node1_id)
        if node1_props:
            print(f"Successfully read node properties: {node1_id}")
            print(
                f"Node description: {node1_props.get('description', 'No description')}"
            )
            print(f"Node type: {node1_props.get('entity_type', 'No type')}")
            print(f"Node keywords: {node1_props.get('keywords', 'No keywords')}")
            # Verify that the returned properties are correct
            assert (
                node1_props.get("entity_id") == node1_id
            ), f"Node ID mismatch: expected {node1_id}, got {node1_props.get('entity_id')}"
            assert (
                node1_props.get("description") == node1_data["description"]
            ), "Node description mismatch"
            assert (
                node1_props.get("entity_type") == node1_data["entity_type"]
            ), "Node type mismatch"
        else:
            print(f"Failed to read node properties: {node1_id}")
            assert False, f"Failed to read node properties: {node1_id}"

        # 5. Read edge properties
        print(f"Reading edge properties: {node1_id} -> {node2_id}")
        edge_props = await storage.get_edge(node1_id, node2_id)
        if edge_props:
            print(f"Successfully read edge properties: {node1_id} -> {node2_id}")
            print(
                f"Edge relationship: {edge_props.get('relationship', 'No relationship')}"
            )
            print(
                f"Edge description: {edge_props.get('description', 'No description')}"
            )
            print(f"Edge weight: {edge_props.get('weight', 'No weight')}")
            # Verify that the returned properties are correct
            assert (
                edge_props.get("relationship") == edge_data["relationship"]
            ), "Edge relationship mismatch"
            assert (
                edge_props.get("description") == edge_data["description"]
            ), "Edge description mismatch"
            assert (
                edge_props.get("weight") == edge_data["weight"]
            ), "Edge weight mismatch"
        else:
            print(f"Failed to read edge properties: {node1_id} -> {node2_id}")
            assert False, f"Failed to read edge properties: {node1_id} -> {node2_id}"

        # 5.1 Verify undirected graph property - read reverse edge properties
        print(f"Reading reverse edge properties: {node2_id} -> {node1_id}")
        reverse_edge_props = await storage.get_edge(node2_id, node1_id)
        if reverse_edge_props:
            print(
                f"Successfully read reverse edge properties: {node2_id} -> {node1_id}"
            )
            print(
                f"Reverse edge relationship: {reverse_edge_props.get('relationship', 'No relationship')}"
            )
            print(
                f"Reverse edge description: {reverse_edge_props.get('description', 'No description')}"
            )
            print(
                f"Reverse edge weight: {reverse_edge_props.get('weight', 'No weight')}"
            )
            # Verify that forward and reverse edge properties are the same
            assert (
                edge_props == reverse_edge_props
            ), "Forward and reverse edge properties are not consistent, undirected graph property verification failed"
            print(
                "Undirected graph property verification successful: forward and reverse edge properties are consistent"
            )
        else:
            print(f"Failed to read reverse edge properties: {node2_id} -> {node1_id}")
            assert False, f"Failed to read reverse edge properties: {node2_id} -> {node1_id}, undirected graph property verification failed"

        print("Basic tests completed, data is preserved in the database.")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during the test: {str(e)}")
        return False


async def test_graph_advanced(storage):
    """
    Test advanced graph database operations:
    1. Use node_degree to get the degree of a node.
    2. Use edge_degree to get the degree of an edge.
    3. Use get_node_edges to get all edges of a node.
    4. Use get_all_labels to get all labels.
    5. Use get_knowledge_graph to get a knowledge graph.
    6. Use delete_node to delete a node.
    7. Use remove_nodes to delete multiple nodes.
    8. Use remove_edges to delete edges.
    9. Use drop to clean up data.
    """
    try:
        # 1. Insert test data
        # Insert node 1: Artificial Intelligence
        node1_id = "Artificial Intelligence"
        node1_data = {
            "entity_id": node1_id,
            "description": "Artificial intelligence is a branch of computer science that aims to understand the essence of intelligence and produce a new kind of intelligent machine that can react in a manner similar to human intelligence.",
            "keywords": "AI,Machine Learning,Deep Learning",
            "entity_type": "Technology Field",
        }
        print(f"Inserting node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # Insert node 2: Machine Learning
        node2_id = "Machine Learning"
        node2_data = {
            "entity_id": node2_id,
            "description": "Machine learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
            "keywords": "Supervised Learning,Unsupervised Learning,Reinforcement Learning",
            "entity_type": "Technology Field",
        }
        print(f"Inserting node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # Insert node 3: Deep Learning
        node3_id = "Deep Learning"
        node3_data = {
            "entity_id": node3_id,
            "description": "Deep learning is a branch of machine learning that uses multi-layered neural networks to simulate the learning process of the human brain.",
            "keywords": "Neural Networks,CNN,RNN",
            "entity_type": "Technology Field",
        }
        print(f"Inserting node 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # Insert edge 1: Artificial Intelligence -> Machine Learning
        edge1_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of machine learning.",
        }
        print(f"Inserting edge 1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Insert edge 2: Machine Learning -> Deep Learning
        edge2_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of machine learning includes the subfield of deep learning.",
        }
        print(f"Inserting edge 2: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 2. Test node_degree - get the degree of a node
        print(f"== Testing node_degree: {node1_id}")
        node1_degree = await storage.node_degree(node1_id)
        print(f"Degree of node {node1_id}: {node1_degree}")
        assert (
            node1_degree == 1
        ), f"Degree of node {node1_id} should be 1, but got {node1_degree}"

        # 2.1 Test degrees of all nodes
        print("== Testing degrees of all nodes")
        node2_degree = await storage.node_degree(node2_id)
        node3_degree = await storage.node_degree(node3_id)
        print(f"Degree of node {node2_id}: {node2_degree}")
        print(f"Degree of node {node3_id}: {node3_degree}")
        assert (
            node2_degree == 2
        ), f"Degree of node {node2_id} should be 2, but got {node2_degree}"
        assert (
            node3_degree == 1
        ), f"Degree of node {node3_id} should be 1, but got {node3_degree}"

        # 3. Test edge_degree - get the degree of an edge
        print(f"== Testing edge_degree: {node1_id} -> {node2_id}")
        edge_degree = await storage.edge_degree(node1_id, node2_id)
        print(f"Degree of edge {node1_id} -> {node2_id}: {edge_degree}")
        assert (
            edge_degree == 3
        ), f"Degree of edge {node1_id} -> {node2_id} should be 3, but got {edge_degree}"

        # 3.1 Test reverse edge degree - verify undirected graph property
        print(f"== Testing reverse edge degree: {node2_id} -> {node1_id}")
        reverse_edge_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"Degree of reverse edge {node2_id} -> {node1_id}: {reverse_edge_degree}")
        assert (
            edge_degree == reverse_edge_degree
        ), "Degrees of forward and reverse edges are not consistent, undirected graph property verification failed"
        print(
            "Undirected graph property verification successful: degrees of forward and reverse edges are consistent"
        )

        # 4. Test get_node_edges - get all edges of a node
        print(f"== Testing get_node_edges: {node2_id}")
        node2_edges = await storage.get_node_edges(node2_id)
        print(f"All edges of node {node2_id}: {node2_edges}")
        assert (
            len(node2_edges) == 2
        ), f"Node {node2_id} should have 2 edges, but got {len(node2_edges)}"

        # 4.1 Verify undirected graph property of node edges
        print("== Verifying undirected graph property of node edges")
        # Check if it includes connections with node1 and node3 (regardless of direction)
        has_connection_with_node1 = False
        has_connection_with_node3 = False
        for edge in node2_edges:
            # Check for connection with node1 (regardless of direction)
            if (edge[0] == node1_id and edge[1] == node2_id) or (
                edge[0] == node2_id and edge[1] == node1_id
            ):
                has_connection_with_node1 = True
            # Check for connection with node3 (regardless of direction)
            if (edge[0] == node2_id and edge[1] == node3_id) or (
                edge[0] == node3_id and edge[1] == node2_id
            ):
                has_connection_with_node3 = True

        assert (
            has_connection_with_node1
        ), f"Edge list of node {node2_id} should include a connection with {node1_id}"
        assert (
            has_connection_with_node3
        ), f"Edge list of node {node2_id} should include a connection with {node3_id}"
        print(
            f"Undirected graph property verification successful: edge list of node {node2_id} contains all relevant edges"
        )

        # 5. Test get_all_labels - get all labels
        print("== Testing get_all_labels")
        all_labels = await storage.get_all_labels()
        print(f"All labels: {all_labels}")
        assert len(all_labels) == 3, f"Should have 3 labels, but got {len(all_labels)}"
        assert node1_id in all_labels, f"{node1_id} should be in the label list"
        assert node2_id in all_labels, f"{node2_id} should be in the label list"
        assert node3_id in all_labels, f"{node3_id} should be in the label list"

        # 6. Test get_knowledge_graph - get a knowledge graph
        print("== Testing get_knowledge_graph")
        kg = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=10)
        print(f"Number of nodes in knowledge graph: {len(kg.nodes)}")
        print(f"Number of edges in knowledge graph: {len(kg.edges)}")
        assert isinstance(
            kg, KnowledgeGraph
        ), "The returned result should be of type KnowledgeGraph"
        assert (
            len(kg.nodes) == 3
        ), f"The knowledge graph should have 3 nodes, but got {len(kg.nodes)}"
        assert (
            len(kg.edges) == 2
        ), f"The knowledge graph should have 2 edges, but got {len(kg.edges)}"

        # 7. Test delete_node - delete a node
        print(f"== Testing delete_node: {node3_id}")
        await storage.delete_node(node3_id)
        node3_props = await storage.get_node(node3_id)
        print(f"Querying node properties after deletion {node3_id}: {node3_props}")
        assert node3_props is None, f"Node {node3_id} should have been deleted"

        # Re-insert node 3 for subsequent tests
        await storage.upsert_node(node3_id, node3_data)
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 8. Test remove_edges - delete edges
        print(f"== Testing remove_edges: {node2_id} -> {node3_id}")
        await storage.remove_edges([(node2_id, node3_id)])
        edge_props = await storage.get_edge(node2_id, node3_id)
        print(
            f"Querying edge properties after deletion {node2_id} -> {node3_id}: {edge_props}"
        )
        assert (
            edge_props is None
        ), f"Edge {node2_id} -> {node3_id} should have been deleted"

        # 8.1 Verify undirected graph property of edge deletion
        print(
            f"== Verifying undirected graph property of edge deletion: {node3_id} -> {node2_id}"
        )
        reverse_edge_props = await storage.get_edge(node3_id, node2_id)
        print(
            f"Querying reverse edge properties after deletion {node3_id} -> {node2_id}: {reverse_edge_props}"
        )
        assert (
            reverse_edge_props is None
        ), f"Reverse edge {node3_id} -> {node2_id} should also be deleted, undirected graph property verification failed"
        print(
            "Undirected graph property verification successful: deleting an edge in one direction also deletes the reverse edge"
        )

        # 9. Test remove_nodes - delete multiple nodes
        print(f"== Testing remove_nodes: [{node2_id}, {node3_id}]")
        await storage.remove_nodes([node2_id, node3_id])
        node2_props = await storage.get_node(node2_id)
        node3_props = await storage.get_node(node3_id)
        print(f"Querying node properties after deletion {node2_id}: {node2_props}")
        print(f"Querying node properties after deletion {node3_id}: {node3_props}")
        assert node2_props is None, f"Node {node2_id} should have been deleted"
        assert node3_props is None, f"Node {node3_id} should have been deleted"

        print("\nAdvanced tests completed.")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during the test: {str(e)}")
        return False


async def test_graph_batch_operations(storage):
    """
    Test batch operations of the graph database:
    1. Use get_nodes_batch to get properties of multiple nodes in batch.
    2. Use node_degrees_batch to get degrees of multiple nodes in batch.
    3. Use edge_degrees_batch to get degrees of multiple edges in batch.
    4. Use get_edges_batch to get properties of multiple edges in batch.
    5. Use get_nodes_edges_batch to get all edges of multiple nodes in batch.
    """
    try:
        chunk1_id = "1"
        chunk2_id = "2"
        chunk3_id = "3"
        # 1. Insert test data
        # Insert node 1: Artificial Intelligence
        node1_id = "Artificial Intelligence"
        node1_data = {
            "entity_id": node1_id,
            "description": "Artificial intelligence is a branch of computer science that aims to understand the essence of intelligence and produce a new kind of intelligent machine that can react in a manner similar to human intelligence.",
            "keywords": "AI,Machine Learning,Deep Learning",
            "entity_type": "Technology Field",
            "source_id": GRAPH_FIELD_SEP.join([chunk1_id, chunk2_id]),
        }
        print(f"Inserting node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # Insert node 2: Machine Learning
        node2_id = "Machine Learning"
        node2_data = {
            "entity_id": node2_id,
            "description": "Machine learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
            "keywords": "Supervised Learning,Unsupervised Learning,Reinforcement Learning",
            "entity_type": "Technology Field",
            "source_id": GRAPH_FIELD_SEP.join([chunk2_id, chunk3_id]),
        }
        print(f"Inserting node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # Insert node 3: Deep Learning
        node3_id = "Deep Learning"
        node3_data = {
            "entity_id": node3_id,
            "description": "Deep learning is a branch of machine learning that uses multi-layered neural networks to simulate the learning process of the human brain.",
            "keywords": "Neural Networks,CNN,RNN",
            "entity_type": "Technology Field",
            "source_id": GRAPH_FIELD_SEP.join([chunk3_id]),
        }
        print(f"Inserting node 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # Insert node 4: Natural Language Processing
        node4_id = "Natural Language Processing"
        node4_data = {
            "entity_id": node4_id,
            "description": "Natural language processing is a branch of artificial intelligence that focuses on enabling computers to understand and process human language.",
            "keywords": "NLP,Text Analysis,Language Models",
            "entity_type": "Technology Field",
        }
        print(f"Inserting node 4: {node4_id}")
        await storage.upsert_node(node4_id, node4_data)

        # Insert node 5: Computer Vision
        node5_id = "Computer Vision"
        node5_data = {
            "entity_id": node5_id,
            "description": "Computer vision is a branch of artificial intelligence that focuses on enabling computers to gain information from images or videos.",
            "keywords": "CV,Image Recognition,Object Detection",
            "entity_type": "Technology Field",
        }
        print(f"Inserting node 5: {node5_id}")
        await storage.upsert_node(node5_id, node5_data)

        # Insert edge 1: Artificial Intelligence -> Machine Learning
        edge1_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of machine learning.",
            "source_id": GRAPH_FIELD_SEP.join([chunk1_id, chunk2_id]),
        }
        print(f"Inserting edge 1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Insert edge 2: Machine Learning -> Deep Learning
        edge2_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of machine learning includes the subfield of deep learning.",
            "source_id": GRAPH_FIELD_SEP.join([chunk2_id, chunk3_id]),
        }
        print(f"Inserting edge 2: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # Insert edge 3: Artificial Intelligence -> Natural Language Processing
        edge3_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of natural language processing.",
            "source_id": GRAPH_FIELD_SEP.join([chunk3_id]),
        }
        print(f"Inserting edge 3: {node1_id} -> {node4_id}")
        await storage.upsert_edge(node1_id, node4_id, edge3_data)

        # Insert edge 4: Artificial Intelligence -> Computer Vision
        edge4_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of computer vision.",
        }
        print(f"Inserting edge 4: {node1_id} -> {node5_id}")
        await storage.upsert_edge(node1_id, node5_id, edge4_data)

        # Insert edge 5: Deep Learning -> Natural Language Processing
        edge5_data = {
            "relationship": "applied to",
            "weight": 0.8,
            "description": "Deep learning techniques are applied in the field of natural language processing.",
        }
        print(f"Inserting edge 5: {node3_id} -> {node4_id}")
        await storage.upsert_edge(node3_id, node4_id, edge5_data)

        # Insert edge 6: Deep Learning -> Computer Vision
        edge6_data = {
            "relationship": "applied to",
            "weight": 0.8,
            "description": "Deep learning techniques are applied in the field of computer vision.",
        }
        print(f"Inserting edge 6: {node3_id} -> {node5_id}")
        await storage.upsert_edge(node3_id, node5_id, edge6_data)

        # 2. Test get_nodes_batch - batch get properties of multiple nodes
        print("== Testing get_nodes_batch")
        node_ids = [node1_id, node2_id, node3_id]
        nodes_dict = await storage.get_nodes_batch(node_ids)
        print(f"Batch get node properties result: {nodes_dict.keys()}")
        assert len(nodes_dict) == 3, f"Should return 3 nodes, but got {len(nodes_dict)}"
        assert node1_id in nodes_dict, f"{node1_id} should be in the result"
        assert node2_id in nodes_dict, f"{node2_id} should be in the result"
        assert node3_id in nodes_dict, f"{node3_id} should be in the result"
        assert (
            nodes_dict[node1_id]["description"] == node1_data["description"]
        ), f"{node1_id} description mismatch"
        assert (
            nodes_dict[node2_id]["description"] == node2_data["description"]
        ), f"{node2_id} description mismatch"
        assert (
            nodes_dict[node3_id]["description"] == node3_data["description"]
        ), f"{node3_id} description mismatch"

        # 3. Test node_degrees_batch - batch get degrees of multiple nodes
        print("== Testing node_degrees_batch")
        node_degrees = await storage.node_degrees_batch(node_ids)
        print(f"Batch get node degrees result: {node_degrees}")
        assert (
            len(node_degrees) == 3
        ), f"Should return degrees of 3 nodes, but got {len(node_degrees)}"
        assert node1_id in node_degrees, f"{node1_id} should be in the result"
        assert node2_id in node_degrees, f"{node2_id} should be in the result"
        assert node3_id in node_degrees, f"{node3_id} should be in the result"
        assert (
            node_degrees[node1_id] == 3
        ), f"Degree of {node1_id} should be 3, but got {node_degrees[node1_id]}"
        assert (
            node_degrees[node2_id] == 2
        ), f"Degree of {node2_id} should be 2, but got {node_degrees[node2_id]}"
        assert (
            node_degrees[node3_id] == 3
        ), f"Degree of {node3_id} should be 3, but got {node_degrees[node3_id]}"

        # 4. Test edge_degrees_batch - batch get degrees of multiple edges
        print("== Testing edge_degrees_batch")
        edges = [(node1_id, node2_id), (node2_id, node3_id), (node3_id, node4_id)]
        edge_degrees = await storage.edge_degrees_batch(edges)
        print(f"Batch get edge degrees result: {edge_degrees}")
        assert (
            len(edge_degrees) == 3
        ), f"Should return degrees of 3 edges, but got {len(edge_degrees)}"
        assert (
            node1_id,
            node2_id,
        ) in edge_degrees, f"Edge {node1_id} -> {node2_id} should be in the result"
        assert (
            node2_id,
            node3_id,
        ) in edge_degrees, f"Edge {node2_id} -> {node3_id} should be in the result"
        assert (
            node3_id,
            node4_id,
        ) in edge_degrees, f"Edge {node3_id} -> {node4_id} should be in the result"
        # Verify edge degrees (sum of source and target node degrees)
        assert (
            edge_degrees[(node1_id, node2_id)] == 5
        ), f"Degree of edge {node1_id} -> {node2_id} should be 5, but got {edge_degrees[(node1_id, node2_id)]}"
        assert (
            edge_degrees[(node2_id, node3_id)] == 5
        ), f"Degree of edge {node2_id} -> {node3_id} should be 5, but got {edge_degrees[(node2_id, node3_id)]}"
        assert (
            edge_degrees[(node3_id, node4_id)] == 5
        ), f"Degree of edge {node3_id} -> {node4_id} should be 5, but got {edge_degrees[(node3_id, node4_id)]}"

        # 5. Test get_edges_batch - batch get properties of multiple edges
        print("== Testing get_edges_batch")
        # Convert list of tuples to list of dicts for Neo4j style
        edge_dicts = [{"src": src, "tgt": tgt} for src, tgt in edges]
        edges_dict = await storage.get_edges_batch(edge_dicts)
        print(f"Batch get edge properties result: {edges_dict.keys()}")
        assert (
            len(edges_dict) == 3
        ), f"Should return properties of 3 edges, but got {len(edges_dict)}"
        assert (
            node1_id,
            node2_id,
        ) in edges_dict, f"Edge {node1_id} -> {node2_id} should be in the result"
        assert (
            node2_id,
            node3_id,
        ) in edges_dict, f"Edge {node2_id} -> {node3_id} should be in the result"
        assert (
            node3_id,
            node4_id,
        ) in edges_dict, f"Edge {node3_id} -> {node4_id} should be in the result"
        assert (
            edges_dict[(node1_id, node2_id)]["relationship"]
            == edge1_data["relationship"]
        ), f"Edge {node1_id} -> {node2_id} relationship mismatch"
        assert (
            edges_dict[(node2_id, node3_id)]["relationship"]
            == edge2_data["relationship"]
        ), f"Edge {node2_id} -> {node3_id} relationship mismatch"
        assert (
            edges_dict[(node3_id, node4_id)]["relationship"]
            == edge5_data["relationship"]
        ), f"Edge {node3_id} -> {node4_id} relationship mismatch"

        # 5.1 Test batch get of reverse edges - verify undirected property
        print("== Testing batch get of reverse edges")
        # Create list of dicts for reverse edges
        reverse_edge_dicts = [{"src": tgt, "tgt": src} for src, tgt in edges]
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)
        print(f"Batch get reverse edge properties result: {reverse_edges_dict.keys()}")
        assert (
            len(reverse_edges_dict) == 3
        ), f"Should return properties of 3 reverse edges, but got {len(reverse_edges_dict)}"

        # Verify that properties of forward and reverse edges are consistent
        for (src, tgt), props in edges_dict.items():
            assert (
                (
                    tgt,
                    src,
                )
                in reverse_edges_dict
            ), f"Reverse edge {tgt} -> {src} should be in the result"
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"Properties of edge {src} -> {tgt} and reverse edge {tgt} -> {src} are inconsistent"

        print(
            "Undirected graph property verification successful: properties of batch-retrieved forward and reverse edges are consistent"
        )

        # 6. Test get_nodes_edges_batch - batch get all edges of multiple nodes
        print("== Testing get_nodes_edges_batch")
        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node3_id])
        print(f"Batch get node edges result: {nodes_edges.keys()}")
        assert (
            len(nodes_edges) == 2
        ), f"Should return edges for 2 nodes, but got {len(nodes_edges)}"
        assert node1_id in nodes_edges, f"{node1_id} should be in the result"
        assert node3_id in nodes_edges, f"{node3_id} should be in the result"
        assert (
            len(nodes_edges[node1_id]) == 3
        ), f"{node1_id} should have 3 edges, but has {len(nodes_edges[node1_id])}"
        assert (
            len(nodes_edges[node3_id]) == 3
        ), f"{node3_id} should have 3 edges, but has {len(nodes_edges[node3_id])}"

        # 6.1 Verify undirected property of batch-retrieved node edges
        print("== Verifying undirected property of batch-retrieved node edges")

        # Check if node 1's edges include all relevant edges (regardless of direction)
        node1_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if src == node1_id
        ]
        node1_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if tgt == node1_id
        ]
        print(f"Outgoing edges of node {node1_id}: {node1_outgoing_edges}")
        print(f"Incoming edges of node {node1_id}: {node1_incoming_edges}")

        # Check for edges to Machine Learning, Natural Language Processing, and Computer Vision
        has_edge_to_node2 = any(tgt == node2_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node4 = any(tgt == node4_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node5 = any(tgt == node5_id for _, tgt in node1_outgoing_edges)

        assert (
            has_edge_to_node2
        ), f"Edge list of node {node1_id} should include an edge to {node2_id}"
        assert (
            has_edge_to_node4
        ), f"Edge list of node {node1_id} should include an edge to {node4_id}"
        assert (
            has_edge_to_node5
        ), f"Edge list of node {node1_id} should include an edge to {node5_id}"

        # Check if node 3's edges include all relevant edges (regardless of direction)
        node3_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if src == node3_id
        ]
        node3_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if tgt == node3_id
        ]
        print(f"Outgoing edges of node {node3_id}: {node3_outgoing_edges}")
        print(f"Incoming edges of node {node3_id}: {node3_incoming_edges}")

        # Check for connections with Machine Learning, Natural Language Processing, and Computer Vision (ignoring direction)
        has_connection_with_node2 = any(
            (src == node2_id and tgt == node3_id)
            or (src == node3_id and tgt == node2_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node4 = any(
            (src == node3_id and tgt == node4_id)
            or (src == node4_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node5 = any(
            (src == node3_id and tgt == node5_id)
            or (src == node5_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )

        assert (
            has_connection_with_node2
        ), f"Edge list of node {node3_id} should include a connection with {node2_id}"
        assert (
            has_connection_with_node4
        ), f"Edge list of node {node3_id} should include a connection with {node4_id}"
        assert (
            has_connection_with_node5
        ), f"Edge list of node {node3_id} should include a connection with {node5_id}"

        print(
            "Undirected graph property verification successful: batch-retrieved node edges include all relevant edges (regardless of direction)"
        )

        print("\nBatch operations tests completed.")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during the test: {str(e)}")
        return False


async def test_graph_special_characters(storage):
    """
    Test the graph database's handling of special characters:
    1. Test node names and descriptions containing single quotes, double quotes, and backslashes.
    2. Test edge descriptions containing single quotes, double quotes, and backslashes.
    3. Verify that special characters are saved and retrieved correctly.
    """
    try:
        # 1. Test special characters in node name
        node1_id = "Node with 'single quotes'"
        node1_data = {
            "entity_id": node1_id,
            "description": "This description contains 'single quotes', \"double quotes\", and \\backslashes",
            "keywords": "special characters,quotes,escaping",
            "entity_type": "Test Node",
        }
        print(f"Inserting node with special characters 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. Test double quotes in node name
        node2_id = 'Node with "double quotes"'
        node2_data = {
            "entity_id": node2_id,
            "description": "This description contains both 'single quotes' and \"double quotes\" and \\a\\path",
            "keywords": "special characters,quotes,JSON",
            "entity_type": "Test Node",
        }
        print(f"Inserting node with special characters 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. Test backslashes in node name
        node3_id = "Node with \\backslashes\\"
        node3_data = {
            "entity_id": node3_id,
            "description": "This description contains a Windows path C:\\Program Files\\ and escape characters \\n\\t",
            "keywords": "backslashes,paths,escaping",
            "entity_type": "Test Node",
        }
        print(f"Inserting node with special characters 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 4. Test special characters in edge description
        edge1_data = {
            "relationship": "special 'relationship'",
            "weight": 1.0,
            "description": "This edge description contains 'single quotes', \"double quotes\", and \\backslashes",
        }
        print(f"Inserting edge with special characters: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 5. Test more complex combination of special characters in edge description
        edge2_data = {
            "relationship": 'complex "relationship"\\type',
            "weight": 0.8,
            "description": "Contains SQL injection attempt: SELECT * FROM users WHERE name='admin'--",
        }
        print(
            f"Inserting edge with complex special characters: {node2_id} -> {node3_id}"
        )
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 6. Verify that node special characters are saved correctly
        print("\n== Verifying node special characters")
        for node_id, original_data in [
            (node1_id, node1_data),
            (node2_id, node2_data),
            (node3_id, node3_data),
        ]:
            node_props = await storage.get_node(node_id)
            if node_props:
                print(f"Successfully read node: {node_id}")
                print(
                    f"Node description: {node_props.get('description', 'No description')}"
                )

                # Verify node ID is saved correctly
                assert (
                    node_props.get("entity_id") == node_id
                ), f"Node ID mismatch: expected {node_id}, got {node_props.get('entity_id')}"

                # Verify description is saved correctly
                assert (
                    node_props.get("description") == original_data["description"]
                ), f"Node description mismatch: expected {original_data['description']}, got {node_props.get('description')}"

                print(f"Node {node_id} special character verification successful")
            else:
                print(f"Failed to read node properties: {node_id}")
                assert False, f"Failed to read node properties: {node_id}"

        # 7. Verify that edge special characters are saved correctly
        print("\n== Verifying edge special characters")
        edge1_props = await storage.get_edge(node1_id, node2_id)
        if edge1_props:
            print(f"Successfully read edge: {node1_id} -> {node2_id}")
            print(
                f"Edge relationship: {edge1_props.get('relationship', 'No relationship')}"
            )
            print(
                f"Edge description: {edge1_props.get('description', 'No description')}"
            )

            # Verify edge relationship is saved correctly
            assert (
                edge1_props.get("relationship") == edge1_data["relationship"]
            ), f"Edge relationship mismatch: expected {edge1_data['relationship']}, got {edge1_props.get('relationship')}"

            # Verify edge description is saved correctly
            assert (
                edge1_props.get("description") == edge1_data["description"]
            ), f"Edge description mismatch: expected {edge1_data['description']}, got {edge1_props.get('description')}"

            print(
                f"Edge {node1_id} -> {node2_id} special character verification successful"
            )
        else:
            print(f"Failed to read edge properties: {node1_id} -> {node2_id}")
            assert False, f"Failed to read edge properties: {node1_id} -> {node2_id}"

        edge2_props = await storage.get_edge(node2_id, node3_id)
        if edge2_props:
            print(f"Successfully read edge: {node2_id} -> {node3_id}")
            print(
                f"Edge relationship: {edge2_props.get('relationship', 'No relationship')}"
            )
            print(
                f"Edge description: {edge2_props.get('description', 'No description')}"
            )

            # Verify edge relationship is saved correctly
            assert (
                edge2_props.get("relationship") == edge2_data["relationship"]
            ), f"Edge relationship mismatch: expected {edge2_data['relationship']}, got {edge2_props.get('relationship')}"

            # Verify edge description is saved correctly
            assert (
                edge2_props.get("description") == edge2_data["description"]
            ), f"Edge description mismatch: expected {edge2_data['description']}, got {edge2_props.get('description')}"

            print(
                f"Edge {node2_id} -> {node3_id} special character verification successful"
            )
        else:
            print(f"Failed to read edge properties: {node2_id} -> {node3_id}")
            assert False, f"Failed to read edge properties: {node2_id} -> {node3_id}"

        print("\nSpecial character tests completed, data is preserved in the database.")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during the test: {str(e)}")
        return False


async def test_graph_undirected_property(storage):
    """
    Specifically test the undirected graph property of the storage:
    1. Verify that after inserting an edge in one direction, a reverse query can retrieve the same result.
    2. Verify that edge properties are consistent in forward and reverse queries.
    3. Verify that after deleting an edge in one direction, the edge in the other direction is also deleted.
    4. Verify the undirected property in batch operations.
    """
    try:
        # 1. Insert test data
        # Insert node 1: Computer Science
        node1_id = "Computer Science"
        node1_data = {
            "entity_id": node1_id,
            "description": "Computer science is the study of computers and their applications.",
            "keywords": "computer,science,technology",
            "entity_type": "Discipline",
        }
        print(f"Inserting node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # Insert node 2: Data Structures
        node2_id = "Data Structures"
        node2_data = {
            "entity_id": node2_id,
            "description": "A data structure is a fundamental concept in computer science used to organize and store data.",
            "keywords": "data,structure,organization",
            "entity_type": "Concept",
        }
        print(f"Inserting node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # Insert node 3: Algorithms
        node3_id = "Algorithms"
        node3_data = {
            "entity_id": node3_id,
            "description": "An algorithm is a set of steps and methods for solving problems.",
            "keywords": "algorithm,steps,methods",
            "entity_type": "Concept",
        }
        print(f"Inserting node 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 2. Test undirected property after edge insertion
        print("\n== Testing undirected property after edge insertion")

        # Insert edge 1: Computer Science -> Data Structures
        edge1_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "Computer science includes the concept of data structures.",
        }
        print(f"Inserting edge 1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Verify forward query
        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(f"Forward edge properties: {forward_edge}")
        assert (
            forward_edge is not None
        ), f"Failed to read forward edge properties: {node1_id} -> {node2_id}"

        # Verify reverse query
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(f"Reverse edge properties: {reverse_edge}")
        assert (
            reverse_edge is not None
        ), f"Failed to read reverse edge properties: {node2_id} -> {node1_id}"

        # Verify that forward and reverse edge properties are consistent
        assert (
            forward_edge == reverse_edge
        ), "Forward and reverse edge properties are inconsistent, undirected property verification failed"
        print(
            "Undirected property verification successful: forward and reverse edge properties are consistent"
        )

        # 3. Test undirected property of edge degree
        print("\n== Testing undirected property of edge degree")

        # Insert edge 2: Computer Science -> Algorithms
        edge2_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "Computer science includes the concept of algorithms.",
        }
        print(f"Inserting edge 2: {node1_id} -> {node3_id}")
        await storage.upsert_edge(node1_id, node3_id, edge2_data)

        # Verify degrees of forward and reverse edges
        forward_degree = await storage.edge_degree(node1_id, node2_id)
        reverse_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"Degree of forward edge {node1_id} -> {node2_id}: {forward_degree}")
        print(f"Degree of reverse edge {node2_id} -> {node1_id}: {reverse_degree}")
        assert (
            forward_degree == reverse_degree
        ), "Degrees of forward and reverse edges are inconsistent, undirected property verification failed"
        print(
            "Undirected property verification successful: degrees of forward and reverse edges are consistent"
        )

        # 4. Test undirected property of edge deletion
        print("\n== Testing undirected property of edge deletion")

        # Delete forward edge
        print(f"Deleting edge: {node1_id} -> {node2_id}")
        await storage.remove_edges([(node1_id, node2_id)])

        # Verify forward edge is deleted
        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(
            f"Querying forward edge properties after deletion {node1_id} -> {node2_id}: {forward_edge}"
        )
        assert (
            forward_edge is None
        ), f"Edge {node1_id} -> {node2_id} should have been deleted"

        # Verify reverse edge is also deleted
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(
            f"Querying reverse edge properties after deletion {node2_id} -> {node1_id}: {reverse_edge}"
        )
        assert (
            reverse_edge is None
        ), f"Reverse edge {node2_id} -> {node1_id} should also be deleted, undirected property verification failed"
        print(
            "Undirected property verification successful: deleting an edge in one direction also deletes the reverse edge"
        )

        # 5. Test undirected property in batch operations
        print("\n== Testing undirected property in batch operations")

        # Re-insert edge
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Batch get edge properties
        edge_dicts = [
            {"src": node1_id, "tgt": node2_id},
            {"src": node1_id, "tgt": node3_id},
        ]
        reverse_edge_dicts = [
            {"src": node2_id, "tgt": node1_id},
            {"src": node3_id, "tgt": node1_id},
        ]

        edges_dict = await storage.get_edges_batch(edge_dicts)
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)

        print(f"Batch get forward edge properties result: {edges_dict.keys()}")
        print(f"Batch get reverse edge properties result: {reverse_edges_dict.keys()}")

        # Verify that properties of forward and reverse edges are consistent
        for (src, tgt), props in edges_dict.items():
            assert (
                (
                    tgt,
                    src,
                )
                in reverse_edges_dict
            ), f"Reverse edge {tgt} -> {src} should be in the result"
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"Properties of edge {src} -> {tgt} and reverse edge {tgt} -> {src} are inconsistent"

        print(
            "Undirected property verification successful: properties of batch-retrieved forward and reverse edges are consistent"
        )

        # 6. Test undirected property of batch-retrieved node edges
        print("\n== Testing undirected property of batch-retrieved node edges")

        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node2_id])
        print(f"Batch get node edges result: {nodes_edges.keys()}")

        # Check if node 1's edges include all relevant edges (regardless of direction)
        node1_edges = nodes_edges[node1_id]
        node2_edges = nodes_edges[node2_id]

        # Check if node 1 has edges to node 2 and node 3
        has_edge_to_node2 = any(
            (src == node1_id and tgt == node2_id) for src, tgt in node1_edges
        )
        has_edge_to_node3 = any(
            (src == node1_id and tgt == node3_id) for src, tgt in node1_edges
        )

        assert (
            has_edge_to_node2
        ), f"Edge list of node {node1_id} should include an edge to {node2_id}"
        assert (
            has_edge_to_node3
        ), f"Edge list of node {node1_id} should include an edge to {node3_id}"

        # Check if node 2 has a connection with node 1
        has_edge_to_node1 = any(
            (src == node2_id and tgt == node1_id)
            or (src == node1_id and tgt == node2_id)
            for src, tgt in node2_edges
        )
        assert (
            has_edge_to_node1
        ), f"Edge list of node {node2_id} should include a connection with {node1_id}"

        print(
            "Undirected property verification successful: batch-retrieved node edges include all relevant edges (regardless of direction)"
        )

        print("\nUndirected property tests completed.")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during the test: {str(e)}")
        return False


async def main():
    """Main function"""
    # Display program title
    ASCIIColors.cyan("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            General Graph Storage Test Program                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Check for .env file
    if not check_env_file():
        return

    # Load environment variables
    load_dotenv(dotenv_path=".env", override=False)

    # Get graph storage type
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    ASCIIColors.magenta(
        f"\nCurrently configured graph storage type: {graph_storage_type}"
    )
    ASCIIColors.white(
        f"Supported graph storage types: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
    )

    # Initialize storage instance
    storage = await initialize_graph_storage()
    if not storage:
        ASCIIColors.red("Failed to initialize storage instance, exiting test program.")
        return

    try:
        # Display test options
        ASCIIColors.yellow("\nPlease select a test type:")
        ASCIIColors.white("1. Basic Test (Node and edge insertion, reading)")
        ASCIIColors.white(
            "2. Advanced Test (Degree, labels, knowledge graph, deletion, etc.)"
        )
        ASCIIColors.white(
            "3. Batch Operations Test (Batch get node/edge properties, degrees, etc.)"
        )
        ASCIIColors.white(
            "4. Undirected Property Test (Verify undirected properties of the storage)"
        )
        ASCIIColors.white(
            "5. Special Characters Test (Verify handling of single/double quotes, backslashes, etc.)"
        )
        ASCIIColors.white("6. All Tests")

        choice = input("\nEnter your choice (1/2/3/4/5/6): ")

        # Clean data before running tests
        if choice in ["1", "2", "3", "4", "5", "6"]:
            ASCIIColors.yellow("\nCleaning data before running tests...")
            await storage.drop()
            ASCIIColors.green("Data cleanup complete\n")

        if choice == "1":
            await test_graph_basic(storage)
        elif choice == "2":
            await test_graph_advanced(storage)
        elif choice == "3":
            await test_graph_batch_operations(storage)
        elif choice == "4":
            await test_graph_undirected_property(storage)
        elif choice == "5":
            await test_graph_special_characters(storage)
        elif choice == "6":
            ASCIIColors.cyan("\n=== Starting Basic Test ===")
            basic_result = await test_graph_basic(storage)

            if basic_result:
                ASCIIColors.cyan("\n=== Starting Advanced Test ===")
                advanced_result = await test_graph_advanced(storage)

                if advanced_result:
                    ASCIIColors.cyan("\n=== Starting Batch Operations Test ===")
                    batch_result = await test_graph_batch_operations(storage)

                    if batch_result:
                        ASCIIColors.cyan("\n=== Starting Undirected Property Test ===")
                        undirected_result = await test_graph_undirected_property(
                            storage
                        )

                        if undirected_result:
                            ASCIIColors.cyan(
                                "\n=== Starting Special Characters Test ==="
                            )
                            await test_graph_special_characters(storage)
        else:
            ASCIIColors.red("Invalid choice")

    finally:
        # Close connection
        if storage:
            await storage.finalize()
            ASCIIColors.green("\nStorage connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
