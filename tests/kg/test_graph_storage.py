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
- PGTableGraphStorage
- MemgraphStorage

Every test is triggerable two ways, and both run the exact same code through the
same pytest fixture — the command line is a thin wrapper around pytest, not a
second execution path:

    # via pytest (integration tests are opt-in)
    pytest tests/kg/test_graph_storage.py --run-integration
    pytest tests/kg/test_graph_storage.py::test_graph_basic --run-integration
    ./scripts/test.sh tests/kg/test_graph_storage.py --run-integration

    # via the command line — exit code is pytest's; non-interactive under CI/a
    # pipe (skips the confirmation below automatically), or pass -y anywhere
    python tests/kg/test_graph_storage.py                 # every test
    python tests/kg/test_graph_storage.py basic advanced   # a subset
    python tests/kg/test_graph_storage.py --list           # show the names
    python tests/kg/test_graph_storage.py basic -- -x --tb=long   # extra pytest args

When run from an actual terminal, the CLI pauses once up front to name the
backend under test (LIGHTRAG_GRAPH_STORAGE, defaulting to NetworkXStorage) and
which tests are about to run, and repeats the backend name at the end next to
the pass/fail result — pytest's own summary says nothing about which backend
just ran. -y/--yes, or stdin not being a TTY, skips the pause.

No CI workflow runs the tests in this module. Its coverage of the
PGTableGraphStorage contract has been ported into
tests/kg/pgtable_impl/test_pgtable_smoke.py, which pg-smoke.yml runs against a
live PostgreSQL server on every push/PR — see that module's docstring for the
merge. This file remains the manual entry point for the *other* five backends
(NetworkXStorage, Neo4JStorage, MongoDBStorage, PGGraphStorage,
MemgraphStorage): point `.env` at a live instance and run it by hand, via
either trigger above, whenever you touch a backend it covers.
"""

import argparse
import inspect
import os
import sys
import importlib
import numpy as np
import pytest
from dotenv import load_dotenv
from ascii_colors import ASCIIColors

# Add the project root directory to the front of the Python path so this
# script always exercises the checked-out source tree, not a stale installed
# lightrag package from the active virtualenv.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

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
    Warn if the .env file is missing. Never blocks.

    This runs inside the pytest fixture, so it must not prompt: an input() here
    hangs `pytest` in a terminal and dies on EOF under CI or any captured run.
    Configuration problems surface as the fixture's own skip (missing backend /
    env vars) rather than as a question nobody is there to answer.
    """
    if not os.path.exists(".env"):
        ASCIIColors.yellow(
            "Warning: .env file not found in the current directory. "
            "This may affect storage configuration loading."
        )


def _report_and_reraise(exc: Exception) -> None:
    """Print the failure marker CI greps for, then re-raise so pytest fails.

    The test bodies below wrap themselves in try/except purely for this log line.
    Returning a value from that except block — as an earlier version did — does
    NOT fail the test: pytest-asyncio discards a coroutine test's return value,
    so a swallowed assertion was reported as a PASSED test. Re-raising is what
    makes pytest the authority on the result, for both entry points.
    """
    ASCIIColors.red(f"An error occurred during the test: {str(exc)}")
    raise exc


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


@pytest.fixture
async def storage():
    """
    Pytest fixture for graph storage integration tests.

    Each test gets an initialized storage instance with a clean graph state.

    Deliberately silent about which backend it initialized: a bare `pytest
    tests/kg/test_graph_storage.py -m integration --run-integration` is the
    CI-equivalent path — its backend is a config fact (a CI job's env: block, or
    a developer's own just-set env var), not something this fixture needs to
    remind anyone of on every test. That reminder is the CLI's job (see
    _confirm_backend and main()), for the one case where a human might not
    otherwise know what's currently configured.
    """
    load_dotenv(dotenv_path=".env", override=False)
    check_env_file()

    storage_instance = await initialize_graph_storage()
    if storage_instance is None:
        pytest.skip("Graph storage backend is not configured for integration tests")

    try:
        await storage_instance.drop()
        yield storage_instance
    finally:
        try:
            await storage_instance.drop()
        except Exception as exc:
            ASCIIColors.yellow(f"Warning: failed to drop test graph data: {exc}")
        finally:
            await storage_instance.finalize()


@pytest.mark.integration
@pytest.mark.requires_db
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
            assert node1_props.get("entity_id") == node1_id, (
                f"Node ID mismatch: expected {node1_id}, got {node1_props.get('entity_id')}"
            )
            assert node1_props.get("description") == node1_data["description"], (
                "Node description mismatch"
            )
            assert node1_props.get("entity_type") == node1_data["entity_type"], (
                "Node type mismatch"
            )
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
            assert edge_props.get("relationship") == edge_data["relationship"], (
                "Edge relationship mismatch"
            )
            assert edge_props.get("description") == edge_data["description"], (
                "Edge description mismatch"
            )
            assert edge_props.get("weight") == edge_data["weight"], (
                "Edge weight mismatch"
            )
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
            assert edge_props == reverse_edge_props, (
                "Forward and reverse edge properties are not consistent, undirected graph property verification failed"
            )
            print(
                "Undirected graph property verification successful: forward and reverse edge properties are consistent"
            )
        else:
            print(f"Failed to read reverse edge properties: {node2_id} -> {node1_id}")
            assert False, (
                f"Failed to read reverse edge properties: {node2_id} -> {node1_id}, undirected graph property verification failed"
            )

        print("Basic tests completed, data is preserved in the database.")

    except Exception as e:
        _report_and_reraise(e)


@pytest.mark.integration
@pytest.mark.requires_db
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
        assert node1_degree == 1, (
            f"Degree of node {node1_id} should be 1, but got {node1_degree}"
        )

        # 2.1 Test degrees of all nodes
        print("== Testing degrees of all nodes")
        node2_degree = await storage.node_degree(node2_id)
        node3_degree = await storage.node_degree(node3_id)
        print(f"Degree of node {node2_id}: {node2_degree}")
        print(f"Degree of node {node3_id}: {node3_degree}")
        assert node2_degree == 2, (
            f"Degree of node {node2_id} should be 2, but got {node2_degree}"
        )
        assert node3_degree == 1, (
            f"Degree of node {node3_id} should be 1, but got {node3_degree}"
        )

        # 3. Test edge_degree - get the degree of an edge
        print(f"== Testing edge_degree: {node1_id} -> {node2_id}")
        edge_degree = await storage.edge_degree(node1_id, node2_id)
        print(f"Degree of edge {node1_id} -> {node2_id}: {edge_degree}")
        assert edge_degree == 3, (
            f"Degree of edge {node1_id} -> {node2_id} should be 3, but got {edge_degree}"
        )

        # 3.1 Test reverse edge degree - verify undirected graph property
        print(f"== Testing reverse edge degree: {node2_id} -> {node1_id}")
        reverse_edge_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"Degree of reverse edge {node2_id} -> {node1_id}: {reverse_edge_degree}")
        assert edge_degree == reverse_edge_degree, (
            "Degrees of forward and reverse edges are not consistent, undirected graph property verification failed"
        )
        print(
            "Undirected graph property verification successful: degrees of forward and reverse edges are consistent"
        )

        # 4. Test get_node_edges - get all edges of a node
        print(f"== Testing get_node_edges: {node2_id}")
        node2_edges = await storage.get_node_edges(node2_id)
        print(f"All edges of node {node2_id}: {node2_edges}")
        assert len(node2_edges) == 2, (
            f"Node {node2_id} should have 2 edges, but got {len(node2_edges)}"
        )

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

        assert has_connection_with_node1, (
            f"Edge list of node {node2_id} should include a connection with {node1_id}"
        )
        assert has_connection_with_node3, (
            f"Edge list of node {node2_id} should include a connection with {node3_id}"
        )
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
        assert isinstance(kg, KnowledgeGraph), (
            "The returned result should be of type KnowledgeGraph"
        )
        assert len(kg.nodes) == 3, (
            f"The knowledge graph should have 3 nodes, but got {len(kg.nodes)}"
        )
        assert len(kg.edges) == 2, (
            f"The knowledge graph should have 2 edges, but got {len(kg.edges)}"
        )

        # 6.1 Every returned node must carry its entity_id in properties: the
        # WebUI reads properties['entity_id'] to render the node "Name" row and
        # neighbour/edge-endpoint labels. A backend that strips it leaves the
        # property panel nameless (regression guard for all backends).
        expected_ids = {node1_id, node2_id, node3_id}
        for kg_node in kg.nodes:
            assert kg_node.properties.get("entity_id") in expected_ids, (
                f"Node {kg_node.id} is missing entity_id in properties: {kg_node.properties}"
            )

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
        assert edge_props is None, (
            f"Edge {node2_id} -> {node3_id} should have been deleted"
        )

        # 8.1 Verify undirected graph property of edge deletion
        print(
            f"== Verifying undirected graph property of edge deletion: {node3_id} -> {node2_id}"
        )
        reverse_edge_props = await storage.get_edge(node3_id, node2_id)
        print(
            f"Querying reverse edge properties after deletion {node3_id} -> {node2_id}: {reverse_edge_props}"
        )
        assert reverse_edge_props is None, (
            f"Reverse edge {node3_id} -> {node2_id} should also be deleted, undirected graph property verification failed"
        )
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

    except Exception as e:
        _report_and_reraise(e)


@pytest.mark.integration
@pytest.mark.requires_db
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
        assert nodes_dict[node1_id]["description"] == node1_data["description"], (
            f"{node1_id} description mismatch"
        )
        assert nodes_dict[node2_id]["description"] == node2_data["description"], (
            f"{node2_id} description mismatch"
        )
        assert nodes_dict[node3_id]["description"] == node3_data["description"], (
            f"{node3_id} description mismatch"
        )

        # 3. Test node_degrees_batch - batch get degrees of multiple nodes
        print("== Testing node_degrees_batch")
        node_degrees = await storage.node_degrees_batch(node_ids)
        print(f"Batch get node degrees result: {node_degrees}")
        assert len(node_degrees) == 3, (
            f"Should return degrees of 3 nodes, but got {len(node_degrees)}"
        )
        assert node1_id in node_degrees, f"{node1_id} should be in the result"
        assert node2_id in node_degrees, f"{node2_id} should be in the result"
        assert node3_id in node_degrees, f"{node3_id} should be in the result"
        assert node_degrees[node1_id] == 3, (
            f"Degree of {node1_id} should be 3, but got {node_degrees[node1_id]}"
        )
        assert node_degrees[node2_id] == 2, (
            f"Degree of {node2_id} should be 2, but got {node_degrees[node2_id]}"
        )
        assert node_degrees[node3_id] == 3, (
            f"Degree of {node3_id} should be 3, but got {node_degrees[node3_id]}"
        )

        # 4. Test edge_degrees_batch - batch get degrees of multiple edges
        print("== Testing edge_degrees_batch")
        edges = [(node1_id, node2_id), (node2_id, node3_id), (node3_id, node4_id)]
        edge_degrees = await storage.edge_degrees_batch(edges)
        print(f"Batch get edge degrees result: {edge_degrees}")
        assert len(edge_degrees) == 3, (
            f"Should return degrees of 3 edges, but got {len(edge_degrees)}"
        )
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
        assert edge_degrees[(node1_id, node2_id)] == 5, (
            f"Degree of edge {node1_id} -> {node2_id} should be 5, but got {edge_degrees[(node1_id, node2_id)]}"
        )
        assert edge_degrees[(node2_id, node3_id)] == 5, (
            f"Degree of edge {node2_id} -> {node3_id} should be 5, but got {edge_degrees[(node2_id, node3_id)]}"
        )
        assert edge_degrees[(node3_id, node4_id)] == 5, (
            f"Degree of edge {node3_id} -> {node4_id} should be 5, but got {edge_degrees[(node3_id, node4_id)]}"
        )

        # 5. Test get_edges_batch - batch get properties of multiple edges
        print("== Testing get_edges_batch")
        # Convert list of tuples to list of dicts for Neo4j style
        edge_dicts = [{"src": src, "tgt": tgt} for src, tgt in edges]
        edges_dict = await storage.get_edges_batch(edge_dicts)
        print(f"Batch get edge properties result: {edges_dict.keys()}")
        assert len(edges_dict) == 3, (
            f"Should return properties of 3 edges, but got {len(edges_dict)}"
        )
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
        assert len(reverse_edges_dict) == 3, (
            f"Should return properties of 3 reverse edges, but got {len(reverse_edges_dict)}"
        )

        # Verify that properties of forward and reverse edges are consistent
        for (src, tgt), props in edges_dict.items():
            assert (
                tgt,
                src,
            ) in reverse_edges_dict, (
                f"Reverse edge {tgt} -> {src} should be in the result"
            )
            assert props == reverse_edges_dict[(tgt, src)], (
                f"Properties of edge {src} -> {tgt} and reverse edge {tgt} -> {src} are inconsistent"
            )

        print(
            "Undirected graph property verification successful: properties of batch-retrieved forward and reverse edges are consistent"
        )

        # 6. Test get_nodes_edges_batch - batch get all edges of multiple nodes
        print("== Testing get_nodes_edges_batch")
        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node3_id])
        print(f"Batch get node edges result: {nodes_edges.keys()}")
        assert len(nodes_edges) == 2, (
            f"Should return edges for 2 nodes, but got {len(nodes_edges)}"
        )
        assert node1_id in nodes_edges, f"{node1_id} should be in the result"
        assert node3_id in nodes_edges, f"{node3_id} should be in the result"
        assert len(nodes_edges[node1_id]) == 3, (
            f"{node1_id} should have 3 edges, but has {len(nodes_edges[node1_id])}"
        )
        assert len(nodes_edges[node3_id]) == 3, (
            f"{node3_id} should have 3 edges, but has {len(nodes_edges[node3_id])}"
        )

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

        assert has_edge_to_node2, (
            f"Edge list of node {node1_id} should include an edge to {node2_id}"
        )
        assert has_edge_to_node4, (
            f"Edge list of node {node1_id} should include an edge to {node4_id}"
        )
        assert has_edge_to_node5, (
            f"Edge list of node {node1_id} should include an edge to {node5_id}"
        )

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

        assert has_connection_with_node2, (
            f"Edge list of node {node3_id} should include a connection with {node2_id}"
        )
        assert has_connection_with_node4, (
            f"Edge list of node {node3_id} should include a connection with {node4_id}"
        )
        assert has_connection_with_node5, (
            f"Edge list of node {node3_id} should include a connection with {node5_id}"
        )

        print(
            "Undirected graph property verification successful: batch-retrieved node edges include all relevant edges (regardless of direction)"
        )

        print("\nBatch operations tests completed.")

    except Exception as e:
        _report_and_reraise(e)


@pytest.mark.integration
@pytest.mark.requires_db
async def test_graph_batch_upsert(storage):
    """
    Test the batch *write* paths end-to-end against the configured backend:
    1. upsert_nodes_batch inserts many nodes (forced across multiple chunks on
       backends that chunk by record count, e.g. PGGraphStorage).
    2. Same-batch last-write-wins dedup for nodes.
    3. has_node / has_nodes_batch existence checks.
    4. upsert_edges_batch inserts many edges, including a reciprocal duplicate
       that must collapse to the last write (undirected last-write-wins).
    5. Read-back of node/edge properties, degrees, and undirected consistency.

    These paths are otherwise only covered by mock unit tests; this is the only
    place they run against a real graph backend.
    """
    try:
        # Force >1 chunk on backends that chunk batch upserts by record count
        # (PGGraphStorage). Other backends simply ignore this attribute.
        if hasattr(storage, "_max_upsert_records_per_batch"):
            storage._max_upsert_records_per_batch = 2

        # 1. Batch node upsert with a same-batch duplicate (last write wins).
        nodes = [
            ("E1", {"entity_id": "E1", "description": "first", "entity_type": "T"}),
            ("E2", {"entity_id": "E2", "description": "second", "entity_type": "T"}),
            ("E3", {"entity_id": "E3", "description": "third", "entity_type": "T"}),
            ("E4", {"entity_id": "E4", "description": "fourth", "entity_type": "T"}),
            ("E5", {"entity_id": "E5", "description": "fifth", "entity_type": "T"}),
            # duplicate of E1 later in the same batch -> must keep this payload
            (
                "E1",
                {"entity_id": "E1", "description": "first-updated", "entity_type": "T"},
            ),
        ]
        print("== Testing upsert_nodes_batch (multi-chunk + same-batch dedup)")
        await storage.upsert_nodes_batch(nodes)

        # 2. All five distinct nodes exist; has_node / has_nodes_batch.
        for nid in ["E1", "E2", "E3", "E4", "E5"]:
            assert await storage.has_node(nid), (
                f"{nid} should exist after upsert_nodes_batch"
            )
        existing = await storage.has_nodes_batch(["E1", "E3", "E5", "DOES_NOT_EXIST"])
        assert existing == {
            "E1",
            "E3",
            "E5",
        }, f"has_nodes_batch returned unexpected set: {existing}"

        # Last-write-wins: the second E1 in the batch wins.
        e1 = await storage.get_node("E1")
        assert e1 is not None and e1["description"] == "first-updated", (
            "Same-batch node dedup should keep the last write"
        )

        # Batch read-back of the rest.
        nodes_dict = await storage.get_nodes_batch(["E2", "E3", "E4", "E5"])
        assert set(nodes_dict) == {"E2", "E3", "E4", "E5"}
        assert nodes_dict["E3"]["description"] == "third"

        # 3. Batch edge upsert with a reciprocal duplicate (undirected dedup).
        edges = [
            ("E1", "E2", {"relationship": "r12", "weight": 1.0, "description": "d12"}),
            ("E2", "E3", {"relationship": "r23", "weight": 1.0, "description": "d23"}),
            ("E3", "E4", {"relationship": "r34", "weight": 1.0, "description": "d34"}),
            ("E4", "E5", {"relationship": "r45", "weight": 1.0, "description": "d45"}),
            # reciprocal of (E1, E2): undirected -> last write wins
            (
                "E2",
                "E1",
                {
                    "relationship": "r12-updated",
                    "weight": 2.0,
                    "description": "d12-updated",
                },
            ),
        ]
        print("== Testing upsert_edges_batch (multi-chunk + reciprocal dedup)")
        await storage.upsert_edges_batch(edges)

        # 4. Every edge readable in both directions (undirected).
        for a, b in [("E1", "E2"), ("E2", "E3"), ("E3", "E4"), ("E4", "E5")]:
            fwd = await storage.get_edge(a, b)
            rev = await storage.get_edge(b, a)
            assert fwd is not None, f"Edge {a}->{b} missing after upsert_edges_batch"
            assert rev is not None, f"Reverse edge {b}->{a} missing (undirected)"
            assert fwd == rev, f"Edge {a}<->{b} not undirected-consistent"

        # Reciprocal dedup last-write-wins on (E1, E2).
        e12 = await storage.get_edge("E1", "E2")
        assert e12["relationship"] == "r12-updated", "Reciprocal edge dedup lost"
        assert e12["weight"] == 2.0, "Reciprocal edge dedup kept the wrong weight"

        # 5. Degrees reflect exactly the four distinct edges.
        assert await storage.node_degree("E1") == 1
        assert await storage.node_degree("E2") == 2
        assert await storage.node_degree("E3") == 2
        assert await storage.node_degree("E5") == 1

        print("\nBatch upsert tests completed.")

    except Exception as e:
        _report_and_reraise(e)


@pytest.mark.integration
@pytest.mark.requires_db
async def test_graph_query_helpers(storage):
    """
    Cover the whole-graph query helpers that the other tests don't touch:
    1. get_all_nodes  - every node as a dict carrying its "id".
    2. get_all_edges  - every edge as a dict carrying "source"/"target".
    3. get_popular_labels - labels ordered by degree (highest first), INCLUDING
       isolated (degree-0) entities, ties broken on the label ascending.
    4. search_labels  - substring/fuzzy label search.
    5. get_node_edges - None for a node that does not exist, [] for one that
       exists with no relations.
    """
    try:
        # Star topology so degrees are distinct: Alpha=3, others=1. "Orphan" and
        # "Aardvark" stay unconnected (degree 0) to pin two things at once:
        #   * isolated entities are still ranked -- a backend deriving degrees
        #     from its edge store has no row for them, and joining/aggregating
        #     from that side alone drops them;
        #   * their tie is broken on the LABEL, not on insertion order --
        #     "Aardvark" is inserted last on purpose, so a backend that keeps
        #     insertion order for ties returns it after "Orphan".
        node_ids = ["Alpha", "Beta", "Gamma", "Alphabet", "Orphan", "Aardvark"]
        for nid in node_ids:
            await storage.upsert_node(
                nid,
                {
                    "entity_id": nid,
                    "description": f"desc of {nid}",
                    "entity_type": "T",
                },
            )
        star_edges = [("Alpha", "Beta"), ("Alpha", "Gamma"), ("Alpha", "Alphabet")]
        for src, tgt in star_edges:
            await storage.upsert_edge(
                src, tgt, {"relationship": "rel", "weight": 1.0, "description": "d"}
            )

        # 1. get_all_nodes
        print("== Testing get_all_nodes")
        all_nodes = await storage.get_all_nodes()
        assert isinstance(all_nodes, list)
        ids = {n.get("id") for n in all_nodes}
        assert ids == set(node_ids), f"get_all_nodes ids mismatch: {ids}"

        # 2. get_all_edges (undirected: compare unordered endpoint pairs)
        print("== Testing get_all_edges")
        all_edges = await storage.get_all_edges()
        assert isinstance(all_edges, list)
        assert len(all_edges) == 3, (
            f"get_all_edges should return 3 edges, got {len(all_edges)}"
        )
        edge_pairs = {frozenset((e["source"], e["target"])) for e in all_edges}
        assert edge_pairs == {frozenset(p) for p in star_edges}

        # 3. get_popular_labels - highest degree first
        print("== Testing get_popular_labels")
        popular = await storage.get_popular_labels(limit=2)
        assert isinstance(popular, list)
        assert len(popular) <= 2
        assert popular and popular[0] == "Alpha", (
            f"highest-degree label should be 'Alpha', got {popular}"
        )

        # 3.1 Isolated entities must still be ranked (last, at degree 0) rather
        # than excluded — they are entities the user can select in the WebUI —
        # and their tie must break on the label, not on insertion order. The
        # tie-break is what decides which labels survive a `limit`, so getting
        # it wrong silently hides entities from the picker.
        all_popular = await storage.get_popular_labels(limit=len(node_ids) + 5)
        assert set(all_popular) == set(node_ids), (
            f"get_popular_labels must rank every node, got {all_popular}"
        )
        assert all_popular[0] == "Alpha"
        assert all_popular[-2:] == ["Aardvark", "Orphan"], (
            "degree-0 nodes should rank last, ordered by label ascending "
            f"(NOT insertion order), got {all_popular}"
        )

        # 4. search_labels - substring / prefix match, and a clear miss
        print("== Testing search_labels")
        gamma_hits = await storage.search_labels("Gam")
        assert "Gamma" in gamma_hits, f"search 'Gam' should find 'Gamma': {gamma_hits}"
        alpha_hits = await storage.search_labels("Alpha")
        assert "Alpha" in alpha_hits, (
            f"search 'Alpha' should find 'Alpha': {alpha_hits}"
        )
        misses = await storage.search_labels("NoSuchEntityXYZ")
        assert "Alpha" not in misses and "Gamma" not in misses

        # 5. get_node_edges - "no such node" and "node with no edges" are
        # different answers: None vs []. A backend that returns an empty list
        # for both makes a deleted entity indistinguishable from an isolated
        # one. (A backend ERROR is neither value — it must raise.)
        print("== Testing get_node_edges on absent vs isolated nodes")
        assert await storage.get_node_edges("NoSuchEntityXYZ") is None, (
            "get_node_edges must return None for a node that does not exist"
        )
        assert await storage.get_node_edges("Orphan") == [], (
            "get_node_edges must return [] for an existing node with no edges"
        )

        print("\nQuery helper tests completed.")

    except Exception as e:
        _report_and_reraise(e)


@pytest.mark.integration
@pytest.mark.requires_db
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
                assert node_props.get("entity_id") == node_id, (
                    f"Node ID mismatch: expected {node_id}, got {node_props.get('entity_id')}"
                )

                # Verify description is saved correctly
                assert node_props.get("description") == original_data["description"], (
                    f"Node description mismatch: expected {original_data['description']}, got {node_props.get('description')}"
                )

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
            assert edge1_props.get("relationship") == edge1_data["relationship"], (
                f"Edge relationship mismatch: expected {edge1_data['relationship']}, got {edge1_props.get('relationship')}"
            )

            # Verify edge description is saved correctly
            assert edge1_props.get("description") == edge1_data["description"], (
                f"Edge description mismatch: expected {edge1_data['description']}, got {edge1_props.get('description')}"
            )

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
            assert edge2_props.get("relationship") == edge2_data["relationship"], (
                f"Edge relationship mismatch: expected {edge2_data['relationship']}, got {edge2_props.get('relationship')}"
            )

            # Verify edge description is saved correctly
            assert edge2_props.get("description") == edge2_data["description"], (
                f"Edge description mismatch: expected {edge2_data['description']}, got {edge2_props.get('description')}"
            )

            print(
                f"Edge {node2_id} -> {node3_id} special character verification successful"
            )
        else:
            print(f"Failed to read edge properties: {node2_id} -> {node3_id}")
            assert False, f"Failed to read edge properties: {node2_id} -> {node3_id}"

        print("\nSpecial character tests completed, data is preserved in the database.")

    except Exception as e:
        _report_and_reraise(e)


@pytest.mark.integration
@pytest.mark.requires_db
async def test_graph_string_escaping_regressions(storage):
    """
    Regression coverage for entity IDs and properties that require Cypher escaping.

    Covers quoted and backslash-heavy node IDs across single-node reads, batch reads,
    edge retrieval, and delete/remove write paths.
    """
    center_id = 'Danh mục "bài toán lớn"'
    backslash_id = r"C:\Program Files\LightRAG"
    mixed_id = 'Path "C:\\RAG\\docs"'
    single_quote_id = "Node with 'single quotes'"

    node_payloads = {
        center_id: {
            "entity_id": center_id,
            "description": 'Quoted entity with JSON-ish payload {"path": "C:\\\\temp"}',
            "keywords": 'quotes,"double quotes",unicode',
            "entity_type": "Regression Node",
        },
        backslash_id: {
            "entity_id": backslash_id,
            "description": r"Windows path C:\Program Files\LightRAG\bin",
            "keywords": r"paths,C:\temp,backslashes",
            "entity_type": "Regression Node",
        },
        mixed_id: {
            "entity_id": mixed_id,
            "description": 'Mixed quotes "and" slashes \\ in one entity id',
            "keywords": r'mixed,"quoted",C:\RAG\docs',
            "entity_type": "Regression Node",
        },
        single_quote_id: {
            "entity_id": single_quote_id,
            "description": "Single quotes stay literal in entity identifiers",
            "keywords": "single quotes,escaping",
            "entity_type": "Regression Node",
        },
    }

    for node_id, payload in node_payloads.items():
        await storage.upsert_node(node_id, payload)

    edge_payloads = {
        (center_id, backslash_id): {
            "relationship": r'contains "path"\edge',
            "weight": 1.0,
            "description": r'Links "quoted" title to C:\Program Files\LightRAG',
        },
        (center_id, mixed_id): {
            "relationship": 'references "docs"',
            "weight": 0.8,
            "description": r'Contains both "quotes" and \\backslashes\\',
        },
        (center_id, single_quote_id): {
            "relationship": "mentions 'alias'",
            "weight": 0.6,
            "description": 'Single quote entity linked to "quoted" center node',
        },
    }

    for (src_id, tgt_id), payload in edge_payloads.items():
        await storage.upsert_edge(src_id, tgt_id, payload)

    for node_id, payload in node_payloads.items():
        node = await storage.get_node(node_id)
        assert node is not None, f"Expected node {node_id!r} to round-trip"
        assert node["entity_id"] == node_id
        assert node["description"] == payload["description"]

    nodes_batch = await storage.get_nodes_batch(list(node_payloads))
    assert set(nodes_batch) == set(node_payloads)
    for node_id, payload in node_payloads.items():
        assert nodes_batch[node_id]["entity_id"] == node_id
        assert nodes_batch[node_id]["description"] == payload["description"]

    degrees = await storage.node_degrees_batch(list(node_payloads))
    assert degrees[center_id] == 3
    assert degrees[backslash_id] == 1
    assert degrees[mixed_id] == 1
    assert degrees[single_quote_id] == 1

    # Helper: undirected graph has no canonical direction, so accept either (a,b) or (b,a).
    def connects(edges, a, b):
        return any(
            (src == a and tgt == b) or (src == b and tgt == a) for src, tgt in edges
        )

    center_edges = await storage.get_node_edges(center_id)
    assert center_edges is not None
    assert connects(center_edges, center_id, backslash_id), (
        f"center_edges should contain connection to {backslash_id}"
    )
    assert connects(center_edges, center_id, mixed_id), (
        f"center_edges should contain connection to {mixed_id}"
    )
    assert connects(center_edges, center_id, single_quote_id), (
        f"center_edges should contain connection to {single_quote_id}"
    )

    batch_edges = await storage.get_nodes_edges_batch(
        [center_id, mixed_id, backslash_id, single_quote_id]
    )
    assert set(batch_edges) == {center_id, mixed_id, backslash_id, single_quote_id}
    assert connects(batch_edges[center_id], center_id, backslash_id)
    assert connects(batch_edges[center_id], center_id, mixed_id)
    assert connects(batch_edges[center_id], center_id, single_quote_id)
    assert connects(batch_edges[mixed_id], center_id, mixed_id)
    assert connects(batch_edges[backslash_id], center_id, backslash_id)
    assert connects(batch_edges[single_quote_id], center_id, single_quote_id)

    # --- Undirected property: get_edge in both directions ---
    print("\n== Verifying undirected property: get_edge forward and reverse")
    for (src_id, tgt_id), payload in edge_payloads.items():
        fwd = await storage.get_edge(src_id, tgt_id)
        rev = await storage.get_edge(tgt_id, src_id)
        assert fwd is not None, (
            f"get_edge({src_id!r}, {tgt_id!r}) returned None after insertion"
        )
        assert rev is not None, (
            f"get_edge({tgt_id!r}, {src_id!r}) returned None — "
            f"storage is not treating the edge as undirected"
        )
        assert fwd["relationship"] == payload["relationship"]
        assert fwd["description"] == payload["description"]
        assert rev["relationship"] == fwd["relationship"], (
            f"Reverse get_edge returned different relationship for "
            f"({src_id!r}, {tgt_id!r})"
        )
        assert rev["description"] == fwd["description"], (
            f"Reverse get_edge returned different description for "
            f"({src_id!r}, {tgt_id!r})"
        )
    print(
        "Undirected property verification successful: "
        "get_edge returns consistent data in both directions"
    )

    # --- Undirected property: has_edge in both directions ---
    print("\n== Verifying undirected property: has_edge forward and reverse")
    for src_id, tgt_id in edge_payloads:
        assert await storage.has_edge(src_id, tgt_id), (
            f"has_edge({src_id!r}, {tgt_id!r}) returned False after insertion"
        )
        assert await storage.has_edge(tgt_id, src_id), (
            f"has_edge({tgt_id!r}, {src_id!r}) returned False — "
            f"storage is not treating the edge as undirected"
        )
    print(
        "Undirected property verification successful: "
        "has_edge returns True in both directions"
    )

    # --- Undirected property: get_edges_batch forward and reverse ---
    print("\n== Verifying undirected property: get_edges_batch forward and reverse")
    forward_edges = await storage.get_edges_batch(
        [{"src": src_id, "tgt": tgt_id} for src_id, tgt_id in edge_payloads]
    )
    reverse_edges = await storage.get_edges_batch(
        [{"src": tgt_id, "tgt": src_id} for src_id, tgt_id in edge_payloads]
    )

    assert set(forward_edges) == set(edge_payloads)
    for pair, payload in edge_payloads.items():
        assert forward_edges[pair]["relationship"] == payload["relationship"]
        assert forward_edges[pair]["description"] == payload["description"]
        reverse_pair = (pair[1], pair[0])
        assert reverse_pair in reverse_edges, (
            f"get_edges_batch did not return reverse pair {reverse_pair!r}"
        )
        assert reverse_edges[reverse_pair]["relationship"] == payload["relationship"]
        assert reverse_edges[reverse_pair]["description"] == payload["description"]
    print(
        "Undirected property verification successful: "
        "get_edges_batch returns consistent data in both directions"
    )

    # --- Undirected property: edge deletion removes both directions ---
    print("\n== Verifying undirected property: edge deletion removes both directions")
    await storage.remove_edges([(center_id, mixed_id)])
    assert await storage.get_edge(center_id, mixed_id) is None, (
        f"Forward edge ({center_id!r} -> {mixed_id!r}) should be deleted"
    )
    assert await storage.get_edge(mixed_id, center_id) is None, (
        f"Reverse edge ({mixed_id!r} -> {center_id!r}) should also be deleted "
        f"— storage is not treating deletion as undirected"
    )
    remaining_center_edges = await storage.get_node_edges(center_id)
    assert remaining_center_edges is not None
    assert not connects(remaining_center_edges, center_id, mixed_id), (
        "Edge between center and mixed_id should have been removed"
    )
    print(
        "Undirected property verification successful: "
        "deleting an edge removes it in both directions"
    )

    await storage.delete_node(single_quote_id)
    assert await storage.get_node(single_quote_id) is None

    await storage.remove_nodes([center_id, backslash_id])
    assert await storage.get_node(center_id) is None
    assert await storage.get_node(backslash_id) is None
    assert await storage.get_node(mixed_id) is not None


@pytest.mark.integration
@pytest.mark.requires_db
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
        assert forward_edge is not None, (
            f"Failed to read forward edge properties: {node1_id} -> {node2_id}"
        )

        # Verify reverse query
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(f"Reverse edge properties: {reverse_edge}")
        assert reverse_edge is not None, (
            f"Failed to read reverse edge properties: {node2_id} -> {node1_id}"
        )

        # Verify that forward and reverse edge properties are consistent
        assert forward_edge == reverse_edge, (
            "Forward and reverse edge properties are inconsistent, undirected property verification failed"
        )
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
        assert forward_degree == reverse_degree, (
            "Degrees of forward and reverse edges are inconsistent, undirected property verification failed"
        )
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
        assert forward_edge is None, (
            f"Edge {node1_id} -> {node2_id} should have been deleted"
        )

        # Verify reverse edge is also deleted
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(
            f"Querying reverse edge properties after deletion {node2_id} -> {node1_id}: {reverse_edge}"
        )
        assert reverse_edge is None, (
            f"Reverse edge {node2_id} -> {node1_id} should also be deleted, undirected property verification failed"
        )
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
                tgt,
                src,
            ) in reverse_edges_dict, (
                f"Reverse edge {tgt} -> {src} should be in the result"
            )
            assert props == reverse_edges_dict[(tgt, src)], (
                f"Properties of edge {src} -> {tgt} and reverse edge {tgt} -> {src} are inconsistent"
            )

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

        assert has_edge_to_node2, (
            f"Edge list of node {node1_id} should include an edge to {node2_id}"
        )
        assert has_edge_to_node3, (
            f"Edge list of node {node1_id} should include an edge to {node3_id}"
        )

        # Check if node 2 has a connection with node 1
        has_edge_to_node1 = any(
            (src == node2_id and tgt == node1_id)
            or (src == node1_id and tgt == node2_id)
            for src, tgt in node2_edges
        )
        assert has_edge_to_node1, (
            f"Edge list of node {node2_id} should include a connection with {node1_id}"
        )

        print(
            "Undirected property verification successful: batch-retrieved node edges include all relevant edges (regardless of direction)"
        )

        print("\nUndirected property tests completed.")

    except Exception as e:
        _report_and_reraise(e)


# ---------------------------------------------------------------------------
# Command-line entry point
#
# CLI_TESTS maps a short CLI name to the test function itself, derived by
# introspection rather than hand-maintained — there is no second list that can
# drift out of sync with what pytest actually collects, so a test added to this
# module later is automatically reachable from the CLI too, with no extra step
# and nothing to check for staleness. main() hands the selection to
# pytest.main(), so both entry points still run through the same fixture.
# ---------------------------------------------------------------------------


def _cli_name(test_name: str) -> str:
    """Derive a short CLI name from a test function's name.

    Strips the "test_graph_" prefix shared by this module's contract tests
    (falling back to a bare "test_" strip for anything named differently) and
    swaps underscores for hyphens: test_graph_batch_upsert -> batch-upsert.
    """
    stem = test_name.removeprefix("test_graph_")
    if stem == test_name:
        stem = test_name.removeprefix("test_")
    return stem.replace("_", "-")


def _discover_cli_tests() -> dict[str, object]:
    """Every test_*(storage) function in this module, keyed by its CLI name.

    Taking the ``storage`` fixture — not a name prefix — is what makes a test
    one of the backend contract tests, so a differently-named test is still
    picked up.
    """
    candidates = {
        name: obj
        for name, obj in globals().items()
        if name.startswith("test_")
        and callable(obj)
        and "storage" in inspect.signature(obj).parameters
    }
    return {_cli_name(name): func for name, func in candidates.items()}


CLI_TESTS = _discover_cli_tests()


def _summary(func) -> str:
    """First line of a test's docstring, for --list."""
    doc = (func.__doc__ or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


def _confirm_backend(graph_storage_type: str, selected: list[str], skip: bool) -> bool:
    """Pause for a one-time confirmation naming the backend under test.

    Mirrors check_env_file()'s old isatty() guard: the prompt only fires when a
    human is actually watching a terminal. Under CI, a pipe, or -y/--yes it just
    prints the same line and returns True immediately — this must never become a
    second thing that can hang a non-interactive run, which is why the earlier
    input()-based menu had to go in the first place.
    """
    names = ", ".join(selected)
    if skip or not sys.stdin.isatty():
        # ASCIIColors runs printed text through rich-style markup parsing, where
        # a literal "[...]" is a (silently-dropped) markup tag, not text — so this
        # cannot follow the "Tests: ..." line below through the same brackets.
        ASCIIColors.magenta(f"\nTesting backend: {graph_storage_type} (tests: {names})")
        return True

    ASCIIColors.magenta(f"\nAbout to test backend: {graph_storage_type}")
    ASCIIColors.white(f"Tests: {names}")
    try:
        response = input("Press Enter to continue, or type 'n' to abort: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return response.strip().lower() not in ("n", "no")


def main(argv: list[str] | None = None) -> int:
    """Run the selected tests through pytest; return pytest's exit code.

    Non-interactive under CI/a pipe/a Makefile (stdin is not a TTY): no prompt,
    runs straight through. When a human is actually watching a terminal, pauses
    once up front so the backend under test is impossible to miss — see
    `_confirm_backend` — skippable with -y/--yes.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after a literal "--" is forwarded to pytest verbatim. Split it off
    # before argparse, which would otherwise swallow the separator and try to read
    # the pytest flags as test names.
    if "--" in argv:
        separator = argv.index("--")
        argv, forwarded = argv[:separator], argv[separator + 1 :]
    else:
        forwarded = []

    parser = argparse.ArgumentParser(
        prog="python tests/kg/test_graph_storage.py",
        description=(
            "Run the graph storage contract tests against the backend selected by "
            "LIGHTRAG_GRAPH_STORAGE. A thin wrapper around pytest: the tests run "
            "through the same fixture whichever entry point you use."
        ),
        epilog=(
            "examples:\n"
            "  %(prog)s                       run every test\n"
            "  %(prog)s basic advanced        run a subset\n"
            "  %(prog)s --list                list the test names\n"
            "  %(prog)s basic -- -x --tb=long forward extra args to pytest\n"
            "  %(prog)s -y                    skip the confirmation prompt\n"
            "  %(prog)s -- -s                 show each test's live print() output\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tests",
        nargs="*",
        metavar="NAME",
        help="tests to run (default: all). See --list for the names.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the available test names and exit",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (implied when stdin is not a TTY)",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, func in CLI_TESTS.items():
            ASCIIColors.white(f"  {name:<20} {func.__name__} — {_summary(func)}")
        return 0

    unknown = [name for name in args.tests if name not in CLI_TESTS]
    if unknown:
        parser.error(
            f"unknown test name(s): {', '.join(unknown)}\n"
            f"valid names: {', '.join(CLI_TESTS)}"
        )

    ASCIIColors.cyan("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            General Graph Storage Test Program                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    check_env_file()
    load_dotenv(dotenv_path=".env", override=False)

    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    ASCIIColors.white(
        f"Supported graph storage types: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
    )

    selected = args.tests or list(CLI_TESTS)
    if not _confirm_backend(graph_storage_type, selected, skip=args.yes):
        ASCIIColors.red("Aborted.")
        return 1

    # --run-integration is mandatory here: these tests carry the integration
    # marker, so without it conftest skips every one and the script would exit 0
    # having run nothing. No -s by default: pytest captures each test's print()
    # output and only replays it on failure, which is far less noisy than the
    # live firehose these old print()-heavy tests produce — pass "-- -s" to get
    # that firehose back.
    pytest_argv = [
        *(
            f"{os.path.abspath(__file__)}::{CLI_TESTS[name].__name__}"
            for name in selected
        ),
        "--run-integration",
        "-v",
        *forwarded,
    ]
    ASCIIColors.yellow(f"\n$ pytest {' '.join(pytest_argv)}\n")

    # The fixture drops the graph before and after each test, so no cleanup pass
    # is needed here.
    exit_code = pytest.main(pytest_argv)

    # Repeated at the end, next to the result, because pytest's own summary line
    # (and everything above it) says nothing about which backend just ran —
    # exactly the ambiguity the confirmation prompt above exists to head off.
    ASCIIColors.magenta(f"\nBackend tested: {graph_storage_type}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
