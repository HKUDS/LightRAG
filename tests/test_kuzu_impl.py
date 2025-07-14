import asyncio
import tempfile
import os
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv
from lightrag.kg.kuzu_impl import KuzuDBStorage
from lightrag.types import KnowledgeGraph

load_dotenv(dotenv_path=".env", override=False)


class TestKuzuDBStorage:
    """Test suite for KuzuDBStorage implementation"""

    @pytest.fixture
    async def storage(self):
        """Create a test storage instance with temporary database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["KUZU_DB_PATH"] = os.path.join(temp_dir, "test.db")

            # Mock embedding function
            embedding_func = MagicMock(return_value=[0.1, 0.2, 0.3])

            storage = KuzuDBStorage(
                namespace="test",
                global_config={"max_graph_nodes": 1000},
                embedding_func=embedding_func,
                workspace=os.environ.get("KUZU_WORKSPACE", "kuzudb"),
            )

            await storage.initialize()
            yield storage
            await storage.finalize()

    async def test_initialization(self, storage):
        """Test database initialization"""
        assert storage._db is not None
        assert storage._conn is not None
        # Get the actual workspace value rather than assuming from environment
        expected_workspace = storage.workspace or "base"
        assert storage._get_label() == expected_workspace

    async def test_node_operations(self, storage):
        """Test node creation, retrieval, and existence checking"""
        # Test node doesn't exist initially
        # assert await storage.has_node("test_node_1") == False
        if not await storage.has_node("test_node_1"):
            assert True
        else:
            assert False

        # Create a node
        node_data = {
            "entity_id": "test_node_1",
            "entity_type": "Person",
            "description": "A test person",
            "source_id": "chunk_1",
        }
        await storage.upsert_node("test_node_1", node_data)

        # Test node exists now
        # assert await storage.has_node("test_node_1") == True
        if await storage.has_node("test_node_1"):
            assert True
        else:
            assert False

        # Retrieve the node
        retrieved_node = await storage.get_node("test_node_1")
        assert retrieved_node is not None
        assert retrieved_node["entity_id"] == "test_node_1"
        assert retrieved_node["entity_type"] == "Person"
        assert retrieved_node["description"] == "A test person"

    async def test_edge_operations(self, storage):
        """Test edge creation, retrieval, and existence checking"""
        # Create two nodes first
        node1_data = {
            "entity_id": "node_1",
            "entity_type": "Person",
            "description": "First person",
            "source_id": "chunk_1",
        }
        node2_data = {
            "entity_id": "node_2",
            "entity_type": "Person",
            "description": "Second person",
            "source_id": "chunk_1",
        }

        await storage.upsert_node("node_1", node1_data)
        await storage.upsert_node("node_2", node2_data)

        # Test edge doesn't exist initially
        # assert await storage.has_edge("node_1", "node_2") == False
        if not await storage.has_edge("node_1", "node_2"):
            assert True
        else:
            assert False

        # Create an edge
        edge_data = {
            "weight": 0.8,
            "description": "knows",
            "keywords": "relationship",
            "source_id": "chunk_1",
        }
        await storage.upsert_edge("node_1", "node_2", edge_data)

        # Test edge exists now
        # assert await storage.has_edge("node_1", "node_2") == True
        if await storage.has_edge("node_1", "node_2"):
            assert True
        else:
            assert False

        # Retrieve the edge
        retrieved_edge = await storage.get_edge("node_1", "node_2")
        assert retrieved_edge is not None
        assert retrieved_edge["weight"] == 0.8
        assert retrieved_edge["description"] == "knows"

    async def test_node_degree(self, storage):
        """Test node degree calculation"""
        # Create nodes
        await storage.upsert_node(
            "center_node",
            {
                "entity_id": "center_node",
                "entity_type": "Person",
                "description": "Center node",
                "source_id": "chunk_1",
            },
        )

        await storage.upsert_node(
            "connected_node_1",
            {
                "entity_id": "connected_node_1",
                "entity_type": "Person",
                "description": "Connected node 1",
                "source_id": "chunk_1",
            },
        )

        await storage.upsert_node(
            "connected_node_2",
            {
                "entity_id": "connected_node_2",
                "entity_type": "Person",
                "description": "Connected node 2",
                "source_id": "chunk_1",
            },
        )

        # Initially degree should be 0
        assert await storage.node_degree("center_node") == 0

        # Add edges
        await storage.upsert_edge(
            "center_node",
            "connected_node_1",
            {"weight": 0.5, "description": "edge1", "source_id": "chunk_1"},
        )

        await storage.upsert_edge(
            "center_node",
            "connected_node_2",
            {"weight": 0.7, "description": "edge2", "source_id": "chunk_1"},
        )

        # Now degree should be 2
        degree = await storage.node_degree("center_node")
        assert degree == 2

    async def test_get_node_edges(self, storage):
        """Test retrieving edges for a node"""
        # Create test nodes
        node_data = {
            "entity_id": "main_node",
            "entity_type": "Person",
            "description": "Main node",
            "source_id": "chunk_1",
        }
        await storage.upsert_node("main_node", node_data)

        # Create connected nodes
        for i in range(3):
            await storage.upsert_node(
                f"connected_{i}",
                {
                    "entity_id": f"connected_{i}",
                    "entity_type": "Person",
                    "description": f"Connected node {i}",
                    "source_id": "chunk_1",
                },
            )

            # Create edge
            await storage.upsert_edge(
                "main_node",
                f"connected_{i}",
                {
                    "weight": 0.5 + i * 0.1,
                    "description": f"edge to {i}",
                    "source_id": "chunk_1",
                },
            )

        # Get edges for main node
        edges = await storage.get_node_edges("main_node")
        assert edges is not None
        assert len(edges) == 3

        # Check that all expected connections are present
        edge_targets = [edge[1] for edge in edges if edge[0] == "main_node"]
        edge_sources = [edge[0] for edge in edges if edge[1] == "main_node"]

        connected_nodes = set(edge_targets + edge_sources)
        expected_nodes = {"connected_0", "connected_1", "connected_2", "main_node"}
        assert connected_nodes.issubset(expected_nodes)

    async def test_batch_operations(self, storage):
        """Test batch retrieval operations"""
        # Create multiple nodes
        node_ids = ["batch_node_1", "batch_node_2", "batch_node_3"]
        for node_id in node_ids:
            await storage.upsert_node(
                node_id,
                {
                    "entity_id": node_id,
                    "entity_type": "Person",
                    "description": f"Batch node {node_id}",
                    "source_id": "chunk_1",
                },
            )

        # Test batch node retrieval
        nodes = await storage.get_nodes_batch(node_ids)
        assert len(nodes) == 3
        for node_id in node_ids:
            assert node_id in nodes
            assert nodes[node_id]["entity_id"] == node_id

        # Test batch degree retrieval
        degrees = await storage.node_degrees_batch(node_ids)
        assert len(degrees) == 3
        for node_id in node_ids:
            assert node_id in degrees
            assert degrees[node_id] == 0  # No edges created yet

    async def test_chunk_id_queries(self, storage):
        """Test querying by chunk IDs"""
        # Create nodes with specific chunk IDs
        chunk_id = "test_chunk_123"

        await storage.upsert_node(
            "chunk_node_1",
            {
                "entity_id": "chunk_node_1",
                "entity_type": "Person",
                "description": "Node from chunk",
                "source_id": chunk_id,
            },
        )

        await storage.upsert_node(
            "chunk_node_2",
            {
                "entity_id": "chunk_node_2",
                "entity_type": "Person",
                "description": "Another node from chunk",
                "source_id": chunk_id,
            },
        )

        # Create edge with same chunk ID
        await storage.upsert_edge(
            "chunk_node_1",
            "chunk_node_2",
            {"weight": 0.9, "description": "chunk edge", "source_id": chunk_id},
        )

        # Query nodes by chunk ID
        nodes = await storage.get_nodes_by_chunk_ids([chunk_id])
        assert len(nodes) >= 2

        # Check that nodes have the correct chunk ID
        for node in nodes:
            assert chunk_id in node["source_id"]

        # Query edges by chunk ID
        edges = await storage.get_edges_by_chunk_ids([chunk_id])
        assert len(edges) >= 1

        # Check that edges have the correct chunk ID
        for edge in edges:
            assert chunk_id in edge["source_id"]

    async def test_get_all_labels(self, storage):
        """Test retrieving all node labels"""
        # Create some test nodes
        test_nodes = ["label_test_1", "label_test_2", "label_test_3"]
        for node_id in test_nodes:
            await storage.upsert_node(
                node_id,
                {
                    "entity_id": node_id,
                    "entity_type": "TestType",
                    "description": f"Test node {node_id}",
                    "source_id": "chunk_1",
                },
            )

        # Get all labels
        labels = await storage.get_all_labels()

        # Check that our test nodes are in the labels
        for node_id in test_nodes:
            assert node_id in labels

    async def test_deletion_operations(self, storage):
        """Test node and edge deletion"""
        # Create test data
        await storage.upsert_node(
            "delete_node_1",
            {
                "entity_id": "delete_node_1",
                "entity_type": "Person",
                "description": "Node to delete",
                "source_id": "chunk_1",
            },
        )

        await storage.upsert_node(
            "delete_node_2",
            {
                "entity_id": "delete_node_2",
                "entity_type": "Person",
                "description": "Another node to delete",
                "source_id": "chunk_1",
            },
        )

        await storage.upsert_edge(
            "delete_node_1",
            "delete_node_2",
            {"weight": 0.5, "description": "edge to delete", "source_id": "chunk_1"},
        )

        # Verify they exist
        # assert await storage.has_node("delete_node_1") == True
        # assert await storage.has_node("delete_node_2") == True
        # assert await storage.has_edge("delete_node_1", "delete_node_2") == True
        if await storage.has_node("delete_node_1") and await storage.has_node(
            "delete_node_2"
        ):
            assert True
        else:
            assert False

        if await storage.has_edge("delete_node_1", "delete_node_2"):
            assert True
        else:
            assert False
        # Delete edge
        await storage.remove_edges([("delete_node_1", "delete_node_2")])
        # assert await storage.has_edge("delete_node_1", "delete_node_2") == False
        if not await storage.has_edge("delete_node_1", "delete_node_2"):
            assert True
        else:
            assert False

        # Delete nodes
        await storage.remove_nodes(["delete_node_1", "delete_node_2"])
        # assert await storage.has_node("delete_node_1") == False
        # assert await storage.has_node("delete_node_2") == False
        if not await storage.has_node("delete_node_1") and not await storage.has_node(
            "delete_node_2"
        ):
            assert True
        else:
            assert False

    async def test_knowledge_graph_retrieval(self, storage):
        """Test comprehensive knowledge graph retrieval scenarios"""

        # Test 1: Empty graph scenario
        kg_empty = await storage.get_knowledge_graph(
            "nonexistent_node", max_depth=2, max_nodes=10
        )
        assert isinstance(kg_empty, KnowledgeGraph)
        assert len(kg_empty.nodes) == 0
        assert len(kg_empty.edges) == 0
        assert kg_empty.is_truncated == False

        # Test 2: Create a complex connected graph with multiple depths
        # Layer 1: Central node
        await storage.upsert_node(
            "central_node",
            {
                "entity_id": "central_node",
                "entity_type": "Hub",
                "description": "Central hub node",
                "source_id": "chunk_central",
                "keywords": "hub,center",
            },
        )

        # Layer 2: Direct neighbors
        layer2_nodes = ["neighbor_a", "neighbor_b", "neighbor_c"]
        for node_id in layer2_nodes:
            await storage.upsert_node(
                node_id,
                {
                    "entity_id": node_id,
                    "entity_type": "Neighbor",
                    "description": f"Direct neighbor {node_id}",
                    "source_id": "chunk_layer2",
                    "keywords": f"neighbor,{node_id}",
                },
            )
            # Connect to central node
            await storage.upsert_edge(
                "central_node",
                node_id,
                {
                    "weight": 0.8,
                    "description": f"connects to {node_id}",
                    "source_id": "chunk_central",
                    "keywords": "direct",
                },
            )

        # Layer 3: Distant neighbors (only connected to layer 2)
        layer3_nodes = ["distant_x", "distant_y"]
        for i, node_id in enumerate(layer3_nodes):
            await storage.upsert_node(
                node_id,
                {
                    "entity_id": node_id,
                    "entity_type": "Distant",
                    "description": f"Distant node {node_id}",
                    "source_id": "chunk_layer3",
                    "keywords": f"distant,{node_id}",
                },
            )
            # Connect to one of the layer 2 nodes
            await storage.upsert_edge(
                layer2_nodes[i],
                node_id,
                {
                    "weight": 0.6,
                    "description": f"extends to {node_id}",
                    "source_id": "chunk_layer3",
                    "keywords": "distant",
                },
            )

        # Layer 4: Very distant node (should be excluded with depth=2)
        await storage.upsert_node(
            "very_distant",
            {
                "entity_id": "very_distant",
                "entity_type": "VeryDistant",
                "description": "Very distant node",
                "source_id": "chunk_layer4",
                "keywords": "very,distant",
            },
        )
        await storage.upsert_edge(
            "distant_x",
            "very_distant",
            {
                "weight": 0.3,
                "description": "very far connection",
                "source_id": "chunk_layer4",
                "keywords": "very_distant",
            },
        )

        # Isolated component (should not be included)
        isolated_nodes = ["isolated_1", "isolated_2"]
        for node_id in isolated_nodes:
            await storage.upsert_node(
                node_id,
                {
                    "entity_id": node_id,
                    "entity_type": "Isolated",
                    "description": f"Isolated node {node_id}",
                    "source_id": "chunk_isolated",
                    "keywords": f"isolated,{node_id}",
                },
            )
        await storage.upsert_edge(
            "isolated_1",
            "isolated_2",
            {
                "weight": 0.5,
                "description": "isolated connection",
                "source_id": "chunk_isolated",
                "keywords": "isolated",
            },
        )

        # Test 3: BFS traversal with depth limit
        kg_depth1 = await storage.get_knowledge_graph(
            "central_node", max_depth=1, max_nodes=10
        )
        assert isinstance(kg_depth1, KnowledgeGraph)

        # Should include central node + direct neighbors (depth 1)
        expected_nodes_depth1 = {
            "central_node",
            "neighbor_a",
            "neighbor_b",
            "neighbor_c",
        }
        actual_nodes_depth1 = {node.id for node in kg_depth1.nodes}
        assert expected_nodes_depth1.issubset(actual_nodes_depth1)

        # Should not include layer 3 nodes at depth 1
        assert "distant_x" not in actual_nodes_depth1
        assert "distant_y" not in actual_nodes_depth1

        # Verify edge count for depth 1 (3 edges from central to neighbors)
        assert len(kg_depth1.edges) >= 3

        # Test 4: BFS traversal with depth 2
        kg_depth2 = await storage.get_knowledge_graph(
            "central_node", max_depth=2, max_nodes=20
        )
        assert isinstance(kg_depth2, KnowledgeGraph)

        # Should include up to layer 3 nodes
        expected_nodes_depth2 = {
            "central_node",
            "neighbor_a",
            "neighbor_b",
            "neighbor_c",
            "distant_x",
            "distant_y",
        }
        actual_nodes_depth2 = {node.id for node in kg_depth2.nodes}
        assert expected_nodes_depth2.issubset(actual_nodes_depth2)

        # Should not include very distant node at depth 2
        assert "very_distant" not in actual_nodes_depth2

        # Should not include isolated nodes
        assert "isolated_1" not in actual_nodes_depth2
        assert "isolated_2" not in actual_nodes_depth2

        # Test 5: Node limit enforcement
        kg_limited = await storage.get_knowledge_graph(
            "central_node", max_depth=3, max_nodes=3
        )
        assert isinstance(kg_limited, KnowledgeGraph)
        assert len(kg_limited.nodes) <= 3
        # Should be truncated due to node limit
        assert kg_limited.is_truncated == True

        # Test 6: Verify node and edge properties are preserved
        kg_full = await storage.get_knowledge_graph(
            "central_node", max_depth=2, max_nodes=10
        )

        # Find central node and verify its properties
        central_node_kg = next(
            (node for node in kg_full.nodes if node.id == "central_node"), None
        )
        assert central_node_kg is not None
        assert central_node_kg.properties["entity_type"] == "Hub"
        assert central_node_kg.properties["description"] == "Central hub node"
        assert central_node_kg.properties["source_id"] == "chunk_central"
        assert central_node_kg.properties["keywords"] == "hub,center"

        # Find an edge and verify its properties
        central_to_neighbor_edge = next(
            (
                edge
                for edge in kg_full.edges
                if (edge.source == "central_node" and edge.target in layer2_nodes)
                or (edge.target == "central_node" and edge.source in layer2_nodes)
            ),
            None,
        )
        assert central_to_neighbor_edge is not None
        assert central_to_neighbor_edge.properties["weight"] == 0.8
        assert "connects to" in central_to_neighbor_edge.properties["description"]
        assert central_to_neighbor_edge.properties["source_id"] == "chunk_central"
        assert central_to_neighbor_edge.properties["keywords"] == "direct"
        assert central_to_neighbor_edge.type == "UNDIRECTED"

        # Test 7: Wildcard retrieval ("*")
        kg_wildcard = await storage.get_knowledge_graph("*", max_nodes=5)
        assert isinstance(kg_wildcard, KnowledgeGraph)
        assert len(kg_wildcard.nodes) <= 5
        # Should return nodes with highest degree first
        assert len(kg_wildcard.nodes) > 0

        # The central node should be included as it has the highest degree
        wildcard_node_ids = {node.id for node in kg_wildcard.nodes}
        assert "central_node" in wildcard_node_ids

        # Test 8: Edge uniqueness and bidirectionality handling
        # Verify that bidirectional edges are represented as single undirected edges
        kg_edge_test = await storage.get_knowledge_graph(
            "central_node", max_depth=1, max_nodes=10
        )
        edge_pairs = set()
        for edge in kg_edge_test.edges:
            # Normalize edge representation
            edge_pair = tuple(sorted([edge.source, edge.target]))
            assert edge_pair not in edge_pairs, f"Duplicate edge found: {edge_pair}"
            edge_pairs.add(edge_pair)

        # Test 9: Different starting points should yield different but overlapping results
        kg_from_neighbor = await storage.get_knowledge_graph(
            "neighbor_a", max_depth=2, max_nodes=10
        )
        assert isinstance(kg_from_neighbor, KnowledgeGraph)

        # Should include neighbor_a as starting point
        neighbor_node_ids = {node.id for node in kg_from_neighbor.nodes}
        assert "neighbor_a" in neighbor_node_ids

        # Should reach central node and other neighbors through depth-2 traversal
        assert "central_node" in neighbor_node_ids

        # Test 10: Verify graph connectivity
        # All returned nodes should be reachable from the starting node
        def is_connected(nodes, edges, start_node_id):
            """Verify all nodes are reachable from start_node via BFS"""
            if not nodes:
                return True

            # Build adjacency list
            adjacency = {node.id: set() for node in nodes}
            for edge in edges:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)

            # BFS to check connectivity
            visited = set()
            queue = [start_node_id]
            visited.add(start_node_id)

            while queue:
                current = queue.pop(0)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            node_ids = {node.id for node in nodes}
            return visited == node_ids

        kg_connectivity = await storage.get_knowledge_graph(
            "central_node", max_depth=2, max_nodes=10
        )
        assert is_connected(
            kg_connectivity.nodes, kg_connectivity.edges, "central_node"
        )

        print("✓ All knowledge graph retrieval tests passed comprehensively!")

    async def test_drop_operation(self, storage):
        """Test dropping all data"""
        # Create some test data
        await storage.upsert_node(
            "drop_test_node",
            {
                "entity_id": "drop_test_node",
                "entity_type": "TestNode",
                "description": "Node for drop test",
                "source_id": "chunk_1",
            },
        )

        # Verify data exists
        # assert await storage.has_node("drop_test_node") == True
        if await storage.has_node("drop_test_node"):
            assert True
        else:
            assert False

        # Drop all data
        result = await storage.drop()
        assert result["status"] == "success"

        # Verify data is gone
        # assert await storage.has_node("drop_test_node") == False
        if not await storage.has_node("drop_test_node"):
            assert True
        else:
            assert False


# Run the tests
async def run_tests():
    """Run all tests"""
    test_instance = TestKuzuDBStorage()

    # Create storage fixture
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["KUZU_DB_PATH"] = os.path.join(temp_dir, "test.db")

        # Mock embedding function
        def embedding_func(x):
            return [0.1, 0.2, 0.3]

        storage = KuzuDBStorage(
            namespace="test",
            global_config={"max_graph_nodes": 1000},
            embedding_func=embedding_func,
            workspace="test_workspace",
        )

        await storage.initialize()

        try:
            print("Running KuzuDB tests...")

            # Run tests
            await test_instance.test_initialization(storage)
            print("✓ Initialization test passed")

            await test_instance.test_node_operations(storage)
            print("✓ Node operations test passed")

            await test_instance.test_edge_operations(storage)
            print("✓ Edge operations test passed")

            await test_instance.test_node_degree(storage)
            print("✓ Node degree test passed")

            await test_instance.test_get_node_edges(storage)
            print("✓ Get node edges test passed")

            await test_instance.test_batch_operations(storage)
            print("✓ Batch operations test passed")

            await test_instance.test_chunk_id_queries(storage)
            print("✓ Chunk ID queries test passed")

            await test_instance.test_get_all_labels(storage)
            print("✓ Get all labels test passed")

            await test_instance.test_deletion_operations(storage)
            print("✓ Deletion operations test passed")

            await test_instance.test_knowledge_graph_retrieval(storage)
            print("✓ Knowledge graph retrieval test passed")

            await test_instance.test_drop_operation(storage)
            print("✓ Drop operation test passed")

            print("\nAll tests passed! ✅")

        finally:
            await storage.finalize()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests())
