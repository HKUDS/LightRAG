#!/usr/bin/env python
"""
Test script to verify KuzuDB integration with test_graph_storage.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from lightrag.kg.kuzu_impl import KuzuDBStorage
from lightrag.types import KnowledgeGraph
import numpy as np


# Mock embedding function
async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10)


async def test_kuzu_integration():
    """Test basic KuzuDB integration functionality"""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up KuzuDB environment
        kuzu_db_path = os.path.join(temp_dir, "test_kuzu.db")
        os.environ["KUZU_DB_PATH"] = kuzu_db_path
        os.environ["KUZU_WORKSPACE"] = "test_workspace"

        # Initialize KuzuDB storage
        storage = KuzuDBStorage(
            namespace="test_graph",
            global_config={"max_graph_nodes": 1000},
            embedding_func=mock_embedding_func,
            workspace="test_workspace",
        )

        try:
            # Initialize connection
            await storage.initialize()
            print("✓ KuzuDB initialization successful")

            # Test basic node operations
            node_id = "test_node"
            node_data = {
                "entity_id": node_id,
                "entity_type": "Test",
                "description": "A test node",
                "source_id": "chunk_1",
            }

            await storage.upsert_node(node_id, node_data)
            print("✓ Node insertion successful")

            # Test node retrieval
            retrieved_node = await storage.get_node(node_id)
            assert retrieved_node is not None
            assert retrieved_node["entity_id"] == node_id
            print("✓ Node retrieval successful")

            # Test edge operations
            node2_id = "test_node_2"
            node2_data = {
                "entity_id": node2_id,
                "entity_type": "Test",
                "description": "Another test node",
                "source_id": "chunk_1",
            }

            await storage.upsert_node(node2_id, node2_data)

            edge_data = {
                "weight": 0.8,
                "description": "test relationship",
                "keywords": "test",
                "source_id": "chunk_1",
            }

            await storage.upsert_edge(node_id, node2_id, edge_data)
            print("✓ Edge insertion successful")

            # Test edge retrieval
            retrieved_edge = await storage.get_edge(node_id, node2_id)
            assert retrieved_edge is not None
            assert retrieved_edge["weight"] == 0.8
            print("✓ Edge retrieval successful")

            # Test knowledge graph retrieval
            kg = await storage.get_knowledge_graph(node_id, max_depth=2, max_nodes=10)
            assert isinstance(kg, KnowledgeGraph)
            assert len(kg.nodes) > 0
            print("✓ Knowledge graph retrieval successful")

            # Test cleanup
            await storage.drop()
            print("✓ Database cleanup successful")

            print("\n🎉 All KuzuDB integration tests passed!")

        finally:
            await storage.finalize()


if __name__ == "__main__":
    asyncio.run(test_kuzu_integration())
