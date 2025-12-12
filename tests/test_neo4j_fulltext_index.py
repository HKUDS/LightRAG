#!/usr/bin/env python
"""
Test Neo4j full-text index functionality, specifically:
1. Workspace-specific index naming
2. Legacy index migration
3. search_labels functionality with workspace-specific indexes
"""

import asyncio
import os
import sys
import pytest
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.kg.shared_storage import initialize_share_data


# Mock embedding function that returns random vectors
async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10)  # Return 10-dimensional random vectors


@pytest.fixture
async def neo4j_storage():
    """
    Initialize Neo4j storage for testing.
    Requires Neo4j to be running and configured via environment variables.
    """
    # Check if Neo4j is configured
    if not os.getenv("NEO4J_URI"):
        pytest.skip("Neo4j not configured (NEO4J_URI not set)")

    from lightrag.kg.neo4j_impl import Neo4JStorage

    # Initialize shared_storage for locks
    initialize_share_data()

    global_config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),
    }

    storage = Neo4JStorage(
        namespace="test_fulltext_index",
        workspace="test_workspace",
        global_config=global_config,
        embedding_func=mock_embedding_func,
    )

    # Initialize the connection
    await storage.initialize()

    # Clean up any existing data
    await storage.drop()

    yield storage

    # Cleanup
    await storage.drop()
    await storage.finalize()


@pytest.mark.integration
@pytest.mark.requires_db
async def test_fulltext_index_creation(neo4j_storage):
    """
    Test that the full-text index is created with the workspace-specific name.
    """
    storage = neo4j_storage
    workspace_label = storage._get_workspace_label()
    expected_index_name = f"entity_id_fulltext_idx_{workspace_label}"

    # Query Neo4j to check if the index exists
    async with storage._driver.session(database=storage._DATABASE) as session:
        result = await session.run("SHOW FULLTEXT INDEXES")
        indexes = await result.data()
        await result.consume()

        # Check if the workspace-specific index exists
        index_names = [idx["name"] for idx in indexes]
        assert (
            expected_index_name in index_names
        ), f"Expected index '{expected_index_name}' not found. Found indexes: {index_names}"

        # Check if the legacy index doesn't exist (should be migrated if it was there)
        legacy_index_name = "entity_id_fulltext_idx"
        if legacy_index_name in index_names:
            # If legacy index exists, it should be for a different workspace
            # or it means migration didn't happen
            print(
                f"Warning: Legacy index '{legacy_index_name}' still exists alongside '{expected_index_name}'"
            )


@pytest.mark.integration
@pytest.mark.requires_db
async def test_search_labels_with_workspace_index(neo4j_storage):
    """
    Test that search_labels uses the workspace-specific index and returns results.
    """
    storage = neo4j_storage

    # Insert test nodes
    test_nodes = [
        {
            "entity_id": "Artificial Intelligence",
            "description": "AI field",
            "keywords": "AI,ML,DL",
            "entity_type": "Technology",
        },
        {
            "entity_id": "Machine Learning",
            "description": "ML subfield",
            "keywords": "supervised,unsupervised",
            "entity_type": "Technology",
        },
        {
            "entity_id": "Deep Learning",
            "description": "DL subfield",
            "keywords": "neural networks",
            "entity_type": "Technology",
        },
        {
            "entity_id": "Natural Language Processing",
            "description": "NLP field",
            "keywords": "text,language",
            "entity_type": "Technology",
        },
    ]

    for node_data in test_nodes:
        await storage.upsert_node(node_data["entity_id"], node_data)

    # Give the index time to become consistent (eventually consistent index)
    await asyncio.sleep(2)

    # Test search_labels
    results = await storage.search_labels("Learning", limit=10)

    # Should find nodes with "Learning" in them
    assert len(results) > 0, "search_labels should return results for 'Learning'"
    assert any(
        "Learning" in result for result in results
    ), "Results should contain 'Learning'"

    # Test case-insensitive search
    results_lower = await storage.search_labels("learning", limit=10)
    assert len(results_lower) > 0, "search_labels should be case-insensitive"

    # Test partial match
    results_partial = await storage.search_labels("Intelli", limit=10)
    assert (
        len(results_partial) > 0
    ), "search_labels should support partial matching with wildcard"
    assert any(
        "Intelligence" in result for result in results_partial
    ), "Should find 'Artificial Intelligence'"


@pytest.mark.integration
@pytest.mark.requires_db
async def test_search_labels_chinese_text(neo4j_storage):
    """
    Test that search_labels works with Chinese text using the CJK analyzer.
    """
    storage = neo4j_storage

    # Insert Chinese test nodes
    chinese_nodes = [
        {
            "entity_id": "人工智能",
            "description": "人工智能领域",
            "keywords": "AI,机器学习",
            "entity_type": "技术",
        },
        {
            "entity_id": "机器学习",
            "description": "机器学习子领域",
            "keywords": "监督学习,无监督学习",
            "entity_type": "技术",
        },
        {
            "entity_id": "深度学习",
            "description": "深度学习子领域",
            "keywords": "神经网络",
            "entity_type": "技术",
        },
    ]

    for node_data in chinese_nodes:
        await storage.upsert_node(node_data["entity_id"], node_data)

    # Give the index time to become consistent
    await asyncio.sleep(2)

    # Test Chinese text search
    results = await storage.search_labels("学习", limit=10)

    # Should find nodes with "学习" in them
    assert len(results) > 0, "search_labels should return results for Chinese text"
    assert any(
        "学习" in result for result in results
    ), "Results should contain Chinese characters '学习'"


@pytest.mark.integration
@pytest.mark.requires_db
async def test_search_labels_fallback_to_contains(neo4j_storage):
    """
    Test that search_labels falls back to CONTAINS search if the index fails.
    This can happen with older Neo4j versions or if the index is not yet available.
    """
    storage = neo4j_storage

    # Insert test nodes
    test_nodes = [
        {
            "entity_id": "Test Node Alpha",
            "description": "Test node",
            "keywords": "test",
            "entity_type": "Test",
        },
        {
            "entity_id": "Test Node Beta",
            "description": "Test node",
            "keywords": "test",
            "entity_type": "Test",
        },
    ]

    for node_data in test_nodes:
        await storage.upsert_node(node_data["entity_id"], node_data)

    # Even if the full-text index is not available, CONTAINS should work
    results = await storage.search_labels("Alpha", limit=10)

    # Should find the node using fallback CONTAINS search
    assert len(results) > 0, "Fallback CONTAINS search should return results"
    assert "Test Node Alpha" in results, "Should find 'Test Node Alpha'"


@pytest.mark.integration
@pytest.mark.requires_db
async def test_multiple_workspaces_have_separate_indexes(neo4j_storage):
    """
    Test that different workspaces have their own separate indexes.
    """
    from lightrag.kg.neo4j_impl import Neo4JStorage

    # Create storage for workspace 1
    storage1 = neo4j_storage

    # Create storage for workspace 2
    global_config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),
    }

    storage2 = Neo4JStorage(
        namespace="test_fulltext_index",
        workspace="test_workspace_2",
        global_config=global_config,
        embedding_func=mock_embedding_func,
    )

    await storage2.initialize()
    await storage2.drop()

    try:
        # Check that both workspaces have their own indexes
        async with storage1._driver.session(database=storage1._DATABASE) as session:
            result = await session.run("SHOW FULLTEXT INDEXES")
            indexes = await result.data()
            await result.consume()

            index_names = [idx["name"] for idx in indexes]
            workspace1_index = (
                f"entity_id_fulltext_idx_{storage1._get_workspace_label()}"
            )
            workspace2_index = (
                f"entity_id_fulltext_idx_{storage2._get_workspace_label()}"
            )

            assert (
                workspace1_index in index_names
            ), f"Workspace 1 index '{workspace1_index}' should exist"
            assert (
                workspace2_index in index_names
            ), f"Workspace 2 index '{workspace2_index}' should exist"

    finally:
        await storage2.drop()
        await storage2.finalize()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--run-integration"])
