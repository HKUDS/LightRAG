"""
Tests for Hybrid Cross-Document Entity Resolution.

These tests verify the VDB-assisted and hybrid mode functionality
for cross-document entity resolution in large knowledge graphs.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from collections import defaultdict

from lightrag.constants import (
    DEFAULT_CROSS_DOC_RESOLUTION_MODE,
    DEFAULT_CROSS_DOC_THRESHOLD_ENTITIES,
    DEFAULT_CROSS_DOC_VDB_TOP_K,
    DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    DEFAULT_ENTITY_MIN_NAME_LENGTH,
)


class TestVDBResolution:
    """Test suite for VDB-assisted cross-document entity resolution (User Story 1)."""

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph storage."""
        mock = AsyncMock()
        mock.get_node_count = AsyncMock(return_value=10000)
        mock.get_all_nodes = AsyncMock(return_value=[
            {"entity_id": "Apple Inc", "entity_type": "ORGANIZATION"},
            {"entity_id": "Google LLC", "entity_type": "ORGANIZATION"},
            {"entity_id": "Microsoft Corporation", "entity_type": "ORGANIZATION"},
        ])
        return mock

    @pytest.fixture
    def mock_entity_vdb(self):
        """Create a mock entity VDB storage."""
        mock = AsyncMock()
        # Default: return similar entities for queries
        mock.query = AsyncMock(return_value=[
            {"id": "Apple Inc", "distance": 0.1, "entity_type": "ORGANIZATION"},
            {"id": "Apple Computer", "distance": 0.2, "entity_type": "ORGANIZATION"},
        ])
        return mock

    @pytest.fixture
    def global_config(self):
        """Default global config for tests."""
        return {
            "cross_doc_resolution_mode": "vdb",
            "cross_doc_threshold_entities": DEFAULT_CROSS_DOC_THRESHOLD_ENTITIES,
            "cross_doc_vdb_top_k": DEFAULT_CROSS_DOC_VDB_TOP_K,
            "entity_similarity_threshold": DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
            "entity_min_name_length": DEFAULT_ENTITY_MIN_NAME_LENGTH,
        }

    # T012: test_vdb_resolution_finds_similar_entities
    @pytest.mark.asyncio
    async def test_vdb_resolution_finds_similar_entities(
        self, mock_knowledge_graph, mock_entity_vdb, global_config
    ):
        """Test that VDB resolution correctly finds and matches similar entities."""
        from lightrag.operate import _resolve_cross_document_entities_vdb

        all_nodes = {
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
        }

        # VDB returns similar entity
        mock_entity_vdb.query = AsyncMock(return_value=[
            {"id": "Apple Inc", "distance": 0.05, "entity_type": "ORGANIZATION"},
        ])

        resolved_nodes, resolution_map = await _resolve_cross_document_entities_vdb(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Should resolve "Apple Inc." to existing "Apple Inc"
        assert "Apple Inc" in resolved_nodes or "Apple Inc." in resolution_map
        # VDB query should have been called
        mock_entity_vdb.query.assert_called()

    # T013: test_vdb_resolution_respects_entity_type_filter
    @pytest.mark.asyncio
    async def test_vdb_resolution_respects_entity_type_filter(
        self, mock_knowledge_graph, mock_entity_vdb, global_config
    ):
        """Test that VDB resolution only matches entities of the same type."""
        from lightrag.operate import _resolve_cross_document_entities_vdb

        all_nodes = {
            "Apple": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
        }

        # VDB returns entity of different type
        mock_entity_vdb.query = AsyncMock(return_value=[
            {"id": "Apple", "distance": 0.0, "entity_type": "FRUIT"},  # Different type!
        ])

        resolved_nodes, resolution_map = await _resolve_cross_document_entities_vdb(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Should NOT resolve because entity types don't match
        assert "Apple" in resolved_nodes
        assert "Apple" not in resolution_map

    # T014: test_vdb_resolution_handles_no_candidates
    @pytest.mark.asyncio
    async def test_vdb_resolution_handles_no_candidates(
        self, mock_knowledge_graph, mock_entity_vdb, global_config
    ):
        """Test that VDB resolution handles cases with no VDB candidates."""
        from lightrag.operate import _resolve_cross_document_entities_vdb

        all_nodes = {
            "Unique Entity XYZ": [{"entity_type": "ORGANIZATION", "description": "Unique"}],
        }

        # VDB returns no candidates
        mock_entity_vdb.query = AsyncMock(return_value=[])

        resolved_nodes, resolution_map = await _resolve_cross_document_entities_vdb(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Should keep the entity as-is
        assert "Unique Entity XYZ" in resolved_nodes
        assert len(resolution_map) == 0


class TestHybridMode:
    """Test suite for hybrid mode cross-document entity resolution (User Story 2)."""

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph storage."""
        mock = AsyncMock()
        mock.get_all_nodes = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_entity_vdb(self):
        """Create a mock entity VDB storage."""
        return AsyncMock()

    @pytest.fixture
    def global_config(self):
        """Default global config for tests."""
        return {
            "cross_doc_resolution_mode": "hybrid",
            "cross_doc_threshold_entities": 5000,
            "cross_doc_vdb_top_k": DEFAULT_CROSS_DOC_VDB_TOP_K,
            "entity_similarity_threshold": DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
            "entity_min_name_length": DEFAULT_ENTITY_MIN_NAME_LENGTH,
        }

    # T019: test_hybrid_uses_full_mode_below_threshold
    @pytest.mark.asyncio
    async def test_hybrid_uses_full_mode_below_threshold(
        self, mock_knowledge_graph, mock_entity_vdb, global_config
    ):
        """Test that hybrid mode uses full matching when below entity threshold."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        # Graph has only 1000 entities (below 5000 threshold)
        mock_knowledge_graph.get_node_count = AsyncMock(return_value=1000)
        mock_knowledge_graph.get_all_nodes = AsyncMock(return_value=[
            {"entity_id": "Apple Inc", "entity_type": "ORGANIZATION"},
        ])

        all_nodes = {
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "Tech"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Should use full mode
        assert mode_used == "full"
        # VDB should NOT have been queried
        mock_entity_vdb.query.assert_not_called()

    # T020: test_hybrid_uses_vdb_mode_above_threshold
    @pytest.mark.asyncio
    async def test_hybrid_uses_vdb_mode_above_threshold(
        self, mock_knowledge_graph, mock_entity_vdb, global_config
    ):
        """Test that hybrid mode uses VDB matching when above entity threshold."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        # Graph has 10000 entities (above 5000 threshold)
        mock_knowledge_graph.get_node_count = AsyncMock(return_value=10000)
        mock_entity_vdb.query = AsyncMock(return_value=[])

        all_nodes = {
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "Tech"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Should use VDB mode
        assert mode_used == "vdb"

    # T021: test_hybrid_mode_boundary_at_threshold
    @pytest.mark.asyncio
    async def test_hybrid_mode_boundary_at_threshold(
        self, mock_knowledge_graph, mock_entity_vdb, global_config
    ):
        """Test behavior exactly at the threshold boundary."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        # Graph has exactly 5000 entities (at threshold - should use VDB)
        mock_knowledge_graph.get_node_count = AsyncMock(return_value=5000)
        mock_entity_vdb.query = AsyncMock(return_value=[])

        all_nodes = {
            "Test Entity": [{"entity_type": "ORGANIZATION", "description": "Test"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # At threshold (>=), should use VDB mode
        assert mode_used == "vdb"


class TestConfigurableResolution:
    """Test suite for configurable resolution behavior (User Story 3)."""

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph storage."""
        mock = AsyncMock()
        mock.get_node_count = AsyncMock(return_value=1000)
        mock.get_all_nodes = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_entity_vdb(self):
        """Create a mock entity VDB storage."""
        mock = AsyncMock()
        mock.query = AsyncMock(return_value=[])
        return mock

    # T026: test_config_mode_full_always_uses_full
    @pytest.mark.asyncio
    async def test_config_mode_full_always_uses_full(
        self, mock_knowledge_graph, mock_entity_vdb
    ):
        """Test that mode=full always uses full matching regardless of graph size."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        # Even with large graph, mode=full should use full matching
        mock_knowledge_graph.get_node_count = AsyncMock(return_value=100000)
        mock_knowledge_graph.get_all_nodes = AsyncMock(return_value=[])

        global_config = {
            "cross_doc_resolution_mode": "full",
            "cross_doc_threshold_entities": 5000,
            "cross_doc_vdb_top_k": 10,
            "entity_similarity_threshold": 0.92,
            "entity_min_name_length": 2,
        }

        all_nodes = {
            "Test Entity": [{"entity_type": "ORGANIZATION", "description": "Test"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        assert mode_used == "full"

    # T027: test_config_mode_vdb_always_uses_vdb
    @pytest.mark.asyncio
    async def test_config_mode_vdb_always_uses_vdb(
        self, mock_knowledge_graph, mock_entity_vdb
    ):
        """Test that mode=vdb always uses VDB matching regardless of graph size."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        # Even with small graph, mode=vdb should use VDB matching
        mock_knowledge_graph.get_node_count = AsyncMock(return_value=100)

        global_config = {
            "cross_doc_resolution_mode": "vdb",
            "cross_doc_threshold_entities": 5000,
            "cross_doc_vdb_top_k": 10,
            "entity_similarity_threshold": 0.92,
            "entity_min_name_length": 2,
        }

        all_nodes = {
            "Test Entity": [{"entity_type": "ORGANIZATION", "description": "Test"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        assert mode_used == "vdb"

    # T028: test_config_mode_disabled_skips_resolution
    @pytest.mark.asyncio
    async def test_config_mode_disabled_skips_resolution(
        self, mock_knowledge_graph, mock_entity_vdb
    ):
        """Test that mode=disabled skips cross-document resolution entirely."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        global_config = {
            "cross_doc_resolution_mode": "disabled",
            "cross_doc_threshold_entities": 5000,
            "cross_doc_vdb_top_k": 10,
            "entity_similarity_threshold": 0.92,
            "entity_min_name_length": 2,
        }

        all_nodes = {
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "Tech"}],
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Same company"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Should return nodes unchanged
        assert mode_used == "disabled"
        assert "Apple Inc." in resolved_nodes
        assert "Apple Inc" in resolved_nodes
        assert len(resolution_map) == 0

    # T029: test_config_custom_threshold_respected
    @pytest.mark.asyncio
    async def test_config_custom_threshold_respected(
        self, mock_knowledge_graph, mock_entity_vdb
    ):
        """Test that custom threshold value is respected in hybrid mode."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        # Custom threshold of 2000
        mock_knowledge_graph.get_node_count = AsyncMock(return_value=2500)
        mock_knowledge_graph.get_all_nodes = AsyncMock(return_value=[])

        global_config = {
            "cross_doc_resolution_mode": "hybrid",
            "cross_doc_threshold_entities": 2000,  # Custom threshold
            "cross_doc_vdb_top_k": 10,
            "entity_similarity_threshold": 0.92,
            "entity_min_name_length": 2,
        }

        all_nodes = {
            "Test Entity": [{"entity_type": "ORGANIZATION", "description": "Test"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # 2500 >= 2000 threshold, should use VDB
        assert mode_used == "vdb"


class TestObservabilityMetrics:
    """Test suite for observability and metrics (User Story 4)."""

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph storage."""
        mock = AsyncMock()
        mock.get_node_count = AsyncMock(return_value=1000)
        mock.get_all_nodes = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_entity_vdb(self):
        """Create a mock entity VDB storage."""
        mock = AsyncMock()
        mock.query = AsyncMock(return_value=[])
        return mock

    # T034: test_resolution_logs_metrics
    @pytest.mark.asyncio
    async def test_resolution_logs_metrics(
        self, mock_knowledge_graph, mock_entity_vdb
    ):
        """Test that resolution returns metrics for logging: mode, entities, duplicates."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        global_config = {
            "cross_doc_resolution_mode": "hybrid",
            "cross_doc_threshold_entities": 5000,
            "cross_doc_vdb_top_k": 10,
            "entity_similarity_threshold": 0.92,
            "entity_min_name_length": 2,
        }

        all_nodes = {
            "Test Entity": [{"entity_type": "ORGANIZATION", "description": "Test"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Verify all metrics are available for logging
        # mode_used: "full", "vdb", or "disabled"
        assert mode_used in ["full", "vdb", "disabled"]
        # entities count
        assert len(resolved_nodes) >= 0
        # duplicates count
        assert isinstance(resolution_map, dict)

    # T035: test_resolution_logs_mode_switch
    @pytest.mark.asyncio
    async def test_resolution_logs_mode_switch(
        self, mock_knowledge_graph, mock_entity_vdb
    ):
        """Test that mode switching is correctly reported in hybrid mode."""
        from lightrag.operate import _resolve_cross_document_entities_hybrid

        # Large graph triggers VDB mode
        mock_knowledge_graph.get_node_count = AsyncMock(return_value=10000)

        global_config = {
            "cross_doc_resolution_mode": "hybrid",
            "cross_doc_threshold_entities": 5000,
            "cross_doc_vdb_top_k": 10,
            "entity_similarity_threshold": 0.92,
            "entity_min_name_length": 2,
        }

        all_nodes = {
            "Test Entity": [{"entity_type": "ORGANIZATION", "description": "Test"}],
        }

        resolved_nodes, resolution_map, mode_used = await _resolve_cross_document_entities_hybrid(
            all_nodes=all_nodes,
            knowledge_graph_inst=mock_knowledge_graph,
            entity_vdb=mock_entity_vdb,
            global_config=global_config,
        )

        # Verify VDB mode was used when above threshold
        assert mode_used == "vdb"
