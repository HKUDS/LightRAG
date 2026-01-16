"""
Integration tests for Entity Resolution and Conflict Detection.

These tests verify that the entity resolution and conflict detection
features work together correctly in realistic scenarios.
"""

import pytest
from lightrag.entity_resolution import EntityResolver
from lightrag.conflict_detection import ConflictDetector, ConflictInfo


class TestEntityResolutionIntegration:
    """Integration tests for entity resolution workflow."""

    def test_entity_resolution_full_workflow(self):
        """Test complete entity resolution workflow with name variations."""
        # Simulate entities extracted from multiple documents
        all_nodes = {
            # Apple variations
            "Apple Inc": [
                {"entity_type": "ORGANIZATION", "description": "Tech company founded in 1976"},
                {"entity_type": "ORGANIZATION", "description": "Based in Cupertino, California"},
            ],
            "Apple Inc.": [
                {"entity_type": "ORGANIZATION", "description": "Makes iPhones and Macs"},
            ],
            "APPLE INC": [
                {"entity_type": "ORGANIZATION", "description": "Market cap over $2 trillion"},
            ],
            # Google variations
            "Google LLC": [
                {"entity_type": "ORGANIZATION", "description": "Search engine company"},
            ],
            "Google": [
                {"entity_type": "ORGANIZATION", "description": "Subsidiary of Alphabet"},
            ],
            # Tesla - no variations
            "Tesla": [
                {"entity_type": "ORGANIZATION", "description": "Electric vehicle manufacturer"},
            ],
        }

        resolver = EntityResolver(similarity_threshold=0.85)
        result = resolver.consolidate_entities(all_nodes)

        # Should have 3 consolidated entities: Apple, Google, Tesla
        assert len(result) == 3

        # Find the Apple entity
        apple_entities = [k for k in result.keys() if "apple" in k.lower()]
        assert len(apple_entities) == 1
        apple_key = apple_entities[0]

        # All Apple descriptions should be merged
        assert len(result[apple_key]) == 4

        # Google entities should be merged
        google_entities = [k for k in result.keys() if "google" in k.lower()]
        assert len(google_entities) == 1
        google_key = google_entities[0]
        assert len(result[google_key]) == 2

        # Tesla should remain as-is
        assert "Tesla" in result
        assert len(result["Tesla"]) == 1


class TestConflictDetectionIntegration:
    """Integration tests for conflict detection workflow."""

    def test_conflict_detection_full_workflow(self):
        """Test complete conflict detection workflow with multiple sources."""
        descriptions = [
            ("Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning.", "doc_001"),
            ("Tesla was founded in 2004.", "doc_002"),
            ("Tesla is headquartered in Austin, Texas.", "doc_003"),
            ("Tesla produces electric vehicles and energy storage systems.", "doc_004"),
        ]

        detector = ConflictDetector(confidence_threshold=0.7)
        conflicts = detector.detect_conflicts("Tesla", descriptions)

        # Should detect temporal conflict (2003 vs 2004)
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        assert len(temporal_conflicts) >= 1

        # Verify conflict structure
        conflict = temporal_conflicts[0]
        assert conflict.entity_name == "Tesla"
        assert conflict.value_a in ("2003", "2004")
        assert conflict.value_b in ("2003", "2004")
        assert conflict.value_a != conflict.value_b
        assert conflict.confidence >= 0.7

    def test_conflict_detection_with_no_conflicts(self):
        """Test that non-conflicting descriptions don't create false positives."""
        descriptions = [
            ("Apple is headquartered in Cupertino.", "doc_001"),
            ("Apple makes iPhones and Macs.", "doc_002"),
            ("Apple was founded in 1976.", "doc_003"),
            ("Apple is a technology company.", "doc_004"),
        ]

        detector = ConflictDetector(confidence_threshold=0.7)
        conflicts = detector.detect_conflicts("Apple", descriptions)

        # Should not detect any temporal conflicts (all say 1976 or no date)
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        # Only one year mentioned, no conflict
        assert len(temporal_conflicts) == 0


class TestCombinedWorkflow:
    """Test entity resolution and conflict detection working together."""

    def test_resolution_then_conflict_detection(self):
        """Test that conflict detection works on resolved entities."""
        # First, resolve entities
        all_nodes = {
            "Tesla Inc": [
                {"entity_type": "ORGANIZATION", "description": "Founded in 2003", "source_id": "doc_001"},
            ],
            "Tesla Inc.": [
                {"entity_type": "ORGANIZATION", "description": "Founded in 2004", "source_id": "doc_002"},
            ],
            "TESLA": [
                {"entity_type": "ORGANIZATION", "description": "Electric vehicle maker", "source_id": "doc_003"},
            ],
        }

        # Resolve entities
        resolver = EntityResolver(similarity_threshold=0.85)
        resolved = resolver.consolidate_entities(all_nodes)

        # Should have 1 consolidated Tesla entity
        assert len(resolved) == 1

        # Get the canonical name and its descriptions
        canonical_name = list(resolved.keys())[0]
        entities = resolved[canonical_name]

        # Build description tuples for conflict detection
        descriptions_with_sources = [
            (e.get("description", ""), e.get("source_id", "unknown"))
            for e in entities
        ]

        # Detect conflicts
        detector = ConflictDetector(confidence_threshold=0.7)
        conflicts = detector.detect_conflicts(canonical_name, descriptions_with_sources)

        # Should detect the 2003 vs 2004 conflict
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        assert len(temporal_conflicts) >= 1


class TestEdgeCasesIntegration:
    """Test edge cases in integration scenarios."""

    def test_short_names_excluded_from_resolution(self):
        """Test that short names don't get fuzzy matched."""
        all_nodes = {
            "AI": [{"entity_type": "CONCEPT", "description": "Artificial Intelligence"}],
            "AI Inc": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
        }

        resolver = EntityResolver(similarity_threshold=0.85, min_name_length=3)
        result = resolver.consolidate_entities(all_nodes)

        # "AI" should remain separate (too short for fuzzy matching)
        # "AI Inc" should also remain separate (different type)
        assert "AI" in result
        assert len(result) == 2

    def test_different_types_not_merged(self):
        """Test that entities of different types are not merged."""
        all_nodes = {
            "Paris France": [{"entity_type": "LOCATION", "description": "City in France"}],
            "Paris Hotel": [{"entity_type": "ORGANIZATION", "description": "Luxury hotel"}],
        }

        resolver = EntityResolver(similarity_threshold=0.85)
        result = resolver.consolidate_entities(all_nodes)

        # Different types should not merge
        assert len(result) == 2

    def test_empty_description_handling(self):
        """Test handling of empty descriptions in conflict detection."""
        descriptions = [
            ("", "doc_001"),
            ("Tesla was founded in 2003.", "doc_002"),
        ]

        detector = ConflictDetector(confidence_threshold=0.7)
        conflicts = detector.detect_conflicts("Tesla", descriptions)

        # Should not crash, and no conflicts from empty string
        assert isinstance(conflicts, list)
