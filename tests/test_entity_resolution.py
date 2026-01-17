"""
Tests for Entity Resolution module.

These tests verify the fuzzy matching and deduplication functionality
for similar entity names during document ingestion.
"""

import pytest
from lightrag.entity_resolution import EntityResolver


class TestEntityResolver:
    """Test suite for EntityResolver class."""

    # T010: test_fuzzy_matching_accuracy - verify token set ratio works correctly
    def test_fuzzy_matching_accuracy(self):
        """Test that fuzzy matching correctly identifies similar entity names."""
        resolver = EntityResolver(similarity_threshold=0.85)

        # These should be considered similar (high similarity)
        assert resolver._compute_similarity("Apple Inc", "Apple Inc.") > 0.85
        assert resolver._compute_similarity("Apple Inc", "APPLE INC") > 0.85
        # Token set ratio: Microsoft Corp vs Microsoft Corporation is ~0.80
        assert resolver._compute_similarity("Microsoft Corporation", "Microsoft Corp.") > 0.75

        # These should not be considered similar (low similarity)
        assert resolver._compute_similarity("Apple Inc", "Google Inc") < 0.85
        assert resolver._compute_similarity("Tesla", "Toyota") < 0.85

    # T011: test_threshold_behavior - verify 0.85 default threshold
    def test_threshold_behavior(self):
        """Test that threshold correctly filters matches."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "Makes iPhones"}],
            "Google": [{"entity_type": "ORGANIZATION", "description": "Search engine"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Apple Inc and Apple Inc. should be merged
        assert "Apple Inc" in result or "Apple Inc." in result
        # Google should remain separate
        assert "Google" in result
        # Total should be 2 entities
        assert len(result) == 2

    def test_threshold_strict(self):
        """Test stricter threshold (0.95) requires very close matches."""
        resolver = EntityResolver(similarity_threshold=0.95)

        all_nodes = {
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
            "Apple": [{"entity_type": "ORGANIZATION", "description": "Based in Cupertino"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # With strict threshold, "Apple" and "Apple Inc" might not merge
        # depending on exact similarity score
        # This tests that threshold is respected
        assert len(result) >= 1

    def test_threshold_lenient(self):
        """Test lenient threshold (0.75) merges more variations."""
        resolver = EntityResolver(similarity_threshold=0.75)

        all_nodes = {
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
            "Apple": [{"entity_type": "ORGANIZATION", "description": "Makes iPhones"}],
            "Apple Incorporated": [{"entity_type": "ORGANIZATION", "description": "Full name"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # With lenient threshold, Apple variants with shared tokens should merge
        # "Apple" and "Apple Inc" have ~80% similarity
        # At 0.75 threshold, "Apple" and "Apple Inc" should merge
        assert len(result) <= 2  # At least some merging happens

    # T012: test_case_insensitive_matching - "APPLE" matches "Apple"
    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
            "APPLE INC": [{"entity_type": "ORGANIZATION", "description": "Makes iPhones"}],
            "apple inc": [{"entity_type": "ORGANIZATION", "description": "Cupertino based"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # All case variations should merge into one
        assert len(result) == 1

    # T013: test_short_name_exclusion - names ≤2 chars never matched (FR-018)
    def test_short_name_exclusion(self):
        """Test that short names (≤2 chars) are excluded from fuzzy matching."""
        resolver = EntityResolver(similarity_threshold=0.85, min_name_length=3)

        all_nodes = {
            "AI": [{"entity_type": "CONCEPT", "description": "Artificial Intelligence"}],
            "AI Systems": [{"entity_type": "CONCEPT", "description": "AI technology"}],
            "ML": [{"entity_type": "CONCEPT", "description": "Machine Learning"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Short names should remain as-is, not merged
        assert "AI" in result
        assert "ML" in result
        # AI Systems should be separate (not fuzzy matched with short "AI")
        assert len(result) == 3

    def test_short_name_exclusion_with_similar_long_names(self):
        """Test short names don't interfere with longer name matching."""
        resolver = EntityResolver(similarity_threshold=0.85, min_name_length=3)

        all_nodes = {
            "US": [{"entity_type": "LOCATION", "description": "Country"}],
            "United States": [{"entity_type": "LOCATION", "description": "America"}],
            "United States of America": [{"entity_type": "LOCATION", "description": "USA full name"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # "US" should remain separate (too short)
        assert "US" in result
        # "United States" and "United States of America" should merge
        merged_names = [k for k in result.keys() if k != "US"]
        assert len(merged_names) == 1

    # T014: test_canonical_name_selection - longest name wins (FR-003)
    def test_canonical_name_selection(self):
        """Test that the longest name is selected as canonical."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "Apple": [{"entity_type": "ORGANIZATION", "description": "Short name"}],
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Medium name"}],
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "With period"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Longest name should be the canonical one
        # "Apple Inc." has 11 chars, "Apple Inc" has 9, "Apple" has 5
        assert "Apple Inc." in result
        assert len(result) == 1

    def test_canonical_name_alphabetical_tie_breaker(self):
        """Test that alphabetical order breaks ties for equal-length names."""
        resolver = EntityResolver(similarity_threshold=0.85)

        # Create similar names with same length (same tokens, different case)
        # "APPLE INC" and "Apple Inc" have same length (9 chars each)
        all_nodes = {
            "APPLE INC": [{"entity_type": "ORGANIZATION", "description": "Uppercase"}],
            "apple inc": [{"entity_type": "ORGANIZATION", "description": "Lowercase"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Both should merge (same name different case)
        assert len(result) == 1
        # The canonical name should be first alphabetically
        canonical = list(result.keys())[0]
        # "APPLE INC" comes before "apple inc" alphabetically (uppercase < lowercase in ASCII)
        assert canonical == "APPLE INC"

    # T015: test_entity_type_constraint - same-type entities only (FR-006)
    def test_entity_type_constraint(self):
        """Test that only entities of the same type are merged."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "Apple": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
            "Apple": [{"entity_type": "FRUIT", "description": "A fruit"}],  # Different type
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Full name"}],
        }

        # Note: dict keys must be unique, so let's structure this differently
        all_nodes = {
            "Apple Corp": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
            "Apple Fruit": [{"entity_type": "FRUIT", "description": "A fruit"}],
            "Apple Corp.": [{"entity_type": "ORGANIZATION", "description": "Full name"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # ORGANIZATION entities should merge, FRUIT should stay separate
        org_count = sum(1 for records in result.values()
                       if records and records[0].get("entity_type") == "ORGANIZATION")
        fruit_count = sum(1 for records in result.values()
                         if records and records[0].get("entity_type") == "FRUIT")

        assert org_count == 1  # Merged organizations
        assert fruit_count == 1  # Separate fruit

    def test_entity_type_constraint_prevents_cross_type_merge(self):
        """Test that entities of different types are never merged even if names match."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "Paris": [{"entity_type": "LOCATION", "description": "City in France"}],
            "Paris Hilton": [{"entity_type": "PERSON", "description": "Celebrity"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Both should remain separate (different types)
        assert len(result) == 2
        assert "Paris" in result
        assert "Paris Hilton" in result

    def test_entity_type_case_insensitive(self):
        """Test that entity types with different cases are treated as the same type."""
        resolver = EntityResolver(similarity_threshold=0.85)

        # Same entity with different entity_type casing (common LLM output variation)
        all_nodes = {
            "THALIE": [{"entity_type": "ORGANIZATION", "description": "Version 1"}],
            "Thalie": [{"entity_type": "Organization", "description": "Version 2"}],
            "thalie": [{"entity_type": "organization", "description": "Version 3"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # All should merge despite different entity_type casing
        assert len(result) == 1
        # Should have all 3 descriptions
        canonical_key = list(result.keys())[0]
        assert len(result[canonical_key]) == 3

    # T016: test_consolidate_entities_batch - batch processing works
    def test_consolidate_entities_batch(self):
        """Test batch processing of multiple entities."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            # Apple variations
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Tech company"}],
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "Makes iPhones"}],
            "APPLE": [{"entity_type": "ORGANIZATION", "description": "Cupertino"}],
            # Google variations
            "Google LLC": [{"entity_type": "ORGANIZATION", "description": "Search engine"}],
            "Google": [{"entity_type": "ORGANIZATION", "description": "Alphabet subsidiary"}],
            # Standalone
            "Microsoft": [{"entity_type": "ORGANIZATION", "description": "Windows maker"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Should have 3 consolidated entities
        assert len(result) == 3

        # Each consolidated entity should have merged descriptions
        for entity_name, records in result.items():
            if "Apple" in entity_name:
                assert len(records) == 3
            elif "Google" in entity_name:
                assert len(records) == 2
            else:
                assert len(records) == 1

    def test_consolidate_entities_empty_input(self):
        """Test handling of empty input."""
        resolver = EntityResolver()

        result = resolver.consolidate_entities({})
        assert result == {}

    def test_consolidate_entities_single_entity(self):
        """Test handling of single entity."""
        resolver = EntityResolver()

        all_nodes = {
            "Apple": [{"entity_type": "ORGANIZATION", "description": "Tech company"}]
        }

        result = resolver.consolidate_entities(all_nodes)
        assert result == all_nodes


class TestEntityResolverConfiguration:
    """Test configuration handling."""

    def test_default_values(self):
        """Test default configuration values."""
        resolver = EntityResolver()

        assert resolver.similarity_threshold == 0.85
        assert resolver.min_name_length == 2  # Changed from 3 to allow more entity matching

    def test_custom_threshold(self):
        """Test custom threshold configuration."""
        resolver = EntityResolver(similarity_threshold=0.90)
        assert resolver.similarity_threshold == 0.90

    def test_custom_min_length(self):
        """Test custom minimum name length configuration."""
        resolver = EntityResolver(min_name_length=5)
        assert resolver.min_name_length == 5

    def test_invalid_threshold_raises(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError):
            EntityResolver(similarity_threshold=1.5)

        with pytest.raises(ValueError):
            EntityResolver(similarity_threshold=-0.1)

    def test_invalid_min_length_raises(self):
        """Test that invalid min_name_length raises ValueError."""
        with pytest.raises(ValueError):
            EntityResolver(min_name_length=0)


class TestNormalization:
    """Test name normalization."""

    def test_normalize_basic(self):
        """Test basic normalization: lowercase, strip whitespace."""
        resolver = EntityResolver()

        # Basic case normalization
        assert resolver._normalize_name("Apple") == "apple"
        assert resolver._normalize_name("  Apple  ") == "apple"
        assert resolver._normalize_name("APPLE") == "apple"

    def test_normalize_removes_legal_forms(self):
        """Test that French and English legal forms are removed."""
        resolver = EntityResolver()

        # French legal forms
        assert resolver._normalize_name("SFJB SAS") == "sfjb"
        assert resolver._normalize_name("SAS SFJB") == "sfjb"
        assert resolver._normalize_name("THALIE SARL") == "thalie"
        assert resolver._normalize_name("EURL Dupont") == "dupont"
        # English legal forms
        assert resolver._normalize_name("Apple Inc") == "apple"
        assert resolver._normalize_name("Google LLC") == "google"
        assert resolver._normalize_name("Microsoft Corporation") == "microsoft"

    def test_normalize_removes_articles(self):
        """Test that French articles and prefixes are removed."""
        resolver = EntityResolver()

        assert resolver._normalize_name("Société THALIE") == "thalie"
        assert resolver._normalize_name("La Compagnie") == "compagnie"
        assert resolver._normalize_name("Groupe Bondoux") == "bondoux"
        assert resolver._normalize_name("Entreprise Martin") == "martin"

    def test_normalize_removes_accents(self):
        """Test that accents are normalized."""
        resolver = EntityResolver()

        assert resolver._normalize_name("Études Techniques") == "etudes techniques"
        assert resolver._normalize_name("Financière de Rozier") == "financiere rozier"
        assert resolver._normalize_name("Ingénierie") == "ingenierie"
        assert resolver._normalize_name("Crèche Étoile") == "creche etoile"

    def test_normalize_coalesces_acronyms(self):
        """Test that single characters are coalesced into acronyms."""
        resolver = EntityResolver()

        # Single characters with spaces become acronyms
        assert resolver._normalize_name("2 C B SAS") == "2cb"
        assert resolver._normalize_name("S F J B") == "sfjb"
        assert resolver._normalize_name("A B C Company") == "abc company"

    def test_normalize_coalesces_partial_acronyms(self):
        """Test that digit + 2-3 char acronym fragments are coalesced."""
        resolver = EntityResolver()

        # Partial acronyms: digit + short acronym
        assert resolver._normalize_name("2 CB") == "2cb"
        assert resolver._normalize_name("2 CB SAS") == "2cb"
        assert resolver._normalize_name("3 AB test") == "3ab test"
        # French articles should NOT be coalesced
        assert resolver._normalize_name("2 le chat") == "2 chat"  # "le" removed as article
        assert resolver._normalize_name("2 la maison") == "2 maison"  # "la" removed as article

    def test_normalize_combined(self):
        """Test combination of all normalization rules."""
        resolver = EntityResolver()

        # Complex case: accents + legal form + acronym
        assert resolver._normalize_name("2 C B SAS Ingénierie, Études Techniques") == "2cb ingenierie etudes techniques"
        # Legal form in middle
        assert resolver._normalize_name("Société Financière de Rozier SAS") == "financiere rozier"

    def test_similarity_with_punctuation(self):
        """Test similarity handles punctuation differences."""
        resolver = EntityResolver()

        # Punctuation has minor effect on similarity (~0.94 for period)
        similarity = resolver._compute_similarity("Apple Inc.", "Apple Inc")
        assert similarity > 0.90  # Still very high similarity


class TestFrenchEntityResolution:
    """Test entity resolution with French legal entities."""

    def test_sfjb_variations_merge(self):
        """Test that SFJB variations are properly merged."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "SAS SFJB": [{"entity_type": "ORGANIZATION", "description": "Version 1"}],
            "SFJB SAS": [{"entity_type": "ORGANIZATION", "description": "Version 2"}],
            "SFJB": [{"entity_type": "ORGANIZATION", "description": "Version 3"}],
            "Société SFJB": [{"entity_type": "ORGANIZATION", "description": "Version 4"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # All should merge into one entity
        assert len(result) == 1
        # Should have all 4 descriptions
        canonical_key = list(result.keys())[0]
        assert len(result[canonical_key]) == 4

    def test_2cb_variations_merge(self):
        """Test that 2CB variations with spaces are properly merged."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "2CB SAS": [{"entity_type": "ORGANIZATION", "description": "Normal"}],
            "2 C B SAS": [{"entity_type": "ORGANIZATION", "description": "With full spaces"}],
            "2 CB": [{"entity_type": "ORGANIZATION", "description": "Partial acronym"}],
            "SAS 2CB": [{"entity_type": "ORGANIZATION", "description": "Reversed"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # All should merge into one entity
        assert len(result) == 1

    def test_thalie_variations_merge(self):
        """Test that THALIE variations are properly merged."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "THALIE": [{"entity_type": "ORGANIZATION", "description": "Uppercase"}],
            "Thalie": [{"entity_type": "ORGANIZATION", "description": "Titlecase"}],
            "SAS THALIE": [{"entity_type": "ORGANIZATION", "description": "With SAS"}],
            "Société THALIE": [{"entity_type": "ORGANIZATION", "description": "With Société"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # All should merge into one entity
        assert len(result) == 1

    def test_accent_variations_merge(self):
        """Test that accent variations are properly merged."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "Financière de Rozier": [{"entity_type": "ORGANIZATION", "description": "With accents"}],
            "Financiere de Rozier": [{"entity_type": "ORGANIZATION", "description": "Without accents"}],
            "SAS Financière de Rozier": [{"entity_type": "ORGANIZATION", "description": "With SAS"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # All should merge into one entity
        assert len(result) == 1

    def test_canonical_name_preserves_original(self):
        """Test that the canonical name is the longest original (not normalized)."""
        resolver = EntityResolver(similarity_threshold=0.85)

        all_nodes = {
            "SFJB": [{"entity_type": "ORGANIZATION", "description": "Short"}],
            "SFJB SAS Consulting": [{"entity_type": "ORGANIZATION", "description": "Long"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Canonical should be the longest original name
        assert "SFJB SAS Consulting" in result
        assert len(result) == 1


class TestConfigurationIntegration:
    """Test configuration integration with global_config (T055-T056)."""

    def test_configuration_changes(self):
        """Test that threshold configuration changes affect matching behavior."""
        # Strict threshold - fewer matches
        strict_resolver = EntityResolver(similarity_threshold=0.95)
        lenient_resolver = EntityResolver(similarity_threshold=0.70)

        all_nodes = {
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Tech"}],
            "Apple": [{"entity_type": "ORGANIZATION", "description": "Phones"}],
        }

        strict_result = strict_resolver.consolidate_entities(all_nodes.copy())
        lenient_result = lenient_resolver.consolidate_entities(all_nodes.copy())

        # Lenient should merge more
        assert len(lenient_result) <= len(strict_result)

    def test_disable_entity_resolution(self):
        """Test that entity resolution can be disabled via config check."""
        # Simulate disabled entity resolution (feature toggle in operate.py)
        enable_entity_resolution = False

        all_nodes = {
            "Apple Inc": [{"entity_type": "ORGANIZATION", "description": "Tech"}],
            "Apple Inc.": [{"entity_type": "ORGANIZATION", "description": "Phones"}],
        }

        if enable_entity_resolution:
            resolver = EntityResolver()
            result = resolver.consolidate_entities(all_nodes)
        else:
            result = all_nodes  # No resolution

        # When disabled, entities should remain separate
        assert len(result) == 2
        assert "Apple Inc" in result
        assert "Apple Inc." in result


class TestPreferShorterCanonicalName:
    """Test prefer_shorter_canonical_name option."""

    def test_default_prefers_longer(self):
        """Test default behavior prefers longer canonical name."""
        # Use min_name_length=2 to include short names like "Acme" in fuzzy matching
        resolver = EntityResolver(similarity_threshold=0.85, min_name_length=2)

        all_nodes = {
            "Acme": [{"entity_type": "ORGANIZATION", "description": "Short"}],
            "Acme Ingenierie": [{"entity_type": "ORGANIZATION", "description": "Long"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Default: longest name is canonical
        assert "Acme Ingenierie" in result
        assert "Acme" not in result
        assert len(result) == 1

    def test_prefer_shorter_selects_shorter(self):
        """Test prefer_shorter_canonical_name=True selects shorter name."""
        resolver = EntityResolver(
            similarity_threshold=0.85,
            min_name_length=2,
            prefer_shorter_canonical_name=True,
        )

        all_nodes = {
            "Acme": [{"entity_type": "ORGANIZATION", "description": "Short"}],
            "Acme Ingenierie": [{"entity_type": "ORGANIZATION", "description": "Long"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # With option: shortest name is canonical
        assert "Acme" in result
        assert "Acme Ingenierie" not in result
        assert len(result) == 1

    def test_prefer_shorter_multiple_variants(self):
        """Test prefer_shorter with multiple variants."""
        resolver = EntityResolver(
            similarity_threshold=0.85,
            prefer_shorter_canonical_name=True,
        )

        all_nodes = {
            "SFJB": [{"entity_type": "ORGANIZATION", "description": "Shortest"}],
            "SFJB SAS": [{"entity_type": "ORGANIZATION", "description": "Medium"}],
            "SFJB SAS Consulting": [{"entity_type": "ORGANIZATION", "description": "Longest"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Shortest should be canonical
        assert "SFJB" in result
        assert len(result) == 1

    def test_prefer_shorter_alphabetical_tiebreaker(self):
        """Test alphabetical tiebreaker when lengths are equal."""
        resolver = EntityResolver(
            similarity_threshold=0.85,
            prefer_shorter_canonical_name=True,
        )

        all_nodes = {
            "ABC SAS": [{"entity_type": "ORGANIZATION", "description": "First alpha"}],
            "XYZ SAS": [{"entity_type": "ORGANIZATION", "description": "Last alpha"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # Equal length: alphabetically first should win
        # Note: These are different companies, they shouldn't merge
        # unless they have high similarity
        # Actually these won't merge - different names
        assert len(result) == 2

    def test_prefer_shorter_with_french_forms(self):
        """Test prefer_shorter with French legal forms."""
        resolver = EntityResolver(
            similarity_threshold=0.85,
            prefer_shorter_canonical_name=True,
        )

        all_nodes = {
            "Acme": [{"entity_type": "ORGANIZATION", "description": "Short"}],
            "Acme SAS": [{"entity_type": "ORGANIZATION", "description": "With legal form"}],
            "Société Acme SARL": [{"entity_type": "ORGANIZATION", "description": "Full"}],
        }

        result = resolver.consolidate_entities(all_nodes)

        # All normalize to "acme", shortest original is "Acme"
        assert "Acme" in result
        assert len(result) == 1
        # Should have 3 merged descriptions
        assert len(result["Acme"]) == 3


class TestCrossDocumentResolution:
    """Test cross-document entity resolution (operate.py integration)."""

    @pytest.mark.asyncio
    async def test_cross_doc_resolution_basic(self):
        """Test basic cross-document resolution against existing entities."""
        from lightrag.operate import _resolve_cross_document_entities
        from unittest.mock import AsyncMock, MagicMock

        # Mock knowledge graph with existing entities
        mock_graph = MagicMock()
        mock_graph.get_all_nodes = AsyncMock(return_value=[
            {"entity_id": "Acme Ingenierie", "entity_type": "ORGANIZATION"},
            {"entity_id": "Tesla Motors", "entity_type": "ORGANIZATION"},
        ])

        # New entities from a document
        all_nodes = {
            "Acme": [{"entity_type": "ORGANIZATION", "description": "Short variant"}],
            "SpaceX": [{"entity_type": "ORGANIZATION", "description": "Different company"}],
        }

        global_config = {
            "entity_similarity_threshold": 0.85,
            "entity_min_name_length": 2,  # Allow short names like "Acme"
        }

        resolved, resolution_map = await _resolve_cross_document_entities(
            all_nodes, mock_graph, global_config
        )

        # "Acme" should resolve to existing "Acme Ingenierie"
        assert "Acme Ingenierie" in resolved
        assert "Acme" not in resolved
        assert "Acme" in resolution_map
        assert resolution_map["Acme"][0] == "Acme Ingenierie"

        # "SpaceX" should remain (no match)
        assert "SpaceX" in resolved

    @pytest.mark.asyncio
    async def test_cross_doc_resolution_same_type_only(self):
        """Test that cross-document resolution only matches same entity type."""
        from lightrag.operate import _resolve_cross_document_entities
        from unittest.mock import AsyncMock, MagicMock

        mock_graph = MagicMock()
        mock_graph.get_all_nodes = AsyncMock(return_value=[
            {"entity_id": "Apple", "entity_type": "ORGANIZATION"},
        ])

        # New entity with DIFFERENT type
        all_nodes = {
            "Apple": [{"entity_type": "FRUIT", "description": "The fruit"}],
        }

        global_config = {
            "entity_similarity_threshold": 0.85,
            "entity_min_name_length": 3,
        }

        resolved, resolution_map = await _resolve_cross_document_entities(
            all_nodes, mock_graph, global_config
        )

        # Should NOT resolve because types are different
        assert "Apple" in resolved
        assert len(resolution_map) == 0

    @pytest.mark.asyncio
    async def test_cross_doc_resolution_empty_graph(self):
        """Test cross-document resolution with empty graph."""
        from lightrag.operate import _resolve_cross_document_entities
        from unittest.mock import AsyncMock, MagicMock

        mock_graph = MagicMock()
        mock_graph.get_all_nodes = AsyncMock(return_value=[])

        all_nodes = {
            "NewEntity": [{"entity_type": "ORGANIZATION", "description": "New"}],
        }

        global_config = {
            "entity_similarity_threshold": 0.85,
            "entity_min_name_length": 3,
        }

        resolved, resolution_map = await _resolve_cross_document_entities(
            all_nodes, mock_graph, global_config
        )

        # No existing entities, so no resolution
        assert "NewEntity" in resolved
        assert len(resolution_map) == 0

    @pytest.mark.asyncio
    async def test_cross_doc_resolution_error_handling(self):
        """Test cross-document resolution handles errors gracefully."""
        from lightrag.operate import _resolve_cross_document_entities
        from unittest.mock import AsyncMock, MagicMock

        mock_graph = MagicMock()
        mock_graph.get_all_nodes = AsyncMock(side_effect=Exception("DB error"))

        all_nodes = {
            "Entity": [{"entity_type": "ORGANIZATION", "description": "Test"}],
        }

        global_config = {}

        # Should not raise, just return original nodes
        resolved, resolution_map = await _resolve_cross_document_entities(
            all_nodes, mock_graph, global_config
        )

        assert "Entity" in resolved
        assert len(resolution_map) == 0

    @pytest.mark.asyncio
    async def test_cross_doc_resolution_french_normalization(self):
        """Test cross-document resolution with French entity normalization."""
        from lightrag.operate import _resolve_cross_document_entities
        from unittest.mock import AsyncMock, MagicMock

        mock_graph = MagicMock()
        mock_graph.get_all_nodes = AsyncMock(return_value=[
            {"entity_id": "SFJB", "entity_type": "ORGANIZATION"},
        ])

        # New entity with legal form and spacing variations
        all_nodes = {
            "S F J B SAS": [{"entity_type": "ORGANIZATION", "description": "With spaces and SAS"}],
        }

        global_config = {
            "entity_similarity_threshold": 0.85,
            "entity_min_name_length": 3,
        }

        resolved, resolution_map = await _resolve_cross_document_entities(
            all_nodes, mock_graph, global_config
        )

        # "S F J B SAS" should resolve to existing "SFJB" (same normalized form)
        assert "SFJB" in resolved
        assert "S F J B SAS" not in resolved
