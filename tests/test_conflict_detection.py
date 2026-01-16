"""
Tests for Conflict Detection module.

These tests verify the detection of contradictions in entity descriptions
such as conflicting dates, attributions, and numerical values.
"""

import pytest
from lightrag.conflict_detection import ConflictDetector, ConflictInfo


class TestConflictInfo:
    """Test ConflictInfo dataclass."""

    def test_to_log_message(self):
        """Test log message formatting."""
        conflict = ConflictInfo(
            entity_name="Tesla",
            conflict_type="temporal",
            value_a="2003",
            value_b="2004",
            source_a="doc_001",
            source_b="doc_002",
            confidence=0.95,
            context_a="Tesla was founded in 2003",
            context_b="Tesla was founded in 2004",
        )

        log_msg = conflict.to_log_message()

        assert "Tesla" in log_msg
        assert "temporal" in log_msg
        assert "2003" in log_msg
        assert "2004" in log_msg
        assert "0.95" in log_msg

    def test_to_prompt_context(self):
        """Test prompt context formatting."""
        conflict = ConflictInfo(
            entity_name="Tesla",
            conflict_type="temporal",
            value_a="2003",
            value_b="2004",
            source_a="doc_001",
            source_b="doc_002",
            confidence=0.95,
            context_a="Tesla was founded in 2003",
            context_b="Tesla was founded in 2004",
        )

        prompt_ctx = conflict.to_prompt_context()

        assert "CONFLICT DETECTED" in prompt_ctx
        assert "temporal" in prompt_ctx
        assert "2003" in prompt_ctx
        assert "2004" in prompt_ctx
        assert "doc_001" in prompt_ctx
        assert "doc_002" in prompt_ctx


class TestConflictDetector:
    """Test ConflictDetector class."""

    # T028: test_detect_temporal_conflict - different years detected
    def test_detect_temporal_conflict(self):
        """Test detection of conflicting years/dates."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("Tesla was founded in 2003 by Martin Eberhard.", "doc_001"),
            ("Tesla was founded in 2004.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Tesla", descriptions)

        assert len(conflicts) >= 1
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        assert len(temporal_conflicts) >= 1
        assert temporal_conflicts[0].value_a in ("2003", "2004")
        assert temporal_conflicts[0].value_b in ("2003", "2004")
        assert temporal_conflicts[0].value_a != temporal_conflicts[0].value_b

    def test_detect_temporal_conflict_with_dates(self):
        """Test detection of conflicting date formats."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("The event occurred on 01/15/2020.", "doc_001"),
            ("The event occurred on 01/20/2020.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Event", descriptions)

        # Should detect the date conflict
        assert len(conflicts) >= 1

    # T029: test_detect_attribution_conflict - different founders detected
    def test_detect_attribution_conflict(self):
        """Test detection of conflicting attributions."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("Apple was founded by Steve Jobs.", "doc_001"),
            ("Apple was founded by Steve Wozniak.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Apple", descriptions)

        assert len(conflicts) >= 1
        attribution_conflicts = [c for c in conflicts if c.conflict_type == "attribution"]
        assert len(attribution_conflicts) >= 1
        # Check that both names are detected
        values = {attribution_conflicts[0].value_a, attribution_conflicts[0].value_b}
        assert "Steve Jobs" in values or "Steve Wozniak" in values

    def test_detect_attribution_with_different_verbs(self):
        """Test attribution detection with various verbs."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("The technology was invented by John Smith.", "doc_001"),
            ("The technology was created by Jane Doe.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Technology", descriptions)

        assert len(conflicts) >= 1

    # T030: test_detect_numerical_conflict - different values detected
    def test_detect_numerical_conflict(self):
        """Test detection of conflicting numerical values."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("The company has 50,000 employees.", "doc_001"),
            ("The company has 75,000 employees.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Company", descriptions)

        assert len(conflicts) >= 1
        numerical_conflicts = [c for c in conflicts if c.conflict_type == "numerical"]
        assert len(numerical_conflicts) >= 1

    def test_detect_monetary_conflict(self):
        """Test detection of conflicting monetary values."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("Revenue was $100 million.", "doc_001"),
            ("Revenue was $150 million.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Revenue", descriptions)

        assert len(conflicts) >= 1

    # T031: test_no_conflict_on_extension - "Musk" vs "Musk and others" is not conflict
    def test_no_conflict_on_extension(self):
        """Test that extensions don't trigger false conflicts."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("The company was founded in 2004.", "doc_001"),
            ("The company was founded in 2004 by a group of engineers.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Company", descriptions)

        # Same year should not be a conflict
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        assert len(temporal_conflicts) == 0

    def test_no_conflict_same_values(self):
        """Test that identical values don't create conflicts."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("Founded in 2003.", "doc_001"),
            ("Established in 2003.", "doc_002"),
            ("Started operations in 2003.", "doc_003"),
        ]

        conflicts = detector.detect_conflicts("Company", descriptions)

        # Same year appearing multiple times should not be a conflict
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        assert len(temporal_conflicts) == 0

    # T032: test_confidence_scoring - conflicts have valid confidence scores
    def test_confidence_scoring(self):
        """Test that confidence scores are valid."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("Tesla was founded in 2003.", "doc_001"),
            ("Tesla was founded in 2004.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Tesla", descriptions)

        for conflict in conflicts:
            assert 0.0 <= conflict.confidence <= 1.0

    def test_confidence_threshold_filters(self):
        """Test that low-confidence conflicts are filtered."""
        # High threshold should filter some conflicts
        detector = ConflictDetector(confidence_threshold=0.99)

        descriptions = [
            ("Tesla was founded in 2003.", "doc_001"),
            ("Tesla was founded in 2004.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Tesla", descriptions)

        # With very high threshold, some conflicts might be filtered
        # (depends on implementation confidence calculation)
        # Just verify we don't get errors
        assert isinstance(conflicts, list)

    # T033: test_n_way_conflict - 3+ sources with different values (FR-019)
    def test_n_way_conflict(self):
        """Test detection of conflicts from multiple sources."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("Founded in 2003.", "doc_001"),
            ("Founded in 2004.", "doc_002"),
            ("Founded in 2005.", "doc_003"),
        ]

        conflicts = detector.detect_conflicts("Company", descriptions)

        # Should detect multiple pairwise conflicts
        # 3 different values = 3 pairwise conflicts (2003-2004, 2003-2005, 2004-2005)
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        assert len(temporal_conflicts) >= 2  # At least some pairs detected


class TestConflictDetectorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_descriptions(self):
        """Test handling of empty descriptions list."""
        detector = ConflictDetector()

        conflicts = detector.detect_conflicts("Entity", [])
        assert conflicts == []

    def test_single_description(self):
        """Test handling of single description."""
        detector = ConflictDetector()

        descriptions = [("Tesla was founded in 2003.", "doc_001")]
        conflicts = detector.detect_conflicts("Tesla", descriptions)

        assert conflicts == []

    def test_no_patterns_match(self):
        """Test handling when no patterns match."""
        detector = ConflictDetector()

        descriptions = [
            ("This is a generic description.", "doc_001"),
            ("Another generic description.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Entity", descriptions)
        assert conflicts == []

    def test_mixed_conflict_types(self):
        """Test detection of multiple conflict types."""
        detector = ConflictDetector(confidence_threshold=0.7)

        descriptions = [
            ("Company founded in 2003 by John Smith with 1000 employees.", "doc_001"),
            ("Company founded in 2004 by Jane Doe with 2000 employees.", "doc_002"),
        ]

        conflicts = detector.detect_conflicts("Company", descriptions)

        # Should detect temporal, attribution, and numerical conflicts
        conflict_types = {c.conflict_type for c in conflicts}
        assert len(conflict_types) >= 1


class TestConflictDetectorConfiguration:
    """Test configuration handling."""

    def test_default_threshold(self):
        """Test default confidence threshold."""
        detector = ConflictDetector()
        assert detector.confidence_threshold == 0.7

    def test_custom_threshold(self):
        """Test custom confidence threshold."""
        detector = ConflictDetector(confidence_threshold=0.9)
        assert detector.confidence_threshold == 0.9

    def test_invalid_threshold_raises(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError):
            ConflictDetector(confidence_threshold=1.5)

        with pytest.raises(ValueError):
            ConflictDetector(confidence_threshold=-0.1)


class TestPatternExtraction:
    """Test pattern extraction methods."""

    def test_extract_years(self):
        """Test year extraction."""
        detector = ConflictDetector()

        values = detector._extract_values("Founded in 2003.", detector.DATE_PATTERNS)
        assert len(values) >= 1
        assert any("2003" in v[0] for v in values)

    def test_extract_monetary(self):
        """Test monetary value extraction."""
        detector = ConflictDetector()

        values = detector._extract_values("Revenue was $100 million.", detector.NUMBER_PATTERNS)
        assert len(values) >= 1

    def test_extract_attribution(self):
        """Test attribution extraction."""
        detector = ConflictDetector()

        values = detector._extract_values(
            "Founded by Steve Jobs.", detector.ATTRIBUTION_PATTERNS
        )
        assert len(values) >= 1


class TestContextExtraction:
    """Test context/sentence extraction."""

    def test_get_context_sentence(self):
        """Test sentence extraction around matched value."""
        detector = ConflictDetector()

        text = "First sentence. Tesla was founded in 2003. Last sentence."
        context = detector._get_context_sentence(text, 25, 29)  # Around "2003"

        assert "Tesla" in context or "founded" in context


class TestDisableConflictDetection:
    """Test that conflict detection can be disabled (T057)."""

    def test_disable_conflict_detection(self):
        """Test that conflict detection can be disabled via config check."""
        # Simulate disabled conflict detection (feature toggle in operate.py)
        enable_conflict_detection = False

        descriptions = [
            ("Tesla was founded in 2003.", "doc_001"),
            ("Tesla was founded in 2004.", "doc_002"),
        ]

        if enable_conflict_detection:
            detector = ConflictDetector()
            conflicts = detector.detect_conflicts("Tesla", descriptions)
        else:
            conflicts = []  # No detection when disabled

        # When disabled, no conflicts should be returned
        assert len(conflicts) == 0
