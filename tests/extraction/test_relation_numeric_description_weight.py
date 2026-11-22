"""Numeric relationship descriptions must not become edge weights."""

import pytest

from lightrag.operate import _handle_single_relationship_extraction

pytestmark = pytest.mark.offline


# The weight is a constant 1.0 whatever the description says, so one numeric
# description is the whole claim.
@pytest.mark.parametrize("description,expected_weight", [("12.345", 1.0)])
def test_numeric_description_keeps_default_weight(description, expected_weight):
    result = _handle_single_relationship_extraction(
        ["relationship", "Alice", "Bob", "knows", description],
        "chunk-1",
        1,
    )
    assert result is not None
    assert result["description"] == description
    assert result["weight"] == expected_weight
