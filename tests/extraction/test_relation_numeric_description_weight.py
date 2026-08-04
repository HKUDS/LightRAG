"""Numeric relationship descriptions must not become edge weights."""

import pytest

from lightrag.operate import _handle_single_relationship_extraction

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "description,expected_weight",
    [
        ("1000", 1.0),
        ("12.345", 1.0),
        ("+2.5", 1.0),
        ("-1.5", 1.0),
        ("version 1.0", 1.0),
    ],
)
def test_numeric_description_keeps_default_weight(description, expected_weight):
    result = _handle_single_relationship_extraction(
        ["relationship", "Alice", "Bob", "knows", description],
        "chunk-1",
        1,
    )
    assert result is not None
    assert result["description"] == description
    assert result["weight"] == expected_weight


def test_non_numeric_description_keeps_default_weight():
    result = _handle_single_relationship_extraction(
        ["relationship", "Alice", "Bob", "knows", "founded the company"],
        "chunk-1",
        1,
    )
    assert result is not None
    assert result["description"] == "founded the company"
    assert result["weight"] == 1.0
