"""Unit tests for chunk_schema heading normalization — table_header handling."""

import pytest

from lightrag.chunk_schema import format_parent_headings, normalize_chunk_heading


_WRAPPED = '<table format="json">[["H1", "H2"]]</table>'


@pytest.mark.offline
def test_normalize_passes_through_table_header():
    out = normalize_chunk_heading(
        {
            "heading": {
                "level": 2,
                "heading": "Section",
                "parent_headings": ["Doc"],
                "table_header": _WRAPPED,
            }
        }
    )
    assert out == {
        "level": 2,
        "heading": "Section",
        "parent_headings": ["Doc"],
        "table_header": _WRAPPED,
    }


@pytest.mark.offline
def test_normalize_keeps_table_header_when_all_other_fields_empty():
    # heading text + parents + level all empty would normally collapse to None;
    # a present table_header must keep the dict alive so the header is not lost.
    out = normalize_chunk_heading(
        {
            "heading": {
                "level": 0,
                "heading": "",
                "parent_headings": [],
                "table_header": _WRAPPED,
            }
        }
    )
    assert out is not None
    assert out["heading"] == ""
    assert out["parent_headings"] == []
    assert out["table_header"] == _WRAPPED


@pytest.mark.offline
def test_normalize_omits_table_header_key_when_absent():
    out = normalize_chunk_heading(
        {"heading": {"level": 1, "heading": "X", "parent_headings": []}}
    )
    assert out == {"level": 1, "heading": "X", "parent_headings": []}
    assert "table_header" not in out


@pytest.mark.offline
def test_normalize_all_empty_without_table_header_returns_none():
    assert (
        normalize_chunk_heading(
            {"heading": {"level": 0, "heading": "", "parent_headings": []}}
        )
        is None
    )
    assert normalize_chunk_heading({}) is None


@pytest.mark.offline
def test_format_parent_headings_ignores_table_header():
    # The recovered header must not leak into the parent-heading breadcrumb.
    breadcrumb = format_parent_headings(
        {
            "heading": {
                "level": 2,
                "heading": "Sec",
                "parent_headings": ["A", "B"],
                "table_header": _WRAPPED,
            }
        }
    )
    assert breadcrumb == "A → B"
