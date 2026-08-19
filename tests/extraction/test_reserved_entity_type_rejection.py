"""Reserved JavaScript prototype member names must never survive extraction.

An ``entity_type`` of ``__proto__`` / ``constructor`` / ``prototype`` crashes the
WebUI knowledge-graph view: the type-color resolver reads an inherited value off
a plain object and the Legend later calls a string method on the resulting
non-string map key. These names are never valid entity types, so both extraction
paths (delimited-record text AND JSON) drop the whole entity.

Covers both paths directly — a text-only test would pass even if the JSON path
(``ENTITY_EXTRACTION_USE_JSON=true``) still let the value through.
"""

from __future__ import annotations

import pytest

from lightrag.operate import (
    _handle_single_entity_extraction,
    _normalize_and_validate_entity_type,
    _process_json_extraction_result,
)

pytestmark = pytest.mark.offline

_RESERVED = ["__proto__", "constructor", "prototype"]


# --- shared helper -----------------------------------------------------------


@pytest.mark.parametrize("reserved", _RESERVED)
def test_shared_helper_rejects_reserved_names(reserved: str) -> None:
    assert _normalize_and_validate_entity_type(reserved, "ctx") is None


def test_shared_helper_passes_ordinary_type() -> None:
    assert _normalize_and_validate_entity_type("Organization", "ctx") == "organization"


def test_shared_helper_takes_first_token_of_comma_list() -> None:
    # Comma handling is now unified across both extraction paths.
    assert _normalize_and_validate_entity_type("person, extra", "ctx") == "person"


# --- text (delimited-record) path -------------------------------------------


@pytest.mark.parametrize("reserved", _RESERVED)
def test_text_path_drops_reserved_entity_type(reserved: str) -> None:
    attrs = ["entity", "GLOBEX LTD", reserved, "Globex Ltd supplies equipment."]
    assert _handle_single_entity_extraction(attrs, "chunk-1", 1, "doc.txt") is None


def test_text_path_keeps_ordinary_entity() -> None:
    attrs = ["entity", "GLOBEX LTD", "organization", "Globex Ltd supplies equipment."]
    entity = _handle_single_entity_extraction(attrs, "chunk-1", 1, "doc.txt")
    assert entity is not None
    assert entity["entity_type"] == "organization"


# --- JSON path ---------------------------------------------------------------


def _json_with_type(entity_type: str) -> str:
    return (
        '{"entities":[{"name":"Globex","type":"' + entity_type + '",'
        '"description":"Globex supplies equipment"}],"relationships":[]}'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved", _RESERVED)
async def test_json_path_drops_reserved_entity_type(reserved: str) -> None:
    nodes, _edges = await _process_json_extraction_result(
        _json_with_type(reserved), chunk_key="chunk-1", timestamp=1, file_path="doc.pdf"
    )
    assert "Globex" not in nodes


@pytest.mark.asyncio
async def test_json_path_keeps_ordinary_entity() -> None:
    nodes, _edges = await _process_json_extraction_result(
        _json_with_type("organization"),
        chunk_key="chunk-1",
        timestamp=1,
        file_path="doc.pdf",
    )
    assert "Globex" in nodes
    assert nodes["Globex"][0]["entity_type"] == "organization"
