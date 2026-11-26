"""Entity/relation JSON extraction must recover objects wrapped in prose.

These call ``_process_json_extraction_result`` directly (not just the shared
``tolerant_load_json_dict`` helper) so the operate-side wiring is covered:
import, the parse hand-off, and the downstream entity/relation normalization.
A helper-only test would pass even if operate forgot to route through the
helper or mangled the normalization.
"""

from __future__ import annotations

import pytest

from lightrag.operate import _process_json_extraction_result

pytestmark = pytest.mark.offline

# Valid leading object followed by trailing prose braces. On the old code
# (json_repair.loads over the whole string) this returned a list -> non-dict ->
# dropped, losing the entire chunk. The unified helper recovers the object.
_TRAILING_BRACE_JSON = (
    '{"entities":[{"name":"Alice","type":"person","description":"an engineer"}],'
    '"relationships":[{"source":"Alice","target":"Acme",'
    '"keywords":"works at","description":"Alice works at Acme"}]}'
    " trailing {brace}"
)


@pytest.mark.asyncio
async def test_process_json_extraction_recovers_from_trailing_brace_prose() -> None:
    nodes, edges = await _process_json_extraction_result(
        _TRAILING_BRACE_JSON,
        chunk_key="chunk-1",
        timestamp=1,
        file_path="doc.pdf",
    )
    assert "Alice" in nodes, "entity dropped — object not recovered from prose"
    assert nodes["Alice"][0]["entity_type"] == "person"
    assert ("Alice", "Acme") in edges
    assert edges[("Alice", "Acme")][0]["description"] == "Alice works at Acme"


@pytest.mark.asyncio
async def test_process_json_extraction_top_level_array_yields_nothing() -> None:
    """A top-level array is not a single object: extraction yields nothing and
    the caller falls back (rebuild) / re-queries (gleaning)."""
    raw = '[{"name":"Alice","type":"person","description":"x"}]'
    nodes, edges = await _process_json_extraction_result(
        raw, chunk_key="chunk-1", timestamp=1, file_path="doc.pdf"
    )
    assert nodes == {}
    assert edges == {}


@pytest.mark.asyncio
async def test_process_json_extraction_reads_array_fields_as_text() -> None:
    """``"keywords": ["a", "b"]`` and ``"type": ["person"]`` are joined like the
    text format's comma-separated fields, not stored as their Python repr: the
    repr gave the keywords ``['marriage', 'collaboration']`` and, through its
    quotes, made the entity type invalid, which dropped the whole entity."""
    raw = (
        '{"entities":[{"name":"Marie Curie","type":["person"],'
        '"description":["A physicist.","A chemist."]}],'
        '"relationships":[{"source":"Marie Curie","target":"Pierre Curie",'
        '"keywords":["marriage","collaboration"],'
        '"description":"Married and researched radioactivity together."}]}'
    )
    nodes, edges = await _process_json_extraction_result(
        raw, chunk_key="chunk-1", timestamp=1, file_path="doc.pdf"
    )
    assert nodes["Marie Curie"][0]["entity_type"] == "person"
    assert nodes["Marie Curie"][0]["description"] == "A physicist.\nA chemist."
    edge = edges[("Marie Curie", "Pierre Curie")][0]
    assert edge["keywords"] == "marriage, collaboration"


@pytest.mark.asyncio
async def test_process_json_extraction_reads_null_fields_as_empty() -> None:
    """A JSON ``null`` is an empty field, not the string ``"None"``: null
    keywords are stored empty, and a null name or description drops its record
    as an empty one does."""
    raw = (
        '{"entities":[{"name":"Alice","type":"person","description":null},'
        '{"name":null,"type":"person","description":"nameless"},'
        '{"name":"Acme","type":"organization","description":"a company"}],'
        '"relationships":[{"source":"Alice","target":"Acme","keywords":null,'
        '"description":"Alice works at Acme"},'
        '{"source":null,"target":"Acme","keywords":"x","description":"orphan"}]}'
    )
    nodes, edges = await _process_json_extraction_result(
        raw, chunk_key="chunk-1", timestamp=1, file_path="doc.pdf"
    )
    assert set(nodes) == {"Acme"}
    assert set(edges) == {("Alice", "Acme")}
    assert edges[("Alice", "Acme")][0]["keywords"] == ""
