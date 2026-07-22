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
from lightrag.utils import tolerant_load_json_dict

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


def test_helper_recovers_same_trailing_brace_object() -> None:
    """Helper-level companion to the wiring test above."""
    recovered = tolerant_load_json_dict(_TRAILING_BRACE_JSON)
    assert recovered["entities"][0]["name"] == "Alice"
    assert recovered["relationships"][0]["target"] == "Acme"
