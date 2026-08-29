"""An empty JSON extraction is an answer, not a parse failure.

``{"entities": [], "relationships": []}`` is what a chunk holding no reliable
facts legitimately produces in ``ENTITY_EXTRACTION_USE_JSON`` mode. The
cache-rebuild path used to decide "did the JSON parse work?" from the entity
count, so every such cache entry fell through to the delimiter parser — which
found no ``<|COMPLETE|>`` in a JSON document and warned about it, once per
chunk, for the whole rebuild.

The fallback now turns on the response's own format rather than on how much it
yielded: a result carrying the extraction schema is answered from JSON however
empty it turns out to be, unless it also carries the delimiter terminator that
says records were appended to it.

See issue #3744.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lightrag import operate
from lightrag.prompt import PROMPTS

pytestmark = pytest.mark.offline

_EMPTY_JSON = '{"entities": [], "relationships": []}'

_POPULATED_JSON = (
    '{"entities":[{"name":"Alice","type":"person","description":"an engineer"}],'
    '"relationships":[{"source":"Alice","target":"Acme",'
    '"keywords":"works at","description":"Alice works at Acme"}]}'
)

# Parses as an object but answers a different question, so it is not an
# extraction result and the delimiter parser is still the right fallback.
_NON_EXTRACTION_JSON = '{"summary": "no entities here"}'

_TUPLE = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
_COMPLETE = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

# The shape the old entity-count test rescued: an empty JSON object with
# delimiter records appended. The JSON prompt never asks for it, but
# tolerant_load_json_dict accepts the records as trailing prose.
_EMPTY_JSON_WITH_APPENDED_RECORDS = (
    f"{_EMPTY_JSON}\nentity{_TUPLE}Alice{_TUPLE}person{_TUPLE}an engineer\n{_COMPLETE}"
)


class _DummyTextChunksStorage:
    async def get_by_id(self, chunk_id: str):
        return {"file_path": "test.md"}


async def _rebuild(extraction_result: str):
    return await operate._rebuild_from_extraction_result(
        text_chunks_storage=_DummyTextChunksStorage(),
        extraction_result=extraction_result,
        chunk_id="chunk-001",
        timestamp=123,
    )


def _delimiter_parser_must_not_run():
    return patch(
        "lightrag.operate._process_extraction_result",
        side_effect=AssertionError(
            "delimiter parser reached for a valid JSON extraction result"
        ),
    )


@pytest.mark.parametrize(
    "cached",
    [_EMPTY_JSON, f"```json\n{_EMPTY_JSON}\n```"],
    ids=["raw", "fenced"],
)
@pytest.mark.asyncio
async def test_empty_json_extraction_does_not_reach_the_delimiter_parser(cached):
    with _delimiter_parser_must_not_run():
        nodes, edges = await _rebuild(cached)

    assert nodes == {}, "empty extraction must not invent entities"
    assert edges == {}, "empty extraction must not invent relationships"


@pytest.mark.asyncio
async def test_empty_json_extraction_is_logged_as_nothing_at_all():
    """The symptom the issue reports: one misleading warning per rebuilt chunk.

    Neither warning belongs here — not the delimiter parser's, and not the
    JSON parser's own "empty or unrecoverable", which would be just as wrong
    about a response that parsed perfectly well.
    """
    with patch("lightrag.operate.logger") as mock_logger:
        await _rebuild(_EMPTY_JSON)

    warnings = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
    assert warnings == [], warnings


@pytest.mark.asyncio
async def test_json_whose_records_are_all_skipped_is_still_a_json_answer():
    """Records dropped by validation leave the same empty result as an empty
    array, and must be read the same way — the schema is what decides."""
    all_records_dropped = (
        '{"entities":[{"name":"Alice","type":"person","description":"  "}],'
        '"relationships":[]}'
    )

    with _delimiter_parser_must_not_run():
        nodes, edges = await _rebuild(all_records_dropped)

    assert nodes == {}, "an entity with no description is dropped, not rescued"
    assert edges == {}


@pytest.mark.asyncio
async def test_populated_json_extraction_still_parses_as_json():
    with _delimiter_parser_must_not_run():
        nodes, edges = await _rebuild(_POPULATED_JSON)

    assert set(nodes) == {"Alice"}, "entity dropped — JSON answer not used"
    assert ("Alice", "Acme") in edges
    assert nodes["Alice"][0]["file_path"] == "test.md"


@pytest.mark.asyncio
async def test_object_without_the_extraction_schema_still_falls_back():
    """The fallback is narrowed, not removed: an object that is not an
    extraction result keeps reaching the delimiter parser."""
    with patch(
        "lightrag.operate._process_extraction_result",
        return_value=({"FromDelimiter": []}, {}),
    ) as delimiter_parser:
        nodes, _edges = await _rebuild(_NON_EXTRACTION_JSON)

    assert delimiter_parser.await_count == 1, "non-extraction JSON must fall back"
    assert nodes == {"FromDelimiter": []}


@pytest.mark.asyncio
async def test_unparseable_json_still_falls_back():
    with patch(
        "lightrag.operate._process_extraction_result",
        return_value=({"FromDelimiter": []}, {}),
    ) as delimiter_parser:
        nodes, _edges = await _rebuild("{ not json at all")

    assert delimiter_parser.await_count == 1, "unparseable JSON must fall back"
    assert nodes == {"FromDelimiter": []}


@pytest.mark.asyncio
async def test_records_appended_to_an_empty_object_are_still_recovered():
    """Narrowing the fallback must not drop what the old entity-count test
    rescued. The appended terminator is what says the response is also in
    delimiter format, so it reaches the delimiter parser — which finds its
    terminator there and emits no warning about it."""
    with patch("lightrag.operate.logger") as mock_logger:
        nodes, _edges = await _rebuild(_EMPTY_JSON_WITH_APPENDED_RECORDS)

    assert "Alice" in nodes, "appended delimiter records were dropped"
    warnings = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
    assert not any("Complete delimiter" in warning for warning in warnings), warnings


def test_extraction_shape_is_decided_by_the_fields_not_their_contents():
    assert operate._has_json_extraction_shape({"entities": [], "relationships": []})
    assert operate._has_json_extraction_shape({"entities": [{"name": "Alice"}]})
    # One field is enough: models routinely omit the empty one.
    assert operate._has_json_extraction_shape({"relationships": []})

    assert not operate._has_json_extraction_shape({"summary": "no entities here"})
    # Present but not a list: the builder cannot read it, so it is not the shape.
    assert not operate._has_json_extraction_shape({"entities": "none"})
