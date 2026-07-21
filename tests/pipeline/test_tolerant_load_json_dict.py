"""Trailing prose braces must not drop recoverable JSON objects."""

from __future__ import annotations

import pytest

from lightrag.utils import tolerant_load_json_dict

pytestmark = pytest.mark.offline


def test_trailing_brace_prose_recovers_object() -> None:
    raw = '{"facts":[{"text":"ok"}]} trailing {brace}'
    assert tolerant_load_json_dict(raw) == {"facts": [{"text": "ok"}]}


def test_prose_apostrophe_before_object_still_recovers_object() -> None:
    raw = 'Here\'s the result: {"facts":[{"text":"ok"}]} trailing {brace}'
    assert tolerant_load_json_dict(raw) == {"facts": [{"text": "ok"}]}


def test_hash_prefixed_prose_before_object_still_recovers_object() -> None:
    raw = 'Result #1: {"name":"n","description":"d"}'
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


@pytest.mark.parametrize(
    "raw",
    [
        '// result: {"name":"n","description":"d"}',
        'Note // result: {"name":"n","description":"d"}',
    ],
)
def test_slash_prefixed_prose_before_object_still_recovers_object(raw: str) -> None:
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


def test_quoted_prose_apostrophe_before_object_still_recovers_object() -> None:
    raw = 'Result: \'Here\'s context\' {"facts":[{"text":"ok"}]} trailing {brace}'
    assert tolerant_load_json_dict(raw) == {"facts": [{"text": "ok"}]}


def test_greedy_regex_would_fail_same_input() -> None:
    import json_repair
    import re

    raw = '{"facts":[{"text":"ok"}]} trailing {brace}'
    m = re.search(r"\{[\s\S]*\}", raw)
    assert m is not None
    try:
        obj = json_repair.loads(m.group(0))
        recovered = obj if isinstance(obj, dict) else {}
    except Exception:
        recovered = {}
    assert recovered == {}


def test_plain_object_still_loads() -> None:
    assert tolerant_load_json_dict('{"a": 1}') == {"a": 1}


@pytest.mark.parametrize(
    "raw",
    [
        '[{"name":"first"},{"name":"second"}]',
        '```json\n[{"name":"first"},{"name":"second"}]\n```',
        'Here is the result: [{"name":"first"},{"name":"second"}]',
        'Here is the result: [{name:"first"},{name:"second"}]',
        '[{"name":"first"},{"name":"second"}',
        'Here is the result: [ {"name":"first"},{"name":"second"}',
        'Here is the result: ["note", {"name":"first"}',
        "['note', {'name':'first','description':'x'}]",
        "[note, {name:'first',description:'x'}]",
        '[/* note */ {"name":"first","description":"x"}]',
        '[// note\n{"name":"first","description":"x"}]',
        '[# note\n{"name":"first","description":"x"}]',
        '[1, /* note */ {"name":"first","description":"x"}]',
        '[/* ] */ {"name":"first","description":"x"}]',
        '[// ]\n{"name":"first","description":"x"}]',
        '[# ]\n{"name":"first","description":"x"}]',
        "['note ]', {'name':'first','description':'x'}]",
        '[http://example, {"name":"first","description":"x"}]',
    ],
)
def test_top_level_array_is_rejected(raw: str) -> None:
    assert tolerant_load_json_dict(raw) == {}


def test_bracketed_prefix_prose_still_repairs_object() -> None:
    # Weaker VLMs sometimes prefix a repairable object with bracketed notes.
    raw = 'analysis: [draft] {name:"x", type:"Chart", description:"ok",}'
    assert tolerant_load_json_dict(raw) == {
        "name": "x",
        "type": "Chart",
        "description": "ok",
    }


def test_unmatched_bracket_prefix_still_recovers_object() -> None:
    raw = 'Analysis [draft: {"name":"n","description":"d"}'
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


def test_leading_unmatched_bracketed_prose_still_recovers_object() -> None:
    raw = '[draft: {"name":"n","description":"d"}'
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


def test_closed_bracketed_prose_still_recovers_object() -> None:
    raw = 'Analysis [draft: {"name":"n","description":"d"}]'
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


@pytest.mark.parametrize(
    "url",
    [
        "http://example",
        "https://example.test/path",
        "git+ssh://example/repo//tree#readme",
    ],
)
def test_prose_url_before_object_still_recovers_object(url: str) -> None:
    raw = f'Source: {url} {{"name":"n","description":"d"}}'
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


def test_bracketed_prose_url_before_object_still_recovers_object() -> None:
    raw = '[draft: Source http://example {"name":"n","description":"d"}]'
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


def test_repaired_object_excludes_trailing_prose() -> None:
    raw = "analysis: [draft] {name:n,description:d} trailing"
    assert tolerant_load_json_dict(raw) == {"name": "n", "description": "d"}


def test_balanced_object_slice_respects_nested_and_quoted_braces() -> None:
    raw = "analysis {name:n,description:'brace } ok',meta:{unit:x}} trailing"
    assert tolerant_load_json_dict(raw) == {
        "name": "n",
        "description": "brace } ok",
        "meta": {"unit": "x"},
    }
