"""tolerant_load_json_dict: recover one JSON object from LLM/VLM text.

Trailing prose (including trailing braces), leading prose, fences, and the
malformed-object slips json_repair fixes must all be recovered; top-level
arrays must be rejected so callers retry / fall back. Objects hidden behind
bracketed prose (``[draft] {...}``) are intentionally NOT recovered — they are
indistinguishable from a real array without heuristics and do not occur in
practice; the caller's retry / text fallback covers the rare case.
"""

from __future__ import annotations

import pytest

from lightrag.utils import strip_markdown_code_fence, tolerant_load_json_dict

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
    """The old greedy ``\\{.*\\}`` slice over-extends across trailing braces and
    drops the object; the new helper recovers it."""
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
    assert tolerant_load_json_dict(raw) == {"facts": [{"text": "ok"}]}


def test_plain_object_still_loads() -> None:
    assert tolerant_load_json_dict('{"a": 1}') == {"a": 1}


def test_single_line_fence_is_stripped() -> None:
    """The shared fence stripper handles fences with no interior newlines,
    which the old inline pipeline regex (mandatory ``\\n``) missed."""
    assert strip_markdown_code_fence('```json {"a":1}```').strip() == '{"a":1}'
    assert tolerant_load_json_dict('```json {"a":1}```') == {"a": 1}


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


def test_balanced_object_slice_respects_nested_and_quoted_braces() -> None:
    raw = "analysis {name:n,description:'brace } ok',meta:{unit:x}} trailing"
    assert tolerant_load_json_dict(raw) == {
        "name": "n",
        "description": "brace } ok",
        "meta": {"unit": "x"},
    }


@pytest.mark.parametrize(
    "raw",
    [
        # Bracketed prose before the object reads as a top-level array without
        # heuristics; intentionally rejected (caller retries / falls back).
        'analysis: [draft] {name:"x", type:"Chart", description:"ok",}',
        'Analysis [draft: {"name":"n","description":"d"}',
        '[draft: {"name":"n","description":"d"}',
    ],
)
def test_bracketed_prose_prefix_is_rejected(raw: str) -> None:
    assert tolerant_load_json_dict(raw) == {}


def test_broken_array_shell_yielding_dict_is_rejected() -> None:
    """A broken top-level array shell where json_repair salvages the inner
    object must still be rejected — a leading '[' is a top-level array. This
    proves the structural check runs *before* a json_repair dict is trusted:
    with the old ordering json_repair.loads returned the inner dict and the
    helper leaked it, bypassing the array-rejection contract."""
    import json_repair

    raw = '[}{"name":"fig-1","type":"Chart","description":"ok"}]'
    # json_repair on its own hands back the inner object (the old-ordering leak)...
    assert isinstance(json_repair.loads(raw), dict)
    # ...but the contract rejects any top-level array.
    assert tolerant_load_json_dict(raw) == {}


@pytest.mark.parametrize(
    "raw",
    [
        '[}{"a":1}]',
        'Here is the result: [}{"a":1}]',
    ],
)
def test_broken_array_shell_variants_rejected(raw: str) -> None:
    assert tolerant_load_json_dict(raw) == {}


def test_unclosed_quote_before_broken_array_is_rejected() -> None:
    """An unclosed double quote makes the opener scan miss the structure
    (returns None); json_repair would still salvage the inner object from the
    broken array shell. Only a clear '{' opener is trusted, so this is
    rejected — a None opener must not fall through to json_repair."""
    import json_repair

    raw = '"oops [}{"a":1}]'
    assert isinstance(json_repair.loads(raw), dict)  # json_repair would leak it
    assert tolerant_load_json_dict(raw) == {}


def test_closed_quote_prose_before_object_still_recovers() -> None:
    """Guard against over-rejection: a properly closed quoted phrase (even one
    containing a '[') before the object leaves the first opener as '{', so the
    object is still recovered."""
    raw = 'Source: "see [1] here" {"a":1}'
    assert tolerant_load_json_dict(raw) == {"a": 1}
