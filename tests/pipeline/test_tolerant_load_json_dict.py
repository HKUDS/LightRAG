"""Trailing prose braces must not drop recoverable JSON objects."""

from __future__ import annotations

import pytest

from lightrag.utils import tolerant_load_json_dict

pytestmark = pytest.mark.offline


def test_trailing_brace_prose_recovers_object() -> None:
    raw = '{"facts":[{"text":"ok"}]} trailing {brace}'
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


def test_bracketed_prefix_prose_still_repairs_object():
    # Weaker VLMs sometimes prefix a repairable object with bracketed notes.
    raw = 'analysis: [draft] {name:"x", type:"Chart", description:"ok",}'
    assert tolerant_load_json_dict(raw) == {
        "name": "x",
        "type": "Chart",
        "description": "ok",
    }
