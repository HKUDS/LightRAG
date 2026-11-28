"""tolerant_load_json_dict: recover one JSON object from LLM/VLM text.

Trailing prose (including trailing braces), leading prose, fences, and the
malformed-object slips json_repair fixes must all be recovered; top-level
arrays must be rejected so callers retry / fall back. Objects hidden behind
bracketed prose (``[draft] {...}``) are intentionally NOT recovered — they are
indistinguishable from a real array without heuristics and do not occur in
practice; the caller's retry / text fallback covers the rare case.
"""

from __future__ import annotations

import time

import pytest

from lightrag.utils import strip_markdown_code_fence, tolerant_load_json_dict

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            'Here\'s the result: {"a":1} trailing {brace}',
            id="apostrophe-trailing-brace",
        ),
        pytest.param('Result #1: {"a":1}', id="hash"),
        pytest.param('// result: {"a":1}', id="slash-comment"),
        pytest.param('Source: git+ssh://example/repo//tree#readme {"a":1}', id="url"),
        pytest.param('Source: "see [1] here" {"a":1}', id="closed-double-quote"),
        pytest.param('Source: "a \\" [x" {"a":1}', id="escaped-quote-in-quote"),
    ],
)
def test_prose_before_object_still_recovers_object(raw: str) -> None:
    """Only a ``[`` outside double quotes is a top-level array: apostrophes,
    comment-like prose, URLs and quoted phrases before the object (including a
    ``[`` inside an escaped-quote phrase) leave the opener at ``{``."""
    assert tolerant_load_json_dict(raw) == {"a": 1}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('[{"name":"first"},{"name":"second"}]', id="array"),
        pytest.param('```json\n[{"name":"first"}]\n```', id="fenced-array"),
        pytest.param('Here is the result: [{"name":"first"}]', id="prose-array"),
        pytest.param('[{"name":"first"},{"name":"second"}', id="truncated-array"),
        # json_repair scavenges the object out of this one if it is ever asked.
        pytest.param('[# ]\n{"name":"first","description":"x"}]', id="comment-array"),
        # Bracketed prose before an object reads as a top-level array without
        # heuristics; intentionally rejected (caller retries / falls back).
        pytest.param(
            'analysis: [draft] {name:"x", type:"Chart", description:"ok",}',
            id="bracketed-prose",
        ),
    ],
)
def test_top_level_array_is_rejected(raw: str) -> None:
    assert tolerant_load_json_dict(raw) == {}


def test_balanced_object_slice_respects_nested_and_quoted_braces() -> None:
    raw = "analysis {name:n,description:'brace } ok',meta:{unit:x}} trailing"
    assert tolerant_load_json_dict(raw) == {
        "name": "n",
        "description": "brace } ok",
        "meta": {"unit": "x"},
    }


def test_bare_apostrophe_slice_stops_at_object_end() -> None:
    """The fix lives in _first_balanced_object_slice: a bare apostrophe in an
    unquoted token (O'Reilly, it's) must not be read as a single-quote string
    start, or the slice runs past the object's real '}'. Asserted at the slice
    level because the boundary is deterministic — the exact repaired dict is
    json_repair-version-dependent, but the slice boundary is what the fix
    controls."""
    from lightrag.utils import _first_balanced_object_slice

    assert (
        _first_balanced_object_slice("{name: O'Reilly, type: Chart} tail {x}")
        == "{name: O'Reilly, type: Chart}"
    )
    assert _first_balanced_object_slice("{a: it's ok} tail {x}") == "{a: it's ok}"


def test_bare_apostrophe_object_recovers_without_trailing_pollution() -> None:
    """End-to-end symptom of the bug: trailing prose folded into a value. The
    exact repaired dict depends on the json_repair version, so this asserts only
    that no value absorbs the trailing marker — which proves the slice ended at
    the object's real '}' rather than running to the trailing '{brace}'."""
    raw = "{name: O'Reilly, type: Chart, description: ok} trailing {brace}"
    out = tolerant_load_json_dict(raw)
    assert out, "object should be recovered"
    assert all("trailing" not in str(v) and "brace" not in str(v) for v in out.values())


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


def test_unclosed_quote_before_broken_array_is_rejected() -> None:
    """An unclosed double quote makes the opener scan miss the structure
    (returns None); json_repair would still salvage the inner object from the
    broken array shell. Only a clear '{' opener is trusted, so this is
    rejected — a None opener must not fall through to json_repair."""
    import json_repair

    raw = '"oops [}{"a":1}]'
    assert isinstance(json_repair.loads(raw), dict)  # json_repair would leak it
    assert tolerant_load_json_dict(raw) == {}


def test_pseudo_object_prefix_before_array_is_rejected() -> None:
    """Fix-proof: whole-string json_repair scavenges the inner object out of a
    pseudo-object-prose ('{note}') + array input. The helper must not trust that
    — the real payload is a top-level array, so it is rejected (caller retries /
    falls back) rather than returning an element out of the array."""
    import json_repair

    raw = 'Context {note} [{"name":"fig","type":"Chart","description":"ok"}]'
    # whole-string json_repair hands back a dict scavenged from inside the array
    # (the leak the helper must not reproduce). isinstance, not ==, so the test
    # is not pinned to a specific json_repair version's exact repair.
    assert isinstance(json_repair.loads(raw), dict)
    # the contract rejects a top-level array behind pseudo-object prose.
    assert tolerant_load_json_dict(raw) == {}


def test_malformed_object_with_trailing_citation_recovers() -> None:
    """Guard against over-rejection: a genuine (malformed) leading object
    followed by trailing text starting with '[' — e.g. a '[1]' citation — must
    still be recovered. The first balanced slice is repaired to an object
    *before* any trailing consideration, so the object wins over the bracket."""
    raw = '{"name":"fig","type":"Chart","description":"ok",} [1]'
    assert tolerant_load_json_dict(raw) == {
        "name": "fig",
        "type": "Chart",
        "description": "ok",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            '```json {"a":1}```',
            '{"a":1}',
            id="3bt-single-line",
        ),
        pytest.param(
            '````json\n{"code": "```py\\nx\\n```"}\n````',
            '{"code": "```py\\nx\\n```"}',
            id="4bt-nested-triple-backtick",
        ),
        pytest.param(
            '~~~json\n{"a": 1}\n~~~',
            '{"a": 1}',
            id="tilde-multiline",
        ),
        pytest.param(
            '~~~~JSON\n{"a": 1}\n~~~~',
            '{"a": 1}',
            id="4tilde-uppercase-json",
        ),
        pytest.param(
            '```json\n{"a": 1}\n````',
            '{"a": 1}',
            id="commonmark-longer-backtick-closer",
        ),
    ],
)
def test_strip_markdown_code_fence_commonmark_variants(raw: str, expected: str) -> None:
    """Fences with no interior newline, 4+ backticks around embedded triple
    backticks, tilde fences and longer closers are all stripped."""
    assert strip_markdown_code_fence(raw).strip() == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('````json\n{"a":1}\n```', id="4-open-3-close-mismatch"),
        pytest.param('```json\n{"a":1}\n~~~', id="backtick-tilde-mismatch"),
    ],
)
def test_strip_markdown_code_fence_rejects_mismatched_fences_by_identity(
    raw: str,
) -> None:
    assert strip_markdown_code_fence(raw) is raw


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("```json" + " " * 1500 + "x", id="fence-then-spaces"),
        pytest.param('```json\n{"d":"' + " " * 16000, id="truncated-spaces"),
        pytest.param("```json\n" + "\n " * 1000, id="newline-space-runs"),
    ],
)
def test_strip_markdown_code_fence_is_linear_on_whitespace_runs(raw: str) -> None:
    """Degenerate LLM output (whitespace repeated up to the token limit) must
    not backtrack: a regex with a lazy body between whitespace runs takes
    seconds on these inputs and blocks the event loop."""
    started = time.perf_counter()
    assert strip_markdown_code_fence(raw) is raw
    assert time.perf_counter() - started < 0.5
