"""Unit tests for ``resolve_user_prompt`` — the prompt-prefix composition.

``USER_PROMPT_PREFIX`` lets an operator prepend global instructions to every
request's ``QueryParam.user_prompt``. The composition is deliberately dumb:
byte concatenation with no normalization on either side. These tests pin the
two properties that make that safe.

Coverage:
- Composition order, and that NO separator is injected — the operator owns
  formatting and ends the prefix with its own newlines.
- Neither side is stripped, so a prefix's trailing ``"\\n\\n"`` survives.
- ``slot`` falls back to ``"n/a"`` only when the composed text is empty,
  matching the pre-existing template convention.
- With no prefix configured, ``text`` is byte-identical to the value the
  answer cache keyed on before this feature existed. This is what lets the
  cache-policy version stay at v2 instead of invalidating every entry.
- ``disable_prefix`` drops the prefix and nothing else.
"""

from __future__ import annotations

import pytest

from lightrag.utils import resolve_user_prompt


pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Composition: order, and no injected separator.
# ---------------------------------------------------------------------------


def test_prefix_precedes_user_prompt():
    result = resolve_user_prompt("Be concise.", "Answer in Chinese.\n\n")
    assert result.text == "Answer in Chinese.\n\nBe concise."


def test_no_separator_is_injected():
    """The operator owns formatting; a missing separator stays missing.

    This is the documented cost of letting the operator control the join, and
    env.example/docs show a trailing newline for exactly this reason.
    """
    result = resolve_user_prompt("Be concise.", "Answer in Chinese.")
    assert result.text == "Answer in Chinese.Be concise."


def test_prefix_trailing_whitespace_is_preserved():
    """Stripping the prefix would eat the separator the operator added."""
    result = resolve_user_prompt("Be concise.", "  Answer in Chinese.  \n\n")
    assert result.text == "  Answer in Chinese.  \n\nBe concise."


def test_user_prompt_is_not_stripped():
    result = resolve_user_prompt("  Be concise.  ", "")
    assert result.text == "  Be concise.  "


# ---------------------------------------------------------------------------
# The nine None/""/whitespace combinations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", [None, "", "   "])
@pytest.mark.parametrize("user_prompt", [None, "", "   "])
def test_empty_and_whitespace_combinations(prefix, user_prompt):
    result = resolve_user_prompt(user_prompt, prefix)
    assert result.text == (prefix or "") + (user_prompt or "")
    # "n/a" appears only when nothing at all was composed.
    if result.text:
        assert result.slot == f"\n\n{result.text}"
    else:
        assert result.slot == "n/a"


def test_slot_is_na_only_when_nothing_composed():
    assert resolve_user_prompt(None, None).slot == "n/a"
    assert resolve_user_prompt("", "").slot == "n/a"
    # Whitespace-only is NOT nothing — it matches the pre-existing truthiness
    # check `if query_param.user_prompt`.
    assert resolve_user_prompt("   ", "").slot == "\n\n   "


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_user_prompt_makes_the_prefix_the_whole_instruction(empty):
    """The governing rule: an empty user_prompt does not disable the prefix.

    When a caller sends nothing, the operator's prefix alone becomes the
    instructions that reach the LLM. This is the common deployment -- one
    server-side policy, callers sending no per-request text -- and it is also
    what the WebUI relies on, since it ships ``user_prompt: ""`` as its default.
    Only ``disable_prefix`` suppresses the prefix.
    """
    prefix = "Answer in the language of the question.\n\n"
    result = resolve_user_prompt(empty, prefix)
    assert result.text == prefix
    assert result.slot == f"\n\n{prefix}"


def test_na_requires_both_sides_empty():
    """The placeholder is reached only when there is nothing at all to say."""
    assert resolve_user_prompt(None, None).slot == "n/a"
    assert resolve_user_prompt(None, "House style.").slot != "n/a"
    assert resolve_user_prompt("Be concise.", None).slot != "n/a"


def test_prefix_only_is_indistinguishable_from_a_caller_sending_it():
    """Why prompt.py needs no second placeholder."""
    prefix_only = resolve_user_prompt(None, "House style.")
    caller_sent = resolve_user_prompt("House style.", None)
    assert prefix_only == caller_sent
    assert prefix_only.slot == "\n\nHouse style."


# ---------------------------------------------------------------------------
# Cache invariant: no prefix configured => byte-identical to the old key input.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_prompt", [None, "", "   ", "Be concise.", " padded ", "多行\n文本"]
)
@pytest.mark.parametrize("prefix", [None, ""])
def test_unconfigured_prefix_preserves_legacy_cache_key_component(user_prompt, prefix):
    """``text`` must equal the expression the answer cache keyed on before.

    ``operate.py`` previously hashed ``query_param.user_prompt or ""``. Keeping
    this byte-identical is what allows ``_ANSWER_CACHE_POLICY_VERSION`` to stay
    at v2 — existing cached answers keep hitting for every deployment that
    never sets a prefix.
    """
    legacy = user_prompt or ""
    assert resolve_user_prompt(user_prompt, prefix).text == legacy


# ---------------------------------------------------------------------------
# The opt-out switch.
# ---------------------------------------------------------------------------


def test_disable_prefix_drops_only_the_prefix():
    result = resolve_user_prompt(
        "Be concise.", "Answer in Chinese.\n\n", disable_prefix=True
    )
    assert result.text == "Be concise."
    assert result.slot == "\n\nBe concise."


def test_disable_prefix_with_no_user_prompt_yields_na():
    result = resolve_user_prompt(None, "Answer in Chinese.\n\n", disable_prefix=True)
    assert result.text == ""
    assert result.slot == "n/a"


def test_disable_prefix_matches_never_configuring_one():
    """The switch itself must not be a distinct cache key.

    ``disable=True`` with a prefix set and ``disable=False`` with none produce
    byte-identical prompts and so should share a cache entry.
    """
    disabled = resolve_user_prompt("Be concise.", "House style.", disable_prefix=True)
    unconfigured = resolve_user_prompt("Be concise.", None)
    assert disabled == unconfigured
