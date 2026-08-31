"""``QueryParam`` is a plain dataclass, so its field ORDER is public API.

It has no ``kw_only=True``, which means SDK callers may construct it
positionally. Inserting a new field in the middle silently rebinds every
positional argument after it -- the kind of break that produces no error, just
different behaviour: a positional ``False`` meant for ``enable_rerank`` lands
on the inserted field while reranking quietly keeps its environment default.

New fields therefore go at the END of the class. This test pins that rule
against the field order at the time ``disable_user_prompt_prefix`` was added.
"""

from __future__ import annotations

import dataclasses

import pytest

from lightrag.base import QueryParam


pytestmark = pytest.mark.offline


# The positional prefix of the signature as it stood before
# disable_user_prompt_prefix existed. Do NOT reorder to match a future refactor
# -- if this list stops matching, the constructor signature changed and every
# positional caller downstream changed behaviour with it.
_LEGACY_POSITIONAL_ORDER = [
    "mode",
    "only_need_context",
    "only_need_prompt",
    "response_type",
    "stream",
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "hl_keywords",
    "ll_keywords",
    "conversation_history",
    "user_prompt",
    "enable_rerank",
    "include_references",
]


def test_legacy_positional_field_order_is_unchanged():
    names = [f.name for f in dataclasses.fields(QueryParam)]
    assert names[: len(_LEGACY_POSITIONAL_ORDER)] == _LEGACY_POSITIONAL_ORDER


def test_new_fields_are_appended_after_the_legacy_ones():
    names = [f.name for f in dataclasses.fields(QueryParam)]
    assert "disable_user_prompt_prefix" in names
    assert names.index("disable_user_prompt_prefix") >= len(
        _LEGACY_POSITIONAL_ORDER
    )


def test_positional_construction_still_binds_enable_rerank():
    """The concrete break: the 15th positional argument.

    Before the field was moved to the end, this ``False`` bound to
    ``disable_user_prompt_prefix`` and ``enable_rerank`` silently kept its
    ``RERANK_BY_DEFAULT`` value.
    """
    param = QueryParam(
        "mix",
        False,
        False,
        "Multiple Paragraphs",
        False,
        40,
        20,
        6000,
        8000,
        30000,
        [],
        [],
        [],
        "be concise",
        False,
    )
    assert param.user_prompt == "be concise"
    assert param.enable_rerank is False
    assert param.disable_user_prompt_prefix is False


def test_positional_construction_still_binds_include_references():
    """The 16th positional argument, shifted by the same insertion."""
    param = QueryParam(
        "mix",
        False,
        False,
        "Multiple Paragraphs",
        False,
        40,
        20,
        6000,
        8000,
        30000,
        [],
        [],
        [],
        None,
        True,
        True,
    )
    assert param.enable_rerank is True
    assert param.include_references is True
    assert param.disable_user_prompt_prefix is False
