"""Input ceilings on QueryRequest (GHSA-r8jh-295g-vv42).

Fixing the whitelist would narrow who can trigger the stall but not remove it:
an authenticated 8 MiB ``/query`` blocks the loop too, because the graph modes
tokenize the keyword prompt before retrieval runs. These tests bound the fields
that feed that work.

They validate the model rather than driving HTTP: ``QueryRequest`` is what the
three query routes share, and a rejection here is by construction a rejection
before the handler — and therefore before any tokenizer or LLM call — runs.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from pydantic import ValidationError

# lightrag.api.config parses argv at import time, so the import has to be guarded
# the way the rest of the API tests guard it.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _query_routes.QueryRequest

from lightrag.constants import (  # noqa: E402
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS_PER_LIST,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES_PER_REQUEST,
    MAX_QUERY_CHARS,
    MAX_QUERY_TOKEN_BUDGET,
    MAX_QUERY_TOP_K,
    MAX_REQUEST_TEXT_CHARS,
    MAX_RESPONSE_TYPE_CHARS,
    MAX_ROLE_CHARS,
)

pytestmark = pytest.mark.offline


def _build(**overrides) -> QueryRequest:
    payload = {"query": "what is this about"}
    payload.update(overrides)
    return QueryRequest(**payload)


# --------------------------------------------------------------------------- #
# Text volume
# --------------------------------------------------------------------------- #


def test_oversized_query_is_refused():
    with pytest.raises(ValidationError):
        _build(query="q" * (MAX_QUERY_CHARS + 1))


def test_oversized_user_prompt_is_refused():
    with pytest.raises(ValidationError):
        _build(user_prompt="p" * (MAX_QUERY_CHARS + 1))


def test_aggregate_text_is_bounded_across_fields():
    """Each field is legal on its own; together they rebuild the payload.

    query and user_prompt at their maxima already reach the aggregate budget
    exactly, so a single further message tips it over — which is the point: the
    per-field limits alone leave the total unbounded as fields are added.
    """
    assert MAX_QUERY_CHARS * 2 == MAX_REQUEST_TEXT_CHARS
    with pytest.raises(ValidationError, match="total request text"):
        _build(
            query="q" * MAX_QUERY_CHARS,
            user_prompt="p" * MAX_QUERY_CHARS,
            conversation_history=[{"role": "user", "content": "one more"}],
        )


def test_oversized_response_type_is_refused():
    with pytest.raises(ValidationError):
        _build(response_type="r" * (MAX_RESPONSE_TYPE_CHARS + 1))


# --------------------------------------------------------------------------- #
# conversation_history — per-field and aggregate input limits
# --------------------------------------------------------------------------- #


def test_oversized_history_role_is_refused():
    """A role has its own size cap, so it cannot carry a large payload."""
    history = [{"role": "u" * (MAX_ROLE_CHARS + 1), "content": "hi"}]
    with pytest.raises(ValidationError, match="Each message 'role' must be at most"):
        _build(conversation_history=history)


def test_unenumerated_history_fields_consume_aggregate_budget():
    """Forwarded extras must share the query's one request budget.

    Before this check, both the 64 Ki-character query and the approximately
    64 Ki-character metadata payload fit their separate limits. Counting the
    serialized history in the aggregate correctly refuses their combined input.
    """
    history = [
        {
            "role": "user",
            "content": "hi",
            "metadata": "x" * (MAX_REQUEST_TEXT_CHARS - MAX_QUERY_CHARS),
        }
    ]
    with pytest.raises(ValidationError, match="total request text"):
        _build(query="q" * MAX_QUERY_CHARS, conversation_history=history)


def test_oversized_unenumerated_history_field_is_refused():
    """History alone cannot exceed the aggregate budget through extra fields."""
    history = [
        {
            "role": "user",
            "content": "hi",
            "metadata": "x" * (MAX_REQUEST_TEXT_CHARS // 4),
        }
        for _ in range(8)
    ]
    with pytest.raises(ValidationError, match="total request text"):
        _build(conversation_history=history)


def test_history_with_ordinary_extra_keys_is_still_accepted():
    """Extra keys must not be forbidden outright.

    Clients following the OpenAI convention send ``name`` / ``tool_call_id``;
    rejecting those would be a gratuitous break, which is why the bound is on
    size rather than on shape.
    """
    request = _build(
        conversation_history=[
            {"role": "user", "content": "hi", "name": "alice", "tool_call_id": "t-1"}
        ]
    )
    assert request.conversation_history[0]["name"] == "alice"


def test_oversized_history_content_is_refused():
    with pytest.raises(ValidationError):
        _build(
            conversation_history=[
                {"role": "user", "content": "c" * (MAX_MESSAGE_CHARS + 1)}
            ]
        )


def test_too_many_history_messages_are_refused():
    with pytest.raises(ValidationError):
        _build(
            conversation_history=[
                {"role": "user", "content": "hi"}
                for _ in range(MAX_MESSAGES_PER_REQUEST + 1)
            ]
        )


def test_a_realistic_conversation_is_accepted():
    """The count cap is a sanity guard, not a boundary; it must not bite."""
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "hello there " * 20}
        for i in range(MAX_MESSAGES_PER_REQUEST)
    ]
    request = _build(conversation_history=history)
    assert len(request.conversation_history) == MAX_MESSAGES_PER_REQUEST


# --------------------------------------------------------------------------- #
# Keywords
# --------------------------------------------------------------------------- #


def test_too_many_keywords_are_refused():
    with pytest.raises(ValidationError):
        _build(hl_keywords=["k"] * (MAX_KEYWORDS_PER_LIST + 1))


def test_oversized_keyword_is_refused():
    with pytest.raises(ValidationError):
        _build(ll_keywords=["k" * (MAX_KEYWORD_CHARS + 1)])


# --------------------------------------------------------------------------- #
# Numeric knobs — character limits alone do not bound the work
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["top_k", "chunk_top_k"])
def test_retrieval_width_is_bounded(field):
    """A three-character query with ``top_k=10**6`` still puts the retrieval
    volume, and the tokenization that follows it, under the caller's control."""
    _build(**{field: MAX_QUERY_TOP_K})
    with pytest.raises(ValidationError):
        _build(**{field: MAX_QUERY_TOP_K + 1})


@pytest.mark.parametrize(
    "field", ["max_entity_tokens", "max_relation_tokens", "max_total_tokens"]
)
def test_token_budgets_are_bounded(field):
    _build(**{field: MAX_QUERY_TOKEN_BUDGET})
    with pytest.raises(ValidationError):
        _build(**{field: MAX_QUERY_TOKEN_BUDGET + 1})


# --------------------------------------------------------------------------- #
# Regression: the limits must not disturb ordinary requests
# --------------------------------------------------------------------------- #


def test_an_ordinary_request_is_unaffected():
    request = _build(
        query="  what is this about  ",
        top_k=60,
        chunk_top_k=20,
        hl_keywords=["alpha", "beta"],
        conversation_history=[{"role": "user", "content": "earlier question"}],
    )
    assert request.query == "what is this about"  # strip still applies
    assert request.top_k == 60
    assert len(request.query) + len("earlier question") < MAX_REQUEST_TEXT_CHARS
