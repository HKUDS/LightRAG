"""``disable_user_prompt_prefix`` on QueryRequest, and what it must NOT change.

The switch is the client's only handle on the server-side prompt prefix: a
request may opt out of it, never read or replace it. It rides the existing
``to_query_params`` model_dump with no special casing, so these tests pin the
three-state ``Optional[bool]`` behaviour and the fact that the operator's
prefix stays outside the client-supplied text budget.

Like ``test_query_input_limits``, this validates the model rather than driving
HTTP: ``QueryRequest`` is what the three query routes share.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# lightrag.api.config parses argv at import time, so the import has to be guarded
# the way the rest of the API tests guard it.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _query_routes.QueryRequest

from lightrag.constants import (  # noqa: E402
    MAX_QUERY_CHARS,
    MAX_REQUEST_TEXT_CHARS,
)

pytestmark = pytest.mark.offline


def _build(**overrides) -> QueryRequest:
    payload = {"query": "what is this about"}
    payload.update(overrides)
    return QueryRequest(**payload)


# ---------------------------------------------------------------------------
# Three-state Optional[bool] -> QueryParam bool.
# ---------------------------------------------------------------------------


def test_omitted_switch_falls_back_to_the_queryparam_default():
    """Absent means "prefix applies" -- the upgrade-safe default.

    ``exclude_none=True`` drops the field, so ``QueryParam`` supplies ``False``.
    This is the path every existing client and every WebUI install that has not
    yet written the new setting takes.
    """
    param = _build().to_query_params(is_stream=False)
    assert param.disable_user_prompt_prefix is False


@pytest.mark.parametrize("value", [True, False])
def test_explicit_switch_is_forwarded(value):
    param = _build(disable_user_prompt_prefix=value).to_query_params(is_stream=False)
    assert param.disable_user_prompt_prefix is value


def test_switch_survives_alongside_a_user_prompt():
    param = _build(
        user_prompt="Reply in one sentence.", disable_user_prompt_prefix=True
    ).to_query_params(is_stream=False)
    assert param.user_prompt == "Reply in one sentence."
    assert param.disable_user_prompt_prefix is True


def test_switch_is_not_leaked_as_a_queryparam_prefix_field():
    """A request must have no way to SET the prefix, only to opt out of it."""
    param = _build().to_query_params(is_stream=False)
    assert not hasattr(param, "user_prompt_prefix")


# ---------------------------------------------------------------------------
# The operator prefix stays outside the client text budget.
# ---------------------------------------------------------------------------


def test_request_at_the_text_budget_still_validates():
    """The aggregate budget bounds client text only.

    If the server-side prefix were folded into ``bound_aggregate_text``, a
    request that is legal today would start failing the moment an operator
    edited ``USER_PROMPT_PREFIX`` -- with nothing about the request having
    changed. This pins that a maximal request is accepted regardless.
    """
    # Both text fields at their own per-field caps, summing to exactly the
    # aggregate budget: the largest request the API accepts at all.
    query = "x" * MAX_QUERY_CHARS
    user_prompt = "y" * (MAX_REQUEST_TEXT_CHARS - MAX_QUERY_CHARS)
    request = _build(query=query, user_prompt=user_prompt)
    assert len(request.query) + len(request.user_prompt) == MAX_REQUEST_TEXT_CHARS
    param = request.to_query_params(is_stream=False)
    assert param.disable_user_prompt_prefix is False
