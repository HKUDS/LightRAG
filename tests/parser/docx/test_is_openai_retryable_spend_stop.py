"""``is_openai_retryable`` must classify permanent spend stops as non-retryable.

The helper's documented contract used to be "429 -> always retry", which is
wrong for the two 429s that no backoff can clear: a LiteLLM Proxy
``budget_exceeded`` and an OpenAI ``insufficient_quota``. It shares the
classifier with the openai binding's tenacity predicate
(``lightrag.llm._error_utils.is_permanent_rate_limit_error``) so the two cannot
drift apart.
"""

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from lightrag.parser.docx.utils import is_openai_retryable

pytestmark = pytest.mark.offline

BUDGET_BODY = {
    "message": "Budget has been exceeded! Team=abc Max budget: 180.0",
    "type": "budget_exceeded",
    "code": "429",
}
QUOTA_BODY = {
    "message": "You exceeded your current quota.",
    "type": "insufficient_quota",
    "code": "insufficient_quota",
}
THROTTLE_BODY = {
    "message": "Rate limit reached. Please try again in 1s.",
    "type": "requests",
    "code": "rate_limit_exceeded",
}


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://proxy.example/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(BUDGET_BODY, False, id="budget-exceeded"),
        pytest.param(QUOTA_BODY, False, id="insufficient-quota"),
        pytest.param(THROTTLE_BODY, True, id="throughput-throttle"),
        pytest.param(None, True, id="no-body"),
    ],
)
def test_rate_limit_error_classification(body, expected):
    err = RateLimitError(
        f"Error code: 429 - {body}", response=_response(429), body=body
    )
    assert is_openai_retryable(err) is expected


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(BUDGET_BODY, False, id="budget-exceeded"),
        pytest.param(THROTTLE_BODY, True, id="throughput-throttle"),
    ],
)
def test_bare_429_status_error_classification(body, expected):
    """A proxy may surface a 429 as a plain APIStatusError, not RateLimitError."""
    err = APIStatusError(
        f"Error code: 429 - {body}", response=_response(429), body=body
    )
    assert is_openai_retryable(err) is expected


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_server_errors_still_retryable(status_code):
    err = APIStatusError("boom", response=_response(status_code), body=None)
    assert is_openai_retryable(err) is True
