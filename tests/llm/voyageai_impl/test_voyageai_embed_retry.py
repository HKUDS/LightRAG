"""Regression tests for retry/reraise behavior of ``voyageai_embed``.

``voyageai_embed`` is decorated with tenacity's ``@retry`` but, unlike every
other LLM/embedding binding in ``lightrag/llm/`` (see ``openai.py``,
``jina.py``, ``ollama.py``, ``gemini.py``, ``zhipu.py``,
``openai_responses.py``, ``lmdeploy.py``, ``lollms.py``), it does not pass
``reraise=True``. tenacity's default behavior, when every attempt is
exhausted, is to raise its own ``tenacity.RetryError`` wrapping the last
attempt instead of re-raising the last attempt's own exception. That hides
the real VoyageAI error (type and message) from any caller that catches a
specific exception (e.g. ``voyageai.error.RateLimitError``) or logs the
error text, and is inconsistent with the documented convention used
everywhere else in the codebase.

Fix-proof: the persistent-failure test fails on the pre-fix code because it
raises ``tenacity.RetryError`` instead of the original ``RateLimitError``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import voyageai.error as voyageai_error
from tenacity import wait_none

from lightrag.llm.voyageai import voyageai_embed

pytestmark = pytest.mark.offline


@pytest.fixture
def no_retry_wait(monkeypatch):
    """Drop the 4-60s exponential backoff for the duration of one test."""
    retrying = getattr(voyageai_embed.func, "retry", None)
    if retrying is not None:
        monkeypatch.setattr(retrying, "wait", wait_none())


def _make_flaky_client(*, error: Exception, failures: int):
    """A fake AsyncClient whose embed() raises `error` `failures` times then
    succeeds, mirroring a transient VoyageAI rate limit that clears up
    before the caller gives up."""
    calls = {"count": 0}

    async def embed(**kwargs):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise error
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])

    return SimpleNamespace(embed=embed), calls


@pytest.mark.asyncio
async def test_transient_rate_limit_is_retried(no_retry_wait):
    """A rate limit that clears on retry must still succeed."""
    error = voyageai_error.RateLimitError("rate limited")
    fake_client, calls = _make_flaky_client(error=error, failures=1)

    with patch("lightrag.llm.voyageai.voyageai.AsyncClient", return_value=fake_client):
        result = await voyageai_embed.func(["hello"], api_key="fake")

    assert np.allclose(result, np.array([[0.1, 0.2, 0.3]]))
    assert calls["count"] == 2, "a transient rate limit should be retried"


@pytest.mark.asyncio
async def test_persistent_rate_limit_reraises_original_error_after_three_attempts(
    no_retry_wait,
):
    """Fix-proof: three exhausted attempts must surface the ORIGINAL
    ``RateLimitError`` -- not a ``tenacity.RetryError``, which would hide the
    real error type and message from any caller catching it specifically or
    logging ``str(exc)``."""
    error = voyageai_error.RateLimitError("rate limited, try again later")
    fake_client, calls = _make_flaky_client(error=error, failures=10)

    with patch("lightrag.llm.voyageai.voyageai.AsyncClient", return_value=fake_client):
        with pytest.raises(voyageai_error.RateLimitError) as exc_info:
            await voyageai_embed.func(["hello"], api_key="fake")

    assert type(exc_info.value) is voyageai_error.RateLimitError
    assert "rate limited, try again later" in str(exc_info.value)
    assert calls["count"] == 3, "a persistent rate limit stops at attempt 3"
