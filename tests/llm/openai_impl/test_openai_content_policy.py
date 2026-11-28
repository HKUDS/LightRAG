"""Provider content-moderation verdicts fail fast as ContentPolicyViolationError.

Aliyun DashScope hosted Qwen models answer a flagged prompt with HTTP 400
``data_inspection_failed`` (or, mid-stream, an SSE error chunk the openai SDK
raises as ``APIError``); Azure answers 400 ``content_filter``; OpenAI/Azure
also stop generation with ``finish_reason="content_filter"`` on empty AND
non-empty content. All are deterministic verdicts: retrying re-buys the same
rejection, and returning flagged partial text would index or serve it as a
complete answer. Attempt counts call the DECORATED function so the retry
predicate's exclusion of the new type is what the count proves.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIError, BadRequestError
from tenacity import wait_none

from lightrag.exceptions import ContentPolicyViolationError
from lightrag.llm.openai import InvalidResponseError, openai_complete_if_cache

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """Strip the exponential backoff so the retrying control cases stay fast."""
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())


DASHSCOPE_BODY = {
    "code": "data_inspection_failed",
    "type": "data_inspection_failed",
    "param": None,
    "message": "Input data may contain inappropriate content.",
}

AZURE_BODY = {
    "code": "content_filter",
    "innererror": {"code": "ResponsibleAIPolicyViolation"},
    "message": (
        "The response was filtered due to the prompt triggering Azure OpenAI's "
        "content management policy. Please modify your prompt and retry."
    ),
}

GENERIC_BODY = {
    "code": None,
    "type": "invalid_request_error",
    "param": "temperature",
    "message": "Invalid value for 'temperature': must be <= 2.",
}


def _make_bad_request(body: dict) -> BadRequestError:
    request = httpx.Request(
        "POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


def _make_client(payload) -> SimpleNamespace:
    """Fake client whose create() returns a payload or raises an exception."""
    if isinstance(payload, Exception):
        create = AsyncMock(side_effect=payload)
    else:
        create = AsyncMock(return_value=payload)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )


def _make_completion(content: str, finish_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content, parsed=None, reasoning_content=""
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


class _FakeAsyncStream:
    """Async-iterable fake; an Exception item is raised mid-iteration, the way
    the openai SDK raises APIError on a DashScope SSE error chunk."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.aclose = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        item = self._chunks.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_chunk(content=None, finish_reason=None, with_delta=True):
    choice = SimpleNamespace(finish_reason=finish_reason)
    if with_delta:
        choice.delta = SimpleNamespace(content=content, reasoning_content="")
    return SimpleNamespace(choices=[choice], usage=None)


# ---------------------------------------------------------------------------
# HTTP 400 verdicts at create()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, expected_type, reason_fragment",
    [
        pytest.param(
            DASHSCOPE_BODY,
            ContentPolicyViolationError,
            "inappropriate content",
            id="dashscope-data-inspection-failed",
        ),
        pytest.param(
            AZURE_BODY,
            ContentPolicyViolationError,
            "content management policy",
            id="azure-content-filter",
        ),
        # Control: a genuine parameter 400 keeps its own type and fail-fast
        # behaviour; the classifier must not swallow it.
        pytest.param(
            GENERIC_BODY, BadRequestError, "temperature", id="generic-400-control"
        ),
    ],
)
async def test_policy_400_fails_after_one_attempt(body, expected_type, reason_fragment):
    fake_client = _make_client(_make_bad_request(body))

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(expected_type) as excinfo:
            await openai_complete_if_cache(model="qwen-plus", prompt="hello")

    assert fake_client.chat.completions.create.await_count == 1, (
        "a deterministic moderation verdict must not be retried"
    )
    # The provider's own reason survives into the error (and doc_status).
    assert reason_fragment in str(excinfo.value)
    fake_client.close.assert_awaited()


# ---------------------------------------------------------------------------
# finish_reason="content_filter" on the non-streaming path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish_reason, expected_type, attempts",
    [
        pytest.param(
            "content_filter",
            ContentPolicyViolationError,
            1,
            id="policy-stop-one-attempt",
        ),
        # Control: a genuinely empty response is a sampling artifact and keeps
        # its three retryable attempts.
        pytest.param("stop", InvalidResponseError, 3, id="empty-stop-retried"),
    ],
)
async def test_empty_content_finish_reason_selects_retry_policy(
    finish_reason, expected_type, attempts
):
    fake_client = _make_client(_make_completion("", finish_reason))

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(expected_type) as excinfo:
            await openai_complete_if_cache(model="qwen-plus", prompt="hello")

    assert fake_client.chat.completions.create.await_count == attempts
    assert f"finish_reason={finish_reason}" in str(excinfo.value)
    fake_client.close.assert_awaited()


async def test_non_empty_content_filter_raises_instead_of_returning_flagged_text():
    """A policy stop is not caller-budget truncation: the flagged tail must
    never be returned (or cached) as a complete answer, and the error names
    the stop without quoting the flagged content."""
    content = "flagged partial answer"
    fake_client = _make_client(_make_completion(content, "content_filter"))

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            await openai_complete_if_cache(model="qwen-plus", prompt="hello")

    assert fake_client.chat.completions.create.await_count == 1
    message = str(excinfo.value)
    assert "finish_reason=content_filter" in message
    assert f"content_len={len(content)}" in message
    assert content not in message, "flagged content must not be logged"
    fake_client.close.assert_awaited()


# ---------------------------------------------------------------------------
# Streaming verdicts
# ---------------------------------------------------------------------------


async def test_streaming_content_filter_terminal_raises_after_partial_text():
    """The terminal chunk carries finish_reason with NO delta, so the check
    must run before the no-delta skip; text yielded before it is preserved."""
    stream = _FakeAsyncStream(
        [
            _make_chunk(content="par"),
            _make_chunk(finish_reason="content_filter", with_delta=False),
        ]
    )
    fake_client = _make_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache(
            model="qwen-plus", prompt="hello", stream=True
        )
        chunks: list[str] = []
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            async for chunk in gen:
                chunks.append(chunk)

    assert "".join(chunks) == "par"
    assert "content_filter" in str(excinfo.value)
    assert fake_client.chat.completions.create.await_count == 1
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()


async def test_streaming_stop_terminal_with_empty_delta_completes():
    """Control: a normal stop terminal (finish_reason=stop, no delta) ends the
    stream successfully; only content_filter raises."""
    stream = _FakeAsyncStream(
        [
            _make_chunk(content="ok"),
            _make_chunk(finish_reason="stop", with_delta=False),
        ]
    )
    fake_client = _make_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache(
            model="qwen-plus", prompt="hello", stream=True
        )
        result = "".join([chunk async for chunk in gen])

    assert result == "ok"
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()


async def test_streaming_mid_stream_policy_apierror_raises():
    """DashScope streaming moderation arrives as HTTP 200 + an SSE error chunk
    AFTER partial content; the openai SDK raises APIError (not BadRequestError)
    mid-iteration, so the streaming except branch must classify it too."""
    request = httpx.Request(
        "POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    error = APIError(
        f"Error code: 400 - {DASHSCOPE_BODY}", request=request, body=DASHSCOPE_BODY
    )
    stream = _FakeAsyncStream([_make_chunk(content="par"), error])
    fake_client = _make_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache(
            model="qwen-plus", prompt="hello", stream=True
        )
        chunks: list[str] = []
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            async for chunk in gen:
                chunks.append(chunk)

    assert "".join(chunks) == "par"
    assert "inappropriate content" in str(excinfo.value)
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()
