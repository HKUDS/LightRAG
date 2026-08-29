"""Offline tests for the OpenAI Responses (``/v1/responses``) LLM binding.

The driver is exercised against a fake client so the whole file runs in the
offline CI job with no network and no API key.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from tenacity import RetryError

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.llm.openai_responses import (
    InvalidResponseError,
    openai_responses_complete_if_cache,
)
from lightrag.utils import TruncatedResponse, is_truncated_response

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _FakeResponses:
    """Captures the create() call and returns a canned payload."""

    def __init__(self, payload: Any):
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    def __init__(self, payload: Any):
        self.responses = _FakeResponses(payload)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _reasoning(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _response(
    *,
    output: list[Any] | None = None,
    status: str = "completed",
    incomplete_reason: str | None = None,
    usage: Any = None,
    output_text: str | None = None,
) -> SimpleNamespace:
    payload = SimpleNamespace(
        output=output or [],
        status=status,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        ),
        usage=usage,
    )
    # `output_text` is an SDK convenience property; omitting it exercises the
    # fallback walk over output[].
    if output_text is not None:
        payload.output_text = output_text
    return payload


def _usage(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


def _patch_client(monkeypatch, payload: Any) -> _FakeClient:
    client = _FakeClient(payload)
    monkeypatch.setattr(
        "lightrag.llm.openai_responses.create_openai_async_client",
        lambda **kwargs: client,
    )
    return client


async def test_returns_output_text_and_closes_client(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("hello")]))

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert result == "hello"
    assert client.closed, "client must be closed on the non-streaming path"


async def test_system_prompt_is_sent_as_instructions(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache("gpt-5.6", "hi", system_prompt="be terse")

    call = client.responses.calls[0]
    assert call["instructions"] == "be terse"
    # Responses has no system role; a leaked system message would be a bug.
    assert all(item["role"] != "system" for item in call["input"])


async def test_history_assistant_turn_uses_output_text_part(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache(
        "gpt-5.6",
        "next",
        history_messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ],
    )

    api_input = client.responses.calls[0]["input"]
    assert api_input[0]["content"][0]["type"] == "input_text"
    # The API rejects an assistant turn tagged as input_text.
    assert api_input[1]["content"][0]["type"] == "output_text"
    assert api_input[-1]["content"][0]["type"] == "input_text"


@pytest.mark.parametrize(
    "alias", ["max_output_tokens", "max_completion_tokens", "max_tokens"]
)
async def test_output_budget_aliases_map_to_max_output_tokens(monkeypatch, alias):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache("gpt-5.6", "hi", **{alias: 321})

    call = client.responses.calls[0]
    assert call["max_output_tokens"] == 321
    # The Chat Completions spellings must not reach the Responses API.
    assert "max_completion_tokens" not in call
    assert "max_tokens" not in call


async def test_reasoning_effort_becomes_nested_reasoning(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache("gpt-5.6", "hi", reasoning_effort="low")

    call = client.responses.calls[0]
    assert call["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in call


async def test_explicit_reasoning_dict_wins_over_flat_alias(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", reasoning={"effort": "high"}, reasoning_effort="low"
    )

    assert client.responses.calls[0]["reasoning"] == {"effort": "high"}


async def test_response_format_becomes_text_format(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message('{"a":1}')]))

    await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", response_format={"type": "json_object"}
    )

    call = client.responses.calls[0]
    assert call["text"] == {"format": {"type": "json_object"}}
    assert "response_format" not in call


async def test_unsupported_chat_completions_params_are_dropped(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", frequency_penalty=0.5, presence_penalty=0.2, n=2
    )

    call = client.responses.calls[0]
    for dropped in ("frequency_penalty", "presence_penalty", "n"):
        assert dropped not in call, f"{dropped} would be rejected by the API"


async def test_cot_wraps_reasoning_when_body_is_empty(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(output=[_reasoning("thinking hard"), _message("")]),
    )

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", enable_cot=True)

    assert result == "<think>thinking hard</think>"


async def test_cot_is_ignored_when_body_text_present(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(output=[_reasoning("trace"), _message("answer")]),
    )

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", enable_cot=True)

    assert result == "answer", "reasoning is dropped once body text exists"


async def test_structured_output_disables_cot(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(output=[_reasoning("trace"), _message('{"a": 1}')]),
    )

    result = await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        enable_cot=True,
        response_format={"type": "json_object"},
    )

    # <think> tags would corrupt JSON the caller is about to parse.
    assert result == '{"a": 1}'


async def test_token_tracker_maps_responses_usage_keys(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(output=[_message("ok")], usage=_usage(11, 7)),
    )
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    await openai_responses_complete_if_cache("gpt-5.6", "hi", token_tracker=tracker)

    assert recorded == [
        {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    ]


async def test_truncated_output_is_wrapped_for_the_cache_layer(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(
            output=[_message('{"partial": ')],
            status="incomplete",
            incomplete_reason="max_output_tokens",
        ),
    )

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert isinstance(result, TruncatedResponse)
    assert is_truncated_response(result), "cache layer must skip persisting this"
    assert str(result) == '{"partial": '


async def test_empty_truncated_output_raises_non_retryable(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(
            output=[],
            status="incomplete",
            incomplete_reason="max_output_tokens",
            usage=_usage(10, 500, reasoning_tokens=500),
        ),
    )

    # Budget exhaustion is deterministic, so it must not buy two more retries.
    with pytest.raises(EmptyTruncatedResponseError) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert "max_output_tokens" in str(excinfo.value)


async def test_empty_output_without_truncation_raises_retryable(monkeypatch):
    _patch_client(monkeypatch, _response(output=[_message("")]))

    with pytest.raises((InvalidResponseError, Exception)) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert not isinstance(excinfo.value, EmptyTruncatedResponseError)


async def test_usage_is_counted_even_when_output_is_unusable(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(
            output=[],
            status="incomplete",
            incomplete_reason="max_output_tokens",
            usage=_usage(9, 400, reasoning_tokens=400),
        ),
    )
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    with pytest.raises(EmptyTruncatedResponseError):
        await openai_responses_complete_if_cache("gpt-5.6", "hi", token_tracker=tracker)

    # The request burned its budget whether or not the output was usable.
    assert recorded == [
        {"prompt_tokens": 9, "completion_tokens": 400, "total_tokens": 409}
    ]


async def test_output_text_property_is_preferred(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(output=[_message("from-walk")], output_text="from-property"),
    )

    assert await openai_responses_complete_if_cache("gpt-5.6", "hi") == (
        "from-property"
    )


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, events: list[Any]):
        self._events = events
        self.aclosed = False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()

    async def aclose(self) -> None:
        self.aclosed = True


def _delta(event_type: str, delta: str) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, delta=delta)


async def _collect(agen) -> str:
    return "".join([chunk async for chunk in agen])


async def test_streaming_yields_output_text_deltas(monkeypatch):
    stream = _FakeStream(
        [
            _delta("response.output_text.delta", "hel"),
            _delta("response.output_text.delta", "lo"),
        ]
    )
    client = _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    assert await _collect(result) == "hello"
    assert stream.aclosed
    assert client.closed


async def test_streaming_wraps_reasoning_in_think_tags(monkeypatch):
    stream = _FakeStream(
        [
            _delta("response.reasoning_summary_text.delta", "step one"),
            _delta("response.output_text.delta", "answer"),
        ]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )

    # The reasoning block must be closed as soon as body text starts.
    assert await _collect(result) == "<think>step one</think>answer"


async def test_streaming_closes_think_tag_when_no_body_text_follows(monkeypatch):
    stream = _FakeStream(
        [_delta("response.reasoning_summary_text.delta", "only thinking")]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )

    assert await _collect(result) == "<think>only thinking</think>"


async def test_streaming_drops_reasoning_when_cot_disabled(monkeypatch):
    stream = _FakeStream(
        [
            _delta("response.reasoning_summary_text.delta", "trace"),
            _delta("response.output_text.delta", "answer"),
        ]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    assert await _collect(result) == "answer"


async def test_streaming_records_usage_from_completed_event(monkeypatch):
    stream = _FakeStream(
        [
            _delta("response.output_text.delta", "ok"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=_usage(3, 4)),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, token_tracker=tracker
    )
    await _collect(result)

    assert recorded == [{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}]


async def test_streaming_records_usage_from_incomplete_event(monkeypatch):
    stream = _FakeStream(
        [
            _delta("response.output_text.delta", "partial"),
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    usage=_usage(3, 4),
                ),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, token_tracker=tracker
    )

    assert await _collect(result) == "partial", "partial output is kept"
    assert recorded == [{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}]


async def test_streaming_empty_truncation_raises_non_retryable(monkeypatch):
    stream = _FakeStream(
        [
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    usage=_usage(3, 120, reasoning_tokens=120),
                ),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    with pytest.raises(EmptyTruncatedResponseError):
        await _collect(result)


async def test_streaming_incomplete_without_budget_reason_does_not_raise(monkeypatch):
    """`response.incomplete` also covers non-budget stops (e.g. content filter),
    which are not token-limit truncation and must not take the raise path."""
    stream = _FakeStream(
        [
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="content_filter"),
                    usage=_usage(3, 0),
                ),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    assert await _collect(result) == ""


async def test_streaming_reasoning_only_truncation_is_not_treated_as_empty(monkeypatch):
    """Reasoning text already reached the consumer inside <think>, so the
    stream ends normally rather than raising."""
    stream = _FakeStream(
        [
            _delta("response.reasoning_summary_text.delta", "thinking"),
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    usage=_usage(3, 9, reasoning_tokens=9),
                ),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )

    assert await _collect(result) == "<think>thinking</think>"


async def test_streaming_failed_terminal_raises_with_the_provider_reason(monkeypatch):
    """`response.failed` is neither a completed/incomplete terminal nor the
    separate `error` event, so an unhandled one would end the iterator
    successfully on the partial text it had already yielded."""
    stream = _FakeStream(
        [
            _delta("response.output_text.delta", "partial"),
            SimpleNamespace(
                type="response.failed",
                response=SimpleNamespace(
                    status="failed",
                    error=SimpleNamespace(
                        code="server_error", message="upstream exploded"
                    ),
                    usage=_usage(3, 4),
                ),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    with pytest.raises(InvalidResponseError) as excinfo:
        await _collect(result)
    assert "server_error" in str(excinfo.value)
    assert "upstream exploded" in str(excinfo.value)


async def test_streaming_failed_terminal_still_counts_usage(monkeypatch):
    """The attempt was billed whether or not it produced a usable answer,
    matching how the truncated terminal accounts before raising."""
    stream = _FakeStream(
        [
            SimpleNamespace(
                type="response.failed",
                response=SimpleNamespace(
                    status="failed",
                    error=SimpleNamespace(code="server_error", message="boom"),
                    usage=_usage(3, 4),
                ),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, token_tracker=tracker
    )

    with pytest.raises(InvalidResponseError):
        await _collect(result)
    assert recorded == [{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}]


async def test_streaming_failed_terminal_closes_an_open_think_tag(monkeypatch):
    """Reasoning already opened `<think>`; the error path must still close it
    so the consumer is not left holding an unterminated tag."""
    chunks: list[str] = []
    stream = _FakeStream(
        [
            _delta("response.reasoning_summary_text.delta", "thinking"),
            SimpleNamespace(
                type="response.failed",
                response=SimpleNamespace(
                    status="failed",
                    error=SimpleNamespace(code="server_error", message="boom"),
                    usage=None,
                ),
            ),
        ]
    )
    _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )

    with pytest.raises(InvalidResponseError):
        async for chunk in result:
            chunks.append(chunk)
    assert "".join(chunks) == "<think>thinking</think>"


async def _non_streaming_failure(monkeypatch, payload, **kwargs) -> BaseException:
    """Return the exception a non-streaming call ends on.

    `InvalidResponseError` is in the driver's retry set, so the decorator
    exhausts its attempts and re-raises as `RetryError`; the real error is only
    reachable through the last attempt.
    """
    _patch_client(monkeypatch, payload)

    with pytest.raises(RetryError) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi", **kwargs)
    return excinfo.value.last_attempt.exception()


async def test_failed_status_raises_rather_than_reporting_empty_output(monkeypatch):
    """The non-streaming sibling of `response.failed`: a request accepted at
    the transport layer that failed during generation comes back as a payload
    with status "failed", not as an exception from responses.create()."""
    payload = _response(status="failed")
    payload.error = SimpleNamespace(code="server_error", message="upstream exploded")

    error = await _non_streaming_failure(monkeypatch, payload)

    assert isinstance(error, InvalidResponseError)
    assert "server_error" in str(error)
    assert "upstream exploded" in str(error)


async def test_failed_status_is_not_masked_by_partial_output(monkeypatch):
    """Content on a failed payload is the tail of a generation that did not
    finish, so returning it would present a failure as a short answer."""
    payload = _response(output=[_message("partial")], status="failed")
    payload.error = SimpleNamespace(code="server_error", message="boom")

    error = await _non_streaming_failure(monkeypatch, payload)

    assert isinstance(error, InvalidResponseError)


async def test_failed_status_counts_usage_before_raising(monkeypatch):
    payload = _response(status="failed", usage=_usage(5, 2))
    payload.error = SimpleNamespace(code="server_error", message="boom")
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    await _non_streaming_failure(monkeypatch, payload, token_tracker=tracker)

    # One entry per attempt: each retry is a separately billed request.
    assert recorded
    assert all(
        counts == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        for counts in recorded
    )
