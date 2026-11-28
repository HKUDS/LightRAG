"""Offline tests for Bedrock moderation verdict classification.

``content_filtered`` is the model's Responsible-AI refusal (it can arrive with
no message content at all) and ``guardrail_intervened`` is a guardrail
intervention whose assessments decide whether the masked output is usable
(ANONYMIZED) or must be rejected (any BLOCKED, or no readable trace). Both are
deterministic verdicts: the raised ContentPolicyViolationError must survive
``_handle_bedrock_exception`` and cost exactly one attempt.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tenacity import wait_none

from lightrag.exceptions import ContentPolicyViolationError
from lightrag.llm.bedrock import (
    _handle_bedrock_exception,
    bedrock_complete_if_cache,
)
from lightrag.utils import TokenTracker

pytestmark = pytest.mark.offline

_BLOCKED_TEXT = "Sorry, I can't answer questions about heavy metal music."


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    monkeypatch.setattr(bedrock_complete_if_cache.retry, "wait", wait_none())


class _ConverseClient:
    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **kwargs):
        self._calls.append(kwargs)
        return self._response


def _converse_patch(response, calls):
    client = _ConverseClient(response, calls)
    session = SimpleNamespace(client=lambda *_args, **_kwargs: client)
    return patch("lightrag.llm.bedrock.aioboto3.Session", return_value=session)


def _content_filtered_response(content):
    return {
        "output": {"message": {"content": content}},
        "stopReason": "content_filtered",
        "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
    }


def _guardrail_response(content, trace):
    response = {
        "output": {"message": {"content": content}},
        "stopReason": "guardrail_intervened",
        "usage": {"inputTokens": 9, "outputTokens": 11, "totalTokens": 20},
    }
    if trace is not None:
        response["trace"] = {"guardrail": trace}
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        pytest.param([], id="empty-content"),
        pytest.param([{"text": "flagged output"}], id="flagged-text"),
    ],
)
async def test_content_filtered_raises_content_policy_error(content):
    """Raised whether or not text survived, and before the empty-content
    validation can mask the verdict as a generic BedrockError."""
    calls: list[dict] = []

    with _converse_patch(_content_filtered_response(content), calls):
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            await bedrock_complete_if_cache(model="bedrock-model", prompt="hello")

    message = str(excinfo.value)
    assert "stopReason=content_filtered" in message
    assert "flagged output" not in message
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_content_filtered_still_records_billed_usage():
    """The API call returned (and was billed) before the verdict raised."""
    tracker = TokenTracker()
    calls: list[dict] = []

    with _converse_patch(_content_filtered_response([]), calls):
        with pytest.raises(ContentPolicyViolationError):
            await bedrock_complete_if_cache(
                model="bedrock-model", prompt="hello", token_tracker=tracker
            )

    assert tracker.get_usage() == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "call_count": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trace", "expected_actions"),
    [
        pytest.param(
            {
                "inputAssessment": {
                    "gr-1": {
                        "topicPolicy": {
                            "topics": [
                                {
                                    "name": "Heavy metal",
                                    "type": "DENY",
                                    "action": "BLOCKED",
                                }
                            ]
                        }
                    }
                }
            },
            "actions=['BLOCKED']",
            id="input-topic-blocked",
        ),
        pytest.param(
            {
                "outputAssessments": {
                    "gr-1": [
                        {
                            "contentPolicy": {
                                "filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]
                            }
                        }
                    ]
                }
            },
            "actions=['BLOCKED']",
            id="output-content-blocked",
        ),
        pytest.param(
            {
                "inputAssessment": {
                    "gr-1": {
                        "sensitiveInformationPolicy": {
                            "piiEntities": [{"type": "NAME", "action": "ANONYMIZED"}]
                        }
                    }
                },
                "outputAssessments": {
                    "gr-1": [
                        {
                            "contentPolicy": {
                                "filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]
                            }
                        }
                    ]
                },
            },
            "actions=['ANONYMIZED', 'BLOCKED']",
            id="mixed-anonymized-and-blocked",
        ),
        pytest.param(None, "actions=[]", id="missing-trace"),
    ],
)
async def test_guardrail_intervention_without_anonymization_raises(
    trace, expected_actions
):
    calls: list[dict] = []

    with _converse_patch(_guardrail_response([{"text": _BLOCKED_TEXT}], trace), calls):
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            await bedrock_complete_if_cache(model="bedrock-model", prompt="hello")

    message = str(excinfo.value)
    assert "stopReason=guardrail_intervened" in message
    assert expected_actions in message
    assert _BLOCKED_TEXT not in message
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_guardrail_anonymize_returns_masked_content():
    """ANONYMIZE is a usable response, not a rejection (AWS caveat)."""
    trace = {
        "inputAssessment": {
            "gr-1": {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"match": "John Doe", "action": "ANONYMIZED"}]
                }
            }
        },
        "outputAssessments": {
            "gr-1": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [{"match": "John Doe", "action": "ANONYMIZED"}]
                    }
                }
            ]
        },
    }
    calls: list[dict] = []

    with _converse_patch(
        _guardrail_response([{"text": "Hello [NAME]."}], trace), calls
    ):
        result = await bedrock_complete_if_cache(model="bedrock-model", prompt="hello")

    assert result == "Hello [NAME]."
    assert len(calls) == 1


def test_handle_bedrock_exception_preserves_content_policy_error():
    error = ContentPolicyViolationError("Bedrock content moderation rejected")

    # The handler's pass-through branch is a bare ``raise``, so it must be
    # called from an active handler exactly as both call sites do.
    try:
        raise error
    except ContentPolicyViolationError as caught_error:
        with pytest.raises(ContentPolicyViolationError) as caught:
            _handle_bedrock_exception(caught_error, "Bedrock converse")

    assert caught.value is error


class _EventStream:
    def __init__(self, events):
        self._events = iter(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def close(self):
        self.closed = True


class _StreamingClient:
    def __init__(self, events, calls):
        self._events = events
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse_stream(self, **kwargs):
        self._calls.append(kwargs)
        return {"stream": _EventStream(self._events)}


def _stream_patch(events, calls):
    client = _StreamingClient(events, calls)
    session = SimpleNamespace(client=lambda *_args, **_kwargs: client)
    return patch("lightrag.llm.bedrock.aioboto3.Session", return_value=session)


@pytest.mark.asyncio
async def test_streaming_content_filtered_raises_after_partial_text():
    calls: list[dict] = []
    events = [
        {"contentBlockDelta": {"delta": {"text": "partial"}}},
        {"messageStop": {"stopReason": "content_filtered"}},
    ]

    with _stream_patch(events, calls):
        stream = await bedrock_complete_if_cache(
            model="bedrock-model", prompt="hello", stream=True
        )
        chunks = []
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            async for chunk in stream:
                chunks.append(chunk)

    assert chunks == ["partial"]
    assert "stopReason=content_filtered" in str(excinfo.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_streaming_guardrail_blocked_raises_from_metadata_trace():
    calls: list[dict] = []
    events = [
        {"contentBlockDelta": {"delta": {"text": _BLOCKED_TEXT}}},
        {"messageStop": {"stopReason": "guardrail_intervened"}},
        {
            "metadata": {
                "trace": {
                    "guardrail": {
                        "inputAssessment": {
                            "gr-1": {
                                "topicPolicy": {
                                    "topics": [
                                        {
                                            "name": "Heavy metal",
                                            "type": "DENY",
                                            "action": "BLOCKED",
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
    ]

    with _stream_patch(events, calls):
        stream = await bedrock_complete_if_cache(
            model="bedrock-model", prompt="hello", stream=True
        )
        chunks = []
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            async for chunk in stream:
                chunks.append(chunk)

    assert chunks == [_BLOCKED_TEXT]
    assert "actions=['BLOCKED']" in str(excinfo.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_streaming_guardrail_anonymize_returns_masked_text():
    calls: list[dict] = []
    events = [
        {"contentBlockDelta": {"delta": {"text": "Hello [NAME]."}}},
        {"messageStop": {"stopReason": "guardrail_intervened"}},
        {
            "metadata": {
                "trace": {
                    "guardrail": {
                        "inputAssessment": {
                            "gr-1": {
                                "sensitiveInformationPolicy": {
                                    "piiEntities": [
                                        {"match": "John Doe", "action": "ANONYMIZED"}
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
    ]

    with _stream_patch(events, calls):
        stream = await bedrock_complete_if_cache(
            model="bedrock-model", prompt="hello", stream=True
        )
        chunks = [chunk async for chunk in stream]

    assert chunks == ["Hello [NAME]."]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_streaming_guardrail_without_metadata_fails_closed():
    calls: list[dict] = []
    events = [
        {"contentBlockDelta": {"delta": {"text": "unverified"}}},
        {"messageStop": {"stopReason": "guardrail_intervened"}},
    ]

    with _stream_patch(events, calls):
        stream = await bedrock_complete_if_cache(
            model="bedrock-model", prompt="hello", stream=True
        )
        chunks = []
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            async for chunk in stream:
                chunks.append(chunk)

    assert chunks == ["unverified"]
    assert "actions=[]" in str(excinfo.value)
    assert len(calls) == 1
