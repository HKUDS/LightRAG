"""Retry policy, lifecycle and request-parameter defaults of the Responses driver.

These pin the places where the Responses driver must stay aligned with the
Chat Completions driver (shared retry predicates, ``reraise=True``, client
cleanup on cancellation and consumer disconnect) and the request defaults the
driver adds on top of the raw API (``store``, ``reasoning.summary``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIStatusError, InternalServerError, RateLimitError
from tenacity import wait_none

from lightrag.llm.binding_options import OpenAIResponsesLLMOptions
from lightrag.llm.openai_responses import (
    InvalidResponseError,
    ResponseRefusedError,
    openai_responses_complete_if_cache,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

_URL = "https://api.openai.com/v1/responses"


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    monkeypatch.setattr(openai_responses_complete_if_cache.retry, "wait", wait_none())


class _SequencedResponses:
    """``responses.create`` that plays a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = (
            self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Client:
    def __init__(self, responses: _SequencedResponses):
        self.responses = responses
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _install(monkeypatch, outcomes: list[Any]) -> tuple[_SequencedResponses, list]:
    """Every tenacity attempt builds a fresh client; record all of them."""
    responses = _SequencedResponses(outcomes)
    clients: list[_Client] = []

    def factory(**kwargs: Any) -> _Client:
        client = _Client(responses)
        clients.append(client)
        return client

    monkeypatch.setattr(
        "lightrag.llm.openai_responses.create_openai_async_client", factory
    )
    return responses, clients


def _ok(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        status="completed",
        incomplete_details=None,
        usage=None,
    )


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", _URL)
    response = httpx.Response(status_code=status_code, request=request)
    return APIStatusError(f"Error code: {status_code}", response=response, body=None)


def _rate_limit(body: dict[str, Any]) -> RateLimitError:
    request = httpx.Request("POST", _URL)
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError(f"Error code: 429 - {body}", response=response, body=body)


QUOTA_BODY = {
    "message": "You exceeded your current quota, please check your plan.",
    "type": "insufficient_quota",
    "param": None,
    "code": "insufficient_quota",
}

THROTTLE_BODY = {
    "message": "Rate limit reached. Please try again in 1s.",
    "type": "requests",
    "param": None,
    "code": "rate_limit_exceeded",
}


# --------------------------------------------------------------------------
# Retry policy (shared with the Chat Completions driver)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [408, 409])
async def test_transient_status_is_retried(monkeypatch, status_code):
    """The client is built with ``max_retries=0``, so 408 / 409 — which the
    SDK loop used to retry — are retried only if tenacity covers them."""
    responses, _ = _install(monkeypatch, [_status_error(status_code), _ok("done")])

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert result == "done"
    assert len(responses.calls) == 2


async def test_permanent_spend_stop_fails_on_the_first_attempt(monkeypatch):
    responses, _ = _install(monkeypatch, [_rate_limit(QUOTA_BODY)])

    with pytest.raises(RateLimitError) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert len(responses.calls) == 1
    assert "insufficient_quota" in str(excinfo.value)


async def test_exhausted_retries_raise_the_provider_error_not_retry_error(
    monkeypatch,
):
    """``reraise=True``: the caller and the doc-status ``error_msg`` get the
    provider's own exception, not a ``tenacity.RetryError`` hiding it."""
    responses, _ = _install(monkeypatch, [_rate_limit(THROTTLE_BODY)])

    with pytest.raises(RateLimitError) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert len(responses.calls) == 3
    assert "rate_limit_exceeded" in str(excinfo.value)


async def test_refusal_is_not_retried(monkeypatch):
    """A refusal is a verdict on the prompt; retrying buys the same verdict."""
    refused = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="refusal", refusal="I can't help")],
            )
        ],
        status="completed",
        incomplete_details=None,
        usage=None,
    )
    responses, _ = _install(monkeypatch, [refused])

    with pytest.raises(ResponseRefusedError):
        await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert len(responses.calls) == 1


# --------------------------------------------------------------------------
# Client lifecycle
# --------------------------------------------------------------------------

_COMPLETED = SimpleNamespace(
    type="response.completed",
    response=SimpleNamespace(status="completed", usage=None),
)


async def test_cancellation_during_create_closes_the_client(monkeypatch):
    """CancelledError is a BaseException and escapes ``except Exception``."""
    _, clients = _install(monkeypatch, [asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert len(clients) == 1, "cancellation must not be retried"
    assert clients[0].closed


class _FailingStream:
    """Yields a reasoning delta, then fails mid-stream."""

    def __init__(self):
        self.aclosed = False

    def __aiter__(self):
        async def gen():
            yield SimpleNamespace(
                type="response.reasoning_summary_text.delta", delta="step"
            )
            raise RuntimeError("stream broke")

        return gen()

    async def aclose(self) -> None:
        self.aclosed = True


async def test_disconnect_while_closing_think_on_error_still_cleans_up(monkeypatch):
    """The consumer disconnects at the ``</think>`` the error path yields.

    Without the GeneratorExit guard the finally block yields again, aclose()
    raises "async generator ignored GeneratorExit", and the stream and client
    are never closed.
    """
    stream = _FailingStream()
    _, clients = _install(monkeypatch, [stream])

    agen = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )
    assert await agen.__anext__() == "<think>"
    assert await agen.__anext__() == "step"
    # The stream fails here; the error path yields the closing tag first.
    assert await agen.__anext__() == "</think>"

    await agen.aclose()

    assert stream.aclosed
    assert clients[0].closed


class _StallingStream:
    """Yields a reasoning delta, then blocks the next read forever."""

    def __init__(self):
        self.aclosed = False

    def __aiter__(self):
        async def gen():
            yield SimpleNamespace(
                type="response.reasoning_summary_text.delta", delta="step"
            )
            await asyncio.Event().wait()

        return gen()

    async def aclose(self) -> None:
        self.aclosed = True


async def test_cancelled_read_mid_cot_propagates_and_cleans_up(monkeypatch):
    """A cancelled read must raise CancelledError, not return ``</think>``.

    CancelledError escapes ``except Exception``; without its own handler the
    finally block yields the closing tag into the cancelled read, swallowing
    the cancellation and suspending before the stream and client are closed.
    """
    stream = _StallingStream()
    _, clients = _install(monkeypatch, [stream])

    agen = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )
    assert await agen.__anext__() == "<think>"
    assert await agen.__anext__() == "step"

    pending = asyncio.ensure_future(agen.__anext__())
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert stream.aclosed
    assert clients[0].closed


class _CutOffStream:
    """Emits body text, then ends cleanly without any terminal event."""

    def __init__(self):
        self.aclosed = False

    def __aiter__(self):
        async def gen():
            yield SimpleNamespace(type="response.output_text.delta", delta="partial")

        return gen()

    async def aclose(self) -> None:
        self.aclosed = True


async def test_stream_without_terminal_event_raises(monkeypatch):
    """A gateway closing the SSE response before the terminal event leaves a
    fragment; presenting it as a complete answer would hide the cut-off."""
    stream = _CutOffStream()
    _, clients = _install(monkeypatch, [stream])

    agen = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)
    chunks: list[str] = []
    with pytest.raises(InvalidResponseError) as excinfo:
        async for chunk in agen:
            chunks.append(chunk)

    assert chunks == ["partial"]
    assert "terminal event" in str(excinfo.value)
    assert stream.aclosed
    assert clients[0].closed


# --------------------------------------------------------------------------
# Request parameters
# --------------------------------------------------------------------------


async def test_store_defaults_to_false(monkeypatch):
    """The API's own default retains every request; LightRAG must opt out."""
    responses, _ = _install(monkeypatch, [_ok()])

    await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert responses.calls[0]["store"] is False


async def test_explicit_store_true_is_honoured(monkeypatch):
    responses, _ = _install(monkeypatch, [_ok()])

    await openai_responses_complete_if_cache("gpt-5.6", "hi", store=True)

    assert responses.calls[0]["store"] is True


async def test_reasoning_summary_is_requested_when_cot_is_enabled(monkeypatch):
    responses, _ = _install(monkeypatch, [_ok()])

    await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        enable_cot=True,
        reasoning_effort="low",
        reasoning_summary="auto",
    )

    call = responses.calls[0]
    assert call["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "reasoning_summary" not in call, "the option itself must not leak"


async def test_reasoning_summary_is_not_requested_without_cot(monkeypatch):
    """The summary would be discarded, so it is not paid for."""
    responses, _ = _install(monkeypatch, [_ok()])

    await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", enable_cot=False, reasoning_summary="auto"
    )

    call = responses.calls[0]
    assert "reasoning" not in call
    assert "reasoning_summary" not in call


async def test_explicit_reasoning_summary_wins_over_the_option(monkeypatch):
    responses, _ = _install(monkeypatch, [_ok()])

    await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        enable_cot=True,
        reasoning={"summary": "detailed"},
        reasoning_summary="auto",
    )

    assert responses.calls[0]["reasoning"] == {"summary": "detailed"}


async def test_reasoning_summary_is_skipped_for_structured_output(monkeypatch):
    """Structured output disables COT, so no summary is requested either."""
    responses, _ = _install(monkeypatch, [_ok("{}")])

    await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        enable_cot=True,
        response_format={"type": "json_object"},
        reasoning_summary="auto",
    )

    assert "reasoning" not in responses.calls[0]


class _ReasoningThenJsonStream:
    def __init__(self):
        self.aclosed = False

    def __aiter__(self):
        async def gen():
            yield SimpleNamespace(
                type="response.reasoning_summary_text.delta", delta="thinking"
            )
            yield SimpleNamespace(type="response.output_text.delta", delta="{}")
            yield _COMPLETED

        return gen()

    async def aclose(self) -> None:
        self.aclosed = True


async def test_native_text_format_disables_cot(monkeypatch):
    """``text={"format": ...}`` is structured output just as ``response_format``
    is; ``<think>`` markup ahead of the JSON body would corrupt it."""
    _install(monkeypatch, [_ReasoningThenJsonStream()])

    agen = await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        stream=True,
        enable_cot=True,
        text={"format": {"type": "json_object"}},
    )

    assert "".join([chunk async for chunk in agen]) == "{}"


async def test_plain_text_format_keeps_cot(monkeypatch):
    """``{"type": "text"}`` is the plain-text format, not structured output."""
    _install(monkeypatch, [_ReasoningThenJsonStream()])

    agen = await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        stream=True,
        enable_cot=True,
        text={"format": {"type": "text"}},
    )

    assert "".join([chunk async for chunk in agen]) == "<think>thinking</think>{}"


# 1x1 transparent PNG.
_PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode("ascii")


async def test_image_inputs_become_input_image_parts(monkeypatch):
    """The VLM role routes here when VLM_LLM_BINDING=openai_responses."""
    responses, _ = _install(monkeypatch, [_ok("a pixel")])

    await openai_responses_complete_if_cache(
        "gpt-5.6",
        "describe",
        image_inputs=[{"base64": _PNG, "mime_type": "image/png"}],
    )

    user_turn = responses.calls[0]["input"][-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"][0] == {"type": "input_text", "text": "describe"}
    image_part = user_turn["content"][1]
    assert image_part["type"] == "input_image"
    assert image_part["image_url"] == f"data:image/png;base64,{_PNG}"


async def test_reasoning_summary_option_reads_its_own_env(monkeypatch):
    monkeypatch.setenv("OPENAI_RESPONSES_LLM_REASONING_SUMMARY", "concise")
    parser = argparse.ArgumentParser()
    OpenAIResponsesLLMOptions.add_args(parser)
    options = OpenAIResponsesLLMOptions.options_dict(parser.parse_args([]))

    assert options["reasoning_summary"] == "concise"


# --------------------------------------------------------------------------
# Shared configuration and log hygiene
# --------------------------------------------------------------------------


async def test_shared_text_config_is_not_polluted_by_a_json_call(monkeypatch):
    """``llm_model_kwargs`` is shared across calls: a keyword-extraction call's
    ``text.format`` must not leak into the next, plain-text call (where it
    would force JSON mode and, via the native-format check, disable COT)."""
    responses, _ = _install(monkeypatch, [_ok("{}")])
    shared = {"text": {"verbosity": "low"}}

    await openai_responses_complete_if_cache(
        "gpt-5.6", "kw", response_format={"type": "json_object"}, **shared
    )
    await openai_responses_complete_if_cache("gpt-5.6", "answer", **shared)

    assert shared == {"text": {"verbosity": "low"}}
    assert responses.calls[0]["text"] == {
        "verbosity": "low",
        "format": {"type": "json_object"},
    }
    assert responses.calls[1]["text"] == {"verbosity": "low"}


async def test_shared_reasoning_config_is_not_polluted(monkeypatch):
    """A COT call's ``reasoning.summary`` must not stick to the shared dict and
    be sent (and billed) on later non-COT calls."""
    responses, _ = _install(monkeypatch, [_ok()])
    shared = {"reasoning": {"effort": "low"}}

    await openai_responses_complete_if_cache(
        "gpt-5.6", "q", enable_cot=True, reasoning_summary="auto", **shared
    )
    await openai_responses_complete_if_cache("gpt-5.6", "kw", **shared)

    assert shared == {"reasoning": {"effort": "low"}}
    assert responses.calls[0]["reasoning"] == {"effort": "low", "summary": "auto"}
    assert responses.calls[1]["reasoning"] == {"effort": "low"}


async def test_error_log_does_not_contain_the_system_prompt(
    monkeypatch, lightrag_caplog
):
    """The system prompt carries retrieved document context; a provider 5xx
    must not dump it into ordinary ERROR (or DEBUG) logs."""
    request = httpx.Request("POST", _URL)
    error = InternalServerError(
        "Error code: 500",
        response=httpx.Response(status_code=500, request=request),
        body=None,
    )
    _install(monkeypatch, [error])
    secret = "CONFIDENTIAL-DOCUMENT-CONTEXT-1234"

    with lightrag_caplog.at_level("DEBUG", logger="lightrag"):
        with pytest.raises(InternalServerError):
            await openai_responses_complete_if_cache(
                "gpt-5.6", "hi", system_prompt=f"Context:\n{secret}"
            )

    # `verbose_debug("System prompt: ...")` is the sanctioned prompt channel,
    # shared with the Chat Completions driver; only parameter dumps are pinned.
    param_dumps = [
        record.getMessage()
        for record in lightrag_caplog.records
        if "Params:" in record.getMessage()
        or record.getMessage().startswith("Additional kwargs:")
    ]
    assert any("OpenAI Responses API Call Failed" in m for m in param_dumps)
    assert any(m.startswith("Additional kwargs:") for m in param_dumps)
    assert all(secret not in m for m in param_dumps)
    assert all(f"<{len('Context:') + 1 + len(secret)} chars>" in m for m in param_dumps)
