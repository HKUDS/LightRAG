"""Offline tests: the Ollama binding reports token usage.

Ollama returns ``prompt_eval_count`` / ``eval_count`` on every completed call,
and this module already read them -- but only to decorate a truncation error
message. Nothing surfaced them as usage, so the ``TokenTracker`` contract the
OpenAI and Gemini bindings honour was unimplemented here.

On the chat path, passing ``token_tracker`` was not merely ignored before this:
everything left in ``**kwargs`` is forwarded verbatim to
``AsyncClient.chat()``, which declares no ``**kwargs``, so it raised
``TypeError`` on every call. (``ollama_embed`` builds its client arguments
explicitly, so there the same kwarg was silently dropped instead.) The
parameter is explicit now, and the forwarding test below is what keeps it out
of the wire call.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.llm.ollama import (
    _ollama_model_if_cache,
    _ollama_usage_counts,
    ollama_embed,
)
from lightrag.utils import TokenTracker

pytestmark = pytest.mark.offline


def _make_fake_client(response):
    return SimpleNamespace(
        chat=AsyncMock(return_value=response),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


def _chat_response(content="answer", **extra):
    return {"message": {"content": content}, "done_reason": "stop", **extra}


async def _call(response, **kwargs):
    fake_client = _make_fake_client(response)
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="Q", **kwargs)
    return result, fake_client


# -- non-streaming ---------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_records_prompt_and_eval_counts():
    tracker = TokenTracker()

    result, _ = await _call(
        _chat_response(prompt_eval_count=31, eval_count=11),
        token_tracker=tracker,
    )

    assert result == "answer"
    assert tracker.get_usage() == {
        "prompt_tokens": 31,
        "completion_tokens": 11,
        "total_tokens": 42,
        "call_count": 1,
    }, "Ollama's eval counters were not recorded as usage"


@pytest.mark.asyncio
async def test_token_tracker_is_not_forwarded_to_the_wire_call():
    """`AsyncClient.chat` declares no **kwargs, so a `token_tracker` left in
    kwargs is a TypeError on every call rather than a no-op. It must be
    consumed by the binding."""
    _, fake_client = await _call(
        _chat_response(prompt_eval_count=1, eval_count=1),
        token_tracker=TokenTracker(),
    )

    assert "token_tracker" not in fake_client.chat.await_args.kwargs, (
        "token_tracker reached AsyncClient.chat(), which declares no **kwargs"
    )


@pytest.mark.asyncio
async def test_usage_is_counted_before_the_empty_truncation_raise():
    """The budget was spent whether or not the output was usable, and the
    empty-and-cut-off case is exactly the one that spent it."""
    from lightrag.llm.ollama import InvalidResponseError

    tracker = TokenTracker()

    with pytest.raises(InvalidResponseError):
        await _call(
            {
                "message": {"content": "", "thinking": "thought for a while"},
                "done_reason": "length",
                "prompt_eval_count": 12,
                "eval_count": 400,
            },
            token_tracker=tracker,
        )

    assert tracker.get_usage()["total_tokens"] == 412, (
        "a call that burned its whole budget on the reasoning trace must "
        "still be billed"
    )


@pytest.mark.asyncio
async def test_a_response_without_counters_records_nothing():
    """Older servers omit the counters; a missing count is not a zero count."""
    tracker = TokenTracker()

    await _call(_chat_response(), token_tracker=tracker)

    assert tracker.get_usage()["call_count"] == 0, (
        "a response reporting no counters was billed as a zero-token call"
    )


def test_the_retry_policy_still_guards_the_call():
    """The `@retry` block sits immediately above `_ollama_model_if_cache`, so a
    helper defined between the two silently steals it and leaves every Ollama
    call unretried. Nothing else in this directory pins it."""
    assert hasattr(_ollama_model_if_cache, "retry"), (
        "_ollama_model_if_cache lost its tenacity @retry decorator"
    )
    assert not hasattr(_ollama_usage_counts, "retry"), (
        "the @retry block slid onto _ollama_usage_counts"
    )


# -- streaming -------------------------------------------------------------


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


@pytest.mark.asyncio
async def test_streaming_records_usage_from_the_terminal_chunk():
    tracker = TokenTracker()
    stream = _FakeStream(
        [
            {"message": {"content": "hel"}},
            {"message": {"content": "lo"}},
            # Terminal chunk: empty content, counters attached.
            {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 7,
                "eval_count": 2,
            },
        ]
    )

    result, _ = await _call(stream, stream=True, token_tracker=tracker)
    collected = "".join([chunk async for chunk in result])

    assert collected == "hello"
    assert tracker.get_usage() == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
        "call_count": 1,
    }, "the terminal chunk's counters were not recorded"


@pytest.mark.asyncio
async def test_streaming_without_counters_records_nothing():
    tracker = TokenTracker()
    stream = _FakeStream([{"message": {"content": "hi"}}])

    result, _ = await _call(stream, stream=True, token_tracker=tracker)
    assert "".join([chunk async for chunk in result]) == "hi"

    assert tracker.get_usage()["call_count"] == 0, (
        "a stream reporting no counters was billed as a zero-token call"
    )


@pytest.mark.asyncio
async def test_a_disconnected_stream_is_not_billed():
    """Accounting sits after the loop and inside the try, so GeneratorExit on
    consumer disconnect skips it -- the OpenAI binding's choice. Gemini bills a
    disconnect from a `finally`; this file follows OpenAI."""
    tracker = TokenTracker()
    stream = _FakeStream(
        [
            {"message": {"content": "hel"}},
            {"message": {"content": "lo"}},
            {"message": {"content": ""}, "done": True, "eval_count": 2},
        ]
    )

    result, _ = await _call(stream, stream=True, token_tracker=tracker)
    agen = result.__aiter__()
    assert await agen.__anext__() == "hel"
    await agen.aclose()

    assert tracker.get_usage()["call_count"] == 0, (
        "a stream the consumer abandoned was billed"
    )


# -- embeddings ------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_records_prompt_tokens_only():
    """An embed call has no generation, so Ollama reports no `eval_count`."""
    tracker = TokenTracker()
    fake_client = SimpleNamespace(
        embed=AsyncMock(
            return_value={"embeddings": [[0.1, 0.2]], "prompt_eval_count": 6}
        ),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        vectors = await ollama_embed.func(["hello"], token_tracker=tracker)

    assert isinstance(vectors, np.ndarray)
    assert tracker.get_usage() == {
        "prompt_tokens": 6,
        "completion_tokens": 0,
        "total_tokens": 6,
        "call_count": 1,
    }, "embed usage was not recorded, or invented completion tokens"


# -- the mapping itself ----------------------------------------------------


def test_usage_counts_sums_a_total_ollama_does_not_report():
    assert _ollama_usage_counts({"prompt_eval_count": 4, "eval_count": 6}) == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
    }


def test_usage_counts_distinguishes_absent_from_zero():
    # Neither counter: an older or non-conforming server, not a call that
    # consumed nothing.
    assert _ollama_usage_counts({}) is None
    assert _ollama_usage_counts({"message": {"content": "x"}}) is None
    # One counter present is real usage; the absent half counts as zero.
    assert _ollama_usage_counts({"eval_count": 3}) == {
        "prompt_tokens": 0,
        "completion_tokens": 3,
        "total_tokens": 3,
    }


def test_usage_counts_reads_an_ollama_0_4_chat_response():
    """The raw-dict fixtures above stand in for ollama<0.4. Pin the >=0.4
    object shape too, since `SubscriptableBaseModel.get` is what makes that
    dict-style access work against a real client."""
    from ollama import ChatResponse

    response = ChatResponse(
        model="test-model",
        message={"role": "assistant", "content": "hi"},
        prompt_eval_count=8,
        eval_count=4,
    )

    assert _ollama_usage_counts(response) == {
        "prompt_tokens": 8,
        "completion_tokens": 4,
        "total_tokens": 12,
    }


def test_usage_counts_degrades_on_a_payload_it_cannot_read():
    """A shape without `.get` must not take down a usable response."""
    assert _ollama_usage_counts(object()) is None
