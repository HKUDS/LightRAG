"""Gemini provider-moderation verdicts raise ContentPolicyViolationError once.

A safety-family ``finish_reason`` or a ``prompt_feedback.block_reason`` is a
deterministic provider verdict, not caller-budget truncation and not a sampling
artifact: it must raise the non-retryable ``ContentPolicyViolationError`` after
ONE attempt (non-streaming) or mid-iterator (streaming), naming the reason,
instead of the retryable empty-response path or a silent empty stream. The
controls pin that STOP / MAX_TOKENS keep their existing behavior.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from tenacity import wait_none

from lightrag.exceptions import (
    ContentPolicyViolationError,
    EmptyTruncatedResponseError,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


# The safety-family finish reasons confirmed against google-genai 2.34.0. Each
# member is its own branch of the membership test, so each gets a case.
_SAFETY_MEMBERS = [
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
]


def _load_gemini_module(monkeypatch, request):
    """Fresh-import the binding against a stubbed google-genai client.

    Mirrors tests/llm/gemini_impl/test_gemini_llm.py's loader, but the fake
    ``FinishReason`` namespace carries the whole safety family (string sentinels
    stand in for the SDK enum members, exactly as the sibling tests do for
    MAX_TOKENS / STOP) so the membership test can be exercised offline.
    """
    fake_pm = SimpleNamespace(is_installed=lambda name: True, install=lambda name: None)

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    finish_reason_ns = SimpleNamespace(
        MAX_TOKENS="MAX_TOKENS", STOP="STOP", **{m: m for m in _SAFETY_MEMBERS}
    )
    fake_types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        HttpOptions=FakeHttpOptions,
        FinishReason=finish_reason_ns,
        BlockedReason=SimpleNamespace(
            SAFETY="SAFETY",
            OTHER="OTHER",
            BLOCKLIST="BLOCKLIST",
            PROHIBITED_CONTENT="PROHIBITED_CONTENT",
            IMAGE_SAFETY="IMAGE_SAFETY",
            MODEL_ARMOR="MODEL_ARMOR",
            JAILBREAK="JAILBREAK",
        ),
    )

    class FakeAPIError(Exception):
        def __init__(self, code):
            self.code = code
            super().__init__(f"fake gemini api error {code}")

    fake_genai_errors = SimpleNamespace(
        APIError=FakeAPIError,
        ClientError=type("ClientError", (FakeAPIError,), {}),
        ServerError=type("ServerError", (FakeAPIError,), {}),
    )
    fake_genai = SimpleNamespace(
        Client=lambda **kwargs: SimpleNamespace(kwargs=kwargs), errors=fake_genai_errors
    )
    fake_google_module = ModuleType("google")
    fake_google_module.genai = fake_genai
    fake_api_exceptions = SimpleNamespace(
        **{
            name: type(name, (Exception,), {})
            for name in (
                "InternalServerError",
                "ServiceUnavailable",
                "ResourceExhausted",
                "GatewayTimeout",
                "BadGateway",
                "DeadlineExceeded",
                "Aborted",
                "Unknown",
            )
        }
    )
    fake_google_api_core = ModuleType("google.api_core")
    fake_google_api_core.exceptions = fake_api_exceptions

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        SimpleNamespace(types=fake_types, errors=fake_genai_errors),
    )
    monkeypatch.setitem(sys.modules, "google.api_core", fake_google_api_core)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", fake_api_exceptions)

    parent = sys.modules.get("lightrag.llm")
    original_gemini = sys.modules.get("lightrag.llm.gemini")
    original_parent_attr = getattr(parent, "gemini", None) if parent else None
    sys.modules.pop("lightrag.llm.gemini", None)
    if parent is not None and hasattr(parent, "gemini"):
        delattr(parent, "gemini")

    def _restore_gemini():
        if original_gemini is not None:
            sys.modules["lightrag.llm.gemini"] = original_gemini
        else:
            sys.modules.pop("lightrag.llm.gemini", None)
        if parent is not None:
            if original_parent_attr is not None:
                parent.gemini = original_parent_attr
            elif hasattr(parent, "gemini"):
                delattr(parent, "gemini")

    request.addfinalizer(_restore_gemini)
    return importlib.import_module("lightrag.llm.gemini")


def _make_response(
    regular_text="", thought_text="", finish_reason=None, block_reason=None
):
    """A fake GenerateContentResponse.

    ``block_reason`` models an input-side block: candidates are then absent
    (None), as the SDK returns them, and prompt_feedback carries the reason.
    """
    parts = []
    if thought_text:
        parts.append(SimpleNamespace(text=thought_text, thought=True))
    if regular_text:
        parts.append(SimpleNamespace(text=regular_text, thought=False))

    if block_reason is not None:
        candidates = None
        prompt_feedback = SimpleNamespace(block_reason=block_reason)
    else:
        candidates = [
            SimpleNamespace(
                content=SimpleNamespace(parts=parts), finish_reason=finish_reason
            )
        ]
        prompt_feedback = SimpleNamespace(block_reason=None)

    return SimpleNamespace(
        candidates=candidates,
        prompt_feedback=prompt_feedback,
        usage_metadata=SimpleNamespace(
            prompt_token_count=1, candidates_token_count=2, total_token_count=3
        ),
    )


def _counting_client(response):
    """Non-streaming client whose generate_content counts attempts."""
    calls = {"n": 0}

    async def _generate_content(**kwargs):
        calls["n"] += 1
        return response

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=_generate_content))
    )
    return client, calls


def _no_backoff(monkeypatch, gemini_module):
    monkeypatch.setattr(
        gemini_module.gemini_complete_if_cache.retry, "wait", wait_none()
    )


@pytest.mark.parametrize("member", _SAFETY_MEMBERS)
async def test_safety_finish_reason_raises_policy_error_once(
    monkeypatch, request, member
):
    """Each safety-family finish_reason is a deterministic verdict: it raises
    ContentPolicyViolationError after ONE decorated attempt and names the
    reason, never the retryable empty-response path."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    _no_backoff(monkeypatch, gemini_module)
    client, calls = _counting_client(_make_response(finish_reason=member))
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: client)

    with pytest.raises(ContentPolicyViolationError) as excinfo:
        await gemini_module.gemini_complete_if_cache(
            model="gemini-model", prompt="hi", api_key="test-key"
        )

    assert calls["n"] == 1, "a deterministic policy verdict must not retry"
    assert f"finish_reason={member}" in str(excinfo.value)


async def test_safety_finish_reason_with_partial_text_raises_not_returns(
    monkeypatch, request
):
    """A policy stop is not caller-budget truncation: flagged partial text must
    never be returned or cached as a complete answer, so it raises even though
    the response carries usable-looking text."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    _no_backoff(monkeypatch, gemini_module)
    client, calls = _counting_client(
        _make_response(regular_text="flagged partial answer", finish_reason="SAFETY")
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: client)

    with pytest.raises(ContentPolicyViolationError) as excinfo:
        await gemini_module.gemini_complete_if_cache(
            model="gemini-model", prompt="hi", api_key="test-key"
        )

    assert calls["n"] == 1
    message = str(excinfo.value)
    assert "finish_reason=SAFETY" in message
    assert "flagged partial answer" not in message, "must never log response content"


@pytest.mark.parametrize("reason", ["SAFETY", "MODEL_ARMOR", "JAILBREAK"])
async def test_prompt_block_reason_raises_policy_error_once(
    monkeypatch, request, reason
):
    """An input-side block (prompt_feedback.block_reason, candidates absent)
    raises ContentPolicyViolationError naming the actual block reason after ONE
    attempt, instead of 'model produced no output' retried three times."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    _no_backoff(monkeypatch, gemini_module)
    client, calls = _counting_client(_make_response(block_reason=reason))
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: client)

    with pytest.raises(ContentPolicyViolationError) as excinfo:
        await gemini_module.gemini_complete_if_cache(
            model="gemini-model", prompt="hi", api_key="test-key"
        )

    assert calls["n"] == 1
    assert f"block_reason={reason}" in str(excinfo.value)


async def test_stream_safety_finish_reason_raises_mid_iterator(monkeypatch, request):
    """A blocked stream used to end empty and successful. A safety finish_reason
    on a later chunk must raise mid-iterator, after partial text was yielded."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    first = _make_response(regular_text="partial ", finish_reason=None)
    blocked = _make_response(finish_reason="SAFETY")

    async def _stream(**kwargs):
        async def _gen():
            yield first
            yield blocked

        return _gen()

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=_stream))
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: client)

    stream = await gemini_module.gemini_complete_if_cache(
        model="gemini-model", prompt="hi", api_key="test-key", stream=True
    )
    collected = []
    with pytest.raises(ContentPolicyViolationError) as excinfo:
        async for chunk in stream:
            collected.append(chunk)

    assert collected == ["partial "], "partial text precedes the mid-iterator raise"
    assert "finish_reason=SAFETY" in str(excinfo.value)


async def test_stream_prompt_block_reason_raises_mid_iterator(monkeypatch, request):
    """An input-side block surfaced on a stream chunk raises mid-iterator rather
    than letting the stream complete empty and successful."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    blocked = _make_response(block_reason="JAILBREAK")

    async def _stream(**kwargs):
        async def _gen():
            yield blocked

        return _gen()

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=_stream))
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: client)

    stream = await gemini_module.gemini_complete_if_cache(
        model="gemini-model", prompt="hi", api_key="test-key", stream=True
    )
    with pytest.raises(ContentPolicyViolationError) as excinfo:
        async for _chunk in stream:
            pass

    assert "block_reason=JAILBREAK" in str(excinfo.value)


@pytest.mark.parametrize(
    ("finish_reason", "expected_type", "expected_attempts"),
    [
        pytest.param("STOP", "invalid", 3, id="stop-empty-stays-retryable"),
        pytest.param("MAX_TOKENS", "truncated", 1, id="max-tokens-unchanged"),
    ],
)
async def test_non_safety_finish_reasons_are_not_policy(
    monkeypatch, request, finish_reason, expected_type, expected_attempts
):
    """Control: the safety classification must not swallow the neighboring
    finish reasons. STOP-empty stays retryable (three decorated attempts);
    MAX_TOKENS-empty stays the fail-fast EmptyTruncatedResponseError (one)."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    _no_backoff(monkeypatch, gemini_module)
    client, calls = _counting_client(_make_response(finish_reason=finish_reason))
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: client)

    expected = (
        gemini_module.InvalidResponseError
        if expected_type == "invalid"
        else EmptyTruncatedResponseError
    )
    with pytest.raises(expected) as excinfo:
        await gemini_module.gemini_complete_if_cache(
            model="gemini-model", prompt="hi", api_key="test-key"
        )

    assert calls["n"] == expected_attempts
    assert not isinstance(excinfo.value, ContentPolicyViolationError)
