"""``settings`` on the two llama_index entry points.

``llama_index_embed`` applies the caller's ``Settings`` object through
``configure_llama_index()``; ``llama_index_complete`` documents the same
parameter but dropped it, so a deployment that configures LlamaIndex once
through ``llm_model_kwargs={"settings": ...}`` had it honoured for embeddings
and silently ignored for completions.
"""

from __future__ import annotations

import pytest

from lightrag.llm import llama_index_impl
from lightrag.llm.llama_index_impl import llama_index_complete, llama_index_embed

pytestmark = pytest.mark.offline


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeLLM:
    def __init__(self, content: str = "answer") -> None:
        self.content = content
        self.received_kwargs = None

    async def achat(self, messages, **kwargs):
        self.received_kwargs = kwargs
        return _FakeChatResponse(self.content)


class _FakeEmbedModel:
    async def _aget_text_embeddings(self, texts):
        return [[0.1, 0.2] for _ in texts]


def _record_configure(monkeypatch):
    calls = []

    def _fake_configure(settings=None, **kwargs):
        calls.append(settings)
        return settings

    monkeypatch.setattr(llama_index_impl, "configure_llama_index", _fake_configure)
    return calls


@pytest.mark.asyncio
async def test_complete_applies_settings(monkeypatch):
    """settings is documented on llama_index_complete and must reach
    configure_llama_index() the same way it does on llama_index_embed."""
    calls = _record_configure(monkeypatch)
    settings = object()

    result = await llama_index_complete(
        "hi", llm_instance=_FakeLLM(), settings=settings
    )

    assert result == "answer"
    assert calls == [settings]


@pytest.mark.asyncio
async def test_complete_without_settings_does_not_configure(monkeypatch):
    """An absent/None settings must stay a no-op: only an explicit Settings
    object may replace the global one."""
    calls = _record_configure(monkeypatch)

    await llama_index_complete("hi", llm_instance=_FakeLLM())

    assert calls == []


@pytest.mark.asyncio
async def test_complete_settings_does_not_reach_the_llm(monkeypatch):
    """settings is consumed here, not forwarded: llama_index_complete_if_cache
    has no such parameter, so leaking it would raise TypeError instead."""
    _record_configure(monkeypatch)
    llm = _FakeLLM()

    await llama_index_complete("hi", llm_instance=llm, settings=object())

    assert llm.received_kwargs == {}


@pytest.mark.asyncio
async def test_embed_and_complete_apply_settings_symmetrically(monkeypatch):
    """Both entry points are wired as a pair (llm_model_func /
    embedding_func) from the same llm_model_kwargs, so one settings entry has
    to configure both."""
    calls = _record_configure(monkeypatch)
    settings = object()

    await llama_index_complete("hi", llm_instance=_FakeLLM(), settings=settings)
    await llama_index_embed.func(
        ["a"], embed_model=_FakeEmbedModel(), settings=settings
    )

    assert calls == [settings, settings]
