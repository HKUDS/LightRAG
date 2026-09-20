"""``azure_openai_embed`` must truncate long texts like ``openai_embed`` does.

``EmbeddingFunc.__call__`` auto-injects ``max_token_size`` from the decorator
only when the wrapped function's own signature declares that parameter
(``lightrag.utils.EmbeddingFunc.__call__``). ``azure_openai_embed`` is
decorated with ``max_token_size=8192`` but its signature never declared the
parameter, and its call into ``openai_embed.func`` never forwarded one
either, so the injection silently no-opped: a text longer than the model's
context window reached Azure's embeddings API untouched instead of being cut
to fit, unlike the identical path for the standard OpenAI binding.

Fixed by adding ``max_token_size`` to ``azure_openai_embed``'s signature and
forwarding it to ``openai_embed.func``.
"""

from types import SimpleNamespace

import pytest

from lightrag.llm.azure_openai import azure_openai_embed
from lightrag.llm.openai import _get_tiktoken_encoding_for_model

pytestmark = pytest.mark.offline

MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class _FakeEmbeddingClient:
    """Captures the ``input`` list that actually reaches the embeddings API."""

    def __init__(self, captured):
        self._captured = captured
        self.embeddings = SimpleNamespace(create=self._create)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _create(self, **params):
        self._captured.append(params)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.0] * EMBEDDING_DIM)
                for _ in params["input"]
            ]
        )


def _token_count(text):
    encoding = _get_tiktoken_encoding_for_model(MODEL)
    return len(encoding.encode(text, disallowed_special=()))


@pytest.mark.asyncio
async def test_azure_openai_embed_truncates_oversized_text(monkeypatch):
    """A text far beyond the declared 8192-token budget must be cut down
    before reaching the Azure embeddings API, exactly as ``openai_embed``
    truncates for the standard OpenAI binding.
    """
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", MODEL)

    text = "filler words here more filler " * 2000
    assert _token_count(text) > EMBEDDING_DIM * 5  # comfortably over 8192

    await azure_openai_embed([text])

    assert len(captured) == 1
    (sent,) = captured[0]["input"]
    assert sent != text
    assert _token_count(sent) <= 8192


@pytest.mark.asyncio
async def test_azure_openai_embed_passes_short_text_through(monkeypatch):
    """Stability: text within budget still reaches the API unchanged."""
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", MODEL)

    text = "a short sentence"

    await azure_openai_embed([text])

    assert captured[0]["input"] == [text]
