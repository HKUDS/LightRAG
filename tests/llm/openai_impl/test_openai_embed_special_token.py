"""``openai_embed`` must not fail a batch over a literal special-token string.

``tiktoken``'s ``Encoding.encode`` defaults to ``disallowed_special=ALL``, so it
raises ``ValueError`` as soon as the text merely *contains* a marker such as
``"<|endoftext|>"``. ``openai_embed``'s truncation block ran that encode for
every non-empty text, so a single chunk of user content quoting the marker --
documentation, notes, captured model output -- failed the whole embedding batch
regardless of length. Fixed by encoding the markers as ordinary text, which is
what every other tokenizing path in the codebase already does via
``lightrag.utils.Tokenizer.encode``.

Scope: the truncation block only runs where ``max_token_size`` actually arrives,
i.e. direct library use (``LightRAG(embedding_func=openai_embed)``), because
``EmbeddingFunc.__call__`` injects it only when the wrapped function's own
signature declares the parameter. The API server wraps the binding in an
``optimized_embedding_function`` whose signature does not, and
``azure_openai_embed`` never forwards it either -- separate gaps, not covered
here.

The truncation itself (slice the token list, decode it back) is deliberately
left as it was: it fills the budget exactly, which no character-prefix scheme
can do, and its only artefact is at most one trailing U+FFFD when the cut lands
inside a multi-byte code point -- pinned below, and negligible for an embedding
vector.
"""

from types import SimpleNamespace

import pytest

from lightrag.llm.openai import _get_tiktoken_encoding_for_model, openai_embed

pytestmark = pytest.mark.offline

MODEL = "text-embedding-3-small"
MARKER = "<|endoftext|>"
REPLACEMENT = "�"


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
            data=[SimpleNamespace(embedding=[0.0, 1.0]) for _ in params["input"]]
        )


async def _embed(monkeypatch, texts, *, max_token_size, **kwargs):
    """Drive the real ``openai_embed`` body; return the texts sent to the API.

    ``.func`` unwraps ``@wrap_embedding_func_with_attrs`` so the fake client's
    2-element vectors do not trip the wrapper's dimension check.
    """
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )
    await openai_embed.func(
        texts, model=MODEL, api_key="test-key", max_token_size=max_token_size, **kwargs
    )
    assert len(captured) == 1
    return captured[0]["input"]


def _token_count(text):
    encoding = _get_tiktoken_encoding_for_model(MODEL)
    # disallowed_special=() so this measurement is of the budget, never of
    # tiktoken's special-token guard.
    return len(encoding.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Fix proof: each of these raises ValueError before the fix.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_containing_a_special_token_is_embedded(monkeypatch):
    """A short text quoting the marker embeds unchanged, like any other text."""
    text = f"an example of the {MARKER} marker in prose"

    assert await _embed(monkeypatch, [text], max_token_size=8192) == [text]


@pytest.mark.asyncio
async def test_one_offending_text_does_not_fail_its_batch_siblings(monkeypatch):
    """The old failure took down the whole batch, not just the offending item."""
    texts = ["a perfectly ordinary chunk", f"another chunk mentioning {MARKER}"]

    assert await _embed(monkeypatch, texts, max_token_size=8192) == texts


@pytest.mark.asyncio
async def test_oversized_text_containing_a_special_token(monkeypatch):
    """Truncation and the marker in one input: encode, then cut to budget."""
    text = f"filler words here {MARKER} more filler " * 40
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    assert sent != text
    assert _token_count(sent) <= budget


# ---------------------------------------------------------------------------
# Stability: these pass before and after the fix, and pin the properties that
# the one-line change deliberately preserves.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_truncation_still_fills_the_budget(monkeypatch):
    """No under-fill: the cut is at a token boundary, so the budget is used up.

    This is why the token-slice truncation is kept rather than replaced by a
    character-prefix scheme, which can only ever land at or below the budget.
    """
    text = "hello world " * 100
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    assert _token_count(sent) == budget


@pytest.mark.asyncio
async def test_a_cut_inside_a_code_point_costs_one_trailing_replacement(monkeypatch):
    """The accepted artefact, bounded: at most one U+FFFD, only at the end.

    Every token before the cut decodes to complete characters; only the code
    point whose remaining bytes lived in the next token is lost. The result
    still fits the budget, which is what the API cares about.
    """
    text = "hello world " * 10 + "\U0001f44d" * 5
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    assert sent.count(REPLACEMENT) <= 1
    assert REPLACEMENT not in sent[:-1]
    assert _token_count(sent) <= budget


@pytest.mark.asyncio
async def test_text_within_budget_and_empty_text_are_passed_through(monkeypatch):
    sent = await _embed(monkeypatch, ["", "a short sentence"], max_token_size=8192)

    assert sent == ["", "a short sentence"]


@pytest.mark.asyncio
async def test_prefix_is_applied_before_truncation(monkeypatch):
    """The budget covers the prefix, so the prefix cannot push the input over."""
    prefix = "search_document: "
    budget = 12

    (sent,) = await _embed(
        monkeypatch,
        ["hello world " * 10],
        max_token_size=budget,
        context="document",
        document_prefix=prefix,
    )

    assert sent.startswith(prefix)
    assert _token_count(sent) <= budget
