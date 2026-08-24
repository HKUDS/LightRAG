"""``openai_embed``'s own truncation must honour the safe-tokenizer contract.

#3565 introduced ``lightrag.utils.Tokenizer``'s verified split/truncate
contract and migrated the paths it covered, listing this one under "Known
remaining paths": ``openai_embed`` sliced the raw tiktoken token list and
``decode()``d it back, bypassing ``Tokenizer`` entirely with no
re-verification. Two consequences, both live wherever the truncation block runs at all. That is
direct library use -- ``LightRAG(embedding_func=openai_embed)``, as in
``examples/lightrag_openai_demo.py`` -- where ``EmbeddingFunc`` injects the
decorator's ``max_token_size=8192`` because ``openai_embed``'s own signature
declares the parameter. (The API server wraps the binding in an
``optimized_embedding_function`` whose signature does not, so nothing is
injected there and the block is skipped entirely; ``azure_openai_embed``
likewise never forwards the parameter. Both are separate gaps, out of scope
here.)

1. ``Encoding.encode`` defaults to ``disallowed_special=ALL``, so a chunk whose
   text merely *contains* the literal ``"<|endoftext|>"`` raised ``ValueError``
   and failed the whole embedding batch -- regardless of its length, since the
   encode ran for every non-empty text once the block was entered.
   ``Tokenizer.encode`` exists precisely to treat those strings as ordinary
   text.
2. ``decode(tokens[:n])`` decodes with ``errors="replace"``, so a budget landing
   inside a multi-token code point yielded a string ending in U+FFFD. What was
   embedded was then not a prefix of the input at all.

Nothing re-encoded the result either, which the contract requires because BPE
token count is not monotonic in text length.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightrag.llm.openai import (
    _get_tiktoken_encoding_for_model,
    openai_embed,
)
from lightrag.utils import TokenBudgetError


pytestmark = pytest.mark.offline

MODEL = "text-embedding-3-small"


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


async def _embed(monkeypatch, texts, *, max_token_size, model=MODEL, **kwargs):
    """Drive the real ``openai_embed`` body; return the texts sent to the API."""
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )
    # ``.func`` unwraps @wrap_embedding_func_with_attrs, ``__wrapped__`` the
    # tenacity retry, so a raised error surfaces once instead of three times.
    await openai_embed.func.__wrapped__(
        texts, model=model, api_key="test-key", max_token_size=max_token_size, **kwargs
    )
    assert len(captured) == 1
    return captured[0]["input"]


@pytest.mark.asyncio
async def test_literal_special_token_does_not_fail_the_batch(monkeypatch):
    """A chunk containing "<|endoftext|>" must embed like any other text."""
    text = "an example of the <|endoftext|> marker in prose"

    sent = await _embed(monkeypatch, [text], max_token_size=8192)

    assert sent == [text]


@pytest.mark.asyncio
async def test_truncation_lands_on_a_code_point_boundary(monkeypatch):
    """The truncated text is a real prefix of the input, never U+FFFD-tailed."""
    text = "hello world " * 10 + "\U0001f44d" * 5

    (sent,) = await _embed(monkeypatch, [text], max_token_size=24)

    assert "�" not in sent
    assert text.startswith(sent)
    assert sent != text


@pytest.mark.asyncio
async def test_truncated_text_is_re_verified_against_the_budget(monkeypatch):
    """Re-encoding the result actually fits -- BPE count is not monotonic."""
    text = "hello world " * 10 + "\U0001f44d" * 5
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    # Count with disallowed_special=() so this assertion measures the budget,
    # not tiktoken's special-token guard.
    encoding = _get_tiktoken_encoding_for_model(MODEL)
    assert len(encoding.encode(sent, disallowed_special=())) <= budget


@pytest.mark.asyncio
async def test_text_within_budget_is_sent_unchanged(monkeypatch):
    text = "a short sentence"

    sent = await _embed(monkeypatch, [text], max_token_size=8192)

    assert sent == [text]


@pytest.mark.asyncio
async def test_empty_text_is_passed_through(monkeypatch):
    sent = await _embed(monkeypatch, ["", "kept"], max_token_size=8192)

    assert sent == ["", "kept"]


@pytest.mark.asyncio
async def test_prefix_is_applied_before_truncation(monkeypatch):
    """Prefix-then-truncate order is unchanged, so the budget covers the prefix."""
    prefix = "search_document: "
    text = "hello world " * 10

    (sent,) = await _embed(
        monkeypatch,
        [text],
        max_token_size=12,
        context="document",
        document_prefix=prefix,
    )

    assert sent.startswith(prefix)
    assert (prefix + text).startswith(sent)
    # The two assertions above both survive a prefix-after-truncate swap; this
    # one does not -- swapping puts the payload prefix_tokens over budget.
    encoding = _get_tiktoken_encoding_for_model(MODEL)
    assert len(encoding.encode(sent, disallowed_special=())) <= 12


@pytest.mark.asyncio
async def test_unknown_model_name_still_truncates(monkeypatch):
    """Deployment / gateway names tiktoken cannot resolve fall back, not raise.

    ``TiktokenTokenizer(model_name=...)`` raises ``ValueError`` for these; the
    binding must keep ``_get_tiktoken_encoding_for_model``'s cl100k_base
    fallback.
    """
    text = "hello world " * 10

    (sent,) = await _embed(
        monkeypatch, [text], max_token_size=12, model="my-azure-deployment-xyz"
    )

    assert text.startswith(sent)
    assert sent != text


@pytest.mark.asyncio
async def test_budget_smaller_than_one_code_point_is_surfaced(monkeypatch):
    """A budget no single code point can fit is an error, not a silent U+FFFD.

    The ERROR line is asserted too, so deleting the binding's own
    ``except TokenBudgetError`` handler is caught -- a bare ``pytest.raises``
    would still pass on the tokenizer's own exception alone. Matches
    ``_truncate_vdb_content``'s handling of the same condition, and the
    position tells the operator which input in the batch was at fault.
    """
    with patch("lightrag.llm.openai.logger") as mock_logger:
        with pytest.raises(TokenBudgetError):
            await _embed(monkeypatch, ["ok", "\U0001f44d" * 10], max_token_size=1)

    logged = " ".join(str(call) for call in mock_logger.error.call_args_list)
    assert "Embedding input 2/2 cannot fit token limit 1" in logged


@pytest.mark.asyncio
async def test_oversized_text_containing_a_special_token(monkeypatch):
    """The composite case: both defects meet in one input.

    Over budget AND carrying a literal special token, so the result must come
    from encoding the marker as ordinary text and then cutting on a code-point
    boundary -- neither half of the fix alone produces it.
    """
    text = "filler words here <|endoftext|> more filler " * 40
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    assert text.startswith(sent)
    assert sent != text
    assert "\ufffd" not in sent
    encoding = _get_tiktoken_encoding_for_model(MODEL)
    assert len(encoding.encode(sent, disallowed_special=())) <= budget
