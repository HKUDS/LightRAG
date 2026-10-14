"""hf_embed() must pass an explicit, correctly-resolved max_length to the
tokenizer so truncation=True actually truncates oversized inputs.

Regression for issue #2396: a text that tokenizes longer than the embedding
model's real position-embedding limit crashed inside the model's forward
pass ("The expanded size of the tensor ... must match the existing size ...
at non-singleton dimension 1"), with transformers warning "Asking to
truncate to max_length but no maximum length is provided and the model has
no predefined maximum length. Default to no truncation."

That happens because truncation=True alone is a no-op when the tokenizer's
own model_max_length is unset -- transformers then falls back to a sentinel
value like int(1e30) rather than a real limit, and nothing gets truncated.
The fix resolves a real max_length -- preferring a sanely bounded
tokenizer.model_max_length, falling back to
embed_model.config.max_position_embeddings when the tokenizer's value is
that huge unset sentinel -- and passes it into the tokenizer call.

lightrag/llm/hf.py imports transformers and torch at module level; neither
is installed in the offline test job, so both are stubbed here following the
same pattern as test_hf_embed_masked_pooling.py.
"""

from __future__ import annotations

import sys
import types
import importlib

import numpy as np
import pytest

pytestmark = pytest.mark.offline

# transformers' own sentinel for "no max length configured" (int(1e30)),
# the exact shape of value the issue's warning is about.
UNSET_SENTINEL = int(1e30)


class _FakeDType:
    def __init__(self, name):
        self.name = name


FLOAT32 = _FakeDType("float32")


class FakeTensor:
    """Minimal numpy-backed stand-in for the torch.Tensor ops hf_embed()
    touches on the embedding/pooling path."""

    def __init__(self, array, dtype=FLOAT32):
        self.array = np.asarray(array, dtype=np.float64)
        self.dtype = dtype

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.array, dim), self.dtype)

    def to(self, target):
        return self

    def sum(self, dim):
        return FakeTensor(self.array.sum(axis=dim), self.dtype)

    def clamp_min(self, value):
        return FakeTensor(np.clip(self.array, a_min=value, a_max=None), self.dtype)

    def __mul__(self, other):
        return FakeTensor(self.array * other.array, self.dtype)

    def __truediv__(self, other):
        return FakeTensor(self.array / other.array, self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array.astype(np.float32)


def zeros(*shape):
    return FakeTensor(np.zeros(shape))


def ones(*shape):
    return FakeTensor(np.ones(shape))


def install_fake_transformers_and_torch(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = FLOAT32
    fake_torch.float16 = _FakeDType("float16")
    fake_torch.bfloat16 = _FakeDType("bfloat16")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.device = lambda name: name
    fake_torch.no_grad = _NullContext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)


@pytest.fixture
def hf_module(monkeypatch):
    install_fake_transformers_and_torch(monkeypatch)
    sys.modules.pop("lightrag.llm.hf", None)
    return importlib.import_module("lightrag.llm.hf")


class _FakeTokenizerOutput(dict):
    def to(self, device):
        return self


class _FakeTokenizerConfig:
    """Shaped like a BERT-family tokenizer with no model_max_length
    configured: transformers falls back to its huge unset sentinel."""

    def __init__(self, model_max_length=UNSET_SENTINEL):
        self.model_max_length = model_max_length


class _OversizedTokenizer(_FakeTokenizerConfig):
    """A tokenizer whose input tokenizes to more tokens (1277) than the
    model's real position-embedding limit (512) -- the issue's exact
    scenario. Records the kwargs it was called with so the test can see
    whether a max_length was actually passed through."""

    def __init__(self, raw_token_count=1277, model_max_length=UNSET_SENTINEL):
        super().__init__(model_max_length=model_max_length)
        self.raw_token_count = raw_token_count
        self.last_kwargs = None

    def __call__(
        self, texts, return_tensors="pt", padding=True, truncation=True, max_length=None
    ):
        self.last_kwargs = {"truncation": truncation, "max_length": max_length}
        seq_len = self.raw_token_count
        if truncation and max_length is not None:
            seq_len = min(seq_len, max_length)
        batch = len(texts)
        return _FakeTokenizerOutput(
            {"input_ids": zeros(batch, seq_len), "attention_mask": ones(batch, seq_len)}
        )


class _FakeModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _PositionLimitedEmbedModel:
    """Stands in for a real BERT-family model: its forward pass can only
    add position embeddings up to max_position_embeddings, and raises the
    same shape-mismatch error the issue reports when fed a longer sequence
    (mirrors "expanded size ... must match the existing size ... at
    non-singleton dimension 1")."""

    def __init__(self, max_position_embeddings, dim=1024):
        self.config = types.SimpleNamespace(
            max_position_embeddings=max_position_embeddings
        )
        self._dim = dim

    def to(self, device):
        return self

    def parameters(self):
        yield zeros(1)

    def __call__(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_position_embeddings:
            raise RuntimeError(
                f"The expanded size of the tensor ({seq_len}) must match the "
                f"existing size ({self.config.max_position_embeddings}) at "
                "non-singleton dimension 1."
            )
        batch = input_ids.shape[0]
        hidden = np.ones((batch, seq_len, self._dim), dtype=np.float64)
        return _FakeModelOutput(FakeTensor(hidden))


@pytest.mark.asyncio
async def test_oversized_input_with_unset_tokenizer_limit_does_not_crash(hf_module):
    """The issue's exact reproduction: a 1277-token input against a model
    whose real limit is 512, and a tokenizer whose model_max_length was
    never configured (the huge unset sentinel). Before the fix this
    crashes inside the model's forward pass; the fix must truncate to 512
    so the call succeeds."""
    tokenizer = _OversizedTokenizer(
        raw_token_count=1277, model_max_length=UNSET_SENTINEL
    )
    embed_model = _PositionLimitedEmbedModel(max_position_embeddings=512)

    result = await hf_module.hf_embed(["x" * 5000], tokenizer, embed_model)

    assert result.shape == (1, 1024)
    assert tokenizer.last_kwargs["max_length"] == 512
    assert tokenizer.last_kwargs["truncation"] is True


@pytest.mark.asyncio
async def test_sanely_bounded_tokenizer_limit_is_used_directly(hf_module):
    """When the tokenizer already reports a sane, bounded model_max_length,
    that value is used rather than falling back to the model config."""
    tokenizer = _OversizedTokenizer(raw_token_count=900, model_max_length=384)
    embed_model = _PositionLimitedEmbedModel(max_position_embeddings=512)

    result = await hf_module.hf_embed(["x" * 5000], tokenizer, embed_model)

    assert result.shape == (1, 1024)
    assert tokenizer.last_kwargs["max_length"] == 384


@pytest.mark.asyncio
async def test_input_within_limit_is_left_untruncated(hf_module):
    """A normal, in-range input must not be clipped by the new max_length
    plumbing -- the resolved limit itself is the cap, not a smaller one."""
    tokenizer = _OversizedTokenizer(
        raw_token_count=100, model_max_length=UNSET_SENTINEL
    )
    embed_model = _PositionLimitedEmbedModel(max_position_embeddings=512)

    result = await hf_module.hf_embed(["short text"], tokenizer, embed_model)

    assert result.shape == (1, 1024)
    assert tokenizer.last_kwargs["max_length"] == 512
