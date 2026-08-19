"""hf_embed(): mean pooling must be attention_mask-weighted.

Plain .mean(dim=1) counts padding-token hidden states, so the same text's
embedding shifts depending on what else shares its batch (padding length
varies per batch). attention_mask is already computed and passed to the
model one line above the pooling step, but was unused there.

lightrag/llm/hf.py imports transformers and torch at module level. Neither
is a project dependency -- both are lazily pip-installed by hf.py itself
only when a caller actually uses the hf binding, and CI's offline test job
(.github/workflows/tests.yml) never installs them. So both are stubbed here:
transformers with a bare placeholder (unused by hf_embed() itself), and
torch with a minimal, numpy-backed FakeTensor that implements exactly the
tensor operations hf_embed()'s pooling step performs (unsqueeze, elementwise
multiply, sum(dim=), clamp_min, divide, dtype comparison, detach/cpu/numpy).
Backing it with numpy -- a real, always-installed core dependency -- keeps
the arithmetic genuine rather than a call-recording mock.
"""

from __future__ import annotations

import sys
import types
import importlib

import numpy as np
import pytest

pytestmark = pytest.mark.offline


class _FakeDType:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _FakeDType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


FLOAT32 = _FakeDType("float32")
BFLOAT16 = _FakeDType("bfloat16")


class FakeTensor:
    """Numpy-backed stand-in for the subset of torch.Tensor that
    hf_embed()'s pooling step actually calls."""

    def __init__(self, array, dtype=FLOAT32):
        self.array = np.asarray(array, dtype=np.float64)
        self.dtype = dtype

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.array, dim), self.dtype)

    def to(self, target):
        if isinstance(target, _FakeDType):
            return FakeTensor(self.array, target)
        return self  # device argument -- no-op

    def sum(self, dim):
        return FakeTensor(self.array.sum(axis=dim), self.dtype)

    def clamp_min(self, value):
        return FakeTensor(np.clip(self.array, a_min=value, a_max=None), self.dtype)

    def mean(self, dim):
        return FakeTensor(self.array.mean(axis=dim), self.dtype)

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


def randn(*shape, rng):
    return FakeTensor(rng.standard_normal(shape))


def full(shape, value):
    return FakeTensor(np.full(shape, value))


def tensor(data):
    return FakeTensor(np.array(data))


def cat(tensors, dim):
    return FakeTensor(np.concatenate([t.array for t in tensors], axis=dim))


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
    fake_torch.bfloat16 = BFLOAT16
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


class _FakeTokenizer:
    def __init__(self, encoded):
        self._encoded = encoded

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        return _FakeTokenizerOutput(self._encoded)


class _FakeModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEmbedModel:
    def __init__(self, hidden_states):
        self._hidden_states = hidden_states

    def to(self, device):
        return self

    def __call__(self, input_ids, attention_mask):
        return _FakeModelOutput(self._hidden_states)

    def parameters(self):
        yield zeros(1)


@pytest.mark.asyncio
async def test_same_text_gets_the_same_embedding_regardless_of_batch_padding(
    hf_module,
):
    """The real-world symptom: identical text embeds differently only
    because it happened to share a batch with a longer document."""
    dim = 1024
    rng = np.random.default_rng(0)
    real_tokens = randn(1, 3, dim, rng=rng)
    pad_tokens = full((1, 3, dim), 5.0)  # distinct, non-zero padding

    hidden_alone = real_tokens
    hidden_batched = cat([real_tokens, pad_tokens], dim=1)

    embed_model_alone = _FakeEmbedModel(hidden_alone)
    tokenizer_alone = _FakeTokenizer(
        {"input_ids": zeros(1, 3), "attention_mask": ones(1, 3)}
    )
    emb_alone = await hf_module.hf_embed(["hello"], tokenizer_alone, embed_model_alone)

    embed_model_batched = _FakeEmbedModel(hidden_batched)
    tokenizer_batched = _FakeTokenizer(
        {
            "input_ids": zeros(1, 6),
            "attention_mask": tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
        }
    )
    emb_batched = await hf_module.hf_embed(
        ["hello"], tokenizer_batched, embed_model_batched
    )

    assert emb_alone.shape == emb_batched.shape
    assert np.allclose(emb_alone, emb_batched, atol=1e-8)


@pytest.mark.asyncio
async def test_no_padding_present_matches_plain_mean(hf_module):
    """Control: with no padding, masked pooling must reproduce the exact
    same result as a plain, unweighted mean -- this must not regress."""
    dim = 1024
    rng = np.random.default_rng(1)
    hidden = randn(2, 5, dim, rng=rng)
    plain_mean = hidden.mean(dim=1).numpy()

    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer({"input_ids": zeros(2, 5), "attention_mask": ones(2, 5)})
    result = await hf_module.hf_embed(["a", "b"], tokenizer, embed_model)

    assert np.allclose(result, plain_mean, atol=1e-8)


@pytest.mark.asyncio
async def test_fully_masked_row_does_not_produce_nan_or_inf(hf_module):
    """An all-padding row (e.g. an empty string) must not divide by zero."""
    dim = 1024
    rng = np.random.default_rng(2)
    hidden = randn(1, 3, dim, rng=rng)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer(
        {"input_ids": zeros(1, 3), "attention_mask": zeros(1, 3)}  # fully masked
    )

    result = await hf_module.hf_embed([""], tokenizer, embed_model)

    assert bool(np.isfinite(result).all())


@pytest.mark.asyncio
async def test_output_shape_and_dtype_are_preserved(hf_module):
    dim = 1024
    rng = np.random.default_rng(3)
    hidden = randn(3, 4, dim, rng=rng)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer({"input_ids": zeros(3, 4), "attention_mask": ones(3, 4)})

    result = await hf_module.hf_embed(["a", "b", "c"], tokenizer, embed_model)

    assert result.shape == (3, dim)
    assert result.dtype == np.float32


@pytest.mark.asyncio
async def test_bfloat16_conversion_path_still_triggers(hf_module):
    """Regression guard for the existing dtype branch just below pooling."""
    dim = 1024
    rng = np.random.default_rng(4)
    hidden = randn(1, 2, dim, rng=rng).to(BFLOAT16)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer({"input_ids": zeros(1, 2), "attention_mask": ones(1, 2)})

    result = await hf_module.hf_embed(["a"], tokenizer, embed_model)

    assert result.dtype == np.float32  # converted from bfloat16 before .numpy()
