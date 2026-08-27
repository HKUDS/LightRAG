"""hf_embed(): mean pooling must be attention_mask-weighted and reduced
in float32, regardless of the model's own hidden-state dtype.

Plain .mean(dim=1) counts padding-token hidden states, so the same text's
embedding shifts depending on what else shares its batch (padding length
varies per batch). attention_mask is already computed and passed to the
model one line above the pooling step, but was unused there.

Separately, accumulating the masked sum and the token count directly in a
low-precision dtype (fp16/bf16) risks overflow (fp16 max ~65504) and loses
exact integer counting (fp16 represents integers exactly only up to 2048;
bf16 only to 256) -- so the reduction itself must run in float32, casting
back to the original hidden-state dtype only for the final result.

lightrag/llm/hf.py imports transformers and torch at module level. Neither
is a project dependency -- both are lazily pip-installed by hf.py itself
only when a caller actually uses the hf binding, and CI's offline test job
(.github/workflows/tests.yml) never installs them. So both are stubbed here:
transformers with a bare placeholder (unused by hf_embed() itself), and
torch with a minimal, numpy-backed FakeTensor that implements exactly the
tensor operations hf_embed()'s pooling step performs (unsqueeze, elementwise
multiply, sum(dim=), clamp_min, divide, dtype comparison, detach/cpu/numpy).

FakeTensor backs most dtypes with float64 (ample precision, so ordinary
correctness assertions aren't sensitive to rounding). The one exception is
float16, which is backed by real numpy float16 -- numpy supports it
natively, so the fp16 overflow and integer-count tests below exercise
genuine low-precision arithmetic, not a simulation of it.
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
FLOAT16 = _FakeDType("float16")
BFLOAT16 = _FakeDType("bfloat16")

# Only float16 gets genuine low-precision numpy backing (numpy has no
# native bfloat16, and float32/bfloat16 tags stay at full float64
# precision internally so ordinary correctness assertions aren't
# sensitive to rounding -- see module docstring).
_GENUINE_NUMPY_DTYPE = {"float16": np.float16}


class FakeTensor:
    """Numpy-backed stand-in for the subset of torch.Tensor that
    hf_embed()'s pooling step actually calls."""

    def __init__(self, array, dtype=FLOAT32):
        backing = _GENUINE_NUMPY_DTYPE.get(dtype.name, np.float64)
        self.array = np.asarray(array, dtype=backing)
        self.dtype = dtype

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.array, dim), self.dtype)

    def to(self, target):
        if isinstance(target, _FakeDType):
            backing = _GENUINE_NUMPY_DTYPE.get(target.name, np.float64)
            return FakeTensor(self.array.astype(backing), target)
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
        backing = _GENUINE_NUMPY_DTYPE.get(self.dtype.name, np.float32)
        return self.array.astype(backing)


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
    fake_torch.float16 = FLOAT16
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


@pytest.mark.asyncio
async def test_fp16_hidden_states_upcast_to_float32_avoid_overflow(hf_module):
    """Codex review: accumulating in fp16 can overflow to infinity on long
    inputs even though the true mean is finite. Confirmed with real
    np.float16 arithmetic (not simulated) that this exact scenario
    overflows when summed at fp16 precision; hf_embed() must upcast to
    float32 before reducing, so its result stays finite."""
    seq = 8192
    dim = 1024

    with np.errstate(over="ignore"):  # the overflow below is the point being proven
        naive_fp16_sum = np.full(seq, 10.0, dtype=np.float16).sum()
    assert not np.isfinite(naive_fp16_sum), (
        "test setup invalid: this scenario doesn't actually overflow in real fp16"
    )

    hidden_np = np.full((1, seq, dim), 10.0, dtype=np.float16)
    hidden = FakeTensor(hidden_np, dtype=FLOAT16)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer(
        {"input_ids": zeros(1, seq), "attention_mask": ones(1, seq)}
    )

    result = await hf_module.hf_embed(["x"], tokenizer, embed_model)

    assert np.isfinite(result).all()
    assert np.allclose(result, 10.0, atol=1e-2)


@pytest.mark.asyncio
async def test_token_count_and_hidden_state_reduction_occur_in_float32(hf_module):
    """seq=3001 exceeds fp16's exact-integer range (2048), so a token
    count -- or a hidden-state sum -- accumulated in fp16 would round.
    Confirmed with real np.float16 arithmetic that this scenario measurably
    diverges between an fp16 reduction and the true (float64) reduction of
    the same fp16-stored values (max abs diff ~0.09 for this seed). hf_embed()
    must land on the precise side: its result -- cast back to fp16 only at
    the very end, per the fix -- should match the true reduction to within a
    single fp16 rounding step (max abs diff ~0.0005 for this seed), not the
    much larger fp16-reduction error."""
    seq = 3001
    dim = 1024
    rng = np.random.default_rng(7)
    hidden_np = rng.uniform(0.5, 2.0, size=(1, seq, dim)).astype(np.float16)

    true_mean = hidden_np.astype(np.float64).mean(axis=1)
    naive_fp16_mean = hidden_np.mean(axis=1, dtype=np.float16)
    assert not np.allclose(naive_fp16_mean, true_mean, atol=2e-3), (
        "test setup invalid: fp16 reduction doesn't actually diverge here"
    )

    hidden = FakeTensor(hidden_np, dtype=FLOAT16)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer(
        {"input_ids": zeros(1, seq), "attention_mask": ones(1, seq)}
    )

    result = await hf_module.hf_embed(["x"], tokenizer, embed_model)

    # atol covers the single expected fp16 rounding step on the final
    # cast-back (~5e-4 observed), while staying far tighter than the
    # naive fp16-reduction error (~0.09 observed) -- so this still fails
    # if the reduction itself regresses to low precision.
    assert np.allclose(result, true_mean, atol=2e-3)


@pytest.mark.asyncio
async def test_pooled_embedding_cast_back_to_original_hidden_state_dtype(hf_module):
    """The final embedding must be cast back to the model's own hidden-
    state dtype (fp16 here), matching pre-fix output-dtype behaviour, even
    though the reduction itself runs in float32."""
    dim = 1024
    hidden_np = np.full((1, 3, dim), 2.0, dtype=np.float16)
    hidden = FakeTensor(hidden_np, dtype=FLOAT16)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer({"input_ids": zeros(1, 3), "attention_mask": ones(1, 3)})

    result = await hf_module.hf_embed(["x"], tokenizer, embed_model)

    assert result.dtype == np.float16
