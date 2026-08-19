"""hf_embed(): mean pooling must be attention_mask-weighted.

Plain .mean(dim=1) counts padding-token hidden states, so the same text's
embedding shifts depending on what else shares its batch (padding length
varies per batch). attention_mask is already computed and passed to the
model one line above the pooling step, but was unused there.

lightrag/llm/hf.py imports transformers at module level; it isn't installed
in this environment (a real install is a large, unnecessary download for a
pure-tensor test), so it's stubbed in sys.modules. torch itself is a real
install here -- these tests exercise genuine tensor arithmetic, not a fake.
"""

from __future__ import annotations

import sys
import types
import importlib

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.offline


@pytest.fixture
def hf_module(monkeypatch):
    """Import lightrag.llm.hf with transformers stubbed (not installed here)
    and its pipmaster install-check short-circuited."""
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)

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
        yield torch.zeros(1)


@pytest.mark.asyncio
async def test_same_text_gets_the_same_embedding_regardless_of_batch_padding(
    hf_module,
):
    """The real-world symptom: identical text embeds differently only
    because it happened to share a batch with a longer document."""
    dim = 1024
    torch.manual_seed(0)
    real_tokens = torch.randn(1, 3, dim)
    pad_tokens = torch.full((1, 3, dim), 5.0)  # distinct, non-zero padding

    hidden_alone = real_tokens
    hidden_batched = torch.cat([real_tokens, pad_tokens], dim=1)

    embed_model_alone = _FakeEmbedModel(hidden_alone)
    tokenizer_alone = _FakeTokenizer(
        {"input_ids": torch.zeros(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3)}
    )
    emb_alone = await hf_module.hf_embed(["hello"], tokenizer_alone, embed_model_alone)

    embed_model_batched = _FakeEmbedModel(hidden_batched)
    tokenizer_batched = _FakeTokenizer(
        {
            "input_ids": torch.zeros(1, 6, dtype=torch.long),
            "attention_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
        }
    )
    emb_batched = await hf_module.hf_embed(
        ["hello"], tokenizer_batched, embed_model_batched
    )

    assert emb_alone.shape == emb_batched.shape
    assert np.allclose(emb_alone, emb_batched, atol=1e-5)


@pytest.mark.asyncio
async def test_no_padding_present_matches_plain_mean(hf_module):
    """Control: with no padding, masked pooling must reproduce the exact
    same result as a plain, unweighted mean -- this must not regress."""
    dim = 1024
    torch.manual_seed(1)
    hidden = torch.randn(2, 5, dim)
    plain_mean = hidden.mean(dim=1)

    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer(
        {
            "input_ids": torch.zeros(2, 5, dtype=torch.long),
            "attention_mask": torch.ones(2, 5),
        }
    )
    result = await hf_module.hf_embed(["a", "b"], tokenizer, embed_model)

    assert np.allclose(result, plain_mean.numpy(), atol=1e-5)


@pytest.mark.asyncio
async def test_fully_masked_row_does_not_produce_nan_or_inf(hf_module):
    """An all-padding row (e.g. an empty string) must not divide by zero."""
    dim = 1024
    hidden = torch.randn(1, 3, dim)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer(
        {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "attention_mask": torch.zeros(1, 3),  # fully masked
        }
    )

    result = await hf_module.hf_embed([""], tokenizer, embed_model)

    assert bool((result == result).all())  # not NaN
    assert bool((abs(result) < float("inf")).all())


@pytest.mark.asyncio
async def test_output_shape_and_dtype_are_preserved(hf_module):
    dim = 1024
    hidden = torch.randn(3, 4, dim, dtype=torch.float32)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer(
        {
            "input_ids": torch.zeros(3, 4, dtype=torch.long),
            "attention_mask": torch.ones(3, 4),
        }
    )

    result = await hf_module.hf_embed(["a", "b", "c"], tokenizer, embed_model)

    assert result.shape == (3, dim)
    assert result.dtype.name == "float32"


@pytest.mark.asyncio
async def test_bfloat16_conversion_path_still_triggers(hf_module):
    """Regression guard for the existing dtype branch just below pooling."""
    dim = 1024
    hidden = torch.randn(1, 2, dim).to(torch.bfloat16)
    embed_model = _FakeEmbedModel(hidden)
    tokenizer = _FakeTokenizer(
        {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2),
        }
    )

    result = await hf_module.hf_embed(["a"], tokenizer, embed_model)

    assert result.dtype.name == "float32"  # converted from bfloat16 before .numpy()
