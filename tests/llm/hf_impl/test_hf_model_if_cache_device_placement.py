"""hf_model_if_cache(): generation input tensors must follow wherever the
model actually is (hf_model.device), not a hard-coded "cuda".

Before the fix, the tokenizer output was force-moved with `.to("cuda")`,
then a *separate*, correctly device-aware `inputs` dict was built from it
(`v.to(hf_model.device)` per tensor) -- but only used later for prompt-
length slicing. `generate()` itself was called with the stale, hard-coded
`input_ids`, not `inputs`. Two failures followed from this:

1. On a host with no CUDA runtime at all, `.to("cuda")` raised immediately,
   regardless of where the model was loaded.
2. On a host with CUDA, but with this particular model placed on CPU (e.g.
   `device_map="auto"` offload -- the exact condition reported in issue
   #249), `generate()` received CUDA tensors for a CPU model and raised
   "Expected all tensors to be on the same device".

The fix drops the hard-coded `.to("cuda")` and passes the already-correct
`inputs` dict (moved to `hf_model.device`) into `generate()`.

lightrag/llm/hf.py imports transformers and torch at module level. Neither
is a project dependency -- both are lazily pip-installed by hf.py itself,
and CI's offline test job (.github/workflows/tests.yml) never installs
them. So both are stubbed here with a minimal, self-contained FakeTensor
that models exactly the device semantics hf_model_if_cache() depends on
(`.to(device)`, `.device`, `len()`, indexing/slicing) -- not the numeric
pooling behaviour that tests/llm/hf_impl/test_hf_embed_masked_pooling.py
already covers for hf_embed().
"""

from __future__ import annotations

import sys
import types
import importlib

import pytest

pytestmark = pytest.mark.offline


class FakeDevice:
    def __init__(self, name):
        self.type = name.type if isinstance(name, FakeDevice) else name.split(":")[0]

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and other.type == self.type

    def __hash__(self):
        return hash(self.type)

    def __repr__(self):
        return f"FakeDevice({self.type!r})"


class FakeTensor:
    """Minimal stand-in for the subset of torch.Tensor that
    hf_model_if_cache()'s generation path actually calls: `.to(device)`,
    `.device`, `len()`, and integer/slice indexing."""

    def __init__(self, data, device="cpu", host_has_cuda=True):
        self.data = data
        self.device = FakeDevice(device)
        self._host_has_cuda = host_has_cuda

    def to(self, device):
        device = FakeDevice(device)
        if device.type == "cuda" and not self._host_has_cuda:
            # Mirrors real torch: moving to "cuda" on a host with no CUDA
            # runtime raises immediately, regardless of the caller's intent.
            raise RuntimeError("Torch not compiled with CUDA enabled")
        return FakeTensor(self.data, device, self._host_has_cuda)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return FakeTensor(self.data[idx], self.device, self._host_has_cuda)

    def __repr__(self):
        return f"FakeTensor(data={self.data!r}, device={self.device.type!r})"


class FakeModel:
    def __init__(self, device):
        self.device = FakeDevice(device)
        self.received_kwargs = None

    def generate(self, **kwargs):
        self.received_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        if input_ids.device.type != self.device.type:
            # Mirrors real torch's generate() failure when inputs and model
            # weights disagree on device -- this is issue #249's exact error.
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found "
                f"at least two devices, {self.device.type} and "
                f"{input_ids.device.type}:0!"
            )
        prompt_ids = input_ids.data[0]
        generated_ids = prompt_ids + [901, 902]  # arbitrary new-token suffix
        return FakeTensor([generated_ids], self.device, input_ids._host_has_cuda)


class FakeTokenizer:
    def __init__(self, prompt_ids, host_has_cuda=True):
        self._prompt_ids = list(prompt_ids)
        self._host_has_cuda = host_has_cuda

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<rendered-prompt>"

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        # A real HF tokenizer call with no `.to(...)` chained returns
        # tensors on CPU by default -- so does this fake.
        return {
            "input_ids": FakeTensor(
                [list(self._prompt_ids)], "cpu", self._host_has_cuda
            ),
            "attention_mask": FakeTensor(
                [[1] * len(self._prompt_ids)], "cpu", self._host_has_cuda
            ),
        }

    def decode(self, tensor, skip_special_tokens=True):
        return f"decoded:{tensor.data}"


def install_fake_transformers_and_torch(monkeypatch, host_has_cuda):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: host_has_cuda)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)


def make_hf_module(
    monkeypatch, *, model_device, host_has_cuda, prompt_ids=(11, 12, 13)
):
    install_fake_transformers_and_torch(monkeypatch, host_has_cuda)
    sys.modules.pop("lightrag.llm.hf", None)
    hf_module = importlib.import_module("lightrag.llm.hf")

    fake_model = FakeModel(model_device)
    fake_tokenizer = FakeTokenizer(prompt_ids, host_has_cuda)
    monkeypatch.setattr(
        hf_module,
        "initialize_hf_model",
        lambda model_name: (fake_model, fake_tokenizer),
    )
    return hf_module, fake_model, fake_tokenizer


@pytest.mark.asyncio
async def test_cpu_only_host_cpu_model_generation_succeeds(monkeypatch):
    """Item 1: a CPU-only host with the model on CPU must generate
    successfully -- and must never attempt to move anything to CUDA. Before
    the fix, the hard-coded `.to("cuda")` raised here immediately."""
    hf_module, fake_model, _ = make_hf_module(
        monkeypatch, model_device="cpu", host_has_cuda=False
    )

    result = await hf_module.hf_model_if_cache("fake-model", "hello world")

    assert fake_model.received_kwargs["input_ids"].device.type == "cpu"
    # Item 5: prompt-length slicing excludes the prompt tokens from decode.
    assert result == "decoded:[901, 902]"


@pytest.mark.asyncio
async def test_cuda_model_receives_moved_inputs_not_raw_tokenizer_output(monkeypatch):
    """Item 2 + item 4: with the model on CUDA, generate() must receive
    tensors moved to CUDA. The tokenizer's raw output is always CPU (see
    FakeTokenizer.__call__), so this can only pass if generate() was
    actually called with the device-adjusted `inputs` dict, not the
    original, unmoved `input_ids`."""
    hf_module, fake_model, _ = make_hf_module(
        monkeypatch, model_device="cuda", host_has_cuda=True
    )

    result = await hf_module.hf_model_if_cache("fake-model", "hello world")

    assert fake_model.received_kwargs["input_ids"].device.type == "cuda"
    assert fake_model.received_kwargs["attention_mask"].device.type == "cuda"
    assert result == "decoded:[901, 902]"


@pytest.mark.asyncio
async def test_cpu_offloaded_model_on_cuda_capable_host_uses_cpu_tensors(monkeypatch):
    """Item 3: regression test for issue #249. The host supports CUDA, but
    this particular model was placed on CPU (e.g. device_map="auto"
    offload). Generation must receive CPU tensors, matching the model --
    not CUDA tensors forced by a hard-coded `.to("cuda")`, which previously
    raised "Expected all tensors to be on the same device"."""
    hf_module, fake_model, _ = make_hf_module(
        monkeypatch, model_device="cpu", host_has_cuda=True
    )

    result = await hf_module.hf_model_if_cache("fake-model", "hello world")

    assert fake_model.received_kwargs["input_ids"].device.type == "cpu"
    assert result == "decoded:[901, 902]"


@pytest.mark.asyncio
async def test_prompt_length_slicing_excludes_prompt_tokens(monkeypatch):
    """Item 5, isolated: decode must only see the newly generated tokens,
    using inputs["input_ids"] (not the raw tokenizer output) for the
    prompt-length offset."""
    hf_module, fake_model, _ = make_hf_module(
        monkeypatch, model_device="cpu", host_has_cuda=False, prompt_ids=(1, 2, 3, 4, 5)
    )

    result = await hf_module.hf_model_if_cache("fake-model", "hello world")

    full_output = fake_model.received_kwargs is not None
    assert full_output
    assert result == "decoded:[901, 902]"  # prompt ids [1..5] sliced off


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_kwargs", "expected_max_new_tokens"),
    [
        ({}, 512),
        ({"max_tokens": 37}, 37),
        ({"max_new_tokens": 41}, 41),
        ({"max_tokens": 37, "max_new_tokens": 41}, 41),
    ],
)
async def test_generation_token_limit_precedence(
    monkeypatch, generation_kwargs, expected_max_new_tokens
):
    """The HF-native setting wins over LightRAG's generic alias, while
    preserving the existing 512-token default when neither is supplied."""
    hf_module, fake_model, _ = make_hf_module(
        monkeypatch, model_device="cpu", host_has_cuda=False
    )

    await hf_module.hf_model_if_cache("fake-model", "hello world", **generation_kwargs)

    assert fake_model.received_kwargs["max_new_tokens"] == expected_max_new_tokens
    assert "max_tokens" not in fake_model.received_kwargs
