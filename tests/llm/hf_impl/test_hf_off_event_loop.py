"""hf_model_if_cache() and hf_embed() run real, synchronous PyTorch compute
(hf_model.generate() / a forward pass) that can take seconds to minutes.
Calling either directly from these async functions would block the whole
event loop for that duration, stalling every other concurrent task. Both
must run their blocking call on a worker thread instead.

lightrag/llm/hf.py imports transformers and torch at module level. Neither
is a project dependency, so both are stubbed here, same approach as the
sibling hf tests in this directory.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
import importlib

import numpy as np
import pytest

pytestmark = pytest.mark.offline


class FakeDevice:
    def __init__(self, name):
        self.type = name


class FakeTensor:
    def __init__(self, data, device="cpu"):
        self.data = data
        self.device = FakeDevice(device)

    def to(self, device):
        return FakeTensor(self.data, device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return FakeTensor(self.data[idx], self.device.type)

    def item(self):
        return self.data


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
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
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


@pytest.mark.asyncio
async def test_hf_model_if_cache_runs_generate_off_the_event_loop_thread(
    hf_module, monkeypatch
):
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    class FakeModel:
        def __init__(self):
            self.device = FakeDevice("cpu")
            self.generation_config = types.SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            call_thread_id["id"] = threading.get_ident()
            input_ids = kwargs["input_ids"]
            return FakeTensor([input_ids.data[0] + [901, 902]], "cpu")

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return "<prompt>"

        def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
            return {
                "input_ids": FakeTensor([[1, 2, 3]]),
                "attention_mask": FakeTensor([[1, 1, 1]]),
            }

        def decode(self, tensor, skip_special_tokens=True):
            return f"decoded:{tensor.data}"

    fake_model = FakeModel()
    monkeypatch.setattr(
        hf_module, "initialize_hf_model", lambda name: (fake_model, FakeTokenizer())
    )

    result = await hf_module.hf_model_if_cache("fake-model", "hello world")

    assert result == "decoded:[901, 902]"
    assert call_thread_id["id"] != main_thread_id


@pytest.mark.asyncio
async def test_hf_model_if_cache_logs_and_repropagates_cancellation(
    hf_module, monkeypatch
):
    """Cancelling the outer await (e.g. an execution timeout) still has to
    propagate CancelledError, with a warning noting generate() keeps
    running -- and keeps holding its allocated memory -- in the background
    thread until it completes on its own."""
    call_started = threading.Event()
    release_call = threading.Event()
    warnings_logged = []

    class FakeModel:
        def __init__(self):
            self.device = FakeDevice("cpu")
            self.generation_config = types.SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            call_started.set()
            release_call.wait(timeout=5)
            input_ids = kwargs["input_ids"]
            return FakeTensor([input_ids.data[0] + [901, 902]], "cpu")

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return "<prompt>"

        def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
            return {
                "input_ids": FakeTensor([[1, 2, 3]]),
                "attention_mask": FakeTensor([[1, 1, 1]]),
            }

        def decode(self, tensor, skip_special_tokens=True):
            return f"decoded:{tensor.data}"

    monkeypatch.setattr(
        hf_module, "initialize_hf_model", lambda name: (FakeModel(), FakeTokenizer())
    )
    monkeypatch.setattr(
        hf_module.logger, "warning", lambda msg: warnings_logged.append(msg)
    )

    task = asyncio.ensure_future(
        hf_module.hf_model_if_cache("fake-model", "hello world")
    )
    for _ in range(500):
        if call_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert call_started.is_set()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    release_call.set()
    assert len(warnings_logged) == 1
    assert "cancelled while awaiting generate()" in warnings_logged[0]


@pytest.mark.asyncio
async def test_hf_embed_runs_forward_pass_off_the_event_loop_thread(hf_module):
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    class _FakeModelOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    class _FakeHidden:
        dtype = "float32"

        def unsqueeze(self, dim):
            return self

        def to(self, target):
            return self

        def __mul__(self, other):
            return self

        def sum(self, dim):
            return self

        def clamp_min(self, value):
            return self

        def __truediv__(self, other):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 1024), dtype=np.float32)

    class _FakeEmbedModel:
        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            call_thread_id["id"] = threading.get_ident()
            return _FakeModelOutput(_FakeHidden())

        def parameters(self):
            yield _FakeHidden()

    class _FakeTokenizerOutput(dict):
        def to(self, device):
            return self

    class _FakeTokenizer:
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
            return _FakeTokenizerOutput(
                {"input_ids": _FakeHidden(), "attention_mask": _FakeHidden()}
            )

    result = await hf_module.hf_embed(["hello"], _FakeTokenizer(), _FakeEmbedModel())

    assert result.shape == (1, 1024)
    assert call_thread_id["id"] != main_thread_id


@pytest.mark.asyncio
async def test_hf_embed_runs_cpu_conversion_off_the_event_loop_thread(hf_module):
    """.cpu() on a CUDA tensor synchronizes the device (blocks until
    pending GPU work finishes), which can take as long as the forward pass
    itself -- it must run in the same background thread, not back on the
    event loop after the forward pass returns."""
    main_thread_id = threading.get_ident()
    cpu_call_thread_id = {}

    class _FakeModelOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    class _FakeHidden:
        dtype = "float32"

        def unsqueeze(self, dim):
            return self

        def to(self, target):
            return self

        def __mul__(self, other):
            return self

        def sum(self, dim):
            return self

        def clamp_min(self, value):
            return self

        def __truediv__(self, other):
            return self

        def detach(self):
            return self

        def cpu(self):
            cpu_call_thread_id["id"] = threading.get_ident()
            return self

        def numpy(self):
            return np.zeros((1, 1024), dtype=np.float32)

    class _FakeEmbedModel:
        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            return _FakeModelOutput(_FakeHidden())

        def parameters(self):
            yield _FakeHidden()

    class _FakeTokenizerOutput(dict):
        def to(self, device):
            return self

    class _FakeTokenizer:
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
            return _FakeTokenizerOutput(
                {"input_ids": _FakeHidden(), "attention_mask": _FakeHidden()}
            )

    await hf_module.hf_embed(["hello"], _FakeTokenizer(), _FakeEmbedModel())

    assert cpu_call_thread_id["id"] != main_thread_id


@pytest.mark.asyncio
async def test_hf_embed_logs_and_repropagates_cancellation(hf_module, monkeypatch):
    call_started = threading.Event()
    release_call = threading.Event()
    warnings_logged = []

    class _FakeModelOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    class _FakeHidden:
        dtype = "float32"

        def unsqueeze(self, dim):
            return self

        def to(self, target):
            return self

        def __mul__(self, other):
            return self

        def sum(self, dim):
            return self

        def clamp_min(self, value):
            return self

        def __truediv__(self, other):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 1024), dtype=np.float32)

    class _FakeEmbedModel:
        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            call_started.set()
            release_call.wait(timeout=5)
            return _FakeModelOutput(_FakeHidden())

        def parameters(self):
            yield _FakeHidden()

    class _FakeTokenizerOutput(dict):
        def to(self, device):
            return self

    class _FakeTokenizer:
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
            return _FakeTokenizerOutput(
                {"input_ids": _FakeHidden(), "attention_mask": _FakeHidden()}
            )

    monkeypatch.setattr(
        hf_module.logger, "warning", lambda msg: warnings_logged.append(msg)
    )

    task = asyncio.ensure_future(
        hf_module.hf_embed(["hello"], _FakeTokenizer(), _FakeEmbedModel())
    )
    for _ in range(500):
        if call_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert call_started.is_set()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    release_call.set()
    assert len(warnings_logged) == 1
    assert "cancelled while awaiting the forward pass" in warnings_logged[0]
