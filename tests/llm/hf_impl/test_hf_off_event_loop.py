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
import os
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


def test_hf_inference_executor_works_across_successive_event_loops(hf_module):
    async def run_once():
        return await hf_module._run_hf_inference(lambda: "ok")

    assert asyncio.run(run_once()) == "ok"
    assert asyncio.run(run_once()) == "ok"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX-only")
def test_inference_executor_resets_after_fork(hf_module):
    """A forked child (e.g. a gunicorn pre-fork worker) inherits a COPY of
    the parent's ThreadPoolExecutor object and its guard lock, but fork()
    only carries the calling thread into the child -- the pool's own
    worker thread (and, if some other thread held the guard lock at fork
    time, the thread that would release it) do not exist there. Submitting
    through the stale executor would hang forever. os.register_at_fork
    must reset both so the child lazily builds a fresh pair."""
    hf_module._get_hf_inference_executor()
    assert hf_module._HF_INFERENCE_EXECUTOR is not None

    pid = os.fork()
    if pid == 0:
        # Child: exit immediately via os._exit so control never returns to
        # pytest's own machinery here (fixture teardown, etc. must run
        # exactly once, in the parent only).
        ok = hf_module._HF_INFERENCE_EXECUTOR is None
        os._exit(0 if ok else 1)

    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 0


def test_cancelled_queued_hf_inference_does_not_run(hf_module):
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []

    def run(marker):
        calls.append(marker)
        if marker == "first":
            first_started.set()
            release_first.wait(timeout=5)
        return marker

    async def exercise():
        first = asyncio.create_task(hf_module._run_hf_inference(run, "first"))
        assert await asyncio.to_thread(first_started.wait, 5)
        queued = asyncio.create_task(hf_module._run_hf_inference(run, "queued"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release_first.set()
        assert await first == "first"

    asyncio.run(exercise())
    assert calls == ["first"]


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
async def test_hf_model_if_cache_serializes_concurrent_generate_calls(
    hf_module, monkeypatch
):
    """initialize_hf_model caches a single model instance shared by every
    concurrent hf_model_if_cache call (e.g. extract_entities() fanning out
    up to llm_model_max_async chunks). generate() is not safe to call
    concurrently against one model from multiple threads -- blocking the
    event loop used to serialize this by accident; asyncio.to_thread does
    not, so only one generate() call must ever be in flight at a time."""
    in_generate = threading.Event()
    release_generate = threading.Event()
    concurrent_calls = {"count": 0, "max": 0}
    lock = threading.Lock()

    class FakeModel:
        def __init__(self):
            self.device = FakeDevice("cpu")
            self.generation_config = types.SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            with lock:
                concurrent_calls["count"] += 1
                concurrent_calls["max"] = max(
                    concurrent_calls["max"], concurrent_calls["count"]
                )
            in_generate.set()
            release_generate.wait(timeout=5)
            with lock:
                concurrent_calls["count"] -= 1
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

    task1 = asyncio.ensure_future(
        hf_module.hf_model_if_cache("fake-model", "hello world")
    )
    for _ in range(500):
        if in_generate.is_set():
            break
        await asyncio.sleep(0.01)
    assert in_generate.is_set()

    # A second call must queue behind the lock instead of starting a
    # second concurrent generate().
    task2 = asyncio.ensure_future(
        hf_module.hf_model_if_cache("fake-model", "hello again")
    )
    await asyncio.sleep(0.05)
    assert concurrent_calls["count"] == 1

    # release_generate is a one-shot latch: once set it stays set, so
    # task2's own generate() call (once it gets the lock) will not block
    # on it either -- that's fine, the queueing behavior was already
    # proven by the count==1 check above.
    release_generate.set()
    await task1
    await task2

    assert concurrent_calls["max"] == 1


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
async def test_cancelled_generate_remains_serialized_until_worker_finishes(
    hf_module, monkeypatch
):
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    class FakeModel:
        device = FakeDevice("cpu")
        generation_config = types.SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                release_first.wait(timeout=5)
            else:
                second_started.set()
            return FakeTensor([kwargs["input_ids"].data[0] + [901]], "cpu")

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return "<prompt>"

        def __call__(self, *args, **kwargs):
            return {"input_ids": FakeTensor([[1]]), "attention_mask": FakeTensor([[1]])}

        def decode(self, tensor, skip_special_tokens=True):
            return "decoded"

    model = FakeModel()
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        hf_module, "initialize_hf_model", lambda name: (model, tokenizer)
    )

    first = asyncio.create_task(hf_module.hf_model_if_cache("model", "first"))
    assert await asyncio.to_thread(first_started.wait, 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(hf_module.hf_model_if_cache("model", "second"))
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    release_first.set()
    assert await second == "decoded"


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


@pytest.mark.asyncio
async def test_hf_embed_serializes_concurrent_forward_passes(hf_module):
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    class FakeHidden:
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

    class FakeModel:
        def to(self, device):
            return self

        def parameters(self):
            yield FakeHidden()

        def __call__(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                release_first.wait(timeout=5)
            else:
                second_started.set()
            return types.SimpleNamespace(last_hidden_state=FakeHidden())

    class FakeEncoded(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return FakeEncoded(input_ids=FakeHidden(), attention_mask=FakeHidden())

    model = FakeModel()
    tokenizer = FakeTokenizer()
    first = asyncio.create_task(hf_module.hf_embed(["first"], tokenizer, model))
    assert await asyncio.to_thread(first_started.wait, 5)
    second = asyncio.create_task(hf_module.hf_embed(["second"], tokenizer, model))
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    release_first.set()
    await asyncio.gather(first, second)
