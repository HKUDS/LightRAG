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
import subprocess
import sys
import threading
import types
import importlib
from pathlib import Path

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
def test_inference_executor_resets_after_fork():
    """A forked child (e.g. a gunicorn pre-fork worker) inherits a COPY of
    the parent's ThreadPoolExecutor object and its guard lock, but fork()
    only carries the calling thread into the child -- the pool's own
    worker thread (and, if some other thread held the guard lock at fork
    time, the thread that would release it) do not exist there. Submitting
    through the stale executor would hang forever. os.register_at_fork
    must reset both so the child lazily builds a fresh pair.

    The fork runs in a subprocess, not here: a fresh single-threaded
    interpreter is the only place CPython's multi-threaded-fork warning can
    be asserted absent instead of filtered away. _fork_probe.py explains the
    rest, including the native thread-pool caps that keep it single-threaded."""
    probe = Path(__file__).with_name("_fork_probe.py")
    # The probe caps the BLAS/OpenMP pools itself, so running it directly
    # behaves the same as running it from here.
    result = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fork probe failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


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
    # Teardown -- shutting the inference pool down and dropping the module --
    # belongs to the autouse fixture in this directory's conftest, which also
    # covers the sibling files that build the module through a plain helper.
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
async def test_hf_model_if_cache_serializes_model_loading_with_generate(
    hf_module, monkeypatch
):
    """initialize_hf_model() is an lru_cache(maxsize=1). If it runs on the
    event loop before the await, a second concurrent call with a different
    model_name can load its own model while the first call's model is still
    resident and mid-generate(), doubling peak GPU memory. Model
    acquisition must be serialized together with generate() inside the
    single-worker executor so at most one model is ever in flight."""
    live_models = set()
    peak_live_models = {"count": 0}
    registry_lock = threading.Lock()
    a_generate_started = threading.Event()
    b_initialize_called = threading.Event()
    release_a = threading.Event()

    class FakeModel:
        def __init__(self, name):
            self.name = name
            self.device = FakeDevice("cpu")
            self.generation_config = types.SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            if self.name == "model-a":
                a_generate_started.set()
                release_a.wait(timeout=5)
            with registry_lock:
                live_models.discard(self.name)
            input_ids = kwargs["input_ids"]
            return FakeTensor([input_ids.data[0] + [901]], "cpu")

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return "<prompt>"

        def __call__(self, *args, **kwargs):
            return {
                "input_ids": FakeTensor([[1]]),
                "attention_mask": FakeTensor([[1]]),
            }

        def decode(self, tensor, skip_special_tokens=True):
            return f"decoded:{tensor.data}"

    def fake_initialize(model_name):
        if model_name == "model-b":
            b_initialize_called.set()
        with registry_lock:
            live_models.add(model_name)
            peak_live_models["count"] = max(peak_live_models["count"], len(live_models))
        return FakeModel(model_name), FakeTokenizer()

    monkeypatch.setattr(hf_module, "initialize_hf_model", fake_initialize)

    task_a = asyncio.create_task(hf_module.hf_model_if_cache("model-a", "hello a"))
    assert await asyncio.to_thread(a_generate_started.wait, 5)

    # A second call with a different model_name must queue behind the
    # single worker instead of loading its own model while A's model is
    # still resident and mid-generate().
    task_b = asyncio.create_task(hf_module.hf_model_if_cache("model-b", "hello b"))
    await asyncio.sleep(0.05)
    assert not b_initialize_called.is_set()

    release_a.set()
    await task_a
    await task_b

    assert peak_live_models["count"] == 1


@pytest.mark.asyncio
async def test_hf_model_if_cache_keeps_decoding_inside_the_serialized_job(
    hf_module, monkeypatch
):
    """Returning hf_model/inputs/output for the caller to decode afterwards
    would keep model A's references alive in this coroutine's locals while
    the now-free worker starts loading model B for a second queued call --
    reopening the residency window generate()-serialization was meant to
    close, just narrower. Decoding and the truncation check must run inside
    the closure that holds the executor slot instead."""
    a_decode_started = threading.Event()
    release_a_decode = threading.Event()
    b_initialize_called = threading.Event()

    class FakeModel:
        def __init__(self, name):
            self.name = name
            self.device = FakeDevice("cpu")
            self.generation_config = types.SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            return FakeTensor([input_ids.data[0] + [901]], "cpu")

    class FakeTokenizer:
        def __init__(self, name):
            self.name = name
            self.eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return "<prompt>"

        def __call__(self, *args, **kwargs):
            return {
                "input_ids": FakeTensor([[1]]),
                "attention_mask": FakeTensor([[1]]),
            }

        def decode(self, tensor, skip_special_tokens=True):
            if self.name == "model-a":
                a_decode_started.set()
                release_a_decode.wait(timeout=5)
            return f"decoded:{tensor.data}"

    def fake_initialize(model_name):
        if model_name == "model-b":
            b_initialize_called.set()
        return FakeModel(model_name), FakeTokenizer(model_name)

    monkeypatch.setattr(hf_module, "initialize_hf_model", fake_initialize)

    task_a = asyncio.create_task(hf_module.hf_model_if_cache("model-a", "hello a"))
    assert await asyncio.to_thread(a_decode_started.wait, 5)

    # generate() has already returned by now -- only decode() is still
    # running. A second queued call must not be able to start loading its
    # own model until decode() (still inside A's closure) finishes too.
    task_b = asyncio.create_task(hf_module.hf_model_if_cache("model-b", "hello b"))
    await asyncio.sleep(0.05)
    assert not b_initialize_called.is_set()

    release_a_decode.set()
    await task_a
    await task_b

    assert b_initialize_called.is_set()


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
