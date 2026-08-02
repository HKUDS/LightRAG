"""Offline tests for the tokenizer thread-safety contract (GHSA-r8jh / GHSA-26pm).

Token counting is CPU-bound, so it is moved off the asyncio event loop into
worker threads. That makes concurrent calls into one tokenizer routine, and the
contract places responsibility for handling them on the injected implementation
rather than on LightRAG.

Two things have to keep holding for that to be the right call, and each is easy
to break without noticing:

* the built-in ``TiktokenTokenizer`` really is safe to call concurrently — an
  upgrade that changed this would turn every token count into a data race;
* LightRAG really does not serialize callers itself — a lock owned here would be
  waited on by the event loop behind a worker thread, recreating the freeze the
  offload exists to remove.

The tests also pin *why* thread safety, rather than "deep-copy into independent
state", is the property the contract asks for: copies are not independent, and
never were.
"""

import copy
import dataclasses
import threading

import numpy as np
import pytest

from lightrag import LightRAG

# Accessed via the module (not ``from``-imports) on purpose: another test in the
# suite reloads lightrag.utils in place, which would leave from-imported class
# references pointing at the pre-reload class.
from lightrag import utils as lr_utils
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _PlainTokenizer:
    """Minimal conforming implementation: immutable, therefore thread-safe."""

    def encode(self, content: str) -> list[int]:
        return [len(content)]

    def decode(self, tokens: list[int]) -> str:
        return "x" * sum(tokens)


class _OverlapDetectingTokenizer:
    """Records whether ``encode`` is ever entered by two threads at once."""

    def __init__(self, hold: float = 0.0):
        self._hold = hold
        self._inside = 0
        self._guard = threading.Lock()
        self.max_concurrency = 0

    def encode(self, content: str) -> list[int]:
        with self._guard:
            self._inside += 1
            self.max_concurrency = max(self.max_concurrency, self._inside)
        try:
            if self._hold:
                # Plain sleep: this stands in for CPU work, so it must not yield
                # to the event loop.
                threading.Event().wait(self._hold)
            return [len(content)]
        finally:
            with self._guard:
                self._inside -= 1

    def decode(self, tokens: list[int]) -> str:
        return "x" * sum(tokens)


class _LockHoldingTokenizer:
    """Achieves thread safety with an internal lock, the documented way.

    ``copy.deepcopy`` of a ``threading.Lock`` raises ``TypeError: cannot pickle
    '_thread.lock' object``, and ``LightRAG`` is a dataclass whose
    ``_build_global_config`` runs ``dataclasses.asdict``. So the contract asks a
    lock-based implementation to define ``__deepcopy__`` returning ``self`` —
    correct precisely because being thread-safe is what makes it shareable.
    """

    def __init__(self, deepcopyable: bool = True):
        self._lock = threading.Lock()
        if deepcopyable:
            self.__deepcopy__ = lambda _memo: self

    def encode(self, content: str) -> list[int]:
        with self._lock:
            return [len(content)]

    def decode(self, tokens: list[int]) -> str:
        with self._lock:
            return "x" * sum(tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 16))


async def _mock_llm(*_args, **_kwargs) -> str:
    return "mock"


def _make_rag(tmp_path, tokenizer):
    return LightRAG(
        working_dir=str(tmp_path / "tokenizer-contract"),
        workspace="tokenizer-contract",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=16, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# The property the contract now rests on: tiktoken tolerates concurrent calls
# ---------------------------------------------------------------------------


def test_builtin_tokenizer_survives_concurrent_encode_and_decode():
    """Guards the assumption a tiktoken upgrade could silently invalidate.

    ``tiktoken`` itself fans ``Encoding.encode``/``decode`` out across a
    ``ThreadPoolExecutor`` in ``encode_batch``/``decode_batch``, so this is a
    documented capability rather than an accident. If it ever stops holding, the
    whole "the implementation owns its thread safety" contract has to be
    revisited — so it fails here rather than as corrupted token counts.
    """
    tokenizer = lr_utils.TiktokenTokenizer()
    texts = [f"hello {i} 中文测试 " * 200 for i in range(8)]
    expected = [tokenizer.encode(text) for text in texts]

    failures: list[str] = []

    def work(index: int) -> None:
        try:
            for _ in range(10):
                tokens = tokenizer.encode(texts[index])
                if tokens != expected[index]:
                    failures.append(f"encode mismatch at {index}")
                if tokenizer.decode(tokens) != texts[index]:
                    failures.append(f"decode mismatch at {index}")
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(texts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []


# ---------------------------------------------------------------------------
# Why the contract is thread safety and not "deep-copy into independent state"
# ---------------------------------------------------------------------------


def test_builtin_tokenizers_all_share_one_underlying_encoding():
    """``tiktoken`` caches encodings process-wide, so copies are not isolation.

    Every ``TiktokenTokenizer`` for a given model — in LightRAG, in the Ollama
    routes, in the reranker — resolves to the same ``Encoding``. Any design that
    tried to buy thread safety by handing each consumer "its own" tokenizer would
    be reasoning about an object that does not exist.
    """
    first = lr_utils.TiktokenTokenizer()
    second = lr_utils.TiktokenTokenizer()
    assert first.tokenizer is second.tokenizer


def test_deepcopy_of_the_builtin_still_shares_the_same_core_bpe():
    """A deep copy produces a new wrapper over the *same* BPE engine.

    ``Encoding.__getstate__`` returns just the encoding name for a registered
    encoding and ``__setstate__`` rebinds the new object's ``__dict__`` to the
    registered instance's. So the copy is superficial exactly where it would have
    had to be deep. A ``KeyError`` here means tiktoken restructured its
    internals and this reasoning must be re-verified.
    """
    tokenizer = lr_utils.TiktokenTokenizer()
    clone = copy.deepcopy(tokenizer)

    assert clone.tokenizer is not tokenizer.tokenizer  # wrapper differs...
    assert vars(clone.tokenizer)["_core_bpe"] is vars(tokenizer.tokenizer)["_core_bpe"]
    assert clone.encode("hello world") == tokenizer.encode("hello world")


# ---------------------------------------------------------------------------
# LightRAG must not serialize callers, and must not copy the tokenizer
# ---------------------------------------------------------------------------


def test_concurrent_callers_are_not_serialized_by_the_wrapper():
    """Fix-proof against reintroducing a LightRAG-owned tokenizer lock.

    Such a lock would be acquired by whichever thread got there first and waited
    on by the others — including the event loop, which is precisely the freeze
    being removed. Two threads must genuinely overlap inside one wrapper.
    """
    underlying = _OverlapDetectingTokenizer(hold=0.15)
    tokenizer = Tokenizer("test-model", underlying)
    barrier = threading.Barrier(2)

    def work() -> None:
        barrier.wait(timeout=5)
        tokenizer.encode("abcdef")

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert underlying.max_concurrency == 2


def test_build_global_config_hands_out_the_tokenizer_itself(tmp_path):
    """``asdict`` deep-copies it; ``_build_global_config`` restores the original.

    An identity guarantee, not a saving — the copy is still made and then
    discarded, which is why the deep-copy requirement stays in the contract (see
    ``test_a_lock_based_tokenizer_without_deepcopy_fails_at_construction``). What
    it buys is that every consumer reading ``global_config["tokenizer"]`` holds
    the same object as ``LightRAG.tokenizer``, instead of a per-operation copy
    that no one can trace back — and which was never independent anyway.
    """
    tokenizer = Tokenizer("test-model", _PlainTokenizer())
    rag = _make_rag(tmp_path, tokenizer)

    assert rag._build_global_config()["tokenizer"] is tokenizer


def test_a_lock_based_tokenizer_is_injectable_when_it_declares_deepcopy(tmp_path):
    """Pins the escape hatch the deep-copy clause of the contract prescribes.

    An internal lock is a legitimate way to satisfy "be thread-safe", but
    ``LightRAG`` is a dataclass and ``__post_init__`` runs ``asdict`` over it, so
    a bare lock would fail at construction. Declaring ``__deepcopy__`` makes such
    an implementation usable.
    """
    tokenizer = Tokenizer("test-model", _LockHoldingTokenizer())
    rag = _make_rag(tmp_path, tokenizer)

    global_config = rag._build_global_config()
    assert global_config["tokenizer"] is tokenizer
    assert global_config["tokenizer"].encode("abcd") == [4]


def test_a_lock_based_tokenizer_without_deepcopy_fails_at_construction(tmp_path):
    """Documents the boundary the escape hatch exists for.

    This is the failure mode the contract's deep-copy clause warns about, pinned
    so the warning cannot quietly stop being true (or quietly start applying to
    implementations that do declare ``__deepcopy__``).
    """
    tokenizer = Tokenizer("test-model", _LockHoldingTokenizer(deepcopyable=False))

    with pytest.raises(TypeError, match="cannot pickle"):
        _make_rag(tmp_path, tokenizer)


def test_tokenizer_wrapper_is_deepcopyable(tmp_path):
    """The wrapper keeps no un-copyable state of its own.

    Deep copies of a ``LightRAG`` remain possible outside the hot path, so a lock
    stored on the wrapper instance would surface as ``cannot pickle RLock`` here.
    """

    @dataclasses.dataclass
    class Holder:
        tokenizer: object

    tokenizer = Tokenizer("test-model", _PlainTokenizer())
    assert copy.deepcopy(tokenizer).encode("abc") == [3]
    assert dataclasses.asdict(Holder(tokenizer=tokenizer))["tokenizer"].encode(
        "abcd"
    ) == [4]
