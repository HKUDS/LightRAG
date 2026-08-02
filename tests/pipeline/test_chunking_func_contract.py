"""The ``chunking_func`` extension contract survives the executor change.

Moving chunking off the event loop must not move the *extension point* off it.
``chunking_func`` is typed ``Union[List[Dict], Awaitable[List[Dict]]]`` and its
docstring opens with "Synchronous or async", and the pipeline awaits whatever it
returns — so "it is called synchronously today, therefore it cannot depend on the
running loop" is not a sound inference. A synchronous factory that touches the
loop when called is a supported implementation and would fail outright in a
worker thread; an ``async def`` would gain nothing from the hop, since its body
runs on the loop either way.

Only the built-in default is dispatched to the executor. These tests pin that
split in both directions.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.full((len(texts), 32), 0.1, dtype=np.float32)


async def _mock_llm(prompt, **kwargs):
    return '{"name":"x","summary":"s","detail_description":"d"}'


def _new_rag(tmp_path: Path, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"chunkfunc-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


def _chunks():
    return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]


async def _ingest(rag: LightRAG, doc_id: str):
    await rag.apipeline_enqueue_documents(
        f"body text for {doc_id}",
        ids=[doc_id],
        file_paths=f"{doc_id}.txt",
        track_id=f"track-{doc_id}",
        process_options="",
    )
    await rag.apipeline_process_enqueue_documents()


def _run(tmp_path, chunking_func, doc_id):
    async def _main():
        rag = _new_rag(tmp_path, chunking_func=chunking_func)
        await rag.initialize_storages()
        try:
            await _ingest(rag, doc_id)
        finally:
            await rag.finalize_storages()

    asyncio.run(_main())


def test_a_synchronous_custom_chunker_still_works(tmp_path):
    seen = {}

    def _custom(tokenizer, content, *args, **kwargs):
        seen["called"] = True
        return _chunks()

    _run(tmp_path, _custom, "doc-sync")
    assert seen.get("called") is True


def test_an_async_custom_chunker_is_still_awaited(tmp_path):
    seen = {}

    async def _custom(tokenizer, content, *args, **kwargs):
        seen["called"] = True
        return _chunks()

    _run(tmp_path, _custom, "doc-async")
    assert seen.get("called") is True


def test_a_custom_chunker_that_touches_the_running_loop_still_works(tmp_path):
    """The case that rules out "just run everything in a thread".

    A synchronous factory calling ``get_running_loop()`` / ``create_task()`` is a
    supported implementation of this contract. Dispatched to a worker thread it
    raises ``RuntimeError: no running event loop`` before doing any work.
    """
    seen = {}

    def _custom(tokenizer, content, *args, **kwargs):
        loop = asyncio.get_running_loop()
        seen["loop"] = loop is not None

        async def _produce():
            return _chunks()

        return loop.create_task(_produce())

    _run(tmp_path, _custom, "doc-loop-aware")
    assert seen.get("loop") is True


def test_a_custom_chunker_runs_on_the_event_loop_and_the_builtin_does_not(tmp_path):
    """Pins the split itself, in both directions.

    The custom implementation seeing a running loop and the built-in one not is
    the whole of the dispatch rule; asserting only one side would let a change
    that routes everything one way pass.
    """
    observations: dict[str, bool] = {}

    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _custom(tokenizer, content, *args, **kwargs):
        observations["custom_on_loop"] = _has_running_loop()
        return _chunks()

    _run(tmp_path, _custom, "doc-custom-side")

    import lightrag.chunker as chunker_pkg

    original_builtin = chunker_pkg.chunking_by_token_size

    def _builtin_spy(*args, **kwargs):
        observations["builtin_on_loop"] = _has_running_loop()
        observations["builtin_thread"] = threading.current_thread().name
        return original_builtin(*args, **kwargs)

    async def _main():
        rag = _new_rag(tmp_path)
        # Identity selects the branch, so both references must be the spy.
        chunker_pkg.chunking_by_token_size = _builtin_spy
        rag.chunking_func = _builtin_spy
        await rag.initialize_storages()
        try:
            await _ingest(rag, "doc-builtin-side")
        finally:
            await rag.finalize_storages()
            chunker_pkg.chunking_by_token_size = original_builtin

    asyncio.run(_main())

    assert observations["custom_on_loop"] is True
    assert observations["builtin_on_loop"] is False
    assert observations["builtin_thread"].startswith("lightrag-chunking")
