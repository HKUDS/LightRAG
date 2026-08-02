"""Chunking must not hold the event loop (GHSA-26pm-px5v-8c4w).

A 414 KiB ``POST /documents/text`` was answered ``200 OK`` in 1.6 ms and the next
``GET /health`` — a route needing no authentication and touching no storage — took
63 seconds. The response is not the signal: the work happens afterwards, in a
background task, on the only thread serving HTTP.

Every test here measures the same thing: does an ``asyncio.sleep(0)`` heartbeat
advance *while the chunker is inside its blocking section*. Counting beats across
the whole run would prove nothing, since the pipeline awaits plenty of other
things; the stub therefore samples the counter itself, on both sides of its own
sleep.

The branch list is exhaustive on purpose. Testing P/R/F only would miss half the
surface: V degenerates to a synchronous R run when no embedding function is
configured, and the pre-embedding hard split runs for every strategy after all of
them.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
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


def _new_rag(tmp_path: Path, *, with_embedding: bool = True, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"chunk-loop-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_mock_embedding
        )
        if with_embedding
        else None,
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


class _Heartbeat:
    """Counts loop iterations, and lets a worker thread sample the count."""

    def __init__(self):
        self.beats = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.observed: tuple[int, int] | None = None

    def start(self) -> None:
        async def _run():
            while not self._stop.is_set():
                self.beats += 1
                await asyncio.sleep(0)

        self._task = asyncio.create_task(_run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    def sample_across(self, seconds: float = 0.15) -> None:
        """Called FROM the chunker. Records the beat count either side of a
        synchronous sleep, which is the window the loop must stay alive in."""
        before = self.beats
        threading.Event().wait(seconds)
        self.observed = (before, self.beats)

    def assert_loop_stayed_alive(self) -> None:
        assert self.observed is not None, "the instrumented chunker never ran"
        before, after = self.observed
        assert after > before, (
            "the event loop did not advance while the chunker was working "
            f"(beats {before} -> {after})"
        )


async def _ingest(rag: LightRAG, *, doc_id: str, process_options: str, body: str):
    await rag.apipeline_enqueue_documents(
        body,
        ids=[doc_id],
        file_paths=f"{doc_id}.txt",
        track_id=f"track-{doc_id}",
        process_options=process_options,
    )
    await rag.apipeline_process_enqueue_documents()


def _run_with_heartbeat(coro_factory):
    """Run ``coro_factory(heartbeat)`` with a heartbeat task alongside it."""
    heartbeat = _Heartbeat()

    async def _main():
        heartbeat.start()
        try:
            await coro_factory(heartbeat)
        finally:
            await heartbeat.stop()

    asyncio.run(_main())
    return heartbeat


@pytest.mark.parametrize(
    "option,chunker_name",
    [
        ("P", "chunking_by_paragraph_semantic"),
        ("R", "chunking_by_recursive_character"),
        ("F", "chunking_by_fixed_token"),
    ],
)
def test_explicit_strategies_do_not_hold_the_loop(
    tmp_path, monkeypatch, option, chunker_name
):
    import lightrag.chunker as chunker_pkg

    def _make_stub(heartbeat):
        def _stub(tokenizer, content, chunk_token_size, **kwargs):
            heartbeat.sample_across()
            return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

        return _stub

    async def _body(heartbeat):
        monkeypatch.setattr(chunker_pkg, chunker_name, _make_stub(heartbeat))
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id=f"doc-{option}", process_options=option, body="body text"
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_the_builtin_legacy_chunker_does_not_hold_the_loop(tmp_path, monkeypatch):
    """No explicit selector: the ``self.chunking_func`` path."""
    import lightrag.chunker as chunker_pkg

    async def _body(heartbeat):
        rag = _new_rag(tmp_path)
        original = rag.chunking_func

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        # Object identity is what routes to the executor, so the stub has to be
        # installed BOTH on the instance and as the module attribute the
        # dispatcher compares it against — patching only one makes the stub look
        # like a user-supplied chunker and this would silently test the other
        # branch.
        monkeypatch.setattr(rag, "chunking_func", _stub)
        monkeypatch.setattr(chunker_pkg, "chunking_by_token_size", _stub)

        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id="doc-legacy", process_options="", body="body text"
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_v_without_an_embedding_function_does_not_hold_the_loop(monkeypatch):
    """V's ``await asyncio.to_thread`` is never reached in this configuration.

    Without an embedding function the whole call degenerates to a synchronous R
    run, so treating V as "already offloaded" leaves the R attack surface fully
    exposed under an ordinary deployment choice. Driven directly rather than
    through the pipeline because ``LightRAG`` refuses to construct without an
    embedding function at all — the None branch is reachable only by calling the
    chunker.
    """
    import lightrag.chunker.recursive_character as rc_mod
    from lightrag.chunker.semantic_vector import chunking_by_semantic_vector

    async def _body(heartbeat):
        original = rc_mod.chunking_by_recursive_character

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        monkeypatch.setattr(rc_mod, "chunking_by_recursive_character", _stub)

        chunks = await chunking_by_semantic_vector(
            Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
            "body text " * 200,
            120,
            embedding_func=None,
        )
        assert chunks

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_v_post_processing_does_not_hold_the_loop(monkeypatch):
    """The second half of the V path.

    ``await asyncio.to_thread`` covers the embedding-driven grouping only; what
    follows encodes every piece and runs a whole R pass over the oversized ones.
    """
    import lightrag.chunker.recursive_character as rc_mod
    import lightrag.chunker.semantic_vector as sv_mod

    async def _body(heartbeat):
        # One oversized semantic group, so the post-loop has to re-split it.
        monkeypatch.setattr(
            sv_mod,
            "_semantic_groups_with_spans",
            lambda splitter, content: [(content, 0, len(content))],
        )
        monkeypatch.setattr(sv_mod, "SemanticChunker", lambda **kwargs: object())
        monkeypatch.setattr(sv_mod, "_LANGCHAIN_EXPERIMENTAL_AVAILABLE", True)

        original = rc_mod.chunking_by_recursive_character

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        monkeypatch.setattr(rc_mod, "chunking_by_recursive_character", _stub)

        chunks = await chunking_by_semantic_vector_stub()
        assert chunks

    async def chunking_by_semantic_vector_stub():
        return await sv_mod.chunking_by_semantic_vector(
            Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
            "body text " * 400,
            50,
            embedding_func=EmbeddingFunc(
                embedding_dim=32, max_token_size=4096, func=_mock_embedding
            ),
        )

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_the_pre_embedding_hard_split_does_not_hold_the_loop(tmp_path, monkeypatch):
    """Runs after chunking, for every strategy — so offloading the branches is
    not on its own enough."""
    import lightrag.pipeline as pipeline_mod

    async def _body(heartbeat):
        original = pipeline_mod.enforce_chunk_token_limit_before_embedding

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        monkeypatch.setattr(
            pipeline_mod, "enforce_chunk_token_limit_before_embedding", _stub
        )

        rag = _new_rag(tmp_path)
        rag.embedding_token_limit = 32
        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id="doc-hardsplit", process_options="F", body="body text " * 80
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_a_chunker_failure_still_fails_the_document(tmp_path, monkeypatch):
    """Exceptions must survive the executor hop with their handling intact."""
    import lightrag.chunker as chunker_pkg

    def _boom(*args, **kwargs):
        raise RuntimeError("chunker exploded")

    monkeypatch.setattr(chunker_pkg, "chunking_by_fixed_token", _boom)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(rag, doc_id="doc-boom", process_options="F", body="body text")
            row = await rag.doc_status.get_by_id("doc-boom")
            assert row is not None
            # DocStatus is a ``str, Enum``: compare on the value, since ``str()``
            # of the member renders as "DocStatus.FAILED".
            assert DocStatus(row["status"]) is DocStatus.FAILED
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def _with_blocks_path(monkeypatch, blocks_path: Path):
    """Make the sidecar-dependent branches reachable from a plain-text ingest.

    A raw text document parses to ``blocks_path=''``, so the branches guarded by
    it never run. Injecting the path at the process-stage boundary is the
    smallest intervention that exercises them without standing up a parser that
    emits real artifacts.
    """
    import lightrag.pipeline as pipeline_mod

    original = pipeline_mod._PipelineMixin.process_single_document

    async def _patched(self, *, doc_id, status_doc, parsed_data, ctx):
        parsed_data["blocks_path"] = str(blocks_path)
        return await original(
            self, doc_id=doc_id, status_doc=status_doc, parsed_data=parsed_data, ctx=ctx
        )

    monkeypatch.setattr(
        pipeline_mod._PipelineMixin, "process_single_document", _patched
    )


def test_the_sidecar_backfill_does_not_hold_the_loop(tmp_path, monkeypatch):
    """It sits between chunking and the storage writes, and is not cheap.

    ``backfill_chunk_sidecars`` parses blocks.jsonl and then scans every block
    for every chunk — O(chunks x blocks) of CPU plus synchronous file I/O. A
    document with many chunks freezes the loop here for seconds, so offloading
    the chunkers alone leaves the same stall a little further down the function.
    """
    import lightrag.sidecar as sidecar_mod

    async def _body(heartbeat):
        def _stub(chunking_result, blocks_path):
            heartbeat.sample_across()

        monkeypatch.setattr(sidecar_mod, "backfill_chunk_sidecars", _stub)
        _with_blocks_path(monkeypatch, tmp_path / "doc-backfill.blocks.jsonl")

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(
                rag,
                doc_id="doc-backfill",
                process_options="F",
                body="body text " * 200,
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_the_multimodal_chunk_builder_does_not_hold_the_loop(tmp_path, monkeypatch):
    """Its truncation loop is driven by VLM output length, which no request-side
    ceiling bounds."""
    import lightrag.pipeline as pipeline_mod

    async def _body(heartbeat):
        def _stub(self, **kwargs):
            heartbeat.sample_across()
            return []

        monkeypatch.setattr(
            pipeline_mod._PipelineMixin, "_build_mm_chunks_from_sidecars", _stub
        )
        _with_blocks_path(monkeypatch, tmp_path / "doc-mmchunks.blocks.jsonl")

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(
                rag,
                doc_id="doc-mmchunks",
                process_options="F",
                body="body text " * 200,
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()
