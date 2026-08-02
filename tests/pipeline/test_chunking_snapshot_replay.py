"""A pre-fix separator snapshot must not re-freeze the worker on resume.

``chunk_options`` is snapshotted at ENQUEUE time into ``full_docs``, so capping
``separators`` on the request model only protects new requests. A build that
still accepted 900 separators persisted them AND left the document in
``PROCESSING`` when the worker froze — and ``PROCESSING`` is an auto-resume
status, so an upgraded server reloads the stored cascade and freezes again. That
is a boot loop restarting cannot clear, which is why the cap has to live where
every source of a cascade passes through rather than only at the HTTP boundary.

The same reasoning already appears for ``sentence_split_regex`` in
``apply_trusted_sentence_split_regex``; this is the R-strategy analogue.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import MAX_R_SEPARATORS
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

# What a pre-fix build would have accepted and stored.
POISONED_SEPARATORS = ["Q"] * 900


class _CountingTokenizerImpl:
    def __init__(self):
        self.encodes = 0

    def encode(self, content: str):
        self.encodes += 1
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.full((len(texts), 32), 0.1, dtype=np.float32)


async def _mock_llm(prompt, **kwargs):
    return '{"name":"x","summary":"s","detail_description":"d"}'


def _new_rag(tmp_path: Path, tokenizer_impl) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"snapshot-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", tokenizer_impl),
    )


def test_a_poisoned_snapshot_is_bounded_at_process_time(tmp_path, monkeypatch):
    """End-to-end: the stored cascade reaches the chunker already bounded.

    Asserting on what the chunker *received* rather than on the stored row: a
    fix that bounded the value on the way in but splatted the original on the
    way out would pass a source-level reading and fail here.
    """
    import lightrag.chunker as chunker_pkg
    import lightrag.chunker.recursive_character as rc_mod

    captured: dict = {}
    original_chunker = chunker_pkg.chunking_by_recursive_character

    def _chunker_spy(tokenizer, content, chunk_token_size, **kwargs):
        # What the dispatcher handed over, i.e. straight out of the snapshot.
        captured["dispatched"] = kwargs.get("separators")
        return original_chunker(tokenizer, content, chunk_token_size, **kwargs)

    original_normalize = rc_mod.normalize_r_separators

    def _normalize_spy(separators, **kwargs):
        result = original_normalize(separators, **kwargs)
        captured["normalized"] = result
        return result

    monkeypatch.setattr(chunker_pkg, "chunking_by_recursive_character", _chunker_spy)
    monkeypatch.setattr(rc_mod, "normalize_r_separators", _normalize_spy)

    async def _run():
        rag = _new_rag(tmp_path, _CountingTokenizerImpl())
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Q" + ("X " * 2000),
                ids=["doc-poisoned-snapshot"],
                file_paths="poisoned.txt",
                track_id="track-poisoned",
                process_options="R",
                chunk_options={
                    "recursive_character": {"separators": POISONED_SEPARATORS}
                },
            )
            # The snapshot really does carry the payload, or this test would
            # pass for the wrong reason.
            row = await rag.full_docs.get_by_id("doc-poisoned-snapshot")
            stored = row["chunk_options"]["recursive_character"]["separators"]
            assert len(stored) == len(POISONED_SEPARATORS)

            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())

    assert captured.get("dispatched") is not None, "the R chunker never ran"
    # Bounded by the time it is dispatched. The dispatcher normalizes before it
    # formats, logs or splats, so nothing downstream — including the chunker —
    # ever handles the stored cascade at full size.
    assert len(captured["dispatched"]) <= MAX_R_SEPARATORS
    # The chunker normalizes too, and must: it is the chokepoint for callers
    # that never pass through the dispatcher at all.
    assert captured.get("normalized") is not None
    assert len(captured["normalized"]) <= MAX_R_SEPARATORS


def test_replaying_a_poisoned_snapshot_costs_a_bounded_number_of_encodes(
    tmp_path,
):
    """The cost, not just the shape.

    900 separators over the advisory's payload used to mean ~900 whole-text
    encodes for a result of one chunk. Encodes rather than wall clock: timing is
    unreliable in CI, and the encodes are what the seconds were made of.
    """
    tokenizer_impl = _CountingTokenizerImpl()

    async def _run():
        rag = _new_rag(tmp_path, tokenizer_impl)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Q" + ("X " * 2000),
                ids=["doc-poisoned-cost"],
                file_paths="poisoned-cost.txt",
                track_id="track-poisoned-cost",
                process_options="R",
                chunk_options={
                    "recursive_character": {"separators": POISONED_SEPARATORS}
                },
            )
            before = tokenizer_impl.encodes
            await rag.apipeline_process_enqueue_documents()
            return tokenizer_impl.encodes - before
        finally:
            await rag.finalize_storages()

    encodes = asyncio.run(_run())

    # Generous: the pipeline encodes for its own bookkeeping too. The point is
    # that it does not scale with the length of the cascade.
    assert encodes < len(POISONED_SEPARATORS) / 4


def test_an_oversized_snapshot_never_reaches_the_log_or_doc_status(tmp_path):
    """The cascade must be bounded before anything RENDERS it.

    Bounding it inside the chunker is too late for one consumer that runs first:
    ``_format_chunking_params`` interpolates the kwargs about to be splatted into
    a string that is logged at INFO and persisted to
    ``doc_status.metadata['chunk_opts']``. A snapshot holding a single 5 MB
    separator therefore built a ~5 MB parameter string on the event loop and
    pushed it into doc_status — where it can exceed a backend's value limit and
    strand the document in FAILED — even though the cascade the chunker actually
    used was empty, every entry having been dropped as over-long.

    Asserting on the persisted string rather than on log capture: it is the
    durable half, and the one that can break a storage write.
    """
    huge_separator = "Z" * (5 * 1024 * 1024)

    async def _run():
        rag = _new_rag(tmp_path, _CountingTokenizerImpl())
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Q" + ("X " * 2000),
                ids=["doc-huge-separator"],
                file_paths="huge-separator.txt",
                track_id="track-huge-separator",
                process_options="R",
                chunk_options={"recursive_character": {"separators": [huge_separator]}},
            )
            await rag.apipeline_process_enqueue_documents()
            return await rag.doc_status.get_by_id("doc-huge-separator")
        finally:
            await rag.finalize_storages()

    row = asyncio.run(_run())

    rendered = (row.get("metadata") or {}).get("chunk_opts", "")
    # Pre-fix this was len(huge_separator) + change. The bound is deliberately
    # loose — the point is that it does not scale with the separator.
    assert len(rendered) < 4096, f"chunk_opts rendered {len(rendered)} characters"
    assert huge_separator not in rendered
