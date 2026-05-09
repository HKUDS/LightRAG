"""Tests for the ``chunk_options`` snapshot mechanism.

Three properties under test:

1. **env-driven snapshot**: env vars (CHUNK_R_OVERLAP_SIZE,
   CHUNK_V_BREAKPOINT_THRESHOLD_TYPE, …) flow into
   ``addon_params['chunker']`` via
   :func:`lightrag.parser_routing.default_chunker_config`, then into
   ``full_docs[doc_id]['chunk_options']`` at enqueue time via
   :func:`lightrag.parser_routing.resolve_chunk_options`.

2. **caller-supplied chunk_options**: an explicit ``chunk_options``
   kwarg passed to ``apipeline_enqueue_documents`` is persisted
   verbatim and reaches the dispatched chunker as keyword args.

3. **per-file chunk_options as a list**: when chunk_options is a
   ``list[dict]`` aligned with ``input``, each doc gets its own
   independent persisted snapshot.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


class _SimpleTokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt, **kwargs):
    return '{"name":"x","summary":"s","detail_description":"d"}'


_ROLE_FIELD_SUFFIXES = (
    ("_llm_model_func", "func"),
    ("_llm_model_kwargs", "kwargs"),
    ("_llm_model_max_async", "max_async"),
    ("_llm_timeout", "timeout"),
)


def _new_rag(tmp_path: Path, **kwargs) -> LightRAG:
    role_configs: dict[str, RoleLLMConfig] = {}
    for spec in ROLES:
        bucket = {}
        for suffix, target in _ROLE_FIELD_SUFFIXES:
            key = f"{spec.name}{suffix}"
            if key in kwargs:
                bucket[target] = kwargs.pop(key)
        if bucket:
            role_configs[spec.name] = RoleLLMConfig(**bucket)
    if role_configs:
        kwargs["role_llm_configs"] = role_configs

    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"chunkopts-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


@pytest.mark.offline
def test_env_driven_snapshot_persisted_in_full_docs(tmp_path, monkeypatch):
    """Env vars + ainsert split args land in ``full_docs.chunk_options``."""
    monkeypatch.setenv("CHUNK_R_OVERLAP_SIZE", "42")
    monkeypatch.setenv("CHUNK_V_BREAKPOINT_THRESHOLD_TYPE", "interquartile")
    monkeypatch.setenv("CHUNK_V_BUFFER_SIZE", "3")

    doc_id = "doc-snap-aaaaa"

    async def _run():
        from lightrag.parser_routing import resolve_chunk_options

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Mirror what ``LightRAG.ainsert`` does: build the
            # chunk_options snapshot from addon_params + F-strategy
            # runtime args, then hand it to enqueue.  Avoids ainsert
            # itself so the test stays a focused enqueue-only check.
            chunk_opts = resolve_chunk_options(
                rag.addon_params,
                split_by_character="\n\n",
                split_by_character_only=True,
            )
            await rag.apipeline_enqueue_documents(
                "Body for chunk_options snapshot test.",
                ids=[doc_id],
                file_paths="snap.txt",
                track_id="track-snap",
                chunk_options=chunk_opts,
            )
            row = await rag.full_docs.get_by_id(doc_id)
        finally:
            await rag.finalize_storages()
        return row

    row = asyncio.run(_run())
    assert row is not None, "doc must be persisted to full_docs"
    chunk_opts = row.get("chunk_options")

    assert chunk_opts is not None, "chunk_options must be persisted"
    assert chunk_opts["recursive_character"]["chunk_overlap_token_size"] == 42
    assert (
        chunk_opts["semantic_vector"]["breakpoint_threshold_type"]
        == "interquartile"
    )
    assert chunk_opts["semantic_vector"]["buffer_size"] == 3
    assert chunk_opts["fixed_token"]["split_by_character"] == "\n\n"
    assert chunk_opts["fixed_token"]["split_by_character_only"] is True


@pytest.mark.offline
def test_caller_supplied_chunk_options_reach_chunker(tmp_path, monkeypatch):
    """A caller-supplied ``chunk_options`` dict is persisted verbatim
    and the dispatcher splats it into the chunker call."""
    pytest.importorskip("langchain_text_splitters")

    import lightrag.chunker as chunker_pkg

    custom_options = {
        "chunk_token_size": 100,
        "fixed_token": {
            "chunk_overlap_token_size": 5,
            "split_by_character": None,
            "split_by_character_only": False,
        },
        "recursive_character": {
            "chunk_overlap_token_size": 0,
            "separators": ["|", ""],
        },
        "semantic_vector": {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": None,
            "buffer_size": 1,
        },
        "paragraph_semantic": {},
    }

    captured: dict = {}

    def _r_spy(tokenizer, content, chunk_token_size, **kwargs):
        captured["chunk_token_size"] = chunk_token_size
        captured["kwargs"] = dict(kwargs)
        return [
            {"tokens": 5, "content": "stub", "chunk_order_index": 0},
        ]

    monkeypatch.setattr(chunker_pkg, "chunking_by_recursive_character", _r_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "alpha|beta|gamma|delta",
                file_paths="caller.[native-R].txt",
                track_id="track-caller",
                process_options="R",
                chunk_options=custom_options,
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())

    assert captured.get("chunk_token_size") == 100, (
        f"R chunker must receive caller-supplied chunk_token_size; got {captured!r}"
    )
    assert captured["kwargs"]["separators"] == ["|", ""]
    assert captured["kwargs"]["chunk_overlap_token_size"] == 0


@pytest.mark.offline
def test_per_file_chunk_options_list(tmp_path, monkeypatch):
    """A ``chunk_options`` list aligned with ``input`` writes
    independent snapshots per doc."""

    opts_a = {
        "chunk_token_size": 1200,
        "fixed_token": {
            "chunk_overlap_token_size": 100,
            "split_by_character": None,
            "split_by_character_only": False,
        },
        "recursive_character": {
            "chunk_overlap_token_size": 100,
            "separators": ["A_SEP"],
        },
        "semantic_vector": {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": None,
            "buffer_size": 1,
        },
        "paragraph_semantic": {},
    }
    opts_b = {
        "chunk_token_size": 1200,
        "fixed_token": {
            "chunk_overlap_token_size": 100,
            "split_by_character": None,
            "split_by_character_only": False,
        },
        "recursive_character": {
            "chunk_overlap_token_size": 100,
            "separators": ["B_SEP"],
        },
        "semantic_vector": {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": None,
            "buffer_size": 1,
        },
        "paragraph_semantic": {},
    }

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                ["doc one body", "doc two body"],
                ids=["doc-aaaaa-list", "doc-bbbbb-list"],
                file_paths=["a.txt", "b.txt"],
                track_id="track-list",
                chunk_options=[opts_a, opts_b],
            )
            row_a = await rag.full_docs.get_by_id("doc-aaaaa-list")
            row_b = await rag.full_docs.get_by_id("doc-bbbbb-list")
        finally:
            await rag.finalize_storages()
        return row_a, row_b

    row_a, row_b = asyncio.run(_run())
    assert row_a is not None and row_b is not None

    sep_a = row_a["chunk_options"]["recursive_character"]["separators"]
    sep_b = row_b["chunk_options"]["recursive_character"]["separators"]
    assert sep_a == ["A_SEP"]
    assert sep_b == ["B_SEP"]

    # Independence: mutating one snapshot must not bleed into the other.
    sep_a.append("MUT")
    assert "MUT" not in row_b["chunk_options"]["recursive_character"]["separators"]


@pytest.mark.offline
def test_runtime_addon_params_mutation_affects_subsequent_enqueue(tmp_path):
    """Mutating ``rag.addon_params['chunker']`` after construction must
    take effect for documents enqueued *after* the mutation, while
    documents enqueued *before* keep their frozen snapshot.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Doc A enqueued under default config.
            await rag.apipeline_enqueue_documents(
                "first doc body",
                ids=["doc-pre-mutation"],
                file_paths=["pre.txt"],
                track_id="track-pre",
            )
            row_pre = await rag.full_docs.get_by_id("doc-pre-mutation")
            sep_pre = list(
                row_pre["chunk_options"]["recursive_character"]["separators"]
            )

            # Mutate the runtime defaults.
            rag.addon_params["chunker"]["recursive_character"]["separators"] = [
                "##",
                "\n",
            ]

            # Doc B enqueued under the mutated defaults.
            await rag.apipeline_enqueue_documents(
                "second doc body",
                ids=["doc-post-mutation"],
                file_paths=["post.txt"],
                track_id="track-post",
            )
            row_post = await rag.full_docs.get_by_id("doc-post-mutation")
        finally:
            await rag.finalize_storages()
        return sep_pre, row_post

    sep_pre, row_post = asyncio.run(_run())

    # Pre-mutation doc keeps the original LangChain default cascade.
    assert sep_pre == ["\n\n", "\n", " ", ""]
    # Post-mutation doc reflects the runtime change.
    assert (
        row_post["chunk_options"]["recursive_character"]["separators"]
        == ["##", "\n"]
    )
