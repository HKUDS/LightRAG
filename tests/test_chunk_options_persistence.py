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
from lightrag.constants import DEFAULT_R_SEPARATORS
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
    assert chunk_opts["semantic_vector"]["breakpoint_threshold_type"] == "interquartile"
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

    assert (
        captured.get("chunk_token_size") == 100
    ), f"R chunker must receive caller-supplied chunk_token_size; got {captured!r}"
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
def test_constructor_chunk_size_overlays_addon_params(tmp_path, monkeypatch):
    """``LightRAG(chunk_token_size=N, chunk_overlap_token_size=M)`` must
    actually take effect — the per-doc snapshot is built from
    ``addon_params['chunker']``, so the constructor values have to be
    overlaid onto it (otherwise env-driven defaults would silently win).
    """
    # Set env vars to non-default values so the env path would be
    # observably different from the constructor path.
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "100")

    async def _run():
        rag = _new_rag(
            tmp_path,
            chunk_token_size=7,
            chunk_overlap_token_size=2,
        )
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for constructor overlay test.",
                ids=["doc-ctor-overlay"],
                file_paths="ctor.txt",
                track_id="track-ctor",
            )
            row = await rag.full_docs.get_by_id("doc-ctor-overlay")
        finally:
            await rag.finalize_storages()
        return row, rag.addon_params

    row, addon_params = asyncio.run(_run())
    assert row is not None
    chunk_opts = row["chunk_options"]
    # Top-level chunk_token_size carries the constructor value.
    assert chunk_opts["chunk_token_size"] == 7
    # F and R sub-dicts both pick up the legacy overlap field; V
    # doesn't have chunk_overlap_token_size and must remain unchanged.
    assert chunk_opts["fixed_token"]["chunk_overlap_token_size"] == 2
    assert chunk_opts["recursive_character"]["chunk_overlap_token_size"] == 2
    assert "chunk_overlap_token_size" not in chunk_opts["semantic_vector"]
    # addon_params reflects the same overlay so subsequent runtime
    # mutations operate on the constructor-supplied baseline.
    assert addon_params["chunker"]["chunk_token_size"] == 7
    assert addon_params["chunker"]["fixed_token"]["chunk_overlap_token_size"] == 2


@pytest.mark.offline
def test_addon_params_chunker_wins_when_constructor_field_unset(tmp_path):
    """If the constructor field is left at its default (``None``), an
    explicit ``addon_params={'chunker': {...}}`` must NOT be clobbered.
    """

    async def _run():
        rag = _new_rag(
            tmp_path,
            addon_params={
                "chunker": {
                    "chunk_token_size": 5000,
                    "fixed_token": {
                        "chunk_overlap_token_size": 250,
                        "split_by_character": None,
                        "split_by_character_only": False,
                    },
                    "recursive_character": {
                        "chunk_overlap_token_size": 250,
                        "separators": ["\n\n", "\n", " ", ""],
                    },
                    "semantic_vector": {
                        "breakpoint_threshold_type": "percentile",
                        "breakpoint_threshold_amount": None,
                        "buffer_size": 1,
                    },
                    "paragraph_semantic": {},
                },
            },
        )
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for addon-only overlay test.",
                ids=["doc-addon-only"],
                file_paths="addon.txt",
                track_id="track-addon",
            )
            row = await rag.full_docs.get_by_id("doc-addon-only")
        finally:
            await rag.finalize_storages()
        return row, rag.chunk_token_size, rag.chunk_overlap_token_size

    row, ctor_size, ctor_overlap = asyncio.run(_run())
    assert row is not None
    assert row["chunk_options"]["chunk_token_size"] == 5000
    assert row["chunk_options"]["fixed_token"]["chunk_overlap_token_size"] == 250
    # Legacy instance fields back-fill from addon_params (not env defaults).
    assert ctor_size == 5000
    assert ctor_overlap == 250


@pytest.mark.offline
def test_strategy_env_wins_over_legacy_ctor_field(tmp_path, monkeypatch):
    """Specificity-ordered precedence: strategy-specific env vars beat
    the strategy-agnostic legacy constructor field.

    Setup: ``CHUNK_R_OVERLAP_SIZE=42`` is strategy-specific for R.
    ``LightRAG(chunk_overlap_token_size=2)`` is the legacy
    strategy-agnostic field.  R must end up at 42 (env wins on its own
    strategy slot), F at 2 (no F-specific env, so legacy field fills).
    """
    monkeypatch.setenv("CHUNK_R_OVERLAP_SIZE", "42")
    monkeypatch.delenv("CHUNK_F_OVERLAP_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP_SIZE", raising=False)

    async def _run():
        rag = _new_rag(tmp_path, chunk_overlap_token_size=2)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for strategy-vs-ctor precedence test.",
                ids=["doc-strategy-vs-ctor"],
                file_paths="prec.txt",
                track_id="track-prec",
            )
            row = await rag.full_docs.get_by_id("doc-strategy-vs-ctor")
        finally:
            await rag.finalize_storages()
        return row, rag.chunk_overlap_token_size

    row, ctor_field = asyncio.run(_run())
    assert row is not None
    chunk_opts = row["chunk_options"]
    assert chunk_opts["recursive_character"]["chunk_overlap_token_size"] == 42, (
        "Strategy-specific CHUNK_R_OVERLAP_SIZE must win over the "
        "legacy chunk_overlap_token_size constructor field."
    )
    assert chunk_opts["fixed_token"]["chunk_overlap_token_size"] == 2, (
        "Without a CHUNK_F_OVERLAP_SIZE override, the F slot falls back "
        "to the legacy constructor field."
    )
    # self.chunk_overlap_token_size mirrors the F-strategy resolved value.
    assert ctor_field == 2


@pytest.mark.offline
def test_legacy_env_is_final_fallback(tmp_path, monkeypatch):
    """When neither a strategy env nor the legacy ctor field is set,
    the legacy ``CHUNK_OVERLAP_SIZE`` env is the final fallback for
    every per-strategy overlap slot."""
    monkeypatch.delenv("CHUNK_F_OVERLAP_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_R_OVERLAP_SIZE", raising=False)
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "77")

    async def _run():
        rag = _new_rag(tmp_path)  # no chunk_overlap_token_size kwarg
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for legacy-env fallback test.",
                ids=["doc-legacy-env"],
                file_paths="legacy.txt",
                track_id="track-legacy",
            )
            row = await rag.full_docs.get_by_id("doc-legacy-env")
        finally:
            await rag.finalize_storages()
        return row, rag.chunk_overlap_token_size

    row, ctor_field = asyncio.run(_run())
    chunk_opts = row["chunk_options"]
    assert chunk_opts["fixed_token"]["chunk_overlap_token_size"] == 77
    assert chunk_opts["recursive_character"]["chunk_overlap_token_size"] == 77
    assert ctor_field == 77

    # Mixed case: F-specific env set, legacy still acts as R's fallback.
    monkeypatch.setenv("CHUNK_F_OVERLAP_SIZE", "10")

    async def _run_mixed():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for mixed-env fallback test.",
                ids=["doc-mixed-env"],
                file_paths="mixed.txt",
                track_id="track-mixed",
            )
            row = await rag.full_docs.get_by_id("doc-mixed-env")
        finally:
            await rag.finalize_storages()
        return row

    row = asyncio.run(_run_mixed())
    chunk_opts = row["chunk_options"]
    assert chunk_opts["fixed_token"]["chunk_overlap_token_size"] == 10
    assert chunk_opts["recursive_character"]["chunk_overlap_token_size"] == 77


@pytest.mark.offline
def test_p_strategy_uses_dedicated_chunk_size_env(tmp_path, monkeypatch):
    """``CHUNK_P_SIZE`` must give P its own ``chunk_token_size``,
    decoupled from the global ``CHUNK_SIZE`` shared by F/R/V."""
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_P_SIZE", "999")

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    def _p_spy(tokenizer, content, chunk_token_size, *, blocks_path=None, **kwargs):
        captured["chunk_token_size"] = chunk_token_size
        captured["blocks_path"] = blocks_path
        captured["kwargs"] = dict(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_paragraph_semantic", _p_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "stand-in body for paragraph-semantic chunker",
                file_paths="ctor.[native-P].txt",
                track_id="track-p-size",
                process_options="P",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == 999, (
        "P chunker must receive CHUNK_P_SIZE-derived chunk_token_size, "
        f"not the global CHUNK_SIZE; got {captured!r}"
    )
    # And the dispatcher must not double-pass chunk_token_size as kwarg.
    assert "chunk_token_size" not in captured["kwargs"]


@pytest.mark.offline
def test_p_strategy_falls_back_to_global_chunk_size(tmp_path, monkeypatch):
    """When ``CHUNK_P_SIZE`` is unset and no per-doc P override is
    supplied, P inherits the top-level ``chunk_token_size`` resolved
    from the standard chain (here: ``LightRAG(chunk_token_size=…)``)."""
    monkeypatch.delenv("CHUNK_P_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    def _p_spy(tokenizer, content, chunk_token_size, *, blocks_path=None, **kwargs):
        captured["chunk_token_size"] = chunk_token_size
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_paragraph_semantic", _p_spy)

    async def _run():
        rag = _new_rag(tmp_path, chunk_token_size=333)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "fallback body",
                file_paths="ctor.[native-P].txt",
                track_id="track-p-fallback",
                process_options="P",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == 333


@pytest.mark.offline
def test_addon_params_strategy_wins_over_strategy_env(tmp_path, monkeypatch):
    """Highest tier check: a value sitting in
    ``addon_params['chunker'][<strategy>]['chunk_overlap_token_size']``
    must beat even a strategy-specific env."""
    monkeypatch.setenv("CHUNK_R_OVERLAP_SIZE", "42")

    async def _run():
        rag = _new_rag(
            tmp_path,
            addon_params={
                "chunker": {
                    "recursive_character": {
                        "chunk_overlap_token_size": 999,
                        "separators": ["\n\n", "\n", " ", ""],
                    },
                },
            },
        )
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for addon-vs-env precedence test.",
                ids=["doc-addon-vs-env"],
                file_paths="addon.txt",
                track_id="track-addon",
            )
            row = await rag.full_docs.get_by_id("doc-addon-vs-env")
        finally:
            await rag.finalize_storages()
        return row

    row = asyncio.run(_run())
    chunk_opts = row["chunk_options"]
    assert (
        chunk_opts["recursive_character"]["chunk_overlap_token_size"] == 999
    ), "addon_params explicit value must beat strategy-specific env."


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

    # Pre-mutation doc keeps the env-driven default cascade.
    assert sep_pre == list(DEFAULT_R_SEPARATORS)
    # Post-mutation doc reflects the runtime change.
    assert row_post["chunk_options"]["recursive_character"]["separators"] == [
        "##",
        "\n",
    ]


@pytest.mark.offline
def test_r_strategy_uses_dedicated_chunk_size_env(tmp_path, monkeypatch):
    """``CHUNK_R_SIZE`` must give R its own ``chunk_token_size``,
    decoupled from the global ``CHUNK_SIZE`` shared by F/V."""
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_R_SIZE", "777")

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    def _r_spy(tokenizer, content, chunk_token_size, **kwargs):
        captured["chunk_token_size"] = chunk_token_size
        captured["kwargs"] = dict(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_recursive_character", _r_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "stand-in body for recursive-character chunker",
                file_paths="ctor.[native-R].txt",
                track_id="track-r-size",
                process_options="R",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == 777, (
        "R chunker must receive CHUNK_R_SIZE-derived chunk_token_size, "
        f"not the global CHUNK_SIZE; got {captured!r}"
    )
    # Dispatcher must not double-pass chunk_token_size as kwarg.
    assert "chunk_token_size" not in captured["kwargs"]


@pytest.mark.offline
def test_r_strategy_falls_back_to_global_chunk_size(tmp_path, monkeypatch):
    """When ``CHUNK_R_SIZE`` is unset and no per-doc R override is
    supplied, R inherits the top-level ``chunk_token_size`` resolved
    from the standard chain (here: ``LightRAG(chunk_token_size=…)``)."""
    monkeypatch.delenv("CHUNK_R_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    def _r_spy(tokenizer, content, chunk_token_size, **kwargs):
        captured["chunk_token_size"] = chunk_token_size
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_recursive_character", _r_spy)

    async def _run():
        rag = _new_rag(tmp_path, chunk_token_size=444)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "fallback body",
                file_paths="ctor.[native-R].txt",
                track_id="track-r-fallback",
                process_options="R",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == 444


@pytest.mark.offline
def test_v_strategy_uses_dedicated_chunk_size_env(tmp_path, monkeypatch):
    """``CHUNK_V_SIZE`` must give V its own ``chunk_token_size`` advisory
    ceiling, decoupled from the global ``CHUNK_SIZE`` shared by F/R."""
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_V_SIZE", "2500")

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    async def _v_spy(
        tokenizer, content, chunk_token_size, *, embedding_func=None, **kwargs
    ):
        captured["chunk_token_size"] = chunk_token_size
        captured["kwargs"] = dict(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_semantic_vector", _v_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "stand-in body for semantic-vector chunker",
                file_paths="ctor.[native-V].txt",
                track_id="track-v-size",
                process_options="V",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == 2500, (
        "V chunker must receive CHUNK_V_SIZE-derived chunk_token_size, "
        f"not the global CHUNK_SIZE; got {captured!r}"
    )
    # Dispatcher must not double-pass chunk_token_size as kwarg.
    assert "chunk_token_size" not in captured["kwargs"]


@pytest.mark.offline
def test_v_strategy_falls_back_to_global_chunk_size(tmp_path, monkeypatch):
    """When ``CHUNK_V_SIZE`` is unset and no per-doc V override is
    supplied, V inherits the top-level ``chunk_token_size`` resolved
    from the standard chain (here: ``LightRAG(chunk_token_size=…)``)."""
    monkeypatch.delenv("CHUNK_V_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    async def _v_spy(
        tokenizer, content, chunk_token_size, *, embedding_func=None, **kwargs
    ):
        captured["chunk_token_size"] = chunk_token_size
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_semantic_vector", _v_spy)

    async def _run():
        rag = _new_rag(tmp_path, chunk_token_size=555)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "fallback body",
                file_paths="ctor.[native-V].txt",
                track_id="track-v-fallback",
                process_options="V",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == 555
