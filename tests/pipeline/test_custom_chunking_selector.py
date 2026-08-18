"""End-to-end contract for the explicit ``C`` custom-chunking selector.

The legacy no-selector path remains backward compatible.  ``C`` is the
explicit, persisted opt-in that routes to the same six-argument callback while
making custom success and fixed-token fallback observable.
"""

import asyncio
import logging
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocStatus
from lightrag.chunker import chunking_by_token_size
from lightrag.utils import EmbeddingFunc, Tokenizer


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


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
        workspace=f"custom-selector-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _metadata(row: dict) -> dict:
    metadata = row.get("metadata")
    assert isinstance(metadata, dict)
    return metadata


async def _ingest(
    rag: LightRAG,
    *,
    doc_id: str,
    process_options: str,
    body: str = "Body for explicit custom chunking.",
    chunk_options: dict | None = None,
) -> dict:
    await rag.apipeline_enqueue_documents(
        body,
        ids=[doc_id],
        file_paths=[f"{doc_id}.txt"],
        track_id=f"track-{doc_id}",
        process_options=process_options,
        chunk_options=chunk_options,
    )
    await rag.apipeline_process_enqueue_documents()
    row = await rag.doc_status.get_by_id(doc_id)
    assert isinstance(row, dict)
    return row


def _force_blocks_path(monkeypatch, blocks_path: Path) -> None:
    """Inject a process-stage sidecar path without standing up a parser."""
    import lightrag.pipeline as pipeline_mod

    original = pipeline_mod._PipelineMixin.process_single_document

    async def _patched(self, *, doc_id, status_doc, parsed_data, ctx):
        parsed_data["blocks_path"] = str(blocks_path)
        return await original(
            self,
            doc_id=doc_id,
            status_doc=status_doc,
            parsed_data=parsed_data,
            ctx=ctx,
        )

    monkeypatch.setattr(
        pipeline_mod._PipelineMixin,
        "process_single_document",
        _patched,
    )


@pytest.mark.offline
def test_c_invokes_sync_custom_chunker_with_legacy_args_and_no_bypass_warning(
    tmp_path,
):
    captured: dict = {"calls": 0}

    def _custom(tokenizer, content, split_by, split_only, overlap, size):
        captured.update(
            {
                "calls": captured["calls"] + 1,
                "loop": asyncio.get_running_loop(),
                "content": content,
                "split_by": split_by,
                "split_only": split_only,
                "overlap": overlap,
                "size": size,
            }
        )
        return [{"tokens": len(content), "content": content, "chunk_order_index": 0}]

    async def _run():
        rag = _new_rag(tmp_path, chunking_func=_custom)
        await rag.initialize_storages()
        handler = _ListHandler()
        logger = logging.getLogger("lightrag")
        logger.addHandler(handler)
        try:
            row = await _ingest(
                rag,
                doc_id="doc-custom-sync",
                process_options="C!",
                chunk_options={
                    "chunk_token_size": 1200,
                    "fixed_token": {
                        "chunk_token_size": 256,
                        "split_by_character": "\n",
                        "split_by_character_only": True,
                        "chunk_overlap_token_size": 7,
                    },
                },
            )
            full_doc = await rag.full_docs.get_by_id("doc-custom-sync")
        finally:
            logger.removeHandler(handler)
            await rag.finalize_storages()

        assert DocStatus(row["status"]) is DocStatus.PROCESSED
        assert captured == {
            "calls": 1,
            "loop": asyncio.get_running_loop(),
            "content": "Body for explicit custom chunking.",
            "split_by": "\n",
            "split_only": True,
            "overlap": 7,
            "size": 256,
        }
        assert isinstance(full_doc, dict)
        assert full_doc["process_options"] == "C!"
        assert _metadata(row)["chunk_method"] == "custom_chunking_func"
        assert not any(
            "Custom chunking_func bypassed" in record.getMessage()
            for record in handler.records
        )

    asyncio.run(_run())


@pytest.mark.offline
def test_c_awaits_async_custom_chunker_on_the_event_loop(tmp_path):
    captured: dict = {"calls": 0}

    async def _custom(tokenizer, content, split_by, split_only, overlap, size):
        captured["calls"] += 1
        captured["loop"] = asyncio.get_running_loop()
        await asyncio.sleep(0)
        return [{"tokens": len(content), "content": content, "chunk_order_index": 0}]

    async def _run():
        rag = _new_rag(tmp_path, chunking_func=_custom)
        await rag.initialize_storages()
        try:
            running_loop = asyncio.get_running_loop()
            row = await _ingest(
                rag,
                doc_id="doc-custom-async",
                process_options="C!",
            )
        finally:
            await rag.finalize_storages()

        assert DocStatus(row["status"]) is DocStatus.PROCESSED
        assert captured == {"calls": 1, "loop": running_loop}
        assert _metadata(row)["chunk_method"] == "custom_chunking_func"

    asyncio.run(_run())


@pytest.mark.offline
def test_c_without_custom_chunker_warns_once_and_runs_exact_f_fallback(
    tmp_path, monkeypatch
):
    import lightrag.chunker as chunker_pkg
    import lightrag.sidecar as sidecar_mod

    fixed_calls: list[dict] = []
    backfill_calls: list[str] = []

    def _fixed(tokenizer, content, size, **kwargs):
        fixed_calls.append({"content": content, "size": size, **kwargs})
        return [
            {
                "tokens": len(content),
                "content": content,
                "chunk_order_index": 0,
                "_source_span": {"start": 0, "end": len(content)},
            }
        ]

    def _backfill(chunking_result, blocks_path):
        backfill_calls.append(blocks_path)

    monkeypatch.setattr(chunker_pkg, "chunking_by_fixed_token", _fixed)
    monkeypatch.setattr(sidecar_mod, "backfill_chunk_sidecars", _backfill)
    blocks_path = tmp_path / "fallback.blocks.jsonl"
    _force_blocks_path(monkeypatch, blocks_path)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        handler = _ListHandler()
        logger = logging.getLogger("lightrag")
        logger.addHandler(handler)
        try:
            row = await _ingest(
                rag,
                doc_id="doc-custom-fallback",
                process_options="C!",
            )
        finally:
            logger.removeHandler(handler)
            await rag.finalize_storages()

        warnings = [
            record.getMessage()
            for record in handler.records
            if record.levelno == logging.WARNING
            and "Custom chunking_func unavailable for selector C" in record.getMessage()
        ]
        assert len(warnings) == 1
        assert "doc-custom-fallback" in warnings[0]
        assert len(fixed_calls) == 1
        assert fixed_calls[0]["_emit_source_span"] is True
        assert backfill_calls == [str(blocks_path)]
        assert DocStatus(row["status"]) is DocStatus.PROCESSED
        method = _metadata(row)["chunk_method"]
        assert method == "custom_chunking_fallback_fixed_token"
        assert method != "fixed_token_fallback"

    asyncio.run(_run())


@pytest.mark.offline
def test_c_callback_failure_fails_document_without_fallback(tmp_path, monkeypatch):
    import lightrag.chunker as chunker_pkg

    fixed_calls = 0

    def _fixed(*args, **kwargs):
        nonlocal fixed_calls
        fixed_calls += 1
        return []

    def _custom(*args, **kwargs):
        raise RuntimeError("custom chunker exploded")

    monkeypatch.setattr(chunker_pkg, "chunking_by_fixed_token", _fixed)

    async def _run():
        rag = _new_rag(tmp_path, chunking_func=_custom)
        await rag.initialize_storages()
        try:
            row = await _ingest(
                rag,
                doc_id="doc-custom-failure",
                process_options="C!",
            )
        finally:
            await rag.finalize_storages()

        assert DocStatus(row["status"]) is DocStatus.FAILED
        assert "C custom chunking_func failed" in row["error_msg"]
        assert "custom chunker exploded" in row["error_msg"]
        assert fixed_calls == 0

    asyncio.run(_run())


@pytest.mark.offline
def test_c_combines_with_multimodal_flags_without_enabling_backfill(
    tmp_path, monkeypatch
):
    import lightrag.pipeline as pipeline_mod
    import lightrag.sidecar as sidecar_mod

    captured: dict = {"custom_calls": 0, "builder_options": None}

    def _custom(tokenizer, content, split_by, split_only, overlap, size):
        captured["custom_calls"] += 1
        return [{"tokens": len(content), "content": content, "chunk_order_index": 0}]

    def _mm_builder(self, **kwargs):
        captured["builder_options"] = kwargs["process_options"]
        base = kwargs["base_order_index"]
        return [
            {
                "tokens": 1,
                "content": f"modality-{index}",
                "chunk_order_index": base + index,
            }
            for index in range(3)
        ]

    def _unexpected_backfill(*args, **kwargs):
        raise AssertionError("custom callback output must not be sidecar-backfilled")

    monkeypatch.setattr(
        pipeline_mod._PipelineMixin,
        "_build_mm_chunks_from_sidecars",
        _mm_builder,
    )
    monkeypatch.setattr(sidecar_mod, "backfill_chunk_sidecars", _unexpected_backfill)
    _force_blocks_path(monkeypatch, tmp_path / "multimodal.blocks.jsonl")

    async def _run():
        rag = _new_rag(tmp_path, chunking_func=_custom)
        await rag.initialize_storages()
        try:
            row = await _ingest(
                rag,
                doc_id="doc-custom-multimodal",
                process_options="Cite!",
            )
        finally:
            await rag.finalize_storages()

        assert DocStatus(row["status"]) is DocStatus.PROCESSED
        assert captured == {"custom_calls": 1, "builder_options": "Cite!"}
        assert row["chunks_count"] == 4
        assert _metadata(row)["mm_chunks"] == 3
        assert _metadata(row)["chunk_method"] == "custom_chunking_func"

    asyncio.run(_run())


@pytest.mark.offline
def test_reprocess_persisted_c_after_callback_removal_uses_observable_fallback(
    tmp_path,
):
    custom_calls = 0

    def _custom(tokenizer, content, split_by, split_only, overlap, size):
        nonlocal custom_calls
        custom_calls += 1
        return [{"tokens": len(content), "content": content, "chunk_order_index": 0}]

    async def _run():
        rag = _new_rag(tmp_path, chunking_func=_custom)
        await rag.initialize_storages()
        try:
            first = await _ingest(
                rag,
                doc_id="doc-custom-reprocess",
                process_options="C!",
            )
            assert _metadata(first)["chunk_method"] == "custom_chunking_func"

            persisted = await rag.full_docs.get_by_id("doc-custom-reprocess")
            assert isinstance(persisted, dict)
            assert persisted["process_options"] == "C!"

            rag.chunking_func = chunking_by_token_size
            pending = dict(first)
            pending["status"] = DocStatus.PENDING.value
            await rag.doc_status.upsert({"doc-custom-reprocess": pending})
            await rag.doc_status.index_done_callback()

            handler = _ListHandler()
            logger = logging.getLogger("lightrag")
            logger.addHandler(handler)
            try:
                await rag.apipeline_process_enqueue_documents()
            finally:
                logger.removeHandler(handler)

            second = await rag.doc_status.get_by_id("doc-custom-reprocess")
            assert isinstance(second, dict)
            warnings = [
                record.getMessage()
                for record in handler.records
                if "Custom chunking_func unavailable for selector C"
                in record.getMessage()
            ]
        finally:
            await rag.finalize_storages()

        assert custom_calls == 1
        assert DocStatus(second["status"]) is DocStatus.PROCESSED
        assert _metadata(second)["chunk_method"] == (
            "custom_chunking_fallback_fixed_token"
        )
        assert len(warnings) == 1

    asyncio.run(_run())
