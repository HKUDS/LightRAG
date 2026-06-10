"""Offline tests for the parse-stage FAILED doc_status field mapping.

PR #3235 deferred legacy extraction to the parse worker, so extraction
errors stopped producing the enqueue-time error documents that carried a
human-readable ``content_summary`` and ``metadata.error_type`` for the
WebUI. ``doc_status_parse_failure_fields`` restores those fields at the
parse worker's FAILED upsert — uniformly for every engine (legacy /
external / native / third-party):

* ``content_summary`` gets a ``[File Extraction]``-prefixed description,
  but only when the document has none (pending_parse docs enqueue with an
  empty summary; raw passthrough docs keep their real one);
* ``metadata.error_type`` keeps the legacy ``file_extraction_error``
  value, ``metadata.error_stage`` distinguishes the parse stage, and
  ``metadata.parse_engine`` records the engine that failed;
* none of the error keys is in the carry-over / directive whitelists, so
  a successful retry drops them automatically (mirroring ``error_msg``).
"""

from __future__ import annotations

import asyncio
import sys
import types
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.parser import registry
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id
from lightrag.utils_pipeline import doc_status_parse_failure_fields

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Unit tests: doc_status_parse_failure_fields
# ---------------------------------------------------------------------------


def test_empty_summary_gets_error_description():
    extra, meta = doc_status_parse_failure_fields(
        ValueError("boom"), status_doc={"content_summary": "", "metadata": {}}
    )
    assert extra["error_msg"] == "boom"
    assert extra["content_summary"] == "[File Extraction]boom"
    assert meta["error_type"] == "file_extraction_error"
    assert meta["error_stage"] == "parse"


def test_existing_summary_is_not_overwritten():
    extra, _meta = doc_status_parse_failure_fields(
        ValueError("boom"),
        status_doc={"content_summary": "real document summary", "metadata": {}},
    )
    assert "content_summary" not in extra
    assert extra["error_msg"] == "boom"


def test_whitespace_summary_counts_as_empty():
    extra, _meta = doc_status_parse_failure_fields(
        ValueError("boom"), status_doc={"content_summary": "   \n", "metadata": {}}
    )
    assert extra["content_summary"] == "[File Extraction]boom"


def test_engine_hint_fills_missing_parse_engine():
    _extra, meta = doc_status_parse_failure_fields(
        ValueError("boom"),
        status_doc={"content_summary": "", "metadata": {}},
        engine_hint="mineru",
    )
    assert meta["parse_engine"] == "mineru"


def test_engine_hint_does_not_override_existing_parse_engine():
    _extra, meta = doc_status_parse_failure_fields(
        ValueError("boom"),
        status_doc={"content_summary": "", "metadata": {"parse_engine": "docling"}},
        engine_hint="native",
    )
    # The stamped value rides the carry-over whitelist instead.
    assert "parse_engine" not in meta


def test_long_error_text_is_truncated_in_summary():
    long_error = "x" * 1000
    extra, _meta = doc_status_parse_failure_fields(
        ValueError(long_error), status_doc={"content_summary": "", "metadata": {}}
    )
    summary = extra["content_summary"]
    assert summary.startswith("[File Extraction]")
    # get_content_summary caps the description at 250 chars (plus ellipsis).
    assert len(summary) <= len("[File Extraction]") + 253
    # error_msg keeps the full text.
    assert extra["error_msg"] == long_error


# ---------------------------------------------------------------------------
# End-to-end: parse worker FAILED upsert carries the mapped fields
# ---------------------------------------------------------------------------


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _deterministic_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [
        {"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0},
    ]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"parsefail-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


@pytest.mark.asyncio
async def test_legacy_parse_failure_maps_ui_fields_and_retry_clears_them(
    tmp_path, monkeypatch
):
    """A legacy worker-stage extraction failure surfaces the pre-deferral UI
    fields; a successful retry replaces/clears all of them."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    rag = await _build_rag(tmp_path)
    try:
        source_path = input_dir / "missing.txt"  # deliberately not created
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="legacy",
            process_options="i",
        )
        doc_id = compute_mdhash_id("missing.txt", prefix="doc-")

        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        assert "legacy source file not found" in status["error_msg"]
        assert status["content_summary"].startswith("[File Extraction]")
        assert "legacy source file not found" in status["content_summary"]
        metadata = status["metadata"]
        assert metadata["error_type"] == "file_extraction_error"
        assert metadata["error_stage"] == "parse"
        assert metadata["parse_engine"] == "legacy"
        # Carry-over directives survive the FAILED upsert.
        assert metadata["process_options"] == "i"

        # Retry: the source file now exists -> parse succeeds and every
        # error field is replaced/cleared (error keys are not carried over).
        source_path.write_text("recovered body text for retry")
        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.PROCESSED
        assert not status.get("error_msg")
        assert status["content_summary"] == "recovered body text for retry"
        metadata = status["metadata"]
        assert "error_type" not in metadata
        assert "error_stage" not in metadata
        assert metadata["parse_engine"] == "legacy"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_third_party_engine_failure_records_engine_name(tmp_path, monkeypatch):
    """Any registered engine whose parse() raises gets the same field mapping,
    with metadata.parse_engine attributing the failing engine."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    class _BoomParser:
        engine_name = "boomengine"

        async def parse(self, ctx):
            raise RuntimeError("boom engine exploded")

    fake_mod = types.ModuleType("_test_boom_parser_mod")
    fake_mod.BoomParser = _BoomParser
    monkeypatch.setitem(sys.modules, "_test_boom_parser_mod", fake_mod)
    registry.register_parser(
        registry.ParserSpec(
            engine_name="boomengine",
            impl="_test_boom_parser_mod:BoomParser",
            suffixes=frozenset({"txt"}),
            queue_group="native",
        )
    )
    rag = await _build_rag(tmp_path)
    try:
        source_path = input_dir / "doc.txt"
        source_path.write_text("some body")
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="boomengine",
        )
        doc_id = compute_mdhash_id("doc.txt", prefix="doc-")

        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        assert status["error_msg"] == "boom engine exploded"
        assert status["content_summary"] == "[File Extraction]boom engine exploded"
        metadata = status["metadata"]
        assert metadata["error_type"] == "file_extraction_error"
        assert metadata["error_stage"] == "parse"
        assert metadata["parse_engine"] == "boomengine"
    finally:
        registry._REGISTRY.pop("boomengine", None)
        registry._INSTANCE_CACHE.pop(
            ("boomengine", "_test_boom_parser_mod:BoomParser"), None
        )
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_raw_document_failure_keeps_real_summary(tmp_path, monkeypatch):
    """A raw passthrough document already has a real content_summary at
    enqueue; a parse-stage failure must not overwrite it."""
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            "raw body that becomes the summary", file_paths="raw.txt"
        )
        doc_id = compute_mdhash_id("raw.txt", prefix="doc-")

        # Fail inside the parse worker (full_docs read at the worker stage);
        # bypass the pre-enqueue consistency read so the worker is the first
        # to hit the failing get_by_id.
        async def _passthrough_validate(docs, pipeline_status, pipeline_status_lock):
            return docs

        monkeypatch.setattr(
            rag, "_validate_and_fix_document_consistency", _passthrough_validate
        )
        orchestrator_read_done = {"flag": False}
        orig_get = rag.full_docs.get_by_id

        async def _flaky_get(d_id):
            if d_id == doc_id and orchestrator_read_done["flag"]:
                raise ConnectionError("storage briefly down")
            if d_id == doc_id:
                orchestrator_read_done["flag"] = True
            return await orig_get(d_id)

        monkeypatch.setattr(rag.full_docs, "get_by_id", _flaky_get)

        await rag.apipeline_process_enqueue_documents()
        await asyncio.sleep(0)

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        # The enqueue-time summary of the raw body is preserved.
        assert status["content_summary"] == "raw body that becomes the summary"
        assert "storage briefly down" in status["error_msg"]
        assert status["metadata"]["error_type"] == "file_extraction_error"
        assert status["metadata"]["error_stage"] == "parse"
    finally:
        await rag.finalize_storages()
