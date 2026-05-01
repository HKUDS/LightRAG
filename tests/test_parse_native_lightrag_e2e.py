"""End-to-end test: native docx → LightRAG Document → stable cache key.

The original bug this guards against: ``parse_native`` used to write the
full interchange JSONL string (with a runtime ``parsed_at`` field) into
``full_docs.content``, so re-parsing the same docx produced different
chunk-0 content and therefore different LLM cache keys.

After the fix, ``parse_native`` writes ``.blocks.jsonl`` + sidecars and
``full_docs`` is in LIGHTRAG format. ``_load_lightrag_document_content``
skips the ``meta`` line (which contains ``parsed_time``) and concatenates
only ``"type": "content"`` rows, so re-parsing must yield byte-identical
``merged_text`` and stable downstream chunk-0 content.
"""

import asyncio
import importlib

import pytest

from lightrag import LightRAG
from lightrag.constants import (
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_DIR_NAME,
)
from lightrag.utils import compute_args_hash


class _MiniFullDocs:
    def __init__(self):
        self.data = {}

    async def upsert(self, payload):
        self.data.update(payload)

    async def get_by_id(self, doc_id):
        return self.data.get(doc_id)

    async def index_done_callback(self):
        return None


class _MiniDocStatus:
    async def get_by_id(self, doc_id):
        return None

    async def upsert(self, data):
        return None


class _MiniRag:
    """Just enough surface for parse_native + _write_lightrag_document_*."""

    _archive_docx_source_after_full_docs_sync = (
        LightRAG._archive_docx_source_after_full_docs_sync
    )
    _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs
    _input_dir_path = LightRAG._input_dir_path
    _parsed_dir_for_source = LightRAG._parsed_dir_for_source
    _parsed_artifact_dir_for_source = LightRAG._parsed_artifact_dir_for_source
    _resolve_lightrag_document_path = LightRAG._resolve_lightrag_document_path
    _write_lightrag_document_from_content_list = (
        LightRAG._write_lightrag_document_from_content_list
    )
    _load_lightrag_document_content = LightRAG._load_lightrag_document_content

    def __init__(self, working_dir):
        self.working_dir = str(working_dir)
        self.full_docs = _MiniFullDocs()
        self.doc_status = _MiniDocStatus()

    def _resolve_source_file_for_parser(self, file_path):
        return file_path


@pytest.mark.offline
def test_native_lightrag_path_produces_stable_merged_text(tmp_path, monkeypatch):
    """Re-parsing the same docx must yield byte-identical merged_text and
    therefore identical chunk_args_hash on chunk-0."""

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "stable.docx"
        source_path.write_bytes(b"fake docx bytes")

        # Stub the docx parser so we get a deterministic content_list both
        # times — the real cache stability guarantee comes from LightRAG
        # writing a stable .blocks.jsonl content section, not from the
        # parser itself being deterministic.
        parse_document = importlib.import_module("lightrag.extraction.parse_document")
        stable_content_list = [
            {"type": "section_header", "text": "Title", "text_level": 1},
            {"type": "text", "text": "First paragraph body."},
            {"type": "text", "text": "Second paragraph body."},
        ]

        def _stub_parse(file_bytes, source_file, doc_id):
            return list(stable_content_list), {}

        monkeypatch.setattr(
            parse_document, "parse_docx_to_lightrag_content_list", _stub_parse
        )

        rag = _MiniRag(tmp_path / "work")

        # ---- First parse ----
        # parse_native archives the source after writing, so re-create it
        # before the second parse for a fair comparison.
        result1 = await LightRAG.parse_native(
            rag,
            "doc-stable",
            str(source_path),
            {"format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )
        merged1 = result1["content"]
        assert merged1, "first parse produced empty merged_text"

        # ---- Second parse ----
        # Restore the source file (archive moved it), reset the in-memory
        # full_docs row, and remove the parsed_dir so the writer rewrites
        # both meta (with a fresh parsed_time) and content lines.
        source_path.write_bytes(b"fake docx bytes")
        rag.full_docs.data.clear()
        parsed_artifact_dir = input_dir / PARSED_DIR_NAME / f"{source_path.name}.parsed"
        if parsed_artifact_dir.exists():
            import shutil

            shutil.rmtree(parsed_artifact_dir)

        result2 = await LightRAG.parse_native(
            rag,
            "doc-stable",
            str(source_path),
            {"format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )
        merged2 = result2["content"]

        # Core invariant: merged_text byte-identical across runs even
        # though parsed_time in the .blocks.jsonl meta line differs.
        assert merged1 == merged2

        # And: a hash computed over a chunk-0 derived from merged_text
        # must also be identical — that is what powers LLM cache hits.
        prompt_template = "EXTRACT_PROMPT::{text}"
        chunk0_a = prompt_template.format(text=merged1[:200])
        chunk0_b = prompt_template.format(text=merged2[:200])
        assert chunk0_a == chunk0_b
        assert compute_args_hash(chunk0_a) == compute_args_hash(chunk0_b)

        # And: full_docs.content uses the {{LRdoc}} marker plus a leading
        # summary derived from merged_text (not the legacy placeholder).
        record = rag.full_docs.data["doc-stable"]
        assert record["format"] == "lightrag"
        assert record["content"].startswith("{{LRdoc}}")
        assert merged1[:40] in record["content"]

    asyncio.run(_run())


@pytest.mark.offline
def test_native_lightrag_path_writes_blocks_jsonl_and_skips_meta_on_load(
    tmp_path, monkeypatch
):
    """Sanity check: ``_load_lightrag_document_content`` must skip the
    meta line (where the runtime ``parsed_time`` lives) and only return
    body content. This is what lets re-parsing produce stable text."""

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "skipmeta.docx"
        source_path.write_bytes(b"fake")

        parse_document = importlib.import_module("lightrag.extraction.parse_document")
        monkeypatch.setattr(
            parse_document,
            "parse_docx_to_lightrag_content_list",
            lambda *_, **__: ([{"type": "text", "text": "the body"}], {}),
        )

        rag = _MiniRag(tmp_path / "work")
        result = await LightRAG.parse_native(
            rag,
            "doc-skip",
            str(source_path),
            {"format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

        # The .blocks.jsonl on disk DOES contain "parsed_time" inside the
        # meta line; the merged_text returned by parse_native MUST NOT.
        blocks_path = result["blocks_path"]
        on_disk = open(blocks_path, "r", encoding="utf-8").read()
        assert "parsed_time" in on_disk
        assert "parsed_time" not in result["content"]
        assert result["content"].strip() == "the body"

    asyncio.run(_run())
