"""End-to-end: the smart-heading parse_warnings split survives the full parse
pipeline, not just the parser boundary.

The parse-stage split (smart_/title_block_ → sidecar smart_audit.json, the rest
→ ParseResult.parse_warnings) is unit-tested in
``tests/parser/docx/test_smart_heading_wiring.py``. This pins the *downstream*
half: ``_parse_worker`` mirrors ``ParseResult.parse_warnings`` onto
``status_doc.metadata`` and ``_upsert_doc_status_transition`` persists it
(pipeline.py), so the final ``doc_status.metadata["parse_warnings"]`` must carry
only the non-smart warnings while the smart ones live in the audit file.
"""

from __future__ import annotations

import asyncio
import json
from unittest import mock
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


_STUB_BLOCKS = [
    {
        "uuid": "p1",
        "heading": "Chapter One",
        "content": "# Chapter One\nBody paragraph with enough words to look real.",
        "type": "text",
        "parent_headings": [],
        "level": 1,
    }
]


def _stub_extract_mixed(
    file_path,
    *,
    drawing_context=None,
    parse_warnings=None,
    parse_metadata=None,
    **_kwargs,
):
    """extract_docx_blocks stand-in seeding a mixed warning set + audit ledger:
    two smart-heading keys (one ``smart_``, one ``title_block_``) and one
    non-smart key."""
    if parse_warnings is not None:
        parse_warnings.update(
            {
                "smart_cb1_tripped": 2,
                "title_block_empty_members": 1,
                "missing_paraid_count": 4,
            }
        )
    if parse_metadata is not None:
        parse_metadata["smart_audit"] = {"shadow_diff": {}}
    return [dict(b) for b in _STUB_BLOCKS]


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
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"pwsplit-{uuid4().hex[:8]}",
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


def test_pipeline_diverts_smart_warnings_keeps_nonsmart_on_doc_status(
    tmp_path, monkeypatch
):
    """A PENDING_PARSE .docx through the native engine: doc_status.metadata
    keeps only the non-smart warning; the smart ones land in the audit file."""

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        rag = await _build_rag(tmp_path)
        try:
            source_path = input_dir / "doc.docx"
            source_path.write_bytes(b"fake-docx")

            with mock.patch(
                "lightrag.parser.docx.parse_document.extract_docx_blocks",
                _stub_extract_mixed,
            ):
                await rag.apipeline_enqueue_documents(
                    "",
                    file_paths=str(source_path),
                    docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                    parse_engine="native",
                )
                doc_id = compute_mdhash_id("doc.docx", prefix="doc-")
                await rag.apipeline_process_enqueue_documents()

            status = await rag.doc_status.get_by_id(doc_id)
            assert status is not None
            assert status["status"] == DocStatus.PROCESSED
            metadata = status["metadata"]
            persisted = metadata.get("parse_warnings")

            # Only the non-smart warning reaches doc_status ...
            assert persisted == {"missing_paraid_count": 4}
            # ... no smart-heading diagnostics leaked into the persisted blob.
            assert not any(
                k.startswith(("smart_", "title_block_")) for k in (persisted or {})
            )

            # ... while the smart-heading warnings are in the sidecar audit file,
            # merged with the ledger the parser left in metadata["smart_audit"].
            (audit_path,) = input_dir.glob("**/*.smart_audit.json")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            assert audit["parse_warnings"] == {
                "smart_cb1_tripped": 2,
                "title_block_empty_members": 1,
            }
            assert "shadow_diff" in audit  # ledger preserved
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
