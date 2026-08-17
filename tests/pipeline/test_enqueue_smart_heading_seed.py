"""Direct-enqueue seeding of the ``DOCX_SMART_HEADING`` global default.

``apipeline_enqueue_documents(docs_format="pending_parse", parse_engine=...)``
bypasses upload-time ``resolve_parser_directives``, so ``_parse_engine_at``
must materialize the seed itself (via ``seed_smart_heading_param``): a bare
``native`` on a .docx persists as ``native(smart_heading=true)`` when the
switch is on, while an explicit ``native(smart_heading=false)`` stays the
opt-out and non-docx files never carry the seed.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE, FULL_DOCS_FORMAT_RAW
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt, **kwargs):
    return "ok"


def _new_rag(tmp_path: Path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"enqueue-seed-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )


def _enqueue_and_read_engines(
    tmp_path: Path,
    *,
    parse_engine: list[str],
    file_paths: list[str],
    docs_format: str = FULL_DOCS_FORMAT_PENDING_PARSE,
    input: list[str] | None = None,
) -> list[str | None]:
    """Enqueue docs and return each persisted full_docs parse_engine."""

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                input if input is not None else [""] * len(file_paths),
                docs_format=docs_format,
                parse_engine=parse_engine,
                file_paths=file_paths,
            )
            engines = []
            for name in file_paths:
                doc_id = compute_mdhash_id(name, prefix="doc-")
                row = await rag.full_docs.get_by_id(doc_id)
                assert row is not None, f"full_docs row missing for {name}"
                engines.append(row.get("parse_engine"))
            return engines
        finally:
            await rag.finalize_storages()

    return asyncio.run(_run())


def test_direct_enqueue_seeds_bare_native_docx(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    engines = _enqueue_and_read_engines(
        tmp_path,
        parse_engine=["native", "native(smart_heading=false)", "native"],
        file_paths=["seeded.docx", "optout.docx", "notes.md"],
    )
    assert engines == [
        "native(smart_heading=true)",  # bare native on .docx gets the seed
        "native(smart_heading=false)",  # explicit param stays the opt-out
        "native",  # non-docx never carries the seed
    ]


def test_direct_enqueue_raw_docx_metadata_not_seeded(tmp_path, monkeypatch):
    """RAW enqueue: parse_engine records the engine that ALREADY extracted
    the content — no docx parser will run on this doc, so the seed must not
    rewrite the metadata even for a .docx source with the switch on."""
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    engines = _enqueue_and_read_engines(
        tmp_path,
        docs_format=FULL_DOCS_FORMAT_RAW,
        input=["Already extracted body text with enough words."],
        parse_engine=["native"],
        file_paths=["extracted.docx"],
    )
    assert engines == ["native"]


def test_direct_enqueue_switch_off_keeps_bare_native(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCX_SMART_HEADING", raising=False)
    engines = _enqueue_and_read_engines(
        tmp_path,
        parse_engine=["native"],
        file_paths=["plain.docx"],
    )
    assert engines == ["native"]
