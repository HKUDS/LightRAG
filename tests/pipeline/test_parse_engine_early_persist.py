"""parse_engine / parse_format are stamped into doc_status at the PARSING
stage (once the engine has run), not deferred to PROCESSING.

Background: these two fields are determined by the parser the moment it
finishes (native/mineru/docling each report a fixed engine; the resulting
``parse_format`` is likewise known).  Historically they only landed in
``doc_status.metadata`` at the PROCESSING/PROCESSED transition, bundled into
``extraction_meta`` — so a document mid-flight (PARSING/ANALYZING) did not
expose which engine/format produced it.

These tests pin the new contract:

* ``resolve_doc_status_parse_engine`` — the shared resolver used by both the
  parse stage and the process stage (so the value never differs between the
  early and final writes).
* ``doc_status_metadata_carry_over`` now preserves ``parse_format`` /
  ``parse_engine`` across transitions (they were added to the carry-over
  allowlist), while ``doc_status_reset_metadata`` still drops them (they stay
  per-attempt directives that a re-queue regenerates).
* End-to-end: the first transition carrying ``parse_engine`` is the PARSING
  one (proving early persistence), and the value never "jumps" between the
  parse-stage write and the process-stage rewrite.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_RAW,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id
from lightrag.utils_pipeline import (
    _DOC_STATUS_METADATA_CARRY_OVER_KEYS,
    doc_status_metadata_carry_over,
    doc_status_reset_metadata,
    resolve_doc_status_parse_engine,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# resolve_doc_status_parse_engine — unit
# ---------------------------------------------------------------------------


def test_resolve_engine_explicit_wins():
    # A parser that reported its engine (or an enqueue-time directive) is
    # honoured verbatim regardless of format.
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_LIGHTRAG, PARSER_ENGINE_MINERU)
        == PARSER_ENGINE_MINERU
    )
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_RAW, PARSER_ENGINE_DOCLING)
        == PARSER_ENGINE_DOCLING
    )


def test_resolve_engine_format_based_fallback():
    # No explicit engine: structured lightrag → native, everything else
    # (raw passthrough, unknown) → legacy. Mirrors how a pre-engine corpus
    # was processed.
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_LIGHTRAG, None)
        == PARSER_ENGINE_NATIVE
    )
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_RAW, None)
        == PARSER_ENGINE_LEGACY
    )
    assert resolve_doc_status_parse_engine(None, None) == PARSER_ENGINE_LEGACY
    assert resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_RAW, "") == (
        PARSER_ENGINE_LEGACY
    )


# ---------------------------------------------------------------------------
# carry-over / reset allowlist interplay — unit
# ---------------------------------------------------------------------------


def test_carry_over_preserves_parse_engine_and_format():
    # Both keys are now in the carry-over allowlist, so a value stamped at
    # PARSING survives every later transition's metadata replace.
    assert "parse_format" in _DOC_STATUS_METADATA_CARRY_OVER_KEYS
    assert "parse_engine" in _DOC_STATUS_METADATA_CARRY_OVER_KEYS

    carried = doc_status_metadata_carry_over(
        {
            "metadata": {
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "parse_engine": PARSER_ENGINE_MINERU,
                "process_options": "iF",
            }
        }
    )
    assert carried["parse_format"] == FULL_DOCS_FORMAT_LIGHTRAG
    assert carried["parse_engine"] == PARSER_ENGINE_MINERU


def test_reset_still_drops_parse_engine_and_format():
    # Carry-over (across transitions) is independent of reset (back to
    # PENDING): a re-queue regenerates these per-attempt directives, so they
    # must NOT survive a reset even though they survive transitions.
    reset = doc_status_reset_metadata(
        {
            "metadata": {
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "parse_engine": PARSER_ENGINE_MINERU,
                "process_options": "iF",
                "source_file": "report.pdf",
            }
        }
    )
    assert reset == {"process_options": "iF", "source_file": "report.pdf"}
    assert "parse_engine" not in reset
    assert "parse_format" not in reset


# ---------------------------------------------------------------------------
# End-to-end: early persistence + no value jump
# ---------------------------------------------------------------------------


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt, **kwargs):
    return '{"name":"x","summary":"s","detail_description":"d"}'


def _new_rag(tmp_path: Path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"parse-engine-early-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )


def _attach_transition_spy(rag: LightRAG) -> list[tuple[str, dict]]:
    """Record (status, written-metadata) at every state-transition upsert.

    The written metadata mirrors what ``_upsert_doc_status_transition``
    persists (carry-over allowlist + ``metadata_extra``), reconstructed via
    the same helper so the recording matches storage exactly.
    """
    from lightrag.utils_pipeline import doc_status_transition_metadata

    records: list[tuple[str, dict]] = []
    orig = rag._upsert_doc_status_transition

    async def _spy(doc_id, status, status_doc, file_path, **kwargs):
        written = doc_status_transition_metadata(
            status_doc, extra=kwargs.get("metadata_extra")
        )
        status_text = status.value if isinstance(status, DocStatus) else str(status)
        records.append((status_text, dict(written)))
        return await orig(doc_id, status, status_doc, file_path, **kwargs)

    rag._upsert_doc_status_transition = _spy
    return records


def _assert_early_and_stable(records, *, expected_format, expected_engine):
    """Assert parse_* lands at PARSING and never changes afterwards."""
    with_engine = [(status, md) for status, md in records if "parse_engine" in md]
    assert with_engine, "parse_engine never written to any doc_status transition"

    # Early persistence: the FIRST transition that carries parse_engine is a
    # PARSING one — not deferred to PROCESSING/PROCESSED.
    first_status = with_engine[0][0]
    assert first_status == DocStatus.PARSING.value, (
        f"parse_engine first appeared at {first_status!r}; expected PARSING "
        f"(early persistence). Full sequence: {[s for s, _ in records]}"
    )

    # No value jump: every transition that carries the fields agrees.
    engines = {md["parse_engine"] for _, md in with_engine}
    formats = {md["parse_format"] for _, md in with_engine if "parse_format" in md}
    assert engines == {expected_engine}, (
        f"parse_engine jumped across transitions: {engines!r}; "
        f"expected a single value {expected_engine!r}"
    )
    assert formats == {expected_format}, (
        f"parse_format jumped across transitions: {formats!r}; "
        f"expected a single value {expected_format!r}"
    )


def test_raw_doc_records_legacy_engine_at_parsing(tmp_path):
    """A raw passthrough doc: parse_engine resolves to ``legacy`` and is
    stamped at PARSING, identical through PROCESSED (no jump)."""

    async def _run():
        rag = _new_rag(tmp_path / "raw")
        await rag.initialize_storages()
        records = _attach_transition_spy(rag)
        try:
            await rag.apipeline_enqueue_documents(
                "Alpha body paragraph with enough words to look real.",
                file_paths="early_raw.txt",
                track_id="track-raw",
            )
            await rag.apipeline_process_enqueue_documents()

            _assert_early_and_stable(
                records,
                expected_format=FULL_DOCS_FORMAT_RAW,
                expected_engine=PARSER_ENGINE_LEGACY,
            )

            doc_id = compute_mdhash_id("early_raw.txt", prefix="doc-")
            stored = await rag.doc_status.get_by_id(doc_id)
            assert stored is not None
            metadata = (
                stored.get("metadata")
                if isinstance(stored, dict)
                else getattr(stored, "metadata", None)
            )
            assert metadata.get("parse_format") == FULL_DOCS_FORMAT_RAW
            assert metadata.get("parse_engine") == PARSER_ENGINE_LEGACY
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def _write_lightrag_blocks(blocks_path: Path, body_paragraphs: list[str]) -> None:
    lines = [
        json.dumps(
            {
                "type": "meta",
                "format": "lightrag",
                "version": "1.0",
                "format_version": "1.0",
            },
            ensure_ascii=False,
        )
    ]
    for i, para in enumerate(body_paragraphs):
        lines.append(
            json.dumps(
                {
                    "type": "content",
                    "blockid": f"b{i}",
                    "format": "plain_text",
                    "content": para,
                },
                ensure_ascii=False,
            )
        )
    blocks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_lightrag_doc_records_native_engine_at_parsing(tmp_path, monkeypatch):
    """A lightrag-format doc (cache-hit, parser does not re-persist):
    parse_engine resolves to ``native`` and is stamped at PARSING, identical
    through PROCESSED."""

    async def _run():
        input_dir = tmp_path / "input"
        parsed_dir = input_dir / "__parsed__"
        parsed_dir.mkdir(parents=True)
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        blocks_path = parsed_dir / "early.blocks.jsonl"
        _write_lightrag_blocks(blocks_path, ["Body paragraph for lightrag doc."])

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()
        records = _attach_transition_spy(rag)
        try:
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="early.lightrag",
                docs_format=FULL_DOCS_FORMAT_LIGHTRAG,
                lightrag_document_paths="__parsed__/early.blocks.jsonl",
                track_id="track-lr",
            )
            await rag.apipeline_process_enqueue_documents()

            _assert_early_and_stable(
                records,
                expected_format=FULL_DOCS_FORMAT_LIGHTRAG,
                expected_engine=PARSER_ENGINE_NATIVE,
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
