"""F-chunking parity between raw and lightrag formats.

After the F-chunking unification, ``apipeline_process_enqueue_documents``
strips the ``{{LRdoc}}`` marker from lightrag-format content and feeds the
result into the same ``chunking_func`` used by raw documents.  These tests
guard the contract end-to-end:

* T1: identical input text produces identical chunking inputs whether it
  arrives as raw or as an already-parsed lightrag full_docs row pulled
  back through the parse queue on resume (ReuseParser) — the only
  production path for lightrag rows now that the ``docs_format="lightrag"``
  enqueue entrypoint is removed (the in-run parse → chunk path is T6).
* T2: after a pending_parse document is parsed by the native engine,
  ``full_docs.content`` carries the *full* merged text with the
  ``{{LRdoc}}`` marker (written by ``_persist_parsed_full_docs``), while
  ``doc_status`` reports the bare body length / summary (no marker
  leakage).
* T3: ``extraction_meta["parse_format"]`` (surfaced via
  ``doc_status.metadata``) is ``"lightrag"`` for natively-parsed docs —
  previously a structured-parse fallback always tagged ``raw`` and
  silently mislabelled the persisted record.
* T4: a raw document whose body coincidentally *looks* like structured
  JSONL is still tokenised as plain text — guards against re-introducing
  dropped structured-format detection in the raw path.
* T5: ``process_options`` selecting R/V/P logs the deferred-strategy
  warning and falls back to fixed-token chunking.
* T6: a ``pending_parse`` document that resolves to lightrag at parse
  time ends up with a real ``content_summary`` after PROCESSED — the
  ANALYZING transition refreshes the summary from the parsed body so
  pending-parse rows no longer carry the empty enqueue-time placeholder
  through to the user-facing list APIs.
* T7: a raw document whose body *literally* starts with ``{{LRdoc}}``
  is chunked verbatim — guards against accidental re-introduction of an
  unconditional ``strip_lightrag_doc_prefix`` at the chunking boundary
  (which would silently drop the user's first 9 characters).
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    LIGHTRAG_DOC_CONTENT_PREFIX,
)
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    get_content_summary,
)
from lightrag.utils_pipeline import make_lightrag_doc_content


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors the harness used by test_pipeline_release_closure)
# ---------------------------------------------------------------------------


class _SimpleTokenizerImpl:
    """Char-level tokenizer so 1 char ≈ 1 token; keeps assertions readable."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
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
        workspace=f"chunking-parity-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


def _attach_chunking_spy(rag: LightRAG) -> dict:
    """Replace ``rag.chunking_func`` with a recording wrapper.

    Returns a dict whose ``input`` key receives the second positional arg
    (the content string) at every chunking call.  The original chunker
    runs normally so the pipeline reaches PROCESSED.
    """
    captured: dict = {"input": None, "calls": 0}
    real = rag.chunking_func

    def _spy(tokenizer, content, *args, **kwargs):
        captured["input"] = content
        captured["calls"] += 1
        return real(tokenizer, content, *args, **kwargs)

    rag.chunking_func = _spy
    return captured


# ---------------------------------------------------------------------------
# T1 — parity: raw vs lightrag produce identical chunking input
# ---------------------------------------------------------------------------


async def _seed_lightrag_row(rag: LightRAG, doc_id: str, *, body: str, file_path: str):
    """Seed an already-parsed lightrag full_docs row + a PENDING doc_status.

    Mirrors how lightrag-format rows exist in production: written by the
    parsers (``_persist_parsed_full_docs``) and pulled back through the
    parse queue (→ ReuseParser) only on resume/retry.
    """
    from datetime import datetime, timezone

    await rag.full_docs.upsert(
        {
            doc_id: {
                "content": make_lightrag_doc_content(body),
                "file_path": file_path,
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
            }
        }
    )
    now = datetime.now(timezone.utc).isoformat()
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": "pending",
                "content_summary": get_content_summary(body),
                "content_length": len(body),
                "file_path": file_path,
                "created_at": now,
                "updated_at": now,
                "track_id": "track-lr",
            }
        }
    )


@pytest.mark.offline
def test_chunking_input_parity_raw_vs_lightrag(tmp_path):
    """Same body text must reach ``chunking_func`` with byte-identical input
    whether it arrives as raw or as a resumed lightrag full_docs row."""

    paragraphs = [
        "Alpha paragraph with enough words to make it look real.",
        "Beta paragraph extends the body so chunking has substance.",
        "Gamma paragraph closes the document with a few more sentences.",
    ]
    expected_merged = "\n\n".join(paragraphs)

    async def _run():
        # ---- RAW path ----
        rag_raw = _new_rag(tmp_path / "raw")
        await rag_raw.initialize_storages()
        spy_raw = _attach_chunking_spy(rag_raw)
        try:
            await rag_raw.apipeline_enqueue_documents(
                expected_merged,
                file_paths="parity_raw.txt",
                track_id="track-raw",
            )
            await rag_raw.apipeline_process_enqueue_documents()
        finally:
            await rag_raw.finalize_storages()

        # ---- LIGHTRAG path (resume: seeded parsed row → ReuseParser) ----
        rag_lr = _new_rag(tmp_path / "lr")
        await rag_lr.initialize_storages()
        spy_lr = _attach_chunking_spy(rag_lr)
        try:
            await _seed_lightrag_row(
                rag_lr,
                "doc-parity-lr",
                body=expected_merged,
                file_path="parity.lightrag",
            )
            await rag_lr.apipeline_process_enqueue_documents()
        finally:
            await rag_lr.finalize_storages()

        assert spy_raw["calls"] >= 1, "raw doc never reached chunking_func"
        assert spy_lr["calls"] >= 1, "lightrag doc never reached chunking_func"
        assert spy_lr["input"] == spy_raw["input"] == expected_merged, (
            "chunking_func received different inputs for raw vs lightrag; "
            f"raw={spy_raw['input']!r}\nlr={spy_lr['input']!r}"
        )
        assert not spy_lr["input"].startswith(LIGHTRAG_DOC_CONTENT_PREFIX), (
            "{{LRdoc}} marker leaked into chunking_func input"
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# T2 — full_docs.content carries full text; doc_status reports bare body
# ---------------------------------------------------------------------------


def _stub_docx_blocks(monkeypatch, body_paragraphs: list[str]) -> None:
    """Stub the docx extractor so native parse yields deterministic blocks;
    the adapter still writes the canonical .blocks.jsonl + sidecars and
    persists the lightrag-format full_docs row."""

    def _stub_extract(file_path, fixlevel=None, drawing_context=None, **kwargs):
        return [
            {
                "uuid": f"para-{i}",
                "uuid_end": f"para-{i}",
                "heading": "",
                "content": para,
                "type": "text",
                "parent_headings": [],
                "level": 0,
                "table_chunk_role": "none",
            }
            for i, para in enumerate(body_paragraphs)
        ]

    monkeypatch.setattr(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        _stub_extract,
    )


@pytest.mark.offline
def test_full_docs_content_carries_full_merged_text(tmp_path, monkeypatch):
    """After the native engine parses a pending_parse upload,
    ``_persist_parsed_full_docs`` stores the *full* merged text with the
    ``{{LRdoc}}`` marker while doc_status reports bare-body semantics."""

    body = "x" * 5000  # single paragraph, 5000 chars

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        (input_dir / "big.docx").write_bytes(b"fake docx bytes")
        _stub_docx_blocks(monkeypatch, [body])

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="big.docx",
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine="native",
                track_id="track-big",
            )
            await rag.apipeline_process_enqueue_documents()

            doc_id = compute_mdhash_id("big.docx", prefix="doc-")
            full_doc = await rag.full_docs.get_by_id(doc_id)
            assert full_doc is not None
            # full_docs preserves the marker AND the full merged text.
            assert full_doc["content"] == LIGHTRAG_DOC_CONTENT_PREFIX + body
            assert full_doc.get("parse_format") == FULL_DOCS_FORMAT_LIGHTRAG
            assert full_doc.get("sidecar_location"), (
                "native parse must record the sidecar_location on the "
                "lightrag full_docs row"
            )

            # doc_status reports body-length semantics (no marker leakage).
            status_doc = await rag.doc_status.get_by_id(doc_id)
            assert status_doc is not None
            length = (
                status_doc.get("content_length")
                if isinstance(status_doc, dict)
                else getattr(status_doc, "content_length", None)
            )
            summary = (
                status_doc.get("content_summary")
                if isinstance(status_doc, dict)
                else getattr(status_doc, "content_summary", "")
            )
            assert length == 5000, f"content_length should match body, got {length}"
            assert not summary.startswith(LIGHTRAG_DOC_CONTENT_PREFIX)
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# T3 — extraction_meta.parse_format reflects persisted format (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_extraction_meta_records_lightrag_parse_format(tmp_path, monkeypatch):
    """Before the unification, a structured-parse fallback tagged
    ``extraction_meta.parse_format = raw`` for lightrag docs, silently
    mislabelling them in ``doc_status.metadata``.  Assert the tag now
    reflects the persisted format end-to-end (pending_parse upload →
    native parse → lightrag full_docs row)."""

    paragraphs = ["Body paragraph for parse_format tagging test."]

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        (input_dir / "tag.docx").write_bytes(b"fake docx bytes")
        _stub_docx_blocks(monkeypatch, paragraphs)

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()

        try:
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="tag.docx",
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine="native",
                track_id="track-tag",
            )
            await rag.apipeline_process_enqueue_documents()

            doc_id = compute_mdhash_id("tag.docx", prefix="doc-")
            status_doc = await rag.doc_status.get_by_id(doc_id)
            assert status_doc is not None
            metadata = (
                status_doc.get("metadata")
                if isinstance(status_doc, dict)
                else getattr(status_doc, "metadata", None)
            )
            assert isinstance(metadata, dict), (
                f"doc_status.metadata should be a dict, got {type(metadata)!r}"
            )
            assert metadata.get("parse_format") == FULL_DOCS_FORMAT_LIGHTRAG, (
                f"doc_status.metadata.parse_format="
                f"{metadata.get('parse_format')!r}; "
                f"expected {FULL_DOCS_FORMAT_LIGHTRAG!r} so the multimodal "
                f"sidecar merge path opens"
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# T4 — JSONL-shaped raw text is still treated as plain text
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_jsonl_shaped_raw_text_chunks_as_plain_text(tmp_path):
    """A raw document whose body coincidentally resembles structured JSONL
    must be tokenised plainly — guarding against accidental
    re-introduction of removed structured-format detection."""

    # No trailing newline — sanitize_text_for_encoding strips trailing
    # whitespace on raw enqueue, and that pre-chunking cleanup is unrelated
    # to structured-format detection.
    pseudo_jsonl = (
        json.dumps({"type": "meta", "format_version": "1.0"})
        + "\n"
        + json.dumps(
            {
                "type": "text",
                "chunk_id": "c0",
                "chunk_order_index": 0,
                "content": "fake structured line",
            }
        )
    )

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        spy = _attach_chunking_spy(rag)
        try:
            await rag.apipeline_enqueue_documents(
                pseudo_jsonl,
                file_paths="pseudo.txt",
                track_id="track-pseudo",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

        # The full pseudo-jsonl text reaches chunking_func; nothing parses
        # it as JSONL and hijacks the chunks list.
        assert spy["input"] == pseudo_jsonl

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# T5 — R/V/P process_options trigger the deferred-strategy warning
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    """Capture log records into an in-memory list.

    The ``lightrag`` logger has ``propagate = False`` so pytest's caplog
    fixture cannot intercept its records via the root logger; this handler
    attaches directly to the logger we care about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.offline
def test_explicit_R_dispatches_to_recursive_character(tmp_path, monkeypatch):
    """``process_options=R`` must invoke
    :func:`chunking_by_recursive_character` (the new file-chunker
    contract) rather than the legacy ``chunking_func``.

    Verifies the explicit-selector dispatch contract:
      1. ``chunking_by_recursive_character`` runs at least once.
      2. The legacy ``chunking_func`` is bypassed entirely.
      3. The deprecated "R/V not yet implemented" warning no longer
         appears (now that R has a real implementation).
    """

    pytest.importorskip("langchain_text_splitters")

    import lightrag.chunker as chunker_pkg
    from lightrag.chunker import chunking_by_recursive_character as real_r

    captured = {"calls": 0}

    def _r_spy(*args, **kwargs):
        captured["calls"] += 1
        return real_r(*args, **kwargs)

    # The dispatcher does ``from lightrag.chunker import …`` inside the
    # function body, which re-resolves the name from the package each
    # call — patching the package attribute is enough to intercept it.
    monkeypatch.setattr(chunker_pkg, "chunking_by_recursive_character", _r_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        legacy_spy = _attach_chunking_spy(rag)

        lightrag_logger = logging.getLogger("lightrag")
        list_handler = _ListHandler()
        list_handler.setLevel(logging.WARNING)
        lightrag_logger.addHandler(list_handler)
        try:
            await rag.apipeline_enqueue_documents(
                "Body paragraph one.\n\nBody paragraph two for R dispatch test.",
                file_paths="rs.[native-R].txt",
                track_id="track-rs",
                process_options="R",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            lightrag_logger.removeHandler(list_handler)
            await rag.finalize_storages()

        assert captured["calls"] >= 1, "R must route to chunking_by_recursive_character"
        assert legacy_spy["calls"] == 0, (
            "explicit process_options selector must bypass legacy "
            "chunking_func; got "
            f"{legacy_spy['calls']} calls"
        )
        warning_messages = [
            rec.getMessage()
            for rec in list_handler.records
            if rec.levelno == logging.WARNING
        ]
        assert not any(
            "R/V strategies are not yet implemented" in msg for msg in warning_messages
        ), (
            "deprecated 'not yet implemented' warning must be gone now "
            f"that R is wired up; saw: {warning_messages!r}"
        )

    asyncio.run(_run())


@pytest.mark.offline
def test_explicit_V_dispatches_to_semantic_vector(tmp_path, monkeypatch):
    """``process_options=V`` must invoke
    :func:`chunking_by_semantic_vector` and bypass the legacy
    ``chunking_func``.  The test installs a stub embedding (the spy
    short-circuits before the real LangChain SemanticChunker runs) so
    the assertion is purely about dispatch routing, not chunk quality.
    """

    pytest.importorskip("langchain_experimental")

    import lightrag.chunker as chunker_pkg

    captured = {"calls": 0}

    async def _v_spy(*args, **kwargs):
        # Short-circuit: skip langchain SemanticChunker entirely and
        # return one synthetic chunk.  We're only verifying that the
        # dispatcher routed here with the right keyword args.
        captured["calls"] += 1
        captured["embedding_func"] = kwargs.get("embedding_func")
        captured["chunk_token_size"] = args[2] if len(args) > 2 else None
        return [
            {"tokens": 5, "content": "stub", "chunk_order_index": 0},
        ]

    monkeypatch.setattr(chunker_pkg, "chunking_by_semantic_vector", _v_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        legacy_spy = _attach_chunking_spy(rag)
        try:
            await rag.apipeline_enqueue_documents(
                "Body for V dispatch test. Sentence one. Sentence two.",
                file_paths="vs.[native-V].txt",
                track_id="track-vs",
                process_options="V",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

        assert captured["calls"] >= 1, "V must route to chunking_by_semantic_vector"
        assert captured.get("embedding_func") is rag.embedding_func, (
            "dispatcher must hand the LightRAG embedding_func to the V chunker"
        )
        assert legacy_spy["calls"] == 0, (
            "explicit process_options selector must bypass legacy chunking_func"
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# T6 — pending_parse → lightrag summary is populated after PROCESSED
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_pending_parse_lightrag_summary_populated_after_processed(
    tmp_path, monkeypatch
):
    """A document enqueued as ``pending_parse`` has empty content at
    enqueue time, so ``content_summary`` starts empty.  After
    ``parse_native`` produces ``.blocks.jsonl`` and the state machine
    moves through ANALYZING → PROCESSING → PROCESSED, the summary must
    reflect the parsed body — not the enqueue-time placeholder."""

    body_paragraphs = [
        "Pending-parse summary regression body paragraph one.",
        "Body paragraph two carries enough text for a meaningful preview.",
        "Body paragraph three closes the document.",
    ]

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "summary.docx"
        source_path.write_bytes(b"fake docx bytes")
        _stub_docx_blocks(monkeypatch, body_paragraphs)

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="summary.docx",
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine="native",
                track_id="track-summary",
            )

            doc_id = compute_mdhash_id("summary.docx", prefix="doc-")

            pending = await rag.doc_status.get_by_id(doc_id)
            assert pending is not None
            pending_summary = (
                pending.get("content_summary")
                if isinstance(pending, dict)
                else getattr(pending, "content_summary", "")
            )
            # At enqueue time pending_parse content is "" so summary is empty.
            assert pending_summary == "", (
                f"pending_parse should start with empty summary, got "
                f"{pending_summary!r}"
            )

            await rag.apipeline_process_enqueue_documents()

            final = await rag.doc_status.get_by_id(doc_id)
            assert final is not None
            final_summary = (
                final.get("content_summary")
                if isinstance(final, dict)
                else getattr(final, "content_summary", "")
            )
            final_length = (
                final.get("content_length")
                if isinstance(final, dict)
                else getattr(final, "content_length", 0)
            )

            assert final_summary, (
                "content_summary still empty after PROCESSED; ANALYZING "
                "refresh did not propagate"
            )
            assert not final_summary.startswith(LIGHTRAG_DOC_CONTENT_PREFIX), (
                f"{{LRdoc}} marker leaked into doc_status summary: {final_summary!r}"
            )
            # The parser stub produces these paragraphs verbatim; the
            # blocks.jsonl writer joins them with a blank line, so the
            # summary must be a prefix of that merged text.
            merged_text = "\n\n".join(body_paragraphs)
            assert final_summary == get_content_summary(merged_text), (
                f"summary should match get_content_summary(merged_text); "
                f"got {final_summary!r} vs "
                f"{get_content_summary(merged_text)!r}"
            )
            assert final_length == len(merged_text), (
                f"content_length should equal len(merged_text)={len(merged_text)}, "
                f"got {final_length}"
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# T7 — raw text starting with {{LRdoc}} must not be stripped at chunking
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_raw_text_starting_with_marker_chunked_verbatim(tmp_path):
    """A raw document whose body literally begins with ``{{LRdoc}}`` is a
    legitimate user input — the chunking branch must not strip those 9
    characters.  ``strip_lightrag_doc_prefix`` is a lightrag-only contract
    enforced by ``parse_native``; raw paths return ``content_data["content"]``
    verbatim, so chunking must hand the body to ``chunking_func`` unchanged."""

    body_with_marker = LIGHTRAG_DOC_CONTENT_PREFIX + (
        "literal-marker-prefix raw document body that should survive "
        "the chunking boundary intact."
    )

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        spy = _attach_chunking_spy(rag)
        try:
            await rag.apipeline_enqueue_documents(
                body_with_marker,
                file_paths="marker_raw.txt",
                track_id="track-marker",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

        assert spy["calls"] >= 1, "raw doc never reached chunking_func"
        # The full body — including the literal {{LRdoc}} prefix — must
        # reach chunking_func; nothing in the chunking branch should
        # treat the marker as a stripping signal for raw content.
        assert spy["input"] == body_with_marker, (
            "chunking_func received corrupted input: "
            f"got {spy['input']!r}, expected {body_with_marker!r}"
        )
        assert spy["input"].startswith(LIGHTRAG_DOC_CONTENT_PREFIX), (
            "literal marker prefix lost at chunking boundary"
        )

    asyncio.run(_run())
