import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocStatus
from lightrag.constants import (
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_DIR_NAME,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
)
from lightrag.operate import (
    _get_relationship_vdb_timeout_seconds,
    _parse_mm_display_name,
)
from lightrag.parser.routing import (
    FilenameParserHintError,
    ParserRoutingConfigError,
    canonicalize_parser_hinted_basename,
    resolve_file_parser_engine,
    resolve_stored_document_parser_engine,
    validate_parser_routing_config,
)
from lightrag.parser.base import ParseContext
from lightrag.parser.registry import get_parser
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    safe_vdb_operation_with_exception,
)


async def _parse_via_registry(rag, engine, doc_id, file_path, content_data):
    """Drive a parser the way the pipeline worker does (registry dispatch)."""
    result = await get_parser(engine).parse(
        ParseContext(rag, doc_id, file_path, content_data)
    )
    return result.to_dict()


class _SimpleTokenizerImpl:
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

    # analyze_multimodal short-circuits when vlm_process_enable is False; this
    # helper drives several VLM-specific tests, so default the switch ON.
    kwargs.setdefault("vlm_process_enable", True)

    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"test-release-closure-{tmp_path.name}",
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
def test_parse_engine_routing_by_filename_and_env(monkeypatch):
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")
    assert (
        resolve_stored_document_parser_engine("a.[docling-iet].docx", {}) == "docling"
    )

    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:mineru-iet,*:native")
    assert resolve_stored_document_parser_engine("paper.pdf", {}) == "mineru"
    # A row that is not pending_parse is already-extracted content -> the
    # passthrough handler (legacy now means worker-stage extraction).
    assert (
        resolve_stored_document_parser_engine("paper.pdf", {"parse_engine": "native"})
        == "passthrough"
    )


@pytest.mark.offline
def test_parse_engine_rule_fallback_and_default_legacy(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:native,*:legacy")
    assert resolve_stored_document_parser_engine("paper.pdf", {}) == "legacy"

    monkeypatch.setenv("LIGHTRAG_PARSER", "pptx:docling,*:legacy")
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    assert resolve_stored_document_parser_engine("slides.pptx", {}) == "legacy"

    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    assert resolve_stored_document_parser_engine("slides.pptx", {}) == "legacy"


@pytest.mark.offline
def test_canonicalize_parser_hinted_basename():
    assert canonicalize_parser_hinted_basename("abc.[native].docx") == "abc.docx"
    assert canonicalize_parser_hinted_basename("/tmp/a.b.[mineru].pdf") == "a.b.pdf"
    assert canonicalize_parser_hinted_basename("abc.[draft].docx") == "abc.[draft].docx"
    # Engine token is case-insensitive (normalize_parser_engine lower-cases).
    assert canonicalize_parser_hinted_basename("abc.[NATIVE].docx") == "abc.docx"
    # Engine sub-variants like "mineru-iet" normalize to a base engine.
    assert canonicalize_parser_hinted_basename("abc.[mineru-iet].pdf") == "abc.pdf"
    # No extension after the bracket: pattern requires ``.[engine].ext``.
    assert canonicalize_parser_hinted_basename("abc.[native]") == "abc.[native]"
    # Plain basename without any hint is returned unchanged.
    assert canonicalize_parser_hinted_basename("abc.docx") == "abc.docx"
    # Bracket without a leading dot is not a hint.
    assert canonicalize_parser_hinted_basename("[native].docx") == "[native].docx"
    # Nested hints: only the outermost segment is stripped.
    assert (
        canonicalize_parser_hinted_basename("name.[native].[mineru].pdf")
        == "name.[native].pdf"
    )
    # New options-only and engine+options forms strip cleanly too.
    assert canonicalize_parser_hinted_basename("foo.[-!].docx") == "foo.docx"
    assert canonicalize_parser_hinted_basename("foo.[native-iet].docx") == "foo.docx"
    assert canonicalize_parser_hinted_basename("foo.[mineru-R!].pdf") == "foo.pdf"
    # Options without the leading hyphen and unknown parsers are left alone.
    assert canonicalize_parser_hinted_basename("foo.[!].docx") == "foo.[!].docx"
    assert (
        canonicalize_parser_hinted_basename("foo.[native-].docx")
        == "foo.[native-].docx"
    )
    assert canonicalize_parser_hinted_basename("foo.[xyz].docx") == "foo.[xyz].docx"


@pytest.mark.offline
def test_filename_parser_directives_decodes_engine_and_options():
    from lightrag.parser.routing import filename_parser_directives

    assert filename_parser_directives("paper.[native-iet].docx") == ("native", "iet")
    assert filename_parser_directives("memo.[native-R!].md") == ("native", "R!")
    assert filename_parser_directives("report.[-!].pdf") == (None, "!")
    assert filename_parser_directives("doc.[mineru].docx") == ("mineru", "")
    assert filename_parser_directives("foo.docx") == (None, "")
    # Unsupported tokens and old options-only syntax stay unparsed.
    assert filename_parser_directives("foo.[!].docx") == (None, "")
    assert filename_parser_directives("foo.[draft].docx") == (None, "")


@pytest.mark.offline
def test_filename_hint_rejects_invalid_engine_qualified_options():
    """Engine-qualified hints with bad option chars must fail validation
    during parser directive resolution instead of silently falling back to
    parser rules/defaults.
    """
    from lightrag.parser.routing import (
        canonicalize_parser_hinted_basename,
        filename_parser_directives,
        resolve_file_parser_directives,
    )

    # Low-level helpers stay non-throwing for scan grouping/canonicalization.
    assert filename_parser_directives("foo.[native-FR].docx") == (None, "")
    assert filename_parser_directives("foo.[native-Q].docx") == (None, "")

    assert (
        canonicalize_parser_hinted_basename("foo.[native-FR].docx")
        == "foo.[native-FR].docx"
    )
    assert (
        canonicalize_parser_hinted_basename("foo.[native-Q].docx")
        == "foo.[native-Q].docx"
    )

    with pytest.raises(FilenameParserHintError, match="multiple chunking modes"):
        resolve_file_parser_directives("foo.[native-FR].docx")
    with pytest.raises(FilenameParserHintError, match="unsupported character"):
        resolve_file_parser_directives("foo.[native-Q].docx")
    with pytest.raises(FilenameParserHintError, match="options-only filename hints"):
        resolve_file_parser_directives("foo.[!].docx")
    with pytest.raises(FilenameParserHintError, match="options-only filename hints"):
        resolve_file_parser_directives("foo.[iet].docx")
    with pytest.raises(
        FilenameParserHintError, match="unsupported parser engine 'abc'"
    ):
        resolve_file_parser_directives("foo.[abc].docx")
    with pytest.raises(
        FilenameParserHintError, match="unsupported parser engine 'xyz'"
    ):
        resolve_file_parser_directives("foo.[xyz].docx")
    with pytest.raises(FilenameParserHintError, match="is empty"):
        resolve_file_parser_directives("foo.[].docx")
    with pytest.raises(FilenameParserHintError, match="empty process options"):
        resolve_file_parser_directives("foo.[-].docx")
    with pytest.raises(FilenameParserHintError, match="empty process options"):
        resolve_file_parser_directives("foo.[native-].docx")


@pytest.mark.offline
def test_filename_hint_missing_required_endpoint_rejects(monkeypatch):
    from lightrag.parser.routing import resolve_file_parser_directives

    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)

    with pytest.raises(FilenameParserHintError, match="requires DOCLING_ENDPOINT"):
        resolve_file_parser_directives("foo.[docling].docx")


@pytest.mark.offline
def test_parse_process_options_decodes_flags():
    from lightrag.parser.routing import parse_process_options

    opts = parse_process_options("iet")
    assert opts.images and opts.tables and opts.equations
    assert not opts.skip_kg
    assert opts.chunking == "F"

    opts = parse_process_options("R!")
    assert opts.skip_kg and opts.chunking == "R"
    assert not opts.images and not opts.tables and not opts.equations

    opts = parse_process_options("P")
    assert opts.chunking == "P"

    opts = parse_process_options("")
    assert not (opts.images or opts.tables or opts.equations or opts.skip_kg)
    assert opts.chunking == "F"


@pytest.mark.offline
def test_validate_process_options_rejects_invalid_combos():
    from lightrag.parser.routing import validate_process_options

    assert validate_process_options("iet") == []
    assert validate_process_options("R!") == []
    # F+R conflict is reported.
    errs = validate_process_options("FR")
    assert any("multiple chunking modes" in m for m in errs)
    # Lowercase chunking selectors are not valid.
    errs = validate_process_options("f")
    assert any("'f'" in m for m in errs)
    # Unknown chars are reported individually.
    errs = validate_process_options("xyz")
    assert sum(1 for m in errs if "unsupported character" in m) == 3


@pytest.mark.offline
def test_lightrag_parser_rule_supports_options_suffix(monkeypatch):
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    # Valid options suffix passes validation.
    validate_parser_routing_config("docx:native-iet,*:legacy")

    # Invalid options suffix is rejected with the rule label and message.
    with pytest.raises(ParserRoutingConfigError, match="multiple chunking modes"):
        validate_parser_routing_config("docx:native-FR,*:legacy")

    with pytest.raises(ParserRoutingConfigError, match="unsupported character"):
        validate_parser_routing_config("docx:native-Q,*:legacy")


@pytest.mark.offline
def test_resolve_file_parser_directives_priority(monkeypatch):
    from lightrag.parser.routing import resolve_file_parser_directives

    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native-iet,*:legacy")

    # Filename hint takes precedence for engine and options.
    engine, options = resolve_file_parser_directives("paper.[native-R!].docx")
    assert engine == "native"
    assert options == "R!"

    # No filename hint → fall through to LIGHTRAG_PARSER defaults for both.
    engine, options = resolve_file_parser_directives("plain.docx")
    assert engine == "native"
    assert options == "iet"

    # Options-only hint keeps engine from rule but uses hinted options.
    engine, options = resolve_file_parser_directives("plain.[-!].docx")
    assert engine == "native"
    assert options == "!"


@pytest.mark.offline
def test_doc_status_metadata_carry_over_helper():
    """``doc_status_transition_metadata`` preserves long-lived per-doc fields
    and layers in any transition-specific extras passed via ``extra=``.
    Empty / missing carry-over fields are dropped, not written as null.
    """
    from lightrag.utils_pipeline import doc_status_transition_metadata

    class _StubStatusDoc:
        def __init__(self, metadata):
            self.metadata = metadata

    # Carries process_options forward.
    md = doc_status_transition_metadata(_StubStatusDoc({"process_options": "iet"}))
    assert md == {"process_options": "iet"}

    # Carries the internal pending-parse source basename forward for retries.
    md = doc_status_transition_metadata(
        _StubStatusDoc({"source_file": "demo.[mineru].pdf"})
    )
    assert md == {"source_file": "demo.[mineru].pdf"}

    # Layers in transition extras while keeping the carry-over.
    md = doc_status_transition_metadata(
        _StubStatusDoc({"process_options": "R!"}),
        extra={"process_start_time": 12345},
    )
    assert md == {"process_options": "R!", "process_start_time": 12345}

    # No carry-over when metadata is missing or empty.
    assert doc_status_transition_metadata(_StubStatusDoc({})) == {}
    assert doc_status_transition_metadata(None) == {}

    # Empty / None process_options are not written as null.
    assert doc_status_transition_metadata(_StubStatusDoc({"process_options": ""})) == {}
    assert (
        doc_status_transition_metadata(_StubStatusDoc({"process_options": None})) == {}
    )


@pytest.mark.offline
def test_carry_over_keys_grouped_by_stage():
    """Strict tuple-equality guard on ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``.

    The tuple order is the WebUI ``DocumentStatusDetailsDialog`` render order,
    so per-stage fields must stay grouped (parse-stage fields then analyze-stage
    trio). ``parse_format`` / ``parse_engine`` join the parse-stage group so
    they are stamped at PARSING (once the engine has run) and carried through
    to PROCESSED instead of only landing at PROCESSING. Locking the order here
    forces any future field addition to update this assertion alongside the
    tuple, preventing silent regressions in the dialog's timeline display.
    """
    from lightrag.utils_pipeline import _DOC_STATUS_METADATA_CARRY_OVER_KEYS

    assert _DOC_STATUS_METADATA_CARRY_OVER_KEYS == (
        "process_options",
        "source_file",
        "parse_warnings",
        "chunk_opts",
        "parse_start_time",
        "parse_end_time",
        "parse_stage_skipped",
        "parse_format",
        "parse_engine",
        "analyzing_start_time",
        "analyzing_end_time",
        "analyzing_stage_skipped",
    )


@pytest.mark.offline
def test_carry_over_helper_propagates_end_times_and_skipped():
    """Stage-end timestamps and skipped flags must survive carry-over so the
    PROCESSING / PROCESSED / FAILED upserts keep them visible for post-mortem
    stage-duration analysis.
    """
    from lightrag.utils_pipeline import doc_status_transition_metadata

    class _StubStatusDoc:
        def __init__(self, metadata):
            self.metadata = metadata

    md = doc_status_transition_metadata(
        _StubStatusDoc(
            {
                "parse_start_time": 1700000000,
                "parse_end_time": 1700000010,
                "analyzing_start_time": 1700000020,
                "analyzing_end_time": 1700000050,
            }
        )
    )
    assert md == {
        "parse_start_time": 1700000000,
        "parse_end_time": 1700000010,
        "analyzing_start_time": 1700000020,
        "analyzing_end_time": 1700000050,
    }

    # Skipped flags (bool True) also survive carry-over.
    md = doc_status_transition_metadata(
        _StubStatusDoc(
            {
                "parse_stage_skipped": True,
                "analyzing_stage_skipped": True,
            }
        )
    )
    assert md == {
        "parse_stage_skipped": True,
        "analyzing_stage_skipped": True,
    }


def _status_value_text(status):
    """Helper: extract the value of a DocStatus enum or raw status string."""
    if hasattr(status, "value"):
        return status.value
    return str(status)


@pytest.mark.offline
def test_doc_status_metadata_survives_processed_transition(tmp_path):
    """End-to-end: a document enqueued with process_options must keep
    ``metadata.process_options`` set in ``doc_status`` after the pipeline
    drives it all the way to PROCESSED.  This exercises the full state
    machine (PENDING → PARSING → ANALYZING → PROCESSING → PROCESSED) and
    asserts the carry-over works at every transition.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Some content body for chunking.",
                file_paths="metadata_carry.txt",
                track_id="track-md-carry",
                process_options="iet!",
            )

            doc_id = compute_mdhash_id("metadata_carry.txt", prefix="doc-")
            pending_status = await rag.doc_status.get_by_id(doc_id)
            assert pending_status is not None
            assert (pending_status.get("metadata") or {}).get(
                "process_options"
            ) == "iet!"

            # Run the pipeline through to PROCESSED.
            await rag.apipeline_process_enqueue_documents()

            final_status = await rag.doc_status.get_by_id(doc_id)
            assert final_status is not None
            assert _status_value_text(final_status.get("status")) == "processed"
            metadata = final_status.get("metadata") or {}
            assert metadata.get("process_options") == "iet!", (
                f"process_options dropped during state-machine transitions; "
                f"final metadata: {metadata!r}"
            )
            # parse_native on FULL_DOCS_FORMAT_RAW does not actually parse —
            # it passes content through verbatim — so the skip branch fires
            # and ``parse_end_time`` stays absent. ``parse_stage_skipped``
            # is the cache-hit / no-parse-work sentinel (same field used by
            # parse_mineru / parse_docling for raw-bundle cache hits).
            assert isinstance(metadata.get("parse_start_time"), int)
            assert metadata.get("parse_stage_skipped") is True
            assert "parse_end_time" not in metadata
            # parse_native on raw content returns blocks_path="", which makes
            # analyze_multimodal take the "no blocks_path" early-return branch
            # and set analyzing_stage_skipped=True (no analyzing_end_time).
            assert isinstance(metadata.get("analyzing_start_time"), int)
            assert metadata.get("analyzing_stage_skipped") is True
            assert "analyzing_end_time" not in metadata
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_legacy_source_file_name_migrated_and_carried_over(tmp_path):
    """Regression: a document persisted before the ``source_file_name`` →
    ``source_file`` rename carries its source hint only as
    ``metadata['source_file_name']`` (and full_docs lacks the key entirely).

    The parse worker must normalize the legacy key onto the new
    ``source_file`` key on the in-memory status_doc BEFORE the PARSING upsert.
    Otherwise the carry-over allowlist (which no longer lists the legacy key)
    drops the hint from doc_status, so a retry after a parse failure — before
    full_docs is rewritten — can no longer resolve the hinted source file.
    Here we assert the hint survives all the way to PROCESSED under the new key.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            doc_id = compute_mdhash_id("legacy_hint.txt", prefix="doc-")

            # full_docs has NO source hint — the only copy lives under the
            # legacy metadata key on doc_status, exactly as a pre-rename doc
            # would have persisted it.
            await rag.full_docs.upsert(
                {
                    doc_id: {
                        "content": "legacy body for chunking.",
                        "file_path": "legacy_hint.txt",
                        "parse_format": "raw",
                        "parse_engine": "legacy",
                        "content_hash": "legacyhash",
                    }
                }
            )
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PENDING,
                        "content_summary": "legacy body for chunking.",
                        "content_length": len("legacy body for chunking."),
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:01+00:00",
                        "file_path": "legacy_hint.txt",
                        "track_id": "track-legacy",
                        "content_hash": "legacyhash",
                        "metadata": {"source_file_name": "legacy_hint.[mineru].pdf"},
                    }
                }
            )

            await rag.apipeline_process_enqueue_documents()

            final_status = await rag.doc_status.get_by_id(doc_id)
            assert final_status is not None
            assert _status_value_text(final_status.get("status")) == "processed"
            metadata = final_status.get("metadata") or {}
            assert metadata.get("source_file") == "legacy_hint.[mineru].pdf", (
                "legacy source_file_name was not migrated to source_file and "
                f"carried over; final metadata: {metadata!r}"
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_soft_failure_writes_neither_end_time_nor_skipped(tmp_path):
    """If ``analyze_multimodal`` returns without setting either the
    ``multimodal_processed`` (completion) or ``analyzing_stage_skipped``
    (user/config skip) sentinel — e.g. the generic ``except Exception``
    soft-swallow path or a malformed-sidecar early return — the worker must
    treat the attempt as a soft failure and leave BOTH fields absent. This
    distinguishes "analyze actually completed" from "analyze attempted but
    bailed" without falsely claiming a duration.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            original_analyze_multimodal = rag.analyze_multimodal

            async def _soft_failing_analyze(*args, **kwargs):
                result = await original_analyze_multimodal(*args, **kwargs)
                # Strip both sentinels: simulate analyze_multimodal returning
                # parsed_data after the generic except Exception soft-swallow.
                result.pop("analyzing_stage_skipped", None)
                result.pop("multimodal_processed", None)
                return result

            rag.analyze_multimodal = _soft_failing_analyze  # type: ignore[assignment]

            await rag.apipeline_enqueue_documents(
                "Soft-fail body for analyze stage.",
                file_paths="analyze_soft_fail.txt",
                track_id="track-analyze-softfail",
                process_options="iet!",
            )
            doc_id = compute_mdhash_id("analyze_soft_fail.txt", prefix="doc-")
            await rag.apipeline_process_enqueue_documents()

            final_status = await rag.doc_status.get_by_id(doc_id)
            assert final_status is not None
            assert _status_value_text(final_status.get("status")) == "processed"
            metadata = final_status.get("metadata") or {}
            assert isinstance(metadata.get("analyzing_start_time"), int)
            assert "analyzing_end_time" not in metadata, (
                f"soft-failed analyze incorrectly stamped analyzing_end_time; "
                f"final metadata: {metadata!r}"
            )
            assert "analyzing_stage_skipped" not in metadata, (
                f"soft-failed analyze incorrectly stamped analyzing_stage_skipped; "
                f"final metadata: {metadata!r}"
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_stage_end_outcomes_persist_within_their_own_stage(tmp_path):
    """The parse-stage outcome (parse_end_time / parse_stage_skipped) must be
    persisted to doc_status during the PARSING stage — before the doc waits in
    q_analyze and the ANALYZING transition fires — and the analyze-stage outcome
    during ANALYZING, before PROCESSING. Otherwise these signals only land at the
    next stage's transition via carry-over, so a doc sitting in a backlogged queue
    shows its prior status with no end-of-stage signal.

    Wraps doc_status.upsert to capture the (status, metadata-keys) sequence and
    asserts each stage-end signal appears under its own status ahead of the next
    stage's first upsert. parse_native on raw content takes the skip branches, so
    the signals here are the skipped flags rather than the *_end_time stamps.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        calls: list[tuple[str, set[str]]] = []
        try:
            original_upsert = rag.doc_status.upsert

            async def _recording_upsert(data, *args, **kwargs):
                for payload in data.values():
                    status_text = _status_value_text(payload.get("status"))
                    metadata = payload.get("metadata") or {}
                    calls.append((status_text, set(metadata.keys())))
                return await original_upsert(data, *args, **kwargs)

            rag.doc_status.upsert = _recording_upsert  # type: ignore[assignment]

            await rag.apipeline_enqueue_documents(
                "Some content body for chunking.",
                file_paths="stage_end_timing.txt",
                track_id="track-stage-end-timing",
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

        first_analyzing = next(
            (i for i, (s, _) in enumerate(calls) if s == "analyzing"), None
        )
        first_processing = next(
            (i for i, (s, _) in enumerate(calls) if s == "processing"), None
        )
        assert first_analyzing is not None, f"no ANALYZING upsert; sequence: {calls!r}"
        assert first_processing is not None, (
            f"no PROCESSING upsert; sequence: {calls!r}"
        )

        assert any(
            s == "parsing" and "parse_stage_skipped" in keys
            for s, keys in calls[:first_analyzing]
        ), (
            f"parse-stage outcome not persisted under PARSING before the "
            f"ANALYZING transition; upsert sequence: {calls!r}"
        )
        assert any(
            s == "analyzing" and "analyzing_stage_skipped" in keys
            for s, keys in calls[first_analyzing:first_processing]
        ), (
            f"analyze-stage outcome not persisted under ANALYZING before the "
            f"PROCESSING transition; upsert sequence: {calls!r}"
        )

    asyncio.run(_run())


@pytest.mark.offline
def test_apipeline_enqueue_persists_process_options(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "alpha body",
                file_paths="abc.[native-R!].docx",
                track_id="track-opts",
                process_options="R!",
            )
            doc_id = compute_mdhash_id("abc.docx", prefix="doc-")
            full_doc = await rag.full_docs.get_by_id(doc_id)
            assert full_doc is not None
            # full_docs stores the canonical (hint-stripped) basename only.
            assert full_doc["file_path"] == "abc.docx"
            assert "canonical_basename" not in full_doc
            assert full_doc.get("process_options") == "R!"

            status_doc = await rag.doc_status.get_by_id(doc_id)
            assert status_doc is not None
            metadata = (
                status_doc.get("metadata")
                if isinstance(status_doc, dict)
                else getattr(status_doc, "metadata", {})
            )
            assert metadata.get("process_options") == "R!"
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_purge_doc_chunks_and_kg_is_noop_for_empty_chunks(tmp_path):
    """``_purge_doc_chunks_and_kg`` with an empty chunk_ids list must be a
    no-op so callers (including the resume branch) can invoke it
    unconditionally without first checking for non-empty chunks_list.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )
            # Empty list: must return immediately without touching storage.
            await rag._purge_doc_chunks_and_kg(
                "doc-empty",
                [],
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )
            # No exceptions → success.  Calling twice in a row is also fine
            # since the helper is idempotent on the empty input.
            await rag._purge_doc_chunks_and_kg(
                "doc-empty",
                [],
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_purge_doc_chunks_and_kg_clears_chunks_for_unknown_doc(tmp_path):
    """When the doc has chunk_ids but no graph contributions yet
    (full_entities / full_relations empty), the helper must still clear
    the chunks from chunks_vdb / text_chunks without raising.  This
    exercises the resume path for documents whose previous run was
    interrupted between chunking and entity extraction.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Seed text_chunks + chunks_vdb with two stale chunks.
            await rag.text_chunks.upsert(
                {
                    "doc-X-chunk-0": {
                        "content": "stale chunk 0",
                        "chunk_order_index": 0,
                        "full_doc_id": "doc-X",
                        "tokens": 4,
                        "file_path": "x.txt",
                    },
                    "doc-X-chunk-1": {
                        "content": "stale chunk 1",
                        "chunk_order_index": 1,
                        "full_doc_id": "doc-X",
                        "tokens": 4,
                        "file_path": "x.txt",
                    },
                }
            )
            await rag.chunks_vdb.upsert(
                {
                    "doc-X-chunk-0": {
                        "content": "stale chunk 0",
                        "chunk_order_index": 0,
                        "full_doc_id": "doc-X",
                        "tokens": 4,
                        "file_path": "x.txt",
                    },
                    "doc-X-chunk-1": {
                        "content": "stale chunk 1",
                        "chunk_order_index": 1,
                        "full_doc_id": "doc-X",
                        "tokens": 4,
                        "file_path": "x.txt",
                    },
                }
            )
            await rag.text_chunks.index_done_callback()
            await rag.chunks_vdb.index_done_callback()

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            await rag._purge_doc_chunks_and_kg(
                "doc-X",
                ["doc-X-chunk-0", "doc-X-chunk-1"],
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )

            # Both chunks gone from text_chunks.
            remaining = await rag.text_chunks.get_by_ids(
                ["doc-X-chunk-0", "doc-X-chunk-1"]
            )
            assert remaining == [None, None]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_resume_purges_old_chunks_when_content_already_extracted(tmp_path):
    """When ``apipeline_process_enqueue_documents`` picks up a document
    whose content is already extracted (full_docs.format=raw with content)
    and whose doc_status carries a non-empty chunks_list from a previous
    half-finished run, the resume branch must call
    ``_purge_doc_chunks_and_kg`` with the old chunk-IDs *before* the
    chunking and entity-extraction stages run.  This test wraps the
    helper so we can assert it is invoked exactly once with the expected
    inputs, then bails out so we don't have to mock the whole VLM /
    entity-extract stack.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            doc_id = compute_mdhash_id("resume.txt", prefix="doc-")

            # Seed full_docs as if extraction already completed.
            await rag.full_docs.upsert(
                {
                    doc_id: {
                        "content": "previously extracted body",
                        "file_path": "resume.txt",
                        "parse_format": "raw",
                        "parse_engine": "legacy",
                        "content_hash": "deadbeef",
                    }
                }
            )
            # Seed doc_status as PROCESSING with chunks_list from a prior
            # half-finished run so the resume branch has something to purge.
            stale_chunks = [f"{doc_id}-chunk-{i:03d}" for i in range(2)]
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PROCESSING,
                        "content_summary": "previously extracted body",
                        "content_length": len("previously extracted body"),
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:01+00:00",
                        "file_path": "resume.txt",
                        "track_id": "track-resume",
                        "content_hash": "deadbeef",
                        "chunks_list": stale_chunks,
                        "chunks_count": len(stale_chunks),
                    }
                }
            )

            # Wrap the helper to record invocations, and raise after the call
            # so the test exits cleanly without exercising downstream stages.
            calls: list[tuple[str, set[str]]] = []
            original = rag._purge_doc_chunks_and_kg

            class _ResumePurged(Exception):
                pass

            async def _wrapped(doc_id_arg, chunk_ids_arg, **kwargs):
                calls.append((doc_id_arg, set(chunk_ids_arg)))
                # Run the real helper so the side-effects (chunks gone from
                # storage) are observable, then short-circuit.
                await original(doc_id_arg, chunk_ids_arg, **kwargs)
                raise _ResumePurged()

            rag._purge_doc_chunks_and_kg = _wrapped  # type: ignore[method-assign]

            # Pipeline will pick up the PROCESSING document, hit the resume
            # branch, call our wrapped purge, and our wrapper raises.
            await rag.apipeline_process_enqueue_documents()

            # Helper was invoked exactly once with the stale chunk-IDs.
            assert len(calls) == 1
            invoked_doc_id, invoked_chunks = calls[0]
            assert invoked_doc_id == doc_id
            assert invoked_chunks == set(stale_chunks)
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_resume_skips_purge_when_chunks_list_empty(tmp_path):
    """If the doc was extracted but never chunked (chunks_list empty),
    the resume branch must NOT call the purge helper — there's nothing
    to clean up.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            doc_id = compute_mdhash_id("noskip.txt", prefix="doc-")

            await rag.full_docs.upsert(
                {
                    doc_id: {
                        "content": "fresh body",
                        "file_path": "noskip.txt",
                        "parse_format": "raw",
                        "parse_engine": "legacy",
                        "content_hash": "fresh",
                    }
                }
            )
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PARSING,
                        "content_summary": "fresh body",
                        "content_length": len("fresh body"),
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:01+00:00",
                        "file_path": "noskip.txt",
                        "track_id": "track-noskip",
                        "content_hash": "fresh",
                        "chunks_list": [],
                        "chunks_count": 0,
                    }
                }
            )

            calls: list[tuple[str, set[str]]] = []

            async def _spy(doc_id_arg, chunk_ids_arg, **kwargs):
                calls.append((doc_id_arg, set(chunk_ids_arg)))
                # Don't actually purge; just record the call and let the
                # pipeline continue past this test boundary.
                raise RuntimeError("test stop after purge check")

            rag._purge_doc_chunks_and_kg = _spy  # type: ignore[method-assign]

            try:
                await rag.apipeline_process_enqueue_documents()
            except Exception:
                # Whether the pipeline reaches our spy or fails downstream
                # doesn't matter for this test; we only care that the spy
                # was NOT called for an empty chunks_list.
                pass

            assert calls == [], (
                "purge helper should not be called when chunks_list is empty"
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_apipeline_enqueue_allows_concurrent_with_busy(tmp_path):
    """``busy=True`` no longer blocks enqueue.  Concurrent processing is
    explicitly permitted — the running loop's request_pending mechanism
    picks up newly-enqueued docs after the current batch.  Enqueue
    nudges request_pending so a freshly-arrived doc is never stranded
    when the call site does not subsequently invoke
    ``apipeline_process_enqueue_documents``.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            # Simulate an in-flight indexing job.
            async with pipeline_status_lock:
                pipeline_status["busy"] = True
                pipeline_status["request_pending"] = False
            try:
                returned_track_id = await rag.apipeline_enqueue_documents(
                    "concurrent with busy",
                    file_paths="concurrent.txt",
                    track_id="track-concurrent",
                )
                assert returned_track_id == "track-concurrent"
                # Enqueue nudged the running loop.
                assert pipeline_status.get("request_pending") is True
            finally:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["request_pending"] = False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_apipeline_enqueue_rejects_when_scanning(tmp_path):
    """Scan is the only state that blocks new enqueues — scan reads
    doc_status to make classification decisions and would race with
    mid-flight writes.  The last-line guard inside
    ``apipeline_enqueue_documents`` enforces this: HTTP endpoints catch
    it earlier and return 409, but core API callers must surface the
    invariant violation as a RuntimeError.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            # Scan classification phase rejects.  ``scanning_exclusive``
            # is the field that gates the enqueue last-line guard, not
            # plain ``scanning`` (which covers the whole scan lifecycle
            # including its processing phase, where concurrent enqueue
            # is allowed).
            async with pipeline_status_lock:
                pipeline_status["scanning"] = True
                pipeline_status["scanning_exclusive"] = True
            try:
                with pytest.raises(RuntimeError, match="scan is classifying"):
                    await rag.apipeline_enqueue_documents(
                        "should not enqueue",
                        file_paths="scan.txt",
                        track_id="track-scan",
                    )
            finally:
                async with pipeline_status_lock:
                    pipeline_status["scanning"] = False
                    pipeline_status["scanning_exclusive"] = False

            # Scan processing phase (scanning=True, scanning_exclusive=False)
            # ALLOWS concurrent enqueue — same as upload-during-busy.
            async with pipeline_status_lock:
                pipeline_status["scanning"] = True
                pipeline_status["scanning_exclusive"] = False
            try:
                track_processing = await rag.apipeline_enqueue_documents(
                    "allowed during scan processing",
                    file_paths="scan_processing.txt",
                    track_id="track-scan-processing",
                )
                assert track_processing == "track-scan-processing"
            finally:
                async with pipeline_status_lock:
                    pipeline_status["scanning"] = False

            # When idle, the same call succeeds — proving the guard is the
            # only thing blocking, not some side effect of the test setup.
            await rag.apipeline_enqueue_documents(
                "now allowed",
                file_paths="ok.txt",
                track_id="track-ok",
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_enqueue_during_busy_sets_request_pending(tmp_path):
    """While the processing loop is running (busy=True), a concurrent
    enqueue must set ``request_pending`` so the loop knows to scan
    doc_status again after its current batch.  This is the mechanism
    that makes "upload while pipeline is busy" actually drain the new
    work — without it, freshly enqueued docs would be stranded until
    an unrelated trigger.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            async with pipeline_status_lock:
                pipeline_status["busy"] = True
                pipeline_status["request_pending"] = False
            try:
                # First enqueue: nudges request_pending.
                await rag.apipeline_enqueue_documents(
                    "first while busy",
                    file_paths="first.txt",
                    track_id="track-first",
                )
                assert pipeline_status.get("request_pending") is True

                # Second enqueue while busy: stays True (idempotent).
                async with pipeline_status_lock:
                    pipeline_status["request_pending"] = False
                await rag.apipeline_enqueue_documents(
                    "second while busy",
                    file_paths="second.txt",
                    track_id="track-second",
                )
                assert pipeline_status.get("request_pending") is True
            finally:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["request_pending"] = False

            # When idle, enqueue does NOT set request_pending — there is
            # no running loop to nudge.
            await rag.apipeline_enqueue_documents(
                "while idle",
                file_paths="idle.txt",
                track_id="track-idle",
            )
            assert pipeline_status.get("request_pending") is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_atomic_release_busy_or_consume_pending(tmp_path):
    """The loop-exit handoff is atomic via
    ``_atomic_release_busy_or_consume_pending``: the same critical
    section that reads ``request_pending`` also writes ``busy=False``.

    This closes the race where a concurrent enqueue could set
    ``request_pending=True`` between the loop's read of the flag and
    the finally block's ``busy=False`` write — leaving newly-enqueued
    docs stranded in PENDING with no running loop to consume them.

    The helper has two outcomes:
      * ``request_pending=True`` at entry → flag cleared, return False
        (caller must continue the loop, refetching doc_status).
      * ``request_pending=False`` at entry → ``busy`` cleared, return
        True (caller must break out without re-clearing busy).

    Tested directly because the closure pattern inside
    ``apipeline_process_enqueue_documents`` is otherwise hard to
    exercise from a unit test without orchestrating real concurrency.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            # Case 1: simulate the race — request_pending was set by a
            # concurrent enqueue while busy=True.  Helper must consume
            # the flag and return False (continue loop) rather than
            # silently exit.
            async with pipeline_status_lock:
                pipeline_status["busy"] = True
                pipeline_status["request_pending"] = True
            released = await rag._atomic_release_busy_or_consume_pending(
                pipeline_status, pipeline_status_lock
            )
            assert released is False
            assert pipeline_status["busy"] is True  # NOT released
            assert pipeline_status["request_pending"] is False  # consumed

            # Case 2: clean exit path — no concurrent enqueue.  Helper
            # releases busy under the SAME lock so any post-call
            # enqueue can either see busy=False (and trigger its own
            # process pass) or had to set request_pending BEFORE this
            # call (handled by Case 1).  No stranded flag possible.
            async with pipeline_status_lock:
                pipeline_status["busy"] = True
                pipeline_status["request_pending"] = False
            released = await rag._atomic_release_busy_or_consume_pending(
                pipeline_status, pipeline_status_lock
            )
            assert released is True
            assert pipeline_status["busy"] is False  # released
            assert pipeline_status["request_pending"] is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_apipeline_enqueue_rejects_when_destructive_busy(tmp_path):
    """``destructive_busy`` (set by /documents/clear and per-doc delete)
    must reject enqueue at the last-line guard.  These jobs DROP
    storages and remove input files; concurrent enqueue would write to
    storages mid-drop and silently lose the document.  Note: this is
    different from plain ``busy=True`` (the processing loop), which is
    explicitly compatible with concurrent enqueue.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            async with pipeline_status_lock:
                pipeline_status["busy"] = True
                pipeline_status["destructive_busy"] = True
            try:
                with pytest.raises(RuntimeError, match="clearing or deleting"):
                    await rag.apipeline_enqueue_documents(
                        "should not enqueue",
                        file_paths="while_clearing.txt",
                        track_id="track-clearing",
                    )
                # ``from_scan`` does NOT bypass destructive_busy: scan
                # is also a writer and would race with the drop.
                with pytest.raises(RuntimeError, match="clearing or deleting"):
                    await rag.apipeline_enqueue_documents(
                        "should not enqueue",
                        file_paths="while_clearing_scan.txt",
                        track_id="track-clearing-scan",
                        from_scan=True,
                    )
            finally:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["destructive_busy"] = False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_concurrent_enqueue_dedupes_same_content_different_filenames(tmp_path):
    """Two concurrent ``apipeline_enqueue_documents`` calls with the
    same content but different filenames must not both end up as
    PENDING.  The dedup-and-upsert critical section is serialised by
    a workspace-scoped lock so the second call always sees the first's
    upserted row and is recorded as ``duplicate_kind=content_hash``.

    The race only matters now that concurrent enqueue is permitted
    (busy=True doesn't block, scan's processing phase doesn't block).
    Without the lock, two enqueues can both read doc_status before
    either upserts, both miss the content_hash dedup, and both write
    PENDING — bypassing the dedup that's supposed to land one of them
    as FAILED.

    Determinism trick: patch ``get_existing_doc_by_content_hash`` to
    yield via ``asyncio.sleep(0)`` before reading.  This guarantees the
    asyncio scheduler interleaves the two coroutines at the dedup
    read, so without the serialise lock both would miss the existing
    row.  With the lock, the second coroutine waits until the first
    has finished upserting, then sees the row.
    """

    async def _run():
        import lightrag.pipeline as pipeline_module

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            original = pipeline_module.get_existing_doc_by_content_hash

            async def yielding_get_by_content_hash(doc_status, content_hash):
                # Yield to the event loop so the SECOND enqueue gets a
                # chance to run its dedup read before we proceed.  This
                # is the exact interleaving the lock must defeat.
                await asyncio.sleep(0)
                return await original(doc_status, content_hash)

            import unittest.mock

            with unittest.mock.patch.object(
                pipeline_module,
                "get_existing_doc_by_content_hash",
                yielding_get_by_content_hash,
            ):
                # Same content, two distinct filenames so the basename
                # dedup misses and the content_hash dedup is the gate.
                shared_content = "shared content for dedup race"
                results = await asyncio.gather(
                    rag.apipeline_enqueue_documents(
                        shared_content,
                        file_paths="first.txt",
                        track_id="track-first",
                    ),
                    rag.apipeline_enqueue_documents(
                        shared_content,
                        file_paths="second.txt",
                        track_id="track-second",
                    ),
                )
                # First call enqueues the doc and returns its track_id.
                # Second call sees the upserted row inside the
                # serialised dedup section, finds zero unique docs, and
                # returns None (the existing "no new unique docs"
                # early-exit path).  The duplicate record is still
                # written to doc_status as FAILED.
                assert results[0] == "track-first"
                assert results[1] is None

            # Exactly ONE PENDING doc should exist for this content,
            # not two.  The second enqueue must have been recorded as a
            # content_hash duplicate (FAILED with metadata).
            pending_docs = await rag.doc_status.get_docs_by_statuses(
                [DocStatus.PENDING]
            )
            assert len(pending_docs) == 1, (
                f"Expected exactly 1 PENDING doc after concurrent enqueue, "
                f"got {len(pending_docs)}: {list(pending_docs.keys())}"
            )

            # The duplicate record (FAILED + duplicate_kind=content_hash)
            # carries the second filename and the metadata pointer back
            # to the original.
            failed_docs = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
            duplicate_records = [
                d
                for d in failed_docs.values()
                if (
                    getattr(d, "metadata", None)
                    and d.metadata.get("duplicate_kind") == "content_hash"
                )
            ]
            assert len(duplicate_records) == 1, (
                f"Expected exactly 1 content_hash-duplicate FAILED row, "
                f"got {len(duplicate_records)}"
            )
            dup = duplicate_records[0]
            assert dup.metadata["is_duplicate"] is True
            assert dup.metadata["original_doc_id"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_apipeline_enqueue_from_scan_bypasses_scanning_guard(tmp_path):
    """The scan-owned background task sets ``scanning=True`` itself, so its
    own enqueue calls must be allowed through.  External callers (without
    ``from_scan=True``) remain blocked.  ``busy=True`` no longer rejects
    enqueue (concurrent processing is permitted under the new contract),
    so it is not exercised here.
    """

    async def _run():
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            # Scan classification phase: scanning_exclusive=True, but
            # ``from_scan=True`` lifts the guard so scan can enqueue
            # files it just discovered.  Non-scan callers are still
            # rejected.
            async with pipeline_status_lock:
                pipeline_status["scanning"] = True
                pipeline_status["scanning_exclusive"] = True
            try:
                returned_track_id = await rag.apipeline_enqueue_documents(
                    "scan-owned content",
                    file_paths="scan_owned.txt",
                    track_id="track-scan-owned",
                    from_scan=True,
                )
                assert returned_track_id == "track-scan-owned"

                with pytest.raises(RuntimeError, match="scan is classifying"):
                    await rag.apipeline_enqueue_documents(
                        "external content",
                        file_paths="external.txt",
                        track_id="track-external",
                    )
            finally:
                async with pipeline_status_lock:
                    pipeline_status["scanning"] = False
                    pipeline_status["scanning_exclusive"] = False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_multimodal_overwrites_already_analyzed_items(tmp_path):
    """Re-running analyze_multimodal recomputes enabled modalities and
    overwrites any prior ``llm_analyze_result`` from the sidecar.
    """

    async def _run():
        vlm_calls = {"n": 0}
        extract_calls = {"n": 0}

        async def _vlm(_prompt, **_kwargs):
            vlm_calls["n"] += 1
            return json.dumps(
                {
                    "name": "Image",
                    "type": "Chart",
                    "description": "details",
                }
            )

        async def _extract(_prompt, **_kwargs):
            extract_calls["n"] += 1
            return json.dumps(
                {
                    "name": "Item",
                    "description": "table content summary",
                }
            )

        rag = _new_rag(
            tmp_path,
            vlm_llm_model_func=_vlm,
            extract_llm_model_func=_extract,
        )
        await rag.initialize_storages()

        # Minimal blocks file with valid meta.
        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps({"type": "meta", "format_version": "1.0"}),
                    json.dumps({"type": "content", "content": "body"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        # 64x64 PNG so the image-pixel skip guard does NOT short-circuit
        # before the VLM call.
        img_path = tmp_path / "img1.png"
        import struct
        import zlib

        def _png_bytes(w: int, h: int) -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
            crc = zlib.crc32(b"IHDR" + ihdr).to_bytes(4, "big")
            ihdr_chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + crc
            idat_payload = b"\x00" * (w * h * 4 + h)
            compressed = zlib.compress(idat_payload)
            crc_idat = zlib.crc32(b"IDAT" + compressed).to_bytes(4, "big")
            idat_chunk = (
                struct.pack(">I", len(compressed)) + b"IDAT" + compressed + crc_idat
            )
            iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"
            return sig + ihdr_chunk + idat_chunk + iend_chunk

        img_path.write_bytes(_png_bytes(64, 64))

        # Drawings sidecar with ONE item already analyzed (status=success).
        drawings = tmp_path / "demo.drawings.json"
        drawings.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "drawings": {
                        "id1": {
                            "id": "id1",
                            "caption": "fig1",
                            "path": str(img_path),
                            "llm_analyze_result": {
                                "name": "Existing",
                                "type": "Photo",
                                "description": "kept as-is",
                                "analyze_time": 1700000000,
                                "status": "success",
                                "message": "",
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        # Tables sidecar with one fresh item (no prior result).
        tables = tmp_path / "demo.tables.json"
        tables.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "tables": {
                        "tbl1": {
                            "id": "tbl1",
                            "caption": "tbl",
                            "format": "html",
                            "content": "Header|Row",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed, process_options="it")

        drawings_payload = json.loads(drawings.read_text(encoding="utf-8"))
        existing = drawings_payload["drawings"]["id1"]["llm_analyze_result"]
        # Existing result was overwritten by the new VLM result.
        assert existing["name"] == "Image"
        assert existing["description"] == "details"
        assert existing["status"] == "success"

        tables_payload = json.loads(tables.read_text(encoding="utf-8"))
        new_result = tables_payload["tables"]["tbl1"]["llm_analyze_result"]
        assert new_result["name"] == "Item"
        assert new_result["status"] == "success"

        # Drawings are recomputed through VLM; tables take the EXTRACT role
        # (per design §3.1), not VLM.
        assert vlm_calls["n"] == 1
        assert extract_calls["n"] == 1

    asyncio.run(_run())


@pytest.mark.offline
def test_enqueue_dedupes_by_filename_and_content_hash(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Distinct filenames with distinct content both get enqueued.
            await rag.apipeline_enqueue_documents(
                ["alpha body", "beta body"],
                file_paths=["first.txt", "second.txt"],
                track_id="track-a",
            )
            first_id = compute_mdhash_id("first.txt", prefix="doc-")
            second_id = compute_mdhash_id("second.txt", prefix="doc-")
            first_doc = await rag.full_docs.get_by_id(first_id)
            second_doc = await rag.full_docs.get_by_id(second_id)
            assert first_doc is not None
            assert second_doc is not None
            assert first_doc.get("content_hash")
            assert second_doc.get("content_hash")
            assert first_doc["content_hash"] != second_doc["content_hash"]

            # Same filename basename with new content is rejected (filename dedup).
            await rag.apipeline_enqueue_documents(
                "changed content",
                file_paths="/tmp/first.txt",
                track_id="track-b",
            )
            first_doc = await rag.full_docs.get_by_id(first_id)
            assert first_doc["content"] == "alpha body"

            # New filename but same content as an existing doc is rejected
            # (content_hash dedup).
            await rag.apipeline_enqueue_documents(
                "alpha body",
                file_paths="third.txt",
                track_id="track-c",
            )
            third_id = compute_mdhash_id("third.txt", prefix="doc-")
            assert await rag.full_docs.get_by_id(third_id) is None

            failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            kinds = {
                getattr(doc, "metadata", {}).get("duplicate_kind")
                for doc in failed_docs.values()
                if getattr(doc, "metadata", {}).get("is_duplicate")
            }
            assert {"filename", "content_hash"}.issubset(kinds)
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_enqueue_dedupes_parser_hinted_filename_variants(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "alpha body",
                file_paths="abc.docx",
                track_id="track-a",
            )
            first_id = compute_mdhash_id("abc.docx", prefix="doc-")
            first_doc = await rag.full_docs.get_by_id(first_id)
            assert first_doc is not None
            # ``file_path`` is the canonical (hint-stripped) basename and
            # serves as the dedup key — no separate ``canonical_basename``
            # field is written.
            assert first_doc["file_path"] == "abc.docx"
            assert "canonical_basename" not in first_doc

            await rag.apipeline_enqueue_documents(
                "changed body",
                file_paths="/tmp/abc.[native].docx",
                track_id="track-b",
            )
            assert (await rag.full_docs.get_by_id(first_id))["content"] == "alpha body"

            failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            # The duplicate record stores the canonical basename — hint is
            # not preserved anywhere in the new schema.
            assert any(
                getattr(doc, "metadata", {}).get("duplicate_kind") == "filename"
                and getattr(doc, "file_path", "") == "abc.docx"
                for doc in failed_docs.values()
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_delete_result_uses_canonical_file_path(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "",
                file_paths=str(tmp_path / "abc.[native].docx"),
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine=PARSER_ENGINE_NATIVE,
                track_id="track-delete-source",
            )

            doc_id = compute_mdhash_id("abc.docx", prefix="doc-")
            result = await rag.adelete_by_doc_id(doc_id)

            assert result.status == "success"
            # New schema: file_path is the canonical (hint-stripped)
            # basename; the ``source_path`` field is no longer carried on
            # DeletionResult.
            assert result.file_path == "abc.docx"
            assert not hasattr(result, "source_path")
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_enqueue_without_file_paths_uses_content_ids(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            docs = ["alpha without source", "beta without source"]
            await rag.apipeline_enqueue_documents(docs, track_id="track-no-source")

            for content in docs:
                doc_id = compute_mdhash_id(content, prefix="doc-")
                full_doc = await rag.full_docs.get_by_id(doc_id)
                status = await rag.doc_status.get_by_id(doc_id)
                assert full_doc is not None
                assert status is not None
                assert full_doc["content"] == content
                assert full_doc["file_path"] == "unknown_source"
                assert status["file_path"] == "unknown_source"
                assert full_doc.get("content_hash")
                assert status.get("content_hash") == full_doc.get("content_hash")

            failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            duplicate_failures = [
                doc
                for doc in failed_docs.values()
                if getattr(doc, "track_id", "") == "track-no-source"
                and getattr(doc, "metadata", {}).get("is_duplicate")
            ]
            assert duplicate_failures == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_legacy_empty_file_paths_do_not_block_unsourced_insert(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.doc_status.upsert(
                {
                    "legacy-empty": {
                        "status": DocStatus.PROCESSED,
                        "content_summary": "legacy empty",
                        "content_length": 0,
                        "file_path": "",
                        "track_id": "legacy",
                        "created_at": "2025-01-01T00:00:00+00:00",
                        "updated_at": "2025-01-01T00:00:00+00:00",
                        "chunks_list": [],
                    },
                    "legacy-no-file": {
                        "status": DocStatus.PROCESSED,
                        "content_summary": "legacy no-file",
                        "content_length": 0,
                        "file_path": "no-file-path",
                        "track_id": "legacy",
                        "created_at": "2025-01-01T00:00:00+00:00",
                        "updated_at": "2025-01-01T00:00:00+00:00",
                        "chunks_list": [],
                    },
                }
            )

            content = "fresh unsourced body"
            await rag.apipeline_enqueue_documents(content, track_id="track-fresh")
            doc_id = compute_mdhash_id(content, prefix="doc-")
            full_doc = await rag.full_docs.get_by_id(doc_id)
            status = await rag.doc_status.get_by_id(doc_id)
            assert full_doc is not None
            assert status is not None
            assert full_doc["file_path"] == "unknown_source"
            assert status["file_path"] == "unknown_source"
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_basename_lookup_requires_canonical_stored_file_path(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Storage does not normalize or apply legacy basename fallback.
            # Business-layer writes must persist the canonical basename.
            noncanonical_id = "doc-noncanonical-1"
            await rag.doc_status.upsert(
                {
                    noncanonical_id: {
                        "status": DocStatus.PROCESSED,
                        "content_summary": "noncanonical",
                        "content_length": 7,
                        "file_path": "/inputs/legacy.txt",
                        "track_id": "noncanonical-track",
                        "created_at": "2025-01-01T00:00:00+00:00",
                        "updated_at": "2025-01-01T00:00:00+00:00",
                        "chunks_list": [],
                    }
                }
            )

            match = await rag.doc_status.get_doc_by_file_basename("legacy.txt")
            assert match is None

            # Re-enqueueing through the business path stores the canonical
            # basename and is not blocked by a noncanonical storage row.
            await rag.apipeline_enqueue_documents(
                "fresh body",
                file_paths="legacy.txt",
                track_id="track-x",
            )
            new_id = compute_mdhash_id("legacy.txt", prefix="doc-")
            full_doc = await rag.full_docs.get_by_id(new_id)
            status = await rag.doc_status.get_by_id(new_id)
            assert full_doc is not None
            assert status is not None
            assert full_doc["file_path"] == "legacy.txt"
            assert status["file_path"] == "legacy.txt"
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_content_hash_lookup_via_storage(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "shared body",
                file_paths="alpha.txt",
                track_id="track-a",
            )
            alpha_id = compute_mdhash_id("alpha.txt", prefix="doc-")
            alpha_full = await rag.full_docs.get_by_id(alpha_id)
            assert alpha_full is not None
            content_hash = alpha_full["content_hash"]

            match = await rag.doc_status.get_doc_by_content_hash(content_hash)
            assert match is not None
            doc_id, _ = match
            assert doc_id == alpha_id
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_enqueue_rejects_removed_or_unknown_docs_format(tmp_path):
    """The 'lightrag' ingestion entrypoint was removed: enqueue accepts only
    raw / pending_parse and raises explicitly for anything else (previously
    an unknown value was silently treated as raw)."""

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            with pytest.raises(ValueError, match="Unsupported docs_format"):
                await rag.apipeline_enqueue_documents(
                    "",
                    file_paths="first.lightrag",
                    docs_format="lightrag",
                )
            with pytest.raises(ValueError, match="Unsupported docs_format"):
                await rag.apipeline_enqueue_documents(
                    "some content",
                    file_paths="doc.txt",
                    docs_format="bogus",
                )
            # The companion parameter is gone entirely (no compat shim).
            with pytest.raises(TypeError):
                await rag.apipeline_enqueue_documents(  # type: ignore[call-arg]
                    "",
                    file_paths="first.lightrag",
                    lightrag_document_paths="__parsed__/doc.blocks.jsonl",
                )
            # Nothing was enqueued by the rejected calls.
            failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            assert failed == {}
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_persist_parsed_full_docs_syncs_hash_to_doc_status(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Enqueue a pending_parse record: no content_hash should exist yet.
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="pending.txt",
                docs_format="pending_parse",
                track_id="track-pending",
            )
            doc_id = compute_mdhash_id("pending.txt", prefix="doc-")
            full_doc = await rag.full_docs.get_by_id(doc_id)
            assert full_doc is not None
            assert full_doc.get("content_hash") in (None, "")
            status_pre = await rag.doc_status.get_by_id(doc_id)
            assert (status_pre or {}).get("content_hash") in (None, "")

            # Simulate a parse_* completion that converts the record to RAW.
            content = "extracted body text"
            await rag._persist_parsed_full_docs(
                doc_id,
                {
                    "content": content,
                    "file_path": "pending.txt",
                    "parse_format": "raw",
                    "parse_engine": "native",
                },
            )

            full_doc = await rag.full_docs.get_by_id(doc_id)
            status_post = await rag.doc_status.get_by_id(doc_id)
            expected_hash = compute_mdhash_id(content, prefix="")
            assert full_doc["content_hash"] == expected_hash
            assert (status_post or {}).get("content_hash") == expected_hash

            # The hash should be queryable via the storage index.
            match = await rag.doc_status.get_doc_by_content_hash(expected_hash)
            assert match is not None
            assert match[0] == doc_id
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_persist_parsed_full_docs_preserves_pending_metadata(tmp_path):
    """``_persist_parsed_full_docs`` must keep process_options seeded at
    enqueue time so downstream stages (analyze_multimodal, chunking
    selection, KG-skip) still see the user's original opt-ins after the
    parse-result record overwrites the pending_parse row.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="report.[native-iet!].docx",
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine=PARSER_ENGINE_NATIVE,
                process_options="iet!",
                track_id="track-merge",
            )
            doc_id = compute_mdhash_id("report.docx", prefix="doc-")

            pre = await rag.full_docs.get_by_id(doc_id)
            assert pre is not None
            assert pre.get("process_options") == "iet!"
            assert "canonical_basename" not in pre
            assert pre.get("file_path") == "report.docx"

            # Simulate a parse_* completion: pass only the fresh fields the
            # parsers actually emit and verify that pre-existing metadata
            # survives the upsert.
            await rag._persist_parsed_full_docs(
                doc_id,
                {
                    "content": "extracted body",
                    "file_path": "report.docx",
                    "parse_format": "raw",
                    "parse_engine": PARSER_ENGINE_NATIVE,
                    "update_time": 12345,
                },
            )

            post = await rag.full_docs.get_by_id(doc_id)
            assert post is not None
            # Parser-supplied fields take precedence...
            assert post["content"] == "extracted body"
            assert post["parse_format"] == "raw"
            # ...while metadata seeded at enqueue time is preserved.
            assert post.get("process_options") == "iet!"
            assert post.get("file_path") == "report.docx"
            # And content_hash is freshly computed from the parsed body.
            assert post["content_hash"] == compute_mdhash_id(
                "extracted body", prefix=""
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_state_machine_upsert_preserves_content_hash(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "raw body",
                file_paths="alpha.txt",
                track_id="track-state",
            )
            doc_id = compute_mdhash_id("alpha.txt", prefix="doc-")
            initial_hash = (await rag.doc_status.get_by_id(doc_id))["content_hash"]
            assert initial_hash

            # Simulate the production state-machine upsert pattern: read
            # status_doc, then write a new payload that includes content_hash.
            status_doc = (await rag.doc_status.get_docs_by_status(DocStatus.PENDING))[
                doc_id
            ]
            for next_status in (
                DocStatus.PARSING,
                DocStatus.ANALYZING,
                DocStatus.PROCESSED,
            ):
                await rag.doc_status.upsert(
                    {
                        doc_id: {
                            "status": next_status,
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                            "updated_at": "now",
                            "file_path": status_doc.file_path,
                            "track_id": status_doc.track_id,
                            "content_hash": status_doc.content_hash,
                        }
                    }
                )
                current = await rag.doc_status.get_by_id(doc_id)
                assert current.get("content_hash") == initial_hash
                # And the index lookup must still return this doc.
                match = await rag.doc_status.get_doc_by_content_hash(initial_hash)
                assert match is not None and match[0] == doc_id
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_pending_parse_duplicate_hash_fails_and_archives_source(tmp_path, monkeypatch):
    """Two PENDING_PARSE docx files with identical extracted bodies must be
    detected as content_hash duplicates and the loser archived.

    ``content_hash`` for LIGHTRAG-format docs is the MD5 of the normalized
    ``merged_text`` (sidecar item ids and ``<base>.blocks.assets/`` prefixes
    stripped via :func:`normalize_merged_text_for_hash`), so identical
    bodies under different filenames produce the same hash and
    ``_mark_duplicate_after_parse`` fires.
    """

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()
        try:
            from datetime import datetime, timezone

            import lightrag.lightrag as lightrag_module
            import lightrag.pipeline as pipeline_module

            class _FrozenDateTime(datetime):
                @classmethod
                def now(cls, tz=None):  # noqa: D401
                    return datetime(2026, 1, 1, tzinfo=tz or timezone.utc)

            monkeypatch.setattr(lightrag_module, "datetime", _FrozenDateTime)
            monkeypatch.setattr(pipeline_module, "datetime", _FrozenDateTime)

            # Both docx files emit the same blocks list, so combined with the
            # frozen datetime the resulting .blocks.jsonl bytes are equal.
            stable_block = {
                "uuid": "p1",
                "uuid_end": "p1",
                "heading": "",
                "content": "same extracted body",
                "type": "text",
                "parent_headings": [],
                "level": 0,
                "table_chunk_role": "none",
            }
            monkeypatch.setattr(
                "lightrag.parser.docx.parse_document.extract_docx_blocks",
                lambda *args, **kwargs: [dict(stable_block)],
            )

            # First original docx: enqueue, parse, mark PROCESSED.
            original_path = input_dir / "original.docx"
            original_path.write_bytes(b"original docx bytes")
            await rag.apipeline_enqueue_documents(
                "",
                file_paths=str(original_path),
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine=PARSER_ENGINE_NATIVE,
                track_id="track-original",
            )
            await rag.apipeline_process_enqueue_documents()
            original_id = compute_mdhash_id("original.docx", prefix="doc-")
            original_status = await rag.doc_status.get_by_id(original_id)
            assert original_status is not None
            original_status["status"] = DocStatus.PROCESSED
            await rag.doc_status.upsert({original_id: original_status})

            # Second docx: distinct filename so filename dedup misses, but
            # content_hash should match the first because content_list +
            # frozen datetime → identical .blocks.jsonl bytes.
            source_path = input_dir / "duplicate.docx"
            source_path.write_bytes(b"docx bytes")

            async def _fail_extract(*args, **kwargs):
                raise AssertionError("duplicate document should not reach extraction")

            monkeypatch.setattr(rag, "_process_extract_entities", _fail_extract)

            await rag.apipeline_enqueue_documents(
                "",
                file_paths=str(source_path),
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine=PARSER_ENGINE_NATIVE,
                track_id="track-dup",
            )
            await rag.apipeline_process_enqueue_documents()

            duplicate_id = compute_mdhash_id("duplicate.docx", prefix="doc-")
            duplicate_status = await rag.doc_status.get_by_id(duplicate_id)
            assert duplicate_status["status"] == DocStatus.FAILED
            assert duplicate_status["metadata"]["is_duplicate"] is True
            assert duplicate_status["metadata"]["duplicate_kind"] == "content_hash"
            assert duplicate_status["metadata"]["original_doc_id"] == original_id
            assert not source_path.exists()
            assert (input_dir / PARSED_DIR_NAME / source_path.name).exists()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parser_routing_accepts_semicolon_rules(monkeypatch):
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")

    rules = "*:mineru;html:docling"
    validate_parser_routing_config(rules)
    assert resolve_file_parser_engine("paper.pdf", parser_rules=rules) == "mineru"
    assert resolve_file_parser_engine("index.html", parser_rules=rules) == "docling"
    assert resolve_file_parser_engine("notes.txt", parser_rules=rules) == "legacy"


@pytest.mark.offline
def test_parser_routing_validation_requires_external_endpoints(monkeypatch):
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")

    with pytest.raises(ParserRoutingConfigError, match="MINERU_LOCAL_ENDPOINT"):
        validate_parser_routing_config("*:mineru;html:docling")

    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    with pytest.raises(ParserRoutingConfigError, match="DOCLING_ENDPOINT"):
        validate_parser_routing_config("*:mineru;html:docling")


@pytest.mark.offline
def test_parser_routing_validation_honors_mineru_api_mode(monkeypatch):
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")

    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    with pytest.raises(ParserRoutingConfigError, match="MINERU_API_TOKEN"):
        validate_parser_routing_config("pdf:mineru")
    monkeypatch.setenv("MINERU_API_TOKEN", "token")
    validate_parser_routing_config("pdf:mineru")

    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    with pytest.raises(ParserRoutingConfigError, match="MINERU_LOCAL_ENDPOINT"):
        validate_parser_routing_config("pdf:mineru")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-local")
    validate_parser_routing_config("pdf:mineru")


@pytest.mark.offline
def test_parser_routing_validation_rejects_invalid_rules(monkeypatch):
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

    with pytest.raises(ParserRoutingConfigError, match=r"\*\.pdf"):
        validate_parser_routing_config("*.pdf:mineru")

    with pytest.raises(ParserRoutingConfigError, match="unsupported parser engine"):
        validate_parser_routing_config("pdf:unknown")

    with pytest.raises(ParserRoutingConfigError, match="does not match any suffix"):
        validate_parser_routing_config("pdf:native")


@pytest.mark.offline
def test_three_phase_status_flow(tmp_path, monkeypatch):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        async def _fake_extract(*args, **kwargs):
            return []

        async def _fake_merge(*args, **kwargs):
            return None

        async def _fake_analyze(doc_id, file_path, parsed_data, **kwargs):
            parsed_data["multimodal_processed"] = True
            return parsed_data

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)
        monkeypatch.setattr("lightrag.pipeline.merge_nodes_and_edges", _fake_merge)
        # "sample text" enqueues as RAW; the worker dispatches it to the
        # PassthroughParser (no parse_* wrapper involved), so no parse stub is
        # needed — the status-flow assertions below exercise the real path.
        monkeypatch.setattr(rag, "analyze_multimodal", _fake_analyze)

        status_seq: list[str] = []
        original_upsert = rag.doc_status.upsert

        async def _record_upsert(data):
            for _, val in data.items():
                if isinstance(val, dict) and "status" in val:
                    status_seq.append(str(val["status"]))
            return await original_upsert(data)

        monkeypatch.setattr(rag.doc_status, "upsert", _record_upsert)

        await rag.apipeline_enqueue_documents("sample text", file_paths="s.txt")
        await rag.apipeline_process_enqueue_documents()

        joined = " ".join(status_seq)
        assert "DocStatus.PARSING" in joined
        assert "DocStatus.ANALYZING" in joined
        assert "DocStatus.PROCESSING" in joined
        assert "DocStatus.PROCESSED" in joined

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_multimodal_invalid_json_hard_fails(tmp_path):
    """An invalid VLM response is a hard failure under the new contract:
    the sidecar item gets status='failure' and MultimodalAnalysisError
    bubbles up so the document fails (no silent conservative fallback)."""

    async def _run():
        calls = {"n": 0}

        async def _broken_vlm(prompt, **kwargs):
            calls["n"] += 1
            return "not-json"

        rag = _new_rag(tmp_path, vlm_llm_model_func=_broken_vlm)
        await rag.initialize_storages()
        # 64x64 PNG so the image-pixel skip guard does NOT short-circuit
        # before the VLM call.
        img_path = tmp_path / "img1.png"
        import struct
        import zlib

        def _png_bytes(w: int, h: int) -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
            crc = zlib.crc32(b"IHDR" + ihdr).to_bytes(4, "big")
            ihdr_chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + crc
            idat_payload = b"\x00" * (w * h * 4 + h)
            compressed = zlib.compress(idat_payload)
            crc_idat = zlib.crc32(b"IDAT" + compressed).to_bytes(4, "big")
            idat_chunk = (
                struct.pack(">I", len(compressed)) + b"IDAT" + compressed + crc_idat
            )
            iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"
            return sig + ihdr_chunk + idat_chunk + iend_chunk

        img_path.write_bytes(_png_bytes(64, 64))

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps({"type": "meta", "format_version": "1.0"}),
                    json.dumps({"type": "content", "content": "body"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        drawings = tmp_path / "demo.drawings.json"
        drawings.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "drawings": {
                        "id1": {
                            "id": "id1",
                            "caption": "图1 测试图",
                            "footnotes": [],
                            "path": str(img_path),
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        from lightrag.exceptions import MultimodalAnalysisError

        with pytest.raises(MultimodalAnalysisError):
            await rag.analyze_multimodal(
                "doc-1", "demo.pdf", parsed, process_options="i"
            )

        drawings_payload = json.loads(drawings.read_text(encoding="utf-8"))
        result = drawings_payload["drawings"]["id1"]["llm_analyze_result"]
        # One JSON conformance retry, then the hard failure surfaces.
        assert calls["n"] == 2
        # Sidecar carries a failure marker so a re-run sees the prior failure
        # and does not silently consume it.
        assert result["status"] == "failure"
        assert "missing or invalid field" in result["message"]

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_multimodal_uses_effective_vlm_max_async_when_role_none(tmp_path):
    async def _run():
        rag = _new_rag(
            tmp_path,
            llm_model_max_async=3,
            vlm_llm_model_max_async=None,
        )
        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            json.dumps({"type": "meta", "format_version": "1.0"}) + "\n",
            encoding="utf-8",
        )

        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        result = await rag.analyze_multimodal(
            "doc-1", "demo.pdf", parsed, process_options="ite"
        )

        assert result["multimodal_processed"] is True

    asyncio.run(_run())


@pytest.mark.offline
def test_safe_vdb_operation_times_out_with_context():
    async def _run():
        async def _hang():
            await asyncio.sleep(0.2)

        with pytest.raises(TimeoutError) as exc_info:
            await safe_vdb_operation_with_exception(
                operation=_hang,
                operation_name="relationship_upsert",
                entity_name="A->B",
                max_retries=1,
                retry_delay=0,
                timeout_seconds=0.05,
            )

        assert "relationship_upsert" in str(exc_info.value)
        assert "A->B" in str(exc_info.value)
        assert "timeout" in str(exc_info.value).lower()

    asyncio.run(_run())


@pytest.mark.offline
def test_relationship_vdb_timeout_has_120s_floor():
    assert _get_relationship_vdb_timeout_seconds({}) == 120.0
    assert (
        _get_relationship_vdb_timeout_seconds({"default_embedding_timeout": 10})
        == 120.0
    )
    assert (
        _get_relationship_vdb_timeout_seconds({"default_embedding_timeout": 50})
        == 150.0
    )


@pytest.mark.offline
def test_analyze_multimodal_unknown_image_type_folds_to_other(tmp_path):
    """Model output with an out-of-enum ``type`` is folded to ``Other``
    instead of failing the document (per design §3.4)."""

    async def _run():
        async def _vlm(_prompt, **_kwargs):
            return json.dumps(
                {
                    "name": "Figure A",
                    "type": "diagram",  # not in IMAGE_TYPE_ENUM
                    "description": "details",
                },
                ensure_ascii=False,
            )

        rag = _new_rag(tmp_path, vlm_llm_model_func=_vlm)
        await rag.initialize_storages()
        import struct
        import zlib

        def _png(w, h):
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
            crc = zlib.crc32(b"IHDR" + ihdr).to_bytes(4, "big")
            return sig + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + crc

        img_path = tmp_path / "img1.png"
        img_path.write_bytes(_png(64, 64))

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps({"type": "meta", "format_version": "1.0"}),
                    json.dumps({"type": "content", "content": "body"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        drawings = tmp_path / "demo.drawings.json"
        drawings.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "drawings": {
                        "id1": {
                            "id": "id1",
                            "caption": "图1 测试图",
                            "footnotes": [],
                            "path": str(img_path),
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed, process_options="i")

        payload = json.loads(drawings.read_text(encoding="utf-8"))
        result = payload["drawings"]["id1"]["llm_analyze_result"]
        assert result["status"] == "success"
        assert result["type"] == "Other"
        assert result["description"] == "details"
        assert "analyze_time" in result

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_multimodal_skips_tiny_image_without_vlm_call(tmp_path):
    """Images smaller than VLM_MIN_IMAGE_PIXEL (default 32px) are flagged
    status=skipped without invoking the VLM."""

    async def _run():
        calls = {"n": 0}

        async def _vlm(_prompt, **_kwargs):
            calls["n"] += 1
            return "{}"

        rag = _new_rag(tmp_path, vlm_llm_model_func=_vlm)
        await rag.initialize_storages()
        # 1x1 PNG.
        img_path = tmp_path / "tiny.png"
        img_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00"
            b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps({"type": "meta", "format_version": "1.0"}),
                    json.dumps({"type": "content", "content": "body"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        drawings = tmp_path / "demo.drawings.json"
        drawings.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "drawings": {
                        "id1": {
                            "id": "id1",
                            "caption": "tiny",
                            "path": str(img_path),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed, process_options="i")

        payload = json.loads(drawings.read_text(encoding="utf-8"))
        result = payload["drawings"]["id1"]["llm_analyze_result"]
        assert result["status"] == "skipped"
        assert "smaller than" in result["message"]
        assert calls["n"] == 0

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_multimodal_table_without_image_uses_textual_analysis(tmp_path):
    async def _run():
        # Tables now route to the EXTRACT role, not VLM (per design §3.1).
        async def _extract(_prompt, **_kwargs):
            return json.dumps(
                {
                    "name": "model_benchmark_metrics",
                    "description": "表格包含三列，分别为符号、代表意义和单位，列出了 A、F、e 等符号。",
                },
                ensure_ascii=False,
            )

        async def _vlm_unused(_prompt, **_kwargs):
            raise AssertionError("VLM must not be called for tables")

        rag = _new_rag(
            tmp_path,
            vlm_llm_model_func=_vlm_unused,
            extract_llm_model_func=_extract,
        )
        await rag.initialize_storages()

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps({"type": "meta", "format_version": "1.0"}),
                    json.dumps({"type": "content", "content": "body"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        tables = tmp_path / "demo.tables.json"
        tables.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "tables": {
                        "id1": {
                            "id": "id1",
                            "caption": "表1 指标说明",
                            "footnotes": ["单位：国际标准单位"],
                            "format": "html",
                            "content": "<table><tr><th>符号</th><th>代表意义</th><th>单位</th></tr><tr><td>A</td><td>面积</td><td>m2</td></tr></table>",
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed, process_options="t")

        payload = json.loads(tables.read_text(encoding="utf-8"))
        result = payload["tables"]["id1"]["llm_analyze_result"]
        assert result["status"] == "success"
        assert result["name"] == "model_benchmark_metrics"
        assert "符号、代表意义和单位" in result["description"]
        # Cache id was written back so document delete can clean it up.
        assert any(
            cid.startswith("default:analysis:")
            for cid in payload["tables"]["id1"].get("llm_cache_list", [])
        )

    asyncio.run(_run())


@pytest.mark.offline
def test_parser_source_resolver_finds_hint_variant_by_canonical_name(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    hinted = input_dir / "demo.[mineru].pdf"
    hinted.write_bytes(b"fake-pdf")
    rag = _new_rag(tmp_path / "work")

    resolved = rag._resolve_source_file_for_parser(
        "demo.pdf",
        parser_engine=PARSER_ENGINE_MINERU,
    )

    assert Path(resolved) == hinted


@pytest.mark.offline
def test_parser_source_resolver_prefers_exact_canonical_file(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    exact = input_dir / "demo.pdf"
    hinted = input_dir / "demo.[mineru].pdf"
    exact.write_bytes(b"canonical")
    hinted.write_bytes(b"hinted")
    rag = _new_rag(tmp_path / "work")

    resolved = rag._resolve_source_file_for_parser(
        "demo.pdf",
        parser_engine=PARSER_ENGINE_MINERU,
    )

    assert Path(resolved) == exact


@pytest.mark.offline
def test_parse_mineru_to_lightrag_document(tmp_path, monkeypatch):
    """End-to-end: parse_mineru routes through MinerURawClient + sidecar
    writer and produces spec-compliant *.parsed/ + *.mineru_raw/ artifacts.

    With the unified pipeline (introduced alongside the MinerU raw bundle
    cache), the MinerU download choreography happens inside
    :meth:`MinerURawClient.download_into`. We stub that method directly.
    """
    from lightrag.parser.external.mineru import compute_size_and_hash
    from lightrag.parser.external.mineru.cache import current_mineru_options_signature
    from lightrag.parser.external.mineru.client import MinerURawClient
    from lightrag.parser.external.mineru.manifest import (
        Manifest,
        ManifestFile,
        write_manifest,
    )

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()

        src_file = input_dir / "demo.pdf"
        src_file.write_bytes(b"fake-pdf")

        async def _fake_download(self, raw_dir, source_file_path, **_kwargs):
            assert source_file_path == src_file
            raw_dir.mkdir(parents=True, exist_ok=True)
            content_list = [
                {"type": "text", "text": "第一段正文"},
                {
                    "type": "image",
                    "img_path": "assets/img1.png",
                    "image_caption": ["图1 架构图"],
                    "image_footnote": ["示意图"],
                },
                {
                    "type": "table",
                    "table_body": "<table><tr><td>A</td></tr></table>",
                    "table_caption": ["表1 指标"],
                    "table_footnote": ["单位：%"],
                },
                {"type": "equation", "text": "$$E=mc^2$$"},
            ]
            (raw_dir / "content_list.json").write_text(
                json.dumps(content_list, ensure_ascii=False),
                encoding="utf-8",
            )
            (raw_dir / "assets").mkdir()
            (raw_dir / "assets" / "img1.png").write_bytes(b"\x89PNGfake")
            src_size, src_hash = compute_size_and_hash(source_file_path)
            crit_size, crit_hash = compute_size_and_hash(raw_dir / "content_list.json")
            manifest = Manifest(
                source_content_hash=src_hash,
                source_size_bytes=src_size,
                source_filename_at_parse=source_file_path.name,
                critical_file=ManifestFile(
                    path="content_list.json", size=crit_size, sha256=crit_hash
                ),
                files=[
                    ManifestFile(
                        path="assets/img1.png",
                        size=(raw_dir / "assets" / "img1.png").stat().st_size,
                    )
                ],
                total_size_bytes=crit_size,
                task_id="fake-task",
                api_mode="local",
                options_signature=current_mineru_options_signature(),
            )
            write_manifest(raw_dir, manifest)
            return manifest

        monkeypatch.setattr(MinerURawClient, "download_into", _fake_download)
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

        parsed = await _parse_via_registry(
            rag,
            "mineru",
            doc_id="doc-1",
            file_path=str(src_file),
            content_data={"content": ""},
        )

        assert parsed["parse_format"] == "lightrag"
        assert parsed["blocks_path"]
        blocks_path = Path(parsed["blocks_path"])
        assert blocks_path.exists()

        lines = blocks_path.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        assert meta["type"] == "meta"
        assert meta["format"] == "lightrag"
        assert meta["drawing_file"] is True
        assert meta["table_file"] is True
        assert meta["equation_file"] is True

        base = str(blocks_path)[: -len(".blocks.jsonl")]
        drawings = json.loads(Path(base + ".drawings.json").read_text(encoding="utf-8"))
        tables = json.loads(Path(base + ".tables.json").read_text(encoding="utf-8"))
        equations = json.loads(
            Path(base + ".equations.json").read_text(encoding="utf-8")
        )
        assert drawings["drawings"]
        assert tables["tables"]
        assert equations["equations"]

        full_doc = await rag.full_docs.get_by_id("doc-1")
        assert full_doc["parse_format"] == "lightrag"
        # Per docs/FileProcessingConfiguration-zh.md spec, ``content`` is now
        # ``{{LRdoc}}`` followed by a leading-text summary of the document.
        assert full_doc["content"].startswith("{{LRdoc}}")
        assert full_doc["sidecar_location"].startswith("file://")
        assert full_doc["sidecar_location"].endswith("/")
        assert str(blocks_path.parent.resolve()) in full_doc["sidecar_location"]

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_mineru_uses_hint_source_and_canonical_upload_name(tmp_path, monkeypatch):
    from lightrag.parser.external.mineru import compute_size_and_hash
    from lightrag.parser.external.mineru.cache import current_mineru_options_signature
    from lightrag.parser.external.mineru.client import MinerURawClient
    from lightrag.parser.external.mineru.manifest import (
        Manifest,
        ManifestFile,
        write_manifest,
    )

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()

        hinted_name = "LightRAG - Simple and Fast RAG.[mineru].pdf"
        canonical_name = "LightRAG - Simple and Fast RAG.pdf"
        src_file = input_dir / hinted_name
        src_file.write_bytes(b"fake-pdf")

        async def _fake_download(self, raw_dir, source_file_path, **kwargs):
            assert source_file_path == src_file
            assert kwargs.get("upload_name") == canonical_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            content_list = [{"type": "text", "text": "第一段正文"}]
            content_path = raw_dir / "content_list.json"
            content_path.write_text(
                json.dumps(content_list, ensure_ascii=False),
                encoding="utf-8",
            )
            src_size, src_hash = compute_size_and_hash(source_file_path)
            crit_size, crit_hash = compute_size_and_hash(content_path)
            manifest = Manifest(
                source_content_hash=src_hash,
                source_size_bytes=src_size,
                source_filename_at_parse=kwargs.get("upload_name"),
                critical_file=ManifestFile(
                    path="content_list.json",
                    size=crit_size,
                    sha256=crit_hash,
                ),
                files=[],
                total_size_bytes=crit_size,
                task_id="fake-task",
                api_mode="local",
                options_signature=current_mineru_options_signature(),
            )
            write_manifest(raw_dir, manifest)
            return manifest

        monkeypatch.setattr(MinerURawClient, "download_into", _fake_download)

        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(src_file),
            track_id="track-hint",
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=PARSER_ENGINE_MINERU,
        )

        doc_id = compute_mdhash_id(canonical_name, prefix="doc-")
        status = await rag.doc_status.get_by_id(doc_id)
        assert status is not None
        assert status["file_path"] == canonical_name
        assert status["metadata"]["source_file"] == hinted_name

        content_data = await rag.full_docs.get_by_id(doc_id)
        assert content_data is not None
        content_data["source_file"] = status["metadata"]["source_file"]

        parsed = await _parse_via_registry(
            rag,
            "mineru",
            doc_id=doc_id,
            file_path=status["file_path"],
            content_data=content_data,
        )

        blocks_path = Path(parsed["blocks_path"])
        expected_parsed_dir = input_dir / PARSED_DIR_NAME / f"{canonical_name}.parsed"
        expected_raw_dir = (
            input_dir
            / PARSED_DIR_NAME
            / ("LightRAG - Simple and Fast RAG.pdf.mineru_raw")
        )
        archived_source = input_dir / PARSED_DIR_NAME / hinted_name

        assert blocks_path.parent == expected_parsed_dir
        assert expected_raw_dir.is_dir()
        assert not src_file.exists()
        assert archived_source.is_file()

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_mm_chunks_and_modality_relations_from_sidecars(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "meta",
                            "format": "lightrag",
                            "version": "1.0",
                            "doc_id": "doc-1",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "type": "content",
                            "blockid": "b1",
                            "format": "plain_text",
                            "content": "正文",
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        drawings = tmp_path / "demo.drawings.json"
        drawings.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "drawings": {
                        "d1": {
                            "id": "d1",
                            "heading": "章节A",
                            "caption": "图1 架构",
                            "llm_cache_list": [
                                "default:analysis:abc123",
                            ],
                            "llm_analyze_result": {
                                "name": "系统架构图",
                                "type": "Chart",
                                "description": "模块交互关系",
                                "analyze_time": 1700000000,
                                "status": "success",
                                "message": "",
                            },
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        mm_chunks = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=2,
        )
        assert len(mm_chunks) == 1
        chunk = mm_chunks[0]
        # New nested schema: heading + sidecar + llm_cache_list merge.
        assert chunk["content"].startswith("[Image Name]")
        assert "[Image Type]Chart" in chunk["content"]
        assert chunk["sidecar"] == {
            "type": "drawing",
            "id": "d1",
            "refs": [{"type": "drawing", "id": "d1"}],
        }
        assert chunk["heading"] == {
            "level": 0,
            "heading": "章节A",
            "parent_headings": [],
        }
        assert chunk["llm_cache_list"] == ["default:analysis:abc123"]
        # Multimodal entity injection now lives in
        # operate.extract_entities._process_single_content; this test only
        # covers chunk assembly. The companion regression below
        # (test_parse_mm_display_name_matches_chunk_format) pins the
        # builder/consumer format contract.

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_mm_chunks_sanitize_vlm_control_characters(tmp_path):
    """Regression: VLM analysis fields parsed from LLM JSON can carry
    control characters (unescaped LaTeX ``\\frac`` decodes as ``\\x0c`` +
    ``rac``). The builder must strip them — they propagate into chunk
    content, vector stores and graph node attributes, where XML-illegal
    characters crash the GraphML flush with "All strings must be XML
    compatible".
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            json.dumps(
                {
                    "type": "meta",
                    "format": "lightrag",
                    "version": "1.0",
                    "doc_id": "doc-1",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        tables = tmp_path / "demo.tables.json"
        tables.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "tables": {
                        "t1": {
                            "id": "t1",
                            "heading": "章节\x0bB",
                            "footnotes": ["脚注\x00一"],
                            "llm_analyze_result": {
                                "name": "成本对比表\x00",
                                "description": "GraphRAG消耗$\x0crac{610}{C}$次调用",
                                "analyze_time": 1700000000,
                                "status": "success",
                                "message": "",
                            },
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        mm_chunks = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
        )
        assert len(mm_chunks) == 1
        chunk = mm_chunks[0]

        illegal = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
        assert not illegal.search(chunk["content"])
        assert not illegal.search(chunk["heading"]["heading"])
        # Control chars are removed, surrounding text retained.
        assert "[Table Name]成本对比表" in chunk["content"]
        assert "$rac{610}{C}$" in chunk["content"]
        assert "[Table Footnotes]脚注一" in chunk["content"]
        assert chunk["heading"]["heading"] == "章节B"

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_mm_display_name_matches_chunk_format():
    """Pin the builder/consumer contract: the chunk content emitted by
    ``_build_mm_chunks_from_sidecars`` must parse cleanly via
    ``operate._parse_mm_display_name`` for all three modalities.
    Regression for the case where the renderer's label format diverged
    from the consumer's regex and display names silently fell back to
    sidecar ids.
    """

    drawing_content = (
        "[Image Name]系统架构图\n[Image Type]Chart\n\n模块交互关系\n\n"
        "[Image Footnotes]脚注1; 脚注2"
    )
    assert _parse_mm_display_name(drawing_content, "d1") == "系统架构图"

    table_content = "[Table Name]性能对比表\n\n各方法的指标对比"
    assert _parse_mm_display_name(table_content, "t1") == "性能对比表"

    equation_content = "E = mc^2\n[Equation Name]质能方程\n\n爱因斯坦的质能等效公式"
    assert _parse_mm_display_name(equation_content, "e1") == "质能方程"

    # Fallbacks: missing marker, empty content, marker with blank name.
    assert _parse_mm_display_name("no marker here", "fallback-id") == "fallback-id"
    assert _parse_mm_display_name("", "fallback-id") == "fallback-id"
    assert _parse_mm_display_name("[Image Name]   ", "fallback-id") == "fallback-id"


@pytest.mark.offline
def test_parse_mm_display_name_on_real_builder_output(tmp_path):
    """End-to-end pin: feed actual chunks from
    ``_build_mm_chunks_from_sidecars`` for all three modalities straight
    into the consumer's parser, and require the analysis ``name`` field
    to come back. This locks the bidirectional builder/consumer contract
    so that renaming the ``[Image|Table|Equation] Name`` label without
    updating the regex in ``operate._parse_mm_display_name`` immediately
    breaks here instead of silently degrading relation descriptions.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        blocks = tmp_path / "demo.pdf.blocks.jsonl"
        blocks.write_text("", encoding="utf-8")

        heading = {"level": 1, "heading": "章节A", "parent_headings": []}

        (tmp_path / "demo.pdf.drawings.json").write_text(
            json.dumps(
                {
                    "drawings": {
                        "d1": {
                            "heading": heading,
                            "footnotes": [],
                            "llm_cache_list": [],
                            "llm_analyze_result": {
                                "name": "系统架构图",
                                "type": "Chart",
                                "description": "模块交互关系",
                                "analyze_time": 1700000000,
                                "status": "success",
                                "message": "",
                            },
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (tmp_path / "demo.pdf.tables.json").write_text(
            json.dumps(
                {
                    "tables": {
                        "t1": {
                            "heading": heading,
                            "footnotes": [],
                            "llm_cache_list": [],
                            "llm_analyze_result": {
                                "name": "性能对比表",
                                "description": "各方法的指标对比",
                                "analyze_time": 1700000001,
                                "status": "success",
                                "message": "",
                            },
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (tmp_path / "demo.pdf.equations.json").write_text(
            json.dumps(
                {
                    "equations": {
                        "e1": {
                            "heading": heading,
                            "footnotes": [],
                            "llm_cache_list": [],
                            "llm_analyze_result": {
                                "name": "质能方程",
                                "equation": "E = mc^2",
                                "description": "爱因斯坦的质能等效公式",
                                "analyze_time": 1700000002,
                                "status": "success",
                                "message": "",
                            },
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        mm_chunks = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
            process_options="ite",
        )
        # Index by sidecar type for stable assertions independent of
        # iteration order in the builder.
        by_type = {chunk["sidecar"]["type"]: chunk for chunk in mm_chunks}
        assert set(by_type.keys()) == {"drawing", "table", "equation"}

        expected = {
            "drawing": ("d1", "系统架构图"),
            "table": ("t1", "性能对比表"),
            "equation": ("e1", "质能方程"),
        }
        for kind, (sidecar_id, name) in expected.items():
            display = _parse_mm_display_name(by_type[kind]["content"], sidecar_id)
            assert display == name, (
                f"{kind}: parser failed to extract '{name}' from builder "
                f"output, got '{display}' (content: {by_type[kind]['content']!r})"
            )

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_mineru_empty_service_result_raises_without_fallback(
    tmp_path, monkeypatch
):
    """When MinerU produces no content_list.json the IR builder raises and the
    pipeline propagates the error — no silent fallback to raw text.

    With the unified pipeline, an "empty result" surfaces as a missing
    critical file inside ``*.mineru_raw/``; the IR builder's
    ``normalize_from_workdir`` raises :class:`FileNotFoundError` and the
    parse fails fast.
    """
    from lightrag.parser.external.mineru.client import MinerURawClient

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        src_file = tmp_path / "demo.pdf"
        src_file.write_bytes(b"fake-pdf")

        async def _fake_download(self, raw_dir, source_file_path, **_kwargs):
            # Simulate a "MinerU returned nothing useful" bundle: dir is
            # touched but no content_list.json is produced.
            raw_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(MinerURawClient, "download_into", _fake_download)
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake")

        with pytest.raises(FileNotFoundError, match="content_list.json"):
            await _parse_via_registry(
                rag,
                "mineru",
                doc_id="doc-local-1",
                file_path=str(src_file),
                content_data={"content": "native fallback content"},
            )

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_build_chunks_dict_preserves_existing_llm_cache_list():
    """Regression: build_chunks_dict_from_chunking_result must not overwrite
    a chunk's pre-existing llm_cache_list — multimodal chunks arrive with
    analysis cache ids already attached so document deletion can clean
    them up via the per-chunk llm_cache_list."""
    from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result

    chunking_result = [
        {
            "chunk_order_index": 0,
            "content": "first chunk",
            "tokens": 4,
        },
        {
            "chunk_id": "doc-1-mm-drawing-000",
            "chunk_order_index": 1,
            "content": "second chunk",
            "tokens": 6,
            "llm_cache_list": [
                "default:analysis:abc",
                "default:analysis:abc",  # dedup verification
                "default:analysis:def",
            ],
        },
    ]

    chunks = build_chunks_dict_from_chunking_result(
        chunking_result, doc_id="doc-1", file_path="demo.pdf"
    )
    # Order is chunking_result order; locate by chunk_id.
    mm_chunk = next(
        v for v in chunks.values() if v.get("chunk_id") == "doc-1-mm-drawing-000"
    )
    text_chunk = next(
        v for v in chunks.values() if v.get("chunk_id") != "doc-1-mm-drawing-000"
    )
    assert mm_chunk["llm_cache_list"] == [
        "default:analysis:abc",
        "default:analysis:def",
    ]
    # Plain text chunks still start with an empty list (no pre-existing ids).
    assert text_chunk["llm_cache_list"] == []


@pytest.mark.offline
def test_build_mm_chunks_respects_process_options_filter(tmp_path):
    """Regression: _build_mm_chunks_from_sidecars must gate sidecar reads
    by the active process_options.  A document re-processed after opting
    out of i/t/e MUST NOT pick up stale success results from a prior pass.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps({"type": "meta", "format_version": "1.0"}),
                    json.dumps({"type": "content", "content": "body"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        # Both modalities carry a stale ``success`` from a prior pass.
        drawings = tmp_path / "demo.drawings.json"
        drawings.write_text(
            json.dumps(
                {
                    "drawings": {
                        "d1": {
                            "id": "d1",
                            "llm_analyze_result": {
                                "name": "old",
                                "type": "Chart",
                                "description": "stale drawing",
                                "analyze_time": 1700000000,
                                "status": "success",
                                "message": "",
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        tables = tmp_path / "demo.tables.json"
        tables.write_text(
            json.dumps(
                {
                    "tables": {
                        "t1": {
                            "id": "t1",
                            "llm_analyze_result": {
                                "name": "old",
                                "description": "stale table",
                                "analyze_time": 1700000000,
                                "status": "success",
                                "message": "",
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        # process_options="t" → only tables are considered; the drawing
        # success entry must NOT generate a chunk.
        only_tables = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
            process_options="t",
        )
        assert len(only_tables) == 1
        assert only_tables[0]["sidecar"]["type"] == "table"

        # Empty/None process_options → no modalities active → no chunks.
        none_active = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
            process_options="",
        )
        assert none_active == []

        # Backwards-compat: callers that pass process_options=None see
        # every modality (legacy behaviour for ad-hoc unit tests).
        legacy = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
        )
        assert {ch["sidecar"]["type"] for ch in legacy} == {"drawing", "table"}

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_strip_internal_multimodal_markup_cleans_table_id():
    """Regression: parser-emitted ``<table id="tb-...">`` tags must have
    their internal id stripped before the entity-extraction prompt sees
    them. ``format`` / ``caption`` and the row body stay verbatim so the
    extractor still recognizes the structured element."""
    from lightrag.chunk_schema import (
        strip_internal_multimodal_markup_for_extraction,
    )

    source = (
        '<table id="tb-1-0001" format="json" caption="Indicator metrics">'
        '[["a","b"],["1","2"]]'
        "</table>"
    )
    cleaned = strip_internal_multimodal_markup_for_extraction(source)
    assert "tb-1-0001" not in cleaned
    assert 'format="json"' in cleaned
    assert 'caption="Indicator metrics"' in cleaned
    # Row body preserved.
    assert '[["a","b"],["1","2"]]' in cleaned


@pytest.mark.offline
def test_strip_internal_multimodal_markup_cite_default_unwraps():
    """Default (keep_cite_tag=False) is the entity-extraction path: the
    ``<cite>`` wrapper is stripped so the extractor does not surface it
    as a structural entity — only the visible label survives.

    The surrounding-context path overrides this via keep_cite_tag=True
    (verified in tests/pipeline/test_multimodal_surrounding_context.py); this
    test pins the default to prevent regressions on the extraction
    path when callers refactor the function signature.
    """
    from lightrag.chunk_schema import (
        strip_internal_multimodal_markup_for_extraction,
    )

    source = (
        'see <cite type="table" refid="tb-1-0001">表 1</cite> and '
        '<cite type="equation" refid="eq-1-0002">公式 2</cite> for details.'
    )
    cleaned = strip_internal_multimodal_markup_for_extraction(source)
    # Wrappers and ids both gone; visible labels survive as plain text.
    assert "<cite" not in cleaned
    assert "refid=" not in cleaned
    assert "tb-1-0001" not in cleaned
    assert "eq-1-0002" not in cleaned
    assert "表 1" in cleaned
    assert "公式 2" in cleaned


@pytest.mark.offline
def test_strip_internal_multimodal_markup_cite_keep_tag_strips_refid_only():
    """keep_cite_tag=True (surrounding-context path): preserve the
    ``<cite type="…">label</cite>`` wrapper but drop the parser-
    internal ``refid``.  Other identifier transformations
    (``<table id=…>`` / ``<drawing id=…/>`` / ``<equation id=…>``) are
    unaffected by the flag and still apply."""
    from lightrag.chunk_schema import (
        strip_internal_multimodal_markup_for_extraction,
    )

    source = (
        'see <cite type="table" refid="tb-1-0001">表 1</cite>; '
        '<drawing id="im-1" path="a.png" src="a" caption="Fig" />'
    )
    cleaned = strip_internal_multimodal_markup_for_extraction(
        source, keep_cite_tag=True
    )
    assert '<cite type="table">表 1</cite>' in cleaned
    assert "refid=" not in cleaned
    assert "tb-1-0001" not in cleaned
    # Non-cite cleaning still applies in this mode.
    assert '<drawing caption="Fig" />' in cleaned
    assert 'id="im-1"' not in cleaned
    assert "path=" not in cleaned


@pytest.mark.offline
def test_reinsert_without_process_options_skips_stale_mm_chunks(tmp_path):
    """Regression for the call-site fallback in process_single_document.

    A document re-inserted without ``process_options`` is signalled by a
    missing / falsy ``content_data["process_options"]`` field.  The
    pipeline must pass ``""`` (not ``None``) to
    ``_build_mm_chunks_from_sidecars`` so the builder honors the
    "no modalities" contract: stale ``status="success"`` sidecar entries
    from an earlier i/t/e pass MUST NOT be re-indexed.

    The new builder happens to handle ``None`` by enabling every
    modality for ad-hoc callers (unit tests), so this test pins the
    call-site behaviour rather than the helper's default — passing the
    same falsy value via ``or ""`` makes the intent explicit.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        blocks = tmp_path / "demo.blocks.jsonl"
        blocks.write_text(
            "\n".join(
                [
                    json.dumps({"type": "meta", "format_version": "1.0"}),
                    json.dumps({"type": "content", "content": "body"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        # Stale success from an earlier multimodal run.
        drawings = tmp_path / "demo.drawings.json"
        drawings.write_text(
            json.dumps(
                {
                    "drawings": {
                        "d1": {
                            "id": "d1",
                            "llm_analyze_result": {
                                "name": "old",
                                "type": "Chart",
                                "description": "stale drawing",
                                "analyze_time": 1700000000,
                                "status": "success",
                                "message": "",
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        # Simulate the call-site contract: ``content_data`` has no
        # ``process_options`` key, so ``.get(...) or ""`` yields "".
        content_data: dict[str, str] = {}
        effective = (content_data or {}).get("process_options") or ""
        assert effective == ""

        mm_chunks = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
            process_options=effective,
        )
        assert mm_chunks == []

        await rag.finalize_storages()

    asyncio.run(_run())


def test_engine_params_survive_persist_to_full_docs(tmp_path, monkeypatch):
    """Per-file engine params encoded in parse_engine survive the parse persist.

    Regression for the ``{**existing, **record}`` merge in
    ``_persist_parsed_full_docs``: the external parser must re-encode
    ``engine_name(params)`` so full_docs keeps the per-file params instead of
    reverting to the bare engine name.
    """
    from lightrag.parser.external.mineru import compute_size_and_hash
    from lightrag.parser.external.mineru.cache import (
        current_mineru_options_signature,
    )
    from lightrag.parser.external.mineru.client import MinerURawClient
    from lightrag.parser.external.mineru.manifest import (
        Manifest,
        ManifestFile,
        write_manifest,
    )

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        monkeypatch.setenv("MINERU_API_MODE", "local")
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()

        src_file = input_dir / "demo.pdf"
        src_file.write_bytes(b"fake-pdf")

        async def _fake_download(self, raw_dir, source_file_path, **_kwargs):
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "content_list.json").write_text(
                json.dumps([{"type": "text", "text": "正文"}], ensure_ascii=False),
                encoding="utf-8",
            )
            src_size, src_hash = compute_size_and_hash(source_file_path)
            crit_size, crit_hash = compute_size_and_hash(raw_dir / "content_list.json")
            write_manifest(
                raw_dir,
                Manifest(
                    source_content_hash=src_hash,
                    source_size_bytes=src_size,
                    source_filename_at_parse=source_file_path.name,
                    critical_file=ManifestFile(
                        path="content_list.json", size=crit_size, sha256=crit_hash
                    ),
                    files=[],
                    total_size_bytes=crit_size,
                    task_id="fake-task",
                    api_mode="local",
                    options_signature=current_mineru_options_signature(
                        {"page_range": "1-3"}
                    ),
                ),
            )

        monkeypatch.setattr(MinerURawClient, "download_into", _fake_download)

        await _parse_via_registry(
            rag,
            "mineru",
            doc_id="doc-ep",
            file_path=str(src_file),
            content_data={"content": "", "parse_engine": "mineru(page_range=1-3)"},
        )

        stored = await rag.full_docs.get_by_id("doc-ep")
        assert stored is not None
        # The encoded directive (with params) is preserved, not reverted to bare.
        assert stored["parse_engine"] == "mineru(page_range=1-3)"

        await rag.finalize_storages()

    asyncio.run(_run())
