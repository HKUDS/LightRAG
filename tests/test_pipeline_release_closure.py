import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocStatus
from lightrag.constants import (
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_DIR_NAME,
    PARSER_ENGINE_NATIVE,
)
from lightrag.operate import _get_relationship_vdb_timeout_seconds
from lightrag.parser_routing import (
    ParserRoutingConfigError,
    canonicalize_parser_hinted_basename,
    resolve_file_parser_engine,
    resolve_stored_document_parser_engine,
    validate_parser_routing_config,
)
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    safe_vdb_operation_with_exception,
)


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

    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:mineru-iet,*:native")
    assert resolve_stored_document_parser_engine("paper.pdf", {}) == "mineru"
    assert (
        resolve_stored_document_parser_engine("paper.pdf", {"parse_engine": "native"})
        == "legacy"
    )


@pytest.mark.offline
def test_parse_engine_rule_fallback_and_default_legacy(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:native,*:legacy")
    assert resolve_stored_document_parser_engine("paper.pdf", {}) == "legacy"

    monkeypatch.setenv("LIGHTRAG_PARSER", "pptx:docling,*:legacy")
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    assert resolve_stored_document_parser_engine("slides.pptx", {}) == "legacy"

    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    monkeypatch.setenv("MINERU_ENDPOINT", "")
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
    assert canonicalize_parser_hinted_basename("foo.[!].docx") == "foo.docx"
    assert canonicalize_parser_hinted_basename("foo.[native-iet].docx") == "foo.docx"
    assert canonicalize_parser_hinted_basename("foo.[mineru-R!].pdf") == "foo.pdf"
    # Invalid options-only hint (unknown chars) is left alone.
    assert canonicalize_parser_hinted_basename("foo.[xyz].docx") == "foo.[xyz].docx"


@pytest.mark.offline
def test_filename_parser_directives_decodes_engine_and_options():
    from lightrag.parser_routing import filename_parser_directives

    assert filename_parser_directives("paper.[native-iet].docx") == ("native", "iet")
    assert filename_parser_directives("memo.[native-R!].md") == ("native", "R!")
    assert filename_parser_directives("report.[!].pdf") == (None, "!")
    assert filename_parser_directives("doc.[mineru].docx") == ("mineru", "")
    assert filename_parser_directives("foo.docx") == (None, "")
    # Unsupported tokens leave the hint untouched and unparsed.
    assert filename_parser_directives("foo.[draft].docx") == (None, "")


@pytest.mark.offline
def test_filename_hint_rejects_invalid_engine_qualified_options():
    """Engine-qualified hints with bad option chars must fail validation
    the same way options-only hints do, so the documented behaviour
    "invalid characters → whole hint fails → defaults apply" holds across
    all hint shapes (otherwise foo.[native-FR].docx would be canonicalised
    even though its options conflict).
    """
    from lightrag.parser_routing import (
        canonicalize_parser_hinted_basename,
        filename_parser_directives,
    )

    # F+R conflict → hint dropped; engine and options are NOT applied.
    assert filename_parser_directives("foo.[native-FR].docx") == (None, "")
    # Unknown char Q → hint dropped; engine native is also NOT applied.
    assert filename_parser_directives("foo.[native-Q].docx") == (None, "")

    # The basename must remain unchanged so the documented "defaults apply"
    # path in the dedup index reflects the literal file the user supplied.
    assert (
        canonicalize_parser_hinted_basename("foo.[native-FR].docx")
        == "foo.[native-FR].docx"
    )
    assert (
        canonicalize_parser_hinted_basename("foo.[native-Q].docx")
        == "foo.[native-Q].docx"
    )


@pytest.mark.offline
def test_parse_process_options_decodes_flags():
    from lightrag.parser_routing import parse_process_options

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
    from lightrag.parser_routing import validate_process_options

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
    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")
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
    from lightrag.parser_routing import resolve_file_parser_directives

    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")
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
    engine, options = resolve_file_parser_directives("plain.[!].docx")
    assert engine == "native"
    assert options == "!"


@pytest.mark.offline
def test_doc_status_metadata_carry_over_helper():
    """``doc_status_transition_metadata`` preserves long-lived per-doc fields
    (currently ``process_options``) and layers in any transition-specific
    extras passed via ``extra=``.  Empty / missing carry-over fields are
    dropped, not written as null.
    """
    from lightrag.utils_pipeline import doc_status_transition_metadata

    class _StubStatusDoc:
        def __init__(self, metadata):
            self.metadata = metadata

    # Carries process_options forward.
    md = doc_status_transition_metadata(_StubStatusDoc({"process_options": "iet"}))
    assert md == {"process_options": "iet"}

    # Layers in transition extras while keeping the carry-over.
    md = doc_status_transition_metadata(
        _StubStatusDoc({"process_options": "R!"}),
        extra={"processing_start_time": 12345},
    )
    assert md == {"process_options": "R!", "processing_start_time": 12345}

    # No carry-over when metadata is missing or empty.
    assert doc_status_transition_metadata(_StubStatusDoc({})) == {}
    assert doc_status_transition_metadata(None) == {}

    # Empty / None process_options are not written as null.
    assert doc_status_transition_metadata(_StubStatusDoc({"process_options": ""})) == {}
    assert (
        doc_status_transition_metadata(_StubStatusDoc({"process_options": None})) == {}
    )


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
        finally:
            await rag.finalize_storages()

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
            # full_docs preserves the user-visible name and the canonical key.
            assert full_doc["file_path"] == "abc.[native-R!].docx"
            assert full_doc.get("canonical_basename") == "abc.docx"
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
    """``_purge_doc_chunks_and_kg`` with an empty chunk_ids set must be a
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
            # Empty set: must return immediately without touching storage.
            await rag._purge_doc_chunks_and_kg(
                "doc-empty",
                set(),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )
            # No exceptions → success.  Calling twice in a row is also fine
            # since the helper is idempotent on the empty input.
            await rag._purge_doc_chunks_and_kg(
                "doc-empty",
                set(),
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
                {"doc-X-chunk-0", "doc-X-chunk-1"},
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
                        "canonical_basename": "resume.txt",
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
                        "canonical_basename": "resume.txt",
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
                        "canonical_basename": "noskip.txt",
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
                        "canonical_basename": "noskip.txt",
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

            assert (
                calls == []
            ), "purge helper should not be called when chunks_list is empty"
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
def test_analyze_multimodal_skips_already_analyzed_items(tmp_path):
    """Re-running analyze_multimodal must not re-analyze items that already
    carry an ``llm_analyze_result`` from a prior pass.  This makes
    enabling a new modality (e.g. add ``t`` after a prior ``i``-only pass)
    cheap: the drawings sidecar is fully populated and skipped, while the
    tables sidecar is newly populated.
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
                            "path": "missing.png",
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
        await rag.analyze_multimodal(
            "doc-1", "demo.pdf", parsed, process_options="it"
        )

        drawings_payload = json.loads(drawings.read_text(encoding="utf-8"))
        existing = drawings_payload["drawings"]["id1"]["llm_analyze_result"]
        # Existing result preserved verbatim — VLM was NOT called for this item.
        assert existing["name"] == "Existing"
        assert existing["status"] == "success"

        tables_payload = json.loads(tables.read_text(encoding="utf-8"))
        new_result = tables_payload["tables"]["tbl1"]["llm_analyze_result"]
        assert new_result["name"] == "Item"
        assert new_result["status"] == "success"

        # Drawings were idempotently skipped (VLM never called); tables took
        # the EXTRACT role (per design §3.1), not VLM.
        assert vlm_calls["n"] == 0
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
            # file_path keeps the user's original basename verbatim, while
            # canonical_basename carries the dedup key.
            assert first_doc["file_path"] == "abc.docx"
            assert first_doc.get("canonical_basename") == "abc.docx"

            await rag.apipeline_enqueue_documents(
                "changed body",
                file_paths="/tmp/abc.[native].docx",
                track_id="track-b",
            )
            assert (await rag.full_docs.get_by_id(first_id))["content"] == "alpha body"

            failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            # The duplicate record reflects the second attempt's user-visible
            # basename (hint preserved); the canonical dedup happened against
            # ``abc.docx`` regardless.
            assert any(
                getattr(doc, "metadata", {}).get("duplicate_kind") == "filename"
                and getattr(doc, "file_path", "") == "abc.[native].docx"
                for doc in failed_docs.values()
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_delete_result_preserves_parser_hinted_source_path(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            source_path = str(tmp_path / "abc.[native].docx")
            await rag.apipeline_enqueue_documents(
                "",
                file_paths=source_path,
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine=PARSER_ENGINE_NATIVE,
                track_id="track-delete-source",
            )

            doc_id = compute_mdhash_id("abc.docx", prefix="doc-")
            result = await rag.adelete_by_doc_id(doc_id)

            assert result.status == "success"
            # ``file_path`` now preserves the parser-hint segment for UI
            # display; canonicalisation only affects the dedup key.
            assert result.file_path == "abc.[native].docx"
            assert result.source_path == source_path
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
def test_basename_lookup_finds_legacy_full_path_records(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Simulate a legacy doc_status row whose file_path still holds a
            # full path; the new basename lookup should still resolve it.
            legacy_id = "doc-legacy-1"
            await rag.doc_status.upsert(
                {
                    legacy_id: {
                        "status": DocStatus.PROCESSED,
                        "content_summary": "legacy",
                        "content_length": 7,
                        "file_path": "/inputs/legacy.txt",
                        "track_id": "legacy-track",
                        "created_at": "2025-01-01T00:00:00+00:00",
                        "updated_at": "2025-01-01T00:00:00+00:00",
                        "chunks_list": [],
                    }
                }
            )

            match = await rag.doc_status.get_doc_by_file_basename("legacy.txt")
            assert match is not None
            doc_id, doc = match
            assert doc_id == legacy_id
            assert doc["file_path"] == "/inputs/legacy.txt"

            # Re-enqueueing the same basename hits the filename guard.
            await rag.apipeline_enqueue_documents(
                "fresh body",
                file_paths="legacy.txt",
                track_id="track-x",
            )
            new_id = compute_mdhash_id("legacy.txt", prefix="doc-")
            assert await rag.full_docs.get_by_id(new_id) is None
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
def test_lightrag_format_uses_blocks_file_hash(tmp_path, monkeypatch):
    async def _run():
        input_dir = tmp_path / "input"
        parsed_dir = input_dir / "__parsed__"
        parsed_dir.mkdir(parents=True)
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        rag = _new_rag(tmp_path / "work")
        rag.workspace = "test-pending-parse-duplicate"
        await rag.initialize_storages()
        try:
            blocks_path = parsed_dir / "doc.blocks.jsonl"
            blocks_path.write_text(
                json.dumps({"type": "header"})
                + "\n"
                + json.dumps({"type": "content", "text": "hello"})
                + "\n",
                encoding="utf-8",
            )

            # Enqueue twice with different filenames pointing at the same
            # blocks file: the second one must be rejected as content_hash dup.
            # ``content`` arg is ignored on the LIGHTRAG path — the LightRAG
            # Document file is read to derive both content_hash and the
            # ``{{LRdoc}}`` summary — so any string here is fine.
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="first.lightrag",
                docs_format="lightrag",
                lightrag_document_paths="__parsed__/doc.blocks.jsonl",
                track_id="track-a",
            )
            await rag.apipeline_enqueue_documents(
                "",
                file_paths="second.lightrag",
                docs_format="lightrag",
                lightrag_document_paths="__parsed__/doc.blocks.jsonl",
                track_id="track-b",
            )
            second_id = compute_mdhash_id("second.lightrag", prefix="doc-")
            assert await rag.full_docs.get_by_id(second_id) is None

            failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            kinds = {
                getattr(doc, "metadata", {}).get("duplicate_kind")
                for doc in failed.values()
                if getattr(doc, "metadata", {}).get("is_duplicate")
            }
            assert "content_hash" in kinds
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
    """``_persist_parsed_full_docs`` must keep process_options / canonical_basename
    seeded at enqueue time so downstream stages (analyze_multimodal,
    chunking selection, KG-skip) still see the user's original opt-ins after
    the parse-result record overwrites the pending_parse row.
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
            assert pre.get("canonical_basename") == "report.docx"
            assert pre.get("file_path") == "report.[native-iet!].docx"

            # Simulate a parse_* completion: pass only the fresh fields the
            # parsers actually emit and verify that pre-existing metadata
            # survives the upsert.
            await rag._persist_parsed_full_docs(
                doc_id,
                {
                    "content": "extracted body",
                    "file_path": "report.[native-iet!].docx",
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
            assert post.get("canonical_basename") == "report.docx"
            assert post.get("file_path") == "report.[native-iet!].docx"
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
@pytest.mark.xfail(
    reason=(
        "Native parsing now produces LIGHTRAG-format full_docs and "
        "content_hash is the MD5 of the .blocks.jsonl file. The writer "
        "embeds doc_id into every block's blockid (md5 of "
        "doc_id:idx:heading:content), so two distinct docx filenames "
        "deterministically produce different .blocks.jsonl bytes even when "
        "their underlying content is identical. Cross-document content_hash "
        "dedup for native docx is therefore architecturally impossible "
        "without changing the blockid scheme; tracked separately."
    ),
    strict=True,
)
def test_pending_parse_duplicate_hash_fails_and_archives_source(tmp_path, monkeypatch):
    """Two PENDING_PARSE docx files producing identical .blocks.jsonl content
    must be detected as content_hash duplicates and the loser archived.

    See xfail reason above for why this is currently impossible to satisfy
    without a blockid scheme change. Test body retained so that a future
    refactor of the blockid algorithm (e.g., dropping doc_id from the hash
    inputs) can flip this back to passing.
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
                "lightrag.native_parser.docx.lightrag_adapter.extract_docx_blocks",
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
    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")

    rules = "*:mineru;html:docling"
    validate_parser_routing_config(rules)
    assert resolve_file_parser_engine("paper.pdf", parser_rules=rules) == "mineru"
    assert resolve_file_parser_engine("index.html", parser_rules=rules) == "docling"
    assert resolve_file_parser_engine("notes.txt", parser_rules=rules) == "legacy"


@pytest.mark.offline
def test_parser_routing_validation_requires_external_endpoints(monkeypatch):
    monkeypatch.delenv("MINERU_ENDPOINT", raising=False)
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")

    with pytest.raises(ParserRoutingConfigError, match="MINERU_ENDPOINT"):
        validate_parser_routing_config("*:mineru;html:docling")

    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    with pytest.raises(ParserRoutingConfigError, match="DOCLING_ENDPOINT"):
        validate_parser_routing_config("*:mineru;html:docling")


@pytest.mark.offline
def test_parser_routing_validation_rejects_invalid_rules(monkeypatch):
    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")

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

        async def _fake_parse_native(doc_id, file_path, content_data):
            return {
                "doc_id": doc_id,
                "file_path": file_path,
                "parse_format": "raw",
                "content": "hello world",
                "blocks_path": "",
            }

        async def _fake_analyze(doc_id, file_path, parsed_data):
            parsed_data["multimodal_processed"] = True
            return parsed_data

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)
        monkeypatch.setattr("lightrag.pipeline.merge_nodes_and_edges", _fake_merge)
        monkeypatch.setattr(rag, "parse_native", _fake_parse_native)
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
        # No retry: VLM mock called exactly once.
        assert calls["n"] == 1
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
        await rag.analyze_multimodal(
            "doc-1", "demo.pdf", parsed, process_options="i"
        )

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
        await rag.analyze_multimodal(
            "doc-1", "demo.pdf", parsed, process_options="i"
        )

        payload = json.loads(drawings.read_text(encoding="utf-8"))
        result = payload["drawings"]["id1"]["llm_analyze_result"]
        assert result["status"] == "skipped"
        assert "smaller than" in result["message"]
        assert calls["n"] == 0

    asyncio.run(_run())


@pytest.mark.offline
def test_write_lightrag_document_preserves_headings_and_table_dimensions(
    tmp_path, monkeypatch
):
    async def _run():
        monkeypatch.setenv("INPUT_DIR", str(tmp_path))
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        source_path = tmp_path / "demo.docx"
        source_path.write_bytes(b"docx bytes")

        content_list = [
            {"type": "section_header", "text": "第一章 绪论", "level": 1},
            {"type": "section_header", "text": "1.1 研究背景", "level": 2},
            {"type": "text", "text": "这是正文段落。"},
            {
                "type": "table",
                "table_caption": ["表1 指标说明"],
                "table_body": {
                    "num_rows": 2,
                    "num_cols": 3,
                    "grid": [
                        [{"text": "符号"}, {"text": "含义"}, {"text": "单位"}],
                        [{"text": "A"}, {"text": "面积"}, {"text": "m2"}],
                    ],
                },
            },
            {
                "type": "image",
                "img_path": "/tmp/a.png",
                "image_caption": ["图1 架构图"],
            },
        ]

        parsed = await rag._write_lightrag_document_from_content_list(
            doc_id="doc-1",
            file_path="demo.docx",
            content_list=content_list,
            engine="docling",
            source_path=str(source_path),
        )

        blocks_path = Path(parsed["blocks_path"])
        assert blocks_path == (
            tmp_path / PARSED_DIR_NAME / "demo.docx.parsed" / "demo.blocks.jsonl"
        )
        assert not source_path.exists()
        assert (tmp_path / PARSED_DIR_NAME / source_path.name).exists()
        blocks = [
            json.loads(line)
            for line in blocks_path.read_text(encoding="utf-8").splitlines()
        ]
        content_blocks = blocks[1:]
        body_block = next(x for x in content_blocks if x["content"] == "这是正文段落。")
        table_block = next(
            x for x in content_blocks if 'refid="tb-doc-1-0001"' in x["content"]
        )
        image_block = next(
            x for x in content_blocks if 'id="dr-doc-1-0001"' in x["content"]
        )

        assert body_block["heading"] == "1.1 研究背景"
        assert body_block["parent_headings"] == ["第一章 绪论"]
        assert table_block["heading"] == "1.1 研究背景"
        assert image_block["heading"] == "1.1 研究背景"

        base = str(blocks_path)[: -len(".blocks.jsonl")]
        tables = json.loads(Path(base + ".tables.json").read_text(encoding="utf-8"))
        table_entry = tables["tables"]["tb-doc-1-0001"]
        assert table_entry["heading"] == "1.1 研究背景"
        assert table_entry["dimension"] == [2, 3]
        assert table_entry["format"] == "json"
        assert json.loads(table_entry["content"]) == [
            ["符号", "含义", "单位"],
            ["A", "面积", "m2"],
        ]

        drawings = json.loads(Path(base + ".drawings.json").read_text(encoding="utf-8"))
        assert drawings["drawings"]["dr-doc-1-0001"]["heading"] == "1.1 研究背景"

        full_doc = await rag.full_docs.get_by_id("doc-1")
        assert full_doc["lightrag_document_path"] == (
            "__parsed__/demo.docx.parsed/demo.blocks.jsonl"
        )

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_write_lightrag_document_strips_parser_hint_from_artifact_names(
    tmp_path, monkeypatch
):
    async def _run():
        monkeypatch.setenv("INPUT_DIR", str(tmp_path))
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            source_path = tmp_path / "demo.[native].docx"
            source_path.write_bytes(b"docx bytes")

            parsed = await rag._write_lightrag_document_from_content_list(
                doc_id="doc-hinted",
                file_path="demo.[native].docx",
                content_list=[{"type": "text", "text": "body"}],
                engine="native",
                source_path=str(source_path),
            )

            blocks_path = Path(parsed["blocks_path"])
            assert blocks_path == (
                tmp_path / PARSED_DIR_NAME / "demo.docx.parsed" / "demo.blocks.jsonl"
            )
            assert not source_path.exists()
            assert (tmp_path / PARSED_DIR_NAME / source_path.name).exists()
            full_doc = await rag.full_docs.get_by_id("doc-hinted")
            assert full_doc["lightrag_document_path"] == (
                "__parsed__/demo.docx.parsed/demo.blocks.jsonl"
            )
        finally:
            await rag.finalize_storages()

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
        await rag.analyze_multimodal(
            "doc-1", "demo.pdf", parsed, process_options="t"
        )

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
def test_parse_mineru_to_lightrag_document(tmp_path, monkeypatch):
    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()

        src_file = input_dir / "demo.pdf"
        src_file.write_bytes(b"fake-pdf")

        async def _fake_service(protocol, file_path):
            assert file_path == str(src_file)
            return json.dumps(
                {
                    "content": [
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
                        {
                            "type": "equation",
                            "text": "$$E=mc^2$$",
                        },
                    ]
                },
                ensure_ascii=False,
            )

        monkeypatch.setattr(rag, "_call_protocol_parse_service", _fake_service)
        monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")

        parsed = await rag.parse_mineru(
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
        assert full_doc["lightrag_document_path"] == str(
            blocks_path.relative_to(input_dir)
        )

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
        assert chunk["content"].startswith("- Image Name:")
        assert "- Image Type:\nChart" in chunk["content"]
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
        # covers chunk assembly.

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_mineru_empty_service_result_raises_without_fallback(
    tmp_path, monkeypatch
):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        src_file = tmp_path / "demo.pdf"
        src_file.write_bytes(b"fake-pdf")

        async def _fake_service(protocol, file_path):
            return None

        monkeypatch.setattr(rag, "_call_protocol_parse_service", _fake_service)
        monkeypatch.setenv("MINERU_ENDPOINT", "http://fake")

        with pytest.raises(ValueError, match="empty content"):
            await rag.parse_mineru(
                doc_id="doc-local-1",
                file_path=str(src_file),
                content_data={"content": "native fallback content"},
            )

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_docling_empty_service_result_raises_without_fallback(
    tmp_path, monkeypatch
):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        src_file = tmp_path / "demo.pptx"
        src_file.write_bytes(b"fake-pptx")

        async def _fake_service(protocol, file_path):
            return None

        monkeypatch.setattr(rag, "_call_protocol_parse_service", _fake_service)
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake")

        with pytest.raises(ValueError, match="empty content"):
            await rag.parse_docling(
                doc_id="doc-local-2",
                file_path=str(src_file),
                content_data={"content": "native fallback content"},
            )

        await rag.finalize_storages()

    asyncio.run(_run())
