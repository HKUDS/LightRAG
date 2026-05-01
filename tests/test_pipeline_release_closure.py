import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocStatus
from lightrag.constants import (
    FULL_DOCS_CONTENT_STORED_IN_LIGHTRAG_DOCUMENT,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_DIR_NAME,
    PARSER_ENGINE_NATIVE,
)
from lightrag.operate import _get_relationship_vdb_timeout_seconds
from lightrag.parser_routing import (
    ParserRoutingConfigError,
    canonicalize_parser_hinted_basename,
    resolve_file_parser_engine,
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
def test_parse_engine_routing_by_filename_and_env(tmp_path, monkeypatch):
    rag = _new_rag(tmp_path)
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")
    assert rag._resolve_parser_engine("a.[docling-iet].docx", {}) == "docling"

    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:mineru-iet,*:native")
    assert rag._resolve_parser_engine("paper.pdf", {}) == "mineru"
    assert (
        rag._resolve_parser_engine("paper.pdf", {"parsed_engine": "native"}) == "legacy"
    )


@pytest.mark.offline
def test_parse_engine_rule_fallback_and_default_legacy(tmp_path, monkeypatch):
    rag = _new_rag(tmp_path)
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:native,*:legacy")
    assert rag._resolve_parser_engine("paper.pdf", {}) == "legacy"

    monkeypatch.setenv("LIGHTRAG_PARSER", "pptx:docling,*:legacy")
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    assert rag._resolve_parser_engine("slides.pptx", {}) == "legacy"

    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    monkeypatch.setenv("MINERU_ENDPOINT", "")
    assert rag._resolve_parser_engine("slides.pptx", {}) == "legacy"


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

            await rag.apipeline_enqueue_documents(
                "changed body",
                file_paths="/tmp/abc.[native].docx",
                track_id="track-b",
            )
            assert (await rag.full_docs.get_by_id(first_id))["content"] == "alpha body"

            failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            assert any(
                getattr(doc, "metadata", {}).get("duplicate_kind") == "filename"
                and getattr(doc, "file_path", "") == "abc.docx"
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
                parsed_engine=PARSER_ENGINE_NATIVE,
                track_id="track-delete-source",
            )

            doc_id = compute_mdhash_id("abc.docx", prefix="doc-")
            result = await rag.adelete_by_doc_id(doc_id)

            assert result.status == "success"
            assert result.file_path == "abc.docx"
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
            await rag.apipeline_enqueue_documents(
                FULL_DOCS_CONTENT_STORED_IN_LIGHTRAG_DOCUMENT,
                file_paths="first.lightrag",
                docs_format="lightrag",
                lightrag_document_paths="__parsed__/doc.blocks.jsonl",
                track_id="track-a",
            )
            await rag.apipeline_enqueue_documents(
                FULL_DOCS_CONTENT_STORED_IN_LIGHTRAG_DOCUMENT,
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
                    "format": "raw",
                    "parsed_engine": "native",
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
    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        rag = _new_rag(tmp_path / "work")
        await rag.initialize_storages()
        try:
            content = "same extracted body"
            await rag.apipeline_enqueue_documents(
                content,
                file_paths="existing.txt",
                track_id="track-original",
            )
            original_id = compute_mdhash_id("existing.txt", prefix="doc-")
            original_status = await rag.doc_status.get_by_id(original_id)
            original_status["status"] = DocStatus.PROCESSED
            await rag.doc_status.upsert({original_id: original_status})

            source_path = input_dir / "duplicate.docx"
            source_path.write_bytes(b"docx bytes")
            parse_document = __import__(
                "lightrag.extraction.parse_document",
                fromlist=["parse_docx_to_interchange_jsonl"],
            )
            monkeypatch.setattr(
                parse_document,
                "parse_docx_to_interchange_jsonl",
                lambda file_bytes, source_file, doc_id, output_dir: content,
            )

            async def _fail_extract(*args, **kwargs):
                raise AssertionError("duplicate document should not reach extraction")

            monkeypatch.setattr(rag, "_process_extract_entities", _fail_extract)

            await rag.apipeline_enqueue_documents(
                "",
                file_paths=str(source_path),
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parsed_engine=PARSER_ENGINE_NATIVE,
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
                "format": "raw",
                "content": "hello world",
                "blocks_path": "",
            }

        async def _fake_analyze(doc_id, file_path, parsed_data):
            parsed_data["multimodal_processed"] = True
            return parsed_data

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)
        monkeypatch.setattr("lightrag.lightrag.merge_nodes_and_edges", _fake_merge)
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
def test_analyze_multimodal_json_retry_and_writeback(tmp_path):
    async def _run():
        calls = {"n": 0}

        async def _retry_vlm(prompt, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return "not-json"
            return '{"name":"Figure A","image_type":"Chart","summary":"ok","detail_description":"details"}'

        rag = _new_rag(tmp_path, vlm_llm_model_func=_retry_vlm)
        # 1x1 transparent PNG for grounded image-path flow
        img_path = tmp_path / "img1.png"
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
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed)

        meta = json.loads(blocks.read_text(encoding="utf-8").splitlines()[0])
        assert meta.get("analyze_time")

        drawings_payload = json.loads(drawings.read_text(encoding="utf-8"))
        result = drawings_payload["drawings"]["id1"]["llm_analyze_result"]
        assert result["name"] == "Figure A"
        assert result["summary"] == "ok"
        assert result["detail_description"] == "details"
        assert calls["n"] >= 2

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
        result = await rag.analyze_multimodal("doc-1", "demo.pdf", parsed)

        meta = json.loads(blocks.read_text(encoding="utf-8").splitlines()[0])
        assert meta.get("analyze_time")
        assert result["analyze_time"]
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
def test_analyze_multimodal_normalizes_string_grounded_to_bool(tmp_path):
    async def _run():
        async def _vlm(_prompt, **_kwargs):
            return json.dumps(
                {
                    "name": "Figure A",
                    "image_type": "diagram",
                    "summary": "ok",
                    "detail_description": "details",
                    "grounded": "true",
                    "grounding_reason": "visual_evidence",
                },
                ensure_ascii=False,
            )

        rag = _new_rag(tmp_path, vlm_llm_model_func=_vlm)
        img_path = tmp_path / "img1.png"
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
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed)

        payload = json.loads(drawings.read_text(encoding="utf-8"))
        result = payload["drawings"]["id1"]["llm_analyze_result"]
        assert result["grounded"] is True
        assert isinstance(result["grounded"], bool)

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_multimodal_without_image_uses_conservative_output(tmp_path):
    async def _run():
        async def _vlm(_prompt, **_kwargs):
            return '{"name":"X","summary":"hallucinated rich details","detail_description":"very specific claims","grounded":false}'

        rag = _new_rag(tmp_path, vlm_llm_model_func=_vlm)

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
                        "id1": {"id": "id1", "caption": "图1 测试图", "footnotes": []}
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
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed)

        payload = json.loads(drawings.read_text(encoding="utf-8"))
        result = payload["drawings"]["id1"]["llm_analyze_result"]
        assert result["grounded"] is False
        assert "Conservative summary only" in result["summary"]
        assert "missing_image" in result["detail_description"]

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
        async def _vlm(_prompt, **_kwargs):
            return json.dumps(
                {
                    "name": "指标表",
                    "summary": "该表展示符号、代表意义和单位的对应关系。",
                    "detail_description": "表格包含三列，分别为符号、代表意义和单位，列出了 A、F、e 等符号。",
                    "grounded": True,
                    "grounding_reason": "textual_content_only",
                },
                ensure_ascii=False,
            )

        rag = _new_rag(tmp_path, vlm_llm_model_func=_vlm)

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
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed)

        payload = json.loads(tables.read_text(encoding="utf-8"))
        result = payload["tables"]["id1"]["llm_analyze_result"]
        assert result["grounded"] is True
        assert result["grounding_reason"] == "textual_content_only"
        assert "Conservative summary only" not in result["summary"]
        assert "符号、代表意义和单位" in result["summary"]

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

        assert parsed["format"] == "lightrag"
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
        assert full_doc["format"] == "lightrag"
        assert full_doc["content"] == FULL_DOCS_CONTENT_STORED_IN_LIGHTRAG_DOCUMENT
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
                            "llm_analyze_result": {
                                "name": "系统架构图",
                                "image_type": "Chart",
                                "summary": "展示系统模块",
                                "detail_description": "模块交互关系",
                            },
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        mm_chunks, mm_specs = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=2,
        )
        assert len(mm_chunks) == 1
        assert mm_chunks[0]["content"].startswith("Image_Name:")
        assert len(mm_specs) == 1
        assert mm_specs[0]["entity_name"].startswith("drawing-")

        # Simulate extracted entities from same mm chunk
        chunk_id = mm_specs[0]["chunk_id"]
        chunk_results = [
            (
                {
                    "系统模块": [
                        {
                            "entity_name": "系统模块",
                            "entity_type": "concept",
                            "description": "模块",
                            "source_id": chunk_id,
                            "file_path": "demo.pdf",
                            "timestamp": 1,
                        }
                    ]
                },
                {},
            )
        ]
        augmented = rag._augment_chunk_results_with_mm_entities(
            chunk_results=chunk_results,
            mm_specs=mm_specs,
            file_path="demo.pdf",
        )
        assert len(augmented) == 2
        mm_nodes, mm_edges = augmented[-1]
        assert any(k.startswith("drawing-") for k in mm_nodes.keys())
        assert mm_edges  # has relation from modality object to extracted entity

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
