"""Scan rollback of failed custom-chunk operations (issue #3400, Phase 4).

``arollback_failed_custom_chunk_patches`` is scan's administrative escape
hatch: incomplete operations are rolled BACK to the previously committed
document state (the SDK caller owns roll-forward by repeating the call).

Real LightRAG object (JSON storages, offline), extraction monkeypatched to a
deterministic fake — same harness as test_custom_chunk_patch.py.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.operate import KGRebuildReport
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag.utils_pipeline import (
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    KG_RECOVERY_WARNINGS_METADATA_KEY,
    make_custom_chunk_id,
)

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"ccroll-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _fake_extraction(rag, monkeypatch):
    async def fake_extract(chunks, *args, **kwargs):
        results = []
        for chunk_id, payload in chunks.items():
            name = payload["content"].split()[0].upper()
            results.append(
                (
                    {
                        name: [
                            {
                                "entity_name": name,
                                "entity_type": "person",
                                "description": f"{name} description",
                                "source_id": chunk_id,
                                "file_path": "custom",
                                "timestamp": 1,
                            }
                        ]
                    },
                    {},
                )
            )
        return results

    monkeypatch.setattr(rag, "_process_extract_entities", fake_extract)


def _status_text(row: dict) -> str:
    raw = row.get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw)


def _journal(row: dict) -> dict | None:
    return (row.get("metadata") or {}).get(CUSTOM_CHUNK_PATCH_METADATA_KEY)


def _chunk_id(doc_key: str, content: str) -> str:
    return make_custom_chunk_id(doc_key, content)


async def _fail_one_merge(monkeypatch):
    calls = {"n": 0}
    orig_merge = lightrag_module.merge_nodes_and_edges

    async def merge_boom(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("merge boom")
        return await orig_merge(**kwargs)

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_boom)
    return calls


async def _fail_after_one_merge(monkeypatch):
    calls = {"n": 0}
    orig_merge = lightrag_module.merge_nodes_and_edges

    async def merge_then_boom(**kwargs):
        calls["n"] += 1
        await orig_merge(**kwargs)
        if calls["n"] == 1:
            raise RuntimeError("post-merge boom")

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_then_boom)
    return calls


async def _seed_base_then_fail_patch(rag, monkeypatch) -> None:
    """Base doc with ALICE committed, then a failed BOB patch left journaled."""
    _fake_extraction(rag, monkeypatch)
    await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")
    await _fail_one_merge(monkeypatch)
    with pytest.raises(RuntimeError, match="merge boom"):
        await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")


@pytest.mark.asyncio
async def test_rollback_restores_exact_pre_patch_state(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_base_then_fail_patch(rag, monkeypatch)
        staged_id = _chunk_id("doc-1", "bob is there")

        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": ["doc-1"], "failed": []}

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None
        assert row["chunks_list"] == [_chunk_id("doc-1", "alice is here")]

        # Staged chunk and its graph contribution are gone...
        assert await rag.text_chunks.get_by_id(staged_id) is None
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is None
        # ...the base contribution is untouched...
        assert await rag.chunk_entity_relation_graph.get_node("ALICE") is not None
        # ...and the anchors were pruned back to the base set.
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert anchors["entity_names"] == ["ALICE"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_missing_cache_rolls_back_with_structural_warning(tmp_path, monkeypatch):
    """Missing base extraction cache must not wedge rollback forever."""
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice base"], doc_id="doc-1")
        base_id = _chunk_id("doc-1", "alice base")
        staged_id = _chunk_id("doc-1", "alice patch")

        await _fail_after_one_merge(monkeypatch)
        with pytest.raises(RuntimeError, match="post-merge boom"):
            await rag.ainsert_custom_chunks("base", ["alice patch"], doc_id="doc-1")

        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": ["doc-1"], "failed": []}

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None
        assert row.get("error_msg") == ""
        warnings = row["metadata"][KG_RECOVERY_WARNINGS_METADATA_KEY]
        assert len(warnings) == 1
        warning = warnings[0]
        assert warning["code"] == "degraded_custom_chunk_rollback"
        assert warning["missing_cache_chunks"] == {
            "count": 1,
            "sample": [base_id],
        }
        assert warning["degraded_entities"] == {
            "count": 1,
            "sample": ["ALICE"],
        }

        node = await rag.chunk_entity_relation_graph.get_node("ALICE")
        assert node["source_id"] == base_id
        assert await rag.text_chunks.get_by_id(staged_id) is None

        # The warning is durable before finalize_storages can mask a missing
        # doc-status flush.
        import json

        status_files = list((tmp_path / "wd").rglob("kv_store_doc_status.json"))
        on_disk = json.loads(status_files[0].read_text())
        assert on_disk["doc-1"]["metadata"][KG_RECOVERY_WARNINGS_METADATA_KEY]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_warning_persistence_failure_keeps_journal_and_retry_converges(
    tmp_path, monkeypatch
):
    """A durable warning is part of rollback commit, not optional logging."""
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice base"], doc_id="doc-1")
        await _fail_after_one_merge(monkeypatch)
        with pytest.raises(RuntimeError, match="post-merge boom"):
            await rag.ainsert_custom_chunks("base", ["alice patch"], doc_id="doc-1")

        original_upsert = rag.doc_status.upsert
        failed = False

        async def fail_warning_upsert_once(data):
            nonlocal failed
            has_warning = any(
                (record.get("metadata") or {}).get(KG_RECOVERY_WARNINGS_METADATA_KEY)
                for record in data.values()
            )
            if has_warning and not failed:
                failed = True
                raise RuntimeError("warning persistence boom")
            return await original_upsert(data)

        monkeypatch.setattr(rag.doc_status, "upsert", fail_warning_upsert_once)

        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": [], "failed": ["doc-1"]}
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.FAILED.value
        assert _journal(row) is not None

        # The graph may already be structurally repaired and staged chunks may
        # already be gone. A journal retry must still rebuild its candidates,
        # persist the warning, and clear the journal idempotently.
        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": ["doc-1"], "failed": []}
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None
        assert row["metadata"][KG_RECOVERY_WARNINGS_METADATA_KEY]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_rollback_of_failed_create_removes_document(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await _fail_one_merge(monkeypatch)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("fresh", ["alice is here"], doc_id="doc-9")

        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": ["doc-9"], "failed": []}

        assert await rag.doc_status.get_by_id("doc-9") is None
        assert await rag.full_docs.get_by_id("doc-9") is None
        assert (
            await rag.text_chunks.get_by_id(_chunk_id("doc-9", "alice is here")) is None
        )
        assert await rag.full_entities.get_by_id("doc-9") is None

        # Codex review (PR #3416): doc_status.delete is deferred-commit on
        # the JSON backend — the rollback must flush it before reporting
        # success, or a crash right after would resurrect the FAILED journal
        # row on restart. Assert durability on DISK, before finalize (which
        # would flush and mask the bug).
        import json

        status_files = list((tmp_path / "wd").rglob("kv_store_doc_status.json"))
        assert status_files, "doc_status JSON file should exist"
        on_disk = json.loads(status_files[0].read_text())
        assert "doc-9" not in on_disk, (
            "create-rollback doc_status delete must be flushed to disk"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_create_rollback_warning_is_attached_to_surviving_owner(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice base"], doc_id="doc-1")

        await _fail_after_one_merge(monkeypatch)
        with pytest.raises(RuntimeError, match="post-merge boom"):
            await rag.ainsert_custom_chunks(
                "new", ["alice from create"], doc_id="doc-9"
            )

        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": ["doc-9"], "failed": []}
        assert await rag.doc_status.get_by_id("doc-9") is None

        owner_row = await rag.doc_status.get_by_id("doc-1")
        warnings = owner_row["metadata"][KG_RECOVERY_WARNINGS_METADATA_KEY]
        assert len(warnings) == 1
        assert warnings[0]["code"] == "degraded_custom_chunk_rollback"
        node = await rag.chunk_entity_relation_graph.get_node("ALICE")
        assert node["source_id"] == _chunk_id("doc-1", "alice base")
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_recovery_warning_deduplicates_and_keeps_latest_ten(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice base"], doc_id="doc-1")
        row = await rag.doc_status.get_by_id("doc-1")
        metadata = dict(row.get("metadata") or {})
        metadata[KG_RECOVERY_WARNINGS_METADATA_KEY] = [
            {
                "code": "degraded_custom_chunk_rollback",
                "operation_id": f"old-{index}",
            }
            for index in range(10)
        ]
        await rag.doc_status.upsert({"doc-1": {**row, "metadata": metadata}})

        report = KGRebuildReport(
            missing_cache_chunk_ids={_chunk_id("doc-1", "alice base")},
            degraded_entities={"ALICE": [_chunk_id("doc-1", "alice base")]},
        )
        status = {"latest_message": "", "history_messages": []}
        import asyncio

        lock = asyncio.Lock()
        journal = {"operation_id": "new-op"}
        await rag._persist_custom_chunk_recovery_warning(
            "doc-1",
            journal,
            report,
            mode="patch",
            pipeline_status=status,
            pipeline_status_lock=lock,
        )
        await rag._persist_custom_chunk_recovery_warning(
            "doc-1",
            journal,
            report,
            mode="patch",
            pipeline_status=status,
            pipeline_status_lock=lock,
        )

        updated = await rag.doc_status.get_by_id("doc-1")
        warnings = updated["metadata"][KG_RECOVERY_WARNINGS_METADATA_KEY]
        assert len(warnings) == 10
        assert [item["operation_id"] for item in warnings].count("new-op") == 1
        assert warnings[-1]["operation_id"] == "new-op"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_failed_rollback_keeps_journal_and_retries(tmp_path, monkeypatch):
    """A rollback failure must keep FAILED + journal (never report success),
    and a later rollback attempt succeeds."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed_base_then_fail_patch(rag, monkeypatch)

        purge_calls = {"n": 0}
        orig_purge = rag._purge_kg_contributions

        async def purge_boom(*args, **kwargs):
            purge_calls["n"] += 1
            if purge_calls["n"] == 1:
                raise RuntimeError("purge boom")
            return await orig_purge(*args, **kwargs)

        monkeypatch.setattr(rag, "_purge_kg_contributions", purge_boom)

        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": [], "failed": ["doc-1"]}
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.FAILED.value
        assert _journal(row) is not None, "journal must survive a failed rollback"

        # The next scan's rollback succeeds.
        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": ["doc-1"], "failed": []}
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_rollback_noop_without_journaled_documents(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")
        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": [], "failed": []}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_sdk_resume_still_possible_before_scan_rolls_back(tmp_path, monkeypatch):
    """Roll-forward stays the SDK caller's choice: if the same call resumes
    and commits before a scan runs, rollback then finds nothing to do."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed_base_then_fail_patch(rag, monkeypatch)

        # SDK retries the same input (merge restored after first boom).
        await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value

        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": [], "failed": []}
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert anchors["entity_names"] == ["ALICE", "BOB"]
    finally:
        await rag.finalize_storages()
