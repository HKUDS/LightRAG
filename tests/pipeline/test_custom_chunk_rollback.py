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
import lightrag.operate as operate_module
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.operate import KGRebuildReport
from lightrag.utils import (
    LLM_TRUNCATION_METADATA_KEY,
    EmbeddingFunc,
    Tokenizer,
    TruncatedResponse,
)
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


async def _build_rag(tmp_path, **overrides) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"ccroll-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
        **overrides,
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
        assert result == {
            "rolled_back_count": 1,
            "failed_count": 0,
            "rolled_back_sample": ["doc-1"],
            "failed_sample": [],
        }

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
        assert result == {
            "rolled_back_count": 1,
            "failed_count": 0,
            "rolled_back_sample": ["doc-1"],
            "failed_sample": [],
        }

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
        assert result == {
            "rolled_back_count": 0,
            "failed_count": 1,
            "rolled_back_sample": [],
            "failed_sample": ["doc-1"],
        }
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.FAILED.value
        assert _journal(row) is not None

        # The graph may already be structurally repaired and staged chunks may
        # already be gone. A journal retry must still rebuild its candidates,
        # persist the warning, and clear the journal idempotently.
        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {
            "rolled_back_count": 1,
            "failed_count": 0,
            "rolled_back_sample": ["doc-1"],
            "failed_sample": [],
        }
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
        assert result == {
            "rolled_back_count": 1,
            "failed_count": 0,
            "rolled_back_sample": ["doc-9"],
            "failed_sample": [],
        }

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
        assert result == {
            "rolled_back_count": 1,
            "failed_count": 0,
            "rolled_back_sample": ["doc-9"],
            "failed_sample": [],
        }
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
        assert result == {
            "rolled_back_count": 0,
            "failed_count": 1,
            "rolled_back_sample": [],
            "failed_sample": ["doc-1"],
        }
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.FAILED.value
        assert _journal(row) is not None, "journal must survive a failed rollback"

        # The next scan's rollback succeeds.
        result = await rag.arollback_failed_custom_chunk_patches()
        assert result == {
            "rolled_back_count": 1,
            "failed_count": 0,
            "rolled_back_sample": ["doc-1"],
            "failed_sample": [],
        }
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
        assert result == {
            "rolled_back_count": 0,
            "failed_count": 0,
            "rolled_back_sample": [],
            "failed_sample": [],
        }
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
        assert result == {
            "rolled_back_count": 0,
            "failed_count": 0,
            "rolled_back_sample": [],
            "failed_sample": [],
        }
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert anchors["entity_names"] == ["ALICE", "BOB"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_rollback_candidate_discovery_uses_strict(tmp_path, monkeypatch):
    """Candidate discovery is scheduling-control-plane: it must query
    doc_status with strict=True so a mid-pagination / per-record backend
    failure aborts the rollback (raise) instead of rolling back a partial
    candidate set and reporting the scan done.  The scan-time caller catches
    the raise and keeps the journal/FAILED rows for the next scan."""
    rag = await _build_rag(tmp_path)
    try:
        # Legacy single-scan discovery path so the failure lands on
        # get_docs_by_statuses; the paged path (page_size>0) is likewise strict
        # (it passes strict=True to get_docs_by_statuses_page).
        rag.pipeline_scheduling_page_size = 0
        seen_strict: list[bool] = []

        async def _raising_query(statuses, strict=False):
            seen_strict.append(strict)
            if strict:
                raise RuntimeError("backend page failure")
            return {}  # relaxed would hide the failure behind a partial result

        monkeypatch.setattr(rag.doc_status, "get_docs_by_statuses", _raising_query)

        with pytest.raises(RuntimeError, match="backend page failure"):
            await rag.arollback_failed_custom_chunk_patches()

        assert seen_strict == [True]  # discovery used strict, not relaxed
    finally:
        await rag.finalize_storages()


async def _seed_journaled_docs(rag, monkeypatch, doc_ids: list[str]) -> None:
    """Leave a failed (journaled) custom-chunk patch on each doc id."""
    _fake_extraction(rag, monkeypatch)
    for doc_id in doc_ids:
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id=doc_id)
        await _fail_one_merge(monkeypatch)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id=doc_id)


@pytest.mark.asyncio
async def test_rollback_interleaves_with_paging(tmp_path, monkeypatch):
    """Fix-proof: the sweep paged its discovery but still collected EVERY
    journaled id before rolling anything back, so peak memory stayed
    proportional to the journaled-document count. Each page must now be rolled
    back before the next one is fetched.

    Shape-independent: measured by when rollbacks happen relative to the page
    fetches, not by the report the call returns.
    """
    rag = await _build_rag(tmp_path)
    try:
        doc_ids = ["doc-a", "doc-b", "doc-c"]
        await _seed_journaled_docs(rag, monkeypatch, doc_ids)
        rag.pipeline_scheduling_page_size = 1

        done = {"n": 0}
        original_rollback_one = rag._rollback_one_custom_chunk_patch
        original_page = rag.doc_status.get_docs_by_statuses_page
        # Rollbacks completed at the moment each page was requested.
        progress_at_fetch: list[int] = []

        async def counting_rollback_one(*args, **kwargs):
            result = await original_rollback_one(*args, **kwargs)
            done["n"] += 1
            return result

        async def observing_page(*args, **kwargs):
            progress_at_fetch.append(done["n"])
            return await original_page(*args, **kwargs)

        monkeypatch.setattr(
            rag, "_rollback_one_custom_chunk_patch", counting_rollback_one
        )
        monkeypatch.setattr(rag.doc_status, "get_docs_by_statuses_page", observing_page)

        result = await rag.arollback_failed_custom_chunk_patches()

        assert result["rolled_back_count"] == len(doc_ids)
        assert result["failed_count"] == 0
        # More than one page was needed (limit=1 over 3 journaled docs).
        assert len(progress_at_fetch) > 1
        # Collect-all-first would leave every fetch at 0 rollbacks done.
        assert progress_at_fetch[-1] > 0, (
            "no rollback had completed by the last page fetch — discovery is "
            "still collecting every journaled id before doing any work"
        )
        assert progress_at_fetch == sorted(progress_at_fetch)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_rollback_report_samples_are_capped(tmp_path, monkeypatch):
    """Counts stay exact while the id lists stop growing at the cap, so the
    report cannot reintroduce an O(journaled-docs) accumulation."""
    rag = await _build_rag(tmp_path)
    try:
        monkeypatch.setattr(lightrag_module, "ROLLBACK_REPORT_SAMPLE_CAP", 2)
        doc_ids = ["doc-a", "doc-b", "doc-c"]
        await _seed_journaled_docs(rag, monkeypatch, doc_ids)

        result = await rag.arollback_failed_custom_chunk_patches()

        assert result["rolled_back_count"] == 3
        assert len(result["rolled_back_sample"]) == 2
        assert set(result["rolled_back_sample"]) <= set(doc_ids)
        assert result["failed_count"] == 0
        assert result["failed_sample"] == []
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_resume_with_a_different_sample_keeps_both_attempts_candidates(
    tmp_path, monkeypatch
):
    """Codex review (PR #3607): truncated extraction results are never cached,
    so a resume re-runs the LLM and can extract a DIFFERENT sample. The first
    attempt's merge may have partially applied its candidates to the graph;
    the applying-phase journal write must UNION across attempts — replacing it
    stranded attempt-1-only objects from both the journal (rollback's anchor)
    and the commit's anchor union."""
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["carol is here"], doc_id="doc-1")

        # Patch attempt 1 extracts ALICE and dies inside the merge (after the
        # applying-phase journal write, after possibly partial graph writes).
        calls = {"n": 0}

        async def _per_attempt_extract(chunks, *args, **kwargs):
            calls["n"] += 1
            name = "ALICE" if calls["n"] == 1 else "BOB"
            return [
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
                for chunk_id in chunks
            ]

        monkeypatch.setattr(rag, "_process_extract_entities", _per_attempt_extract)
        await _fail_one_merge(monkeypatch)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["dave is there"], doc_id="doc-1")

        journal = _journal(await rag.doc_status.get_by_id("doc-1"))
        assert journal["entity_names"] == ["ALICE"], "precondition: attempt 1 journaled"

        # Resume: same input, but extraction now yields BOB (the truncated
        # first response was never cached, so the LLM genuinely re-runs).
        await rag.ainsert_custom_chunks("base", ["dave is there"], doc_id="doc-1")

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert "ALICE" in anchors["entity_names"], (
            "attempt 1's candidate vanished from the recovery anchors: a purge "
            "can no longer discover the objects its merge may have written"
        )
        assert "BOB" in anchors["entity_names"]
        assert "CAROL" in anchors["entity_names"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_resumed_create_unions_both_attempts_into_the_durable_anchors(
    tmp_path, monkeypatch
):
    """Codex review (PR #3607): the commit-time anchor union was restricted
    to patch mode. A create merge writes Phase 0 anchors from the CURRENT
    attempt's chunk_results only, so a resume that extracted a different
    sample (truncated responses are never cached) overwrote the anchors with
    attempt 2's candidates — and the PROCESSED write then cleared the
    journal, leaving attempt 1's possibly-merged objects named nowhere
    durable and stranded from later document purges."""
    rag = await _build_rag(tmp_path)
    try:
        attempts = {"n": 0}

        async def _per_attempt_extract(chunks, *args, **kwargs):
            attempts["n"] += 1
            name = "ALICE" if attempts["n"] == 1 else "BOB"
            return [_entity_result(name, chunk_id) for chunk_id in chunks]

        monkeypatch.setattr(rag, "_process_extract_entities", _per_attempt_extract)
        await _fail_one_merge(monkeypatch)

        # Attempt 1: CREATE fails inside the merge, after possibly partial
        # graph writes; the journal keeps ALICE.
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["carol is here"], doc_id="doc-1")

        # Resume: extraction genuinely re-runs and yields BOB; the create
        # merge's Phase 0 writes anchors from THIS attempt's results.
        await rag.ainsert_custom_chunks("base", ["carol is here"], doc_id="doc-1")

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None, "commit must clear the journal"
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert "ALICE" in anchors["entity_names"], (
            "attempt 1's candidate vanished from the durable anchors: with "
            "the journal cleared, a later purge can no longer discover the "
            "objects its merge may have written"
        )
        assert "BOB" in anchors["entity_names"]
    finally:
        await rag.finalize_storages()


def _entity_result(name: str, chunk_id: str) -> tuple[dict, dict]:
    return (
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


@pytest.mark.asyncio
async def test_a_clean_resume_keeps_the_failed_attempts_truncation_record(
    tmp_path, monkeypatch
):
    """Codex review (PR #3607): truncated extraction responses are never
    cached, so a resume re-runs the LLM and can come back clean — but the
    failed attempt's partial graph mutations stay (the resume is additive,
    it never purges them). The terminal write used to merge only the
    pre-operation snapshot with the CURRENT attempt's tally, so a clean
    resume erased the record of the truncated output still in the graph."""
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["carol is here"], doc_id="doc-1")

        attempts = {"n": 0}

        async def _per_attempt_extract(chunks, *args, truncation_tally=None, **kwargs):
            attempts["n"] += 1
            results = []
            for chunk_id in chunks:
                if attempts["n"] == 1:
                    # Attempt 1's response hit the token limit but was still
                    # parseable — extraction records it and proceeds.
                    truncation_tally.record("initial", chunk_id)
                results.append(
                    _entity_result("ALICE" if attempts["n"] == 1 else "BOB", chunk_id)
                )
            return results

        monkeypatch.setattr(rag, "_process_extract_entities", _per_attempt_extract)
        await _fail_one_merge(monkeypatch)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["dave is there"], doc_id="doc-1")

        # Resume: extraction genuinely re-runs (the truncated response was
        # never cached) and this time comes back clean.
        await rag.ainsert_custom_chunks("base", ["dave is there"], doc_id="doc-1")

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        record = (row.get("metadata") or {}).get(LLM_TRUNCATION_METADATA_KEY)
        assert record is not None, (
            "attempt 1 truncated and its partial output may still be in the "
            "graph, but the clean resume erased the document's truncation "
            "record"
        )
        assert record["stages"] == {"initial": 1}
        assert record["samples"] == [_chunk_id("doc-1", "dave is there")]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_failed_attempts_accumulate_truncation_exactly_once_each(
    tmp_path, monkeypatch
):
    """The accumulated operation record is recomputed from the journal AS
    LOADED (previous attempts only) plus the live tally. Recomputing from the
    journal copy the applying write already folded this attempt into would sum
    the attempt's events twice: a single failed attempt would journal 2 events
    instead of 1."""
    rag = await _build_rag(tmp_path)
    try:

        async def _truncating_extract(chunks, *args, truncation_tally=None, **kwargs):
            results = []
            for chunk_id in chunks:
                truncation_tally.record("initial", chunk_id)
                results.append(_entity_result("ALICE", chunk_id))
            return results

        monkeypatch.setattr(rag, "_process_extract_entities", _truncating_extract)

        async def merge_boom(**kwargs):
            raise RuntimeError("merge boom")

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_boom)

        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        journal = _journal(await rag.doc_status.get_by_id("doc-1"))
        record = journal["operation_llm_truncation"]
        assert record["events"] == 1, (
            "a single failed attempt must journal its one event exactly once — "
            "2 means the FAILED write re-folded the applying write's copy"
        )

        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        row = await rag.doc_status.get_by_id("doc-1")
        record = _journal(row)["operation_llm_truncation"]
        assert record["events"] == 2
        assert record["stages"] == {"initial": 2}
        # Same input, same chunk id: the subject deduplicates while the
        # per-attempt events keep counting.
        assert record["affected"] == 1
        assert record["samples"] == [_chunk_id("doc-1", "alice is here")]

        # The FAILED row's durable metadata carries the same accumulation.
        meta = (row.get("metadata") or {}).get(LLM_TRUNCATION_METADATA_KEY)
        assert meta is not None and meta["events"] == 2
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_hard_crash_after_summary_truncation_survives_resume(
    tmp_path, monkeypatch
):
    """End-to-end Codex scenario (PR #3607): attempt 1 truncates at extraction
    AND at a description summary, its graph writes land, then the process dies
    before the FAILED bookkeeping can run. The resume extracts a different,
    clean sample, never re-touches the truncated entity, and succeeds — the
    PROCESSED row must still carry both stages, fed by the write-ahead
    journal alone (no FAILED write ever happened)."""
    rag = await _build_rag(tmp_path, force_llm_summary_on_merge=3)
    try:
        attempts = {"n": 0}

        def _descriptions(name: str, chunk_id: str, count: int) -> list[dict]:
            return [
                {
                    "entity_name": name,
                    "entity_type": "person",
                    "description": f"{name} description {i}",
                    "source_id": chunk_id,
                    "file_path": "custom",
                    "timestamp": i,
                }
                for i in range(count)
            ]

        async def _per_attempt_extract(chunks, *args, truncation_tally=None, **kwargs):
            attempts["n"] += 1
            results = []
            for chunk_id in chunks:
                if attempts["n"] == 1:
                    # Truncated extraction (uncached, so the resume genuinely
                    # re-runs it) yielding an entity whose three descriptions
                    # force the LLM summary path — which truncates too.
                    truncation_tally.record("initial", chunk_id)
                    results.append(({"ALICE": _descriptions("ALICE", chunk_id, 3)}, {}))
                else:
                    results.append(({"BOB": _descriptions("BOB", chunk_id, 1)}, {}))
            return results

        monkeypatch.setattr(rag, "_process_extract_entities", _per_attempt_extract)

        async def truncated_summary_llm(*args, **kwargs):
            return TruncatedResponse("partial summary"), 0

        monkeypatch.setattr(
            operate_module, "use_llm_func_with_cache", truncated_summary_llm
        )

        # Attempt 1 fails after the merge completed its graph writes, and the
        # "process" is gone before the FAILED bookkeeping runs: that write
        # raises, leaving the journal exactly as the write-ahead left it.
        await _fail_after_one_merge(monkeypatch)
        orig_upsert_status = rag._upsert_custom_chunk_status

        async def crash_on_failed_write(doc_key, status, **kwargs):
            if status == DocStatus.FAILED:
                raise RuntimeError("simulated hard crash")
            return await orig_upsert_status(doc_key, status, **kwargs)

        monkeypatch.setattr(rag, "_upsert_custom_chunk_status", crash_on_failed_write)

        with pytest.raises(RuntimeError, match="post-merge boom"):
            await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        # Resume: clean single-description sample, no summaries, succeeds.
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        record = (row.get("metadata") or {}).get(LLM_TRUNCATION_METADATA_KEY)
        assert record is not None, "both attempts' truncations were erased"
        assert record["stages"] == {"initial": 1, "summary": 1}, (
            "attempt 1's summary truncation was lost: only the FAILED write "
            "knew about it, and a hard crash never runs the FAILED write"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_cancellation_between_merge_phases_keeps_the_journaled_summary_event(
    tmp_path, monkeypatch
):
    """Codex review (PR #3607): a cancellation landing on an inter-phase
    await point inside the merge used to skip the summary-tally absorb; the
    FAILED bookkeeping then recomputed the journal from the unabsorbed
    operation tally, OVERWRITING the write-ahead snapshot that did carry the
    summary event — while the truncated summary itself stayed in the graph."""
    import asyncio

    rag = await _build_rag(tmp_path, force_llm_summary_on_merge=3)
    try:

        async def extract_alice(chunks, *args, truncation_tally=None, **kwargs):
            return [
                (
                    {
                        "ALICE": [
                            {
                                "entity_name": "ALICE",
                                "entity_type": "person",
                                "description": f"ALICE description {i}",
                                "source_id": chunk_id,
                                "file_path": "custom",
                                "timestamp": i,
                            }
                            for i in range(3)
                        ]
                    },
                    {},
                )
                for chunk_id in chunks
            ]

        monkeypatch.setattr(rag, "_process_extract_entities", extract_alice)

        async def truncated_summary_llm(*args, **kwargs):
            return TruncatedResponse("partial summary"), 0

        monkeypatch.setattr(
            operate_module, "use_llm_func_with_cache", truncated_summary_llm
        )

        orig_append = operate_module.append_pipeline_history

        def cancel_at_phase2(status, message):
            if str(message).startswith("Phase 2"):
                raise asyncio.CancelledError()
            return orig_append(status, message)

        monkeypatch.setattr(operate_module, "append_pipeline_history", cancel_at_phase2)

        with pytest.raises(asyncio.CancelledError):
            await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        journal = _journal(await rag.doc_status.get_by_id("doc-1"))
        record = journal["operation_llm_truncation"]
        assert record is not None and "summary" in (record.get("stages") or {}), (
            "the FAILED bookkeeping overwrote the write-ahead journal snapshot "
            "with the unabsorbed tally, erasing the summary event while the "
            "truncated summary stayed in the graph"
        )
    finally:
        await rag.finalize_storages()
