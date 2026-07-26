"""Offline source-conflict repair tool (LR2 Phase 4-f, §5.5).

Runs against a real ``JsonDocStatusStorage`` so the tool is exercised through the
same contract every backend implements: bounded listing, a dry-run that mutates
nothing, and a commit whose compare-and-set token comes from that dry-run rather
than from a fabricated placeholder.
"""

from __future__ import annotations

import pytest

from lightrag.base import SourceConflict, SourceUnique
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.tools.source_conflict_repair import (
    collect_source_conflicts,
    repair_one_conflict,
)

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


def _row(file_path: str, status: str = "pending") -> dict:
    return {
        "status": status,
        "content_summary": "summary",
        "content_length": 3,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "file_path": file_path,
    }


async def _storage(tmp_path, rows: dict) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="conflict-cli",
    )
    await storage.initialize()
    async with storage._storage_lock:
        storage._data.update(rows)
    return storage


@pytest.mark.asyncio
async def test_listing_reports_only_multi_primary_keys(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _row("a.pdf"),
            "doc-2": _row("a.pdf", "failed"),
            "solo": _row("b.pdf"),
        },
    )
    conflicts = await collect_source_conflicts(storage, limit=10)

    assert [c.canonical_source_key for c in conflicts] == ["a.pdf"]
    assert conflicts[0].candidate_count == 2


@pytest.mark.asyncio
async def test_listing_walks_every_page_when_asked(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _row("a.pdf"),
            "doc-2": _row("a.pdf"),
            "doc-3": _row("b.pdf"),
            "doc-4": _row("b.pdf"),
            "doc-5": _row("c.pdf"),
            "doc-6": _row("c.pdf"),
        },
    )
    first_page = await collect_source_conflicts(storage, limit=2)
    assert len(first_page) == 2

    every = await collect_source_conflicts(storage, limit=2, all_pages=True)
    assert [c.canonical_source_key for c in every] == ["a.pdf", "b.pdf", "c.pdf"]


@pytest.mark.asyncio
async def test_dry_run_mutates_nothing(tmp_path):
    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    result = await repair_one_conflict(storage, "a.pdf", "doc-2")

    assert result.committed is False
    assert result.candidate_count == 2
    assert result.demoted_sample_doc_ids == ("doc-1",)
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)
    assert (await storage.get_by_id("doc-1")).get("metadata") is None


@pytest.mark.asyncio
async def test_apply_commits_with_the_token_from_its_own_dry_run(tmp_path):
    """The tool automates the copy-paste, not the guard: the commit still carries
    a count/fingerprint it just read, so the backend can refuse a moved set."""
    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    seen: list[dict] = []
    original = storage.repair_source_conflict

    async def _recording(key, **kwargs):
        seen.append(kwargs)
        return await original(key, **kwargs)

    storage.repair_source_conflict = _recording

    result = await repair_one_conflict(storage, "a.pdf", "doc-2", apply=True)

    assert result.committed is True
    assert [call["dry_run"] for call in seen] == [True, False]
    # The commit echoed the dry-run's token rather than a placeholder.
    assert seen[1]["expected_candidate_count"] == 2
    assert seen[1]["expected_candidate_fingerprint"] == result.fingerprint

    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
    demoted = await storage.get_by_id("doc-1")
    assert demoted["metadata"]["is_duplicate"] is True
    assert demoted["metadata"]["original_doc_id"] == "doc-2"
    # Content and status survive: only the source claim was dropped.
    assert demoted["status"] == "pending"
    assert demoted["content_summary"] == "summary"


@pytest.mark.asyncio
async def test_unknown_primary_is_refused_before_any_write(tmp_path):
    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    with pytest.raises(ValueError, match="not a current primary candidate"):
        await repair_one_conflict(storage, "a.pdf", "doc-typo", apply=True)

    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_commit_holds_the_enqueue_serialize_lock(tmp_path):
    """Fix-proof for the contract gap: no backend can block a NEW primary being
    inserted between the repair's re-read and its demotions, so the commit must
    hold the SAME workspace lock the enqueue critical section holds. Without it,
    an enqueue could interleave and the repair would still report committed=True
    on a key that is still in conflict.
    """
    import asyncio

    from lightrag.constants import ENQUEUE_SERIALIZE_LOCK_NAMESPACE
    from lightrag.kg.shared_storage import get_namespace_lock

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    workspace = "enqueue-ws"
    enqueue_ran: list[str] = []

    # Stand in for the enqueue critical section (filter_keys → dedup → upsert).
    async def _enqueue_critical_section():
        async with get_namespace_lock(
            ENQUEUE_SERIALIZE_LOCK_NAMESPACE, workspace=workspace
        ):
            enqueue_ran.append("enqueue")

    original = storage.repair_source_conflict
    observed_during_repair: list[list[str]] = []

    async def _observing(key, **kwargs):
        # Sampled inside the backend call — precisely the re-read → demote span
        # the CAS cannot protect.
        if not kwargs.get("dry_run", True):
            await asyncio.sleep(0)
            observed_during_repair.append(list(enqueue_ran))
        return await original(key, **kwargs)

    storage.repair_source_conflict = _observing

    async with get_namespace_lock(
        ENQUEUE_SERIALIZE_LOCK_NAMESPACE, workspace=workspace
    ):
        # The enqueue lock is held, so a correctly-locked commit cannot start.
        commit = asyncio.create_task(
            repair_one_conflict(
                storage, "a.pdf", "doc-2", workspace=workspace, apply=True
            )
        )
        await asyncio.sleep(0.05)
        assert not commit.done(), (
            "the commit ran while the enqueue-serialize lock was held — the "
            "phantom-insert window is still open"
        )

    result = await commit
    assert result.committed is True
    # And an enqueue cannot slip inside the backend's re-read → demote span.
    await _enqueue_critical_section()
    assert observed_during_repair == [[]]
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
