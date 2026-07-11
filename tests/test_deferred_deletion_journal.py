"""Durable state for deferred batch document deletion.

These tests deliberately use the filesystem-backed journal because it is the
least-common-denominator durable store across LightRAG's supported backends.
"""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from lightrag.deletion_journal import (
    DeferredDeletionJournal,
    DeferredDeletionJournalStore,
    DeferredDeletionStage,
)


def _journal() -> DeferredDeletionJournal:
    return DeferredDeletionJournal.new(
        batch_id="batch-test-001",
        document_ids=["doc-a", "doc-b"],
        delete_llm_cache=False,
    )


@pytest.mark.parametrize(
    "workspace",
    [r"..\escaped", "..", "../escaped"],
)
def test_workspace_directory_cannot_escape_journal_root(tmp_path: Path, workspace: str):
    store = DeferredDeletionJournalStore(tmp_path, workspace=workspace)
    journal_root = (tmp_path / "deletion_journals").resolve()

    assert (
        store.directory
        == DeferredDeletionJournalStore(tmp_path, workspace=workspace).directory
    )
    assert store.directory.resolve().is_relative_to(journal_root)
    assert "\\" not in store.directory.name
    assert store.directory.name not in {".", ".."}


def test_workspace_directory_preserves_safe_workspace_name(tmp_path: Path):
    store = DeferredDeletionJournalStore(tmp_path, workspace="team-alpha")

    assert store.directory == tmp_path / "deletion_journals" / "team-alpha"


@pytest.mark.parametrize(
    "batch_id",
    [
        "CON",
        "nul",
        "COM1",
        "LPT9",
        "CON.txt",
        "com1.json",
        "LPT9.foo",
        "CLOCK$",
        "clock$.json",
        "a:b",
        "a*",
        "a?b",
        "batch.",
        "batch ",
    ],
)
def test_batch_id_rejects_windows_unsafe_path_components(tmp_path: Path, batch_id: str):
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")

    with pytest.raises(ValueError, match="invalid batch_id"):
        store.load(batch_id)


def test_journal_round_trip_survives_store_recreation(tmp_path: Path):
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    journal = _journal()
    journal.aggregate_targets(
        entities={"Shared": ["chunk-a", "chunk-b"]},
        relationships={("Shared", "Other"): ["chunk-a"]},
    )
    journal.mark_rebuild_pending()

    store.save(journal)

    recovered = DeferredDeletionJournalStore(tmp_path, workspace="default").load(
        journal.batch_id
    )
    assert recovered is not None
    assert recovered.stage is DeferredDeletionStage.REBUILD_PENDING
    assert recovered.document_ids == ["doc-a", "doc-b"]
    assert recovered.entities_to_rebuild == {"Shared": ["chunk-a", "chunk-b"]}
    assert recovered.relationships_to_rebuild == {("Other", "Shared"): ["chunk-a"]}


def test_journal_persists_delete_file_intent_and_snapshot_source_paths(tmp_path: Path):
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    journal = DeferredDeletionJournal.new(
        "delete-source-files", ["doc-a"], False, delete_file=True
    )
    journal.source_file_paths = {"doc-a": "originals/a.pdf"}
    store.save(journal)

    recovered = DeferredDeletionJournalStore(tmp_path, workspace="default").load(
        journal.batch_id
    )

    assert recovered is not None
    assert recovered.delete_file is True
    assert recovered.source_file_paths == {"doc-a": "originals/a.pdf"}


def test_journal_aggregates_shared_targets_as_a_union(tmp_path: Path):
    journal = _journal()

    journal.aggregate_targets(
        entities={"Shared": ["chunk-a", "chunk-b"], "First": ["chunk-a"]},
        relationships={("Shared", "Other"): ["chunk-a"]},
    )
    journal.aggregate_targets(
        entities={"Shared": ["chunk-b", "chunk-c"], "Second": ["chunk-c"]},
        relationships={("Other", "Shared"): ["chunk-b", "chunk-c"]},
    )

    assert journal.entities_to_rebuild == {
        "Shared": ["chunk-a", "chunk-b", "chunk-c"],
        "First": ["chunk-a"],
        "Second": ["chunk-c"],
    }
    assert journal.relationships_to_rebuild == {
        ("Other", "Shared"): ["chunk-a", "chunk-b", "chunk-c"]
    }


def test_failed_rebuild_is_retryable_and_keeps_recovery_metadata(tmp_path: Path):
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    journal = _journal()
    journal.document_metadata = {
        "doc-a": {"file_path": "a.pdf", "chunks_list": ["chunk-a"]}
    }
    journal.mark_rebuild_pending()
    store.save(journal)

    journal.mark_failed_retryable("rebuild failed")
    store.save(journal)

    recovered = store.load(journal.batch_id)
    assert recovered is not None
    assert recovered.stage is DeferredDeletionStage.FAILED_RETRYABLE
    assert recovered.error_detail == "rebuild failed"
    assert recovered.attempt_count == 1
    assert recovered.document_metadata["doc-a"]["chunks_list"] == ["chunk-a"]


def test_commit_is_only_valid_after_rebuild_pending():
    journal = _journal()

    with pytest.raises(ValueError, match="FINALIZATION_PENDING"):
        journal.mark_committed()

    journal.mark_rebuild_pending()
    journal.mark_finalization_pending()
    journal.mark_committed()
    assert journal.stage is DeferredDeletionStage.COMMITTED


def test_store_lists_only_unfinished_journals(tmp_path: Path):
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    committed = DeferredDeletionJournal.new("committed", ["doc-a"], False)
    committed.mark_rebuild_pending()
    committed.mark_finalization_pending()
    committed.mark_committed()
    retryable = DeferredDeletionJournal.new("retryable", ["doc-b"], False)
    retryable.mark_failed_retryable("rebuild failed")
    store.save(committed)
    store.save(retryable)

    assert [journal.batch_id for journal in store.list_unfinished()] == ["retryable"]


def test_store_uses_private_regular_files_and_rejects_symlinked_journals(
    tmp_path: Path,
):
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    journal = _journal()
    store.save(journal)
    path = store._path(journal.batch_id)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    path.unlink()
    os.symlink(target, path)
    with pytest.raises(OSError, match="symlink"):
        store.load(journal.batch_id)


@pytest.mark.asyncio
async def test_batch_lock_is_released_after_lock_holder_process_crashes(tmp_path: Path):
    """OS-owned lock releases on process death; retained lock file is harmless."""
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    store.save(_journal())
    script = """
import asyncio
import sys
from lightrag.deletion_journal import DeferredDeletionJournalStore

async def main():
    store = DeferredDeletionJournalStore(sys.argv[1], "default")
    async with store.batch_lock("batch-test-001"):
        print("LOCKED", flush=True)
        await asyncio.sleep(60)

asyncio.run(main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "LOCKED"
    finally:
        process.terminate()
        process.wait(timeout=10)

    async def acquire_after_crash() -> None:
        async with store.batch_lock("batch-test-001"):
            return None

    await asyncio.wait_for(acquire_after_crash(), timeout=3)


@pytest.mark.asyncio
async def test_cancelled_contended_lock_acquisition_does_not_orphan_descriptor(
    tmp_path: Path,
):
    """Cancelling a waiter cannot leave a later-acquired OS lock held forever."""
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    store.save(_journal())
    script = """
import asyncio
import sys
from lightrag.deletion_journal import DeferredDeletionJournalStore

async def main():
    store = DeferredDeletionJournalStore(sys.argv[1], "default")
    async with store.batch_lock("batch-test-001"):
        print("LOCKED", flush=True)
        await asyncio.sleep(60)

asyncio.run(main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "LOCKED"

        async def wait_for_lock() -> None:
            async with store.batch_lock("batch-test-001"):
                return None

        waiter = asyncio.create_task(wait_for_lock())
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    finally:
        process.terminate()
        process.wait(timeout=10)

    async def acquire_after_cancelled_waiter() -> None:
        async with store.batch_lock("batch-test-001"):
            return None

    await asyncio.wait_for(acquire_after_cancelled_waiter(), timeout=3)


def test_windows_lock_path_uses_msvcrt_lock_and_unlock_when_available(
    tmp_path, monkeypatch
):
    """Exercise the Windows branch without requiring a Windows CI worker."""
    import lightrag.deletion_journal as journal_module

    calls = []

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(descriptor, operation, length):
            calls.append((descriptor, operation, length))

    monkeypatch.setattr(journal_module, "_is_windows", lambda: True)
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)
    store = DeferredDeletionJournalStore(tmp_path, workspace="default")
    lock_path = tmp_path / "deletion_journals" / "default" / ".windows-lock.lock"
    lock_path.parent.mkdir(parents=True)

    descriptor = store._acquire_os_lock(lock_path)
    store._release_os_lock(descriptor)

    assert [operation for _, operation, _ in calls] == [
        FakeMsvcrt.LK_LOCK,
        FakeMsvcrt.LK_UNLCK,
    ]
