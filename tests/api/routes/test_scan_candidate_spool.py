"""The scan's disposable candidate spool (LR2 §8.2/§8.4).

The spool is where ``/documents/scan`` keeps the O(files-in-INPUT_DIR) state
that exact oldest-file-first ordering requires, so that Python memory stays
O(SCAN_ENQUEUE_BATCH_SIZE). Pinned here:

* it lands on real disk under the operator's chosen, per-workspace directory,
  and an unusable directory fails the scan rather than relocating to a possibly
  RAM-backed OS temp dir;
* its fixed name caps what ``kill -9`` can leave on a persistent volume at one
  spool, reclaimed by the next scan;
* a UNIQUE canonical key gives scan-wide first-physical-claim-wins, and names
  the winner by FULL PATH (two subdirectories can hold the same basename);
* ``unknown_source`` carries no identity, so such candidates never claim each
  other's slot;
* a row that fails to land for any OTHER reason raises instead of silently
  dropping a discovered file;
* reads are bounded pages, and each candidate carries the size discovery
  already stat'ed.
"""

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

_ScanCandidateSpool = _document_routes._ScanCandidateSpool
_scan_spool_base_dir = _document_routes._scan_spool_base_dir
UNKNOWN_FILE_SOURCE = _document_routes.UNKNOWN_FILE_SOURCE

pytestmark = pytest.mark.offline


def _aged(path: Path, epoch: int, body: str = "body") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    os.utime(path, (epoch, epoch))
    return path


def _drain(spool: _ScanCandidateSpool, batch_size: int) -> list[list[str]]:
    return [
        [candidate.path.name for candidate in batch]
        for batch in spool.ordered_batches(batch_size)
    ]


# --------------------------------------------------------------------------
# Where the spool lives
# --------------------------------------------------------------------------


def _rag(working_dir: str | None = None, workspace: str = ""):
    attrs = {"workspace": workspace}
    if working_dir is not None:
        attrs["working_dir"] = working_dir
    return type("_Rag", (), attrs)()


def test_base_dir_prefers_scan_spool_dir_then_working_dir(tmp_path, monkeypatch):
    rag = _rag(str(tmp_path / "rag_storage"))

    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    assert (
        _scan_spool_base_dir(rag) == tmp_path / "rag_storage" / "scan_spool" / "unnamed"
    )

    monkeypatch.setattr(
        _document_routes.global_args,
        "scan_spool_dir",
        str(tmp_path / "fast-disk"),
        raising=False,
    )
    assert _scan_spool_base_dir(rag) == tmp_path / "fast-disk" / "unnamed"


def test_each_workspace_gets_its_own_spool_directory(tmp_path, monkeypatch):
    """Two workspaces may share one WORKING_DIR, and the spool file name inside
    the directory is fixed — so the directory itself must separate them."""
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    base = tmp_path / "rag_storage" / "scan_spool"

    assert _scan_spool_base_dir(_rag(str(tmp_path / "rag_storage"), "alpha")) == (
        base / "named" / "alpha"
    )
    assert _scan_spool_base_dir(_rag(str(tmp_path / "rag_storage"), "beta")) == (
        base / "named" / "beta"
    )


def test_no_named_workspace_can_collide_with_the_unnamed_one(tmp_path, monkeypatch):
    """The workspace → directory mapping must be injective.

    ``validate_workspace`` accepts any single path component, so a bare
    sentinel is reachable as a real workspace name: with ``_default`` as the
    stand-in for "", a workspace literally called ``_default`` mapped to the
    same directory. They hold different per-workspace scan locks and can run
    concurrently, so the second one's pre-open discard would delete the first
    one's live database. The type tag (``unnamed`` vs ``named/<ws>``) makes the
    collision unrepresentable.
    """
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    working_dir = str(tmp_path / "rag_storage")

    unnamed = _scan_spool_base_dir(_rag(working_dir, ""))
    for lookalike in ("_default", "unnamed", "named", "default"):
        assert _scan_spool_base_dir(_rag(working_dir, lookalike)) != unnamed


def test_the_unnamed_and_a_lookalike_workspace_hold_spools_concurrently(
    tmp_path, monkeypatch
):
    """End-to-end regression for the collision: two workspaces that can run
    concurrently must not delete each other's live database.

    The directories come from ``_scan_spool_base_dir`` on purpose — resolving
    them by hand would test the spool in isolation and pass even while the
    mapping collided, which is where the defect actually lived.

    Asserted on the inode rather than on the drained rows: POSIX keeps an
    unlinked-but-open file readable, so the victim would still return its own
    candidates while its file had in fact been replaced on disk.
    """
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    working_dir = str(tmp_path / "rag_storage")
    unnamed_dir = _scan_spool_base_dir(_rag(working_dir, ""))
    named_dir = _scan_spool_base_dir(_rag(working_dir, "_default"))

    with _ScanCandidateSpool(4, unnamed_dir) as unnamed:
        asyncio.run(unnamed.claim(tmp_path / "a.txt", "a.txt", 1))
        unnamed_db = unnamed_dir / "candidates.sqlite3"
        inode_before = unnamed_db.stat().st_ino

        # Opening the second spool must not touch the first one's file.
        with _ScanCandidateSpool(4, named_dir) as named:
            asyncio.run(named.claim(tmp_path / "b.txt", "b.txt", 1))

            assert unnamed_db.stat().st_ino == inode_before
            assert _drain(unnamed, 4) == [["a.txt"]]
            assert _drain(named, 4) == [["b.txt"]]

        # And closing one does not reclaim the other's still-live database.
        assert unnamed_db.exists()


def test_a_traversing_workspace_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    with pytest.raises(ValueError):
        _scan_spool_base_dir(_rag(str(tmp_path / "rag_storage"), "../escape"))


def test_base_dir_is_none_only_without_any_working_dir(monkeypatch):
    """A real LightRAG always has working_dir; ``None`` is the test-rig path."""
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    assert _scan_spool_base_dir(_rag(None)) is None


def test_spool_is_created_under_the_requested_dir_and_removed_on_close(tmp_path):
    base = tmp_path / "rag_storage" / "scan_spool" / "unnamed"
    spool = _ScanCandidateSpool(4, base)
    try:
        assert (base / "candidates.sqlite3").exists()
    finally:
        spool.close()

    # Disposable: the scan's ordering state does not outlive the scan.
    assert list(base.glob("candidates.sqlite3*")) == []


def test_an_unusable_base_dir_fails_the_scan_instead_of_relocating(tmp_path):
    """Fail-closed placement.

    Relocating to the OS temp dir is not a degraded-but-correct mode: where
    /tmp is tmpfs it silently restores the O(number of files) RAM cost the
    spool exists to remove, and the operator learns about it as an OOM kill
    partway through a large scan instead of an error naming the directory.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("regular file", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SCAN_SPOOL_DIR"):
        _ScanCandidateSpool(4, blocker / "spool")


def test_a_crashed_scans_leftover_spool_is_reclaimed_not_accumulated(tmp_path):
    """``kill -9`` skips close(); nothing else reclaims a persistent volume.

    A fixed name per workspace caps the residue at one spool: the next scan
    deletes what it finds before opening its own, rather than adding to it.
    """
    base = tmp_path / "scan_spool" / "unnamed"
    base.mkdir(parents=True)
    leftover = base / "candidates.sqlite3"
    leftover.write_bytes(b"corpse of a killed scan")
    (base / "candidates.sqlite3-journal").write_bytes(b"stale journal")

    with _ScanCandidateSpool(4, base) as spool:
        # One spool, not two: the stale journal is gone and the database file
        # was reopened in place rather than a second directory being minted.
        assert list(base.glob("candidates.sqlite3*")) == [leftover]
        # And the corpse's bytes did not survive into this scan's database.
        assert asyncio.run(spool.claim(tmp_path / "a.txt", "a.txt", 1)) is None
        assert _drain(spool, 4) == [["a.txt"]]

    assert list(base.glob("candidates.sqlite3*")) == []


# --------------------------------------------------------------------------
# Claim semantics
# --------------------------------------------------------------------------


def test_first_claim_wins_and_the_loser_is_told_the_winners_full_path(tmp_path):
    """Two subdirectories can hold the same basename, so "x.txt duplicates
    x.txt" would be useless: the conflict names the winning full path."""
    winner = _aged(tmp_path / "one" / "same.txt", 1_600_000_000)
    loser = _aged(tmp_path / "two" / "same.txt", 1_600_000_001)

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        assert asyncio.run(spool.claim(winner, "same.txt", 1)) is None
        claimer = asyncio.run(spool.claim(loser, "same.txt", 2))

        assert claimer == str(winner)
        assert claimer != str(loser)
        # Only the winner is staged; the caller archives the loser.
        assert _drain(spool, 4) == [["same.txt"]]


def test_unknown_source_candidates_never_claim_each_others_slot(tmp_path):
    """``unknown_source`` is "identity cannot be claimed", not an identity.

    SQLite UNIQUE permits multiple NULLs, which is exactly this semantics: every
    such candidate must be staged rather than deduplicated against the others.
    """
    first = _aged(tmp_path / "a.txt", 1_600_000_000)
    second = _aged(tmp_path / "b.txt", 1_600_000_001)
    third = _aged(tmp_path / "c.txt", 1_600_000_002)

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        for index, path in enumerate((first, second, third), start=1):
            assert asyncio.run(spool.claim(path, UNKNOWN_FILE_SOURCE, index)) is None

        assert _drain(spool, 8) == [["a.txt", "b.txt", "c.txt"]]


def test_a_rejected_row_without_a_canonical_claim_raises(tmp_path):
    """``INSERT OR IGNORE`` swallows EVERY constraint violation. Only a
    duplicate canonical key is a legitimate rejection; anything else (here a
    reused sequence) would silently drop a discovered file."""
    first = _aged(tmp_path / "a.txt", 1_600_000_000)
    second = _aged(tmp_path / "b.txt", 1_600_000_001)

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        assert asyncio.run(spool.claim(first, UNKNOWN_FILE_SOURCE, 7)) is None

        with pytest.raises(RuntimeError, match="without a canonical claim"):
            asyncio.run(spool.claim(second, UNKNOWN_FILE_SOURCE, 7))


# --------------------------------------------------------------------------
# Ordering and bounded reads
# --------------------------------------------------------------------------


def test_batches_are_globally_ordered_and_bounded_by_the_page_size(tmp_path):
    paths = [
        _aged(tmp_path / f"f{index}.txt", 1_600_000_000 + index) for index in range(5)
    ]
    arrival = [paths[3], paths[0], paths[4], paths[1], paths[2]]

    with _ScanCandidateSpool(2, tmp_path / "spool") as spool:
        for index, path in enumerate(arrival, start=1):
            asyncio.run(spool.claim(path, path.name, index))

        assert _drain(spool, 2) == [
            ["f0.txt", "f1.txt"],
            ["f2.txt", "f3.txt"],
            ["f4.txt"],
        ]


def test_an_unreadable_mtime_sorts_last_without_losing_the_candidate(tmp_path):
    readable = _aged(tmp_path / "middle.txt", 1_600_000_000)
    gone_z = tmp_path / "z-gone.txt"
    gone_a = tmp_path / "a-gone.txt"

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        for index, path in enumerate((gone_z, readable, gone_a), start=1):
            asyncio.run(spool.claim(path, path.name, index))

        # Missing timestamps lose priority, never their place in the queue, and
        # tie-break deterministically on the path sort key.
        assert _drain(spool, 8) == [["middle.txt", "a-gone.txt", "z-gone.txt"]]


def test_the_candidate_carries_the_size_discovery_stated(tmp_path):
    sized = _aged(tmp_path / "sized.txt", 1_600_000_000, body="0123456789")
    vanished = tmp_path / "vanished.txt"

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        asyncio.run(spool.claim(sized, "sized.txt", 1))
        asyncio.run(spool.claim(vanished, "vanished.txt", 2))

        batches = list(spool.ordered_batches(8))

        # 0, never None: "already read, unknown" — so the enqueue path does
        # not go back to the filesystem for a file that just failed to stat.
        assert [(c.path.name, c.size) for c in batches[0]] == [
            ("sized.txt", 10),
            ("vanished.txt", 0),
        ]


def test_uncommitted_claims_are_visible_to_ordered_batches(tmp_path):
    """The commit interval is a write-batching detail, not a visibility rule: a
    scan that discovers fewer files than one interval must still enqueue them."""
    path = _aged(tmp_path / "lonely.txt", 1_600_000_000)

    # Interval far above the single claim below, so nothing has been committed.
    with _ScanCandidateSpool(1000, tmp_path / "spool") as spool:
        asyncio.run(spool.claim(path, "lonely.txt", 1))

        assert _drain(spool, 8) == [["lonely.txt"]]
