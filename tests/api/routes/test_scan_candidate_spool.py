"""The scan's disposable candidate spool (LR2 §8.2/§8.4).

The spool is where ``/documents/scan`` keeps the O(files-in-INPUT_DIR) state
that exact oldest-file-first ordering requires, so that Python memory stays
O(SCAN_ENQUEUE_BATCH_SIZE). Pinned here:

* it lands on real disk under the operator's chosen directory, and degrades to
  the OS temp dir with a warning rather than failing the scan;
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


def test_base_dir_prefers_scan_spool_dir_then_working_dir(tmp_path, monkeypatch):
    rag = type("_Rag", (), {"working_dir": str(tmp_path / "rag_storage")})()

    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    assert _scan_spool_base_dir(rag) == tmp_path / "rag_storage" / "scan_spool"

    monkeypatch.setattr(
        _document_routes.global_args,
        "scan_spool_dir",
        str(tmp_path / "fast-disk"),
        raising=False,
    )
    assert _scan_spool_base_dir(rag) == tmp_path / "fast-disk"


def test_base_dir_falls_back_to_the_os_temp_dir_without_a_working_dir(monkeypatch):
    """``None`` means "let tempfile decide" — the last resort, not an error."""
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    assert _scan_spool_base_dir(type("_Rag", (), {})()) is None


def test_spool_is_created_under_the_requested_dir_and_removed_on_close(tmp_path):
    base = tmp_path / "rag_storage" / "scan_spool"
    spool = _ScanCandidateSpool(4, base)
    try:
        assert base.exists()
        created = list(base.iterdir())
        assert len(created) == 1
        assert (created[0] / "candidates.sqlite3").exists()
    finally:
        spool.close()

    # Disposable: the scan's ordering state does not outlive the scan.
    assert list(base.iterdir()) == []


def test_an_unusable_base_dir_warns_and_still_produces_a_working_spool(
    tmp_path, caplog
):
    """A misconfigured SCAN_SPOOL_DIR must not fail the whole scan."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("regular file", encoding="utf-8")

    lightrag_logger = importlib.import_module("lightrag.utils").logger
    previous = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger=lightrag_logger.name):
            spool = _ScanCandidateSpool(4, blocker / "spool")
    finally:
        lightrag_logger.propagate = previous

    try:
        assert "SCAN_SPOOL_DIR" in caplog.text
        claimer = asyncio.run(spool.claim(tmp_path / "a.txt", "a.txt", 1))
        assert claimer is None
        assert _drain(spool, 4) == [["a.txt"]]
    finally:
        spool.close()


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

        assert [(c.path.name, c.size) for c in batches[0]] == [
            ("sized.txt", 10),
            ("vanished.txt", None),
        ]


def test_uncommitted_claims_are_visible_to_ordered_batches(tmp_path):
    """The commit interval is a write-batching detail, not a visibility rule: a
    scan that discovers fewer files than one interval must still enqueue them."""
    path = _aged(tmp_path / "lonely.txt", 1_600_000_000)

    # Interval far above the single claim below, so nothing has been committed.
    with _ScanCandidateSpool(1000, tmp_path / "spool") as spool:
        asyncio.run(spool.claim(path, "lonely.txt", 1))

        assert _drain(spool, 8) == [["lonely.txt"]]
