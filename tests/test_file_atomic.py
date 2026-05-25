"""Tests for ``lightrag.file_atomic`` — the shared atomic-write helpers.

These tests cover the helper in isolation. End-to-end coverage of the
individual storage backends that build on it lives in
``test_networkx_atomic_write.py``, ``test_atomic_write_write_json.py``,
``test_atomic_write_faiss.py``, and ``test_atomic_write_nano.py``.
"""

import os
import stat
import sys
import threading
import time
from unittest.mock import patch

import pytest

from lightrag.file_atomic import (
    TMP_REAP_AGE_SECONDS,
    atomic_write,
    reap_orphan_tmp_files,
    tmp_path_for,
)


@pytest.mark.offline
def test_tmp_path_for_unique_across_concurrent_writers():
    """Mirror the production call pattern — each real writer calls
    ``tmp_path_for`` once per atomic_write. Across N concurrent writers,
    every tmp path must be distinct so no writer's ``os.replace`` can hit a
    sibling that another writer already renamed away.

    Same-thread back-to-back calls inside one ns tick are intentionally not
    tested: production never does that (write_fn does real IO between calls)
    and ``time.time_ns()`` resolution on some platforms (notably macOS) is
    coarse enough that a tight loop will collide."""
    paths: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(16)

    def collect():
        barrier.wait()
        p = tmp_path_for("/tmp/x")
        with lock:
            paths.append(p)

    threads = [threading.Thread(target=collect) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(paths) == len(set(paths)), "tmp_path_for must be unique across writers"
    for p in paths:
        assert p.startswith("/tmp/x.tmp.")


@pytest.mark.offline
def test_atomic_write_publishes_file_via_replace(tmp_path):
    dst = str(tmp_path / "out.txt")

    def writer(tmp):
        with open(tmp, "w") as f:
            f.write("hello")

    atomic_write(dst, writer)
    assert open(dst).read() == "hello"
    assert [p for p in os.listdir(tmp_path) if ".tmp." in p] == []


@pytest.mark.offline
def test_atomic_write_write_fn_exception_cleans_tmp_and_preserves_prior(tmp_path):
    """If ``write_fn`` raises, the prior destination must survive and the
    tmp must be removed by ``atomic_write``'s ``finally``."""
    dst = str(tmp_path / "out.txt")

    def commit_v1(tmp):
        with open(tmp, "w") as f:
            f.write("v1")

    atomic_write(dst, commit_v1)

    def boom(tmp):
        with open(tmp, "w") as f:
            f.write("partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        atomic_write(dst, boom)

    assert open(dst).read() == "v1"
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"write_fn exception must clean tmp, got {leftovers}"


@pytest.mark.offline
def test_atomic_write_replace_exception_cleans_tmp_and_preserves_prior(tmp_path):
    """If ``os.replace`` raises, the prior destination must survive and the
    tmp must be removed."""
    dst = str(tmp_path / "out.txt")

    def commit(tmp, payload):
        with open(tmp, "w") as f:
            f.write(payload)

    atomic_write(dst, lambda tmp: commit(tmp, "v1"))

    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            atomic_write(dst, lambda tmp: commit(tmp, "v2"))

    assert open(dst).read() == "v1"
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"os.replace exception must clean tmp, got {leftovers}"


@pytest.mark.offline
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX chmod semantics")
def test_atomic_write_preserves_existing_mode(tmp_path):
    """The inode swap done by ``os.replace`` would otherwise inherit fresh
    tmp permissions and silently widen a 0600 destination."""
    dst = str(tmp_path / "secret.txt")

    atomic_write(dst, lambda tmp: open(tmp, "w").write("seed"))
    os.chmod(dst, 0o600)
    assert stat.S_IMODE(os.stat(dst).st_mode) == 0o600

    atomic_write(dst, lambda tmp: open(tmp, "w").write("updated"))
    assert stat.S_IMODE(os.stat(dst).st_mode) == 0o600
    assert open(dst).read() == "updated"


@pytest.mark.offline
def test_reap_orphan_tmp_files_respects_age_and_locality(tmp_path):
    """Aged tmp siblings get reaped; fresh ones (potentially belonging to a
    live concurrent writer) and unrelated paths are left alone."""
    dst = str(tmp_path / "data.json")
    old_tmp = f"{dst}.tmp.111.222.333"
    young_tmp = f"{dst}.tmp.444.555.666"
    unrelated = str(tmp_path / "other.json.tmp.999")

    for p in (old_tmp, young_tmp, unrelated):
        with open(p, "w") as fh:
            fh.write("partial")

    aged_mtime = time.time() - (TMP_REAP_AGE_SECONDS + 60)
    os.utime(old_tmp, (aged_mtime, aged_mtime))

    reap_orphan_tmp_files(dst)

    assert not os.path.exists(old_tmp)
    assert os.path.exists(young_tmp)
    assert os.path.exists(unrelated)


@pytest.mark.offline
def test_reap_orphan_tmp_files_handles_glob_metacharacters(tmp_path):
    """``file_name`` is composed from workspace + namespace, both of which
    can legitimately contain glob metacharacters on POSIX. The reaper must
    match literally — not miss the real orphan because ``[v2]`` parses as a
    character class, nor widen its pattern to match unrelated siblings."""
    dst = str(tmp_path / "data_[v2].json")
    real_orphan = f"{dst}.tmp.111.222.333"
    decoy = str(tmp_path / "data_v.json.tmp.unrelated")

    for p in (real_orphan, decoy):
        with open(p, "w") as fh:
            fh.write("partial")

    aged_mtime = time.time() - (TMP_REAP_AGE_SECONDS + 60)
    for p in (real_orphan, decoy):
        os.utime(p, (aged_mtime, aged_mtime))

    reap_orphan_tmp_files(dst)

    assert not os.path.exists(real_orphan)
    assert os.path.exists(decoy), "Reaper must not match siblings of an unrelated path"


@pytest.mark.offline
def test_reap_orphan_tmp_files_extra_patterns_clean_legacy_residue(tmp_path):
    """The default ``.tmp.*`` pattern intentionally does not match a bare
    trailing ``.tmp`` (the historical Faiss meta suffix). ``extra_patterns``
    is the migration path for those residues."""
    import glob

    dst = str(tmp_path / "meta.json")
    legacy_tmp = f"{dst}.tmp"

    with open(legacy_tmp, "w") as fh:
        fh.write("legacy partial")

    aged_mtime = time.time() - (TMP_REAP_AGE_SECONDS + 60)
    os.utime(legacy_tmp, (aged_mtime, aged_mtime))

    # Default pattern leaves the legacy residue.
    reap_orphan_tmp_files(dst)
    assert os.path.exists(legacy_tmp)

    # Explicit migration pattern clears it.
    reap_orphan_tmp_files(dst, extra_patterns=(glob.escape(dst) + ".tmp",))
    assert not os.path.exists(legacy_tmp)
