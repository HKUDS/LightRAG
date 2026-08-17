"""Regression tests for the atomic `write_nx_graph` path in NetworkXStorage.

After the migration to the shared ``lightrag.file_atomic`` helper, two
failure modes have distinct tmp-residue semantics:

- Python-level exceptions (``write_fn`` raised, ``os.replace`` raised, ...):
  ``atomic_write``'s ``finally`` removes the tmp best-effort and the prior
  snapshot remains on disk.
- Process-level kills (SIGKILL, OOM, hard reboot): ``finally`` does not run
  and the tmp survives as an orphan; the startup reaper handles it.

Covers concerns introduced when migrating from direct in-place write to
temp-file + os.replace:

1. Concurrent writers (multi-thread) all succeed and the final file parses.
2. A crash between ``write_fn(tmp)`` and ``os.replace`` leaves the prior
   snapshot intact on disk (atomicity guarantee, Python-exception path).
3. The startup orphan-tmp reaper deletes tmp files older than the threshold
   without touching tmp files that may belong to a live concurrent writer.
   This is what salvages real SIGKILL/OOM residue.
4. The rename preserves the destination file's existing permission bits
   instead of widening them to the umask default.
5. The orphan reaper handles glob metacharacters in file_name correctly
   (workspace/namespace strings can legitimately contain '[', '*', '?').
"""

import os
import stat
import sys
import threading
import time
from unittest.mock import patch

import networkx as nx
import pytest

from lightrag.file_atomic import TMP_REAP_AGE_SECONDS, reap_orphan_tmp_files
from lightrag.kg.networkx_impl import NetworkXStorage


@pytest.mark.offline
def test_write_nx_graph_concurrent_writers_no_race(tmp_path):
    """5 threads racing on the same destination must all succeed; the final
    file must parse, and no per-writer .tmp sibling may be left behind."""
    dst = str(tmp_path / "graph_concurrent.graphml")
    errors: list[BaseException] = []
    barrier = threading.Barrier(5)

    def writer(tid: int) -> None:
        try:
            g = nx.Graph()
            g.add_node(f"node-{tid}", label=f"label-{tid}")
            barrier.wait()
            NetworkXStorage.write_nx_graph(g, dst, workspace=f"w{tid}")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent writers raised: {errors}"

    # Final file must parse cleanly.
    loaded = nx.read_graphml(dst)
    assert loaded.number_of_nodes() == 1

    # No per-writer tmp residue.
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Unexpected tmp residue: {leftovers}"


@pytest.mark.offline
def test_write_nx_graph_crash_preserves_prior_snapshot(tmp_path):
    """If ``os.replace`` raises after the tmp is written, the destination
    must still hold the prior snapshot — never a torn/partial file. Because
    the failure is a Python-level exception, ``atomic_write`` runs its
    ``finally`` and removes the tmp; the startup-reaper path is exercised
    separately in ``test_reap_orphan_tmp_files_respects_age_threshold``."""
    dst = str(tmp_path / "graph_crash.graphml")

    # Commit v1.
    v1 = nx.Graph()
    v1.add_node("v1-node")
    NetworkXStorage.write_nx_graph(v1, dst, workspace="crash")
    assert os.path.exists(dst)

    # Attempt v2 with os.replace raising. Patch the symbol in file_atomic
    # because the rename now lives in the shared helper, not in
    # networkx_impl.
    v2 = nx.Graph()
    v2.add_node("v2-node")
    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            NetworkXStorage.write_nx_graph(v2, dst, workspace="crash")

    # Destination must still be v1 — atomicity guarantee.
    reloaded = nx.read_graphml(dst)
    assert "v1-node" in reloaded.nodes
    assert "v2-node" not in reloaded.nodes

    # Python-exception path: tmp must have been cleaned up by atomic_write's
    # finally. Real SIGKILL/OOM residue is the reaper's job, not this one.
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Python-exception path must clean tmp, got {leftovers}"


@pytest.mark.offline
def test_reap_orphan_tmp_files_respects_age_threshold(tmp_path):
    """`reap_orphan_tmp_files` must delete tmp siblings older than the
    threshold and leave younger ones (potentially belonging to a live
    concurrent writer) untouched."""
    dst = str(tmp_path / "graph_reap.graphml")
    old_tmp = f"{dst}.tmp.111.222.333"
    young_tmp = f"{dst}.tmp.444.555.666"
    unrelated = str(tmp_path / "graph_other.graphml.tmp.999")

    for p in (old_tmp, young_tmp, unrelated):
        with open(p, "w") as fh:
            fh.write("partial xml")

    # Age old_tmp past the threshold; leave young_tmp and unrelated fresh.
    aged_mtime = time.time() - (TMP_REAP_AGE_SECONDS + 60)
    os.utime(old_tmp, (aged_mtime, aged_mtime))

    reap_orphan_tmp_files(dst, workspace="reap")

    assert not os.path.exists(old_tmp), "Aged orphan should have been reaped"
    assert os.path.exists(young_tmp), "Fresh tmp may be in-flight — must not be reaped"
    assert os.path.exists(unrelated), (
        "Unrelated path must not be touched by this reaper"
    )


@pytest.mark.offline
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX chmod semantics")
def test_write_nx_graph_preserves_existing_file_mode(tmp_path):
    """If the destination already exists with restrictive mode (e.g. 0600),
    the atomic write must not silently widen it to umask defaults — the
    inode swap done by os.replace would otherwise inherit the tmp file's
    fresh permissions."""
    dst = str(tmp_path / "graph_mode.graphml")

    g = nx.Graph()
    g.add_node("a")
    NetworkXStorage.write_nx_graph(g, dst, workspace="mode")
    os.chmod(dst, 0o600)
    assert stat.S_IMODE(os.stat(dst).st_mode) == 0o600

    # Second write — mode must survive.
    g.add_node("b")
    NetworkXStorage.write_nx_graph(g, dst, workspace="mode")
    assert stat.S_IMODE(os.stat(dst).st_mode) == 0o600, (
        "atomic write must preserve dst permissions across the rename"
    )

    # And content was actually updated.
    assert nx.read_graphml(dst).number_of_nodes() == 2


@pytest.mark.offline
def test_reap_orphan_tmp_files_handles_glob_metacharacters(tmp_path):
    """`file_name` is composed from `working_dir` and `namespace`, both of
    which can legitimately contain glob metacharacters on POSIX. The reaper
    must match literally — not (a) miss the real orphan because '[v2]' was
    parsed as a character class, nor (b) widen its pattern to match
    unrelated tmp files belonging to a sibling storage."""
    dst = str(tmp_path / "graph_[v2].graphml")
    real_orphan = f"{dst}.tmp.111.222.333"
    decoy = str(tmp_path / "graph_v.graphml.tmp.unrelated")

    for p in (real_orphan, decoy):
        with open(p, "w") as fh:
            fh.write("partial xml")

    # Age both past the threshold so age is not the discriminator here —
    # the only thing protecting the decoy is correct pattern escaping.
    aged_mtime = time.time() - (TMP_REAP_AGE_SECONDS + 60)
    for p in (real_orphan, decoy):
        os.utime(p, (aged_mtime, aged_mtime))

    reap_orphan_tmp_files(dst, workspace="meta")

    assert not os.path.exists(real_orphan), (
        "Reaper must match the real orphan even when path contains '['"
    )
    assert os.path.exists(decoy), (
        "Reaper must not match tmp files belonging to unrelated paths"
    )
