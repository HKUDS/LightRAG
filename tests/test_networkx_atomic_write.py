"""Regression tests for the atomic `write_nx_graph` path in NetworkXStorage.

Covers three concerns introduced when migrating from direct in-place write
to temp-file + os.replace:

1. Concurrent writers (multi-thread) all succeed and the final file parses.
2. A crash between `nx.write_graphml(tmp)` and `os.replace(tmp, dst)` leaves
   the prior snapshot intact on disk (atomicity guarantee).
3. The startup orphan-tmp reaper deletes tmp files older than the threshold
   without touching tmp files that may belong to a live concurrent writer.
"""

import os
import threading
import time
from unittest.mock import patch

import networkx as nx
import pytest

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
    """If os.replace fails after the tmp has been written (simulating a kill
    between write_graphml() and rename), the destination must still hold the
    previously-committed snapshot — never a torn/partial file."""
    dst = str(tmp_path / "graph_crash.graphml")

    # Commit v1.
    v1 = nx.Graph()
    v1.add_node("v1-node")
    NetworkXStorage.write_nx_graph(v1, dst, workspace="crash")
    assert os.path.exists(dst)

    # Attempt v2 with os.replace raising — simulates crash between
    # write_graphml() and the atomic rename.
    v2 = nx.Graph()
    v2.add_node("v2-node")
    with patch(
        "lightrag.kg.networkx_impl.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            NetworkXStorage.write_nx_graph(v2, dst, workspace="crash")

    # Destination must still be v1 — atomicity guarantee.
    reloaded = nx.read_graphml(dst)
    assert "v1-node" in reloaded.nodes
    assert "v2-node" not in reloaded.nodes

    # The orphan tmp from the simulated crash should still be on disk; the
    # reaper handles it on next startup (covered by the next test).
    orphans = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert len(orphans) == 1, f"Expected exactly one orphan tmp, got {orphans}"


@pytest.mark.offline
def test_reap_orphan_tmp_files_respects_age_threshold(tmp_path):
    """`_reap_orphan_tmp_files` must delete tmp siblings older than the
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
    threshold = NetworkXStorage._TMP_REAP_AGE_SECONDS
    aged_mtime = time.time() - (threshold + 60)
    os.utime(old_tmp, (aged_mtime, aged_mtime))

    NetworkXStorage._reap_orphan_tmp_files(dst, workspace="reap")

    assert not os.path.exists(old_tmp), "Aged orphan should have been reaped"
    assert os.path.exists(young_tmp), "Fresh tmp may be in-flight — must not be reaped"
    assert os.path.exists(
        unrelated
    ), "Unrelated path must not be touched by this reaper"
