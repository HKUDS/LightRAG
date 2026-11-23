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
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage


@pytest.mark.offline
def test_write_nx_graph_concurrent_writers_no_race(tmp_path):
    """5 threads racing on the same destination must all succeed; the final
    file must parse, and no per-writer .tmp sibling may be left behind."""
    dst = str(tmp_path / "graph_concurrent.graphml")
    barrier = threading.Barrier(5)

    def writer(tid: int) -> None:
        g = nx.Graph()
        g.add_node(f"node-{tid}", label=f"label-{tid}")
        barrier.wait()
        NetworkXStorage.write_nx_graph(g, dst, workspace=f"w{tid}")

    # Consuming Executor.map re-raises the exception of the first failing
    # task in submission order, with its original traceback.
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(writer, range(5)))

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
