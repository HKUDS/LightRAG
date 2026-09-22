"""One process tree at a time on a file-backed configuration directory.

The in-process guard (``tests/kg/json_impl/test_json_init_claim.py``) cannot
see another SERVER: each process tree has its own in-memory copy, so two of
them on one ``working_dir`` each rewrite the whole configuration file from a
private view. An overwritten baseline reads back as absent, and absent is what
lets a start bootstrap -- so the protection disappears with nothing in any log.

These pin the claim that closes it, including the two properties that make an
OS lock the right instrument rather than a PID file: the kernel releases it
when the holder dies, and ``fork`` shares it so a Gunicorn master's workers
inherit rather than fight it.

Every "other process" here is a real subprocess (``_working_dir_lock_probe.py``
in a fresh interpreter), never a ``fork`` of the pytest process: a session
that has run thousands of tests is multi-threaded, and forking it makes CPython
emit a DeprecationWarning that cannot be promoted to an error, only filtered.
The probe is single-threaded, so it can assert the warning ABSENT. Each spawn
costs about a second, which the whole file pays seven times.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightrag.kg import working_dir_lock as wdl
from lightrag.kg.working_dir_lock import (
    LOCK_FILENAME,
    acquire_working_dir_lock,
    holds_working_dir_lock,
    release_working_dir_lock,
    uses_working_dir,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _no_claims_leak():
    yield
    for claim in list(wdl._claims.values()):
        try:
            claim.handle.close()
        except OSError:
            pass
    wdl._claims.clear()


_PROBE = Path(__file__).with_name("_working_dir_lock_probe.py")


def _probe(command: str, path) -> subprocess.CompletedProcess:
    """Run one probe command in a fresh interpreter -- a process tree that did
    NOT inherit this one's bookkeeping or its lock descriptor."""
    result = subprocess.run(
        [sys.executable, str(_PROBE), command, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"probe '{command}' failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _foreign_attempt(path) -> str:
    return _probe("foreign", path).stdout.strip()


def test_only_a_file_backed_configuration_storage_claims_the_directory():
    """A server-backed configuration keeps its rows on the server, so two
    servers there share a container by design and nothing is claimed."""
    assert uses_working_dir("JsonKVStorage") is True
    assert uses_working_dir("RedisKVStorage") is False
    assert uses_working_dir("PGKVStorage") is False
    assert uses_working_dir(None) is False


def test_a_second_process_tree_is_refused(tmp_path):
    acquire_working_dir_lock(str(tmp_path))
    assert _foreign_attempt(tmp_path) == "REFUSED"


def test_the_directory_is_free_once_the_holder_releases(tmp_path):
    acquire_working_dir_lock(str(tmp_path))
    release_working_dir_lock(str(tmp_path))

    assert holds_working_dir_lock(str(tmp_path)) is False
    assert _foreign_attempt(tmp_path) == "ADMITTED"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_forked_workers_inherit_the_masters_claim(tmp_path):
    """The Gunicorn shape: the master claims BEFORE forking, so the workers
    find the claim in their own tree and count themselves in.

    Taken after the fork instead, each worker would open its own descriptor
    and all but one would be refused -- which is why ``on_starting`` is the
    hook that takes it.

    The master is the probe process, not pytest: the probe also fails if the
    fork emitted CPython's multi-threaded-fork warning (see its docstring).
    """
    admitted = _probe("workers", tmp_path).stdout.split()

    assert admitted == ["OK"] * 4


def test_a_holder_that_dies_leaves_nothing_to_reap(tmp_path):
    """Why an OS lock and not a PID file: no stale-file cleanup, no
    read-the-PID-then-probe-liveness race. The kernel releases it."""
    _probe("dying", tmp_path)

    # The lock file still exists; that is not what holds the lock.
    assert (tmp_path / LOCK_FILENAME).exists()
    acquire_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is True


def test_a_second_instance_in_one_tree_counts_itself_in(tmp_path):
    """Several LightRAG instances in one process are legal, so the claim is
    counted and released only by the last one."""
    acquire_working_dir_lock(str(tmp_path))
    acquire_working_dir_lock(str(tmp_path))

    release_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is True
    assert _foreign_attempt(tmp_path) == "REFUSED"

    release_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_a_filesystem_that_cannot_lock_warns_and_proceeds(tmp_path, monkeypatch):
    """NFSv3 without lockd, SMB/CIFS. Refusing there would break deployments
    that work, to protect against a rarer failure -- so it fails OPEN."""
    monkeypatch.setattr(
        wdl, "_try_lock", lambda handle: (_ for _ in ()).throw(OSError("no locks"))
    )

    acquire_working_dir_lock(str(tmp_path))

    assert holds_working_dir_lock(str(tmp_path)) is True
    assert wdl._claims[wdl._lock_path(str(tmp_path))].enforced is False


def test_releasing_without_a_claim_is_a_no_op(tmp_path):
    release_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_two_spellings_of_one_directory_are_one_claim(tmp_path):
    """``_lock_path`` resolves symlinks, and that is load bearing rather than
    tidiness: a relative path and its absolute form, or a symlink and its
    target, must produce ONE key. Two keys mean this tree opens a second
    descriptor on the one file -- and ``flock`` belongs to the open file
    description, so the process refuses ITSELF.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    acquire_working_dir_lock(str(real))
    acquire_working_dir_lock(str(link))  # counts in; does not refuse itself

    assert holds_working_dir_lock(str(link)) is True
    release_working_dir_lock(str(link))
    assert _foreign_attempt(real) == "REFUSED"

    release_working_dir_lock(str(real))
    assert _foreign_attempt(real) == "ADMITTED"
