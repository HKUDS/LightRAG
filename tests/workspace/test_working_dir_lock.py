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
"""

from __future__ import annotations

import os

import pytest

from lightrag.exceptions import WorkingDirectoryInUseError
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


def _foreign_attempt(path) -> str:
    """Acquire from a process that did NOT inherit this tree's bookkeeping."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        os.close(read_fd)
        wdl._claims.clear()
        try:
            acquire_working_dir_lock(str(path))
            os.write(write_fd, b"ADMITTED")
        except WorkingDirectoryInUseError:
            os.write(write_fd, b"REFUSED")
        finally:
            os._exit(0)
    os.close(write_fd)
    answer = os.read(read_fd, 16).decode()
    os.close(read_fd)
    os.waitpid(pid, 0)
    return answer


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


def test_forked_workers_inherit_the_masters_claim(tmp_path):
    """The Gunicorn shape: the master claims BEFORE forking, so the workers
    find the claim in their own tree and count themselves in.

    Taken after the fork instead, each worker would open its own descriptor
    and all but one would be refused -- which is why ``on_starting`` is the
    hook that takes it.
    """
    acquire_working_dir_lock(str(tmp_path))

    admitted = []
    for _ in range(4):
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the child
            os.close(read_fd)
            try:
                acquire_working_dir_lock(str(tmp_path))
                os.write(write_fd, b"OK")
            except WorkingDirectoryInUseError:
                os.write(write_fd, b"REFUSED")
            finally:
                os._exit(0)
        os.close(write_fd)
        admitted.append(os.read(read_fd, 16).decode())
        os.close(read_fd)
        os.waitpid(pid, 0)

    assert admitted == ["OK"] * 4


def test_a_holder_that_dies_leaves_nothing_to_reap(tmp_path):
    """Why an OS lock and not a PID file: no stale-file cleanup, no
    read-the-PID-then-probe-liveness race. The kernel releases it."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        wdl._claims.clear()
        acquire_working_dir_lock(str(tmp_path))
        os._exit(0)  # dies holding it, no release, no cleanup
    os.waitpid(pid, 0)

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
