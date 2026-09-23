"""The anchor lock: shared for starters, exclusive for the offline migration.

Starters that coexist today must still coexist, so they share the lock; the
migration takes it exclusively and is refused while any starter holds it. Where
locking is unavailable, starters warn and proceed and the migration refuses
unless the operator asserts exclusivity.

"Another process" is stood in for by a second open file DESCRIPTION on the
lock file: ``flock`` locks belong to the description, so two descriptions
conflict exactly as two processes do, even inside this one. See
``lightrag/kg/anchor_lock.py``.
"""

from __future__ import annotations

import errno
import os

import pytest

from lightrag.exceptions import ConfigurationAnchorLockError
from lightrag.kg import anchor_lock as al

pytestmark = pytest.mark.offline

fcntl = pytest.importorskip("fcntl")


@pytest.fixture(autouse=True)
def _no_holds_leak():
    yield
    for hold in list(al._shared_holds.values()):
        if hold.handle is not None:
            try:
                hold.handle.close()
            except OSError:
                pass
    al._shared_holds.clear()


def _foreign(tmp_path, mode):
    """A lock taken the way ANOTHER process would hold it."""
    path = al.anchor_lock_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
    return handle


def test_the_lock_file_is_distinct_from_the_config_dir_claim(tmp_path):
    from lightrag.kg.working_dir_lock import LOCK_FILENAME

    path = al.anchor_lock_path(str(tmp_path))
    assert os.path.basename(path) == ".lightrag_anchor.lock"
    assert os.path.basename(path) != LOCK_FILENAME
    assert os.path.dirname(path) == os.path.realpath(tmp_path / "_lightrag_config")


def test_starters_share_it(tmp_path):
    other = _foreign(tmp_path, fcntl.LOCK_SH)
    try:
        al.acquire_anchor_lock_shared(str(tmp_path))
        assert al.holds_anchor_lock(str(tmp_path))
    finally:
        other.close()
    al.release_anchor_lock_shared(str(tmp_path))
    assert not al.holds_anchor_lock(str(tmp_path))


def test_a_starter_is_refused_while_a_migration_holds_it(tmp_path):
    migration = _foreign(tmp_path, fcntl.LOCK_EX)
    try:
        with pytest.raises(ConfigurationAnchorLockError, match="migration"):
            al.acquire_anchor_lock_shared(str(tmp_path))
        assert not al.holds_anchor_lock(str(tmp_path))
    finally:
        migration.close()
    al.acquire_anchor_lock_shared(str(tmp_path))
    al.release_anchor_lock_shared(str(tmp_path))


def test_the_migration_is_refused_while_another_starter_holds_it(tmp_path):
    starter = _foreign(tmp_path, fcntl.LOCK_SH)
    try:
        with pytest.raises(ConfigurationAnchorLockError, match="held by a running"):
            al.acquire_anchor_lock_exclusive(str(tmp_path), assume_exclusive=True)
    finally:
        starter.close()


def test_the_migration_is_refused_while_this_process_is_a_starter(tmp_path):
    al.acquire_anchor_lock_shared(str(tmp_path))
    with pytest.raises(ConfigurationAnchorLockError, match="already holds"):
        al.acquire_anchor_lock_exclusive(str(tmp_path))
    al.release_anchor_lock_shared(str(tmp_path))


def test_the_migration_excludes_every_starter_and_releases(tmp_path):
    lock = al.acquire_anchor_lock_exclusive(str(tmp_path))
    assert lock.enforced is True
    with pytest.raises(BlockingIOError):
        _foreign(tmp_path, fcntl.LOCK_SH)
    lock.release()
    lock.release()  # idempotent
    _foreign(tmp_path, fcntl.LOCK_SH).close()


def test_holds_are_counted_per_process_tree(tmp_path):
    """Two instances (or a master and its worker) count into one hold; the
    lock is given back only at zero."""
    al.acquire_anchor_lock_shared(str(tmp_path))
    al.acquire_anchor_lock_shared(str(tmp_path))
    al.release_anchor_lock_shared(str(tmp_path))
    assert al.holds_anchor_lock(str(tmp_path))
    with pytest.raises(BlockingIOError):
        _foreign(tmp_path, fcntl.LOCK_EX)
    al.release_anchor_lock_shared(str(tmp_path))
    assert not al.holds_anchor_lock(str(tmp_path))
    _foreign(tmp_path, fcntl.LOCK_EX).close()


def test_releasing_without_a_hold_is_a_no_op(tmp_path):
    al.release_anchor_lock_shared(str(tmp_path))


def test_two_spellings_of_one_directory_are_one_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    al.acquire_anchor_lock_shared(str(tmp_path))
    al.acquire_anchor_lock_shared(tmp_path.name)
    assert len(al._shared_holds) == 1
    al.release_anchor_lock_shared(tmp_path.name)
    al.release_anchor_lock_shared(str(tmp_path))
    assert not al._shared_holds


def _cannot_lock(monkeypatch):
    real = fcntl.flock

    def _flock(fd, op):
        if op & fcntl.LOCK_UN:
            return real(fd, op)
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", _flock)


def test_a_filesystem_that_cannot_lock_lets_starters_proceed(tmp_path, monkeypatch):
    _cannot_lock(monkeypatch)
    warnings: list[str] = []
    monkeypatch.setattr(al.logger, "warning", warnings.append)
    al.acquire_anchor_lock_shared(str(tmp_path))
    assert al.holds_anchor_lock(str(tmp_path))
    assert any("proceeding without it" in w for w in warnings)
    al.release_anchor_lock_shared(str(tmp_path))


def test_a_filesystem_that_cannot_lock_refuses_the_migration(tmp_path, monkeypatch):
    _cannot_lock(monkeypatch)
    with pytest.raises(ConfigurationAnchorLockError, match="--assume-exclusive"):
        al.acquire_anchor_lock_exclusive(str(tmp_path))
    lock = al.acquire_anchor_lock_exclusive(str(tmp_path), assume_exclusive=True)
    assert lock.enforced is False
    lock.release()


def test_a_platform_without_shared_locks_fails_open_and_refuses_the_migration(
    tmp_path, monkeypatch
):
    """``msvcrt.locking`` has no shared mode, so on Windows starters cannot
    announce themselves -- and an exclusive lock there would exclude nobody."""
    monkeypatch.setattr(al, "_fcntl", lambda: None)
    al.acquire_anchor_lock_shared(str(tmp_path))
    assert al._shared_holds[al.anchor_lock_path(str(tmp_path))].enforced is False
    al.release_anchor_lock_shared(str(tmp_path))
    with pytest.raises(ConfigurationAnchorLockError):
        al.acquire_anchor_lock_exclusive(str(tmp_path))
    al.acquire_anchor_lock_exclusive(str(tmp_path), assume_exclusive=True).release()


def test_a_read_only_directory_lets_starters_proceed(tmp_path, monkeypatch):
    def _read_only(path):
        raise OSError(errno.EROFS, "Read-only file system", path)

    monkeypatch.setattr(al, "_open_lock_file", _read_only)
    al.acquire_anchor_lock_shared(str(tmp_path))
    assert al.holds_anchor_lock(str(tmp_path))
    al.release_anchor_lock_shared(str(tmp_path))
    with pytest.raises(ConfigurationAnchorLockError):
        al.acquire_anchor_lock_exclusive(str(tmp_path))
