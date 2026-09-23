"""The anchor lock: SHARED for every starter, EXCLUSIVE for the offline
configuration migration.

``<working_dir>/_lightrag_config/.lightrag_anchor.lock``, beside the anchor
file it protects (``lightrag/config_anchor.py``). It is a DIFFERENT file from
the exclusive ``.lightrag_storage.lock`` a file-backed configuration storage
takes on ``config_dir`` (``lightrag/kg/working_dir_lock.py``), even when both
sit in the same directory, and the two answer different questions:

* the ``config_dir`` claim keeps a second SERVER off one JSON configuration
  file -- exclusive between starters;
* this lock keeps a MIGRATION off a deployment that is running -- starters
  share it with each other and are only excluded against the migration.

So starters that coexist today (a server and ``lightrag-rebuild-vdb`` on a
server-backed configuration, two SDK processes) are not newly refused.

Rules:

* **Starters take it shared** -- the server, the Gunicorn master in
  ``on_starting`` before forking (workers inherit its hold and count
  themselves in, exactly as with the ``config_dir`` claim), the SDK,
  ``lightrag-rebuild-vdb``, ``lightrag-clear-storage``. A starter is refused
  only while a migration holds the lock.
* **Starters fail open** where the filesystem or platform cannot lock
  (NFSv3 without lockd, SMB/CIFS, a read-only directory, or Windows, whose
  ``msvcrt.locking`` has no shared mode): a warning, then proceed -- the
  posture the ``config_dir`` claim already takes.
* **The migration refuses** in exactly those places unless the operator
  passes ``assume_exclusive`` (``--assume-exclusive``), which records that
  every reader and writer has been stopped by hand. A lock that works and
  is held is refused regardless.
* **Order**: this lock first, then the ``config_dir`` claim; released in
  reverse, after the storage teardown.

A local directory lock does not cover a deployment using another
``working_dir`` against the same remote configuration container; those must
be stopped and coordinated by the operator. No distributed lock is offered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from lightrag.config_anchor import anchor_dir
from lightrag.exceptions import ConfigurationAnchorLockError
from lightrag.utils import logger

LOCK_FILENAME = ".lightrag_anchor.lock"


@dataclass
class _SharedHold:
    handle: Any
    holders: int = 1
    enforced: bool = True


# Lock path -> this process tree's shared hold. Inherited across ``fork``,
# which is the point: a Gunicorn worker finds its master's hold here and
# counts itself in instead of opening a descriptor of its own. The master's
# count is inherited too, so a worker's release never reaches zero and never
# unlocks the description it shares with its master.
_shared_holds: dict[str, _SharedHold] = {}


def anchor_lock_path(working_dir: str) -> str:
    # ``realpath``: two spellings of one directory must be one key.
    return os.path.join(os.path.realpath(anchor_dir(working_dir)), LOCK_FILENAME)


def _open_lock_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a+")


def _fcntl():
    try:
        import fcntl
    except ImportError:  # Windows: no shared mode in msvcrt
        return None
    return fcntl


def _close(handle) -> None:
    try:
        handle.close()
    except OSError:
        pass


def acquire_anchor_lock_shared(working_dir: str) -> None:
    """Take (or count into) this process tree's shared hold.

    Every caller owes a matching ``release_anchor_lock_shared``. Raises
    ``ConfigurationAnchorLockError`` only when a migration holds the lock
    exclusively; every inability to lock is a warning and a proceed.
    """
    path = anchor_lock_path(working_dir)
    hold = _shared_holds.get(path)
    if hold is not None:
        hold.holders += 1
        return

    fcntl = _fcntl()
    try:
        handle = _open_lock_file(path)
    except OSError as e:
        _warn_unlocked(path, f"{type(e).__name__}: {e}")
        _shared_holds[path] = _SharedHold(handle=None, enforced=False)
        return
    if fcntl is None:
        _warn_unlocked(path, "this platform has no shared file lock")
        _shared_holds[path] = _SharedHold(handle=handle, enforced=False)
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        _close(handle)
        raise ConfigurationAnchorLockError(
            f"The configuration anchor lock {path} is held exclusively: an "
            f"offline configuration migration is running on this working "
            f"directory. Wait for it to finish, then start again."
        ) from None
    except OSError as e:
        _warn_unlocked(path, f"{type(e).__name__}: {e}")
        _shared_holds[path] = _SharedHold(handle=handle, enforced=False)
        return
    _shared_holds[path] = _SharedHold(handle=handle)


def _warn_unlocked(path: str, reason: str) -> None:
    logger.warning(
        f"Could not take the configuration anchor lock {path} ({reason}); "
        f"proceeding without it. An offline configuration migration started "
        f"now could not see this process, so do not run one while it is up."
    )


def release_anchor_lock_shared(working_dir: str) -> None:
    """Give up one shared hold; unlock at zero. Safe to call without a hold."""
    path = anchor_lock_path(working_dir)
    hold = _shared_holds.get(path)
    if hold is None:
        return
    hold.holders -= 1
    if hold.holders > 0:
        return
    _shared_holds.pop(path, None)
    if hold.handle is None:
        return
    fcntl = _fcntl()
    if hold.enforced and fcntl is not None:
        try:
            fcntl.flock(hold.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
    _close(hold.handle)


def holds_anchor_lock(working_dir: str) -> bool:
    """Whether THIS process tree holds a shared hold (tests and probes)."""
    return anchor_lock_path(working_dir) in _shared_holds


class ExclusiveAnchorLock:
    """The migration's hold. ``release()`` is idempotent."""

    def __init__(self, path: str, handle: Any, enforced: bool) -> None:
        self.path = path
        self.enforced = enforced
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        fcntl = _fcntl()
        if self.enforced and fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        _close(handle)


def acquire_anchor_lock_exclusive(
    working_dir: str, *, assume_exclusive: bool = False
) -> ExclusiveAnchorLock:
    """Take the lock exclusively, or raise ``ConfigurationAnchorLockError``.

    Refused while any starter holds it -- in another process, or in this one.
    Where locking is unavailable the lock cannot exclude anything, so it is
    refused unless ``assume_exclusive`` records that the operator has stopped
    every reader and writer; then the result is ``enforced=False``.
    """
    path = anchor_lock_path(working_dir)
    if path in _shared_holds:
        raise ConfigurationAnchorLockError(
            f"This process already holds the configuration anchor lock {path} "
            f"as a starter; a migration cannot run inside a running instance."
        )

    def _unavailable(reason: str, handle: Any = None) -> ExclusiveAnchorLock:
        if not assume_exclusive:
            if handle is not None:
                _close(handle)
            raise ConfigurationAnchorLockError(
                f"Cannot take the configuration anchor lock {path} "
                f"exclusively ({reason}), so this tool cannot prove that no "
                f"server, SDK process or maintenance tool is using the "
                f"deployment. Stop every one of them and re-run with "
                f"--assume-exclusive."
            )
        logger.warning(
            f"Configuration anchor lock {path} unavailable ({reason}); "
            f"proceeding because exclusivity was asserted by the operator."
        )
        return ExclusiveAnchorLock(path, handle, enforced=False)

    fcntl = _fcntl()
    try:
        handle = _open_lock_file(path)
    except OSError as e:
        return _unavailable(f"{type(e).__name__}: {e}")
    if fcntl is None:
        return _unavailable(
            "this platform has no shared file lock, so starters cannot "
            "announce themselves",
            handle,
        )
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _close(handle)
        raise ConfigurationAnchorLockError(
            f"The configuration anchor lock {path} is held by a running "
            f"server, SDK process or maintenance tool. Stop every process "
            f"using this working directory before migrating."
        ) from None
    except OSError as e:
        return _unavailable(f"{type(e).__name__}: {e}", handle)
    return ExclusiveAnchorLock(path, handle, enforced=True)
