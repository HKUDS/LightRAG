"""The anchor lock: SHARED for every starter, EXCLUSIVE for the offline
configuration migration.

``<working_dir>/.lightrag_anchor.lock``, beside the anchor file it protects
(``lightrag/config_anchor.py``). It is a DIFFERENT file from the exclusive
``.lightrag_storage.lock`` a JSON configuration storage takes beside its
workspace snapshot (``lightrag/kg/working_dir_lock.py``) -- for the empty
workspace both sit directly in ``working_dir`` -- and the two answer
different questions:

* the snapshot claim keeps a second SERVER off one JSON configuration
  file -- exclusive between starters;
* this lock keeps a MIGRATION off a deployment that is running -- starters
  share it with each other and are only excluded against the migration.

So starters that coexist today (a server and ``lightrag-rebuild-vdb`` on a
server-backed configuration, two SDK processes) are not newly refused.

Rules:

* **Starters take it shared** -- the server, the Gunicorn master in
  ``on_starting`` before forking (workers inherit its hold and count
  themselves in, exactly as with the snapshot claim), the SDK,
  ``lightrag-rebuild-vdb``, ``lightrag-clear-storage``. A starter is refused
  only while a migration holds the lock.
* **Starters fail open** where the filesystem or platform cannot lock
  (NFSv3 without lockd, SMB/CIFS, a read-only directory, or Windows, whose
  ``msvcrt.locking`` has no shared mode): a warning, then proceed -- the
  posture the snapshot claim already takes.
* **The migration refuses** in exactly those places unless the operator
  passes ``assume_exclusive`` (``--assume-exclusive``), which records that
  every reader and writer has been stopped by hand. A lock that works and
  is held is refused regardless.
* **Order**: this lock first, then the snapshot claim, then (only when
  needed) the bind lock; released in reverse, after the storage teardown.

A third file, ``.lightrag_anchor_bind.lock``, is taken exclusively by a start
that finds no anchor or, on JSON, a workspace not yet listed in the anchor's
members (``anchor_bind_lock``): servers sharing one ``working_dir`` with
different workspaces bind the same identity and append to the same member
list, and the in-tree keyed lock cannot see each other. A held bind lock is
waited for, never bypassed. Where POSIX reports that locking is unsupported
it fails open with a warning (a lost member append is then detected by the
migration's cross-check, see the contract); Windows' ``msvcrt`` cannot tell
"unsupported" from "held", so every failure there is treated as contention.

A local directory lock does not cover a deployment using another
``working_dir`` against the same remote configuration container; those must
be stopped and coordinated by the operator. No distributed lock is offered.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from lightrag.config_anchor import anchor_dir
from lightrag.exceptions import ConfigurationAnchorLockError
from lightrag.namespace import ANCHOR_BIND_LOCK_FILE_NAME, ANCHOR_LOCK_FILE_NAME
from lightrag.utils import logger

LOCK_FILENAME = ANCHOR_LOCK_FILE_NAME
BIND_LOCK_FILENAME = ANCHOR_BIND_LOCK_FILE_NAME

# How long a start waits for another process tree's bind to finish. A bind is
# one identity write, flush and read-back plus a file publish; this is ample.
DEFAULT_BIND_LOCK_TIMEOUT_SECONDS = 120.0


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


def anchor_bind_lock_path(working_dir: str) -> str:
    return os.path.join(os.path.realpath(anchor_dir(working_dir)), BIND_LOCK_FILENAME)


def _msvcrt():
    try:
        import msvcrt
    except ImportError:
        return None
    return msvcrt


@asynccontextmanager
async def anchor_bind_lock(
    working_dir: str, *, timeout: float = DEFAULT_BIND_LOCK_TIMEOUT_SECONDS
) -> AsyncIterator[None]:
    """Serialize a bind or a JSON member append across process trees on one
    host.

    Several servers may share one ``working_dir`` with different workspaces:
    every one of them binds the same container-wide identity and, on JSON,
    appends itself to the same member list. The in-tree keyed lock cannot
    see a second server, so without this two first starts could each create
    a UUID, or two appends could each rewrite the list without the other's
    entry. Taken when there is no anchor or the workspace is not a member --
    a registered start never waits here -- and the caller re-reads the
    anchor inside it. Waits (polling, never blocking the event loop) up to
    ``timeout`` and then raises ``ConfigurationAnchorLockError``. Fails OPEN
    with a warning only where POSIX says locking is unsupported; on Windows
    every ``msvcrt`` failure reads as contention (see the module rules).
    """
    path = anchor_bind_lock_path(working_dir)
    fcntl = _fcntl()
    msvcrt = None if fcntl is not None else _msvcrt()
    try:
        handle = _open_lock_file(path)
    except OSError as e:
        _warn_bind_unlocked(path, f"{type(e).__name__}: {e}")
        yield
        return
    locked = False
    try:
        if fcntl is None and msvcrt is None:
            _warn_bind_unlocked(path, "this platform has no file lock")
        else:
            deadline = time.monotonic() + max(0.0, timeout)
            while not locked:
                try:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except BlockingIOError:
                    pass
                except OSError as e:
                    if fcntl is not None:
                        _warn_bind_unlocked(path, f"{type(e).__name__}: {e}")
                        break
                    # msvcrt: "held by another process" and "cannot lock
                    # here" are the same error; the safe reading is "held".
                if locked:
                    break
                if time.monotonic() >= deadline:
                    raise ConfigurationAnchorLockError(
                        f"Another process has held the configuration anchor "
                        f"bind lock {path} for more than {timeout:.0f}s while "
                        f"binding this working directory or registering a "
                        f"workspace; nothing was written. Retry once it has "
                        f"finished starting."
                    ) from None
                await asyncio.sleep(0.05)
        yield
    finally:
        if locked:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                else:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        _close(handle)


def _warn_bind_unlocked(path: str, reason: str) -> None:
    logger.warning(
        f"Could not take the configuration anchor bind lock {path} ({reason}); "
        f"binding without it. Start servers sharing this working directory "
        f"one at a time until the anchor exists and, with JSON configuration, "
        f"until every workspace has been registered once."
    )
