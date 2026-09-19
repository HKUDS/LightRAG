"""An exclusive claim on a ``working_dir``, held for the life of a process tree.

**Taken for a file-backed CONFIGURATION storage, and only for that.**

A file-backed storage publishes a namespace by rewriting a whole file from an
in-memory copy. That copy is shared inside ONE process tree, which is what
makes a Gunicorn master and its workers safe. It is not shared between process
trees, so two servers started on one ``working_dir`` each load the file, each
accumulate their own view, and each rewrite the whole thing -- the later flush
dropping whatever the other recorded since.

For the configuration namespace that is fatal in a way the others are not: an
overwritten baseline reads back as ABSENT, and absent is the one answer that
lets a start bootstrap. So the next start does not refuse the model change the
baseline existed to refuse -- it records the configured model over vectors
nobody probed, and the protection is gone with nothing in any log.

Nothing inside a process tree can see that, so the claim has to live where both
servers can: on the directory itself.

**Accepted residue: business data in the same directory is not protected.** Two
servers sharing a ``working_dir`` whose configuration is on a server backend but
whose ``full_docs`` / ``doc_status`` / graph / vectors are file-backed still
overwrite each other, and lose more than baselines when they do. That is the
long-standing "separate process trees are unsupported" position, unchanged
here; this claim narrows the blast radius rather than closing it, because the
baseline is the case whose failure is SILENT. Recovery for the rest is
unchanged: run one server per directory, or use server backends. Widening the
claim to any file-backed storage is a deliberate follow-up, not an oversight --
it would refuse deployments that work today.

**An OS lock, not a PID file.** The kernel releases it when the holder dies, so
a `SIGKILL`, an OOM kill or a power cut leaves nothing stale to reap, and there
is no read-PID-then-probe-liveness race. It is also held by the open file
DESCRIPTION, which `fork` shares -- so a Gunicorn master takes it once and its
workers inherit it rather than fighting it.

**It fails OPEN.** Locking is unreliable on NFSv3 without lockd and on SMB/CIFS,
and a `working_dir` on a network volume is ordinary in container deployments.
Refusing to start there would break working deployments to protect against a
rarer one, so a backend that cannot lock gets a warning and proceeds.

See *One file per namespace per process tree* in
``docs/design/FileBackedSnapshotContract.md`` for the in-process guard this sits
beside; the two catch different failures and neither subsumes the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from lightrag.exceptions import WorkingDirectoryInUseError
from lightrag.utils import logger

LOCK_FILENAME = ".lightrag_storage.lock"

# The storages whose data lives under ``working_dir``. Only a deployment using
# at least one of them is claiming the directory -- two servers on one
# ``working_dir`` with server backends share nothing but logs and inputs, and
# refusing that would be an invention.
FILE_BACKED_STORAGES = frozenset(
    {
        "JsonKVStorage",
        "JsonDocStatusStorage",
        "NetworkXStorage",
        "NanoVectorDBStorage",
        "FaissVectorDBStorage",
    }
)


def uses_working_dir(*storage_names: str) -> bool:
    """Whether any of these configured storage classes keeps data on disk."""
    return any(name in FILE_BACKED_STORAGES for name in storage_names if name)


@dataclass
class _Claim:
    handle: Any
    holders: int = 1
    enforced: bool = True
    inherited_pid: int = field(default_factory=os.getpid)


# Path -> claim. Inherited across ``fork``, which is the point: a worker finds
# its master's claim here and counts itself in rather than opening a second
# descriptor, which WOULD conflict.
_claims: dict[str, _Claim] = {}


def _lock_path(working_dir: str) -> str:
    return os.path.join(os.path.abspath(working_dir), LOCK_FILENAME)


def _try_lock(handle) -> bool:
    """True if locked, False if another process holds it.

    Raises ``OSError`` when the platform cannot lock at all, which the caller
    turns into a warning rather than a refusal.
    """
    try:
        import fcntl
    except ImportError:  # Windows
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            # msvcrt does not distinguish "held by another process" from
            # "cannot lock here"; a held lock is the overwhelmingly likely
            # cause, and treating it as such is the safe reading.
            return False

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(handle) -> None:
    try:
        import fcntl
    except ImportError:  # Windows
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def acquire_working_dir_lock(working_dir: str) -> None:
    """Claim ``working_dir`` for this process tree, or refuse.

    Idempotent per tree: a second caller here (another ``LightRAG`` instance, a
    forked worker) counts itself into the existing claim. Every caller owes a
    matching ``release_working_dir_lock``.

    Raises ``WorkingDirectoryInUseError`` when another process tree holds it.
    """
    path = _lock_path(working_dir)

    claim = _claims.get(path)
    if claim is not None:
        claim.holders += 1
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "a+")

    try:
        locked = _try_lock(handle)
    except OSError as e:
        # The filesystem cannot lock. Say so once and carry on unprotected --
        # see the fails-open note in the module docstring.
        logger.warning(
            f"Could not lock the working directory '{working_dir}' ({e}). "
            f"Two servers started on it would overwrite each other's "
            f"file-backed storage without any warning; run one at a time, or "
            f"use a server storage backend."
        )
        _claims[path] = _Claim(handle=handle, enforced=False)
        return

    if not locked:
        handle.close()
        raise WorkingDirectoryInUseError(
            f"working directory '{working_dir}' is already in use by another "
            f"LightRAG process. The file-backed storages rewrite whole files "
            f"from a per-process copy, so both would overwrite each other's "
            f"rows -- the configuration baselines included. Start one at a "
            f"time, give this one its own working directory, or use a server "
            f"storage backend."
        )

    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        # Cosmetic: the PID is for the operator reading the file, never for
        # deciding whether the lock is held.
        pass

    _claims[path] = _Claim(handle=handle)


def release_working_dir_lock(working_dir: str) -> None:
    """Give up one hold; release the directory at zero.

    Safe to call without a hold, so a teardown path may call it
    unconditionally.
    """
    path = _lock_path(working_dir)
    claim = _claims.get(path)
    if claim is None:
        return

    claim.holders -= 1
    if claim.holders > 0:
        return

    _claims.pop(path, None)
    if claim.enforced:
        _unlock(claim.handle)
    try:
        claim.handle.close()
    except OSError:
        pass


def holds_working_dir_lock(working_dir: str) -> bool:
    """Whether THIS process tree holds ``working_dir`` (for tests and probes)."""
    return _lock_path(working_dir) in _claims
