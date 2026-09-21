"""An exclusive claim on a directory, held for the life of a process tree.

**Taken on ``config_dir`` for a file-backed CONFIGURATION storage, and only
for that.** The parameter is still spelled ``working_dir`` throughout because
the claim is on a directory whatever that directory holds; with ``config_dir``
at its default (``<working_dir>/_lightrag_config``) the two name the same
deployment either way.

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

**Accepted residue: business data is not protected.** Two servers sharing a
``working_dir`` whose configuration is on a server backend (or in a different
``config_dir``) but whose ``full_docs`` / ``doc_status`` / graph / vectors are
file-backed still overwrite each other, and lose more than baselines when they
do. That is the
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
from lightrag.namespace import default_config_dir
from lightrag.utils import logger

LOCK_FILENAME = ".lightrag_storage.lock"

# The storages whose data lives on the local filesystem. Only a deployment
# whose CONFIGURATION storage is one of them claims a directory -- two servers
# whose configuration is on a server backend share nothing the claim protects,
# and refusing that would be an invention.
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
    # The slice-1 lock path, held alongside the real one while a deployment
    # running that version can still be on the other side of an upgrade.
    # See ``_acquire_legacy_claim``.
    legacy_handle: Any = None


# Path -> claim. Inherited across ``fork``, which is the point: a worker finds
# its master's claim here and counts itself in rather than opening a second
# descriptor, which WOULD conflict.
_claims: dict[str, _Claim] = {}


def _lock_path(working_dir: str) -> str:
    # ``realpath``, not ``abspath``: two spellings of one directory -- a
    # relative path and its absolute form, a symlink and its target -- must
    # produce ONE key, or this process tree opens a second descriptor on the
    # same file and refuses itself.
    return os.path.join(os.path.realpath(working_dir), LOCK_FILENAME)


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


def _legacy_lock_path(config_dir: str, legacy_working_dir: str | None) -> str | None:
    """The path slice 1 locked for this configuration directory, or ``None``.

    Slice 1 kept the configuration file at ``<working_dir>/_lightrag_config/``
    but locked ``<working_dir>``. This slice locks the directory the file is
    actually in, so the two versions would lock unrelated files: a server from
    each could start on one deployment, and both rewrite that one file from a
    private in-memory copy -- exactly what the claim exists to refuse. Only
    the DEFAULT directory has a slice-1 spelling; one named by
    ``LIGHTRAG_CONFIG_DIR`` is new here and has no older holder.

    ``legacy_working_dir`` is the directory this configuration directory was
    DERIVED from, or ``None`` when the caller was handed one outright. Only
    the caller knows which: the basename cannot tell a default apart from a
    ``LIGHTRAG_CONFIG_DIR`` that merely ENDS in the same component, and
    guessing wrong claims a lock belonging to a different deployment --
    refusing a server that has every right to start.

    Transitional. Remove it once no deployment can still be running a build
    that predates the move, and nothing but this function knows the old path.
    """
    if not legacy_working_dir:
        return None
    # Compared on the spelling, BEFORE any symlink is followed. This module
    # deliberately resolves symlinks everywhere else -- one directory must
    # produce one key however it is spelled -- but resolving first would
    # answer a different question: a default ``<working_dir>/_lightrag_config``
    # that is a symlink to, say, ``/mnt/config`` stops looking derived from
    # its working directory at all, and the deployment most in need of the
    # transitional claim would silently not get one.
    spelled = os.path.normpath(os.path.abspath(config_dir))
    default = os.path.normpath(os.path.abspath(default_config_dir(legacy_working_dir)))
    if spelled != default:
        return None
    # The PATH, though, is resolved: slice 1 locked
    # ``realpath(working_dir)/LOCK_FILENAME``, and this has to name that same
    # file or it claims something the old process never held.
    return os.path.join(os.path.realpath(os.path.dirname(spelled)), LOCK_FILENAME)


def _acquire_legacy_claim(config_dir: str, legacy_working_dir: str | None) -> Any:
    """Take the slice-1 lock too, or return ``None`` if there is none to take.

    Raises ``WorkingDirectoryInUseError`` when a process holding the old path
    is still running -- the refusal this whole module exists to produce, from
    the one direction the new path cannot see. A filesystem that cannot lock
    fails open here for the same reason it does for the real claim; the caller
    has already warned about it.
    """
    path = _legacy_lock_path(config_dir, legacy_working_dir)
    if path is None:
        return None

    try:
        handle = open(path, "a+")
    except OSError:
        # The parent directory may not exist or may not be writable. The real
        # claim is what protects this deployment; this one only reaches back.
        return None

    try:
        locked = _try_lock(handle)
    except OSError:
        return handle

    if not locked:
        handle.close()
        raise WorkingDirectoryInUseError(
            f"working directory '{os.path.dirname(os.path.realpath(config_dir))}' "
            f"is already in use by a LightRAG process from before the "
            f"configuration directory got its own lock. Both would rewrite "
            f"the same configuration file from a per-process copy, losing "
            f"each other's embedding baselines. Stop it before starting this "
            f"one."
        )
    return handle


def acquire_working_dir_lock(
    working_dir: str, *, legacy_working_dir: str | None = None
) -> None:
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
        # The primary path cannot be enforced -- but the slice-1 path may sit
        # on a filesystem that CAN enforce one: a configuration directory
        # symlinked onto a network volume with a local parent is the ordinary
        # shape of that. Failing open here AND skipping the transitional claim
        # would start this server beside an older one holding the only lock
        # either of them is able to take.
        try:
            legacy_handle = _acquire_legacy_claim(working_dir, legacy_working_dir)
        except BaseException:
            # Nothing was locked on the primary path, so there is nothing to
            # unlock -- only the descriptor to give back.
            handle.close()
            raise
        _claims[path] = _Claim(
            handle=handle, enforced=False, legacy_handle=legacy_handle
        )
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
        legacy_handle = _acquire_legacy_claim(working_dir, legacy_working_dir)
    except BaseException:
        # This directory is free but its slice-1 spelling is not. Give back
        # what was just taken, so a refusal leaves nothing held.
        _unlock(handle)
        handle.close()
        raise

    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        # Cosmetic: the PID is for the operator reading the file, never for
        # deciding whether the lock is held.
        pass

    _claims[path] = _Claim(handle=handle, legacy_handle=legacy_handle)


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

    # The transitional claim goes back with the real one, never on its own.
    if claim.legacy_handle is not None:
        try:
            _unlock(claim.legacy_handle)
        except OSError:
            pass
        try:
            claim.legacy_handle.close()
        except OSError:
            pass


def holds_working_dir_lock(working_dir: str) -> bool:
    """Whether THIS process tree holds ``working_dir`` (for tests and probes)."""
    return _lock_path(working_dir) in _claims
