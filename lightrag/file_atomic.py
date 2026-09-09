"""Shared atomic file-write helpers.

Why this lives at the package root rather than under ``lightrag/kg/``:
``lightrag.utils.write_json`` needs ``atomic_write`` to gain crash safety,
and several ``lightrag/kg/*`` modules need both. Hosting the helpers under
``lightrag/kg/`` would create a ``utils -> kg -> utils`` import cycle.
Keeping this module dependency-free (stdlib only) avoids that.

Semantics
---------
``atomic_write`` writes through a per-writer ``.tmp.<pid>.<tid>.<ns>`` sibling
and renames into place with ``os.replace`` — atomic on the same filesystem on
both POSIX (``rename(2)``) and Windows (``MoveFileEx`` with
``MOVEFILE_REPLACE_EXISTING``). Two failure modes are handled differently:

- A Python exception (``write_fn`` raised, ``os.replace`` failed, etc.):
  ``finally`` runs, the in-flight tmp is removed best-effort, and the
  exception propagates. The on-disk destination is the prior snapshot.
- A process-level kill (SIGKILL, OOM, hard reboot) between writing the tmp
  and the rename: ``finally`` does not run, the tmp survives as an orphan,
  and ``reap_orphan_tmp_files`` cleans it on the next startup once it ages
  past the threshold.

What is *not* preserved across the inode swap done by ``os.replace``: owner,
group, ACLs, xattrs, hard-link relationships, and any symlink-target identity.
The mode bits (rwx) are preserved explicitly — see ``_preserve_mode``.
"""

from __future__ import annotations

import glob
import logging
import os
import stat
import sys
import threading
import time
from typing import Callable

logger = logging.getLogger("lightrag")

# Orphan .tmp files older than this are reaped on startup. Large enough that
# an in-flight write from another live process cannot plausibly still be
# running (multi-million-node graphml writes finish in minutes, not hours).
TMP_REAP_AGE_SECONDS = 3600

# Windows-only retry budget for ``os.replace`` (see ``_replace_file``). The
# backoff sleeps run on the process-wide SINGLE-worker storage-io executor
# (``lightrag.utils.get_storage_io_executor``), so the total is the worst-case
# stall imposed on *every* file backend's flush, not just the one retrying.
# Keep it around a second: a handle held by an indexer or antivirus clears in
# tens to a few hundred milliseconds, and a longer hold is better surfaced as a
# failure than paid for by every other namespace queued behind that one worker.
# 7 sleeps of 0.05, 0.075, 0.1125, 0.169, 0.2, 0.2, 0.2 sum to ~1.0s.
WINDOWS_REPLACE_MAX_ATTEMPTS = 8
WINDOWS_REPLACE_INITIAL_DELAY = 0.05
WINDOWS_REPLACE_MAX_DELAY = 0.2


def tmp_path_for(file_name: str) -> str:
    """Return a per-writer tmp sibling for ``file_name``.

    The suffix embeds PID, thread id, and a nanosecond timestamp so that
    multiple concurrent writers — separate processes sharing the same working
    directory, or multiple threads inside one process — cannot trample each
    other's in-flight tmp and leave a "no such file" rename error behind.
    """
    return f"{file_name}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"


def _preserve_mode(tmp: str, dst: str, workspace: str) -> None:
    """Carry ``dst``'s existing mode bits onto ``tmp`` before the rename.

    Without this, ``os.replace`` swaps the inode and the new file inherits
    umask defaults — any intentional restriction (e.g. chmod 0600) on the
    prior snapshot would be silently widened.
    """
    if not os.path.exists(dst):
        return
    try:
        os.chmod(tmp, stat.S_IMODE(os.stat(dst).st_mode))
    except OSError as exc:
        logger.warning(f"[{workspace}] Could not preserve mode of {dst}: {exc}")


def reap_orphan_tmp_files(
    file_name: str,
    workspace: str = "_",
    age_seconds: int = TMP_REAP_AGE_SECONDS,
    extra_patterns: tuple[str, ...] = (),
) -> None:
    """Delete stale tmp siblings of ``file_name`` left behind by hard kills.

    Default pattern matches ``glob.escape(file_name) + ".tmp.*"`` — the suffix
    shape produced by ``tmp_path_for``. ``extra_patterns`` accepts already-built
    glob patterns and is intended for migrating away from legacy naming
    schemes (e.g. Faiss's previous fixed ``<meta>.tmp`` suffix, which the
    default pattern's trailing ``.*`` will not match).

    ``glob.escape`` is required because ``file_name`` is composed from
    ``working_dir + namespace`` and can legitimately contain glob
    metacharacters (workspace ``[v2]``, ``*``, ``?``). Concatenating naively
    would silently miss the real orphan or widen the match to tmp files of
    unrelated storage types.
    """
    patterns = [glob.escape(file_name) + ".tmp.*", *extra_patterns]
    now = time.time()
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                age = now - os.path.getmtime(path)
            except OSError:
                continue
            if age < age_seconds:
                continue
            try:
                os.remove(path)
                logger.info(
                    f"[{workspace}] Reaped orphan tmp file: {path} (age {age:.0f}s)"
                )
            except OSError as exc:
                logger.warning(
                    f"[{workspace}] Failed to reap orphan tmp file {path}: {exc}"
                )


def _replace_file(src: str, dst: str, workspace: str = "_") -> None:
    """Replace src into dst, with exponential backoff retry on Windows NTFS.

    On Windows, os.replace (MoveFileEx) raises PermissionError ([WinError 5]
    ACCESS_DENIED or [WinError 32] SHARING_VIOLATION) if another thread,
    coroutine, or system service (e.g. search indexer, antivirus) momentarily
    holds an open handle to dst. We retry with gentle backoff. ``src`` stays in
    place when the rename fails, so every attempt is idempotent.

    Other platforms take a straight ``os.replace``: the retry loop would add a
    branch and a misleading "after 1 attempts" log line to a path that never
    retries.
    """
    if sys.platform != "win32":
        os.replace(src, dst)
        return

    delay = WINDOWS_REPLACE_INITIAL_DELAY
    for attempt in range(WINDOWS_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if attempt == WINDOWS_REPLACE_MAX_ATTEMPTS - 1:
                logger.warning(
                    f"[{workspace}] Failed to atomically replace {dst} after "
                    f"{WINDOWS_REPLACE_MAX_ATTEMPTS} attempts: {exc}"
                )
                raise
            logger.debug(
                f"[{workspace}] Retrying atomic replace of {dst} in {delay:.2f}s "
                f"(attempt {attempt + 1}/{WINDOWS_REPLACE_MAX_ATTEMPTS}): {exc}"
            )
            time.sleep(delay)
            delay = min(delay * 1.5, WINDOWS_REPLACE_MAX_DELAY)


def atomic_write(
    file_name: str,
    write_fn: Callable[[str], None],
    workspace: str = "_",
) -> None:
    """Run ``write_fn(tmp_path)`` then atomically replace ``file_name`` with it.

    ``write_fn`` is responsible for actually producing the file contents at
    the path it receives. It must not assume the tmp path equals ``file_name``
    — Faiss/Nano callers rely on the tmp path being a real sibling.
    On any exception from ``write_fn`` or from the rename, the tmp is removed
    best-effort and the exception propagates. The destination file is not
    touched in that case.
    """
    tmp = tmp_path_for(file_name)
    try:
        write_fn(tmp)
        _preserve_mode(tmp, file_name, workspace)
        _replace_file(tmp, file_name, workspace)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError as exc:
            logger.warning(
                f"[{workspace}] Failed to remove tmp after failed atomic write: {exc}"
            )
        raise
