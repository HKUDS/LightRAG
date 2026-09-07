"""The file half of the file-backed storages' cross-process fence (#3854).

``NetworkXStorage``, ``NanoVectorDBStorage`` and ``FaissVectorDBStorage`` all
coordinate through a file plus a ``storage_updated`` flag distributed by
``lightrag.kg.shared_storage``. The flag is published with one Manager RPC per
process, so it can be lost -- for one process, for several, or for all -- and a
process that misses it keeps a stale snapshot and, on its write path, saves
that snapshot over the peer's durable rows.

This module is the second channel: the file's own ``(st_mtime_ns, st_size)``,
compared against what the process recorded when it last loaded or wrote it.
Unlike the flag it is *state* rather than a consumable event, so a failed
notification cannot disable it.

The two channels are OR-ed and both are kept, permanently: their blind spots do
not overlap. The flag fails on a transient manager outage, where the commits
are far apart in time and the fingerprint sees them; the fingerprint fails when
two commits land inside one filesystem timestamp tick with an identical size,
which needs a healthy, fast-committing system -- exactly when the flag works.

The mechanism lives here, once, because its hazards are in the details rather
than the shape, and three copies of them is how it rots:

* **Sample BEFORE reading the file, never after.** A fingerprint taken after
  the parse can belong to a newer file than the one now in memory, and
  recording it would suppress the reload that newer file needs -- the one way
  this fence could *introduce* a lost write. Sampling early can only cost a
  redundant reload. Callers own this ordering; the functions cannot enforce it.
* **An unreadable ``stat`` is not an absent file.** ``UNREADABLE`` reports "no
  change" so the fence degrades to the flag alone, i.e. to the behaviour that
  predates it. Failing towards a reload instead would install an empty
  snapshot from a file it cannot read, and the next commit would serialize that
  over the real one.
* **``st_ino`` is deliberately excluded.** ``atomic_write`` renames a tmp file
  over the target, freeing the previous inode, and the allocator hands that
  same inode straight back to the next tmp file in the directory -- measured at
  28 reuses across 30 consecutive commits. It is near-constant across commits
  and discriminates nothing. ``st_size`` is a helper that catches nothing on
  its own (an equal-length attribute rewrite keeps it); ``mtime`` carries the
  signal.
* **Single-process mode has no peer that could have committed**, so the test is
  skipped there: a divergent file means an external edit, and reloading for it
  would discard the process's own uncommitted mutations.

Each storage keeps its own recorded value and wires these into its reload,
commit and drop paths; see ``NetworkXStorage``'s *Cross-process sync protocol*
for the full contract, including the accepted residues.
"""

from __future__ import annotations

import os
from typing import Sequence

from lightrag.utils import log_without_raising, logger

from .shared_storage import is_multiprocess_mode

# Returned by :func:`sample` when a path's metadata could not be read at all.
# Distinct from ``None`` ("the file does not exist", a real state to converge
# on) because the two must not lead to the same decision -- see the module
# docstring.
UNREADABLE = object()

# What a readable sample yields: one ``(st_mtime_ns, st_size)`` per path, or
# ``None`` for a path that does not exist.
Fingerprint = tuple[tuple[int, int] | None, ...]


def fence_enabled() -> bool:
    """Whether the file channel is consulted at all.

    Single-process mode has no peer that could have committed, so the file
    test is skipped there -- see the module docstring. Exposed so the storages
    (and their tests) consult the gate in exactly one place.
    """
    return is_multiprocess_mode()


def sample(paths: Sequence[str], *, workspace: str) -> Fingerprint | object:
    """Sample the identity of every path, or ``UNREADABLE`` if any stat fails.

    All-or-nothing on purpose: a storage whose state spans two files (FAISS's
    index + metadata) has to treat a partially readable pair as "cannot tell",
    not as a change.

    Routed through ``log_without_raising`` because a ``drop`` commit hook
    reaches here after the files are already gone, and every step of such a
    hook must be unable to report a completed destruction as an error -- a
    broken log sink included.
    """
    fingerprints: list[tuple[int, int] | None] = []
    for path in paths:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            fingerprints.append(None)
        except OSError as exc:
            log_without_raising(
                logger.warning,
                f"[{workspace}] Could not stat {path} to check for peer "
                "commits; the reload fence falls back to the notification "
                f"flag alone: {exc}",
            )
            return UNREADABLE
        else:
            fingerprints.append((st.st_mtime_ns, st.st_size))
    return tuple(fingerprints)


def adopted(sampled: Fingerprint | object) -> Fingerprint | None:
    """The value to record for ``sampled``.

    ``UNREADABLE`` becomes ``None``, which differs from any real file: the next
    check therefore re-samples and, if the ``stat`` works by then, reloads
    once. That is the harmless direction.
    """
    return None if sampled is UNREADABLE else sampled  # type: ignore[return-value]


def peer_commit_detected(
    paths: Sequence[str], recorded: Fingerprint | None, *, workspace: str
) -> bool:
    """Whether the files on disk differ from the ones this process recorded.

    The fence's authoritative test -- the one a failed notification cannot
    disable. ``False`` in single-process mode and on an unreadable ``stat``.
    """
    if not fence_enabled():
        return False
    sampled = sample(paths, workspace=workspace)
    if sampled is UNREADABLE:
        return False
    return sampled != recorded
