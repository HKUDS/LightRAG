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
* **A multi-file storage must look completely published**, judged from the
  files themselves: it publishes them in a fixed order and renames the last one
  -- the commit marker -- LAST, so a completed publication leaves the marker
  STRICTLY newer than the files it commits. Reloading a torn set is the
  corruption vector, and strictness is what keeps a coarse filesystem clock
  from passing one off as complete. ``paths`` is therefore given in PUBLICATION
  ORDER. See :func:`publication_complete`.

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

    **A multi-file storage must look completely published**, judged by
    :func:`publication_complete` from the files themselves. A publication
    renames its files one at a time, so there is a window in which they do not
    describe one state, and reloading THAT is what corrupts -- FAISS would pair
    one generation's ``.index`` with another's ``.meta.json`` and bind one
    row's metadata to another's vector. A pair that does not look complete
    reports "no change": keeping an older self-consistent snapshot is always
    better than adopting an inconsistent one, and it is what this process did
    before the fence existed. The writer completes or retries the publication
    (its in-memory state plus its redo logs are the authority), and the next
    check sees a complete set.

    ``recorded is None`` means "nothing recorded, or deliberately
    invalidated" and reports a change for any complete state -- it is how a
    storage arms the fence after a failure it must reload out of.
    """
    if not fence_enabled():
        return False
    sampled = sample(paths, workspace=workspace)
    if sampled is UNREADABLE:
        return False
    if sampled == recorded:
        return False
    if len(paths) > 1 and not publication_complete(sampled):
        log_without_raising(
            logger.warning,
            f"[{workspace}] The storage files ({', '.join(paths)}) do not look "
            "completely published — the commit marker is older than the files "
            "it commits, or the set is partial. Treating this as an "
            "interrupted publication, not a peer commit, and keeping the "
            "snapshot this process already holds; the writer's retry "
            "completes the set.",
        )
        return False
    return True


def publication_complete(sampled: Fingerprint) -> bool:
    """Whether a multi-file sample looks like one completed publication.

    **The last path is the commit marker.** A storage whose state spans
    several files publishes them in a fixed order and renames the marker
    LAST, so a completed publication leaves the marker STRICTLY newer than
    every file it commits, and an interrupted one leaves some file at or
    after the marker. That test is what makes completeness judgeable from the
    files alone -- by any process, at any generation of staleness.

    **Strictly**, not "no older", and that is the whole guard against coarse
    filesystem timestamps. A torn set can land its next file inside the same
    timestamp tick as the previous publication's marker, making the two
    mtimes equal; ``<=`` would call that complete and hand back a set whose
    files describe different generations. ``<`` refuses it.

    The cost is in the safe direction: a COMPLETE publication whose writes all
    land inside one tick is also refused, so its commit goes unnoticed by this
    channel. That is a missed reload, not a torn one -- the process keeps a
    self-consistent snapshot, the flag channel carries that commit (a
    same-tick publication means a healthy, fast-committing writer, which is
    exactly when the flag works), and the next publication that spans a tick
    is seen. Compare the mirror residue on the change-detection side, where
    the same tick collision also costs a detection rather than correctness.

    Comparing per-file changes against what a reader last recorded cannot do
    that: a reader several generations behind sees every file changed even
    when the newest publication is half-landed, so it would take a torn pair
    for a commit. Ordering is a property of the files; a recorded fingerprint
    is a property of one reader.

    Everything absent is complete (the post-``drop`` state, which peers must
    be able to converge on). A present marker with any file missing is not.
    """
    *committed, marker = sampled
    if marker is None:
        return all(entry is None for entry in committed)
    if any(entry is None for entry in committed):
        return False
    marker_mtime = marker[0]
    return all(entry[0] < marker_mtime for entry in committed)
