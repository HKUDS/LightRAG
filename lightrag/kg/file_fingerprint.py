"""The file half of the file-backed storages' cross-process fence.

`NetworkXStorage`, `NanoVectorDBStorage` and `FaissVectorDBStorage` all
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

Four rules a caller must not get wrong, each of which has cost a defect:

* **Sample BEFORE reading the file, never after.** A fingerprint taken after
  the parse can belong to a newer file than the one now in memory, and
  recording it would suppress the reload that file needs -- the one way this
  fence could *introduce* a lost write. Callers own this ordering; the
  functions cannot enforce it.
* **Once a call is committed to adopting, it must not observe the file again.**
  Divergence, counting and adoption all run on ONE sample, passed in.
* **An unreadable ``stat`` is not an absent file.** ``UNREADABLE`` reports "no
  change", degrading the fence to the flag alone; failing towards a reload
  would install an empty snapshot the next commit then serializes over the
  real one.
* **A multi-file storage must look completely published.** ``paths`` is given
  in PUBLICATION ORDER, marker last; see :func:`publication_complete`.

``st_ino`` is deliberately excluded: ``atomic_write`` frees the previous inode
and the allocator hands it straight back (measured at 28 reuses in 30 commits),
so it discriminates nothing. Single-process mode skips the test entirely.

**Full contract: ``docs/design/FileBackedSnapshotContract.md``** -- the
``_missed_notification_reloads`` counter contract and the five ways review
found it biased, the deferred ``os.utime`` remedy and why it waits on that
counter, and the two blind spots that remain by design.
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
    once. That is the safe direction for the *reload decision* — the residue
    below is what it is not harmless about.

    **Accepted residue — one spurious divergence per unreadable adoption.**
    ``None`` means "nothing recorded", and every state reads as a change
    against it, so a reload whose sample came back ``UNREADABLE`` leaves the
    *next* call reporting a divergence even when the file never moved. It is
    the same overcount the module docstring's item 1 rejects, arriving by a
    different door: there the fingerprint was invalidated deliberately, here
    a failed ``stat`` does it.

    Bounded and self-healing: the next readable ``stat`` adopts a concrete
    state, so it cannot repeat without the ``stat`` failing again, and it
    biases the counter *up*, never down — an inflated counter argues for the
    ``os.utime`` remedy the module docstring defers, which is the direction
    that costs work rather than data.

    **Only a storage that can REPLAY may reload on an unreadable sample.**
    That is what keeps the residue at one reload. A reload discards whatever
    the process has mutated in memory and not yet committed, and the
    storages differ in what that costs:

    * The vector backends replay: ``index_done_callback`` reloads and then
      ``_flush_pending_locked`` re-applies the pending buffers over the
      reloaded snapshot. A spurious reload costs them work and nothing else,
      so they adopt ``None`` and move on.
    * ``NetworkXStorage`` has no redo log, so a reload DISCARDS those
      mutations. Between batches that is the intended behaviour; reached
      spuriously it can land *mid*-batch, and then the mutations dropped are
      simply absent from the commit that follows — which SUCCEEDS, marking
      their document PROCESSED. A silent partial loss, neither loud nor
      self-healing. Its ``_reload_locked`` therefore refuses to load on an
      ``UNREADABLE`` sample at all (it raises, which fails the batch loudly
      and retries), so no reload of its graph can reach this function with
      one. What still can are its adoptions of its OWN commit or drop
      (``_record_fingerprint``), where the in-memory graph already equals the
      file — so the redundant reload discards nothing, **provided it happens
      before the next mutation**. It does not on its own: ``UNREADABLE``
      reports "no change", so while the ``stat`` keeps failing no divergence
      fires and the reload would be deferred to whichever later call first
      managed to ``stat`` — mid-batch, dropping what was applied in between.
      ``_get_graph`` therefore settles a ``None`` fingerprint before it
      serves the graph — through the divergence test when it can sample (any
      concrete state is a divergence against ``None``), by refusing when it
      cannot.

    Recorded as an option, not done: the obvious fix is to keep the previously
    recorded fingerprint instead of clearing it (never worse for the reload
    decision, since the file only moves forward and an under-recorded state
    fails towards a redundant reload). It is entangled, though — each
    storage's ``_adopt_fingerprint`` clears ``_counted_peer_fingerprint``
    exactly when this returns a concrete state, so retaining one here would
    clear the dedupe marker on an adoption that did not actually observe the
    file, and re-counting would return through
    :func:`counts_as_a_new_lost_notification`. Both halves have to move
    together, with tests for the pair, which is more than a residue this small
    justifies today.
    """
    return None if sampled is UNREADABLE else sampled  # type: ignore[return-value]


def counts_as_a_new_lost_notification(
    sampled: Fingerprint | object, already_counted: Fingerprint | None
) -> bool:
    """Whether this detection is a NEW lost notification, not a re-detection.

    Every storage here keeps a ``_missed_notification_reloads`` counter, and
    the module docstring designates those counters as the evidence the
    writer-side ``os.utime`` remedy waits on. That only holds if they count
    **distinct unannounced states**, and detection alone does not: a reload
    that raises leaves the reader's recorded fingerprint untouched, so the
    same state is re-detected by every later call. Counted at each detection,
    one state inflates the counter without bound -- and a file that stays
    unreadable for a while is not exotic, since that is what a sick storage
    looks like.

    So each storage remembers the state it last counted and passes it here.
    A genuinely different state counts again. Two things do not, and both are
    by design (see the module docstring): commits inside one timestamp tick
    with an identical size, which produce no new state at all, and commits
    that batch into a single observation because this process did not look in
    between. This function is the wrong place to fix either -- it is handed a
    state and can only compare it with the last one.

    ``UNREADABLE`` counts nothing: it cannot say WHICH state it would be
    counting, so the count could neither be deduplicated nor trusted. The next
    call counts it if the ``stat`` works by then -- but ONLY because the
    caller feeds this same sample to its reload, so an unreadable one adopts
    no fingerprint and leaves the divergence standing. A caller that sampled
    here and let its reload sample independently would lose the event for
    good: this would skip the count while that sample succeeded and adopted
    the peer state. Count and adopt from one observation.

    Counting at detection rather than after a successful reload is deliberate:
    the window occurred whether or not this process could reload out of it,
    and a file that never becomes readable would otherwise erase the evidence
    entirely.
    """
    if sampled is UNREADABLE:
        return False
    return sampled != already_counted


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
    return divergence_detected(
        sample(paths, workspace=workspace), recorded, paths=paths, workspace=workspace
    )


def divergence_detected(
    sampled: Fingerprint | object,
    recorded: Fingerprint | None,
    *,
    paths: Sequence[str],
    workspace: str,
) -> bool:
    """:func:`peer_commit_detected`'s decision, from a sample already taken.

    For the caller that will DECIDE, COUNT and ADOPT within one call: all
    three must come from **one observation**. Taking a fresh ``stat`` for the
    decision lets it fail while the caller's sample succeeded, and then the
    count is skipped while the reload adopts that good sample -- erasing the
    divergence that would have let a later call count the event. The
    one-observation rule in :func:`counts_as_a_new_lost_notification` covers
    counting and adoption; this covers the third participant.

    A caller with nothing else to do with the sample should use
    :func:`peer_commit_detected`, which takes one for itself.
    """
    if not fence_enabled():
        return False
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

    **This test is timestamp-based because the formats give it nothing else.**
    The deferred remedy, recorded so it is not rediscovered: put an explicit
    generation counter in the metadata each file carries (FAISS's
    ``meta.json``), or publish the set atomically -- stage a whole generation
    in a directory and rename that directory into place -- and completeness
    becomes a comparison of equal generation numbers, independent of the
    filesystem clock and of the strictness argument above. It is not done here
    because it is a storage-format change, and these multi-file backends are
    development and test storage today. It is the right answer if they ever
    become production storage.
    """
    *committed, marker = sampled
    if marker is None:
        return all(entry is None for entry in committed)
    if any(entry is None for entry in committed):
        return False
    marker_mtime = marker[0]
    return all(entry[0] < marker_mtime for entry in committed)
