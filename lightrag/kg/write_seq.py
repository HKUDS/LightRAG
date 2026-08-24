"""Per-write ordering token for the file-backed vector stores.

``NanoVectorDBStorage`` and ``FaissVectorDBStorage`` keep a redo log of
materialized-but-unsaved upserts (issue #3688). After a reload the flush has to
decide, for every logged row, whether the row a *foreign* writer committed
under the same (content-hash) id meanwhile is newer than the logged one: a
strictly newer row must be left alone (it is a completed reprocess), an older
one must be overwritten by the replay.

``__created_at__`` alone cannot answer that. ``upsert`` stamps
``int(time.time())``, so two writes inside the same second carry the *same*
timestamp — a normal, reachable outcome rather than an anomaly — and a tie used
to fall through to "replay", which let a stale redo record overwrite a
genuinely newer durable row.

``__write_seq__`` is the tiebreaker. It is stamped into the record at
``upsert`` time next to ``__created_at__``, persisted with the record, and
therefore survives the save/reload round-trip the redo log depends on. Its
value is a nanosecond wall-clock reading bumped past the last value this
process handed out, so it is:

* **strictly increasing per process** — a coarse clock (Windows' ~15ms
  granularity) or a backwards NTP step still yields distinct, ordered tokens
  for our own successive writes;
* **comparable across processes** sharing a store — they share one host clock,
  the same assumption ``__created_at__`` already makes for the whole-second
  comparison.

Two limits are deliberately left in place and documented on the comparison
helpers rather than papered over:

* it orders *stamp* time, not *commit* time — a writer that stamps late and
  commits early is still ordered by when it stamped, which is the intent
  order the supersede rule wants;
* a row written before this field existed carries none, so a comparison
  involving such a row falls back to the whole-second behavior.
"""

import threading
import time

# Record key the token is stored under. Dunder-prefixed like ``__id__`` /
# ``__created_at__`` so it cannot collide with a ``meta_fields`` name, and
# stripped from the public read results (``_format_record`` / ``query``) —
# it is storage bookkeeping, not payload.
WRITE_SEQ_FIELD = "__write_seq__"

_seq_lock = threading.Lock()
_last_seq = 0


def next_write_seq() -> int:
    """Return a strictly increasing per-write ordering token.

    Callers stamp one token per ``upsert`` call (every row in that batch
    shares it): the token orders *writes*, and the redo-log comparison only
    ever puts rows from two different writes side by side.
    """
    global _last_seq
    with _seq_lock:
        seq = max(time.time_ns(), _last_seq + 1)
        _last_seq = seq
        return seq


def row_is_strictly_newer(candidate: dict | None, reference: dict, /) -> bool:
    """True when ``candidate`` was written strictly after ``reference``.

    Both are stored/logged records. Ordering is ``__created_at__`` first
    (whole seconds, present on every row ever written by these backends) and
    ``__write_seq__`` only as the same-second tiebreaker, so a row that
    predates the token still orders correctly against a newer second.

    A missing token is *not* treated as "older": a legacy row carries none, and
    reading its absence as ``0`` would declare it older than any freshly
    stamped row and let the replay overwrite it. When either side lacks the
    token a same-second pair therefore stays a tie, which resolves to False —
    the pre-token behavior (the replay proceeds, the read paths serve the
    logged record).
    """
    if candidate is None:
        return False
    candidate_at = candidate.get("__created_at__", 0)
    reference_at = reference.get("__created_at__", 0)
    if candidate_at != reference_at:
        return candidate_at > reference_at
    candidate_seq = candidate.get(WRITE_SEQ_FIELD)
    reference_seq = reference.get(WRITE_SEQ_FIELD)
    if candidate_seq is None or reference_seq is None:
        return False
    return candidate_seq > reference_seq
