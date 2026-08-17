"""Bounded scan job store (LR2 §8.6).

``/scan`` is asynchronous: it returns a ``track_id`` the instant the scan job
record exists and the managed child has taken over, long before the directory
walk / FAILED reset / document processing finish. Progress is reported into a
job record that is **bounded by construction** so a million-file scan — or a
buggy client — can never materialize an O(total_files) object into the (possibly
Manager-server-hosted) store:

* aggregate integer counters — a capped number of distinct keys, each key
  itself UTF-8-byte-capped, so neither the key set nor one key can grow it;
* three fixed sample buckets (processed / warning / error), each a bounded list
  of UTF-8-byte-capped strings plus ``truncated`` / ``dropped`` markers;
* a whole-record serialized-byte ceiling, measured when the record is created
  and re-checked on every mutation that can grow it — counters, samples and the
  terminal status message alike.

The update protocol only accepts ``count deltas + at most one bounded sample +
the expected owner token & version`` — never a full record — and every limit is
re-validated **inside** the store (server-side under the Manager hub), so a
wrong client cannot bypass them.

This module is the single-workspace reference implementation used directly in
single-process mode (:class:`AsyncioScanJobStore`) and, per-namespace, behind
the Manager hub in multi-worker mode. It is deliberately lock-based
(``threading.Lock``) with no blocking waits, so the identical core runs on the
event loop and on a Manager server thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.managers import BaseProxy
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bounds (module-level so the API layer / tests can override per store).
# ---------------------------------------------------------------------------

# Max concurrent job records per workspace before capacity eviction kicks in.
SCAN_JOB_STORE_CAPACITY = 128
# Max samples retained per bucket (processed / warning / error).
SCAN_JOB_SAMPLE_LIMIT = 20
# UTF-8 byte cap for a single sample string.
SCAN_JOB_SAMPLE_MAX_BYTES = 512
# Whole-record serialized-byte ceiling (belt-and-suspenders over the per-bucket
# and per-sample caps): a sample that would push the record past this is dropped.
SCAN_JOB_RECORD_MAX_BYTES = 65_536
# Max distinct counter keys — bounds the counts dict independently of the
# classification taxonomy, so a buggy client cannot grow it without limit.
SCAN_JOB_MAX_COUNTER_KEYS = 32
# UTF-8 byte cap for ONE counter key. Capping the number of keys alone does not
# bound the counts dict: the record-byte ceiling is only re-checked where the
# payload lives (the sample path), so without this a single 100 KB key would
# sail past ``record_max_bytes``. Every in-tree key is a short taxonomy label
# (the longest is ``resume_same_physical_source``), so this only ever refuses a
# client bug.
SCAN_JOB_COUNTER_KEY_MAX_BYTES = 64
# UTF-8 byte cap for the identifiers a record is created with (``track_id`` /
# ``owner_token``). They are never truncated — a clipped track_id would collide
# with another job and break ``/scan/status/{track_id}`` — so an over-cap one is
# refused at create. In-tree values are a generated track id and a uuid4 hex.
SCAN_JOB_IDENTIFIER_MAX_BYTES = 128
# Ceiling on the magnitude of one counter. Python ints are arbitrary precision,
# so without this a 1000-digit delta is ~450 bytes that ``approx_bytes`` counts
# as 8. 2**53 is also the last integer a JSON client can read back exactly, and
# these records are serialized to JSON — a counter past it is not representable
# anyway, so refusing is more honest than storing it.
SCAN_JOB_COUNTER_VALUE_MAX = 2**53
# Lease renewed on every owner update; a RUNNING job whose lease has expired is
# reaped to ABANDONED (owner presumed dead / stalled).
SCAN_JOB_LEASE_SECONDS = 60.0
# A terminal job is evictable once it is older than this (by updated_at).
SCAN_JOB_TTL_SECONDS = 3_600.0

# The three fixed sample buckets (LR2 §8.6).
SAMPLE_BUCKETS: Tuple[str, ...] = ("processed", "warning", "error")


class ScanJobStatus(str, Enum):
    """Lifecycle status of one scan job (stored as its ``.value`` string)."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


# The status word is measured at its WIDEST enum value rather than its current
# one, for the same reason counter values are measured at a fixed 8 bytes: the
# domain is closed, and a RUNNING → ABANDONED transition (7 → 9 bytes) would
# otherwise grow a record the growth paths had already filled right up to the
# ceiling. Fixing the width makes every status transition size-neutral instead of
# adding a third thing to reserve for.
_STATUS_MAX_BYTES = max(len(status.value.encode("utf-8")) for status in ScanJobStatus)


_TERMINAL_STATUSES = frozenset(
    {
        ScanJobStatus.COMPLETED.value,
        ScanJobStatus.FAILED.value,
        ScanJobStatus.CANCELLED.value,
        ScanJobStatus.ABANDONED.value,
    }
)


class ScanJobCreateOutcome(str, Enum):
    ACCEPTED = "accepted"
    ALREADY_EXISTS = "already_exists"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    # A record the store refuses to hold: an identifier over the per-identifier
    # byte cap, or an empty record already over the whole-record ceiling (a
    # pathological workspace name). A client/config bug, never a reachable state
    # for an in-tree caller.
    INVALID_IDENTIFIER = "invalid_identifier"


class ScanJobUpdateConflict(str, Enum):
    """Why an update / status transition was refused (all non-fatal)."""

    NOT_FOUND = "not_found"
    OWNER = "owner"  # owner token mismatch (a stale/late owner)
    VERSION = "version"  # CAS version mismatch (concurrent update)
    TERMINAL = "terminal"  # job already terminal (or just reaped to ABANDONED)
    INVALID_STATUS = "invalid_status"  # illegal transition target


@dataclass(frozen=True)
class ScanJobCreateResult:
    outcome: ScanJobCreateOutcome
    # Public (owner-token-free) snapshot: the existing job on ALREADY_EXISTS,
    # the new job on ACCEPTED, None on CAPACITY_EXCEEDED / INVALID_IDENTIFIER.
    record: Optional[Dict[str, Any]] = None
    # Why the create was refused (INVALID_IDENTIFIER only).
    message: str = ""


@dataclass(frozen=True)
class ScanJobUpdateResult:
    ok: bool
    conflict: Optional[ScanJobUpdateConflict] = None
    # Current public snapshot (present on success AND on VERSION/TERMINAL so the
    # caller can re-sync); None on NOT_FOUND / OWNER.
    record: Optional[Dict[str, Any]] = None


def _key_bytes(key: str) -> int:
    """UTF-8 byte length of a counter key (the unit every counts bound uses)."""
    return len(key.encode("utf-8", errors="replace"))


def _cap_sample(text: str, max_bytes: int) -> Tuple[str, bool]:
    """Truncate ``text`` to at most ``max_bytes`` UTF-8 bytes on a char boundary.

    Returns ``(capped, truncated)``. Never splits a multi-byte codepoint."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text, False
    # Trim to the last full codepoint within the budget.
    clipped = raw[:max_bytes]
    return clipped.decode("utf-8", errors="ignore"), True


@dataclass
class _SampleBucket:
    items: List[str] = field(default_factory=list)
    truncated: bool = False  # at least one retained sample was byte-truncated
    dropped: int = 0  # samples not retained (count cap or record-byte cap)

    def approx_bytes(self) -> int:
        return sum(len(s.encode("utf-8", errors="replace")) for s in self.items)

    def to_public(self) -> Dict[str, Any]:
        return {
            "items": list(self.items),
            "truncated": self.truncated,
            "dropped": self.dropped,
        }


@dataclass
class _ScanJobRecord:
    track_id: str
    workspace: str
    owner_token: str
    status: str
    counts: Dict[str, int]
    samples: Dict[str, _SampleBucket]
    created_at: float
    updated_at: float
    lease_expires_at: float
    version: int
    message: str = ""
    # Counter deltas refused by a bound (over-long key, distinct-key cap, or the
    # record ceiling). A fixed integer, so surfacing it cannot itself grow the
    # record — and a dropped counter is never silent.
    counters_dropped: int = 0

    def approx_bytes(self) -> int:
        """Cheap serialized-size estimate, in the same UTF-8 bytes the caps use.

        Every variable-length part is measured, not assumed: a flat allowance for
        "the scalar fields" made the ceiling bypassable by whatever the caller
        put in ``track_id`` / ``owner_token`` / ``message``. Two fields are
        counted at a fixed worst-case width instead, both because their domain is
        closed: counter VALUES at 8 bytes (:data:`SCAN_JOB_COUNTER_VALUE_MAX`
        keeps them inside 64 bits) and ``status`` at
        :data:`_STATUS_MAX_BYTES`, which makes a status transition
        size-neutral rather than a 2-byte way past the ceiling.
        """
        counts_bytes = sum(_key_bytes(k) + 8 for k in self.counts)
        sample_bytes = sum(b.approx_bytes() for b in self.samples.values())
        scalar_bytes = _STATUS_MAX_BYTES + sum(
            _key_bytes(value)
            for value in (
                self.track_id,
                self.workspace,
                self.owner_token,
                self.message,
            )
        )
        # Timestamps, version, counters_dropped and the JSON scaffolding.
        return counts_bytes + sample_bytes + scalar_bytes + 128

    def to_public(self) -> Dict[str, Any]:
        """Bounded, owner-token-free snapshot for /scan/status responses."""
        return {
            "track_id": self.track_id,
            "status": self.status,
            "counts": dict(self.counts),
            "counters_dropped": self.counters_dropped,
            "samples": {k: b.to_public() for k, b in self.samples.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "message": self.message,
        }


def _new_samples() -> Dict[str, _SampleBucket]:
    return {bucket: _SampleBucket() for bucket in SAMPLE_BUCKETS}


class AsyncioScanJobStore:
    """Single-workspace bounded scan job store (LR2 §8.6).

    Thread-safe via one ``threading.Lock`` and free of blocking waits, so the
    same instance backs both the single-process asyncio path and one namespace
    of the Manager server hub. Capacity, TTL and lease are enforced per
    workspace; ``clock`` is injectable for deterministic TTL/lease tests.
    """

    def __init__(
        self,
        workspace: str = "",
        *,
        capacity: int = SCAN_JOB_STORE_CAPACITY,
        sample_limit: int = SCAN_JOB_SAMPLE_LIMIT,
        sample_max_bytes: int = SCAN_JOB_SAMPLE_MAX_BYTES,
        record_max_bytes: int = SCAN_JOB_RECORD_MAX_BYTES,
        max_counter_keys: int = SCAN_JOB_MAX_COUNTER_KEYS,
        counter_key_max_bytes: int = SCAN_JOB_COUNTER_KEY_MAX_BYTES,
        counter_value_max: int = SCAN_JOB_COUNTER_VALUE_MAX,
        identifier_max_bytes: int = SCAN_JOB_IDENTIFIER_MAX_BYTES,
        lease_seconds: float = SCAN_JOB_LEASE_SECONDS,
        ttl_seconds: float = SCAN_JOB_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._workspace = workspace
        self._capacity = max(1, capacity)
        self._sample_limit = max(0, sample_limit)
        self._sample_max_bytes = max(1, sample_max_bytes)
        self._record_max_bytes = max(1, record_max_bytes)
        self._max_counter_keys = max(1, max_counter_keys)
        self._counter_key_max_bytes = max(1, counter_key_max_bytes)
        self._counter_value_max = max(1, counter_value_max)
        self._identifier_max_bytes = max(1, identifier_max_bytes)
        self._lease_seconds = lease_seconds
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: Dict[str, _ScanJobRecord] = {}

    # -- internal, lock held ------------------------------------------------

    def _fit_message_locked(self, rec: _ScanJobRecord, message: str) -> str:
        """Cap a status message by its own byte cap AND the record's free budget.

        The message is the last variable-length thing written to a record, and it
        REPLACES whatever was there (so the outgoing message's bytes come back
        into the budget first). Capping it against the remaining budget — not only
        against ``sample_max_bytes`` — is what makes "the whole-record ceiling is
        re-checked on every mutation" true of the status transitions too; without
        it a record filled to the ceiling by samples/counters could still be
        pushed one message past it.

        Never squeezes a real deployment: the sample buckets can hold at most
        ``3 × sample_limit × sample_max_bytes`` (30 KB at the defaults) plus a
        capped counts dict, so the free budget under a 64 KB ceiling is always
        tens of kilobytes. It only bites a store configured with a ceiling
        smaller than one message, where truncating the diagnostic is still better
        than breaching the bound the store advertises.
        """
        remaining = self._record_max_bytes - (
            rec.approx_bytes() - _key_bytes(rec.message)
        )
        budget = min(self._sample_max_bytes, remaining)
        if budget <= 0:
            return ""
        return _cap_sample(message, budget)[0]

    def _maybe_abandon_locked(self, rec: _ScanJobRecord, now: float) -> None:
        """A RUNNING job whose lease expired is atomically reaped to ABANDONED
        (owner SIGKILLed / stalled). A later owner-checked completion then loses
        the CAS, so a stale writer cannot resurrect it."""
        if rec.status == ScanJobStatus.RUNNING.value and rec.lease_expires_at <= now:
            rec.status = ScanJobStatus.ABANDONED.value
            rec.updated_at = now
            rec.version += 1
            rec.message = self._fit_message_locked(
                rec, "Lease expired: owner presumed dead; job abandoned."
            )

    def _reap_locked(self, now: float) -> None:
        """Reap lease-expired RUNNING jobs to ABANDONED, then evict terminal
        jobs older than the TTL."""
        for rec in self._jobs.values():
            self._maybe_abandon_locked(rec, now)
        expired = [
            track_id
            for track_id, rec in self._jobs.items()
            if rec.status in _TERMINAL_STATUSES
            and (now - rec.updated_at) >= self._ttl_seconds
        ]
        for track_id in expired:
            del self._jobs[track_id]

    def _evict_one_terminal_locked(self) -> bool:
        """Evict the OLDEST terminal (by updated_at) to make room. Returns False
        when every job is a still-valid RUNNING one (caller → CAPACITY_EXCEEDED —
        a valid running job is never force-removed)."""
        terminal = [
            (rec.updated_at, track_id)
            for track_id, rec in self._jobs.items()
            if rec.status in _TERMINAL_STATUSES
        ]
        if not terminal:
            return False
        terminal.sort()
        del self._jobs[terminal[0][1]]
        return True

    # -- public API ---------------------------------------------------------

    def create(self, track_id: str, owner_token: str) -> ScanJobCreateResult:
        """Create a RUNNING job record (LR2 §8.6 capacity policy).

        Capacity order: reap expired terminal/abandoned → evict oldest terminal
        → if still full (all valid RUNNING) refuse with CAPACITY_EXCEEDED. An
        existing ``track_id`` returns ALREADY_EXISTS with the existing record
        (idempotent), so a retried create never duplicates or clobbers.

        An identifier past :data:`SCAN_JOB_IDENTIFIER_MAX_BYTES` is refused with
        INVALID_IDENTIFIER: it is the one part of a record that cannot be
        truncated (a clipped ``track_id`` would answer another job's status
        query), so the record ceiling can only hold if the store declines it
        here.

        The freshly built record is then measured against ``record_max_bytes``
        before it is stored, and refused the same way if it does not fit. The
        per-identifier cap alone cannot carry that guarantee: ``workspace`` is a
        scalar ``approx_bytes`` counts but no caller passes to ``create``, so a
        pathological namespace would otherwise seat an over-ceiling record that
        every later mutation then (correctly) refuses to grow. Measuring the built
        record keeps the invariant on the record itself, and covers any scalar
        added later without a second place to remember."""
        for label, value in (("track_id", track_id), ("owner_token", owner_token)):
            if _key_bytes(value) > self._identifier_max_bytes:
                return ScanJobCreateResult(
                    ScanJobCreateOutcome.INVALID_IDENTIFIER,
                    message=(
                        f"{label} exceeds {self._identifier_max_bytes} UTF-8 bytes"
                    ),
                )
        with self._lock:
            now = self._clock()
            self._reap_locked(now)
            existing = self._jobs.get(track_id)
            if existing is not None:
                return ScanJobCreateResult(
                    ScanJobCreateOutcome.ALREADY_EXISTS, existing.to_public()
                )
            rec = _ScanJobRecord(
                track_id=track_id,
                workspace=self._workspace,
                owner_token=owner_token,
                status=ScanJobStatus.RUNNING.value,
                counts={},
                samples=_new_samples(),
                created_at=now,
                updated_at=now,
                lease_expires_at=now + self._lease_seconds,
                version=1,
            )
            # Measured (and refused) BEFORE the capacity step, so a record that
            # cannot be held never evicts a terminal one on its way out.
            base_bytes = rec.approx_bytes()
            if base_bytes > self._record_max_bytes:
                return ScanJobCreateResult(
                    ScanJobCreateOutcome.INVALID_IDENTIFIER,
                    message=(
                        f"an empty job record for this workspace/track_id already "
                        f"measures {base_bytes} bytes, over the "
                        f"{self._record_max_bytes}-byte record ceiling"
                    ),
                )
            if (
                len(self._jobs) >= self._capacity
                and not self._evict_one_terminal_locked()
            ):
                return ScanJobCreateResult(ScanJobCreateOutcome.CAPACITY_EXCEEDED)
            self._jobs[track_id] = rec
            return ScanJobCreateResult(ScanJobCreateOutcome.ACCEPTED, rec.to_public())

    def update(
        self,
        track_id: str,
        owner_token: str,
        *,
        count_deltas: Optional[Dict[str, int]] = None,
        sample: Optional[Tuple[str, str]] = None,
        expected_version: int,
    ) -> ScanJobUpdateResult:
        """Apply ``count_deltas`` + at most one bounded ``sample`` under CAS.

        ``sample`` is ``(bucket, text)`` with ``bucket in SAMPLE_BUCKETS``. Every
        bound is re-validated here (server-side): per-sample byte cap, per-bucket
        count cap, whole-record byte ceiling, distinct-counter-key cap and
        per-counter-key byte cap. A delta whose key trips a bound is refused and
        tallied in ``counters_dropped``; the rest of the update still applies.
        Renews the lease and bumps the version on success. Refuses (without
        mutating) on owner mismatch, version mismatch, or a terminal/abandoned
        job."""
        with self._lock:
            now = self._clock()
            rec = self._jobs.get(track_id)
            if rec is None:
                return ScanJobUpdateResult(False, ScanJobUpdateConflict.NOT_FOUND)
            self._maybe_abandon_locked(rec, now)
            if rec.owner_token != owner_token:
                return ScanJobUpdateResult(False, ScanJobUpdateConflict.OWNER)
            if rec.status != ScanJobStatus.RUNNING.value:
                return ScanJobUpdateResult(
                    False, ScanJobUpdateConflict.TERMINAL, rec.to_public()
                )
            if rec.version != expected_version:
                return ScanJobUpdateResult(
                    False, ScanJobUpdateConflict.VERSION, rec.to_public()
                )

            for key, delta in (count_deltas or {}).items():
                if not isinstance(delta, int) or isinstance(delta, bool):
                    continue
                if abs(delta) > self._counter_value_max:
                    # Arbitrary-precision int: the payload the record ceiling
                    # assumes is 8 bytes. Refuse (and tally) rather than store a
                    # value no JSON client could read back anyway.
                    rec.counters_dropped += 1
                    continue
                if key in rec.counts:
                    # An existing key adds no bytes, but the accumulated value
                    # must stay inside the same bound as a single delta.
                    if abs(rec.counts[key] + delta) > self._counter_value_max:
                        rec.counters_dropped += 1
                        continue
                    rec.counts[key] += delta
                    continue
                key_bytes = _key_bytes(key)
                if (
                    # Over-long key: REFUSED, never truncated — two long keys
                    # sharing a prefix would otherwise merge into one counter,
                    # silently fusing distinct taxonomy labels.
                    key_bytes > self._counter_key_max_bytes
                    # Distinct-key cap.
                    or len(rec.counts) >= self._max_counter_keys
                    # Record ceiling, re-checked here too so the claim holds on
                    # EVERY mutation and not just the sample path.
                    or rec.approx_bytes() + key_bytes + 8 > self._record_max_bytes
                ):
                    rec.counters_dropped += 1
                    continue
                rec.counts[key] = delta

            if sample is not None:
                bucket_name, text = sample
                bucket = rec.samples.get(bucket_name)
                if bucket is not None:
                    capped, truncated = _cap_sample(text, self._sample_max_bytes)
                    would_add = len(capped.encode("utf-8", errors="replace"))
                    if len(bucket.items) >= self._sample_limit:
                        bucket.dropped += 1
                    elif rec.approx_bytes() + would_add > self._record_max_bytes:
                        # Record-byte ceiling reached: drop rather than exceed it.
                        bucket.dropped += 1
                    else:
                        bucket.items.append(capped)
                        bucket.truncated = bucket.truncated or truncated

            rec.version += 1
            rec.updated_at = now
            rec.lease_expires_at = now + self._lease_seconds
            return ScanJobUpdateResult(True, None, rec.to_public())

    def set_status(
        self,
        track_id: str,
        owner_token: str,
        status: ScanJobStatus,
        *,
        expected_version: int,
        message: str = "",
    ) -> ScanJobUpdateResult:
        """Owner-checked, CAS'd transition RUNNING → a terminal status.

        A late completion cannot overwrite a newer status: if the lease already
        expired the job is ABANDONED first, so this owner's expected_version no
        longer matches and the transition loses the CAS (LR2 §8.6 "迟到
        completion 不能覆盖新状态")."""
        if status.value not in _TERMINAL_STATUSES:
            return ScanJobUpdateResult(False, ScanJobUpdateConflict.INVALID_STATUS)
        with self._lock:
            now = self._clock()
            rec = self._jobs.get(track_id)
            if rec is None:
                return ScanJobUpdateResult(False, ScanJobUpdateConflict.NOT_FOUND)
            self._maybe_abandon_locked(rec, now)
            if rec.owner_token != owner_token:
                return ScanJobUpdateResult(False, ScanJobUpdateConflict.OWNER)
            if rec.status != ScanJobStatus.RUNNING.value:
                return ScanJobUpdateResult(
                    False, ScanJobUpdateConflict.TERMINAL, rec.to_public()
                )
            if rec.version != expected_version:
                return ScanJobUpdateResult(
                    False, ScanJobUpdateConflict.VERSION, rec.to_public()
                )
            rec.status = status.value
            rec.updated_at = now
            rec.version += 1
            if message:
                rec.message = self._fit_message_locked(rec, message)
            return ScanJobUpdateResult(True, None, rec.to_public())

    def get(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Bounded public snapshot (reaps lease-expired RUNNING jobs first)."""
        with self._lock:
            now = self._clock()
            rec = self._jobs.get(track_id)
            if rec is None:
                return None
            self._maybe_abandon_locked(rec, now)
            return rec.to_public()

    def cancel(
        self, track_id: str, owner_token: str, *, message: str = ""
    ) -> ScanJobUpdateResult:
        """Owner-checked cancel of a RUNNING job (version-agnostic).

        Used by the reservation-startup compensation chain, which knows the
        owner but not the live version. A NOT_FOUND / already-terminal job is a
        no-op success from the caller's perspective (idempotent teardown)."""
        with self._lock:
            now = self._clock()
            rec = self._jobs.get(track_id)
            if rec is None:
                return ScanJobUpdateResult(False, ScanJobUpdateConflict.NOT_FOUND)
            if rec.owner_token != owner_token:
                return ScanJobUpdateResult(False, ScanJobUpdateConflict.OWNER)
            if rec.status != ScanJobStatus.RUNNING.value:
                return ScanJobUpdateResult(
                    True, ScanJobUpdateConflict.TERMINAL, rec.to_public()
                )
            rec.status = ScanJobStatus.CANCELLED.value
            rec.updated_at = now
            rec.version += 1
            if message:
                rec.message = self._fit_message_locked(rec, message)
            return ScanJobUpdateResult(True, None, rec.to_public())

    def remove_terminal(self, track_id: str) -> bool:
        """Remove one job only if it is terminal (destructive clear support).

        A valid RUNNING job is never removed directly — the caller must
        owner-checked cancel/abandon it first (LR2 §8.6). Returns True iff a
        terminal record was removed."""
        with self._lock:
            rec = self._jobs.get(track_id)
            if rec is None or rec.status not in _TERMINAL_STATUSES:
                return False
            del self._jobs[track_id]
            return True

    def snapshot(self) -> List[Dict[str, Any]]:
        """Bounded public snapshot of ALL jobs (diagnostics/tests)."""
        with self._lock:
            now = self._clock()
            self._reap_locked(now)
            return [rec.to_public() for rec in self._jobs.values()]


# ============================================================================
# Multiprocess: server-side hub + explicit BaseProxy + per-workspace client view
# (same topology as PipelineIngressHub). One AsyncioScanJobStore per namespace.
# ============================================================================


class ScanJobStoreHub:
    """Server-side ``{namespace: AsyncioScanJobStore}`` map — the ONE
    Manager-registered scan-job object. A store is created lazily under the hub
    lock (entirely inside one server-side dispatch), so a SIGKILLed client
    neither strands the registry nor races a duplicate store. Stores live for
    the Manager server's lifetime; a destructive workspace wipe empties one via
    :meth:`clear` (the same accepted lifecycle trade-off as the ingress hub)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stores: Dict[str, AsyncioScanJobStore] = {}

    def _store(self, namespace: str) -> AsyncioScanJobStore:
        with self._lock:
            store = self._stores.get(namespace)
            if store is None:
                store = AsyncioScanJobStore(namespace)
                self._stores[namespace] = store
            return store

    def create(
        self, namespace: str, track_id: str, owner_token: str
    ) -> ScanJobCreateResult:
        return self._store(namespace).create(track_id, owner_token)

    def update(
        self,
        namespace: str,
        track_id: str,
        owner_token: str,
        count_deltas: Optional[Dict[str, int]],
        sample: Optional[Tuple[str, str]],
        expected_version: int,
    ) -> ScanJobUpdateResult:
        return self._store(namespace).update(
            track_id,
            owner_token,
            count_deltas=count_deltas,
            sample=sample,
            expected_version=expected_version,
        )

    def set_status(
        self,
        namespace: str,
        track_id: str,
        owner_token: str,
        status: ScanJobStatus,
        expected_version: int,
        message: str,
    ) -> ScanJobUpdateResult:
        return self._store(namespace).set_status(
            track_id,
            owner_token,
            status,
            expected_version=expected_version,
            message=message,
        )

    def get(self, namespace: str, track_id: str) -> Optional[Dict[str, Any]]:
        return self._store(namespace).get(track_id)

    def cancel(
        self, namespace: str, track_id: str, owner_token: str, message: str
    ) -> ScanJobUpdateResult:
        return self._store(namespace).cancel(track_id, owner_token, message=message)

    def remove_terminal(self, namespace: str, track_id: str) -> bool:
        return self._store(namespace).remove_terminal(track_id)

    def snapshot(self, namespace: str) -> List[Dict[str, Any]]:
        return self._store(namespace).snapshot()

    def clear(self, namespace: str) -> None:
        # Drop the whole per-workspace store (destructive workspace wipe).
        with self._lock:
            self._stores[namespace] = AsyncioScanJobStore(namespace)


class _ScanJobStoreHubProxy(BaseProxy):
    """Explicit proxy for :class:`ScanJobStoreHub` (BaseProxy has no dynamic
    ``__getattr__``, so each exposed method needs a ``_callmethod`` wrapper —
    deterministic, unlike AutoProxy)."""

    _exposed_ = (
        "create",
        "update",
        "set_status",
        "get",
        "cancel",
        "remove_terminal",
        "snapshot",
        "clear",
    )

    def create(self, namespace, track_id, owner_token):
        return self._callmethod("create", (namespace, track_id, owner_token))

    def update(
        self, namespace, track_id, owner_token, count_deltas, sample, expected_version
    ):
        return self._callmethod(
            "update",
            (namespace, track_id, owner_token, count_deltas, sample, expected_version),
        )

    def set_status(
        self, namespace, track_id, owner_token, status, expected_version, message
    ):
        return self._callmethod(
            "set_status",
            (namespace, track_id, owner_token, status, expected_version, message),
        )

    def get(self, namespace, track_id):
        return self._callmethod("get", (namespace, track_id))

    def cancel(self, namespace, track_id, owner_token, message):
        return self._callmethod("cancel", (namespace, track_id, owner_token, message))

    def remove_terminal(self, namespace, track_id):
        return self._callmethod("remove_terminal", (namespace, track_id))

    def snapshot(self, namespace):
        return self._callmethod("snapshot", (namespace,))

    def clear(self, namespace):
        return self._callmethod("clear", (namespace,))


class ManagerScanJobStore:
    """Per-workspace client view over the shared hub proxy — same method surface
    as :class:`AsyncioScanJobStore`, so callers use either identically.
    Stateless besides the ``(hub, namespace)`` binding, so any process reaches
    the same server-side store (workspace identity is the namespace string)."""

    def __init__(self, hub: Any, namespace: str) -> None:
        self._hub = hub
        self.namespace = namespace

    def create(self, track_id: str, owner_token: str) -> ScanJobCreateResult:
        return self._hub.create(self.namespace, track_id, owner_token)

    def update(
        self,
        track_id: str,
        owner_token: str,
        *,
        count_deltas: Optional[Dict[str, int]] = None,
        sample: Optional[Tuple[str, str]] = None,
        expected_version: int,
    ) -> ScanJobUpdateResult:
        return self._hub.update(
            self.namespace,
            track_id,
            owner_token,
            count_deltas,
            sample,
            expected_version,
        )

    def set_status(
        self,
        track_id: str,
        owner_token: str,
        status: ScanJobStatus,
        *,
        expected_version: int,
        message: str = "",
    ) -> ScanJobUpdateResult:
        return self._hub.set_status(
            self.namespace, track_id, owner_token, status, expected_version, message
        )

    def get(self, track_id: str) -> Optional[Dict[str, Any]]:
        return self._hub.get(self.namespace, track_id)

    def cancel(
        self, track_id: str, owner_token: str, *, message: str = ""
    ) -> ScanJobUpdateResult:
        return self._hub.cancel(self.namespace, track_id, owner_token, message)

    def remove_terminal(self, track_id: str) -> bool:
        return self._hub.remove_terminal(self.namespace, track_id)

    def snapshot(self) -> List[Dict[str, Any]]:
        return self._hub.snapshot(self.namespace)

    def clear(self) -> None:
        self._hub.clear(self.namespace)
