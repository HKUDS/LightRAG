"""Bounded scan job store (LR2 §8.6).

``/scan`` is asynchronous: it returns a ``track_id`` the instant the scan job
record exists and the managed child has taken over, long before the directory
walk / FAILED reset / document processing finish. Progress is reported into a
job record that is **bounded by construction** so a million-file scan — or a
buggy client — can never materialize an O(total_files) object into the (possibly
Manager-server-hosted) store:

* aggregate integer counters, a fixed allowlist-capped number of distinct keys;
* three fixed sample buckets (processed / warning / error), each a bounded list
  of UTF-8-byte-capped strings plus ``truncated`` / ``dropped`` markers;
* a whole-record serialized-byte ceiling re-checked on every mutation.

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
    # the new job on ACCEPTED, None on CAPACITY_EXCEEDED.
    record: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ScanJobUpdateResult:
    ok: bool
    conflict: Optional[ScanJobUpdateConflict] = None
    # Current public snapshot (present on success AND on VERSION/TERMINAL so the
    # caller can re-sync); None on NOT_FOUND / OWNER.
    record: Optional[Dict[str, Any]] = None


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

    def approx_bytes(self) -> int:
        # Cheap serialized-size estimate: counters + all sample bytes + a small
        # fixed overhead for the scalar fields. Enough to enforce the ceiling
        # without a real serialize on every append.
        counts_bytes = sum(len(k) + 8 for k in self.counts)
        sample_bytes = sum(b.approx_bytes() for b in self.samples.values())
        return counts_bytes + sample_bytes + 256

    def to_public(self) -> Dict[str, Any]:
        """Bounded, owner-token-free snapshot for /scan/status responses."""
        return {
            "track_id": self.track_id,
            "status": self.status,
            "counts": dict(self.counts),
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
        self._lease_seconds = lease_seconds
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: Dict[str, _ScanJobRecord] = {}

    # -- internal, lock held ------------------------------------------------

    def _maybe_abandon_locked(self, rec: _ScanJobRecord, now: float) -> None:
        """A RUNNING job whose lease expired is atomically reaped to ABANDONED
        (owner SIGKILLed / stalled). A later owner-checked completion then loses
        the CAS, so a stale writer cannot resurrect it."""
        if rec.status == ScanJobStatus.RUNNING.value and rec.lease_expires_at <= now:
            rec.status = ScanJobStatus.ABANDONED.value
            rec.updated_at = now
            rec.version += 1
            rec.message = "Lease expired: owner presumed dead; job abandoned."

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
        (idempotent), so a retried create never duplicates or clobbers."""
        with self._lock:
            now = self._clock()
            self._reap_locked(now)
            existing = self._jobs.get(track_id)
            if existing is not None:
                return ScanJobCreateResult(
                    ScanJobCreateOutcome.ALREADY_EXISTS, existing.to_public()
                )
            if (
                len(self._jobs) >= self._capacity
                and not self._evict_one_terminal_locked()
            ):
                return ScanJobCreateResult(ScanJobCreateOutcome.CAPACITY_EXCEEDED)
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
        count cap, whole-record byte ceiling, distinct-counter-key cap. Renews
        the lease and bumps the version on success. Refuses (without mutating) on
        owner mismatch, version mismatch, or a terminal/abandoned job."""
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
                if not isinstance(delta, int):
                    continue
                if key not in rec.counts and len(rec.counts) >= self._max_counter_keys:
                    # Distinct-key cap: silently drop an over-cap new key rather
                    # than grow unboundedly (a buggy client can't blow the size).
                    continue
                rec.counts[key] = rec.counts.get(key, 0) + delta

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
                rec.message = _cap_sample(message, self._sample_max_bytes)[0]
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
                rec.message = _cap_sample(message, self._sample_max_bytes)[0]
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
