import os
import sys
import asyncio
import multiprocessing as mp
import random
import uuid
from multiprocessing.synchronize import Lock as ProcessLock
from multiprocessing import Manager
import time
import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Mapping, Optional, Union, TypeVar, Generic

from lightrag.constants import (
    DEFAULT_GLOBAL_SLOT_HEARTBEAT_TTL,
    DEFAULT_GLOBAL_SLOT_SUSPECT_GRACE,
    DEFAULT_GLOBAL_SLOT_WAITER_STALE_TTL,
    DEFAULT_QUEUE_STATS_STALE_TTL,
)
from lightrag.exceptions import PipelineNotInitializedError

DEBUG_LOCKS = False


# Define a direct print function for critical logs that must be visible in all processes
def direct_log(message, enable_output: bool = True, level: str = "DEBUG"):
    """
    Log a message directly to stderr to ensure visibility in all processes,
    including the Gunicorn master process.

    Args:
        message: The message to log
        level: Log level for message (control the visibility of the message by comparing with the current logger level)
        enable_output: Enable or disable log message (Force to turn off the message,)
    """
    if not enable_output:
        return

    # Get the current logger level from the lightrag logger
    try:
        from lightrag.utils import logger

        current_level = logger.getEffectiveLevel()
    except ImportError:
        # Fallback if lightrag.utils is not available
        current_level = 20  # INFO

    # Convert string level to numeric level for comparison
    level_mapping = {
        "DEBUG": 10,  # DEBUG
        "INFO": 20,  # INFO
        "WARNING": 30,  # WARNING
        "ERROR": 40,  # ERROR
        "CRITICAL": 50,  # CRITICAL
    }
    message_level = level_mapping.get(level.upper(), logging.DEBUG)

    if message_level >= current_level:
        print(f"{level}: {message}", file=sys.stderr, flush=True)


T = TypeVar("T")
LockType = Union[ProcessLock, asyncio.Lock]

_is_multiprocess = None
_workers = None
_manager = None

# Global singleton data for multi-process keyed locks
_lock_registry: Optional[Dict[str, Any]] = None
_lock_registry_count: Optional[Dict[str, int]] = None
_lock_cleanup_data: Optional[Dict[str, time.time]] = None
_registry_guard = None
# Holder records for multi-process keyed locks (dead-worker recovery). Maps a
# combined key to {owner_pid, process_start_id, lease_id} while the lock is held;
# the record IS the lock (replaces manager.Lock()), so a SIGKILLed holder can be
# reclaimed by a live worker instead of deadlocking every process.
_keyed_lock_holders: Optional[Dict[str, Any]] = None
# Keyed-lock lease poll backoff bounds (seconds).
_KEYED_LEASE_POLL_BASE = 0.02
_KEYED_LEASE_POLL_MAX = 0.5
# Timeout for keyed locks in seconds (Default 300)
CLEANUP_KEYED_LOCKS_AFTER_SECONDS = 300
# Cleanup pending list threshold for triggering cleanup (Default 500)
CLEANUP_THRESHOLD = 500
# Minimum interval between cleanup operations in seconds (Default 30)
MIN_CLEANUP_INTERVAL_SECONDS = 30
# Track the earliest cleanup time for efficient cleanup triggering (multiprocess locks only)
_earliest_mp_cleanup_time: Optional[float] = None
# Track the last cleanup time to enforce minimum interval (multiprocess locks only)
_last_mp_cleanup_time: Optional[float] = None

_initialized = None

# Default workspace for backward compatibility
_default_workspace: Optional[str] = None

# shared data for storage across processes
_shared_dicts: Optional[Dict[str, Any]] = None
_init_flags: Optional[Dict[str, bool]] = None  # namespace -> initialized
_update_flags: Optional[Dict[str, bool]] = None  # namespace -> updated

# locks for mutex access
_internal_lock: Optional[LockType] = None
_data_init_lock: Optional[LockType] = None
# Manager for all keyed locks
_storage_keyed_lock: Optional["KeyedUnifiedLock"] = None

# async locks for coroutine synchronization in multiprocess mode
_async_locks: Optional[Dict[str, asyncio.Lock]] = None

_debug_n_locks_acquired: int = 0

# --- Cross-worker global concurrency gate + queue stats aggregation ---
#
# Read-only configuration set once by the FIRST initialize_share_data() call
# (the gunicorn master, before fork — workers inherit it as a module global).
# Later no-arg calls (e.g. LightRAG.__post_init__) hit the `_initialized`
# guard and never overwrite it.
_global_concurrency_limits: Optional[Dict[str, int]] = None

# Separator between the queue name and the per-pid suffix in queue-stats
# namespace keys. \x1f (ASCII unit separator) cannot appear in queue names.
# (Concurrency gate state needs no separator: one key per group.)
KEY_SEP = "\x1f"

_CONCURRENCY_LEASE_NAMESPACE = "concurrency_leases"
_QUEUE_STATS_NAMESPACE = "queue_stats"

# Heartbeat / staleness parameters (module-level so tests can monkeypatch).
_heartbeat_ttl: float = DEFAULT_GLOBAL_SLOT_HEARTBEAT_TTL
_suspect_grace: float = DEFAULT_GLOBAL_SLOT_SUSPECT_GRACE
_queue_stats_stale_ttl: float = DEFAULT_QUEUE_STATS_STALE_TTL
_waiter_stale_ttl: float = DEFAULT_GLOBAL_SLOT_WAITER_STALE_TTL

# Per-process cached namespace references (avoid the internal lock on every
# publish). Reset by initialize_share_data()/finalize_share_data().
_lease_ns_cache: Optional[Dict[str, Any]] = None
_queue_stats_ns_cache: Optional[Dict[str, Any]] = None

# Rate limiting for acquire-failure warnings (fail-closed path).
_ACQUIRE_FAILURE_LOG_INTERVAL = 30.0
_last_acquire_failure_log: float = 0.0


def get_final_namespace(namespace: str, workspace: str | None = None):
    global _default_workspace
    if workspace is None:
        workspace = _default_workspace

    if workspace is None:
        direct_log(
            f"Error: Invoke namespace operation without workspace, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Invoke namespace operation without workspace")

    final_namespace = f"{workspace}:{namespace}" if workspace else f"{namespace}"
    return final_namespace


def inc_debug_n_locks_acquired():
    global _debug_n_locks_acquired
    if DEBUG_LOCKS:
        _debug_n_locks_acquired += 1
        print(f"DEBUG: Keyed Lock acquired, total: {_debug_n_locks_acquired:>5}")


def dec_debug_n_locks_acquired():
    global _debug_n_locks_acquired
    if DEBUG_LOCKS:
        if _debug_n_locks_acquired > 0:
            _debug_n_locks_acquired -= 1
            print(f"DEBUG: Keyed Lock released, total: {_debug_n_locks_acquired:>5}")
        else:
            raise RuntimeError("Attempting to release lock when no locks are acquired")


def get_debug_n_locks_acquired():
    global _debug_n_locks_acquired
    return _debug_n_locks_acquired


class UnifiedLock(Generic[T]):
    """Provide a unified lock interface type for asyncio.Lock and multiprocessing.Lock"""

    def __init__(
        self,
        lock: Union[ProcessLock, asyncio.Lock],
        is_async: bool,
        name: str = "unnamed",
        enable_logging: bool = True,
        async_lock: Optional[asyncio.Lock] = None,
        mp_is_lease: bool = False,
    ):
        self._lock = lock
        self._is_async = is_async
        self._pid = os.getpid()  # for debug only
        self._name = name  # for debug only
        self._enable_logging = enable_logging  # for debug only
        self._async_lock = async_lock  # auxiliary lock for coroutine synchronization
        # When True, ``_lock`` is a _KeyedLeaseLock whose acquire() is an async
        # holder-record poll (dead-only reclaim) rather than a blocking
        # manager.Lock() acquire offloaded to an executor thread.
        self._mp_is_lease = mp_is_lease

    async def _acquire_mp_lock_in_executor(self) -> None:
        """Acquire the multiprocess lock without blocking the event loop.

        Cancellation safety: if this coroutine is cancelled while the
        executor thread is still blocked inside ``acquire()``, the thread
        cannot be interrupted and WILL take the lock eventually — with no
        owner left to release it, every process would deadlock. The shield +
        done-callback below returns such an orphaned acquisition immediately.
        """
        loop = asyncio.get_running_loop()
        acquire_future = loop.run_in_executor(None, self._lock.acquire)
        try:
            await asyncio.shield(acquire_future)
        except asyncio.CancelledError:

            def _release_orphaned_acquire(f) -> None:
                if f.cancelled() or f.exception() is not None:
                    return
                try:
                    self._lock.release()
                except Exception:
                    pass

            acquire_future.add_done_callback(_release_orphaned_acquire)
            raise

    async def __aenter__(self) -> "UnifiedLock[T]":
        async_gate_acquired = False
        try:
            # If in multiprocess mode and async lock exists, acquire it first
            if not self._is_async and self._async_lock is not None:
                await self._async_lock.acquire()
                async_gate_acquired = True
                direct_log(
                    f"== Lock == Process {self._pid}: Acquired async lock '{self._name}",
                    level="DEBUG",
                    enable_output=self._enable_logging,
                )

            # Acquire the main lock
            # Note: self._lock should never be None here as the check has been moved
            # to get_internal_lock() and get_data_init_lock() functions
            if self._is_async:
                await self._lock.acquire()
            elif self._mp_is_lease:
                # Holder-record lease: an async poll that reclaims a confirmed-
                # dead owner's lease and never blocks the event loop (no executor
                # thread). The async gate above already caps this process to one
                # poller per key.
                await self._lock.acquire()
            else:
                # A Manager lock proxy blocks the calling thread until every
                # other PROCESS ahead of us releases — unbounded. Offload to
                # the default executor so this process's event loop keeps
                # serving while we wait (the async gate above already
                # serializes this process's coroutines, so at most one
                # executor thread per lock key is ever parked here).
                await self._acquire_mp_lock_in_executor()

            direct_log(
                f"== Lock == Process {self._pid}: Acquired lock {self._name} (async={self._is_async})",
                level="INFO",
                enable_output=self._enable_logging,
            )
            return self
        except asyncio.CancelledError:
            # Cancellation can arrive while awaiting the executor-offloaded
            # mp acquire (any orphaned acquisition is returned inside
            # _acquire_mp_lock_in_executor). Roll back the per-process gate
            # we already hold so this process's other coroutines never
            # deadlock on it.
            if async_gate_acquired:
                self._async_lock.release()
            direct_log(
                f"== Lock == Process {self._pid}: Lock acquisition cancelled '{self._name}'",
                level="WARNING",
                enable_output=self._enable_logging,
            )
            raise
        except Exception as e:
            # If main lock acquisition fails, release the async lock if it was acquired
            if async_gate_acquired:
                self._async_lock.release()

            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}': {e}",
                level="ERROR",
                enable_output=True,
            )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        main_lock_released = False
        async_lock_released = False
        try:
            # Release main lock first
            if self._lock is not None:
                if self._is_async:
                    self._lock.release()
                else:
                    self._lock.release()

                direct_log(
                    f"== Lock == Process {self._pid}: Released lock {self._name} (async={self._is_async})",
                    level="INFO",
                    enable_output=self._enable_logging,
                )
                main_lock_released = True

            # Then release async lock if in multiprocess mode
            if not self._is_async and self._async_lock is not None:
                self._async_lock.release()
                direct_log(
                    f"== Lock == Process {self._pid}: Released async lock {self._name}",
                    level="DEBUG",
                    enable_output=self._enable_logging,
                )
                async_lock_released = True

        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}': {e}",
                level="ERROR",
                enable_output=True,
            )

            # If main lock release failed but async lock hasn't been attempted yet, try to release it
            if (
                not main_lock_released
                and not async_lock_released
                and not self._is_async
                and self._async_lock is not None
            ):
                try:
                    direct_log(
                        f"== Lock == Process {self._pid}: Attempting to release async lock after main lock failure",
                        level="DEBUG",
                        enable_output=self._enable_logging,
                    )
                    self._async_lock.release()
                    direct_log(
                        f"== Lock == Process {self._pid}: Successfully released async lock after main lock failure",
                        level="INFO",
                        enable_output=self._enable_logging,
                    )
                except Exception as inner_e:
                    direct_log(
                        f"== Lock == Process {self._pid}: Failed to release async lock after main lock failure: {inner_e}",
                        level="ERROR",
                        enable_output=True,
                    )

            raise

    def __enter__(self) -> "UnifiedLock[T]":
        """For backward compatibility"""
        try:
            if self._is_async or self._mp_is_lease:
                raise RuntimeError("Use 'async with' for shared_storage lock")

            # Acquire the main lock
            # Note: self._lock should never be None here as the check has been moved
            # to get_internal_lock() and get_data_init_lock() functions
            direct_log(
                f"== Lock == Process {self._pid}: Acquiring lock {self._name} (sync)",
                level="DEBUG",
                enable_output=self._enable_logging,
            )
            self._lock.acquire()
            direct_log(
                f"== Lock == Process {self._pid}: Acquired lock {self._name} (sync)",
                level="INFO",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=True,
            )
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For backward compatibility"""
        try:
            if self._is_async:
                raise RuntimeError("Use 'async with' for shared_storage lock")
            direct_log(
                f"== Lock == Process {self._pid}: Releasing lock '{self._name}' (sync)",
                level="DEBUG",
                enable_output=self._enable_logging,
            )
            self._lock.release()
            direct_log(
                f"== Lock == Process {self._pid}: Released lock {self._name} (sync)",
                level="INFO",
                enable_output=self._enable_logging,
            )
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=True,
            )
            raise

    def locked(self) -> bool:
        if self._is_async:
            return self._lock.locked()
        else:
            return self._lock.locked()


def _get_combined_key(factory_name: str, key: str) -> str:
    """Return the combined key for the factory and key."""
    return f"{factory_name}:{key}"


def _perform_lock_cleanup(
    lock_type: str,
    cleanup_data: Dict[str, float],
    lock_registry: Optional[Dict[str, Any]],
    lock_count: Optional[Dict[str, int]],
    earliest_cleanup_time: Optional[float],
    last_cleanup_time: Optional[float],
    current_time: float,
    threshold_check: bool = True,
) -> tuple[int, Optional[float], Optional[float]]:
    """
    Generic lock cleanup function to unify cleanup logic for both multiprocess and async locks.

    Args:
        lock_type: Lock type identifier ("mp" or "async")
        cleanup_data: Cleanup data dictionary
        lock_registry: Lock registry dictionary (can be None for async locks)
        lock_count: Lock count dictionary (can be None for async locks)
        earliest_cleanup_time: Earliest cleanup time
        last_cleanup_time: Last cleanup time
        current_time: Current time
        threshold_check: Whether to check threshold condition (default True, set to False in cleanup_expired_locks)

    Returns:
        tuple: (cleaned_count, new_earliest_time, new_last_cleanup_time)
    """
    if len(cleanup_data) == 0:
        return 0, earliest_cleanup_time, last_cleanup_time

    # If threshold check is needed and threshold not reached, return directly
    if threshold_check and len(cleanup_data) < CLEANUP_THRESHOLD:
        return 0, earliest_cleanup_time, last_cleanup_time

    # Time rollback detection
    if last_cleanup_time is not None and current_time < last_cleanup_time:
        direct_log(
            f"== {lock_type} Lock == Time rollback detected, resetting cleanup time",
            level="WARNING",
            enable_output=False,
        )
        last_cleanup_time = None

    # Check cleanup conditions
    has_expired_locks = (
        earliest_cleanup_time is not None
        and current_time - earliest_cleanup_time > CLEANUP_KEYED_LOCKS_AFTER_SECONDS
    )

    interval_satisfied = (
        last_cleanup_time is None
        or current_time - last_cleanup_time > MIN_CLEANUP_INTERVAL_SECONDS
    )

    if not (has_expired_locks and interval_satisfied):
        return 0, earliest_cleanup_time, last_cleanup_time

    try:
        cleaned_count = 0
        new_earliest_time = None

        # Calculate total count before cleanup
        total_cleanup_len = len(cleanup_data)

        # Perform cleanup operation
        for cleanup_key, cleanup_time in list(cleanup_data.items()):
            if current_time - cleanup_time > CLEANUP_KEYED_LOCKS_AFTER_SECONDS:
                # Remove from cleanup data
                cleanup_data.pop(cleanup_key, None)

                # Remove from lock registry if exists
                if lock_registry is not None:
                    lock_registry.pop(cleanup_key, None)
                if lock_count is not None:
                    lock_count.pop(cleanup_key, None)

                cleaned_count += 1
            else:
                # Track the earliest time among remaining locks
                if new_earliest_time is None or cleanup_time < new_earliest_time:
                    new_earliest_time = cleanup_time

        # Update state only after successful cleanup
        if cleaned_count > 0:
            new_last_cleanup_time = current_time

            # Log cleanup results
            next_cleanup_in = max(
                (new_earliest_time + CLEANUP_KEYED_LOCKS_AFTER_SECONDS - current_time)
                if new_earliest_time
                else float("inf"),
                MIN_CLEANUP_INTERVAL_SECONDS,
            )

            if lock_type == "async":
                direct_log(
                    f"== {lock_type} Lock == Cleaned up {cleaned_count}/{total_cleanup_len} expired {lock_type} locks, "
                    f"next cleanup in {next_cleanup_in:.1f}s",
                    enable_output=False,
                    level="INFO",
                )
            else:
                direct_log(
                    f"== {lock_type} Lock == Cleaned up {cleaned_count}/{total_cleanup_len} expired locks, "
                    f"next cleanup in {next_cleanup_in:.1f}s",
                    enable_output=False,
                    level="INFO",
                )

            return cleaned_count, new_earliest_time, new_last_cleanup_time
        else:
            return 0, earliest_cleanup_time, last_cleanup_time

    except Exception as e:
        direct_log(
            f"== {lock_type} Lock == Cleanup failed: {e}",
            level="ERROR",
            enable_output=True,
        )
        return 0, earliest_cleanup_time, last_cleanup_time


# ============================================================================
# Process identity and liveness helpers
# ============================================================================

_MY_START_ID_CACHE: Optional[str] = None
_MY_START_ID_COMPUTED: bool = False


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness probe; errs on the side of 'alive'."""
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError and friends: the process exists (or we cannot
        # tell) — treat as alive so we never reclaim a live owner's lease.
        return True
    return True


def _read_proc_starttime(pid: int) -> Optional[str]:
    """Return a stable per-process start token (field 22 of ``/proc/<pid>/stat``)
    used to detect PID reuse, or ``None`` when unavailable (non-Linux, the
    process is gone, or the stat file is unreadable).

    A live PID whose start time differs from a previously recorded token is a
    DIFFERENT process that reused the PID. ``None`` means "cannot tell" — callers
    must treat that as *not reused* so a live owner is never reclaimed.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except (FileNotFoundError, ProcessLookupError):
        return None  # process is gone (liveness is decided by _pid_alive)
    except OSError:
        return None  # unreadable → unknown, make no reuse claim
    try:
        # comm (field 2) is wrapped in parentheses and may itself contain spaces
        # or ')' — everything after the LAST ')' is the space-separated tail
        # starting at field 3 (state). starttime is field 22 → tail index 19.
        rparen = data.rindex(b")")
        tail = data[rparen + 1 :].split()
        return tail[19].decode("ascii")
    except (ValueError, IndexError):
        return None


def _my_start_id() -> Optional[str]:
    """This process's start token (see :func:`_read_proc_starttime`), computed
    once and cached. ``None`` on non-Linux / when ``/proc`` is unavailable, in
    which case PID-reuse detection is disabled (dead-only reclaim still works via
    :func:`_pid_alive`)."""
    global _MY_START_ID_CACHE, _MY_START_ID_COMPUTED
    if not _MY_START_ID_COMPUTED:
        _MY_START_ID_CACHE = _read_proc_starttime(os.getpid())
        _MY_START_ID_COMPUTED = True
    return _MY_START_ID_CACHE


def _process_alive(pid: Optional[int], start_id: Optional[str]) -> bool:
    """Dead-only liveness for lock / reservation owners.

    Returns ``False`` ONLY when the owner is *confirmed* dead: the PID is gone,
    or the PID is alive but its start time differs from ``start_id`` (PID reuse =
    a different process now holds that PID). Every uncertainty — no recorded
    identity, no permission to probe, non-Linux, unreadable ``/proc`` — is
    treated as ALIVE, so a live (merely slow) owner is never reclaimed. Shared by
    the keyed-lock and pipeline-reservation dead-owner reclaim layers.
    """
    if pid is None:
        return True  # no owner identity recorded → cannot declare dead
    if pid == os.getpid():
        # Our own PID. Genuinely us only if the recorded start id matches ours:
        # a record carrying our PID but a DIFFERENT start id was written by a
        # dead predecessor whose PID the OS reused for us — that owner is dead,
        # so we must NOT treat the lease as "still alive (me)" or it would never
        # be reclaimed. If either start id is unknown (non-Linux / no /proc), we
        # cannot confirm reuse and conservatively report alive.
        mine = _my_start_id()
        if start_id is not None and mine is not None and start_id != mine:
            return False
        return True
    if not _pid_alive(pid):
        return False
    if start_id is not None:
        current = _read_proc_starttime(pid)
        if current is not None and current != start_id:
            return False  # PID reused by a different process
    return True


def _keyed_lease_backoff(attempt: int) -> float:
    """Exponential backoff with jitter for the keyed-lock lease poll.

    The per-process async gate already caps concurrent pollers to one per process
    per key, so this only spaces out *cross-process* contention; jitter avoids a
    thundering herd when a lease is released."""
    base = min(_KEYED_LEASE_POLL_MAX, _KEYED_LEASE_POLL_BASE * (2 ** min(attempt, 5)))
    return base * (0.5 + random.random() * 0.5)


class _KeyedLeaseLock:
    """Multiprocess keyed lock backed by a holder record, not a manager.Lock().

    The record IS the lock: acquiring writes
    ``_keyed_lock_holders[combined_key] = {owner_pid, process_start_id,
    lease_id}`` under ``_registry_guard`` (a single manager.dict write, atomic);
    releasing pops it iff we still own ``lease_id``. A record whose owner process
    is *confirmed dead* (:func:`_process_alive` — PID gone or start-id mismatch)
    is reclaimed lazily on the next acquire, so a SIGKILLed holder can no longer
    deadlock every other worker. Dead-only: a live (merely slow) owner is never
    preempted, so no fencing token is needed.

    Per-process, per-acquire — each instance holds its own ``lease_id``; the only
    shared state is ``_keyed_lock_holders``. The liveness probe runs OUTSIDE the
    guard so ``os.kill`` / ``/proc`` reads never widen the (otherwise µs-scale)
    guard window.
    """

    __slots__ = ("_combined_key", "_lease_id")

    def __init__(self, combined_key: str) -> None:
        self._combined_key = combined_key
        self._lease_id: Optional[str] = None

    async def acquire(self) -> None:
        attempt = 0
        while True:
            with _registry_guard:
                holder = _keyed_lock_holders.get(self._combined_key)
                if holder is None:
                    lease_id = uuid.uuid4().hex
                    _keyed_lock_holders[self._combined_key] = {
                        "owner_pid": os.getpid(),
                        "process_start_id": _my_start_id(),
                        "lease_id": lease_id,
                    }
                    self._lease_id = lease_id
                    return
                snap = dict(holder)
            # Probe liveness outside the guard (os.kill / /proc can be slow).
            if not _process_alive(snap.get("owner_pid"), snap.get("process_start_id")):
                with _registry_guard:
                    cur = _keyed_lock_holders.get(self._combined_key)
                    # Recheck the SAME lease_id so a holder that changed between
                    # the probe and the reclaim is never clobbered.
                    if cur is not None and cur.get("lease_id") == snap.get("lease_id"):
                        _keyed_lock_holders.pop(self._combined_key, None)
                continue  # retry the claim immediately
            # Live owner → back off (this is the only cancellation point; no
            # lease is held here, so a cancel simply abandons the wait).
            await asyncio.sleep(_keyed_lease_backoff(attempt))
            attempt += 1

    def release(self) -> None:
        lease_id = self._lease_id
        if lease_id is None:
            return
        self._lease_id = None
        with _registry_guard:
            holder = _keyed_lock_holders.get(self._combined_key)
            if holder is not None and holder.get("lease_id") == lease_id:
                _keyed_lock_holders.pop(self._combined_key, None)


def _get_or_create_shared_raw_mp_lock(
    factory_name: str, key: str
) -> Optional["_KeyedLeaseLock"]:
    """Return a fresh holder-lease lock for a keyed lock (multiprocess only).

    Refcount + idle-cleanup bookkeeping is preserved (``_lock_registry`` now holds
    a presence sentinel rather than a manager.Lock); the actual mutual exclusion
    lives in ``_keyed_lock_holders`` via :class:`_KeyedLeaseLock`. A FRESH lease
    object is returned per call — each acquisition owns its own ``lease_id``.
    """
    if not _is_multiprocess:
        return None

    with _registry_guard:
        combined_key = _get_combined_key(factory_name, key)
        raw = _lock_registry.get(combined_key)
        count = _lock_registry_count.get(combined_key)
        if raw is None:
            _lock_registry[combined_key] = True  # presence sentinel (bookkeeping)
            count = 0
        else:
            if count is None:
                raise RuntimeError(
                    f"Shared-Data lock registry for {factory_name} is corrupted for key {key}"
                )
            if (
                count == 0 and combined_key in _lock_cleanup_data
            ):  # Reusing an key waiting for cleanup, remove it from cleanup list
                _lock_cleanup_data.pop(combined_key)
        count += 1
        _lock_registry_count[combined_key] = count
        return _KeyedLeaseLock(combined_key)


def _release_shared_raw_mp_lock(factory_name: str, key: str):
    """Release the *singleton* manager.Lock() proxy for *key*."""
    if not _is_multiprocess:
        return

    global _earliest_mp_cleanup_time, _last_mp_cleanup_time

    with _registry_guard:
        combined_key = _get_combined_key(factory_name, key)
        raw = _lock_registry.get(combined_key)
        count = _lock_registry_count.get(combined_key)
        if raw is None and count is None:
            return
        elif raw is None or count is None:
            raise RuntimeError(
                f"Shared-Data lock registry for {factory_name} is corrupted for key {key}"
            )

        count -= 1
        if count < 0:
            raise RuntimeError(
                f"Attempting to release lock for {key} more times than it was acquired"
            )

        _lock_registry_count[combined_key] = count

        current_time = time.time()
        if count == 0:
            _lock_cleanup_data[combined_key] = current_time

            # Update earliest multiprocess cleanup time (only when earlier)
            if (
                _earliest_mp_cleanup_time is None
                or current_time < _earliest_mp_cleanup_time
            ):
                _earliest_mp_cleanup_time = current_time

        # Use generic cleanup function
        cleaned_count, new_earliest_time, new_last_cleanup_time = _perform_lock_cleanup(
            lock_type="mp",
            cleanup_data=_lock_cleanup_data,
            lock_registry=_lock_registry,
            lock_count=_lock_registry_count,
            earliest_cleanup_time=_earliest_mp_cleanup_time,
            last_cleanup_time=_last_mp_cleanup_time,
            current_time=current_time,
            threshold_check=True,
        )

        # Update global state if cleanup was performed
        if cleaned_count > 0:
            _earliest_mp_cleanup_time = new_earliest_time
            _last_mp_cleanup_time = new_last_cleanup_time


class KeyedUnifiedLock:
    """
    Manager for unified keyed locks, supporting both single and multi-process

    • Keeps only a table of async keyed locks locally
    • Fetches the multi-process keyed lock on every acquire
    • Builds a fresh `UnifiedLock` each time, so `enable_logging`
      (or future options) can vary per call.
    • Supports dynamic namespaces specified at lock usage time
    """

    def __init__(self, *, default_enable_logging: bool = True) -> None:
        self._default_enable_logging = default_enable_logging
        self._async_lock: Dict[str, asyncio.Lock] = {}  # local keyed locks
        self._async_lock_count: Dict[
            str, int
        ] = {}  # local keyed locks referenced count
        self._async_lock_cleanup_data: Dict[
            str, time.time
        ] = {}  # local keyed locks timeout
        self._mp_locks: Dict[
            str, mp.synchronize.Lock
        ] = {}  # multi-process lock proxies
        self._earliest_async_cleanup_time: Optional[float] = (
            None  # track earliest async cleanup time
        )
        self._last_async_cleanup_time: Optional[float] = (
            None  # track last async cleanup time for minimum interval
        )

    def __call__(
        self, namespace: str, keys: list[str], *, enable_logging: Optional[bool] = None
    ):
        """
        Ergonomic helper so you can write:

            async with storage_keyed_lock("namespace", ["key1", "key2"]):
                ...
        """
        if enable_logging is None:
            enable_logging = self._default_enable_logging
        return _KeyedLockContext(
            self,
            namespace=namespace,
            keys=keys,
            enable_logging=enable_logging,
        )

    def _get_or_create_async_lock(self, combined_key: str) -> asyncio.Lock:
        async_lock = self._async_lock.get(combined_key)
        count = self._async_lock_count.get(combined_key, 0)
        if async_lock is None:
            async_lock = asyncio.Lock()
            self._async_lock[combined_key] = async_lock
        elif count == 0 and combined_key in self._async_lock_cleanup_data:
            self._async_lock_cleanup_data.pop(combined_key)
        count += 1
        self._async_lock_count[combined_key] = count
        return async_lock

    def _release_async_lock(self, combined_key: str):
        count = self._async_lock_count.get(combined_key, 0)
        count -= 1

        current_time = time.time()
        if count == 0:
            self._async_lock_cleanup_data[combined_key] = current_time

            # Update earliest async cleanup time (only when earlier)
            if (
                self._earliest_async_cleanup_time is None
                or current_time < self._earliest_async_cleanup_time
            ):
                self._earliest_async_cleanup_time = current_time
        self._async_lock_count[combined_key] = count

        # Use generic cleanup function
        cleaned_count, new_earliest_time, new_last_cleanup_time = _perform_lock_cleanup(
            lock_type="async",
            cleanup_data=self._async_lock_cleanup_data,
            lock_registry=self._async_lock,
            lock_count=self._async_lock_count,
            earliest_cleanup_time=self._earliest_async_cleanup_time,
            last_cleanup_time=self._last_async_cleanup_time,
            current_time=current_time,
            threshold_check=True,
        )

        # Update instance state if cleanup was performed
        if cleaned_count > 0:
            self._earliest_async_cleanup_time = new_earliest_time
            self._last_async_cleanup_time = new_last_cleanup_time

    def _get_lock_for_key(
        self, namespace: str, key: str, enable_logging: bool = False
    ) -> UnifiedLock:
        # 1. Create combined key for this namespace:key combination
        combined_key = _get_combined_key(namespace, key)

        # 2. get (or create) the per‑process async gate for this combined key
        # Is synchronous, so no need to acquire a lock
        async_lock = self._get_or_create_async_lock(combined_key)

        # 3. fetch the shared raw lock
        raw_lock = _get_or_create_shared_raw_mp_lock(namespace, key)
        is_multiprocess = raw_lock is not None
        if not is_multiprocess:
            raw_lock = async_lock

        # 4. build a *fresh* UnifiedLock with the chosen logging flag
        if is_multiprocess:
            return UnifiedLock(
                lock=raw_lock,
                is_async=False,  # holder-lease acquire is driven explicitly
                name=combined_key,
                enable_logging=enable_logging,
                async_lock=async_lock,  # prevents event‑loop blocking
                mp_is_lease=True,  # raw_lock is a _KeyedLeaseLock (async poll)
            )
        else:
            return UnifiedLock(
                lock=raw_lock,
                is_async=True,
                name=combined_key,
                enable_logging=enable_logging,
                async_lock=None,  # No need for async lock in single process mode
            )

    def _release_lock_for_key(self, namespace: str, key: str):
        combined_key = _get_combined_key(namespace, key)
        self._release_async_lock(combined_key)
        _release_shared_raw_mp_lock(namespace, key)

    def cleanup_expired_locks(self) -> Dict[str, Any]:
        """
        Cleanup expired locks for both async and multiprocess locks following the same
        conditions as _release_shared_raw_mp_lock and _release_async_lock functions.

        Only performs cleanup when both has_expired_locks and interval_satisfied conditions are met
        to avoid too frequent cleanup operations.

        Since async and multiprocess locks work together, this method cleans up
        both types of expired locks and returns comprehensive statistics.

        Returns:
            Dict containing cleanup statistics and current status:
            {
                "process_id": 12345,
                "cleanup_performed": {
                    "mp_cleaned": 5,
                    "async_cleaned": 3
                },
                "current_status": {
                    "total_mp_locks": 10,
                    "pending_mp_cleanup": 2,
                    "total_async_locks": 8,
                    "pending_async_cleanup": 1
                }
            }
        """
        global _lock_registry, _lock_registry_count, _lock_cleanup_data
        global _registry_guard, _earliest_mp_cleanup_time, _last_mp_cleanup_time

        cleanup_stats = {"mp_cleaned": 0, "async_cleaned": 0}

        current_time = time.time()

        # 1. Cleanup multiprocess locks using generic function
        if (
            _is_multiprocess
            and _lock_registry is not None
            and _registry_guard is not None
        ):
            try:
                with _registry_guard:
                    if _lock_cleanup_data is not None:
                        # Use generic cleanup function without threshold check
                        cleaned_count, new_earliest_time, new_last_cleanup_time = (
                            _perform_lock_cleanup(
                                lock_type="mp",
                                cleanup_data=_lock_cleanup_data,
                                lock_registry=_lock_registry,
                                lock_count=_lock_registry_count,
                                earliest_cleanup_time=_earliest_mp_cleanup_time,
                                last_cleanup_time=_last_mp_cleanup_time,
                                current_time=current_time,
                                threshold_check=False,  # Force cleanup in cleanup_expired_locks
                            )
                        )

                        # Update global state if cleanup was performed
                        if cleaned_count > 0:
                            _earliest_mp_cleanup_time = new_earliest_time
                            _last_mp_cleanup_time = new_last_cleanup_time
                            cleanup_stats["mp_cleaned"] = cleaned_count

            except Exception as e:
                direct_log(
                    f"Error during multiprocess lock cleanup: {e}",
                    level="ERROR",
                    enable_output=True,
                )

        # 2. Cleanup async locks using generic function
        try:
            # Use generic cleanup function without threshold check
            cleaned_count, new_earliest_time, new_last_cleanup_time = (
                _perform_lock_cleanup(
                    lock_type="async",
                    cleanup_data=self._async_lock_cleanup_data,
                    lock_registry=self._async_lock,
                    lock_count=self._async_lock_count,
                    earliest_cleanup_time=self._earliest_async_cleanup_time,
                    last_cleanup_time=self._last_async_cleanup_time,
                    current_time=current_time,
                    threshold_check=False,  # Force cleanup in cleanup_expired_locks
                )
            )

            # Update instance state if cleanup was performed
            if cleaned_count > 0:
                self._earliest_async_cleanup_time = new_earliest_time
                self._last_async_cleanup_time = new_last_cleanup_time
                cleanup_stats["async_cleaned"] = cleaned_count

        except Exception as e:
            direct_log(
                f"Error during async lock cleanup: {e}",
                level="ERROR",
                enable_output=True,
            )

        # 3. Get current status after cleanup
        current_status = self.get_lock_status()

        return {
            "process_id": os.getpid(),
            "cleanup_performed": cleanup_stats,
            "current_status": current_status,
        }

    def get_lock_status(self) -> Dict[str, int]:
        """
        Get current status of both async and multiprocess locks.

        Returns comprehensive lock counts for both types of locks since
        they work together in the keyed lock system.

        Returns:
            Dict containing lock counts:
            {
                "total_mp_locks": 10,
                "pending_mp_cleanup": 2,
                "total_async_locks": 8,
                "pending_async_cleanup": 1
            }
        """
        global _lock_registry_count, _lock_cleanup_data, _registry_guard

        status = {
            "total_mp_locks": 0,
            "pending_mp_cleanup": 0,
            "total_async_locks": 0,
            "pending_async_cleanup": 0,
        }

        try:
            # Count multiprocess locks
            if _is_multiprocess and _lock_registry_count is not None:
                if _registry_guard is not None:
                    with _registry_guard:
                        status["total_mp_locks"] = len(_lock_registry_count)
                        if _lock_cleanup_data is not None:
                            status["pending_mp_cleanup"] = len(_lock_cleanup_data)

            # Count async locks
            status["total_async_locks"] = len(self._async_lock_count)
            status["pending_async_cleanup"] = len(self._async_lock_cleanup_data)

        except Exception as e:
            direct_log(
                f"Error getting keyed lock status: {e}",
                level="ERROR",
                enable_output=True,
            )

        return status


class _KeyedLockContext:
    def __init__(
        self,
        parent: KeyedUnifiedLock,
        namespace: str,
        keys: list[str],
        enable_logging: bool,
    ) -> None:
        self._parent = parent
        self._namespace = namespace

        # The sorting is critical to ensure proper lock and release order
        # to avoid deadlocks
        self._keys = sorted(keys)
        self._enable_logging = (
            enable_logging
            if enable_logging is not None
            else parent._default_enable_logging
        )
        self._ul: Optional[List[Dict[str, Any]]] = None  # set in __aenter__

    # ----- enter -----
    async def __aenter__(self):
        if self._ul is not None:
            raise RuntimeError("KeyedUnifiedLock already acquired in current context")

        self._ul = []

        try:
            # Acquire locks for all keys in the namespace
            for key in self._keys:
                lock = None
                entry = None

                try:
                    # 1. Get lock object (reference count is incremented here)
                    lock = self._parent._get_lock_for_key(
                        self._namespace, key, enable_logging=self._enable_logging
                    )

                    # 2. Immediately create and add entry to list (critical for rollback to work)
                    entry = {
                        "key": key,
                        "lock": lock,
                        "entered": False,
                        "debug_inc": False,
                        "ref_incremented": True,  # Mark that reference count has been incremented
                    }
                    self._ul.append(
                        entry
                    )  # Add immediately after _get_lock_for_key for rollback to work

                    # 3. Try to acquire the lock
                    # Use try-finally to ensure state is updated atomically
                    lock_acquired = False
                    try:
                        await lock.__aenter__()
                        lock_acquired = True  # Lock successfully acquired
                    finally:
                        if lock_acquired:
                            entry["entered"] = True
                            inc_debug_n_locks_acquired()
                            entry["debug_inc"] = True

                except asyncio.CancelledError:
                    # Lock acquisition was cancelled
                    # The finally block above ensures entry["entered"] is correct
                    direct_log(
                        f"Lock acquisition cancelled for key {key}",
                        level="WARNING",
                        enable_output=self._enable_logging,
                    )
                    raise
                except Exception as e:
                    # Other exceptions, log and re-raise
                    direct_log(
                        f"Lock acquisition failed for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )
                    raise

            return self

        except BaseException:
            # Critical: if any exception occurs (including CancelledError) during lock acquisition,
            # we must rollback all already acquired locks to prevent lock leaks
            # Use shield to ensure rollback completes
            await asyncio.shield(self._rollback_acquired_locks())
            raise

    async def _rollback_acquired_locks(self):
        """Rollback all acquired locks in case of exception during __aenter__"""
        if not self._ul:
            return

        async def rollback_single_entry(entry):
            """Rollback a single lock acquisition"""
            key = entry["key"]
            lock = entry["lock"]
            debug_inc = entry["debug_inc"]
            entered = entry["entered"]
            ref_incremented = entry.get(
                "ref_incremented", True
            )  # Default to True for safety

            errors = []

            # 1. If lock was acquired, release it
            if entered:
                try:
                    await lock.__aexit__(None, None, None)
                except Exception as e:
                    errors.append(("lock_exit", e))
                    direct_log(
                        f"Lock rollback error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            # 2. Release reference count (if it was incremented)
            if ref_incremented:
                try:
                    self._parent._release_lock_for_key(self._namespace, key)
                except Exception as e:
                    errors.append(("ref_release", e))
                    direct_log(
                        f"Lock rollback reference release error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            # 3. Decrement debug counter
            if debug_inc:
                try:
                    dec_debug_n_locks_acquired()
                except Exception as e:
                    errors.append(("debug_dec", e))
                    direct_log(
                        f"Lock rollback counter decrementing error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            return errors

        # Release already acquired locks in reverse order
        for entry in reversed(self._ul):
            # Use shield to protect each lock's rollback
            try:
                await asyncio.shield(rollback_single_entry(entry))
            except Exception as e:
                # Log but continue rolling back other locks
                direct_log(
                    f"Lock rollback unexpected error for {entry['key']}: {e}",
                    level="ERROR",
                    enable_output=True,
                )

        self._ul = None

    # ----- exit -----
    async def __aexit__(self, exc_type, exc, tb):
        if self._ul is None:
            return

        # Snapshot the acquired-lock entries and clear ``self._ul`` BEFORE
        # starting the release task. The release runs on a shielded task that
        # may complete AFTER this coroutine is (re-)cancelled; if the closure
        # below read ``self._ul`` at that later point it would find ``None``
        # (cleared here) and skip releasing the underlying locks, deadlocking
        # every future acquirer. Iterating a stable snapshot keeps the deferred
        # release correct.
        entries = list(self._ul)
        self._ul = None

        async def release_all_locks():
            """Release all locks with comprehensive error handling, protected from cancellation"""

            async def release_single_entry(entry, exc_type, exc, tb):
                """Release a single lock with full protection"""
                key = entry["key"]
                lock = entry["lock"]
                debug_inc = entry["debug_inc"]
                entered = entry["entered"]

                errors = []

                # 1. Release the lock
                if entered:
                    try:
                        await lock.__aexit__(exc_type, exc, tb)
                    except Exception as e:
                        errors.append(("lock_exit", e))
                        direct_log(
                            f"Lock release error for key {key}: {e}",
                            level="ERROR",
                            enable_output=True,
                        )

                # 2. Release reference count
                try:
                    self._parent._release_lock_for_key(self._namespace, key)
                except Exception as e:
                    errors.append(("ref_release", e))
                    direct_log(
                        f"Lock release reference error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

                # 3. Decrement debug counter
                if debug_inc:
                    try:
                        dec_debug_n_locks_acquired()
                    except Exception as e:
                        errors.append(("debug_dec", e))
                        direct_log(
                            f"Lock release counter decrementing error for key {key}: {e}",
                            level="ERROR",
                            enable_output=True,
                        )

                return errors

            all_errors = []

            # Release locks in reverse order
            # This entire loop is protected by the outer shield.
            # Iterate the stable snapshot, not self._ul (already cleared above).
            for entry in reversed(entries):
                try:
                    errors = await release_single_entry(entry, exc_type, exc, tb)
                    for error_type, error in errors:
                        all_errors.append((entry["key"], error_type, error))
                except Exception as e:
                    all_errors.append((entry["key"], "unexpected", e))
                    direct_log(
                        f"Lock release unexpected error for {entry['key']}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            return all_errors

        # CRITICAL: run the release on a FIXED task and wait for THAT SAME task
        # to finish, resisting (repeated) cancellation. A single
        # ``await asyncio.shield(release_all_locks())`` would, on re-cancellation,
        # return before the shielded release actually completes, leaving the
        # underlying keyed lock held forever. The loop below only returns once the
        # release task is done, so __aexit__ never returns while a lock is still
        # held. Any cancellation observed while waiting is deferred and re-raised
        # only after the release has completed.
        release_task = asyncio.ensure_future(release_all_locks())
        pending_cancel = None
        while not release_task.done():
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError as exc:
                if release_task.cancelled():
                    # The release task itself was directly cancelled — it cannot
                    # have finished releasing; do not pretend it did.
                    raise
                # External cancellation of THIS coroutine: record it (even if the
                # task just became done, so it is never swallowed) and keep waiting.
                pending_cancel = pending_cancel or exc

        all_errors = []
        if not release_task.cancelled():
            task_exc = release_task.exception()
            if task_exc is not None:
                direct_log(
                    f"Critical error during __aexit__ cleanup: {task_exc}",
                    level="ERROR",
                    enable_output=True,
                )
            else:
                all_errors = release_task.result()

        # Locks are now definitively released; propagate a deferred cancellation
        # before anything else.
        if pending_cancel is not None:
            raise pending_cancel

        # If there were release errors and no other exception, raise the first release error
        if all_errors and exc_type is None:
            raise all_errors[0][2]  # (key, error_type, error)


def get_internal_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified storage lock for data consistency"""
    if _internal_lock is None:
        raise RuntimeError(
            "Shared data not initialized. Call initialize_share_data() before using locks!"
        )
    async_lock = _async_locks.get("internal_lock") if _is_multiprocess else None
    return UnifiedLock(
        lock=_internal_lock,
        is_async=not _is_multiprocess,
        name="internal_lock",
        enable_logging=enable_logging,
        async_lock=async_lock,
    )


# Workspace based storage_lock is implemented by get_storage_keyed_lock instead.
# Workspace based pipeline_status_lock is implemented by get_storage_keyed_lock instead.
# No need to implement graph_db_lock:
#    data integrity is ensured by entity level keyed-lock and allowing only one process to hold pipeline at a time.


def get_storage_keyed_lock(
    keys: str | list[str], namespace: str = "default", enable_logging: bool = False
) -> _KeyedLockContext:
    """Return unified storage keyed lock for ensuring atomic operations across different namespaces"""
    global _storage_keyed_lock
    if _storage_keyed_lock is None:
        raise RuntimeError("Shared-Data is not initialized")
    if isinstance(keys, str):
        keys = [keys]
    return _storage_keyed_lock(namespace, keys, enable_logging=enable_logging)


def get_data_init_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified data initialization lock for ensuring atomic data initialization"""
    if _data_init_lock is None:
        raise RuntimeError(
            "Shared data not initialized. Call initialize_share_data() before using locks!"
        )
    async_lock = _async_locks.get("data_init_lock") if _is_multiprocess else None
    return UnifiedLock(
        lock=_data_init_lock,
        is_async=not _is_multiprocess,
        name="data_init_lock",
        enable_logging=enable_logging,
        async_lock=async_lock,
    )


def cleanup_keyed_lock() -> Dict[str, Any]:
    """
    Force cleanup of expired keyed locks and return comprehensive status information.

    This function actively cleans up expired locks for both async and multiprocess locks,
    then returns detailed statistics about the cleanup operation and current lock status.

    Returns:
        Same as cleanup_expired_locks in KeyedUnifiedLock
    """
    global _storage_keyed_lock

    # Check if shared storage is initialized
    if not _initialized or _storage_keyed_lock is None:
        return {
            "process_id": os.getpid(),
            "cleanup_performed": {"mp_cleaned": 0, "async_cleaned": 0},
            "current_status": {
                "total_mp_locks": 0,
                "pending_mp_cleanup": 0,
                "total_async_locks": 0,
                "pending_async_cleanup": 0,
            },
        }

    return _storage_keyed_lock.cleanup_expired_locks()


def get_keyed_lock_status() -> Dict[str, Any]:
    """
    Get current status of keyed locks without performing cleanup.

    This function provides a read-only view of the current lock counts
    for both multiprocess and async locks, including pending cleanup counts.

    Returns:
        Same as get_lock_status in KeyedUnifiedLock
    """
    global _storage_keyed_lock

    # Check if shared storage is initialized
    if not _initialized or _storage_keyed_lock is None:
        return {
            "process_id": os.getpid(),
            "total_mp_locks": 0,
            "pending_mp_cleanup": 0,
            "total_async_locks": 0,
            "pending_async_cleanup": 0,
        }

    status = _storage_keyed_lock.get_lock_status()
    status["process_id"] = os.getpid()
    return status


def initialize_share_data(
    workers: int = 1,
    global_concurrency_limits: Optional[Mapping[str, int]] = None,
):
    """
    Initialize shared storage data for single or multi-process mode.

    When used with Gunicorn's preload feature, this function is called once in the
    master process before forking worker processes, allowing all workers to share
    the same initialized data.

    In single-process mode, this function is called in FASTAPI lifespan function.

    The function determines whether to use cross-process shared variables for data storage
    based on the number of workers. If workers=1, it uses thread locks and local dictionaries.
    If workers>1, it uses process locks and shared dictionaries managed by multiprocessing.Manager.

    Args:
        workers (int): Number of worker processes. If 1, single-process mode is used.
                      If > 1, multi-process mode with shared memory is used.
        global_concurrency_limits: Optional mapping of concurrency group name
                      (e.g. "llm:extract", "embedding", "rerank") to the
                      cross-worker global max concurrency for that group.
                      Read-only after initialization; later calls (which hit
                      the already-initialized guard) never overwrite it.
    """
    global \
        _manager, \
        _workers, \
        _is_multiprocess, \
        _lock_registry, \
        _lock_registry_count, \
        _lock_cleanup_data, \
        _registry_guard, \
        _keyed_lock_holders, \
        _internal_lock, \
        _data_init_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized, \
        _update_flags, \
        _async_locks, \
        _storage_keyed_lock, \
        _earliest_mp_cleanup_time, \
        _last_mp_cleanup_time, \
        _global_concurrency_limits, \
        _lease_ns_cache, \
        _queue_stats_ns_cache

    # Check if already initialized
    if _initialized:
        direct_log(
            f"Process {os.getpid()} Shared-Data already initialized (multiprocess={_is_multiprocess})"
        )
        return

    _workers = workers
    _global_concurrency_limits = (
        {
            str(group): int(limit)
            for group, limit in global_concurrency_limits.items()
            if limit is not None and int(limit) > 0
        }
        if global_concurrency_limits
        else {}
    )
    _lease_ns_cache = None
    _queue_stats_ns_cache = None
    if _global_concurrency_limits:
        direct_log(
            f"Process {os.getpid()} Global concurrency limits: {_global_concurrency_limits}",
            level="INFO",
        )

    if workers > 1:
        _is_multiprocess = True
        _manager = Manager()
        _lock_registry = _manager.dict()
        _lock_registry_count = _manager.dict()
        _lock_cleanup_data = _manager.dict()
        _registry_guard = _manager.RLock()
        _keyed_lock_holders = _manager.dict()
        _internal_lock = _manager.Lock()
        _data_init_lock = _manager.Lock()
        _shared_dicts = _manager.dict()
        _init_flags = _manager.dict()
        _update_flags = _manager.dict()

        _storage_keyed_lock = KeyedUnifiedLock()

        # Initialize async locks for multiprocess mode
        _async_locks = {
            "internal_lock": asyncio.Lock(),
            "graph_db_lock": asyncio.Lock(),
            "data_init_lock": asyncio.Lock(),
        }

        direct_log(
            f"Process {os.getpid()} Shared-Data created for Multiple Process (workers={workers})"
        )
    else:
        _is_multiprocess = False
        _internal_lock = asyncio.Lock()
        _data_init_lock = asyncio.Lock()
        _keyed_lock_holders = None  # multiprocess-only; unused single-process
        _shared_dicts = {}
        _init_flags = {}
        _update_flags = {}
        _async_locks = None  # No need for async locks in single process mode

        _storage_keyed_lock = KeyedUnifiedLock()
        direct_log(f"Process {os.getpid()} Shared-Data created for Single Process")

    # Initialize multiprocess cleanup times
    _earliest_mp_cleanup_time = None
    _last_mp_cleanup_time = None

    # Mark as initialized
    _initialized = True


async def initialize_pipeline_status(workspace: str | None = None):
    """
    Initialize pipeline_status share data with default values.
    This function could be called before during FASTAPI lifespan for each worker.

    Args:
        workspace: Optional workspace identifier for pipeline_status of specific workspace.
                   If None or empty string, uses the default workspace set by
                   set_default_workspace().
    """
    pipeline_namespace = await get_namespace_data(
        "pipeline_status", first_init=True, workspace=workspace
    )

    async with get_internal_lock():
        # Check if already initialized by checking for required fields
        if "busy" in pipeline_namespace:
            return

        # Create a shared list object for history_messages
        history_messages = _manager.list() if _is_multiprocess else []
        pipeline_namespace.update(
            {
                "autoscanned": False,  # Auto-scan started
                "busy": False,  # Control concurrent processes
                # Destructive subset of ``busy``: clear / delete jobs that
                # DROP storages or remove input files.  Concurrent enqueue
                # would race against the drop and silently lose the
                # accepted document, so reservation and the enqueue
                # last-line guard reject when this is True.  ``busy`` on
                # its own (the processing loop) remains compatible with
                # concurrent enqueue via request_pending.
                "destructive_busy": False,
                "scanning": False,  # /documents/scan task running (whole lifecycle)
                # Exclusive subset of ``scanning``: only True during the
                # scan's *classification* phase, when run_scanning_process
                # is reading doc_status to classify files (PROCESSED →
                # archive, FAILED-without-full_docs → retry-as-new, etc.)
                # and possibly deleting stale stubs.  After classification
                # the scan transitions to its processing phase (which
                # behaves like any other busy processing run) and clears
                # this flag, allowing concurrent uploads to land in
                # doc_status while the scan-driven processing finishes.
                "scanning_exclusive": False,
                # Counter of upload/insert endpoints that have passed the
                # idle preflight but whose background enqueue has not yet
                # run.  Closes the preflight-to-background race: scan
                # refuses to start while this is > 0 so the bg task is
                # guaranteed to see scanning=False at enqueue time.
                "pending_enqueues": 0,
                "job_name": "-",  # Current job name (indexing files/indexing texts)
                "job_start": None,  # Job start time
                "docs": 0,  # Total number of documents to be indexed
                "batchs": 0,  # Number of batches for processing documents
                "cur_batch": 0,  # Current processing batch
                "request_pending": False,  # Flag for pending request for processing
                "latest_message": "",  # Latest message from pipeline processing
                "history_messages": history_messages,  # 使用共享列表对象
                # ---- reservation owner tokens (cancellation-safe release) ----
                # ``busy``/``scanning`` are held by exactly one task at a time.
                # The owner records which task holds the slot so a release can be
                # owner-checked (never clobber a new owner) and — once the
                # dead-process recovery layer lands — a dead owner can be
                # reclaimed. ``busy_owner`` covers busy + destructive_busy;
                # ``scanning_owner`` covers scanning + scanning_exclusive.
                # ``pending_enqueue_tokens`` is a {token: metadata} set whose
                # length is mirrored in ``pending_enqueues`` (concurrent enqueues
                # are permitted, so it is a set, not a single owner).
                "busy_owner": None,
                "scanning_owner": None,
                "pending_enqueue_tokens": {},
            }
        )

        final_namespace = get_final_namespace("pipeline_status", workspace)
        direct_log(
            f"Process {os.getpid()} Pipeline namespace '{final_namespace}' initialized"
        )


# ============================================================================
# Pipeline reservation primitives (cancellation-safe owner-token release)
# ============================================================================
#
# The pipeline serialises document processing, scans and destructive ops through
# single-holder ``busy``/``scanning`` reservations plus a concurrent
# ``pending_enqueue_tokens`` set. These helpers make acquiring and releasing a
# reservation safe under asyncio cancellation:
#
# * the owner is recorded together with the flags in a SINGLE ``status.update``
#   (one Manager RPC, applied atomically server-side), so a cancellation can
#   never leave a flag set with no matching owner;
# * release/finalize is owner-checked and runs to completion even under repeated
#   cancellation via ``run_to_completion``, so a slot is never wedged and a stale
#   task never clobbers a new owner.
#
# On top of cancellation safety, the dead-process recovery layer reclaims a
# reservation whose owning worker was SIGKILLed: owners are recorded as
# ``{token, pid, process_start_id, kind}`` and ``reconcile_dead_pipeline_reservations``
# (called under the lock, before conflict checks) clears a CONFIRMED-dead owner's
# slot — re-running processing/scan, fencing custom_chunks/delete/clear with
# ``recovery_required``. Gated to Linux multi-worker (single-process dies with
# its state; see ``_reservation_recovery_enabled``).


# pipeline_status fields that are internal bookkeeping for reservation ownership
# / dead-process recovery — never surfaced on the /pipeline_status response.
_INTERNAL_PIPELINE_STATUS_FIELDS = (
    "busy_owner",
    "scanning_owner",
    "pending_enqueue_tokens",
    "operation_record",
    "recovery_required",
)

# Owner ``kind`` values whose work is safely RE-RUNNABLE after a dead-owner
# reclaim (in-flight docs sit in doc_status and are reset to PENDING / retried).
# Every other kind (custom_chunks / delete / clear) may have half-committed and
# is fenced with ``recovery_required`` instead of being cleared for re-run.
_RERUNNABLE_RESERVATION_KINDS = frozenset({"processing", "scan"})


def _reservation_recovery_enabled() -> bool:
    """Dead-process reservation reclaim is Linux-multiworker only.

    Single-process Uvicorn dies with its pipeline_status (no cross-process
    orphan); Windows has no Gunicorn multi-worker. Exposed as a function so tests
    can force it on to exercise the reclaim logic off-Linux.
    """
    return bool(_is_multiprocess) and sys.platform.startswith("linux")


def pipeline_recovery_blocked_message(pipeline_status: Dict[str, Any]) -> str:
    """Human-readable refusal for a mutation attempted while ``recovery_required``
    is set (a worker died mid custom_chunks/delete/clear, which may have
    half-committed). Returns a generic message if the pipeline is not fenced."""
    rec = pipeline_status.get("recovery_required")
    if not isinstance(rec, dict):
        return "Pipeline is not fenced for recovery."
    op = rec.get("operation_record") or {}
    target = op.get("doc_id") or op.get("scope") or ""
    target = f" (target: {target})" if target else ""
    return (
        f"Pipeline is fenced pending recovery: a worker died mid "
        f"'{rec.get('kind')}'{target}, which may have left storage in a "
        "partially-committed state. All mutations are refused until the "
        "workspace is recovered or force-reset."
    )


def make_owner_record(token: str, kind: str) -> Dict[str, Any]:
    """Build a reservation owner record: the cancellation-safety token plus the
    process identity used to detect a dead owner (see :func:`_process_alive`) and
    the ``kind`` that decides recovery semantics."""
    return {
        "token": token,
        "pid": os.getpid(),
        "process_start_id": _my_start_id(),
        "kind": kind,
    }


def _recover_dead_reservation(
    pipeline_status: Dict[str, Any],
    owner_key: str,
    flags: tuple,
    rec: Dict[str, Any],
) -> None:
    """Reclaim a single-holder reservation whose owner is confirmed dead.

    processing / scan → clear flags + owner (the work is re-runnable). Everything
    else (custom_chunks / delete / clear) may have half-committed, so clear the
    flags + owner but raise ``recovery_required`` to fence the workspace against
    all further mutations until an explicit recovery / force-reset. All writes go
    in a SINGLE ``status.update`` so a crash mid-recovery cannot tear them apart.
    """
    updates: Dict[str, Any] = {owner_key: None}
    for flag in flags:
        updates[flag] = False
    if rec.get("kind") not in _RERUNNABLE_RESERVATION_KINDS:
        updates["recovery_required"] = {
            "kind": rec.get("kind"),
            "owner_key": owner_key,
            # snapshot of what the dead owner was doing (doc_id / scope), if any
            "operation_record": pipeline_status.get("operation_record"),
        }
    pipeline_status.update(updates)


def reconcile_dead_pipeline_reservations(pipeline_status: Dict[str, Any]) -> None:
    """Reclaim reservations whose owning process is confirmed dead.

    No-op unless Linux multi-worker (:func:`_reservation_recovery_enabled`).
    Called inside the ``pipeline_status_lock`` critical section, BEFORE conflict
    checks, so a dead owner's slot is lazily reclaimed at the next acquire. Uses
    a FIXED flag→owner mapping (not the generic reject_when set, which only knows
    the acquirer's own owner_key) and only reclaims a CONFIRMED-dead owner
    (:func:`_process_alive`); a live-but-slow owner is never preempted.
    """
    if not _reservation_recovery_enabled():
        return
    # busy_owner covers busy + destructive_busy; scanning_owner covers
    # scanning + scanning_exclusive.
    for owner_key, flags in (
        ("busy_owner", ("busy", "destructive_busy")),
        ("scanning_owner", ("scanning", "scanning_exclusive")),
    ):
        rec = pipeline_status.get(owner_key)
        if not isinstance(rec, dict):
            continue
        if not any(pipeline_status.get(flag) for flag in flags):
            continue
        if _process_alive(rec.get("pid"), rec.get("process_start_id")):
            continue
        _recover_dead_reservation(pipeline_status, owner_key, flags, rec)

    # pending_enqueues: {token: {pid, process_start_id}} — drop confirmed-dead
    # tokens (an enqueue is re-runnable: its doc sits in doc_status). Always
    # recalibrate the mirrored count, covering a crash between "dropped token"
    # and "updated count".
    tokens = dict(pipeline_status.get("pending_enqueue_tokens", {}))
    alive = {
        token: meta
        for token, meta in tokens.items()
        if _process_alive((meta or {}).get("pid"), (meta or {}).get("process_start_id"))
    }
    if len(alive) != len(tokens) or pipeline_status.get("pending_enqueues") != len(
        alive
    ):
        pipeline_status.update(
            {"pending_enqueue_tokens": alive, "pending_enqueues": len(alive)}
        )


async def run_to_completion(factory, *, max_restarts: int = 3):
    """Await ``factory()`` to completion even if THIS coroutine is (repeatedly)
    cancelled while waiting.

    Cancellations of the caller are deferred until the work finishes, then the
    first one is re-raised. If the work task is itself directly cancelled (e.g. a
    shutdown that cancels every task) it is restarted a bounded number of times —
    used only for idempotent releases, so the release still completes.
    """
    pending_cancel = None
    restarts = 0
    task = asyncio.ensure_future(factory())
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.cancelled():
                # The work task itself was cancelled (e.g. a shutdown that
                # cancels every task). Restart the idempotent work so the
                # release still completes; do NOT treat this as a caller cancel.
                if restarts >= max_restarts:
                    raise
                restarts += 1
                task = asyncio.ensure_future(factory())
                continue
            # This coroutine was cancelled while the work is still running:
            # record it (even if the task became done in the same step, so it is
            # never swallowed) and keep waiting for the work to complete.
            pending_cancel = pending_cancel or exc
    if pending_cancel is not None:
        raise pending_cancel
    return result


def _reservation_owner_token(record: Any) -> Any:
    """Extract the token from an owner record (a bare token or a dict)."""
    if isinstance(record, dict):
        return record.get("token")
    return record


async def acquire_reservation(
    pipeline_status: Dict[str, Any],
    pipeline_status_lock,
    *,
    owner_key: str,
    owner: Any,
    flags: Dict[str, Any],
    reject_when,
) -> tuple[bool, Optional[str]]:
    """Atomically take a single-holder reservation.

    ``reject_when`` is a sequence of ``(flag_key, reason)``: if any flag is set
    the reservation is refused and ``(False, reason)`` is returned. Otherwise the
    ``flags`` and ``pipeline_status[owner_key] = owner`` are written in a SINGLE
    ``status.update`` (atomic against a mid-write crash) and ``(True, None)`` is
    returned. ``owner`` may be a bare token or an owner record dict.

    The caller MUST have entered its ``try`` before calling this so a cancel at
    the lock exit still runs the ``finally`` that releases ``owner`` by token.
    """
    async with pipeline_status_lock:
        reconcile_dead_pipeline_reservations(pipeline_status)
        for flag_key, reason in reject_when:
            if pipeline_status.get(flag_key):
                return False, reason
        updates = dict(flags)
        updates[owner_key] = owner
        pipeline_status.update(updates)
    return True, None


async def acquire_enqueue_reservation(
    pipeline_status: Dict[str, Any],
    pipeline_status_lock,
    *,
    token: str,
    reject_when,
) -> tuple[bool, Optional[str]]:
    """Take one of the (concurrent) pending-enqueue reservations.

    ``pending_enqueue_tokens`` is a ``{token: metadata}`` set; several enqueues
    may hold slots at once. Adds ``token`` and mirrors the count into
    ``pending_enqueues`` in a single atomic update.
    """
    async with pipeline_status_lock:
        reconcile_dead_pipeline_reservations(pipeline_status)
        for flag_key, reason in reject_when:
            if pipeline_status.get(flag_key):
                return False, reason
        tokens = dict(pipeline_status.get("pending_enqueue_tokens", {}))
        tokens[token] = {"pid": os.getpid(), "process_start_id": _my_start_id()}
        pipeline_status.update(
            {"pending_enqueue_tokens": tokens, "pending_enqueues": len(tokens)}
        )
    return True, None


async def with_reservation_lock(
    pipeline_status: Dict[str, Any],
    pipeline_status_lock,
    *,
    owner_key: str,
    token: Any,
    action,
):
    """Run ``action(status)`` under ``pipeline_status_lock``, but only while we
    still own the reservation (the token stored at ``status[owner_key]`` equals
    ``token``); otherwise no-op and return ``None``.

    ``action`` is SYNCHRONOUS (no await) and must apply all correctness-critical
    mutations (owner, busy/scanning flags, request_pending, cancellation flags,
    operation_record, recovery state) via a SINGLE ``status.update`` so a crash
    cannot tear them apart. ``history_messages`` (a Manager list) must be mutated
    in place, not replaced. Runs to completion even under repeated cancellation
    (via ``run_to_completion``), so a release can never be interrupted into
    leaving the slot wedged.
    """

    async def _run():
        async with pipeline_status_lock:
            if _reservation_owner_token(pipeline_status.get(owner_key)) != token:
                return None
            return action(pipeline_status)

    return await run_to_completion(_run)


async def with_token_set_reservation_lock(
    pipeline_status: Dict[str, Any],
    pipeline_status_lock,
    *,
    tokens_key: str,
    token: str,
    action=None,
):
    """Release one token from a token-set reservation (e.g. pending enqueues).

    Under the lock: if ``token`` is in the set, remove it, rewrite the set and
    mirror the count in a single atomic update, then run the optional
    ``action(status)``. If the token is absent, no-op (idempotent — an endpoint
    and its background task may both release). Runs to completion under
    cancellation.
    """

    async def _run():
        async with pipeline_status_lock:
            tokens = dict(pipeline_status.get(tokens_key, {}))
            if token not in tokens:
                return None
            del tokens[token]
            pipeline_status.update(
                {tokens_key: tokens, "pending_enqueues": len(tokens)}
            )
            return action(pipeline_status) if action else None

    return await run_to_completion(_run)


async def release_owned_reservation(
    workspace: Optional[str],
    *,
    owner_key: str,
    token: Any,
    action,
):
    """Cancellation-safe, self-fetching form of :func:`with_reservation_lock`.

    Fetches ``pipeline_status`` + its lock AND runs the owner-checked release
    entirely inside ``run_to_completion``, so a cancellation delivered during the
    namespace fetch or the lock acquire/exit is retried/completed rather than
    leaking the reservation. Use this from release helpers that do not already
    hold the status (e.g. background-task releases running during shutdown),
    where the plain ``with_reservation_lock`` would leave the pre-fetch
    unprotected.

    No-op (returns ``None``) when ``pipeline_status`` was never initialised for
    ``workspace`` or a later holder owns the slot. ``action`` must be idempotent
    (it may re-run if the work task is directly cancelled and restarted).
    """

    async def _run():
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )
        except PipelineNotInitializedError:
            return None
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=workspace
        )
        async with pipeline_status_lock:
            if _reservation_owner_token(pipeline_status.get(owner_key)) != token:
                return None
            return action(pipeline_status)

    return await run_to_completion(_run)


async def release_token_set_reservation(
    workspace: Optional[str],
    *,
    tokens_key: str,
    token: str,
    action=None,
):
    """Cancellation-safe, self-fetching form of
    :func:`with_token_set_reservation_lock`.

    Fetches ``pipeline_status`` + its lock AND removes ``token`` from the
    ``tokens_key`` set entirely inside ``run_to_completion``. Idempotent (a no-op
    if the token is absent or the workspace is uninitialised), so an endpoint and
    its background task may both release, and a restart after a direct task
    cancellation is safe.
    """

    async def _run():
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )
        except PipelineNotInitializedError:
            return None
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=workspace
        )
        async with pipeline_status_lock:
            tokens = dict(pipeline_status.get(tokens_key, {}))
            if token not in tokens:
                return None
            del tokens[token]
            pipeline_status.update(
                {tokens_key: tokens, "pending_enqueues": len(tokens)}
            )
            return action(pipeline_status) if action else None

    return await run_to_completion(_run)


# ============================================================================
# Managed reservation-holding background tasks
# ============================================================================
#
# Scans / deletes / enqueues reserve a slot in the request handler, then run the
# actual work in the background. Starlette ``BackgroundTasks`` run only AFTER the
# response body is sent and are not tracked, so a request cancelled mid-send
# would drop the callback and strand the reservation. Instead we start a real
# ``asyncio`` task, track it for shutdown, and hand ownership over via a start
# barrier so a cancellation before takeover releases the slot instead of leaking
# it or letting the child run unreserved.


async def _join_resistant(task) -> Optional[asyncio.CancelledError]:
    """Join ``task`` resisting (repeated) cancellation.

    Returns a caller cancellation observed while waiting (or ``None``); NEVER
    propagates the task's own exception or cancellation, so a caller can always
    run its cleanup (e.g. a backstop release) after the join.
    """
    pending_cancel = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                break  # the task itself is done+cancelled
            pending_cancel = pending_cancel or exc
        except Exception:
            break  # task finished with an exception; retrieved below
    if not task.cancelled():
        task.exception()  # retrieve to avoid "never retrieved" warnings
    return pending_cancel


async def start_reserved_background_task(
    background_tasks: set, *, work, backstop_release
):
    """Start a background task that already holds a reservation and hand ownership
    to it safely.

    ``work(started: asyncio.Event)`` MUST call ``started.set()`` as the first
    statement inside its own ``try`` and release the reservation in its
    ``finally``. ``backstop_release()`` is an owner-checked, idempotent release
    used only if the child never takes over.

    Returns the running task once the child has taken over (``started`` set). If
    this coroutine is cancelled before takeover, or the child ends before
    signalling, the child is cancelled and joined, ``backstop_release`` runs
    (a no-op if the child already released), and the original cancellation — or a
    startup failure — is raised. The child therefore never runs unreserved and
    the reservation is never stranded.
    """
    started = asyncio.Event()
    task = asyncio.ensure_future(work(started))
    background_tasks.add(task)

    def _done(t):
        background_tasks.discard(t)
        if not t.cancelled():
            exc = t.exception()
            if exc is not None:
                direct_log(
                    f"Reserved background task failed: {exc}",
                    level="ERROR",
                    enable_output=True,
                )

    task.add_done_callback(_done)

    waiter = asyncio.ensure_future(started.wait())
    caller_cancel = None
    try:
        await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError as exc:
        caller_cancel = exc
    finally:
        if not waiter.done():
            waiter.cancel()

    if started.is_set() and caller_cancel is None:
        return task  # child took over; its finally owns the release

    # Not taken over (child ended before signalling) or caller cancelled: tear
    # down deterministically so the child never runs unreserved.
    task.cancel()
    join_cancel = await _join_resistant(task)
    await backstop_release()
    cancel = caller_cancel or join_cancel
    if cancel is not None:
        raise cancel
    raise RuntimeError("reserved background task failed to start")


async def drain_reserved_background_tasks(
    background_tasks: set,
) -> Optional[asyncio.CancelledError]:
    """Cancel and join all tracked reserved background tasks (used at shutdown).

    Resists repeated cancellation so every child's ``finally`` releases its
    reservation before shared state is torn down. Returns a deferred caller
    cancellation to re-raise AFTER the caller's own cleanup, or ``None``.
    """
    tasks = list(background_tasks)
    for task in tasks:
        task.cancel()
    pending_cancel = None
    for task in tasks:
        observed = await _join_resistant(task)
        pending_cancel = pending_cancel or observed
    return pending_cancel


async def get_update_flag(namespace: str, workspace: str | None = None):
    """
    Create a namespace's update flag for a workers.
    Returen the update flag to caller for referencing or reset.
    """
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in _update_flags:
            if _is_multiprocess and _manager is not None:
                _update_flags[final_namespace] = _manager.list()
            else:
                _update_flags[final_namespace] = []
            direct_log(
                f"Process {os.getpid()} initialized updated flags for namespace: [{final_namespace}]"
            )

        if _is_multiprocess and _manager is not None:
            new_update_flag = _manager.Value("b", False)
        else:
            # Create a simple mutable object to store boolean value for compatibility with mutiprocess
            class MutableBoolean:
                def __init__(self, initial_value=False):
                    self.value = initial_value

            new_update_flag = MutableBoolean(False)

        _update_flags[final_namespace].append(new_update_flag)
        return new_update_flag


async def set_all_update_flags(namespace: str, workspace: str | None = None):
    """Set all update flag of namespace indicating all workers need to reload data from files"""
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in _update_flags:
            raise ValueError(f"Namespace {final_namespace} not found in update flags")
        # Update flags for both modes
        for i in range(len(_update_flags[final_namespace])):
            _update_flags[final_namespace][i].value = True


async def clear_all_update_flags(namespace: str, workspace: str | None = None):
    """Clear all update flag of namespace indicating all workers need to reload data from files"""
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in _update_flags:
            raise ValueError(f"Namespace {final_namespace} not found in update flags")
        # Update flags for both modes
        for i in range(len(_update_flags[final_namespace])):
            _update_flags[final_namespace][i].value = False


async def get_all_update_flags_status(workspace: str | None = None) -> Dict[str, list]:
    """
    Get update flags status for all namespaces.

    Returns:
        Dict[str, list]: A dictionary mapping namespace names to lists of update flag statuses
    """
    if _update_flags is None:
        return {}

    if workspace is None:
        workspace = get_default_workspace()

    result = {}
    async with get_internal_lock():
        for namespace, flags in _update_flags.items():
            # Check if namespace has a workspace prefix (contains ':')
            if ":" in namespace:
                # Namespace has workspace prefix like "space1:pipeline_status"
                # Only include if workspace matches the prefix
                # Use rsplit to split from the right since workspace can contain colons
                namespace_split = namespace.rsplit(":", 1)
                if not workspace or namespace_split[0] != workspace:
                    continue
            else:
                # Namespace has no workspace prefix like "pipeline_status"
                # Only include if we're querying the default (empty) workspace
                if workspace:
                    continue

            worker_statuses = []
            for flag in flags:
                if _is_multiprocess:
                    worker_statuses.append(flag.value)
                else:
                    worker_statuses.append(flag)
            result[namespace] = worker_statuses

    return result


async def try_initialize_namespace(
    namespace: str, workspace: str | None = None
) -> bool:
    """
    Returns True if the current worker(process) gets initialization permission for loading data later.
    The worker does not get the permission is prohibited to load data from files.
    """
    global _init_flags, _manager

    if _init_flags is None:
        raise ValueError("Try to create nanmespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in _init_flags:
            _init_flags[final_namespace] = True
            direct_log(
                f"Process {os.getpid()} ready to initialize storage namespace: [{final_namespace}]"
            )
            return True
        direct_log(
            f"Process {os.getpid()} storage namespace already initialized: [{final_namespace}]"
        )

    return False


async def get_namespace_data(
    namespace: str, first_init: bool = False, workspace: str | None = None
) -> Dict[str, Any]:
    """get the shared data reference for specific namespace

    Args:
        namespace: The namespace to retrieve
        first_init: If True, allows pipeline_status namespace to create namespace if it doesn't exist.
                    Prevent getting pipeline_status namespace without initialize_pipeline_status().
                    This parameter is used internally by initialize_pipeline_status().
        workspace: Workspace identifier (may be empty string for global namespace)
    """
    if _shared_dicts is None:
        direct_log(
            f"Error: Try to getnanmespace before it is initialized, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Shared dictionaries not initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in _shared_dicts:
            # Special handling for pipeline_status namespace
            if (
                final_namespace.endswith(":pipeline_status")
                or final_namespace == "pipeline_status"
            ) and not first_init:
                # Check if pipeline_status should have been initialized but wasn't
                # This helps users to call initialize_pipeline_status() before get_namespace_data()
                raise PipelineNotInitializedError(final_namespace)

            # For other namespaces or when allow_create=True, create them dynamically
            if _is_multiprocess and _manager is not None:
                _shared_dicts[final_namespace] = _manager.dict()
            else:
                _shared_dicts[final_namespace] = {}

    return _shared_dicts[final_namespace]


class NamespaceLock:
    """
    Reusable namespace lock wrapper that creates a fresh context on each use.

    This class solves the lock re-entrance and concurrent coroutine issues by using
    contextvars.ContextVar to provide per-coroutine storage. Each coroutine gets its
    own independent lock context, preventing state interference between concurrent
    coroutines using the same NamespaceLock instance.

    Example:
        lock = NamespaceLock("my_namespace", "workspace1")

        # Can be used multiple times safely
        async with lock:
            await do_something()

        # Can even be used concurrently without deadlock
        await asyncio.gather(
            coroutine_1(lock),  # Each gets its own context
            coroutine_2(lock)   # No state interference
        )
    """

    def __init__(
        self, namespace: str, workspace: str | None = None, enable_logging: bool = False
    ):
        self._namespace = namespace
        self._workspace = workspace
        self._enable_logging = enable_logging
        # Use ContextVar to provide per-coroutine storage for lock context
        # This ensures each coroutine has its own independent context
        self._ctx_var: ContextVar[Optional[_KeyedLockContext]] = ContextVar(
            "lock_ctx", default=None
        )

    async def __aenter__(self):
        """Create a fresh context each time we enter"""
        # Check if this coroutine already has an active lock context
        if self._ctx_var.get() is not None:
            raise RuntimeError(
                "NamespaceLock already acquired in current coroutine context"
            )

        final_namespace = get_final_namespace(self._namespace, self._workspace)
        ctx = get_storage_keyed_lock(
            ["default_key"],
            namespace=final_namespace,
            enable_logging=self._enable_logging,
        )

        # Acquire the lock first, then store context only after successful acquisition
        # This prevents the ContextVar from being set if acquisition fails (e.g., due to cancellation),
        # which would permanently brick the lock
        result = await ctx.__aenter__()
        self._ctx_var.set(ctx)
        return result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the current context and clean up"""
        # Retrieve this coroutine's context
        ctx = self._ctx_var.get()
        if ctx is None:
            raise RuntimeError("NamespaceLock exited without being entered")

        # Clear the ContextVar in a finally so a cancellation delivered inside
        # ctx.__aexit__ (which awaits an asyncio.shield) does not leave a stale
        # context behind. Otherwise the SAME coroutine re-entering this lock in a
        # finally would hit "already acquired" (RuntimeError, not CancelledError),
        # and any cancel-safe release built on top could never run.
        try:
            return await ctx.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._ctx_var.set(None)


def get_namespace_lock(
    namespace: str, workspace: str | None = None, enable_logging: bool = False
) -> NamespaceLock:
    """Get a reusable namespace lock wrapper.

    This function returns a NamespaceLock instance that can be used multiple times
    safely, even in concurrent scenarios. Each use creates a fresh lock context
    internally, preventing lock re-entrance errors.

    Args:
        namespace: The namespace to get the lock for.
        workspace: Workspace identifier (may be empty string for global namespace)
        enable_logging: Whether to enable lock operation logging

    Returns:
        NamespaceLock: A reusable lock wrapper that can be used with 'async with'

    Example:
        lock = get_namespace_lock("pipeline_status", workspace="space1")

        # Can be used multiple times
        async with lock:
            await do_something()

        async with lock:
            await do_something_else()
    """
    return NamespaceLock(namespace, workspace, enable_logging)


def finalize_share_data():
    """
    Release shared resources and clean up.

    This function should be called when the application is shutting down
    to properly release shared resources and avoid memory leaks.

    In multi-process mode, it shuts down the Manager and releases all shared objects.
    In single-process mode, it simply resets the global variables.
    """
    global \
        _manager, \
        _is_multiprocess, \
        _internal_lock, \
        _data_init_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized, \
        _update_flags, \
        _async_locks, \
        _default_workspace, \
        _global_concurrency_limits, \
        _lease_ns_cache, \
        _queue_stats_ns_cache

    # Check if already initialized
    if not _initialized:
        direct_log(
            f"Process {os.getpid()} storage data not initialized, nothing to finalize"
        )
        return

    direct_log(
        f"Process {os.getpid()} finalizing storage data (multiprocess={_is_multiprocess})"
    )

    # In multi-process mode, shut down the Manager
    if _is_multiprocess and _manager is not None:
        try:
            # Clear shared resources before shutting down Manager
            if _shared_dicts is not None:
                # Clear pipeline status history messages first if exists
                try:
                    pipeline_status = _shared_dicts.get("pipeline_status", {})
                    if "history_messages" in pipeline_status:
                        pipeline_status["history_messages"].clear()
                except Exception:
                    pass  # Ignore any errors during history messages cleanup
                _shared_dicts.clear()
            if _init_flags is not None:
                _init_flags.clear()
            if _update_flags is not None:
                # Clear each namespace's update flags list and Value objects
                try:
                    for namespace in _update_flags:
                        flags_list = _update_flags[namespace]
                        if isinstance(flags_list, list):
                            # Clear Value objects in the list
                            for flag in flags_list:
                                if hasattr(
                                    flag, "value"
                                ):  # Check if it's a Value object
                                    flag.value = False
                            flags_list.clear()
                except Exception:
                    pass  # Ignore any errors during update flags cleanup
                _update_flags.clear()

            # Shut down the Manager - this will automatically clean up all shared resources
            _manager.shutdown()
            direct_log(f"Process {os.getpid()} Manager shutdown complete")
        except Exception as e:
            direct_log(
                f"Process {os.getpid()} Error shutting down Manager: {e}", level="ERROR"
            )

    # Reset global variables
    _manager = None
    _initialized = None
    _is_multiprocess = None
    _shared_dicts = None
    _init_flags = None
    _internal_lock = None
    _data_init_lock = None
    _update_flags = None
    _async_locks = None
    _default_workspace = None
    _global_concurrency_limits = None
    _lease_ns_cache = None
    _queue_stats_ns_cache = None

    direct_log(f"Process {os.getpid()} storage data finalization complete")


def set_default_workspace(workspace: str | None = None):
    """
    Set default workspace for namespace operations for backward compatibility.

    This allows get_namespace_data(),get_namespace_lock() or initialize_pipeline_status() to
    automatically use the correct workspace when called without workspace parameters,
    maintaining compatibility with legacy code that doesn't pass workspace explicitly.

    Args:
        workspace: Workspace identifier (may be empty string for global namespace)
    """
    global _default_workspace
    if workspace is None:
        workspace = ""
    _default_workspace = workspace
    direct_log(
        f"Default workspace set to: '{_default_workspace}' (empty means global)",
        level="DEBUG",
    )


def get_default_workspace() -> str:
    """
    Get default workspace for backward compatibility.

    Returns:
        The default workspace string. Empty string means global namespace. None means not set.
    """
    global _default_workspace
    return _default_workspace


def get_pipeline_status_lock(
    enable_logging: bool = False, workspace: str = None
) -> NamespaceLock:
    """Return unified storage lock for pipeline status data consistency.

    This function is for compatibility with legacy code only.
    """
    global _default_workspace
    actual_workspace = workspace if workspace else _default_workspace
    return get_namespace_lock(
        "pipeline_status", workspace=actual_workspace, enable_logging=enable_logging
    )


# ---------------------------------------------------------------------------
# Cross-worker global concurrency gate (lease + heartbeat semantics)
# ---------------------------------------------------------------------------
#
# Each group's whole gate state lives under a SINGLE key of the
# workspace-less "concurrency_leases" namespace::
#
#     ns[group] = {
#         "leases":  {lease_id: {"pid": int, "updated_at": float,
#                                ("suspect_since": float)}},
#         "waiters": {str(pid): {"pid": int, "wait_start": float,
#                                "last_poll": float}},
#     }
#
# Whole-value replacement only — in multiprocess mode the namespace is a
# Manager dict, so in-place mutation of a retrieved value would NOT persist
# across processes. The single-key layout is deliberate: every proxy access
# is one IPC round trip to the manager process, so an acquire attempt under
# the group's keyed lock costs exactly one read (plus at most one write)
# instead of scanning per-lease keys. Reaping, capacity counting and waiter
# ranking all run on the local copy.
#
# Self-healing: holders refresh ``updated_at`` from their 5s health-check
# heartbeat. A lease is reclaimed when its owner PID is dead (immediately)
# or when its heartbeat expired AND the suspect grace elapsed (protects
# live-but-momentarily-stalled owners from false reclamation). Long-running
# tasks are never reclaimed as long as their owner keeps renewing.
#
# Best-effort cap, not a strict provider-side invariant: the lease table is
# the admission source of truth, so the cap is exact while holders keep
# renewing. A slot is reclaimed only when its owner PID is gone or its
# heartbeat has expired beyond the suspect grace. That prevents permanent
# capacity leaks after kill -9 / OOM and similar external termination, but
# the provider may still be finishing the abandoned HTTP request until its
# own timeout/connection close. During that window, a newly admitted caller
# can overlap with the abandoned provider-side call. Long event-loop stalls
# have the same shape after heartbeat TTL + suspect grace, though normal long
# calls are protected by regular renewal. A reclaimed lease is never
# resurrected (renew_global_slots refuses to re-insert a popped lease), which
# keeps the internal gate self-healing.


def is_share_data_initialized() -> bool:
    """Return True once initialize_share_data() has run in this process."""
    return bool(_initialized)


def is_global_concurrency_limited(group: Optional[str]) -> bool:
    """Synchronous, cacheable check: does ``group`` have a global limit?

    Reads only the module-level read-only configuration — no IPC. Returns
    False when shared data is not initialized, no limits were configured,
    or the group has no positive limit.
    """
    if not group or not _global_concurrency_limits:
        return False
    limit = _global_concurrency_limits.get(group)
    return limit is not None and limit > 0


def get_global_concurrency_limit(group: Optional[str]) -> Optional[int]:
    """Return the configured global limit for ``group`` (None if unlimited)."""
    if not group or not _global_concurrency_limits:
        return None
    return _global_concurrency_limits.get(group)


async def _get_lease_namespace() -> Dict[str, Any]:
    global _lease_ns_cache
    if _lease_ns_cache is None:
        _lease_ns_cache = await get_namespace_data(
            _CONCURRENCY_LEASE_NAMESPACE, workspace=""
        )
    return _lease_ns_cache


async def _get_queue_stats_namespace() -> Dict[str, Any]:
    global _queue_stats_ns_cache
    if _queue_stats_ns_cache is None:
        _queue_stats_ns_cache = await get_namespace_data(
            _QUEUE_STATS_NAMESPACE, workspace=""
        )
    return _queue_stats_ns_cache


def _empty_gate_state() -> Dict[str, Any]:
    return {"leases": {}, "waiters": {}}


def _load_gate_state(ns: Dict[str, Any], group: str) -> Dict[str, Any]:
    """Return a local, independently mutable copy of a group's gate state.

    Exactly one proxy read. The nested dicts are copied so that mutating the
    result never aliases the stored value (a plain dict in single-process
    mode) — callers mutate the copy and write it back whole.
    """
    raw = ns.get(group)
    if raw is None:
        return _empty_gate_state()
    state = dict(raw)
    state["leases"] = {
        lease_id: dict(lease)
        for lease_id, lease in dict(state.get("leases") or {}).items()
    }
    state["waiters"] = {
        pid_key: dict(waiter)
        for pid_key, waiter in dict(state.get("waiters") or {}).items()
    }
    return state


def _reap_gate_state(state: Dict[str, Any], now: float) -> tuple[int, bool]:
    """Reclaim dead/expired leases on a local state copy (no IPC).

    Returns ``(live_lease_count, changed)``. Suspect handling: a lease whose
    heartbeat expired while its PID is still alive is first marked with
    ``suspect_since`` and reclaimed only after ``_suspect_grace`` elapses
    without a renewal; a renewal (fresh ``updated_at``) clears the suspect
    mark. Dead PIDs are reclaimed immediately. Suspect leases still count
    toward capacity so the global limit is never exceeded.

    Waiter records are reaped in the same pass: a process whose lease was
    just reclaimed (timed out / died), whose PID is dead, or whose record
    has not been refreshed within ``_waiter_stale_ttl`` must not keep
    occupying the longest-waiter seat — a ghost favored waiter would push
    every live waiter onto the deferred backoff and waste freed slots.
    """
    leases: Dict[str, Any] = state["leases"]
    waiters: Dict[str, Any] = state["waiters"]
    live = 0
    changed = False
    reclaimed_pids = set()
    # Liveness is constant within this synchronous pass (no await, fixed
    # ``now``), so memoize the probe: a PID holding many leases of this group
    # is checked once instead of once per lease. NEVER cache across calls —
    # PID reuse and staleness could reclaim a live owner's lease.
    alive_cache: Dict[int, bool] = {}

    def _alive(pid: int) -> bool:
        cached = alive_cache.get(pid)
        if cached is None:
            cached = _pid_alive(pid)
            alive_cache[pid] = cached
        return cached

    for lease_id in list(leases.keys()):
        lease = leases[lease_id]
        pid = lease.get("pid")
        updated_at = lease.get("updated_at", 0.0)
        if pid is None or not _alive(pid):
            leases.pop(lease_id)
            changed = True
            if pid is not None:
                reclaimed_pids.add(pid)
            continue
        if now - updated_at > _heartbeat_ttl:
            suspect_since = lease.get("suspect_since")
            if suspect_since is None:
                lease["suspect_since"] = now
                changed = True
                live += 1
            elif now - suspect_since > _suspect_grace:
                leases.pop(lease_id)
                reclaimed_pids.add(pid)
                changed = True
            else:
                live += 1
        else:
            if "suspect_since" in lease:
                lease.pop("suspect_since", None)
                changed = True
            live += 1

    for pid_key in list(waiters.keys()):
        waiter = waiters[pid_key]
        pid = waiter.get("pid")
        last_poll = waiter.get("last_poll", 0.0)
        if (
            pid is None
            or pid in reclaimed_pids
            or not _alive(pid)
            or now - last_poll > _waiter_stale_ttl
        ):
            waiters.pop(pid_key)
            changed = True
    return live, changed


def _log_acquire_failure(group: str, error: Exception) -> None:
    global _last_acquire_failure_log
    now = time.time()
    if now - _last_acquire_failure_log >= _ACQUIRE_FAILURE_LOG_INTERVAL:
        _last_acquire_failure_log = now
        direct_log(
            f"Process {os.getpid()} failed to acquire global slot for group "
            f"'{group}' (fail-closed, task stays queued): {error}",
            level="WARNING",
        )


def _is_longest_live_waiter(state: Dict[str, Any], pid: int, now: float) -> bool:
    """Is ``pid`` the longest-waiting live poller in this gate state?

    Operates on a local state copy after the reap pass has dropped
    dead/stale waiter records — every remaining record belongs to a live,
    actively polling process.
    """
    my_start = None
    others_min = None
    for waiter in state["waiters"].values():
        wait_start = waiter.get("wait_start", now)
        if waiter.get("pid") == pid:
            my_start = wait_start
        elif others_min is None or wait_start < others_min:
            others_min = wait_start
    if my_start is None:
        return False
    return others_min is None or my_start <= others_min


async def _acquire_global_slot(
    group: str, track_wait: bool
) -> tuple[Optional[str], bool]:
    """Shared implementation for the two acquire entry points.

    Returns ``(lease_id, is_priority_waiter)``. When ``track_wait`` is set,
    a failed attempt registers/refreshes this process's waiter record
    (``wait_start`` set once per waiting episode, ``last_poll`` refreshed on
    every attempt) and reports whether this process is the longest-waiting
    live poller; a successful attempt always clears the record.

    IPC budget under the keyed lock: one state read, plus one write only
    when something changed (reap effects, waiter registration, or a new
    lease) — a plain failed attempt on an unchanged gate writes nothing.
    """
    limit = get_global_concurrency_limit(group)
    if limit is None or limit <= 0:
        return None, False
    try:
        ns = await _get_lease_namespace()
        async with get_storage_keyed_lock(
            group, namespace=_CONCURRENCY_LEASE_NAMESPACE, enable_logging=False
        ):
            now = time.time()
            state = _load_gate_state(ns, group)
            in_use, changed = _reap_gate_state(state, now)
            pid = os.getpid()
            pid_key = str(pid)
            if in_use >= limit:
                if not track_wait:
                    if changed:
                        ns[group] = state
                    return None, False
                waiter = state["waiters"].get(pid_key) or {
                    "pid": pid,
                    "wait_start": now,
                }
                waiter["last_poll"] = now
                state["waiters"][pid_key] = waiter
                ns[group] = state
                return None, _is_longest_live_waiter(state, pid, now)
            lease_id = uuid.uuid4().hex
            state["leases"][lease_id] = {"pid": pid, "updated_at": now}
            # Got a slot: this process is no longer waiting. Resetting here
            # (rather than keeping seniority) is what de-prioritizes a
            # backlog-heavy process after each win, yielding approximate
            # round-robin across processes under sustained contention.
            state["waiters"].pop(pid_key, None)
            ns[group] = state
            return lease_id, True
    except Exception as e:
        _log_acquire_failure(group, e)
        return None, False


async def try_acquire_global_slot(group: str) -> Optional[str]:
    """Try to claim one global concurrency slot for ``group`` (non-blocking).

    Returns a lease id on success, or None when the group is at capacity.
    Any shared-storage error is fail-closed: returns None (with a
    rate-limited warning) so the caller keeps the task queued and retries —
    capacity is never exceeded due to infrastructure errors.

    This plain variant never registers waiter records — use
    :func:`try_acquire_global_slot_tracked` from polling loops that want
    longest-waiter fairness.
    """
    lease_id, _ = await _acquire_global_slot(group, track_wait=False)
    return lease_id


async def try_acquire_global_slot_tracked(group: str) -> tuple[Optional[str], bool]:
    """Acquire variant for polling loops: ``(lease_id, is_priority_waiter)``.

    On failure the caller's waiter record is registered/refreshed and the
    second element reports whether this process is currently the
    longest-waiting live poller of the group. Pollers should keep the
    fastest poll interval when favored and back off (bounded) otherwise —
    a soft FIFO across worker processes with no hard gate: any poller that
    finds a free slot still takes it, so a sleeping favored waiter can
    never leave capacity idle indefinitely. Fail-closed errors report
    ``(None, False)``.
    """
    return await _acquire_global_slot(group, track_wait=True)


async def clear_slot_waiter(group: str) -> None:
    """Drop this process's waiter record for ``group`` (idempotent).

    Called when a wrapper shuts down so a no-longer-polling process never
    lingers in the longest-waiter seat; the stale TTL and the reap pass
    cover crashes where this cleanup never runs.
    """
    if not _initialized:
        return
    ns = await _get_lease_namespace()
    async with get_storage_keyed_lock(
        group, namespace=_CONCURRENCY_LEASE_NAMESPACE, enable_logging=False
    ):
        state = _load_gate_state(ns, group)
        if state["waiters"].pop(str(os.getpid()), None) is not None:
            ns[group] = state


async def global_slot_waiters(group: str) -> List[Dict[str, Any]]:
    """Snapshot of processes actively polling for a slot of ``group``.

    Returns ``[{"pid": ..., "waited": seconds}, ...]`` sorted by descending
    wait time; stale records (not refreshed within the waiter TTL) are
    skipped. Read-only and lock-free — intended for observability.
    """
    if not _initialized:
        return []
    ns = await _get_lease_namespace()
    now = time.time()
    state = _load_gate_state(ns, group)
    waiters = []
    for waiter in state["waiters"].values():
        if now - waiter.get("last_poll", 0.0) > _waiter_stale_ttl:
            continue
        waiters.append(
            {
                "pid": waiter.get("pid"),
                "waited": max(0.0, now - waiter.get("wait_start", now)),
            }
        )
    return sorted(waiters, key=lambda w: -w["waited"])


async def release_global_slot(group: str, lease_id: str) -> None:
    """Release a previously acquired global slot (idempotent).

    Raises on shared-storage errors — callers that must never propagate
    (e.g. worker ``finally`` blocks) wrap this and queue the lease for a
    later retry; the heartbeat TTL guarantees eventual reclamation anyway.
    """
    ns = await _get_lease_namespace()
    async with get_storage_keyed_lock(
        group, namespace=_CONCURRENCY_LEASE_NAMESPACE, enable_logging=False
    ):
        state = _load_gate_state(ns, group)
        if state["leases"].pop(lease_id, None) is not None:
            ns[group] = state


async def renew_global_slots(group: str, lease_ids) -> None:
    """Refresh the heartbeat of this process's held leases for ``group``.

    A renewal rewrites the lease whole (clearing any ``suspect_since``
    mark). Leases that have already been reclaimed are NOT resurrected —
    re-inserting could exceed the configured limit; the suspect grace
    exists precisely to make false reclamation unlikely.
    """
    lease_ids = list(lease_ids)
    if not lease_ids:
        return
    ns = await _get_lease_namespace()
    async with get_storage_keyed_lock(
        group, namespace=_CONCURRENCY_LEASE_NAMESPACE, enable_logging=False
    ):
        now = time.time()
        state = _load_gate_state(ns, group)
        changed = False
        for lease_id in lease_ids:
            if lease_id in state["leases"]:
                state["leases"][lease_id] = {"pid": os.getpid(), "updated_at": now}
                changed = True
        if changed:
            ns[group] = state


async def reconcile_global_slots(group: str) -> int:
    """Run the lease reaper for ``group``; return surviving lease count."""
    ns = await _get_lease_namespace()
    async with get_storage_keyed_lock(
        group, namespace=_CONCURRENCY_LEASE_NAMESPACE, enable_logging=False
    ):
        state = _load_gate_state(ns, group)
        live, changed = _reap_gate_state(state, time.time())
        if changed:
            ns[group] = state
        return live


async def global_concurrency_in_use(group: str) -> int:
    """Approximate count of currently held global slots for ``group``.

    Lock-free single read — intended for observability.
    """
    ns = await _get_lease_namespace()
    return len(_load_gate_state(ns, group)["leases"])


# ---------------------------------------------------------------------------
# Cross-worker queue stats (best-effort, debounced snapshots)
# ---------------------------------------------------------------------------
#
# Each worker process publishes per-queue snapshots under
# ``f"{queue_name}{KEY_SEP}{pid}"`` in the workspace-less "queue_stats"
# namespace (whole-value replacement). The local closure counters remain
# the source of truth; the shared area only needs to be "fresh enough"
# (event-triggered publishes debounced by the caller + 5s heartbeat flush).

# Flat counter fields summed across workers during aggregation. These are
# the existing get_queue_stats() fields (schema compatibility for /health
# and the webui) plus the new global_slot_waits / physical_queued counters.
QUEUE_STATS_SUM_FIELDS = (
    "queued",
    "running",
    "in_flight",
    "worker_count",
    "submitted_total",
    "completed_total",
    "failed_total",
    "cancelled_total",
    "rejected_total",
    "global_slot_waits",
    "physical_queued",
)


async def publish_queue_stats(queue_name: str, snapshot: Dict[str, Any]) -> None:
    """Publish this process's snapshot for ``queue_name`` (whole replacement).

    The snapshot must carry ``pid`` and ``updated_at`` (wall-clock time) so
    aggregation can reap stale entries. Best-effort by contract: callers
    must tolerate exceptions.
    """
    if not _initialized:
        return
    ns = await _get_queue_stats_namespace()
    ns[f"{queue_name}{KEY_SEP}{os.getpid()}"] = dict(snapshot)


async def unpublish_queue_stats(queue_name: str) -> None:
    """Remove this process's snapshot for ``queue_name`` (idempotent)."""
    if not _initialized:
        return
    ns = await _get_queue_stats_namespace()
    ns.pop(f"{queue_name}{KEY_SEP}{os.getpid()}", None)


async def aggregate_queue_stats(queue_name: str) -> Dict[str, Any]:
    """Aggregate all workers' published snapshots for ``queue_name``.

    Sums the flat counter fields across live snapshots and returns them
    together with ``reporting_workers`` and the raw ``per_worker`` map.
    Entries owned by dead PIDs or older than the stale TTL are reaped —
    re-checked under the internal lock against the previously observed
    ``updated_at`` so a snapshot republished concurrently is never deleted.
    """
    ns = await _get_queue_stats_namespace()
    now = time.time()
    prefix = f"{queue_name}{KEY_SEP}"
    per_worker: Dict[str, Dict[str, Any]] = {}
    stale: List[tuple] = []
    for key in [k for k in ns.keys() if k.startswith(prefix)]:
        raw = ns.get(key)
        if raw is None:
            continue
        snap = dict(raw)
        pid = snap.get("pid")
        updated_at = snap.get("updated_at", 0.0)
        if (pid is not None and not _pid_alive(pid)) or (
            now - updated_at > _queue_stats_stale_ttl
        ):
            stale.append((key, updated_at))
            continue
        per_worker[str(pid)] = snap

    if stale:
        async with get_internal_lock():
            for key, seen_updated_at in stale:
                current = ns.get(key)
                if current is None:
                    continue
                if dict(current).get("updated_at", 0.0) != seen_updated_at:
                    continue  # republished since we looked — keep it
                ns.pop(key, None)

    aggregated: Dict[str, Any] = {
        field: sum(int(snap.get(field, 0) or 0) for snap in per_worker.values())
        for field in QUEUE_STATS_SUM_FIELDS
    }
    aggregated["reporting_workers"] = len(per_worker)
    aggregated["per_worker"] = per_worker
    return aggregated
