import os
import sys
import asyncio
import multiprocessing as mp
from multiprocessing.synchronize import Lock as ProcessLock
from multiprocessing import Manager
import time
import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

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
_lock_registry: Optional[Dict[str, mp.synchronize.Lock]] = None
_lock_registry_count: Optional[Dict[str, int]] = None
_lock_cleanup_data: Optional[Dict[str, time.time]] = None
_registry_guard = None
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

# shared data for storage across processes
_shared_dicts: Optional[Dict[str, Any]] = None
_init_flags: Optional[Dict[str, bool]] = None  # namespace -> initialized
_update_flags: Optional[Dict[str, bool]] = None  # namespace -> updated

# locks for mutex access
_storage_lock: Optional[LockType] = None
_internal_lock: Optional[LockType] = None
_pipeline_status_lock: Optional[LockType] = None
_graph_db_lock: Optional[LockType] = None
_data_init_lock: Optional[LockType] = None
# Manager for all keyed locks
_storage_keyed_lock: Optional["KeyedUnifiedLock"] = None
# Dictionary to store workspace-specific locks, format: {workspace:lock_name -> lock}
_sync_locks: dict = {}

# async locks for coroutine synchronization in multiprocess mode
_async_locks: Optional[Dict[str, asyncio.Lock]] = None

_debug_n_locks_acquired: int = 0


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
    ):
        self._lock = lock
        self._is_async = is_async
        self._pid = os.getpid()  # for debug only
        self._name = name  # for debug only
        self._enable_logging = enable_logging  # for debug only
        self._async_lock = async_lock  # auxiliary lock for coroutine synchronization

    async def __aenter__(self) -> "UnifiedLock[T]":
        try:
            # If in multiprocess mode and async lock exists, acquire it first
            if not self._is_async and self._async_lock is not None:
                await self._async_lock.acquire()
                direct_log(
                    f"== Lock == Process {self._pid}: Acquired async lock '{self._name}",
                    level="DEBUG",
                    enable_output=self._enable_logging,
                )

            # Then acquire the main lock
            if self._is_async:
                await self._lock.acquire()
            else:
                self._lock.acquire()

            direct_log(
                f"== Lock == Process {self._pid}: Acquired lock {self._name} (async={self._is_async})",
                level="INFO",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            # If main lock acquisition fails, release the async lock if it was acquired
            if (
                not self._is_async
                and self._async_lock is not None
                and self._async_lock.locked()
            ):
                self._async_lock.release()

            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}': {e}",
                level="ERROR",
                enable_output=True,
            )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        main_lock_released = False
        try:
            # Release main lock first
            if self._is_async:
                self._lock.release()
            else:
                self._lock.release()
            main_lock_released = True

            direct_log(
                f"== Lock == Process {self._pid}: Released lock {self._name} (async={self._is_async})",
                level="INFO",
                enable_output=self._enable_logging,
            )

            # Then release async lock if in multiprocess mode
            if not self._is_async and self._async_lock is not None:
                self._async_lock.release()
                direct_log(
                    f"== Lock == Process {self._pid}: Released async lock {self._name}",
                    level="DEBUG",
                    enable_output=self._enable_logging,
                )

        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}': {e}",
                level="ERROR",
                enable_output=True,
            )

            # If main lock release failed but async lock hasn't been released, try to release it
            if (
                not main_lock_released
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
            if self._is_async:
                raise RuntimeError("Use 'async with' for shared_storage lock")
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


def _get_or_create_shared_raw_mp_lock(
    factory_name: str, key: str
) -> Optional[mp.synchronize.Lock]:
    """Return the *singleton* manager.Lock() proxy for keyed lock, creating if needed."""
    if not _is_multiprocess:
        return None

    with _registry_guard:
        combined_key = _get_combined_key(factory_name, key)
        raw = _lock_registry.get(combined_key)
        count = _lock_registry_count.get(combined_key)
        if raw is None:
            raw = _manager.Lock()
            _lock_registry[combined_key] = raw
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
        return raw


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
                is_async=False,  # manager.Lock is synchronous
                name=combined_key,
                enable_logging=enable_logging,
                async_lock=async_lock,  # prevents event‑loop blocking
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
            # This entire loop is protected by the outer shield
            for entry in reversed(self._ul):
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

        # CRITICAL: Protect the entire release process with shield
        # This ensures that even if cancellation occurs, all locks are released
        try:
            all_errors = await asyncio.shield(release_all_locks())
        except Exception as e:
            direct_log(
                f"Critical error during __aexit__ cleanup: {e}",
                level="ERROR",
                enable_output=True,
            )
            all_errors = []
        finally:
            # Always clear the lock list, even if shield was cancelled
            self._ul = None

        # If there were release errors and no other exception, raise the first release error
        if all_errors and exc_type is None:
            raise all_errors[0][2]  # (key, error_type, error)


def _get_workspace_lock(
    lock_type: str,
    global_lock: LockType,
    workspace: str = "",
    enable_logging: bool = False
) -> UnifiedLock:
    """Internal common implementation for workspace-aware locks.

    This function handles lazy creation of workspace-specific locks with proper
    synchronization to avoid race conditions. The creation is protected by
    _registry_guard in multiprocess mode and is naturally safe in single-process
    mode due to asyncio's single-threaded nature.

    Args:
        lock_type: The type of lock (e.g., "storage_lock", "pipeline_status_lock")
        global_lock: The global lock instance to use when workspace is empty
        workspace: Optional workspace identifier for namespace isolation
        enable_logging: Enable lock operation logging

    Returns:
        UnifiedLock instance for the specified lock type and workspace
    """
    # Handle None workspace as empty string
    if workspace is None:
        workspace = ""

    lock_name = f"{workspace}:{lock_type}" if workspace else lock_type

    if workspace:
        # Workspace-specific lock (lazy creation with synchronization)
        if _is_multiprocess:
            # Use registry guard to protect lazy creation in multiprocess mode
            with _registry_guard:
                if lock_name not in _sync_locks:
                    _sync_locks[lock_name] = _manager.RLock()
                    # Create companion async lock for this workspace lock
                    if _async_locks is not None and lock_name not in _async_locks:
                        _async_locks[lock_name] = asyncio.Lock()
        else:
            # Single-process mode - no synchronization needed due to asyncio nature
            if lock_name not in _sync_locks:
                _sync_locks[lock_name] = asyncio.Lock()

        async_lock = _async_locks.get(lock_name) if _is_multiprocess else None
        return UnifiedLock(
            lock=_sync_locks[lock_name],
            is_async=not _is_multiprocess,
            name=lock_name,
            enable_logging=enable_logging,
            async_lock=async_lock,
        )
    else:
        # Global lock (backward compatible)
        async_lock = _async_locks.get(lock_type) if _is_multiprocess else None
        return UnifiedLock(
            lock=global_lock,
            is_async=not _is_multiprocess,
            name=lock_type,
            enable_logging=enable_logging,
            async_lock=async_lock,
        )


def get_internal_lock(workspace: str = "", enable_logging: bool = False) -> UnifiedLock:
    """Return unified internal lock for data consistency.

    Args:
        workspace: Optional workspace identifier for namespace isolation.
        enable_logging: Enable lock operation logging.
    """
    return _get_workspace_lock("internal_lock", _internal_lock, workspace, enable_logging)


def get_storage_lock(workspace: str = "", enable_logging: bool = False) -> UnifiedLock:
    """Return unified storage lock for data consistency.

    Args:
        workspace: Optional workspace identifier for namespace isolation.
        enable_logging: Enable lock operation logging.
    """
    return _get_workspace_lock(
        "storage_lock", _storage_lock, workspace, enable_logging
    )


def get_pipeline_status_lock(workspace: str = "", enable_logging: bool = False) -> UnifiedLock:
    """Return unified pipeline status lock for concurrent processing control.

    Args:
        workspace: Optional workspace identifier for namespace isolation.
        enable_logging: Enable lock operation logging.
    """
    return _get_workspace_lock(
        "pipeline_status_lock", _pipeline_status_lock, workspace, enable_logging
    )


def get_graph_db_lock(workspace: str = "", enable_logging: bool = False) -> UnifiedLock:
    """Return unified graph database lock for ensuring atomic operations.

    Args:
        workspace: Optional workspace identifier for namespace isolation.
        enable_logging: Enable lock operation logging.
    """
    return _get_workspace_lock(
        "graph_db_lock", _graph_db_lock, workspace, enable_logging
    )


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


def get_data_init_lock(
    workspace: str = "", enable_logging: bool = False
) -> UnifiedLock:
    """Return unified data initialization lock for ensuring atomic data initialization.

    Args:
        workspace: Optional workspace identifier for namespace isolation.
        enable_logging: Enable lock operation logging.
    """
    return _get_workspace_lock(
        "data_init_lock", _data_init_lock, workspace, enable_logging
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


def initialize_share_data(workers: int = 1):
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
    """
    global \
        _manager, \
        _workers, \
        _is_multiprocess, \
        _storage_lock, \
        _lock_registry, \
        _lock_registry_count, \
        _lock_cleanup_data, \
        _registry_guard, \
        _internal_lock, \
        _pipeline_status_lock, \
        _graph_db_lock, \
        _data_init_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized, \
        _update_flags, \
        _async_locks, \
        _storage_keyed_lock, \
        _sync_locks, \
        _earliest_mp_cleanup_time, \
        _last_mp_cleanup_time

    # Check if already initialized
    if _initialized:
        direct_log(
            f"Process {os.getpid()} Shared-Data already initialized (multiprocess={_is_multiprocess})"
        )
        return

    _workers = workers

    if workers > 1:
        _is_multiprocess = True
        _manager = Manager()
        _lock_registry = _manager.dict()
        _lock_registry_count = _manager.dict()
        _lock_cleanup_data = _manager.dict()
        _registry_guard = _manager.RLock()
        _internal_lock = _manager.Lock()
        _storage_lock = _manager.Lock()
        _pipeline_status_lock = _manager.Lock()
        _graph_db_lock = _manager.Lock()
        _data_init_lock = _manager.Lock()
        _shared_dicts = _manager.dict()
        _init_flags = _manager.dict()
        _update_flags = _manager.dict()
        _sync_locks = _manager.dict()

        _storage_keyed_lock = KeyedUnifiedLock()

        # Initialize async locks for multiprocess mode
        _async_locks = {
            "internal_lock": asyncio.Lock(),
            "storage_lock": asyncio.Lock(),
            "pipeline_status_lock": asyncio.Lock(),
            "graph_db_lock": asyncio.Lock(),
            "data_init_lock": asyncio.Lock(),
        }

        direct_log(
            f"Process {os.getpid()} Shared-Data created for Multiple Process (workers={workers})"
        )
    else:
        _is_multiprocess = False
        _internal_lock = asyncio.Lock()
        _storage_lock = asyncio.Lock()
        _pipeline_status_lock = asyncio.Lock()
        _graph_db_lock = asyncio.Lock()
        _data_init_lock = asyncio.Lock()
        _shared_dicts = {}
        _init_flags = {}
        _update_flags = {}
        _sync_locks = {}
        _async_locks = None  # No need for async locks in single process mode

        _storage_keyed_lock = KeyedUnifiedLock()
        direct_log(f"Process {os.getpid()} Shared-Data created for Single Process")

    # Initialize multiprocess cleanup times
    _earliest_mp_cleanup_time = None
    _last_mp_cleanup_time = None

    # Mark as initialized
    _initialized = True


async def initialize_pipeline_status():
    """
    Initialize pipeline namespace with default values.
    This function is called during FASTAPI lifespan for each worker.
    """
    pipeline_namespace = await get_namespace_data("pipeline_status", first_init=True)

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
                "job_name": "-",  # Current job name (indexing files/indexing texts)
                "job_start": None,  # Job start time
                "docs": 0,  # Total number of documents to be indexed
                "batchs": 0,  # Number of batches for processing documents
                "cur_batch": 0,  # Current processing batch
                "request_pending": False,  # Flag for pending request for processing
                "latest_message": "",  # Latest message from pipeline processing
                "history_messages": history_messages,  # 使用共享列表对象
            }
        )
        direct_log(f"Process {os.getpid()} Pipeline namespace initialized")


async def get_update_flag(namespace: str):
    """
    Create a namespace's update flag for a workers.
    Returen the update flag to caller for referencing or reset.
    """
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    async with get_internal_lock():
        if namespace not in _update_flags:
            if _is_multiprocess and _manager is not None:
                _update_flags[namespace] = _manager.list()
            else:
                _update_flags[namespace] = []
            direct_log(
                f"Process {os.getpid()} initialized updated flags for namespace: [{namespace}]"
            )

        if _is_multiprocess and _manager is not None:
            new_update_flag = _manager.Value("b", False)
        else:
            # Create a simple mutable object to store boolean value for compatibility with mutiprocess
            class MutableBoolean:
                def __init__(self, initial_value=False):
                    self.value = initial_value

            new_update_flag = MutableBoolean(False)

        _update_flags[namespace].append(new_update_flag)
        return new_update_flag


async def set_all_update_flags(namespace: str):
    """Set all update flag of namespace indicating all workers need to reload data from files"""
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    async with get_internal_lock():
        if namespace not in _update_flags:
            raise ValueError(f"Namespace {namespace} not found in update flags")
        # Update flags for both modes
        for i in range(len(_update_flags[namespace])):
            _update_flags[namespace][i].value = True


async def clear_all_update_flags(namespace: str):
    """Clear all update flag of namespace indicating all workers need to reload data from files"""
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    async with get_internal_lock():
        if namespace not in _update_flags:
            raise ValueError(f"Namespace {namespace} not found in update flags")
        # Update flags for both modes
        for i in range(len(_update_flags[namespace])):
            _update_flags[namespace][i].value = False


async def get_all_update_flags_status() -> Dict[str, list]:
    """
    Get update flags status for all namespaces.

    Returns:
        Dict[str, list]: A dictionary mapping namespace names to lists of update flag statuses
    """
    if _update_flags is None:
        return {}

    result = {}
    async with get_internal_lock():
        for namespace, flags in _update_flags.items():
            worker_statuses = []
            for flag in flags:
                if _is_multiprocess:
                    worker_statuses.append(flag.value)
                else:
                    worker_statuses.append(flag)
            result[namespace] = worker_statuses

    return result


async def try_initialize_namespace(namespace: str) -> bool:
    """
    Returns True if the current worker(process) gets initialization permission for loading data later.
    The worker does not get the permission is prohibited to load data from files.
    """
    global _init_flags, _manager

    if _init_flags is None:
        raise ValueError("Try to create nanmespace before Shared-Data is initialized")

    async with get_internal_lock():
        if namespace not in _init_flags:
            _init_flags[namespace] = True
            direct_log(
                f"Process {os.getpid()} ready to initialize storage namespace: [{namespace}]"
            )
            return True
        direct_log(
            f"Process {os.getpid()} storage namespace already initialized: [{namespace}]"
        )

    return False


async def get_namespace_data(
    namespace: str, first_init: bool = False
) -> Dict[str, Any]:
    """get the shared data reference for specific namespace

    Args:
        namespace: The namespace to retrieve
        allow_create: If True, allows creation of the namespace if it doesn't exist.
                     Used internally by initialize_pipeline_status().
    """
    if _shared_dicts is None:
        direct_log(
            f"Error: try to getnanmespace before it is initialized, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Shared dictionaries not initialized")

    async with get_internal_lock():
        if namespace not in _shared_dicts:
            # Special handling for pipeline_status namespace
            if namespace == "pipeline_status" and not first_init:
                # Check if pipeline_status should have been initialized but wasn't
                # This helps users understand they need to call initialize_pipeline_status()
                raise PipelineNotInitializedError(namespace)

            # For other namespaces or when allow_create=True, create them dynamically
            if _is_multiprocess and _manager is not None:
                _shared_dicts[namespace] = _manager.dict()
            else:
                _shared_dicts[namespace] = {}

    return _shared_dicts[namespace]


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
        _storage_lock, \
        _internal_lock, \
        _pipeline_status_lock, \
        _graph_db_lock, \
        _data_init_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized, \
        _update_flags, \
        _async_locks

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
    _storage_lock = None
    _internal_lock = None
    _pipeline_status_lock = None
    _graph_db_lock = None
    _data_init_lock = None
    _update_flags = None
    _async_locks = None

    direct_log(f"Process {os.getpid()} storage data finalization complete")
