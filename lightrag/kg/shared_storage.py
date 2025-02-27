import os
import sys
from multiprocessing.synchronize import Lock as ProcessLock
from threading import Lock as ThreadLock
from multiprocessing import Manager
from typing import Any, Dict, Optional, Union


# Define a direct print function for critical logs that must be visible in all processes
def direct_log(message, level="INFO"):
    """
    Log a message directly to stderr to ensure visibility in all processes,
    including the Gunicorn master process.
    """
    print(f"{level}: {message}", file=sys.stderr, flush=True)


LockType = Union[ProcessLock, ThreadLock]

_manager = None
_initialized = None
is_multiprocess = None
_global_lock: Optional[LockType] = None

# shared data for storage across processes
_shared_dicts: Optional[Dict[str, Any]] = None
_init_flags: Optional[Dict[str, bool]] = None  # namespace -> initialized

def initialize_share_data(workers: int = 1):
    """
    Initialize shared storage data for single or multi-process mode.

    When used with Gunicorn's preload feature, this function is called once in the
    master process before forking worker processes, allowing all workers to share
    the same initialized data.

    In single-process mode, this function is called during LightRAG object initialization.

    The function determines whether to use cross-process shared variables for data storage
    based on the number of workers. If workers=1, it uses thread locks and local dictionaries.
    If workers>1, it uses process locks and shared dictionaries managed by multiprocessing.Manager.

    Args:
        workers (int): Number of worker processes. If 1, single-process mode is used.
                      If > 1, multi-process mode with shared memory is used.
    """
    global \
        _manager, \
        is_multiprocess, \
        is_multiprocess, \
        _global_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized

    # Check if already initialized
    if _initialized:
        direct_log(
            f"Process {os.getpid()} Shared-Data already initialized (multiprocess={is_multiprocess})"
        )
        return

    _manager = Manager()

    # Force multi-process mode if workers > 1
    if workers > 1:
        is_multiprocess = True
        _global_lock = _manager.Lock()
        # Create shared dictionaries with manager
        _shared_dicts = _manager.dict()
        _init_flags = (
            _manager.dict()
        )  # Use shared dictionary to store initialization flags
        direct_log(
            f"Process {os.getpid()} Shared-Data created for Multiple Process (workers={workers})"
        )
    else:
        is_multiprocess = False
        _global_lock = ThreadLock()
        _shared_dicts = {}
        _init_flags = {}
        direct_log(f"Process {os.getpid()} Shared-Data created for Single Process")

    # Mark as initialized
    _initialized = True


def try_initialize_namespace(namespace: str) -> bool:
    """
    Try to initialize a namespace. Returns True if the current process gets initialization permission.
    Uses atomic operations on shared dictionaries to ensure only one process can successfully initialize.
    """
    global _init_flags, _manager

    if _init_flags is None:
        raise ValueError("Try to create nanmespace before Shared-Data is initialized")

    if namespace not in _init_flags:
        _init_flags[namespace] = True
        direct_log(f"Process {os.getpid()} ready to initialize storage namespace: [{namespace}]")
        return True
    direct_log(f"Process {os.getpid()} storage namespace already to initialized: [{namespace}]")
    return False


def get_storage_lock() -> LockType:
    """return storage lock for data consistency"""
    return _global_lock


def get_namespace_data(namespace: str) -> Dict[str, Any]:
    """get storage space for specific storage type(namespace)"""
    if _shared_dicts is None:
        direct_log(
            f"Error: try to getnanmespace before Shared-Data is initialized, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Shared dictionaries not initialized")

    lock = get_storage_lock()
    with lock:
        if namespace not in _shared_dicts:
            if is_multiprocess and _manager is not None:
                _shared_dicts[namespace] = _manager.dict()
            else:
                _shared_dicts[namespace] = {}
            direct_log(
                f"Created namespace: {{namespace}}({type(_shared_dicts[namespace])}) "
            )

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
        is_multiprocess, \
        _global_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized

    # Check if already initialized
    if not _initialized:
        direct_log(
            f"Process {os.getpid()} storage data not initialized, nothing to finalize"
        )
        return

    direct_log(
        f"Process {os.getpid()} finalizing storage data (multiprocess={is_multiprocess})"
    )

    # In multi-process mode, shut down the Manager
    if is_multiprocess and _manager is not None:
        try:
            # Clear shared dictionaries first
            if _shared_dicts is not None:
                _shared_dicts.clear()
            if _init_flags is not None:
                _init_flags.clear()

            # Shut down the Manager
            _manager.shutdown()
            direct_log(f"Process {os.getpid()} Manager shutdown complete")
        except Exception as e:
            direct_log(
                f"Process {os.getpid()} Error shutting down Manager: {e}", level="ERROR"
            )

    # Reset global variables
    _manager = None
    _initialized = None
    is_multiprocess = None
    _shared_dicts = None
    _init_flags = None
    _global_lock = None

    direct_log(f"Process {os.getpid()} storage data finalization complete")
