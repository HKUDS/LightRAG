import os
import sys
import asyncio
from multiprocessing.synchronize import Lock as ProcessLock
from multiprocessing import Manager
from typing import Any, Dict, Optional, Union, TypeVar, Generic


# Define a direct print function for critical logs that must be visible in all processes
def direct_log(message, level="INFO"):
    """
    Log a message directly to stderr to ensure visibility in all processes,
    including the Gunicorn master process.
    """
    print(f"{level}: {message}", file=sys.stderr, flush=True)


T = TypeVar('T')

class UnifiedLock(Generic[T]):
    """Provide a unified lock interface type for asyncio.Lock and multiprocessing.Lock"""
    def __init__(self, lock: Union[ProcessLock, asyncio.Lock], is_async: bool):
        self._lock = lock
        self._is_async = is_async

    async def __aenter__(self) -> 'UnifiedLock[T]':
        if self._is_async:
            await self._lock.acquire()
        else:
            self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._is_async:
            self._lock.release()
        else:
            self._lock.release()

    def __enter__(self) -> 'UnifiedLock[T]':
        """For backward compatibility"""
        if self._is_async:
            raise RuntimeError("Use 'async with' for asyncio.Lock")
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For backward compatibility"""
        if self._is_async:
            raise RuntimeError("Use 'async with' for asyncio.Lock")
        self._lock.release()


LockType = Union[ProcessLock, asyncio.Lock]

is_multiprocess = None
_workers = None
_manager = None
_initialized = None
_global_lock: Optional[LockType] = None

# shared data for storage across processes
_shared_dicts: Optional[Dict[str, Any]] = None
_init_flags: Optional[Dict[str, bool]] = None  # namespace -> initialized
_update_flags: Optional[Dict[str, bool]] = None # namespace -> updated


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
        _workers, \
        is_multiprocess, \
        is_multiprocess, \
        _global_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized, \
        _update_flags

    # Check if already initialized
    if _initialized:
        direct_log(
            f"Process {os.getpid()} Shared-Data already initialized (multiprocess={is_multiprocess})"
        )
        return

    _manager = Manager()
    _workers = workers

    if workers > 1:
        is_multiprocess = True
        _global_lock = _manager.Lock()
        _shared_dicts = _manager.dict()
        _init_flags = _manager.dict()
        _update_flags = _manager.dict()
        direct_log(
            f"Process {os.getpid()} Shared-Data created for Multiple Process (workers={workers})"
        )
    else:
        is_multiprocess = False
        _global_lock = asyncio.Lock()
        _shared_dicts = {}
        _init_flags = {}
        _update_flags = {}
        direct_log(f"Process {os.getpid()} Shared-Data created for Single Process")

    # Mark as initialized
    _initialized = True


async def initialize_pipeline_namespace():
    """
    Initialize pipeline namespace with default values.
    """
    pipeline_namespace = await get_namespace_data("pipeline_status")

    async with get_storage_lock():
        # Check if already initialized by checking for required fields
        if "busy" in pipeline_namespace:
            return

        # Create a shared list object for history_messages
        history_messages = _manager.list() if is_multiprocess else []
        pipeline_namespace.update({
            "busy": False,  # Control concurrent processes
            "job_name": "Default Job",  # Current job name (indexing files/indexing texts)
            "job_start": None,  # Job start time
            "docs": 0,  # Total number of documents to be indexed
            "batchs": 0,  # Number of batches for processing documents
            "cur_batch": 0,  # Current processing batch
            "request_pending": False,  # Flag for pending request for processing
            "latest_message": "",  # Latest message from pipeline processing
            "history_messages": history_messages,  # 使用共享列表对象
        })
        direct_log(f"Process {os.getpid()} Pipeline namespace initialized")


async def get_update_flag(namespace: str):
    """
    Create a namespace's update flag for a workers.
    Returen the update flag to caller for referencing or reset.
    """
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    async with get_storage_lock():
        if namespace not in _update_flags:
            if is_multiprocess and _manager is not None:
                _update_flags[namespace] = _manager.list()
            else:
                _update_flags[namespace] = []
            direct_log(f"Process {os.getpid()} initialized updated flags for namespace: [{namespace}]")
        
        if is_multiprocess and _manager is not None:
            new_update_flag = _manager.Value('b', False)
        else:
            new_update_flag = False
        
        _update_flags[namespace].append(new_update_flag)
        return new_update_flag

async def set_all_update_flags(namespace: str):
    """Set all update flag of namespace indicating all workers need to reload data from files"""
    global _update_flags
    if _update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")
    
    async with get_storage_lock():
        if namespace not in _update_flags:
            raise ValueError(f"Namespace {namespace} not found in update flags")
        # Update flags for both modes
        for i in range(len(_update_flags[namespace])):
            if is_multiprocess:
                _update_flags[namespace][i].value = True
            else:
                _update_flags[namespace][i] = True


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
        direct_log(
            f"Process {os.getpid()} ready to initialize storage namespace: [{namespace}]"
        )
        return True
    direct_log(
        f"Process {os.getpid()} storage namespace already initialized: [{namespace}]"
    )
    return False


def get_storage_lock() -> UnifiedLock:
    """return unified storage lock for data consistency"""
    return UnifiedLock(
        lock=_global_lock,
        is_async=not is_multiprocess
    )


async def get_namespace_data(namespace: str) -> Dict[str, Any]:
    """get storage space for specific storage type(namespace)"""
    if _shared_dicts is None:
        direct_log(
            f"Error: try to getnanmespace before Shared-Data is initialized, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Shared dictionaries not initialized")

    async with get_storage_lock():
        if namespace not in _shared_dicts:
            if is_multiprocess and _manager is not None:
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
