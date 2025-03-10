import os
import sys
import asyncio
from multiprocessing.synchronize import Lock as ProcessLock
from multiprocessing import Manager
from typing import Any, Dict, Optional, Union, TypeVar, Generic


# Define a direct print function for critical logs that must be visible in all processes
def direct_log(message, level="INFO", enable_output: bool = True):
    """
    Log a message directly to stderr to ensure visibility in all processes,
    including the Gunicorn master process.

    Args:
        message: The message to log
        level: Log level (default: "INFO")
        enable_output: Whether to actually output the log (default: True)
    """
    if enable_output:
        print(f"{level}: {message}", file=sys.stderr, flush=True)


T = TypeVar("T")
LockType = Union[ProcessLock, asyncio.Lock]

is_multiprocess = None
_workers = None
_manager = None
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


class UnifiedLock(Generic[T]):
    """Provide a unified lock interface type for asyncio.Lock and multiprocessing.Lock"""

    def __init__(
        self,
        lock: Union[ProcessLock, asyncio.Lock],
        is_async: bool,
        name: str = "unnamed",
        enable_logging: bool = True,
    ):
        self._lock = lock
        self._is_async = is_async
        self._pid = os.getpid()  # for debug only
        self._name = name  # for debug only
        self._enable_logging = enable_logging  # for debug only

    async def __aenter__(self) -> "UnifiedLock[T]":
        try:
            direct_log(
                f"== Lock == Process {self._pid}: Acquiring lock '{self._name}' (async={self._is_async})",
                enable_output=self._enable_logging,
            )
            if self._is_async:
                await self._lock.acquire()
            else:
                self._lock.acquire()
            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' acquired (async={self._is_async})",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}': {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            direct_log(
                f"== Lock == Process {self._pid}: Releasing lock '{self._name}' (async={self._is_async})",
                enable_output=self._enable_logging,
            )
            if self._is_async:
                self._lock.release()
            else:
                self._lock.release()
            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' released (async={self._is_async})",
                enable_output=self._enable_logging,
            )
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}': {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )
            raise

    def __enter__(self) -> "UnifiedLock[T]":
        """For backward compatibility"""
        try:
            if self._is_async:
                raise RuntimeError("Use 'async with' for shared_storage lock")
            direct_log(
                f"== Lock == Process {self._pid}: Acquiring lock '{self._name}' (sync)",
                enable_output=self._enable_logging,
            )
            self._lock.acquire()
            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' acquired (sync)",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For backward compatibility"""
        try:
            if self._is_async:
                raise RuntimeError("Use 'async with' for shared_storage lock")
            direct_log(
                f"== Lock == Process {self._pid}: Releasing lock '{self._name}' (sync)",
                enable_output=self._enable_logging,
            )
            self._lock.release()
            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' released (sync)",
                enable_output=self._enable_logging,
            )
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )
            raise


def get_internal_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified storage lock for data consistency"""
    return UnifiedLock(
        lock=_internal_lock,
        is_async=not is_multiprocess,
        name="internal_lock",
        enable_logging=enable_logging,
    )


def get_storage_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified storage lock for data consistency"""
    return UnifiedLock(
        lock=_storage_lock,
        is_async=not is_multiprocess,
        name="storage_lock",
        enable_logging=enable_logging,
    )


def get_pipeline_status_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified storage lock for data consistency"""
    return UnifiedLock(
        lock=_pipeline_status_lock,
        is_async=not is_multiprocess,
        name="pipeline_status_lock",
        enable_logging=enable_logging,
    )


def get_graph_db_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified graph database lock for ensuring atomic operations"""
    return UnifiedLock(
        lock=_graph_db_lock,
        is_async=not is_multiprocess,
        name="graph_db_lock",
        enable_logging=enable_logging,
    )


def get_data_init_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified data initialization lock for ensuring atomic data initialization"""
    return UnifiedLock(
        lock=_data_init_lock,
        is_async=not is_multiprocess,
        name="data_init_lock",
        enable_logging=enable_logging,
    )


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
        is_multiprocess, \
        _storage_lock, \
        _internal_lock, \
        _pipeline_status_lock, \
        _graph_db_lock, \
        _data_init_lock, \
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

    _workers = workers

    if workers > 1:
        is_multiprocess = True
        _manager = Manager()
        _internal_lock = _manager.Lock()
        _storage_lock = _manager.Lock()
        _pipeline_status_lock = _manager.Lock()
        _graph_db_lock = _manager.Lock()
        _data_init_lock = _manager.Lock()
        _shared_dicts = _manager.dict()
        _init_flags = _manager.dict()
        _update_flags = _manager.dict()
        direct_log(
            f"Process {os.getpid()} Shared-Data created for Multiple Process (workers={workers})"
        )
    else:
        is_multiprocess = False
        _internal_lock = asyncio.Lock()
        _storage_lock = asyncio.Lock()
        _pipeline_status_lock = asyncio.Lock()
        _graph_db_lock = asyncio.Lock()
        _data_init_lock = asyncio.Lock()
        _shared_dicts = {}
        _init_flags = {}
        _update_flags = {}
        direct_log(f"Process {os.getpid()} Shared-Data created for Single Process")

    # Mark as initialized
    _initialized = True


async def initialize_pipeline_status():
    """
    Initialize pipeline namespace with default values.
    This function is called during FASTAPI lifespan for each worker.
    """
    pipeline_namespace = await get_namespace_data("pipeline_status")

    async with get_internal_lock():
        # Check if already initialized by checking for required fields
        if "busy" in pipeline_namespace:
            return

        # Create a shared list object for history_messages
        history_messages = _manager.list() if is_multiprocess else []
        pipeline_namespace.update(
            {
                "autoscanned": False,  # Auto-scan started
                "busy": False,  # Control concurrent processes
                "job_name": "Default Job",  # Current job name (indexing files/indexing texts)
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
            if is_multiprocess and _manager is not None:
                _update_flags[namespace] = _manager.list()
            else:
                _update_flags[namespace] = []
            direct_log(
                f"Process {os.getpid()} initialized updated flags for namespace: [{namespace}]"
            )

        if is_multiprocess and _manager is not None:
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
            if is_multiprocess:
                _update_flags[namespace][i].value = True
            else:
                # Use .value attribute instead of direct assignment
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
            if is_multiprocess:
                _update_flags[namespace][i].value = False
            else:
                # Use .value attribute instead of direct assignment
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
                if is_multiprocess:
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


async def get_namespace_data(namespace: str) -> Dict[str, Any]:
    """get the shared data reference for specific namespace"""
    if _shared_dicts is None:
        direct_log(
            f"Error: try to getnanmespace before it is initialized, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Shared dictionaries not initialized")

    async with get_internal_lock():
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
        _storage_lock, \
        _internal_lock, \
        _pipeline_status_lock, \
        _graph_db_lock, \
        _data_init_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized, \
        _update_flags

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
    is_multiprocess = None
    _shared_dicts = None
    _init_flags = None
    _storage_lock = None
    _internal_lock = None
    _pipeline_status_lock = None
    _graph_db_lock = None
    _data_init_lock = None
    _update_flags = None

    direct_log(f"Process {os.getpid()} storage data finalization complete")
