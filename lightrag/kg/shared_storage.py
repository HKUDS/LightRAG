import os
import sys
from multiprocessing.synchronize import Lock as ProcessLock
from threading import Lock as ThreadLock
from multiprocessing import Manager
from typing import Any, Dict, Optional, Union
from lightrag.utils import logger

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
_is_multiprocess = None
is_multiprocess = None

# shared data for storage across processes
_shared_dicts: Optional[Dict[str, Any]] = None
_share_objects: Optional[Dict[str, Any]] = None
_init_flags: Optional[Dict[str, bool]] = None  # namespace -> initialized

_global_lock: Optional[LockType] = None


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
    global _manager, _is_multiprocess, is_multiprocess, _global_lock, _shared_dicts, _share_objects, _init_flags, _initialized
        
    # Check if already initialized
    if _initialized and _initialized.value:
        is_multiprocess = _is_multiprocess.value
        direct_log(f"Process {os.getpid()} storage data already initialized (multiprocess={_is_multiprocess.value})")
        return
    
    _manager = Manager()
    _initialized = _manager.Value("b", False)
    _is_multiprocess = _manager.Value("b", False)

    # Force multi-process mode if workers > 1
    if workers > 1:
        _is_multiprocess.value = True
        _global_lock = _manager.Lock() 
        # Create shared dictionaries with manager
        _shared_dicts = _manager.dict()
        _share_objects = _manager.dict()
        _init_flags = _manager.dict()  # Use shared dictionary to store initialization flags
        direct_log(f"Process {os.getpid()} storage data created for Multiple Process (workers={workers})")
    else:
        _is_multiprocess.value = False
        _global_lock = ThreadLock()
        _shared_dicts = {}
        _share_objects = {}
        _init_flags = {}
        direct_log(f"Process {os.getpid()} storage data created for Single Process")

    # Mark as initialized
    _initialized.value = True
    is_multiprocess = _is_multiprocess.value

def try_initialize_namespace(namespace: str) -> bool:
    """
    Try to initialize a namespace. Returns True if the current process gets initialization permission.
    Uses atomic operations on shared dictionaries to ensure only one process can successfully initialize.
    """
    global _init_flags, _manager

    if _is_multiprocess.value:
        if _init_flags is None:
            raise RuntimeError(
                "Shared storage not initialized. Call initialize_share_data() first."
            )
    else:
        if _init_flags is None:
            _init_flags = {}

    logger.info(f"Process {os.getpid()} trying to initialize namespace {namespace}")

    with _global_lock:
        if namespace not in _init_flags:
            _init_flags[namespace] = True
            logger.info(
                f"Process {os.getpid()} ready to initialize namespace {namespace}"
            )
            return True
        logger.info(
            f"Process {os.getpid()} found namespace {namespace} already initialized"
        )
        return False


def _get_global_lock() -> LockType:
    return _global_lock


def get_storage_lock() -> LockType:
    """return storage lock for data consistency"""
    return _get_global_lock()


def get_scan_lock() -> LockType:
    """return scan_progress lock for data consistency"""
    return get_storage_lock()


def get_namespace_object(namespace: str) -> Any:
    """Get an object for specific namespace"""

    if namespace not in _share_objects:
        lock = _get_global_lock()
        with lock:
            if namespace not in _share_objects:
                if _is_multiprocess.value:
                    _share_objects[namespace] = _manager.Value("O", None)
                else:
                    _share_objects[namespace] = None

    return _share_objects[namespace]

def get_namespace_data(namespace: str) -> Dict[str, Any]:
    """get storage space for specific storage type(namespace)"""

    if namespace not in _shared_dicts:
        lock = _get_global_lock()
        with lock:
            if namespace not in _shared_dicts:
                _shared_dicts[namespace] = {}

    return _shared_dicts[namespace]


def get_scan_progress() -> Dict[str, Any]:
    """get storage space for document scanning progress data"""
    return get_namespace_data("scan_progress")


def finalize_share_data():
    """
    Release shared resources and clean up.
    
    This function should be called when the application is shutting down
    to properly release shared resources and avoid memory leaks.
    
    In multi-process mode, it shuts down the Manager and releases all shared objects.
    In single-process mode, it simply resets the global variables.
    """
    global _manager, _is_multiprocess, is_multiprocess, _global_lock, _shared_dicts, _share_objects, _init_flags, _initialized
    
    # Check if already initialized
    if not (_initialized and _initialized.value):
        direct_log(f"Process {os.getpid()} storage data not initialized, nothing to finalize")
        return
    
    direct_log(f"Process {os.getpid()} finalizing storage data (multiprocess={_is_multiprocess.value})")
    
    # In multi-process mode, shut down the Manager
    if _is_multiprocess.value and _manager is not None:
        try:
            # Clear shared dictionaries first
            if _shared_dicts is not None:
                _shared_dicts.clear()
            if _share_objects is not None:
                _share_objects.clear()
            if _init_flags is not None:
                _init_flags.clear()
            
            # Shut down the Manager
            _manager.shutdown()
            direct_log(f"Process {os.getpid()} Manager shutdown complete")
        except Exception as e:
            direct_log(f"Process {os.getpid()} Error shutting down Manager: {e}", level="ERROR")
    
    # Reset global variables
    _manager = None
    _initialized = None
    _is_multiprocess = None
    is_multiprocess = None
    _shared_dicts = None
    _share_objects = None
    _init_flags = None
    _global_lock = None
    
    direct_log(f"Process {os.getpid()} storage data finalization complete")
