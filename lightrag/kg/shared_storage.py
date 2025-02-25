from multiprocessing.synchronize import Lock as ProcessLock
from threading import Lock as ThreadLock
from multiprocessing import Manager
from typing import Any, Dict, Optional, Union

# 定义类型变量
LockType = Union[ProcessLock, ThreadLock]

# 全局变量
_shared_data: Optional[Dict[str, Any]] = None
_namespace_objects: Optional[Dict[str, Any]] = None
_global_lock: Optional[LockType] = None
is_multiprocess = False
manager = None

def initialize_manager():
    """Initialize manager, only for multiple processes where workers > 1"""
    global manager
    if manager is None:
        manager = Manager()

def _get_global_lock() -> LockType:
    global _global_lock, is_multiprocess
    
    if _global_lock is None:
        if is_multiprocess:
            _global_lock = manager.Lock()
        else:
            _global_lock = ThreadLock()
    
    return _global_lock

def get_storage_lock() -> LockType:
    """return storage lock for data consistency"""
    return _get_global_lock()

def get_scan_lock() -> LockType:
    """return scan_progress lock for data consistency"""
    return get_storage_lock()

def get_shared_data() -> Dict[str, Any]:
    """
    return shared data for all storage types
    create mult-process save share data only if need for better performance
    """
    global _shared_data, is_multiprocess
    
    if _shared_data is None:
        lock = _get_global_lock()
        with lock:
            if _shared_data is None:
                if is_multiprocess:
                    _shared_data = manager.dict()
                else:
                    _shared_data = {}
    
    return _shared_data

def get_namespace_object(namespace: str) -> Any:
    """Get an object for specific namespace"""
    global _namespace_objects, is_multiprocess
    
    if _namespace_objects is None:
        lock = _get_global_lock()
        with lock:
            if _namespace_objects is None:
                _namespace_objects = {}
    
    if namespace not in _namespace_objects:
        lock = _get_global_lock()
        with lock:
            if namespace not in _namespace_objects:
                if is_multiprocess:
                    _namespace_objects[namespace] = manager.Value('O', None)
                else:
                    _namespace_objects[namespace] = None
    
    return _namespace_objects[namespace]

def get_namespace_data(namespace: str) -> Dict[str, Any]:
    """get storage space for specific storage type(namespace)"""
    shared_data = get_shared_data()
    lock = _get_global_lock()
    
    if namespace not in shared_data:
        with lock:
            if namespace not in shared_data:
                shared_data[namespace] = {}
    
    return shared_data[namespace]

def get_scan_progress() -> Dict[str, Any]:
    """get storage space for document scanning progress data"""
    return get_namespace_data('scan_progress')
