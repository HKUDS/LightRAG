import os
from multiprocessing.synchronize import Lock as ProcessLock
from threading import Lock as ThreadLock
from multiprocessing import Manager
from typing import Any, Dict, Optional, Union
from lightrag.utils import logger

LockType = Union[ProcessLock, ThreadLock]

is_multiprocess = False

_manager = None
_global_lock: Optional[LockType] = None

# shared data for storage across processes
_shared_dicts: Optional[Dict[str, Any]] = {}
_share_objects: Optional[Dict[str, Any]] = {}
_init_flags: Optional[Dict[str, bool]] = None  # namespace -> initialized

def initialize_share_data():
    """Initialize shared data, only called if multiple processes where workers > 1"""
    global _manager, _shared_dicts, _share_objects, _init_flags, is_multiprocess
    is_multiprocess = True
    
    logger.info(f"Process {os.getpid()} initializing shared storage")
    
    # Initialize manager
    if _manager is None:
        _manager = Manager()
        logger.info(f"Process {os.getpid()} created manager")
    
    # Create shared dictionaries with manager
    _shared_dicts = _manager.dict()
    _share_objects = _manager.dict()
    _init_flags = _manager.dict()  # 使用共享字典存储初始化标志
    logger.info(f"Process {os.getpid()} created shared dictionaries")

def try_initialize_namespace(namespace: str) -> bool:
    """
    尝试初始化命名空间。返回True表示当前进程获得了初始化权限。
    使用共享字典的原子操作确保只有一个进程能成功初始化。
    """
    global _init_flags, _manager
    
    if is_multiprocess:
        if _init_flags is None:
            raise RuntimeError("Shared storage not initialized. Call initialize_share_data() first.")
    else:
        if _init_flags is None:
            _init_flags = {}
    
    logger.info(f"Process {os.getpid()} trying to initialize namespace {namespace}")
    
    # 使用全局锁保护共享字典的访问
    with _get_global_lock():
        # 检查是否已经初始化
        if namespace not in _init_flags:
            # 设置初始化标志
            _init_flags[namespace] = True
            logger.info(f"Process {os.getpid()} ready to initialize namespace {namespace}")
            return True
        
        logger.info(f"Process {os.getpid()} found namespace {namespace} already initialized")
        return False

def _get_global_lock() -> LockType:
    global _global_lock, is_multiprocess, _manager
    
    if _global_lock is None:
        if is_multiprocess:
            _global_lock = _manager.Lock()  # Use manager for lock
        else:
            _global_lock = ThreadLock()
    
    return _global_lock

def get_storage_lock() -> LockType:
    """return storage lock for data consistency"""
    return _get_global_lock()

def get_scan_lock() -> LockType:
    """return scan_progress lock for data consistency"""
    return get_storage_lock()

def get_namespace_object(namespace: str) -> Any:
    """Get an object for specific namespace"""
    global _share_objects, is_multiprocess, _manager
        
    if is_multiprocess and not _manager:
        raise RuntimeError("Multiprocess mode detected but shared storage not initialized. Call initialize_share_data() first.")

    if namespace not in _share_objects:
        lock = _get_global_lock()
        with lock:
            if namespace not in _share_objects:
                if is_multiprocess:
                    _share_objects[namespace] = _manager.Value('O', None)
                else:
                    _share_objects[namespace] = None
    
    return _share_objects[namespace]

# 移除不再使用的函数

def get_namespace_data(namespace: str) -> Dict[str, Any]:
    """get storage space for specific storage type(namespace)"""
    global _shared_dicts, is_multiprocess, _manager
    
    if is_multiprocess and not _manager:
        raise RuntimeError("Multiprocess mode detected but shared storage not initialized. Call initialize_share_data() first.")

    if namespace not in _shared_dicts:
        lock = _get_global_lock()
        with lock:
            if namespace not in _shared_dicts:
                _shared_dicts[namespace] = {}
    
    return _shared_dicts[namespace]

def get_scan_progress() -> Dict[str, Any]:
    """get storage space for document scanning progress data"""
    return get_namespace_data('scan_progress')
