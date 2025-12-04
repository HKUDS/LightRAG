# gunicorn_config.py
import os
import logging
from lightrag.kg.shared_storage import finalize_share_data
from lightrag.utils import setup_logger, get_env_value
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
)


# Get log directory path from environment variable
log_dir = os.getenv("LOG_DIR", os.getcwd())
log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

# Ensure log directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Get log file max size and backup count from environment variables
log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

# These variables will be set by run_with_gunicorn.py
workers = None
bind = None
loglevel = None
certfile = None
keyfile = None

# Enable preload_app option
preload_app = True

# Use Uvicorn worker
worker_class = "uvicorn.workers.UvicornWorker"

# Other Gunicorn configurations

# Logging configuration
errorlog = os.getenv("ERROR_LOG", log_file_path)  # Default write to lightrag.log
accesslog = os.getenv("ACCESS_LOG", log_file_path)  # Default write to lightrag.log

logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": log_file_path,
            "maxBytes": log_max_bytes,
            "backupCount": log_backup_count,
            "encoding": "utf8",
        },
    },
    "filters": {
        "path_filter": {
            "()": "lightrag.utils.LightragPathFilter",
        },
    },
    "loggers": {
        "lightrag": {
            "handlers": ["console", "file"],
            "level": loglevel.upper() if loglevel else "INFO",
            "propagate": False,
        },
        "gunicorn": {
            "handlers": ["console", "file"],
            "level": loglevel.upper() if loglevel else "INFO",
            "propagate": False,
        },
        "gunicorn.error": {
            "handlers": ["console", "file"],
            "level": loglevel.upper() if loglevel else "INFO",
            "propagate": False,
        },
        "gunicorn.access": {
            "handlers": ["console", "file"],
            "level": loglevel.upper() if loglevel else "INFO",
            "propagate": False,
            "filters": ["path_filter"],
        },
    },
}


def on_starting(server):
    """
    Executed when Gunicorn starts, before forking the first worker processes
    You can use this function to do more initialization tasks for all processes
    """
    print("=" * 80)
    print(f"GUNICORN MASTER PROCESS: on_starting jobs for {workers} worker(s)")
    print(f"Process ID: {os.getpid()}")
    print("=" * 80)

    # Memory usage monitoring
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        msg = (
            f"Memory usage after initialization: {memory_info.rss / 1024 / 1024:.2f} MB"
        )
        print(msg)
    except ImportError:
        print("psutil not installed, skipping memory usage reporting")

    # Log the location of the LightRAG log file
    print(f"LightRAG log file: {log_file_path}\n")

    print("Gunicorn initialization complete, forking workers...\n")


def on_exit(server):
    """
    Executed when Gunicorn is shutting down.
    This is a good place to release shared resources.
    """
    print("=" * 80)
    print("GUNICORN MASTER PROCESS: Shutting down")
    print(f"Process ID: {os.getpid()}")

    print("Finalizing shared storage...")
    finalize_share_data()

    print("Gunicorn shutdown complete")
    print("=" * 80)

    print("=" * 80)

    print("=" * 80)
    print("Gunicorn shutdown complete")
    print("=" * 80)


def post_fork(server, worker):
    """
    Executed after a worker has been forked.
    This is a good place to set up worker-specific configurations.
    """
    # Set up main loggers
    log_level = loglevel.upper() if loglevel else "INFO"
    setup_logger("uvicorn", log_level, add_filter=False, log_file_path=log_file_path)
    setup_logger(
        "uvicorn.access", log_level, add_filter=True, log_file_path=log_file_path
    )
    setup_logger("lightrag", log_level, add_filter=True, log_file_path=log_file_path)

    # Set up lightrag submodule loggers
    for name in logging.root.manager.loggerDict:
        if name.startswith("lightrag."):
            setup_logger(name, log_level, add_filter=True, log_file_path=log_file_path)

    # Disable uvicorn.error logger
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers = []
    uvicorn_error_logger.setLevel(logging.CRITICAL)
    uvicorn_error_logger.propagate = False


def worker_exit(server, worker):
    """
    Executed when a worker is about to exit.

    NOTE: When using UvicornWorker (worker_class = "uvicorn.workers.UvicornWorker"),
    this hook may NOT be called reliably. UvicornWorker has its own lifecycle
    management that prioritizes ASGI lifespan shutdown events.

    The primary cleanup mechanism is handled by:
    1. FastAPI lifespan context manager with GUNICORN_CMD_ARGS check (in lightrag_server.py)
       - Workers skip cleanup when GUNICORN_CMD_ARGS is set
    2. on_exit() hook for main process cleanup

    This function serves as a defensive fallback for:
    - Non-UvicornWorker scenarios
    - Future Gunicorn/Uvicorn behavior changes
    - Additional safety layer

    When called, we should only clean up worker-local resources, NOT the shared Manager.
    The Manager should only be shut down by the main process in on_exit().
    """
    print("=" * 80)
    print(f"GUNICORN WORKER PROCESS: Shutting down worker {worker.pid}")
    print(f"Process ID: {os.getpid()}")
    print("=" * 80)

    # Clean up worker-local resources without shutting down the Manager
    # Pass shutdown_manager=False to prevent Manager shutdown
    finalize_share_data(shutdown_manager=False)

    print(f"Worker {worker.pid} cleanup complete")
    print("=" * 80)
