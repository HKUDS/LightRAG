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

# Use Uvicorn worker, subclassed so a worker whose master died stops serving
# instead of holding the listening socket forever (see gunicorn_worker.py).
worker_class = "lightrag.api.gunicorn_worker.LightRAGUvicornWorker"

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

    # Here rather than in a worker: the master is the one process every run
    # has exactly one of, so the deprecation is said once per server start
    # instead of once per worker.
    from lightrag.utils import warn_about_workspace_overrides

    warn_about_workspace_overrides()

    # Claim the configuration directory HERE, in the master, before the fork.
    # The claim is held by the open file description, which forked workers
    # inherit -- so they find it already taken by their own tree and count
    # themselves in, instead of opening a second descriptor and refusing each
    # other. Taken after the fork it would admit exactly one worker.
    #
    # It must be the SAME directory the workers ask for, resolved the same way,
    # or the master's claim is on a path nobody inherits and every worker after
    # the first is refused at startup. Hence the shared resolver rather than a
    # second reading of the environment here.
    from lightrag.config_store import configuration_selection_from_env
    from lightrag.kg.working_dir_lock import (
        acquire_working_dir_lock,
        uses_working_dir,
    )

    config_storage, config_dir = configuration_selection_from_env(
        kv_storage=get_env_value("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        working_dir=get_env_value("WORKING_DIR", "./rag_storage"),
    )
    # The CONFIGURATION storage decides whether a directory is claimed at all;
    # a server-backed one claims nothing.
    if uses_working_dir(config_storage):
        acquire_working_dir_lock(config_dir)

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

    # The master took the directory claim before forking, so the master gives
    # it back -- a worker's own finalize only decrements its inherited count.
    # Resolved through the same helper ``on_starting`` used, so the release
    # cannot name a different directory than the claim.
    from lightrag.config_store import configuration_selection_from_env
    from lightrag.kg.working_dir_lock import (
        release_working_dir_lock,
        uses_working_dir,
    )

    try:
        config_storage, config_dir = configuration_selection_from_env(
            kv_storage=get_env_value("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
            working_dir=get_env_value("WORKING_DIR", "./rag_storage"),
        )
    except ValueError:
        # An unusable selection refused in ``on_starting``, so nothing was
        # claimed and there is nothing to give back.
        config_storage, config_dir = "", ""
    if config_storage and uses_working_dir(config_storage):
        release_working_dir_lock(config_dir)

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
