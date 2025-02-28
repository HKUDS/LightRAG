# gunicorn_config.py
import os
import logging
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
from lightrag.kg.shared_storage import finalize_share_data
from lightrag.api.utils_api import parse_args

# Parse command line arguments
args = parse_args()

# Determine worker count - from environment variable or command line arguments
workers = int(os.getenv("WORKERS", args.workers))

# Binding address
bind = f"{os.getenv('HOST', args.host)}:{os.getenv('PORT', args.port)}"

# Enable preload_app option
preload_app = True

# Use Uvicorn worker
worker_class = "uvicorn.workers.UvicornWorker"

# Other Gunicorn configurations
timeout = int(os.getenv("TIMEOUT", 120))
keepalive = 5

# Optional SSL configuration
if args.ssl:
    certfile = args.ssl_certfile
    keyfile = args.ssl_keyfile

# 获取日志文件路径
log_file_path = os.path.abspath(os.path.join(os.getcwd(), "lightrag.log"))

# Logging configuration
errorlog = os.getenv("ERROR_LOG", log_file_path)  # 默认写入到 lightrag.log
accesslog = os.getenv("ACCESS_LOG", log_file_path)  # 默认写入到 lightrag.log
loglevel = os.getenv("LOG_LEVEL", "info")

# 配置日志系统
logconfig_dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'filename': log_file_path,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
    },
    'loggers': {
        'lightrag': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'gunicorn': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'gunicorn.error': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
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

    print("Gunicorn initialization complete, forking workers...\n")


def on_exit(server):
    """
    Executed when Gunicorn is shutting down.
    This is a good place to release shared resources.
    """
    print("=" * 80)
    print("GUNICORN MASTER PROCESS: Shutting down")
    print(f"Process ID: {os.getpid()}")
    print("=" * 80)

    # Release shared resources
    finalize_share_data()

    print("=" * 80)
    print("Gunicorn shutdown complete")
    print("=" * 80)
