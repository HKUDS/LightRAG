# gunicorn_config.py
import os
import multiprocessing
from lightrag.kg.shared_storage import finalize_share_data
from lightrag.api.utils_api import parse_args

# Parse command line arguments
args = parse_args()

# Determine worker count - from environment variable or command line arguments
workers = int(os.getenv('WORKERS', args.workers))

# If not specified, use CPU count * 2 + 1 (Gunicorn recommended configuration)
if workers <= 1:
    workers = multiprocessing.cpu_count() * 2 + 1

# Binding address
bind = f"{os.getenv('HOST', args.host)}:{os.getenv('PORT', args.port)}"

# Enable preload_app option
preload_app = True

# Use Uvicorn worker
worker_class = "uvicorn.workers.UvicornWorker"

# Other Gunicorn configurations
timeout = int(os.getenv('TIMEOUT', 120))
keepalive = 5

# Optional SSL configuration
if args.ssl:
    certfile = args.ssl_certfile
    keyfile = args.ssl_keyfile

# Logging configuration
errorlog = os.getenv('ERROR_LOG', '-')  # '-' means stderr
accesslog = os.getenv('ACCESS_LOG', '-')  # '-' means stderr
loglevel = os.getenv('LOG_LEVEL', 'info')

def on_starting(server):
    """
    Executed when Gunicorn starts, before forking the first worker processes
    You can use this function to do more initialization tasks for all processes
    """
    print("=" * 80)
    print(f"GUNICORN MASTER PROCESS: on_starting jobs for all {workers} workers")
    print(f"Process ID: {os.getpid()}")
    print("=" * 80)
    
    # Memory usage monitoring
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        msg = f"Memory usage after initialization: {memory_info.rss / 1024 / 1024:.2f} MB"
        print(msg)
    except ImportError:
        print("psutil not installed, skipping memory usage reporting")
    
    print("=" * 80)
    print("Gunicorn initialization complete, forking workers...")
    print("=" * 80)


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
