#!/usr/bin/env python
"""
Start LightRAG server with Gunicorn
"""

import os
import sys
import json
import signal
import argparse
from lightrag.api.utils_api import parse_args, display_splash_screen
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data


# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\n\n" + "=" * 80)
    print("RECEIVED TERMINATION SIGNAL")
    print(f"Process ID: {os.getpid()}")
    print("=" * 80 + "\n")

    # Release shared resources
    finalize_share_data()

    # Exit with success status
    sys.exit(0)


def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command
    # Create a parser to handle Gunicorn-specific parameters
    parser = argparse.ArgumentParser(description="Start LightRAG server with Gunicorn")
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes (overrides the default or config.ini setting)",
    )
    parser.add_argument(
        "--timeout", type=int, help="Worker timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Gunicorn log level",
    )

    # Parse Gunicorn-specific arguments
    gunicorn_args, remaining_args = parser.parse_known_args()

    # Pass remaining arguments to LightRAG's parse_args
    sys.argv = [sys.argv[0]] + remaining_args
    args = parse_args()

    # If workers specified, override args value
    if gunicorn_args.workers:
        args.workers = gunicorn_args.workers
        os.environ["WORKERS"] = str(gunicorn_args.workers)

    # If timeout specified, set environment variable
    if gunicorn_args.timeout:
        os.environ["TIMEOUT"] = str(gunicorn_args.timeout)

    # If log-level specified, set environment variable
    if gunicorn_args.log_level:
        os.environ["LOG_LEVEL"] = gunicorn_args.log_level

    # Save all LightRAG args to environment variable for worker processes
    # This is the key step for passing arguments to lightrag_server.py
    os.environ["LIGHTRAG_ARGS"] = json.dumps(vars(args))

    # Display startup information
    display_splash_screen(args)

    print("🚀 Starting LightRAG with Gunicorn")
    print(f"🔄 Worker management: Gunicorn (workers={args.workers})")
    print("🔍 Preloading app: Enabled")
    print("📝 Note: Using Gunicorn's preload feature for shared data initialization")
    print("\n\n" + "=" * 80)
    print("MAIN PROCESS INITIALIZATION")
    print(f"Process ID: {os.getpid()}")
    print(f"Workers setting: {args.workers}")
    print("=" * 80 + "\n")

    # Start application with Gunicorn using direct Python API
    # Ensure WORKERS environment variable is set before importing gunicorn_config
    if args.workers > 1:
        os.environ["WORKERS"] = str(args.workers)

    # Import Gunicorn's StandaloneApplication
    from gunicorn.app.base import BaseApplication

    # Define a custom application class that loads our config
    class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            # Define valid Gunicorn configuration options
            valid_options = {
                "bind",
                "workers",
                "worker_class",
                "timeout",
                "keepalive",
                "preload_app",
                "errorlog",
                "accesslog",
                "loglevel",
                "certfile",
                "keyfile",
                "limit_request_line",
                "limit_request_fields",
                "limit_request_field_size",
                "graceful_timeout",
                "max_requests",
                "max_requests_jitter",
            }

            # Special hooks that need to be set separately
            special_hooks = {
                "on_starting",
                "on_reload",
                "on_exit",
                "pre_fork",
                "post_fork",
                "pre_exec",
                "pre_request",
                "post_request",
                "worker_init",
                "worker_exit",
                "nworkers_changed",
                "child_exit",
            }

            # Import the gunicorn_config module directly
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "gunicorn_config", "gunicorn_config.py"
            )
            self.config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.config_module)

            # Set configuration options
            for key in dir(self.config_module):
                if key in valid_options:
                    value = getattr(self.config_module, key)
                    # Skip functions like on_starting
                    if not callable(value):
                        self.cfg.set(key, value)
                # Set special hooks
                elif key in special_hooks:
                    value = getattr(self.config_module, key)
                    if callable(value):
                        self.cfg.set(key, value)

            # Override with command line arguments if provided
            if gunicorn_args.workers:
                self.cfg.set("workers", gunicorn_args.workers)
            if gunicorn_args.timeout:
                self.cfg.set("timeout", gunicorn_args.timeout)
            if gunicorn_args.log_level:
                self.cfg.set("loglevel", gunicorn_args.log_level)

        def load(self):
            # Import the application
            from lightrag.api.lightrag_server import get_application

            return get_application()

    # Create the application
    app = GunicornApp("")

    # Directly call initialize_share_data with the correct workers value

    # Force workers to be an integer and greater than 1 for multi-process mode
    workers_count = int(args.workers)
    if workers_count > 1:
        # Set a flag to indicate we're in the main process
        os.environ["LIGHTRAG_MAIN_PROCESS"] = "1"
        initialize_share_data(workers_count)
    else:
        initialize_share_data(1)

    # Run the application
    print("\nStarting Gunicorn with direct Python API...")
    app.run()


if __name__ == "__main__":
    main()
