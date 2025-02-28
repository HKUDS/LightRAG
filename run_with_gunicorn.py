#!/usr/bin/env python
"""
Start LightRAG server with Gunicorn
"""

import os
import sys
import signal
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

    # Parse all arguments using parse_args
    args = parse_args(is_uvicorn_mode=False)

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

            # Import and configure the gunicorn_config module
            import gunicorn_config

            # Set configuration variables in gunicorn_config
            gunicorn_config.workers = int(os.getenv("WORKERS", args.workers))
            gunicorn_config.bind = (
                f"{os.getenv('HOST', args.host)}:{os.getenv('PORT', args.port)}"
            )
            gunicorn_config.loglevel = (
                args.log_level.lower()
                if args.log_level
                else os.getenv("LOG_LEVEL", "info")
            )

            # Set SSL configuration if enabled
            if args.ssl:
                gunicorn_config.certfile = args.ssl_certfile
                gunicorn_config.keyfile = args.ssl_keyfile

            # Set configuration options from the module
            for key in dir(gunicorn_config):
                if key in valid_options:
                    value = getattr(gunicorn_config, key)
                    # Skip functions like on_starting and None values
                    if not callable(value) and value is not None:
                        self.cfg.set(key, value)
                # Set special hooks
                elif key in special_hooks:
                    value = getattr(gunicorn_config, key)
                    if callable(value):
                        self.cfg.set(key, value)

            if hasattr(gunicorn_config, "logconfig_dict"):
                self.cfg.set(
                    "logconfig_dict", getattr(gunicorn_config, "logconfig_dict")
                )

        def load(self):
            # Import the application
            from lightrag.api.lightrag_server import get_application

            return get_application(args)

    # Create the application
    app = GunicornApp("")

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
