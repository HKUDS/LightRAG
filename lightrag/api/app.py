import os
from dotenv import load_dotenv
import uvicorn


from lightrag.api.lightrag_server import (
    check_and_install_dependencies,
    configure_logging,
    create_app,
    display_splash_screen
)
from lightrag.api.utils_api import parse_args

# Load environment variables
load_dotenv(override=True)


def main():
    # Check if running under Gunicorn
    if "GUNICORN_CMD_ARGS" in os.environ:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check and install dependencies
    check_and_install_dependencies()

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    configure_logging()

    args = parse_args(is_uvicorn_mode=True)
    display_splash_screen(args)

    # Create application instance directly instead of using factory function
    app = create_app(args)
    
    uvicorn_config = {
        "app": app, 
        "host": args.host,
        "port": args.port
    }
    if args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }
        )
        
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
