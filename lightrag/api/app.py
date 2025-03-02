from dotenv import load_dotenv
from lightrag.api.lightrag_server import (
    create_app,
    display_splash_screen
)
from lightrag.api.new.new_utils_api import parse_args

# Load environment variables
load_dotenv(override=True)



args = parse_args()
app = create_app(args)


def main():
    import uvicorn

    display_splash_screen(args)
    uvicorn_config = {
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "reload": args.reload,
    }
    if args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }
        )
    uvicorn.run("lightrag.api.app:app", **uvicorn_config)


if __name__ == "__main__":
    main()
