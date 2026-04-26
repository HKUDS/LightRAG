import argparse
import getpass

from lightrag.api.passwords import hash_password


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a bcrypt password value for AUTH_ACCOUNTS."
    )
    parser.add_argument(
        "password",
        nargs="?",
        help="Password to hash. If omitted, a secure prompt is used.",
    )
    parser.add_argument(
        "--username",
        help="Optional username. When provided, output is ready to paste into AUTH_ACCOUNTS.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    password = args.password or getpass.getpass("Password: ")
    if not password:
        parser.error("password cannot be empty")

    hashed_password = hash_password(password)
    if args.username:
        print(f"{args.username}:{hashed_password}")
    else:
        print(hashed_password)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
