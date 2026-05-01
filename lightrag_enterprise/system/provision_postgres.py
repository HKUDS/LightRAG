from __future__ import annotations

import argparse
import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from .db import get_database_url, run_schema


DEFAULT_DATABASE = "lightrag_little_bull"
DEFAULT_ADMIN_DATABASE = "postgres"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_ADMIN_USER = "postgres"
SAFE_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def redact_database_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.password:
        return url
    username = quote(unquote(parts.username or ""), safe="")
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{username}:***@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def replace_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    path = f"/{quote(database, safe='')}"
    return urlunsplit(
        (parts.scheme or "postgresql", parts.netloc, path, parts.query, parts.fragment)
    )


def build_postgres_url(
    *,
    user: str,
    password: str | None,
    host: str,
    port: int,
    database: str,
) -> str:
    auth = quote(user, safe="")
    if password is not None:
        auth = f"{auth}:{quote(password, safe='')}"
    return f"postgresql://{auth}@{host}:{port}/{quote(database, safe='')}"


def quote_identifier(identifier: str) -> str:
    if not SAFE_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError("Postgres identifier must match ^[a-z_][a-z0-9_]{0,62}$")
    return '"' + identifier.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    if "\x00" in value:
        raise ValueError("Postgres literal cannot contain NUL")
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class PostgresProvisionConfig:
    database_url: str | None = None
    admin_url: str | None = None
    database: str = DEFAULT_DATABASE
    app_user: str | None = None
    app_password: str | None = None
    run_schema: bool = True
    write_env: Path | None = None
    functional_enabled: bool = True


@dataclass
class PostgresProvisionResult:
    database_url: str
    mode: str
    created_database: bool = False
    created_role: bool = False
    schema_applied: bool = False
    env_file_written: Path | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def redacted_database_url(self) -> str:
        return redact_database_url(self.database_url)


class AsyncpgConnector:
    async def connect(self, url: str) -> Any:
        import asyncpg

        return await asyncpg.connect(url)


SchemaRunner = Callable[[str | None], Awaitable[bool]]


def config_from_env(args: argparse.Namespace | None = None) -> PostgresProvisionConfig:
    args = args or argparse.Namespace()
    database_url = getattr(args, "database_url", None) or get_database_url()
    admin_url = getattr(args, "admin_url", None) or os.getenv(
        "LIGHTRAG_SYSTEM_POSTGRES_ADMIN_URL"
    )
    database = (
        getattr(args, "database", None)
        or os.getenv("LIGHTRAG_SYSTEM_POSTGRES_DATABASE")
        or DEFAULT_DATABASE
    )
    app_user = getattr(args, "app_user", None) or os.getenv(
        "LIGHTRAG_SYSTEM_POSTGRES_USER"
    )
    app_password = getattr(args, "app_password", None) or os.getenv(
        "LIGHTRAG_SYSTEM_POSTGRES_PASSWORD"
    )
    run_schema_enabled = not getattr(args, "no_schema", False) and not _truthy(
        os.getenv("LIGHTRAG_SYSTEM_POSTGRES_SKIP_SCHEMA")
    )
    write_env = getattr(args, "write_env", None)
    return PostgresProvisionConfig(
        database_url=database_url,
        admin_url=admin_url,
        database=database,
        app_user=app_user,
        app_password=app_password,
        run_schema=run_schema_enabled,
        write_env=Path(write_env) if write_env else None,
        functional_enabled=not getattr(args, "no_functional_enabled", False),
    )


def admin_url_from_env(args: argparse.Namespace | None = None) -> str:
    args = args or argparse.Namespace()
    if getattr(args, "admin_url", None):
        return args.admin_url
    if os.getenv("LIGHTRAG_SYSTEM_POSTGRES_ADMIN_URL"):
        return os.environ["LIGHTRAG_SYSTEM_POSTGRES_ADMIN_URL"]
    return build_postgres_url(
        user=getattr(args, "admin_user", None)
        or os.getenv("LIGHTRAG_SYSTEM_POSTGRES_ADMIN_USER")
        or DEFAULT_ADMIN_USER,
        password=getattr(args, "admin_password", None)
        or os.getenv("LIGHTRAG_SYSTEM_POSTGRES_ADMIN_PASSWORD"),
        host=getattr(args, "host", None)
        or os.getenv("LIGHTRAG_SYSTEM_POSTGRES_HOST")
        or DEFAULT_HOST,
        port=int(
            getattr(args, "port", None)
            or os.getenv("LIGHTRAG_SYSTEM_POSTGRES_PORT")
            or DEFAULT_PORT
        ),
        database=getattr(args, "admin_database", None)
        or os.getenv("LIGHTRAG_SYSTEM_POSTGRES_ADMIN_DATABASE")
        or DEFAULT_ADMIN_DATABASE,
    )


async def _close_quietly(conn: Any) -> None:
    close = getattr(conn, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


async def _assert_connectable(connector: Any, url: str) -> None:
    conn = await connector.connect(url)
    try:
        await conn.execute("SELECT 1")
    finally:
        await _close_quietly(conn)


async def _role_exists(conn: Any, role: str) -> bool:
    return bool(await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = $1", role))


async def _database_exists(conn: Any, database: str) -> bool:
    return bool(
        await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database)
    )


async def provision_postgres(
    config: PostgresProvisionConfig,
    *,
    connector: Any | None = None,
    schema_runner: SchemaRunner = run_schema,
) -> PostgresProvisionResult:
    connector = connector or AsyncpgConnector()
    if config.database_url:
        await _assert_connectable(connector, config.database_url)
        schema_applied = (
            await schema_runner(config.database_url) if config.run_schema else False
        )
        result = PostgresProvisionResult(
            database_url=config.database_url,
            mode="existing",
            schema_applied=schema_applied,
            messages=["Using existing LIGHTRAG_SYSTEM_DATABASE_URL/DATABASE_URL."],
        )
        if config.write_env:
            write_env_file(
                config.write_env,
                result.database_url,
                functional_enabled=config.functional_enabled,
            )
            result.env_file_written = config.write_env
        return result

    admin_url = config.admin_url or admin_url_from_env()
    database = config.database
    app_user = config.app_user or unquote(
        urlsplit(admin_url).username or DEFAULT_ADMIN_USER
    )
    app_password = config.app_password
    app_url = build_postgres_url(
        user=app_user,
        password=app_password
        if app_password is not None
        else urlsplit(admin_url).password,
        host=urlsplit(admin_url).hostname or DEFAULT_HOST,
        port=urlsplit(admin_url).port or DEFAULT_PORT,
        database=database,
    )

    admin_conn = await connector.connect(admin_url)
    created_role = False
    created_database = False
    try:
        if app_password and not await _role_exists(admin_conn, app_user):
            await admin_conn.execute(
                f"CREATE ROLE {quote_identifier(app_user)} LOGIN PASSWORD {quote_literal(app_password)}"
            )
            created_role = True
        if not await _database_exists(admin_conn, database):
            await admin_conn.execute(
                f"CREATE DATABASE {quote_identifier(database)} OWNER {quote_identifier(app_user)}"
            )
            created_database = True
    finally:
        await _close_quietly(admin_conn)

    database_admin_url = replace_database(admin_url, database)
    db_conn = await connector.connect(database_admin_url)
    try:
        await db_conn.execute(
            f"GRANT ALL PRIVILEGES ON DATABASE {quote_identifier(database)} TO {quote_identifier(app_user)}"
        )
        await db_conn.execute(
            f"GRANT ALL ON SCHEMA public TO {quote_identifier(app_user)}"
        )
        await db_conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {quote_identifier(app_user)}"
        )
        await db_conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {quote_identifier(app_user)}"
        )
    finally:
        await _close_quietly(db_conn)

    await _assert_connectable(connector, app_url)
    schema_applied = await schema_runner(app_url) if config.run_schema else False
    result = PostgresProvisionResult(
        database_url=app_url,
        mode="created" if created_database else "existing-admin",
        created_database=created_database,
        created_role=created_role,
        schema_applied=schema_applied,
        messages=[
            "Created or linked a dedicated Little Bull PostgreSQL database.",
            "Set LIGHTRAG_SYSTEM_DATABASE_URL to the returned URL before starting the API.",
        ],
    )
    if config.write_env:
        write_env_file(
            config.write_env,
            result.database_url,
            functional_enabled=config.functional_enabled,
        )
        result.env_file_written = config.write_env
    return result


def write_env_file(
    path: Path, database_url: str, *, functional_enabled: bool = True
) -> None:
    updates = {"LIGHTRAG_SYSTEM_DATABASE_URL": database_url}
    if functional_enabled:
        updates["LITTLE_BULL_FUNCTIONAL_ENABLED"] = "true"
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = (
            stripped.split("=", 1)[0]
            if "=" in stripped and not stripped.startswith("#")
            else None
        )
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision or link the Little Bull system PostgreSQL database."
    )
    parser.add_argument(
        "--database-url", help="Existing system database URL to validate and migrate."
    )
    parser.add_argument(
        "--admin-url",
        help="Admin PostgreSQL URL used to create/link the dedicated database.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Admin PostgreSQL host when --admin-url is not supplied.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Admin PostgreSQL port when --admin-url is not supplied.",
    )
    parser.add_argument(
        "--admin-user",
        default=None,
        help="Admin PostgreSQL user when --admin-url is not supplied.",
    )
    parser.add_argument(
        "--admin-password",
        default=None,
        help="Admin PostgreSQL password when --admin-url is not supplied.",
    )
    parser.add_argument(
        "--admin-database", default=None, help="Admin maintenance database."
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Dedicated Little Bull database to create or link.",
    )
    parser.add_argument(
        "--app-user", default=None, help="Dedicated application role to create or use."
    )
    parser.add_argument(
        "--app-password",
        default=None,
        help="Password for the dedicated application role.",
    )
    parser.add_argument(
        "--write-env",
        default=None,
        help="Optional .env path to update with LIGHTRAG_SYSTEM_DATABASE_URL.",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Do not run the Little Bull system schema.",
    )
    parser.add_argument(
        "--no-functional-enabled",
        action="store_true",
        help="Do not write LITTLE_BULL_FUNCTIONAL_ENABLED=true when --write-env is used.",
    )
    return parser


async def _main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = config_from_env(args)
    if not config.database_url and not config.admin_url:
        config = PostgresProvisionConfig(
            **{**config.__dict__, "admin_url": admin_url_from_env(args)}
        )
    try:
        result = await provision_postgres(config)
    except Exception as exc:
        parser.exit(1, f"Little Bull Postgres provisioning failed: {exc}\n")
    print(f"mode={result.mode}")
    print(f"database_url={result.redacted_database_url}")
    print(f"created_database={str(result.created_database).lower()}")
    print(f"created_role={str(result.created_role).lower()}")
    print(f"schema_applied={str(result.schema_applied).lower()}")
    if result.env_file_written:
        print(f"env_file_written={result.env_file_written}")
    for message in result.messages:
        print(f"note={message}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
