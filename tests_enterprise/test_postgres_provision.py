from __future__ import annotations

from pathlib import Path

import pytest

from lightrag_enterprise.system.provision_postgres import (
    PostgresProvisionConfig,
    build_postgres_url,
    provision_postgres,
    quote_identifier,
    redact_database_url,
    replace_database,
    write_env_file,
)


class FakeConnection:
    def __init__(self, connector):
        self.connector = connector

    async def execute(self, sql, *args):
        self.connector.executed.append((sql, args))
        return "OK"

    async def fetchval(self, sql, *args):
        self.connector.queries.append((sql, args))
        if "pg_roles" in sql:
            return self.connector.role_exists
        if "pg_database" in sql:
            return self.connector.database_exists
        return 1

    async def close(self):
        self.connector.closed += 1


class FakeConnector:
    def __init__(self, *, role_exists=False, database_exists=False):
        self.role_exists = role_exists
        self.database_exists = database_exists
        self.urls = []
        self.executed = []
        self.queries = []
        self.closed = 0

    async def connect(self, url):
        self.urls.append(url)
        return FakeConnection(self)


@pytest.mark.asyncio
async def test_provision_uses_existing_database_url_without_admin_connection():
    connector = FakeConnector()
    schema_calls = []

    async def schema_runner(url):
        schema_calls.append(url)
        return True

    result = await provision_postgres(
        PostgresProvisionConfig(
            database_url="postgresql://app:secret@localhost:5432/lightrag_e2e"
        ),
        connector=connector,
        schema_runner=schema_runner,
    )

    assert result.mode == "existing"
    assert result.schema_applied is True
    assert (
        result.redacted_database_url
        == "postgresql://app:***@localhost:5432/lightrag_e2e"
    )
    assert connector.urls == ["postgresql://app:secret@localhost:5432/lightrag_e2e"]
    assert schema_calls == ["postgresql://app:secret@localhost:5432/lightrag_e2e"]


@pytest.mark.asyncio
async def test_provision_creates_role_database_grants_and_runs_schema():
    connector = FakeConnector(role_exists=False, database_exists=False)
    schema_calls = []

    async def schema_runner(url):
        schema_calls.append(url)
        return True

    result = await provision_postgres(
        PostgresProvisionConfig(
            admin_url="postgresql://postgres:admin@localhost:5432/postgres",
            database="little_bull_e2e",
            app_user="little_bull",
            app_password="app-secret",
        ),
        connector=connector,
        schema_runner=schema_runner,
    )

    executed_sql = [sql for sql, _ in connector.executed]
    assert result.mode == "created"
    assert result.created_database is True
    assert result.created_role is True
    assert "CREATE ROLE \"little_bull\" LOGIN PASSWORD 'app-secret'" in executed_sql
    assert 'CREATE DATABASE "little_bull_e2e" OWNER "little_bull"' in executed_sql
    assert any("GRANT ALL ON SCHEMA public" in sql for sql in executed_sql)
    assert connector.urls == [
        "postgresql://postgres:admin@localhost:5432/postgres",
        "postgresql://postgres:admin@localhost:5432/little_bull_e2e",
        "postgresql://little_bull:app-secret@localhost:5432/little_bull_e2e",
    ]
    assert schema_calls == [
        "postgresql://little_bull:app-secret@localhost:5432/little_bull_e2e"
    ]


@pytest.mark.asyncio
async def test_provision_links_existing_database_without_recreating_role_or_database():
    connector = FakeConnector(role_exists=True, database_exists=True)

    async def schema_runner(_url):
        return False

    result = await provision_postgres(
        PostgresProvisionConfig(
            admin_url="postgresql://postgres@localhost:5432/postgres",
            database="little_bull_e2e",
            app_user="postgres",
            run_schema=False,
        ),
        connector=connector,
        schema_runner=schema_runner,
    )

    executed_sql = [sql for sql, _ in connector.executed]
    assert result.mode == "existing-admin"
    assert result.created_database is False
    assert result.created_role is False
    assert not any(sql.startswith("CREATE ROLE") for sql in executed_sql)
    assert not any(sql.startswith("CREATE DATABASE") for sql in executed_sql)
    assert result.schema_applied is False


def test_url_helpers_redact_and_replace_database():
    url = build_postgres_url(
        user="little bull",
        password="p@ss/word",
        host="localhost",
        port=5432,
        database="db one",
    )

    assert url == "postgresql://little%20bull:p%40ss%2Fword@localhost:5432/db%20one"
    assert (
        redact_database_url(url)
        == "postgresql://little%20bull:***@localhost:5432/db%20one"
    )
    assert replace_database(url, "db two") == (
        "postgresql://little%20bull:p%40ss%2Fword@localhost:5432/db%20two"
    )


@pytest.mark.parametrize("identifier", ["", "Upper", "bad-name", 'safe"name', "x;drop"])
def test_quote_identifier_rejects_unsafe_identifier(identifier):
    with pytest.raises(ValueError):
        quote_identifier(identifier)


def test_quote_identifier_accepts_safe_identifier():
    assert quote_identifier("safe_name_1") == '"safe_name_1"'


def test_write_env_file_preserves_existing_lines_and_upserts(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nLIGHTRAG_SYSTEM_DATABASE_URL=old\n", encoding="utf-8")

    write_env_file(
        env_file,
        "postgresql://app:secret@localhost:5432/little_bull_e2e",
        functional_enabled=True,
    )

    content = env_file.read_text(encoding="utf-8").splitlines()
    assert content == [
        "FOO=bar",
        "LIGHTRAG_SYSTEM_DATABASE_URL=postgresql://app:secret@localhost:5432/little_bull_e2e",
        "LITTLE_BULL_FUNCTIONAL_ENABLED=true",
    ]
