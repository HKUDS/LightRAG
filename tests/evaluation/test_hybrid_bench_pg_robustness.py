"""PG robustness pins, offline (no live database): workspace-override
fail-fast BEFORE storage init, cleanup that deletes only workspace rows and
retains shared tables, cleanup connections that reuse the storage's
POSTGRES_SSL_* settings, and the lazy PG import on the file-backend path.
"""

from __future__ import annotations

import os
import ssl
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest

from lightrag.evaluation.hybrid_bench import rag_setup, write_amp_driver
from lightrag.evaluation.hybrid_bench.rag_setup import BenchConfig, apply_pg_env
from lightrag.kg.postgres_impl import ClientManager

pytestmark = pytest.mark.offline

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pg_env(monkeypatch, tmp_path):
    """Standard PG recipe in an isolated cwd (no stray config.ini)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "bench")
    monkeypatch.setenv("POSTGRES_PASSWORD", "bench_pw")
    monkeypatch.setenv("POSTGRES_DATABASE", "bench_db")
    monkeypatch.delenv("POSTGRES_WORKSPACE", raising=False)
    monkeypatch.delenv("POSTGRES_SSL_MODE", raising=False)
    return tmp_path


@pytest.mark.parametrize(
    "override", ["tenant_ws", "  padded  "], ids=["plain", "padded"]
)
def test_workspace_override_env_fails_fast(pg_env, monkeypatch, override):
    monkeypatch.setenv("POSTGRES_WORKSPACE", override)
    with pytest.raises(ValueError, match="workspace override"):
        apply_pg_env(BenchConfig(storage="pg"))


def test_workspace_override_config_ini_fails_fast(pg_env):
    (pg_env / "config.ini").write_text(
        "[postgres]\nworkspace = ini_ws\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="workspace override"):
        apply_pg_env(BenchConfig(storage="pg"))


@pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
def test_empty_override_sets_nothing(pg_env, monkeypatch, value):
    monkeypatch.setenv("POSTGRES_WORKSPACE", value)
    assert apply_pg_env(BenchConfig(storage="pg")) is None


def test_file_storage_ignores_the_pg_override(pg_env, monkeypatch):
    monkeypatch.setenv("POSTGRES_WORKSPACE", "tenant_ws")
    assert apply_pg_env(BenchConfig(storage="file")) is None


@pytest.mark.asyncio
async def test_guard_fires_before_any_storage_init(pg_env, monkeypatch):
    constructed = []

    class RefuseLightRAG:
        """Records construction and refuses: the guard must fire first."""

        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise AssertionError("storage init started under a workspace override")

    monkeypatch.setattr(rag_setup, "LightRAG", RefuseLightRAG)
    monkeypatch.setenv("POSTGRES_WORKSPACE", "tenant_ws")
    config = BenchConfig(storage="pg", embedding_backend="fake-hash", embedding_dim=8)
    with pytest.raises(ValueError, match="workspace override"):
        await rag_setup.build_rag(config, llm_func=None)
    assert constructed == []


@pytest.mark.parametrize("source", ["env", "config_ini"])
@pytest.mark.asyncio
async def test_driver_guard_fires_before_corpus_read_and_init(
    pg_env, monkeypatch, source
):
    if source == "env":
        monkeypatch.setenv("POSTGRES_WORKSPACE", "tenant_ws")
    else:
        (pg_env / "config.ini").write_text(
            "[postgres]\nworkspace = ini_ws\n", encoding="utf-8"
        )
    constructed = []

    class RefuseLightRAG:
        """Records construction and refuses: the guard must fire first."""

        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise AssertionError("driver storage init started under an override")

    monkeypatch.setattr(write_amp_driver, "LightRAG", RefuseLightRAG)
    # A missing corpus file proves the guard precedes even the corpus read:
    # without it the failure would be FileNotFoundError, not ValueError.
    with pytest.raises(ValueError, match="workspace override"):
        await write_amp_driver.run(str(pg_env / "missing.jsonl"), "wab_test", 8, 4)
    assert constructed == []


class FakeConn:
    """Recording asyncpg connection: workspace-column probes answer True;
    any OTHER probe (an emptiness check) records itself and raises."""

    def __init__(self, tables):
        self.tables = tables
        self.executed = []
        self.probes = []
        self.closed = False

    async def fetch(self, sql):
        assert "pg_tables" in sql
        return [{"tablename": table} for table in self.tables]

    async def fetchval(self, sql, *args):
        self.probes.append(sql)
        if "information_schema.columns" in sql:
            return True
        raise AssertionError(f"unexpected cleanup probe (race window): {sql}")

    async def execute(self, sql, *args):
        self.executed.append((sql, args))

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("target", ["rag_setup", "driver"])
@pytest.mark.asyncio
async def test_cleanup_deletes_only_workspace_rows_and_retains_tables(
    pg_env, monkeypatch, target
):
    tables = [
        "lightrag_doc_chunks",
        "lightrag_vdb_chunks_writeamp_8d",
        "lightrag_config",
    ]
    conn = FakeConn(tables)
    connect_kwargs = []

    async def fake_connect(**kwargs):
        connect_kwargs.append(kwargs)
        return conn

    monkeypatch.setattr(asyncpg, "connect", fake_connect)
    if target == "rag_setup":
        await rag_setup.cleanup_pg_workspace(BenchConfig(workspace="ws_a"))
    else:
        await write_amp_driver.cleanup("ws_a")
    # A concurrent run's rows survive: every statement is a workspace-bound
    # DELETE; no emptiness probe and no DROP is ever issued, even though the
    # shared vector table would report empty after this run's delete.
    assert conn.executed == [
        (f'DELETE FROM "{table}" WHERE workspace=$1', ("ws_a",)) for table in tables
    ]
    assert all("DROP" not in sql for sql, _ in conn.executed)
    assert len(conn.probes) == len(tables)
    assert conn.closed is True
    assert connect_kwargs[0]["host"] == "localhost"
    assert connect_kwargs[0]["port"] == 5432
    assert connect_kwargs[0]["database"] == "bench_db"


@pytest.mark.parametrize(
    "helper",
    [rag_setup._pg_connect_kwargs, write_amp_driver._connect_kwargs],
    ids=["rag_setup", "driver"],
)
@pytest.mark.parametrize(
    ("ssl_mode", "expected"),
    [
        (None, "absent"),
        ("require", True),
        ("prefer", True),
        ("disable", False),
        ("verify-full", "context"),
    ],
    ids=["unset", "require", "prefer", "disable", "verify-full"],
)
def test_cleanup_connection_reuses_storage_ssl_settings(
    pg_env, monkeypatch, helper, ssl_mode, expected
):
    if ssl_mode is None:
        monkeypatch.delenv("POSTGRES_SSL_MODE", raising=False)
    else:
        monkeypatch.setenv("POSTGRES_SSL_MODE", ssl_mode)
    kwargs = helper(ClientManager.get_config())
    if expected == "absent":
        assert "ssl" not in kwargs
    elif expected == "context":
        assert isinstance(kwargs["ssl"], ssl.SSLContext)
    else:
        assert kwargs["ssl"] is expected


@pytest.mark.asyncio
async def test_cleanup_connection_carries_ssl_end_to_end(pg_env, monkeypatch):
    monkeypatch.setenv("POSTGRES_SSL_MODE", "require")
    conn = FakeConn([])
    connect_kwargs = []

    async def fake_connect(**kwargs):
        connect_kwargs.append(kwargs)
        return conn

    monkeypatch.setattr(asyncpg, "connect", fake_connect)
    await rag_setup.cleanup_pg_workspace(BenchConfig(workspace="ws_a"))
    assert connect_kwargs[0]["ssl"] is True


def test_arms_import_never_requires_asyncpg():
    # A file-backend run must import the arms module without the PG driver
    # stack: block asyncpg in a fresh interpreter and import the module.
    code = (
        "import sys; sys.modules['asyncpg'] = None; "
        "import lightrag.evaluation.hybrid_bench.arms; "
        "print('IMPORT-OK')"
    )
    env = {**os.environ, "PYTHON_DOTENV_DISABLED": "1"}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "IMPORT-OK" in completed.stdout
