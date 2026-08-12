"""Regression tests for the Apache AGE version gate in configure_age_extension.

Apache AGE 1.8.0 changed Cypher ``id()`` to return ``graphid``. That breaks
``PGGraphStorage.get_knowledge_graph`` twice over: the labelled branch fails the
``graphid`` -> ``bigint`` column cast, and the wildcard branch can terminate the
backend with SIGSEGV, taking the whole PostgreSQL instance through crash recovery.
See https://github.com/apache/age/issues/2500.

The gate therefore refuses to start on an affected version. These tests pin the
decision table, because a later change to the parser, to the SQL, or to the
boolean handling would otherwise silently re-open the crashing path.
"""

from __future__ import annotations

import inspect
import re

import pytest

from lightrag.kg.postgres_impl import (
    AGE_ALLOW_UNSUPPORTED_ENV,
    AGE_FIRST_UNSUPPORTED_VERSION,
    PGGraphStorage,
    PostgreSQLDB,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConnection:
    """Fake asyncpg connection serving one ``pg_extension`` / ``pg_available_extensions`` row.

    ``installed`` stands in for ``pg_extension.extversion`` (the SQL script last run
    against this database) and ``available`` for ``pg_available_extensions.default_version``
    (the on-disk ``age.control``, which tracks the loaded binaries). Pass ``row=None``
    to model the lookup returning nothing at all.
    """

    def __init__(
        self, installed=None, available=None, row_missing=False, execute_error=None
    ):
        self.calls: list[dict] = []
        self._row = (
            None
            if row_missing
            else {"installed_version": installed, "available_version": available}
        )
        self._execute_error = execute_error

    async def execute(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        if self._execute_error is not None:
            raise self._execute_error
        return ""

    async def fetchrow(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        return self._row


def _allow_unsupported(monkeypatch) -> None:
    monkeypatch.setenv(AGE_ALLOW_UNSUPPORTED_ENV, "true")


# ---------------------------------------------------------------------------
# Supported versions start normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supported_version_starts(monkeypatch):
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.7.0")

    await PostgreSQLDB.configure_age_extension(conn)

    assert any("CREATE EXTENSION" in c["sql"] for c in conn.calls)


@pytest.mark.asyncio
async def test_create_extension_failure_does_not_raise(monkeypatch):
    """AGE-less deployments must keep starting, as they do without the gate."""
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        execute_error=RuntimeError('extension "age" is not available')
    )

    await PostgreSQLDB.configure_age_extension(conn)

    # The version lookup is never reached, so nothing can raise on its behalf.
    assert len(conn.calls) == 1


# ---------------------------------------------------------------------------
# Unsupported versions are refused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_version_refused(monkeypatch):
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available="1.8.0")

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)


@pytest.mark.asyncio
async def test_newer_binaries_over_stale_catalog_refused(monkeypatch):
    """The in-place binary upgrade: extversion lags, default_version tracks the .so.

    Swapping 1.8.0 binaries over a data directory created under 1.7.0 leaves
    ``pg_extension.extversion`` at '1.7.0' while the 1.8.0 library is what executes.
    Reading only the catalog would let that deployment through, and its
    ``get_knowledge_graph`` is already broken.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.8.0")

    with pytest.raises(RuntimeError) as excinfo:
        await PostgreSQLDB.configure_age_extension(conn)

    # Both values are surfaced so the mixed state is diagnosable from the message.
    assert "installed='1.7.0'" in str(excinfo.value)
    assert "available='1.8.0'" in str(excinfo.value)


@pytest.mark.asyncio
async def test_stale_binaries_under_newer_catalog_refused(monkeypatch):
    """The reverse mismatch is refused too: max() of the two decides."""
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available="1.7.0")

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)


# ---------------------------------------------------------------------------
# Unreadable versions fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"row_missing": True}, id="no-row"),
        pytest.param({"installed": None, "available": None}, id="both-null"),
        pytest.param({"installed": "y.y.y", "available": "y.y.y"}, id="unparseable"),
    ],
)
@pytest.mark.asyncio
async def test_undeterminable_version_fails_closed(monkeypatch, kwargs):
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(**kwargs)

    with pytest.raises(RuntimeError, match="Could not determine"):
        await PostgreSQLDB.configure_age_extension(conn)


@pytest.mark.asyncio
async def test_one_unparseable_source_still_decides(monkeypatch):
    """A single unreadable source must not disable the check on the other."""
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="not-a-version", available="1.8.0")

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)


# ---------------------------------------------------------------------------
# The override reaches every refusal path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"installed": "1.8.0", "available": "1.8.0"}, id="unsupported"),
        pytest.param(
            {"installed": "1.7.0", "available": "1.8.0"}, id="in-place-upgrade"
        ),
        pytest.param({"row_missing": True}, id="no-row"),
        pytest.param({"installed": "y.y.y", "available": "y.y.y"}, id="unparseable"),
    ],
)
@pytest.mark.asyncio
async def test_override_bypasses_every_refusal(monkeypatch, kwargs):
    """Every raise must be reachable by the documented escape hatch.

    AGE reports its version per release rather than per commit, so a source build
    that fixes the crash still reports an unsupported number; without this the gate
    would be a permanent block for anyone on such a build.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(**kwargs)

    await PostgreSQLDB.configure_age_extension(conn)


# ---------------------------------------------------------------------------
# The gate must keep reading both sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_queries_both_version_sources(monkeypatch):
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.7.0")

    await PostgreSQLDB.configure_age_extension(conn)

    lookup = next(c["sql"] for c in conn.calls if "pg_available_extensions" in c["sql"])
    assert "pg_extension" in lookup, lookup
    assert "default_version" in lookup, lookup


def test_first_unsupported_version_is_1_8_0():
    """Raising this bound needs evidence that the newer release was verified."""
    assert AGE_FIRST_UNSUPPORTED_VERSION == (1, 8, 0)


# ---------------------------------------------------------------------------
# _bfs_subgraph must not cast id() back to bigint
# ---------------------------------------------------------------------------


def test_bfs_subgraph_declares_no_bigint_columns():
    """Pin the fix the gate cannot protect: id() results are declared agtype.

    On AGE >= 1.8.0 ``id()`` yields ``graphid``, which a column definition list
    refuses to cast to ``bigint``. Reintroducing such a cast would break
    get_knowledge_graph on those versions, and the gate would not catch it — the
    gate refuses them anyway, and the override path runs straight into it.

    Only the ``AS (...)`` column definition lists are inspected, so the prose in
    the surrounding comments (which names the bigint cast) does not match.
    """
    source = inspect.getsource(PGGraphStorage._bfs_subgraph)
    column_lists = re.findall(r"AS \([^)]*\)", source)

    assert column_lists, "no column definition lists found; did the query shape change?"
    for column_list in column_lists:
        assert "bigint" not in column_list.lower(), column_list
