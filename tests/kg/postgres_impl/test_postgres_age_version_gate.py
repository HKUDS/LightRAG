"""Regression tests for the Apache AGE version gate in configure_age_extension.

Apache AGE 1.8.0 changed Cypher ``id()`` to return ``graphid``. That breaks
``PGGraphStorage.get_knowledge_graph`` twice over: the labelled branch fails the
``graphid`` -> ``bigint`` column cast, and the wildcard branch can terminate the
backend with SIGSEGV, taking the whole PostgreSQL instance through crash recovery.
See https://github.com/apache/age/issues/2500.

The gate therefore refuses to start on an affected version. These tests pin the
decision table, because a later change to the parser, to the queries, or to the
boolean handling would otherwise silently re-open the crashing path.

The two version sources are read by two independent queries -- ``AGE_INSTALLED_SQL``
against ``pg_extension`` and ``AGE_AVAILABLE_SQL`` against ``pg_available_extensions``
-- so ``_FakeConnection`` dispatches on the SQL it is handed. That dispatch is
load-bearing: it is what makes "which query supplied which value" observable, and
therefore what makes swapping the two sources a test failure rather than an
invisible change.

The two are also read by different asyncpg methods, and the fake insists on that.
``fetchval`` collapses "no row" and "a row whose column is NULL" into one ``None``,
which for ``default_version`` are different deployments: the first means AGE is not on
disk, the second means a control file is there and names no version. The AVAILABLE
lookup therefore goes through ``fetchrow``, and the fake refuses to serve it any other
way, so a regression to ``fetchval`` -- which is what let a present-but-versionless
control file be read as "AGE is absent" -- fails here.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re

import pytest

from lightrag.kg.postgres_impl import (
    _AGE_VERSION_UNPARSEABLE,
    _AgeVersionUnparseable,
    AGE_ALLOW_UNSUPPORTED_ENV,
    AGE_AVAILABLE_SQL,
    AGE_FIRST_UNSUPPORTED_VERSION,
    AGE_INSTALLED_SQL,
    PGGraphStorage,
    PostgreSQLDB,
    TRANSIENT_DB_EXCEPTIONS,
    _parse_age_version,
)
from lightrag.utils import logger as lightrag_logger

# postgres_impl imports asyncpg unconditionally at module scope (installing it via
# pipmaster if missing), so by the time the import above has succeeded asyncpg is
# importable. A plain import here therefore adds no new dependency on this module --
# it cannot fail unless the import above already did.
import asyncpg

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The failure shape that makes the AVAILABLE lookup fragile in the first place:
# ``pg_available_extensions`` parses *every* control file in the sharedir, so one
# unrelated extension shipping a malformed .control makes the whole view raise. That
# says nothing about AGE either way -- which is the point: a lookup that never
# answered is not evidence that the binaries are safe, so it cannot be waved through.
_MALFORMED_CONTROL_FILE_ERROR = asyncpg.exceptions.PostgresSyntaxError(
    'syntax error in file "/usr/share/postgresql/18/extension/other.control" line 1'
)


class _NullDefaultVersionRow:
    """Marker: ``pg_available_extensions`` has an ``age`` row reporting NULL.

    A distinct state from having no row at all. The column is nullable, so a control
    file can be present in the sharedir and name no version -- AGE is then on disk,
    with binaries that will load, and the catalog has told us nothing about which ones.
    Passed as ``available=`` to ``_FakeConnection``, where ``None`` keeps its older and
    narrower meaning of "no such row".
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<age row with default_version NULL>"


_NULL_DEFAULT_VERSION = _NullDefaultVersionRow()


class _FakeConnection:
    """Fake asyncpg connection answering the two AGE catalog lookups separately.

    ``installed`` is what ``AGE_INSTALLED_SQL`` returns (``pg_extension.extversion``,
    the version of the SQL script last run against this database) and ``available``
    what ``AGE_AVAILABLE_SQL`` reports (``pg_available_extensions.default_version``,
    read from the on-disk ``age.control``, which tracks the loaded binaries).

    Dispatch is on the SQL text, compared against the real module constants rather
    than retyped literals, so that a value can only reach the gate through the query
    that genuinely selects it. ``installed_error`` / ``available_error`` inject a
    failure into exactly one of the two lookups; an unrecognised query is an error
    rather than a silent ``None``, so a renamed or dropped constant surfaces here.

    The AVAILABLE source has four expressible states, because the gate must tell them
    apart and once could not:

    ==========================  =======================================
    ``available=``              ``fetchrow`` returns
    ==========================  =======================================
    ``None``                    ``None`` -- no ``age`` row at all
    ``_NULL_DEFAULT_VERSION``   ``{"default_version": None}``
    any other value             ``{"default_version": <that value>}``
    (``available_error=...``)   raises -- the lookup never answered
    ==========================  =======================================

    Dispatch is also on the *method*: ``AGE_INSTALLED_SQL`` may only be read with
    ``fetchval`` and ``AGE_AVAILABLE_SQL`` only with ``fetchrow``. The middle two rows
    above are indistinguishable through ``fetchval``, so pinning the method is what
    keeps them distinguishable at all.
    """

    def __init__(
        self,
        installed=None,
        available=None,
        execute_error=None,
        installed_error=None,
        available_error=None,
    ):
        self.calls: list[dict] = []
        self._installed = installed
        self._available = available
        self._errors = {
            AGE_INSTALLED_SQL: installed_error,
            AGE_AVAILABLE_SQL: available_error,
        }
        self._execute_error = execute_error

    async def execute(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        if self._execute_error is not None:
            raise self._execute_error
        return ""

    async def fetchval(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        if sql != AGE_INSTALLED_SQL:
            raise AssertionError(f"unexpected fetchval query: {sql!r}")
        self._maybe_raise(sql)
        return self._installed

    async def fetchrow(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        if sql != AGE_AVAILABLE_SQL:
            raise AssertionError(f"unexpected fetchrow query: {sql!r}")
        self._maybe_raise(sql)
        if self._available is None:
            # No row: nothing in the sharedir names 'age'.
            return None
        if self._available is _NULL_DEFAULT_VERSION:
            return {"default_version": None}
        return {"default_version": self._available}

    def _maybe_raise(self, sql) -> None:
        error = self._errors[sql]
        if error is not None:
            raise error

    # -- assertions helpers -------------------------------------------------

    def sql_calls(self) -> list[str]:
        return [c["sql"] for c in self.calls]

    def create_extension_calls(self) -> list[str]:
        return [sql for sql in self.sql_calls() if "CREATE EXTENSION" in sql]


def _allow_unsupported(monkeypatch) -> None:
    monkeypatch.setenv(AGE_ALLOW_UNSUPPORTED_ENV, "true")


@contextlib.contextmanager
def _capture_gate_logs(caplog, level=logging.DEBUG):
    """Capture records emitted on the ``lightrag`` logger.

    ``lightrag/utils.py`` sets ``propagate = False`` on that logger, so caplog --
    which attaches to root -- never sees its records. Flip propagation for the
    duration of the block, then restore it.
    """
    caplog.clear()
    old_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level(level, logger=lightrag_logger.name):
            yield caplog
    finally:
        lightrag_logger.propagate = old_propagate


class _FakeAcquire:
    """Async context manager standing in for ``pool.acquire()``."""

    def __init__(self, connection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, *exc_info):
        return False


class _FakePool:
    def __init__(self, connection):
        self._connection = connection

    def acquire(self):
        return _FakeAcquire(self._connection)

    async def close(self):
        return None


def _make_db(monkeypatch, connection, *, attempts: int = 3) -> PostgreSQLDB:
    """A PostgreSQLDB whose ``_run_with_retry`` runs against ``connection``.

    Only the pool plumbing is faked: the tenacity loop, ``retry_if_exception_type``
    and ``reraise=True`` are the real ones, which is the whole point of the tests
    that use this.
    """
    db = PostgreSQLDB(
        {
            "host": "localhost",
            "port": 5432,
            "user": "u",
            "password": "p",
            "database": "d",
            "workspace": "w",
            "max_connections": 1,
            "connection_retry_attempts": attempts,
            # No backoff: these tests exercise the loop, not the sleeps.
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
            "pool_close_timeout": 1.0,
        }
    )
    pool = _FakePool(connection)

    async def _ensure_pool() -> None:
        # _before_sleep calls _reset_pool(), which clears self.pool; restore it so a
        # retried attempt has a pool to acquire from, as reconnecting would in reality.
        db.pool = pool

    monkeypatch.setattr(db, "_ensure_pool", _ensure_pool)
    db.pool = pool
    return db


# ---------------------------------------------------------------------------
# Supported versions start normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supported_version_starts(monkeypatch):
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.7.0")

    await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls()


@pytest.mark.asyncio
async def test_create_extension_runs_after_the_version_decision(monkeypatch):
    """The CREATE is the *last* thing the gate does, not the first.

    Creating the extension before deciding whether its version is safe means a
    refused startup has already registered ``age`` in a database that had none --
    a side effect the operator then has to undo. Ordering, not mere presence, is
    what makes the refusal below (``test_refused_startup_issues_no_create``) a
    property of the code rather than a coincidence.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.7.0")

    await PostgreSQLDB.configure_age_extension(conn)

    sqls = conn.sql_calls()
    create_index = next(i for i, sql in enumerate(sqls) if "CREATE EXTENSION" in sql)
    assert sqls.index(AGE_INSTALLED_SQL) < create_index, sqls
    assert sqls.index(AGE_AVAILABLE_SQL) < create_index, sqls


@pytest.mark.asyncio
async def test_create_extension_runs_after_the_lookups_under_override(monkeypatch):
    """The override forces a start, and a forced start still creates the extension.

    Otherwise the escape hatch would let the process start against a database in
    which AGE had never been created -- every later graph query failing on a missing
    ``ag_catalog``, with the version warning as the only clue.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed="1.8.0", available="1.8.0")

    await PostgreSQLDB.configure_age_extension(conn)

    sqls = conn.sql_calls()
    create_index = next(i for i, sql in enumerate(sqls) if "CREATE EXTENSION" in sql)
    assert sqls.index(AGE_INSTALLED_SQL) < create_index, sqls
    assert sqls.index(AGE_AVAILABLE_SQL) < create_index, sqls


@pytest.mark.asyncio
async def test_create_extension_failure_does_not_raise(monkeypatch):
    """AGE-less deployments must keep starting, as they do without the gate.

    Neither catalog names ``age``, so there is nothing to gate; the gate goes on to
    issue the CREATE anyway, which fails benignly with "extension age is not
    available". That failure must not be turned into a startup failure.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        installed=None,
        available=None,
        execute_error=RuntimeError('extension "age" is not available'),
    )

    await PostgreSQLDB.configure_age_extension(conn)

    # Both lookups and the CREATE attempt ran; none of them raised out of the gate.
    assert AGE_INSTALLED_SQL in conn.sql_calls()
    assert AGE_AVAILABLE_SQL in conn.sql_calls()
    assert conn.create_extension_calls()


@pytest.mark.asyncio
async def test_absent_age_still_attempts_the_create(monkeypatch):
    """Neither catalog names ``age`` -> no version to gate, but do not return early.

    Returning early here would change what the gate does to a *working* deployment:
    a database where ``CREATE EXTENSION age`` would have succeeded but the catalogs
    were read a moment before the package landed would silently never get the
    extension. Reaching a CREATE that fails harmlessly is the cheaper mistake.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed=None, available=None)

    await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls()


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
async def test_refused_startup_issues_no_create(monkeypatch):
    """A refusal must leave the database exactly as it found it.

    Also subsumes the older ``test_failed_create_extension_still_refuses_unsafe_version``:
    a failing CREATE can no longer mask an unsafe pre-existing installation, because
    the version decision happens before the CREATE is attempted at all.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        installed="1.8.0",
        available="1.8.0",
        execute_error=RuntimeError("permission denied to create extension"),
    )

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls() == [], conn.sql_calls()


@pytest.mark.asyncio
async def test_undeterminable_version_issues_no_create(monkeypatch):
    """The other refusal branch must be just as free of side effects."""
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0-dev", available="1.8.0-dev")

    with pytest.raises(RuntimeError, match="Could not determine"):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls() == [], conn.sql_calls()


# ---------------------------------------------------------------------------
# The stale-catalog cell: unsafe extversion over downgraded binaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_catalog_refusal_has_its_own_guidance(monkeypatch):
    """installed unsafe + available safe is a state no remedy in the generic message fixes.

    The operator put 1.7.0 binaries back under a database whose ``pg_extension``
    still says 1.8.0. ``extversion`` cannot be lowered -- AGE ships no downgrade
    script -- so ``ALTER EXTENSION age UPDATE`` cannot repair it, and AGE is simply
    non-functional until the extension is dropped and recreated. The generic message
    would send the operator to pin a version they have already pinned.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available="1.7.0")

    with pytest.raises(RuntimeError) as excinfo:
        await PostgreSQLDB.configure_age_extension(conn)

    message = str(excinfo.value)
    # The operator-facing facts that only this message carries.
    assert "downgrade" in message.lower(), message
    # What breaks here is narrower than in the crash-prone cells: the newer script's
    # functions are missing from the older library, so writes still land and reads do
    # not. "Every graph query fails" would send the operator hunting the wrong symptom.
    assert "MATCH" in message, message
    assert "read-broken" in message, message
    # The override is named, but as a statement about the outcome rather than as an
    # offer: setting it starts a deployment that still cannot read anything back.
    assert AGE_ALLOW_UNSUPPORTED_ENV in message, message
    # And the generic remedies are not offered at all, rather than offered and then
    # withdrawn: pinning is what produced this state, so proposing it here and taking
    # it back two sentences later reads to the operator as a contradiction.
    assert "Pin AGE to a verified version" not in message, message
    assert "at your own risk" not in message, message
    # Both readings are still surfaced, as in every other refusal.
    assert "installed='1.8.0'" in message, message
    assert "available='1.7.0'" in message, message


@pytest.mark.asyncio
async def test_ordinary_unsupported_refusal_omits_the_stale_catalog_guidance(
    monkeypatch,
):
    """The contrast that makes the test above mean something.

    If the stale-catalog wording leaked into every unsupported refusal, the assertions
    above would pass while the two states stayed indistinguishable to the operator.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available="1.8.0")

    with pytest.raises(RuntimeError) as excinfo:
        await PostgreSQLDB.configure_age_extension(conn)

    message = str(excinfo.value)
    assert "downgrade" not in message.lower(), message
    assert "read-broken" not in message, message
    # The mirror of the two absences asserted above: here the generic remedies *are*
    # the right advice, so withholding them would be the regression.
    assert "Pin AGE to a verified version" in message, message
    assert "at your own risk" in message, message


@pytest.mark.asyncio
async def test_override_still_starts_on_the_stale_catalog_cell(monkeypatch, caplog):
    """ "Does not help" is advice about the outcome, not a second gate.

    The refusal message tells the operator that the override does not help here, and
    it means it in the useful sense: setting the variable starts a process in which
    no graph query works. It does not mean the branch keeps raising. The escape hatch
    stays uniformly reachable -- an operator who has already decided to run anyway
    does not get a second, undocumented refusal to work around -- so this cell
    behaves like every other unsupported one under the override.

    Pinned because the two readings are easy to confuse, and a later change that made
    this branch raise under the override would be a silent behaviour change for
    anyone relying on the hatch.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed="1.8.0", available="1.7.0")

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    warnings = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert any("1.8.0" in m for m in warnings), warnings
    assert conn.create_extension_calls()


# ---------------------------------------------------------------------------
# Unreadable versions fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"installed": "y.y.y", "available": "y.y.y"}, id="unparseable"),
        # A blank string is present-but-unreadable, not absent: a catalog that names
        # ``age`` and reports '' has told us nothing verifiable about the binaries
        # that are on disk, and the safe reading is that they are unverified.
        pytest.param({"installed": "", "available": ""}, id="both-blank"),
        pytest.param({"installed": "", "available": "1.7.0"}, id="blank-and-safe"),
        pytest.param({"installed": "1.7.0", "available": ""}, id="safe-and-blank"),
    ],
)
@pytest.mark.asyncio
async def test_undeterminable_version_fails_closed(monkeypatch, kwargs):
    """A catalog that names ``age`` but reports nothing usable fails closed.

    This is distinct from neither catalog naming ``age`` at all -- see
    ``test_absent_age_still_attempts_the_create`` above, which is not a failure to
    determine a version but the absence of anything to check in the first place.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(**kwargs)

    with pytest.raises(RuntimeError, match="Could not determine"):
        await PostgreSQLDB.configure_age_extension(conn)


@pytest.mark.asyncio
async def test_unsafe_version_wins_even_with_one_unparseable_source(monkeypatch):
    """A known-unsafe reading from one source must not be excused by the other
    being unparseable.

    It only pins the safer half of that idea -- a *known-unsafe* parseable sibling
    still wins. The mirror case, where the unparseable source would need to be
    ignored to *avoid* a refusal, is pinned by
    ``test_unparseable_binaries_source_does_not_defer_to_stale_catalog`` below, and
    refuses instead.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="not-a-version", available="1.8.0")

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)


@pytest.mark.asyncio
async def test_unparseable_binaries_source_does_not_defer_to_stale_catalog(monkeypatch):
    """default_version tracks the binaries; if it is unreadable, extversion must not
    decide alone -- it is exactly the source that does not move on a binary upgrade.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.8.0-dev")

    with pytest.raises(RuntimeError, match="Could not determine"):
        await PostgreSQLDB.configure_age_extension(conn)


@pytest.mark.asyncio
async def test_raising_installed_lookup_fails_closed(monkeypatch):
    """A raising ``pg_extension`` lookup is exactly as undeterminable as an empty one.

    Unlike the AVAILABLE lookup below, this query touches nothing but the ``age`` row
    of one catalog, so there is no unrelated-extension failure mode to tolerate: if
    it fails, the version genuinely could not be determined.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed_error=RuntimeError("relation does not exist"))

    with pytest.raises(RuntimeError, match="Could not determine"):
        await PostgreSQLDB.configure_age_extension(conn)


@pytest.mark.asyncio
async def test_raising_installed_lookup_bypassed_by_override(monkeypatch):
    """override_set must be computed before the lookup it also guards."""
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed_error=RuntimeError("relation does not exist"))

    await PostgreSQLDB.configure_age_extension(conn)


# ---------------------------------------------------------------------------
# A broken pg_available_extensions is an unread source, not a safe one
# ---------------------------------------------------------------------------


def test_malformed_control_file_error_is_not_transient():
    """The premise of the tests below.

    If this error type were in ``TRANSIENT_DB_EXCEPTIONS`` it would be re-raised for
    retry and would never reach the degraded path at all, so those tests would be
    passing for the wrong reason.
    """
    assert not isinstance(_MALFORMED_CONTROL_FILE_ERROR, TRANSIENT_DB_EXCEPTIONS)


@pytest.mark.asyncio
async def test_unreadable_available_lookup_does_not_let_a_stale_installed_decide(
    monkeypatch, caplog
):
    """A lookup that never answered must fail closed, exactly like an unparseable value.

    ``pg_extension.extversion`` is the version of the SQL script last run against this
    database. It does not move when the binaries are swapped underneath it -- which is
    the entire reason the second source exists. So a safe ``installed`` reading, on its
    own, is not evidence that the loaded ``age.so`` is safe; it is evidence about a
    catalog row that a binary upgrade leaves untouched.

    This was reproduced live: an extension created under AGE 1.7.0, then 1.8.0's
    ``age.so`` installed over it with ``age.control`` corrupted so the AVAILABLE view
    raises. ``pg_extension`` still said '1.7.0', the gate started, and the loaded
    binaries were the ones it exists to refuse.

    The asymmetry that settles it: ``(safe installed, unparseable available)`` refuses
    -- see ``test_unparseable_binaries_source_does_not_defer_to_stale_catalog`` -- and
    a failed lookup is the same state, only with even less information. Tolerating the
    broken view here would mean the binary-tracking source can be silenced by any
    unrelated package shipping a malformed ``.control``.

    (This test previously asserted the opposite, on the reasoning that the view's
    failure is a fact about some other extension and so should not refuse "a perfectly
    good 1.7.0 deployment". The code cannot know the deployment is good; that is what
    it was trying to find out.)
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        installed="1.7.0", available_error=_MALFORMED_CONTROL_FILE_ERROR
    )

    with _capture_gate_logs(caplog):
        with pytest.raises(RuntimeError, match="Could not determine"):
            await PostgreSQLDB.configure_age_extension(conn)

    # The degraded lookup still names the view it could not read, so the operator can
    # tell this refusal from one caused by an unreadable *value*.
    warnings = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert any("pg_available_extensions" in m for m in warnings), warnings
    # A refusal leaves the database as it found it.
    assert conn.create_extension_calls() == [], conn.sql_calls()


@pytest.mark.asyncio
async def test_override_bypasses_the_unreadable_available_refusal(monkeypatch, caplog):
    """The escape hatch reaches this refusal too, as it reaches every other one.

    An operator on a build that reports an unsupported number, or one who knows what
    their sharedir contains, must not find this the single unbypassable branch.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(
        installed="1.7.0", available_error=_MALFORMED_CONTROL_FILE_ERROR
    )

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    warnings = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    # Nothing is raised here, so a warning is the whole operator-facing output.
    assert any("could not determine" in m.lower() for m in warnings), warnings
    assert conn.create_extension_calls(), conn.sql_calls()


@pytest.mark.asyncio
async def test_unreadable_available_lookup_does_not_install_an_unchecked_age(
    monkeypatch, caplog
):
    """No row in ``pg_extension`` plus an unreadable AVAILABLE lookup is *undetermined*.

    This cell used to be handled as "neither catalog names age, so there is nothing to
    gate", which conflated two states the gate must keep apart: the view genuinely
    having no ``age`` row, and the view having failed before it could be asked. In the
    second state AGE may well be on disk -- the reproduction is AGE 1.8.0 installed
    alongside one unrelated malformed control file -- and the old handling then went on
    to ``CREATE EXTENSION``, registering and loading exactly the version the gate exists
    to refuse. The first graph request segfaults the instance.

    So the create is skipped here. Not raising is deliberate, and is what separates this
    cell from ``test_unreadable_available_lookup_does_not_let_a_stale_installed_decide``
    above: there ``pg_extension`` names ``age``, so AGE is demonstrably in use and an
    undeterminable version is a startup failure. Here nothing says AGE is present at
    all, and the same unreadable view on a server with no AGE must not become a startup
    failure -- skipping a create that would have been a no-op there costs that
    deployment nothing. The warning is the entire operator-facing output, so it has to
    say what was skipped and how to override it.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        installed=None, available_error=_MALFORMED_CONTROL_FILE_ERROR
    )

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    warnings = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    # The undeterminable state must not be able to install an unchecked AGE.
    assert conn.create_extension_calls() == [], conn.sql_calls()
    # Two warnings are emitted here -- the degraded AVAILABLE lookup logs its own
    # first -- so this deliberately asks for *one record* carrying both facts rather
    # than for the pair to mention one each. An operator who reads only the line about
    # the skip still learns what was not done and what to set to have it done anyway.
    skipped = [
        m
        for m in warnings
        if "CREATE EXTENSION" in m and AGE_ALLOW_UNSUPPORTED_ENV in m
    ]
    assert skipped, (
        f"expected a skipped-create warning naming the override; got: {warnings}"
    )


@pytest.mark.asyncio
async def test_override_installs_age_despite_an_unreadable_available_lookup(
    monkeypatch,
):
    """The escape hatch reaches the undetermined state too.

    An operator who has read the warning above and decided that this server's AGE is
    fine -- or who simply has no AGE and wants the create attempted regardless -- gets
    what they asked for. Without this the skip would be an unbypassable behaviour
    change for anyone whose sharedir contains a broken control file.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(
        installed=None, available_error=_MALFORMED_CONTROL_FILE_ERROR
    )

    await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls(), conn.sql_calls()


@pytest.mark.asyncio
async def test_broken_available_lookup_does_not_rescue_an_unsafe_installed(monkeypatch):
    """A source that *did* answer, and answered unsafe, still decides.

    ``installed`` alone at 1.8.0 is the deleted-control-file shape, and it is refused
    whether ``available`` returned NULL or failed outright. Unchanged by the failed
    lookup now failing closed, and pinned so that the new refusal cannot quietly
    reclassify this one: the operator must still be told the version is unsupported,
    not merely that it could not be determined.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        installed="1.8.0", available_error=_MALFORMED_CONTROL_FILE_ERROR
    )

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls() == [], conn.sql_calls()


# ---------------------------------------------------------------------------
# A NULL default_version is a control file that names no version, not an absent AGE
# ---------------------------------------------------------------------------
#
# ``pg_available_extensions.default_version`` is nullable: the view lists one row per
# control file found in the sharedir, and a control file that sets no default_version
# yields a row whose column is NULL. AGE is then on disk -- its .so will load -- and the
# catalog has said nothing about which build it is.
#
# Read with ``fetchval`` that state is indistinguishable from having no row, and the gate
# read both as "AGE is not on disk", leaving ``pg_extension`` to decide alone. That is the
# one source that cannot see a binary swap, so the reproduction is direct: extversion
# '1.7.0' over 1.8.0 binaries started, and the first graph request is a segfault. Hence
# ``fetchrow`` and the four-state fake above.


def _warnings(caplog) -> list[str]:
    return [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
async def test_null_default_version_does_not_let_a_stale_installed_decide(
    monkeypatch, caplog
):
    """The reproduction: a versionless control file must not read as "no AGE here".

    ``pg_extension`` says 1.7.0 because that is the script last run against this
    database; it says nothing about the ``age.so`` that a later install put on disk. The
    only source that tracks the binaries has a row and no version in it, so the version
    is undetermined -- and an undetermined version fails closed, exactly as it does for a
    value that will not parse or a lookup that never answered.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available=_NULL_DEFAULT_VERSION)

    with _capture_gate_logs(caplog):
        with pytest.raises(RuntimeError, match="Could not determine"):
            await PostgreSQLDB.configure_age_extension(conn)

    # A refusal leaves the database as it found it -- and before the fix this cell did
    # not refuse at all, it created.
    assert conn.create_extension_calls() == [], conn.sql_calls()


@pytest.mark.asyncio
async def test_null_default_version_does_not_rescue_an_unsafe_installed(monkeypatch):
    """A source that did answer, and answered unsafe, still decides.

    The counterpart of ``test_broken_available_lookup_does_not_rescue_an_unsafe_installed``
    for the NULL row: the operator must be told the version is unsupported, not merely
    that it could not be determined, or the remedies they are given are the wrong ones.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available=_NULL_DEFAULT_VERSION)

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls() == [], conn.sql_calls()


@pytest.mark.asyncio
async def test_null_default_version_with_no_installed_row_skips_the_create(
    monkeypatch, caplog
):
    """Nothing registered here, and a control file that names no version: undetermined.

    Same shape as the unreadable-view cell with no ``installed`` row, and skipped for
    the same reason: AGE may well be on disk, so creating would register and load the
    very version the gate never got to check. Not raising is equally deliberate --
    nothing here says AGE is in use, and a versionless control file belonging to a
    server that never runs AGE must not become a startup failure.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed=None, available=_NULL_DEFAULT_VERSION)

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls() == [], conn.sql_calls()
    skipped = [
        m
        for m in _warnings(caplog)
        if "CREATE EXTENSION" in m and AGE_ALLOW_UNSUPPORTED_ENV in m
    ]
    assert skipped, (
        f"expected a skipped-create warning naming the override; got: {_warnings(caplog)}"
    )


@pytest.mark.parametrize(
    "installed", [pytest.param("1.7.0", id="safe"), pytest.param("1.8.0", id="unsafe")]
)
@pytest.mark.asyncio
async def test_override_bypasses_the_null_default_version_refusals(
    monkeypatch, installed
):
    """The escape hatch reaches these two refusals, as it reaches every other one."""
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed=installed, available=_NULL_DEFAULT_VERSION)

    await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls(), conn.sql_calls()


@pytest.mark.asyncio
async def test_an_absent_available_row_still_starts_a_safe_installed(monkeypatch):
    """The contrast that makes the fix a distinction and not a blanket refusal.

    No ``age`` row at all is genuine evidence that nothing is on disk, so a safe
    ``installed`` reading is the whole picture and the gate starts. Answering the NULL
    row by refusing on ``None`` too would have "fixed" the fail-open by breaking every
    ordinary deployment that has AGE registered and no control file in the sharedir.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available=None)

    await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls(), conn.sql_calls()


@pytest.mark.asyncio
async def test_null_default_version_is_not_reported_as_age_being_absent(
    monkeypatch, caplog
):
    """The nothing-to-gate INFO is a claim about the deployment, and here it is false.

    That line says AGE is not installed on this server and there is nothing to gate. A
    versionless control file is AGE sitting in the sharedir, so logging it here would
    tell an operator reading their startup log the opposite of what is true -- and it
    was the visible symptom of the fail-open, emitted on the way to creating the
    extension.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed=None, available=_NULL_DEFAULT_VERSION)

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    absent_claims = [
        rec.getMessage()
        for rec in caplog.records
        if "neither pg_available_extensions nor pg_extension names" in rec.getMessage()
    ]
    assert absent_claims == [], absent_claims


# ---------------------------------------------------------------------------
# Under the override, the warning has to name what is actually being done
# ---------------------------------------------------------------------------
#
# With no ``installed`` row and an AVAILABLE lookup that failed, the override does not
# merely tolerate an unknown version: it goes on to CREATE EXTENSION, installing an AGE
# the gate never saw. The operator consented to starting anyway; what happens is an
# install. Every other override site names its consequence, and this one has to as well,
# because the consequence here is the crash the gate exists to prevent.

_CRASH_RECOVERY = "crash recovery"


def _create_decision_warnings(caplog) -> list[str]:
    """Warnings about what the gate did with the extension, not about the lookup.

    The degraded AVAILABLE lookup logs its own warning in both states below, and it is
    the same line either way, so comparing whole warning sets would compare mostly that.
    """
    return [
        m for m in _warnings(caplog) if "CREATE EXTENSION" in m or _CRASH_RECOVERY in m
    ]


@pytest.mark.asyncio
async def test_override_install_of_an_unchecked_age_says_so(monkeypatch, caplog):
    """Creating AGE under the override is a different act from tolerating a version.

    Nothing is raised here, so the warning is the entire operator-facing output, and it
    has to carry the two facts that distinguish this from the generic
    could-not-determine line: that AGE is being installed without the gate having seen
    it, and that the version so installed may take the instance through crash recovery
    on the first graph request. One record must carry both -- an operator who reads only
    one line still learns what was done to their database.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(
        installed=None, available_error=_MALFORMED_CONTROL_FILE_ERROR
    )

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls(), conn.sql_calls()
    named = [
        m
        for m in _warnings(caplog)
        if "CREATE EXTENSION" in m and _CRASH_RECOVERY in m.lower()
    ]
    assert named, (
        "expected one warning naming both the install and its crash-recovery "
        f"consequence; got: {_warnings(caplog)}"
    )


@pytest.mark.asyncio
async def test_without_the_override_that_state_installs_nothing(monkeypatch, caplog):
    """The half of the pair that must not change: no override, no create.

    Restated next to the override case so the two are read together; the same claim is
    made once more from the state-space sweep at the end of this module.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        installed=None, available_error=_MALFORMED_CONTROL_FILE_ERROR
    )

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls() == [], conn.sql_calls()
    skipped = [
        m
        for m in _warnings(caplog)
        if "CREATE EXTENSION" in m and AGE_ALLOW_UNSUPPORTED_ENV in m
    ]
    assert skipped, _warnings(caplog)


@pytest.mark.asyncio
async def test_the_two_override_states_do_not_share_one_message(monkeypatch, caplog):
    """Consent and no consent must not be reported by the same string.

    The two states differ in what happens to the database -- one installs an unchecked
    AGE, the other leaves it alone -- so a single message covering both would be wrong
    for one of them whatever it said. Asserting the difference directly is what stops a
    later simplification from collapsing them back into the generic
    could-not-determine line, which is the shape the fail-open had.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    skipping = _FakeConnection(
        installed=None, available_error=_MALFORMED_CONTROL_FILE_ERROR
    )
    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(skipping)
        without_override = _create_decision_warnings(caplog)

    _allow_unsupported(monkeypatch)
    creating = _FakeConnection(
        installed=None, available_error=_MALFORMED_CONTROL_FILE_ERROR
    )
    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(creating)
        with_override = _create_decision_warnings(caplog)

    assert without_override, "the skipping state logged nothing about the create"
    assert with_override, "the creating state logged nothing about the create"
    assert set(without_override).isdisjoint(with_override), (
        f"the same message is emitted whether or not the operator consented: "
        f"{sorted(set(without_override) & set(with_override))}"
    )
    # And the difference is not cosmetic: only one of them mentions the consequence the
    # operator is accepting, and only the other says nothing was installed.
    assert any(_CRASH_RECOVERY in m.lower() for m in with_override), with_override
    assert not any(_CRASH_RECOVERY in m.lower() for m in without_override), (
        without_override
    )


# ---------------------------------------------------------------------------
# The second state that installs an AGE nothing has looked at
# ---------------------------------------------------------------------------
#
# There are exactly two of them, and they are not the same state. Above, the INSTALLED
# lookup answered "absent" and only AVAILABLE failed. Here the INSTALLED lookup itself
# failed, which switches the gate to version_known = False -- so the AVAILABLE lookup is
# never even attempted and the gate ends up having seen nothing at all, while the
# CREATE EXTENSION at the end still runs. Same consent, same possible outcome, and for a
# while only the generic could-not-determine line to show for it.
#
# What this state must *not* say is anything about whether 'age' is registered in this
# database: that is exactly the question the failed lookup did not answer.
_REGISTRATION_CLAIM = "not registered"


@pytest.mark.asyncio
async def test_override_on_unreadable_installed_lookup_names_the_install(
    monkeypatch, caplog
):
    """The consent here is to an install nothing inspected, so say that.

    Nothing raises, so the warning is the whole operator-facing output. It has to carry
    both facts -- that the AGE about to be created was never looked at, and that such an
    AGE may take the instance through crash recovery on the first graph request -- and
    it must not assert the registration status the lookup failed to establish.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed_error=RuntimeError("relation does not exist"))

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls(), conn.sql_calls()
    named = [
        m
        for m in _warnings(caplog)
        if "CREATE EXTENSION" in m and _CRASH_RECOVERY in m.lower()
    ]
    assert named, (
        "expected one warning naming both the uninspected install and its "
        f"crash-recovery consequence; got: {_warnings(caplog)}"
    )
    assert not any(_REGISTRATION_CLAIM in m for m in named), (
        f"this state cannot know whether 'age' is registered; got: {named}"
    )


@pytest.mark.asyncio
async def test_the_two_unchecked_install_consents_stay_distinct(monkeypatch, caplog):
    """Two states, two messages -- neither both nor neither.

    Each override state below installs an AGE the gate never saw, but by a different
    route and knowing different things, so one shared string would be wrong for one of
    them whatever it said. Collapsing them back into the generic could-not-determine
    line is the simplification this pins against.
    """
    _allow_unsupported(monkeypatch)

    absent_installed = _FakeConnection(
        installed=None, available_error=_MALFORMED_CONTROL_FILE_ERROR
    )
    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(absent_installed)
        from_absent = _create_decision_warnings(caplog)

    unreadable_installed = _FakeConnection(
        installed_error=RuntimeError("relation does not exist")
    )
    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(unreadable_installed)
        from_unreadable = _create_decision_warnings(caplog)

    assert absent_installed.create_extension_calls(), absent_installed.sql_calls()
    assert unreadable_installed.create_extension_calls(), (
        unreadable_installed.sql_calls()
    )
    # Neither: each state has to speak for itself.
    assert from_absent, "the absent-installed state said nothing about the create"
    assert from_unreadable, (
        "the unreadable-installed state said nothing about the create"
    )
    # Both: one string covering the pair would misdescribe one of them.
    assert set(from_absent).isdisjoint(from_unreadable), (
        "the same message is emitted for two different states: "
        f"{sorted(set(from_absent) & set(from_unreadable))}"
    )
    # And the difference is the fact that separates them: only the first state's lookup
    # established that 'age' is absent from pg_extension.
    assert any(_REGISTRATION_CLAIM in m for m in from_absent), from_absent
    assert not any(_REGISTRATION_CLAIM in m for m in from_unreadable), from_unreadable
    # Both still name the consequence being consented to.
    assert all(
        any(_CRASH_RECOVERY in m.lower() for m in group)
        for group in (from_absent, from_unreadable)
    ), (from_absent, from_unreadable)


# ---------------------------------------------------------------------------
# Transient failures stay retryable, from either lookup
# ---------------------------------------------------------------------------


def test_transient_exceptions_covers_a_non_oserror_asyncpg_error():
    """Guard against the parametrize below going vacuous again.

    ``asyncio.TimeoutError is TimeoutError`` on 3.11+, and both ``TimeoutError`` and
    ``ConnectionResetError`` derive from ``OSError`` -- so a list built only from
    builtins would still pass with every asyncpg-native entry deleted from
    ``TRANSIENT_DB_EXCEPTIONS``. ``ConnectionDoesNotExistError`` derives from
    ``PostgresError``, not ``OSError``, so covering it is only possible while the
    asyncpg-native entries are actually there.
    """
    assert not issubclass(asyncpg.exceptions.ConnectionDoesNotExistError, OSError)
    assert issubclass(
        asyncpg.exceptions.ConnectionDoesNotExistError, asyncpg.exceptions.PostgresError
    )
    assert issubclass(
        asyncpg.exceptions.ConnectionDoesNotExistError, TRANSIENT_DB_EXCEPTIONS
    )


def _transient_exception_params():
    return [
        pytest.param(ConnectionResetError("reset"), id="ConnectionResetError"),
        pytest.param(OSError("broken pipe"), id="OSError"),
        pytest.param(TimeoutError(), id="TimeoutError"),
        # Not an OSError subclass -- see the guard test above. A dropped connection
        # is exactly what a catalog read hits, and it arrives as this type.
        pytest.param(
            asyncpg.exceptions.ConnectionDoesNotExistError("connection is closed"),
            id="asyncpg.ConnectionDoesNotExistError",
        ),
    ]


def _lookup_params():
    return [
        pytest.param("installed_error", id="installed-lookup"),
        pytest.param("available_error", id="available-lookup"),
    ]


@pytest.mark.parametrize("which", _lookup_params())
@pytest.mark.parametrize("exc", _transient_exception_params())
@pytest.mark.asyncio
async def test_transient_lookup_failure_propagates_unwrapped(monkeypatch, exc, which):
    """A connection blip must stay retryable, from either query.

    ``PGGraphStorage.initialize`` calls this through ``_run_with_retry``, whose
    ``retry_if_exception_type(self._transient_exceptions)`` can only recognise the
    original exception type. Wrapping a transient error in ``RuntimeError`` would
    turn a retryable startup blip into a permanent startup failure.

    The AVAILABLE lookup is parametrized in alongside the INSTALLED one because it is
    the one allowed to swallow its own failures: the transient check has to be made
    *before* that degrade-to-None, or a blip becomes "this catalog said nothing" and
    the retry never happens.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.7.0", **{which: exc})

    with pytest.raises(type(exc)) as excinfo:
        await PostgreSQLDB.configure_age_extension(conn)

    assert excinfo.value is exc
    assert isinstance(excinfo.value, TRANSIENT_DB_EXCEPTIONS)


@pytest.mark.parametrize("which", _lookup_params())
@pytest.mark.parametrize("exc", _transient_exception_params())
@pytest.mark.asyncio
async def test_transient_lookup_failure_propagates_even_under_override(
    monkeypatch, exc, which
):
    """The override must not swallow a blip.

    The transient re-raise is placed *before* the override branch precisely so that
    ``POSTGRES_AGE_ALLOW_UNSUPPORTED_VERSION`` cannot convert "the server never got
    to answer" into "we checked and moved on". With the branches in the other order
    the blip would return quietly here, and the retry that should have happened --
    and might well have produced a *refusal* on the second attempt -- never would.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed="1.7.0", available="1.7.0", **{which: exc})

    with pytest.raises(type(exc)) as excinfo:
        await PostgreSQLDB.configure_age_extension(conn)

    assert excinfo.value is exc


@pytest.mark.asyncio
async def test_gate_refusal_survives_run_with_retry_unwrapped(monkeypatch):
    """A gate refusal must reach the operator as its own RuntimeError.

    ``RuntimeError`` is deliberately *not* in ``TRANSIENT_DB_EXCEPTIONS``: "the
    server answered something we cannot use" is permanent and must fail fast. But
    tenacity's default is to raise ``RetryError`` once the loop gives up, which would
    bury the message that tells the operator what to do. ``_run_with_retry`` passes
    ``reraise=True``, which is what keeps the original exception visible -- so this
    drives the real loop and asserts on what comes out of it, rather than relating
    two constants.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available="1.8.0")
    db = _make_db(monkeypatch, conn)
    attempts = []

    async def operation(connection):
        attempts.append(connection)
        await PostgreSQLDB.configure_age_extension(connection)

    # pytest.raises(RuntimeError) is the load-bearing assertion: tenacity's RetryError
    # is not a RuntimeError, so dropping reraise=True fails here.
    with pytest.raises(RuntimeError, match="not supported") as excinfo:
        await db._run_with_retry(operation)

    # The operator-facing remedies survive the retry wrapper intact.
    assert "PGTableGraphStorage" in str(excinfo.value)
    assert AGE_ALLOW_UNSUPPORTED_ENV in str(excinfo.value)
    # Not retried: a permanent condition must not spin the loop.
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_run_with_retry_does_retry_a_transient_gate_failure(monkeypatch):
    """Counterpart to the test above, and its harness check.

    Without this, ``len(attempts) == 1`` there would also pass if the fake pool made
    retrying impossible for reasons unrelated to the exception type.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(
        installed_error=asyncpg.exceptions.ConnectionDoesNotExistError("closed")
    )
    db = _make_db(monkeypatch, conn, attempts=3)
    attempts = []

    async def operation(connection):
        attempts.append(connection)
        await PostgreSQLDB.configure_age_extension(connection)

    with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
        await db._run_with_retry(operation)

    assert len(attempts) == 3


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
        pytest.param(
            {"installed": "1.8.0", "available": None},
            id="installed-only-control-file-gone",
        ),
        pytest.param({"installed": "y.y.y", "available": "y.y.y"}, id="unparseable"),
        pytest.param(
            {"installed": "1.7.0", "available": "1.8.0-dev"},
            id="safe-installed-unparseable-available",
        ),
        pytest.param(
            {"installed": "1.8.0-dev", "available": "1.7.0"},
            id="unparseable-installed-safe-available",
        ),
        # Not a value the lookup returned but a lookup that never returned one, which
        # the gate treats the same way -- so the override has to reach it the same way.
        pytest.param(
            {"installed": "1.7.0", "available_error": _MALFORMED_CONTROL_FILE_ERROR},
            id="safe-installed-unreadable-available",
        ),
    ],
)
@pytest.mark.asyncio
async def test_override_bypasses_every_refusal(monkeypatch, kwargs):
    """Every raise must be reachable by the documented escape hatch.

    AGE reports its version per release rather than per commit, so a source build
    that fixes the crash still reports an unsupported number; without this the gate
    would be a permanent block for anyone on such a build.

    Both catalogs absent is deliberately not in this list: it returns without
    refusing at all, so the case would pass whether or not the override were set and
    would assert nothing. The stale-catalog cell is covered separately by
    ``test_override_still_starts_on_the_stale_catalog_cell``, which additionally
    pins the warning it emits.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(**kwargs)

    await PostgreSQLDB.configure_age_extension(conn)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("false", id="false"),
        pytest.param("False", id="False"),
        pytest.param("0", id="zero"),
        pytest.param("no", id="no"),
        pytest.param("off", id="off"),
        pytest.param("", id="empty"),
        pytest.param("maybe", id="unrecognised"),
    ],
)
@pytest.mark.asyncio
async def test_gate_stays_armed_for_non_truthy_override_values(monkeypatch, value):
    """Setting the variable is not the same as enabling the escape hatch.

    Every other override test sets it to ``"true"``, so nothing pinned what happens
    for the values an operator is just as likely to write. The coercion lives in
    ``lightrag.utils.get_env_value``, outside this gate -- which is exactly why the
    behaviour the gate depends on deserves pinning here: a change there that made
    "the variable is present" mean "enabled" would silently disarm the gate for
    anyone who had written ``=false`` to keep it on.
    """
    monkeypatch.setenv(AGE_ALLOW_UNSUPPORTED_ENV, value)
    conn = _FakeConnection(installed="1.8.0", available="1.8.0")

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)


# ---------------------------------------------------------------------------
# Log levels carry the parts of the decision that raise nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_override_on_unsupported_version_warns_about_crash_recovery(
    monkeypatch, caplog
):
    """The one path that starts a knowingly-crashing deployment must be loud.

    Nothing is raised here, so the WARNING is the entire operator-facing output --
    downgrading it to INFO, or trimming it to "check skipped", would leave an
    instance one wildcard graph query away from crash recovery with no signal at
    startup. It must name that consequence, not merely the fact of the bypass.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed="1.8.0", available="1.8.0")

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    warnings = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    crash_recovery = [m for m in warnings if "crash recovery" in m]
    assert crash_recovery, f"expected a crash-recovery warning; got: {warnings}"
    assert any("1.8.0" in m for m in crash_recovery), crash_recovery


@pytest.mark.asyncio
async def test_override_on_undeterminable_version_warns(monkeypatch, caplog):
    """The other silent-start path also has to say why it started."""
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(installed="1.8.0-dev", available="1.8.0-dev")

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    warnings = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert any("could not determine" in m.lower() for m in warnings), warnings


@pytest.mark.asyncio
async def test_absent_age_is_reported_at_info_or_above(monkeypatch, caplog):
    """ "AGE is absent" is a fact about the deployment, not a debug detail.

    This branch checks no version, so at DEBUG -- off in every default configuration
    -- an operator who expected the gate to run gets no way to tell "AGE is not on
    this server" from "the gate never executed".
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed=None, available=None)

    with _capture_gate_logs(caplog):
        await PostgreSQLDB.configure_age_extension(conn)

    # Matched on level and on the fact that a record exists at all, not on wording:
    # the only record this call is expected to emit besides the CREATE EXTENSION
    # outcome line is the one announcing that no version check was performed.
    records = [
        rec
        for rec in caplog.records
        if "AGE extension enabled" not in rec.getMessage()
        and "Could not create AGE extension" not in rec.getMessage()
    ]
    assert records, f"expected an AGE-absent record; got: {caplog.text}"
    assert all(rec.levelno >= logging.INFO for rec in records), [
        (rec.levelname, rec.getMessage()) for rec in records
    ]
    assert any("age" in rec.getMessage().lower() for rec in records), [
        rec.getMessage() for rec in records
    ]


# ---------------------------------------------------------------------------
# The gate must keep reading both sources, each from its own query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_issues_both_version_queries(monkeypatch):
    """Neither source may be dropped: each sees deployments the other does not.

    ``pg_extension.extversion`` is the version of the SQL script last run against
    this database, so it does not move when the binaries are swapped underneath it.
    ``pg_available_extensions.default_version`` comes from the on-disk ``age.control``
    and therefore tracks the binaries -- but deleting ``age.control`` (a package
    upgrade, or a partial uninstall) empties it while ``pg_extension`` keeps
    ``extversion`` and the functions keep pointing at ``$libdir/age``.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.7.0", available="1.7.0")

    await PostgreSQLDB.configure_age_extension(conn)

    assert conn.sql_calls().count(AGE_INSTALLED_SQL) == 1, conn.sql_calls()
    assert conn.sql_calls().count(AGE_AVAILABLE_SQL) == 1, conn.sql_calls()


@pytest.mark.asyncio
async def test_each_value_comes_from_its_own_query(monkeypatch):
    """The alias-swap test the single-row lookup could not express.

    ``_FakeConnection`` answers per query, and the two sources are not symmetric:
    unsafe-installed-over-safe-available is the stale-catalog state, while its
    mirror is the in-place binary upgrade. Wiring ``installed`` to the
    ``pg_available_extensions`` query (or vice versa) therefore produces the *other*
    message, and this fails.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available="1.7.0")

    with pytest.raises(RuntimeError) as excinfo:
        await PostgreSQLDB.configure_age_extension(conn)

    assert "installed='1.8.0'" in str(excinfo.value), str(excinfo.value)
    assert "available='1.7.0'" in str(excinfo.value), str(excinfo.value)


def test_installed_query_reads_pg_extension_only():
    """TEXT-ONLY: reads ``AGE_INSTALLED_SQL`` as a string; it never executes here.

    The split into two queries exists because ``pg_available_extensions`` parses every
    control file in the sharedir and raises for the whole view if any one of them is
    malformed. That isolation is lost the moment this query joins the other catalog
    back in, so the absence of ``pg_available_extensions`` from this text is the
    property worth pinning.
    """
    normalized = " ".join(AGE_INSTALLED_SQL.lower().split())

    assert "pg_extension" in normalized, AGE_INSTALLED_SQL
    assert "extversion" in normalized, AGE_INSTALLED_SQL
    assert "pg_available_extensions" not in normalized, AGE_INSTALLED_SQL
    assert "join" not in normalized, AGE_INSTALLED_SQL
    assert "'age'" in normalized, AGE_INSTALLED_SQL


def test_available_query_reads_pg_available_extensions_only():
    """TEXT-ONLY, and the mirror of the above.

    If this query pulled in ``pg_extension`` as well, a failure of the fragile view
    would once more take the robust source down with it.
    """
    normalized = " ".join(AGE_AVAILABLE_SQL.lower().split())

    assert "pg_available_extensions" in normalized, AGE_AVAILABLE_SQL
    assert "default_version" in normalized, AGE_AVAILABLE_SQL
    assert "pg_extension" not in normalized, AGE_AVAILABLE_SQL
    assert "join" not in normalized, AGE_AVAILABLE_SQL
    assert "'age'" in normalized, AGE_AVAILABLE_SQL


@pytest.mark.asyncio
async def test_installed_version_alone_still_refuses(monkeypatch):
    """What a deleted control file produces: extversion set, default_version absent.

    It must be gated on the one source that survived, not waved through.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed="1.8.0", available=None)

    with pytest.raises(RuntimeError, match="not supported"):
        await PostgreSQLDB.configure_age_extension(conn)


def test_first_unsupported_version_is_1_8_0():
    """Raising this bound needs evidence that the newer release was verified."""
    assert AGE_FIRST_UNSUPPORTED_VERSION == (1, 8, 0)


# ---------------------------------------------------------------------------
# _parse_age_version: absent vs. unparseable are distinct outcomes
# ---------------------------------------------------------------------------


def test_parse_age_version_returns_none_only_for_none():
    """``None`` -- the SQL NULL -- is the one and only absent case.

    A present-but-blank string is *not* absent: the catalog named ``age`` and
    answered with something, and something we cannot read is an unverified version.
    See the sentinel cases below, which include the blank strings.
    """
    assert _parse_age_version(None) is None


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="space"),
        pytest.param("\t", id="tab"),
        pytest.param("\n", id="newline"),
        pytest.param("1.8.0-dev", id="dev-suffix"),
        pytest.param("v1.8.0", id="v-prefix"),
        pytest.param("1.8.0-rc1", id="rc-suffix"),
        pytest.param("1.9.0beta", id="beta-suffix"),
        pytest.param("2.0.0-beta", id="beta-dash"),
        pytest.param("1.8.0+pg18", id="build-metadata"),
        pytest.param("1.8.0-1.pgdg120", id="debian-revision"),
        pytest.param("y.y.y", id="not-numeric"),
        # Fewer than three components: padding a short form to three would be
        # repairing a value rather than reading it, and a repaired value must not
        # reach the safe side of the comparison.
        pytest.param("1", id="one-component"),
        pytest.param("1.7", id="two-components"),
        pytest.param("1.8.0.1", id="four-components"),
        # Accepted by int() but never emitted by AGE, and each one changes the value
        # it appears to carry: a sign flips the ordering, an underscore hides a digit
        # group, and non-ASCII digits read as a version that is not the one written.
        pytest.param("1.-8.0", id="negative-component"),
        pytest.param("1_8.0.0", id="pep515-underscore"),
        pytest.param("١.٨.٠", id="arabic-indic-digits"),
        pytest.param(" 1.8.0", id="leading-space"),
        pytest.param("1.8.0 ", id="trailing-space"),
    ],
)
def test_parse_age_version_returns_sentinel_when_unparseable(raw):
    """A source string that is present but not exactly three ASCII numeric components.

    Distinct from the absent case: the gate must not treat "we have a string
    but can't read it" the same as "we have nothing at all" -- the former is a
    signal that something might be wrong, not silence.
    """
    assert _parse_age_version(raw) is _AGE_VERSION_UNPARSEABLE


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1.7.0", (1, 7, 0)),
        ("1.8.0", (1, 8, 0)),
        ("0.0.0", (0, 0, 0)),
        ("10.20.30", (10, 20, 30)),
        # Zero padding is still three ASCII numeric components; int() decides the value.
        ("01.07.00", (1, 7, 0)),
    ],
)
def test_parse_age_version_parses_three_numeric_components(raw, expected):
    """The tightening must not have cost the shape AGE actually reports."""
    assert _parse_age_version(raw) == expected


def test_unparseable_sentinel_is_a_named_singleton():
    """The sentinel is compared with ``is``, so its identity is the contract.

    It is a named class rather than a bare ``object()`` so the parser's return
    annotation stays narrower than ``object`` and a type checker can still flag a
    caller that treats the sentinel as a version tuple. Rebinding the module
    attribute to a fresh instance would leave every ``is`` comparison in the gate
    silently false.
    """
    assert isinstance(_AGE_VERSION_UNPARSEABLE, _AgeVersionUnparseable)
    assert not isinstance(_AGE_VERSION_UNPARSEABLE, tuple)
    assert _parse_age_version("y.y.y") is _parse_age_version("1.8.0-dev")
    assert _parse_age_version("y.y.y") is _AGE_VERSION_UNPARSEABLE


# ---------------------------------------------------------------------------
# Exhaustive decision table over (installed, available) source categories
# ---------------------------------------------------------------------------
#
# Four categories per source:
#   safe        parses, below AGE_FIRST_UNSUPPORTED_VERSION      e.g. "1.7.0"
#   unsafe      parses, at or above AGE_FIRST_UNSUPPORTED_VERSION e.g. "1.8.0"
#   unparseable present but not a plain dotted-integer version    e.g. "1.8.0-dev"
#   absent      no value at all (SQL NULL / no row)              None
#
# Branch order under test: stale catalog > known-unsafe > unreadable > start.
#   - stale catalog: installed parses unsafe AND available parses safe. Its own
#     message, because no remedy in the generic one applies.
#   - "not supported": any source parses to >= AGE_FIRST_UNSUPPORTED_VERSION.
#   - "Could not determine": no source parses to a known-unsafe version, and a source
#     is present-but-unparseable.
#   - start (no exception): every present source parses and is below the bound, or
#     neither catalog names age at all.

_SAFE = "1.7.0"
_UNSAFE = "1.8.0"
_UNPARSEABLE = "1.8.0-dev"
_ABSENT = None
# A fifth category, usable only in the sweep below and not in the table: the lookup
# raised instead of returning anything. It is not a *value* a catalog reported, so it
# has no cell in a table whose axes are reported values -- which is exactly how the
# fail-open below hid from the table for as long as it did.
_UNREADABLE = object()

_START = "start"
_NOT_SUPPORTED = "not supported"
_COULD_NOT_DETERMINE = "Could not determine"
_STALE_CATALOG = "stale catalog"

_DECISION_TABLE = [
    (_SAFE, _SAFE, _START),
    (_SAFE, _UNSAFE, _NOT_SUPPORTED),
    (_SAFE, _UNPARSEABLE, _COULD_NOT_DETERMINE),
    (_SAFE, _ABSENT, _START),
    (_UNSAFE, _SAFE, _STALE_CATALOG),
    (_UNSAFE, _UNSAFE, _NOT_SUPPORTED),
    (_UNSAFE, _UNPARSEABLE, _NOT_SUPPORTED),
    (_UNSAFE, _ABSENT, _NOT_SUPPORTED),
    (_UNPARSEABLE, _SAFE, _COULD_NOT_DETERMINE),
    (_UNPARSEABLE, _UNSAFE, _NOT_SUPPORTED),
    (_UNPARSEABLE, _UNPARSEABLE, _COULD_NOT_DETERMINE),
    (_UNPARSEABLE, _ABSENT, _COULD_NOT_DETERMINE),
    (_ABSENT, _SAFE, _START),
    (_ABSENT, _UNSAFE, _NOT_SUPPORTED),
    (_ABSENT, _UNPARSEABLE, _COULD_NOT_DETERMINE),
    # Neither catalog names age: nothing is installed and nothing is on disk, so
    # there is nothing to gate. Under the old single FULL JOIN row this cell was
    # only reachable as a synthetic row (the WHERE could not produce one), and the
    # table recorded it as "Could not determine"; the real deployment it stands for
    # went down the row-is-None path and started. With two independent lookups the
    # cell is the real state, and it starts.
    (_ABSENT, _ABSENT, _START),
]


def _category_id(value):
    return {
        _SAFE: "safe",
        _UNSAFE: "unsafe",
        _UNPARSEABLE: "unparseable",
        _ABSENT: "absent",
        _UNREADABLE: "unreadable",
        _NULL_DEFAULT_VERSION: "null-version-row",
    }[value]


@pytest.mark.parametrize(
    "installed, available, expected",
    _DECISION_TABLE,
    ids=[
        f"installed={_category_id(i)}-available={_category_id(a)}"
        for i, a, _ in _DECISION_TABLE
    ],
)
@pytest.mark.asyncio
async def test_configure_age_extension_decision_table(
    monkeypatch, installed, available, expected
):
    """Pin all 16 combinations so a future change to the branch order shows up here
    instead of as a silently reopened crash path.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(installed=installed, available=available)

    if expected == _START:
        await PostgreSQLDB.configure_age_extension(conn)
        assert conn.create_extension_calls(), conn.sql_calls()
        return

    with pytest.raises(RuntimeError) as excinfo:
        await PostgreSQLDB.configure_age_extension(conn)

    message = str(excinfo.value)
    if expected == _STALE_CATALOG:
        # Classified by what only this message says, so that falling back to the
        # generic unsupported message fails here rather than passing on a substring
        # the two happen to share.
        assert "downgrade" in message.lower(), message
        assert "read-broken" in message, message
    else:
        assert expected in message, message
        assert "downgrade" not in message.lower(), message

    assert conn.create_extension_calls() == [], conn.sql_calls()


def test_decision_table_covers_every_category_pair():
    """The table is exhaustive by construction, not by having been long enough once."""
    categories = [_SAFE, _UNSAFE, _UNPARSEABLE, _ABSENT]
    covered = {(i, a) for i, a, _ in _DECISION_TABLE}

    assert len(_DECISION_TABLE) == 16
    assert covered == {(i, a) for i in categories for a in categories}


# ---------------------------------------------------------------------------
# The invariant behind the table: only a clean start creates the extension
# ---------------------------------------------------------------------------


def _non_starting_states():
    """Every state in which the gate does *not* conclude that AGE is safe to run.

    Two shapes, deliberately in one list. The twelve refusing cells come straight out
    of the decision table, so a cell that is later reclassified as starting cannot
    quietly drop out of this sweep while still looking covered. The two lookup-failure
    states are not expressible as table cells at all -- the table's categories are
    *values* a lookup returned, and these are lookups that never returned one -- and
    the second of them is precisely the state that got an unchecked AGE installed.
    """
    cases = [
        pytest.param(
            {"installed": installed, "available": available},
            True,
            id=f"refuses-installed={_category_id(installed)}"
            f"-available={_category_id(available)}",
        )
        for installed, available, expected in _DECISION_TABLE
        if expected != _START
    ]
    cases.append(
        pytest.param(
            {"installed_error": RuntimeError("relation does not exist")},
            True,
            id="refuses-unreadable-installed",
        )
    )
    cases.append(
        pytest.param(
            {"installed": None, "available_error": _MALFORMED_CONTROL_FILE_ERROR},
            False,
            id="undetermined-absent-installed-unreadable-available",
        )
    )
    cases.extend(
        pytest.param(
            {"installed": installed, "available_error": _MALFORMED_CONTROL_FILE_ERROR},
            True,
            id=f"refuses-{ident}-installed-unreadable-available",
        )
        for installed, ident in ((_SAFE, "safe"), (_UNSAFE, "unsafe"))
    )
    # A row reporting NULL is likewise not a table cell -- the table's AVAILABLE axis is
    # the value the column held, and this is the column holding nothing. Its shape
    # mirrors the unreadable-lookup rows above: a refusal wherever pg_extension shows
    # AGE is in use, and a quiet skip where nothing says it is.
    cases.extend(
        pytest.param(
            {"installed": installed, "available": _NULL_DEFAULT_VERSION},
            raises,
            id=f"{'refuses' if raises else 'undetermined'}-{ident}"
            "-installed-null-available-version",
        )
        for installed, ident, raises in (
            (_SAFE, "safe", True),
            (_UNSAFE, "unsafe", True),
            (_UNPARSEABLE, "unparseable", True),
            (None, "absent", False),
        )
    )
    return cases


@pytest.mark.parametrize("kwargs, raises", _non_starting_states())
@pytest.mark.asyncio
async def test_no_state_short_of_a_clean_start_creates_the_extension(
    monkeypatch, kwargs, raises
):
    """One property over every non-starting state: the database is left untouched.

    The decision table asserts this per cell, and the individual tests above assert it
    for the paths they cover -- but both do so as a detail of some other claim, which
    is how the undetermined state came to issue a ``CREATE EXTENSION`` while every test
    stayed green. Stated once, over a list built from the table itself, a new
    non-starting state cannot be added without either satisfying this or failing here.

    ``raises`` is part of the case rather than inferred, because the two shapes are not
    interchangeable: a refusal is a startup failure the operator sees, while the
    undetermined state returns quietly. Collapsing them would let a refusal silently
    become a quiet skip, or the reverse.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(**kwargs)

    if raises:
        with pytest.raises(RuntimeError):
            await PostgreSQLDB.configure_age_extension(conn)
    else:
        await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls() == [], conn.sql_calls()


# ---------------------------------------------------------------------------
# The other half of that invariant: which states are allowed to start at all
# ---------------------------------------------------------------------------
#
# ``_non_starting_states`` iterates a list of states already believed not to start, so
# it can only catch a state that stops issuing the CREATE it was supposed to withhold.
# It structurally cannot catch a state that *starts* when it should not have -- and
# that is the exact shape of every fail-open this gate has had. The sweep below is
# built the other way round: it enumerates the whole state space, computes what the
# policy says about each cell, and compares. A new fail-open cannot be introduced
# without disagreeing with the predicate.

_SOURCE_CATEGORIES = (_SAFE, _UNSAFE, _UNPARSEABLE, _ABSENT, _UNREADABLE)

# The AVAILABLE axis has one category the INSTALLED axis cannot have: a row that exists
# and reports NULL. ``pg_extension`` is read with fetchval and its ``extversion`` is not
# nullable in any state this gate can act on, so there is no such cell to enumerate on
# that side; ``default_version`` is nullable, and the difference between that row and no
# row is the difference between AGE being on disk and not. The axes are therefore
# deliberately asymmetric -- squaring a single category tuple, as this sweep first did,
# is what left the NULL row outside the enumerated space entirely.
_AVAILABLE_CATEGORIES = _SOURCE_CATEGORIES + (_NULL_DEFAULT_VERSION,)

_ALL_SOURCE_STATES = [
    (installed, available)
    for installed in _SOURCE_CATEGORIES
    for available in _AVAILABLE_CATEGORIES
]

_ALL_SOURCE_STATE_IDS = [
    f"installed={_category_id(i)}-available={_category_id(a)}"
    for i, a in _ALL_SOURCE_STATES
]


def _state_kwargs(installed, available) -> dict:
    """``_FakeConnection`` kwargs for one cell of the state space.

    A category is either a value the lookup returned or, for ``_UNREADABLE``, a
    non-transient failure of that lookup. The errors are the real ones used elsewhere
    in this module: a missing catalog for the INSTALLED query, and the malformed
    control file -- the failure that actually happens in the field -- for AVAILABLE.
    ``_NULL_DEFAULT_VERSION`` needs no special case here: it is a value the fake's
    AVAILABLE lookup knows how to report, which is the point of it being one.
    """
    kwargs: dict = {}
    if installed is _UNREADABLE:
        kwargs["installed_error"] = RuntimeError("relation does not exist")
    else:
        kwargs["installed"] = installed
    if available is _UNREADABLE:
        kwargs["available_error"] = _MALFORMED_CONTROL_FILE_ERROR
    else:
        kwargs["available"] = available
    return kwargs


def _may_start_cleanly(installed, available) -> bool:
    """The policy, stated once as a predicate rather than as a list of cells.

    Without the override, the gate may create the extension only when *every* source
    either reported a version it could read and verify as safe, or reported nothing at
    all because AGE is genuinely not there. Anything else -- an unsafe reading, a value
    it could not parse, or a lookup that never answered -- is an unverified version,
    and an unverified version is not a safe one.

    Note what this deliberately does not distinguish: ``_UNPARSEABLE``, ``_UNREADABLE``
    and ``_NULL_DEFAULT_VERSION`` are all simply "not safe". Treating them differently
    is what each fail-open did -- the last of them by routing a NULL ``default_version``
    into ``_ABSENT``, the one non-safe-looking category that is allowed to start.
    """
    return installed in (_SAFE, _ABSENT) and available in (_SAFE, _ABSENT)


@pytest.mark.parametrize(
    "installed, available", _ALL_SOURCE_STATES, ids=_ALL_SOURCE_STATE_IDS
)
@pytest.mark.asyncio
async def test_only_a_verified_safe_or_absent_state_starts_the_gate(
    monkeypatch, installed, available
):
    """Exactly which states start, over the whole 5x5 space, with no override.

    ``started`` is measured as "a CREATE EXTENSION was issued", not as "no exception
    escaped", because the two differ: the undetermined cell returns quietly *without*
    creating, and a test that only watched for exceptions would score it as a start.
    Refusals are absorbed so that every cell reaches the same assertion -- what is
    under test here is the start/no-start partition, and the refusal *messages* are
    pinned by the decision table above.
    """
    monkeypatch.delenv(AGE_ALLOW_UNSUPPORTED_ENV, raising=False)
    conn = _FakeConnection(**_state_kwargs(installed, available))

    with contextlib.suppress(RuntimeError):
        await PostgreSQLDB.configure_age_extension(conn)

    started = bool(conn.create_extension_calls())
    assert started is _may_start_cleanly(installed, available), conn.sql_calls()


@pytest.mark.parametrize(
    "installed, available", _ALL_SOURCE_STATES, ids=_ALL_SOURCE_STATE_IDS
)
@pytest.mark.asyncio
async def test_override_starts_from_every_state(monkeypatch, installed, available):
    """And with the override, the partition collapses: every state starts.

    The counterpart to the sweep above, and the general form of
    ``test_override_bypasses_every_refusal``: the escape hatch is documented as
    reaching every refusal, so a new refusal that forgets to consult it -- or a new
    quiet-skip path that the operator cannot turn off -- fails here rather than in
    whichever single test happened to cover that cell.
    """
    _allow_unsupported(monkeypatch)
    conn = _FakeConnection(**_state_kwargs(installed, available))

    await PostgreSQLDB.configure_age_extension(conn)

    assert conn.create_extension_calls(), conn.sql_calls()


def test_the_starting_sweep_covers_the_decision_table_and_more():
    """The two sweeps must stay in agreement, and the wider one must stay wider.

    If ``_may_start_cleanly`` ever drifted from ``_DECISION_TABLE`` the two would quietly
    pin different policies, and the fail-open would have somewhere to hide again. The
    inequality is the other half: the sweep has to keep containing the states that are
    not table cells at all, since that is where the live fail-open lived.
    """
    for installed, available, expected in _DECISION_TABLE:
        assert _may_start_cleanly(installed, available) is (expected == _START), (
            installed,
            available,
            expected,
        )

    assert len(_ALL_SOURCE_STATES) == 30
    assert {(i, a) for i, a, _ in _DECISION_TABLE} < set(_ALL_SOURCE_STATES)
    # The two states the table cannot express are both present, and both live: an
    # unreadable lookup and a row reporting no version. Each of them was a fail-open
    # until the sweep was widened to reach it, so the sweep has to keep reaching them.
    assert (_ABSENT, _UNREADABLE) in _ALL_SOURCE_STATES
    assert (_SAFE, _NULL_DEFAULT_VERSION) in _ALL_SOURCE_STATES


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
