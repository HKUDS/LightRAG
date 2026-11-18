"""Retry of the Apache AGE label-cache dangling pointer error in PGGraphStorage.

AGE 1.7.x can make a graph write fail with ``relation "<graph>.<garbage>" does
not exist``: asyncpg raises ``UndefinedTableError`` when the garbage is valid
UTF-8 and ``UnicodeDecodeError`` when it is not. The four retried write paths
(``upsert_node``, ``upsert_edge``, ``_upsert_node_chunk``, ``_upsert_edge_chunk``)
must retry that error and only that error, which needs each path to put its
graph name on the PGGraphQueryException it raises.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from tenacity import wait_none

from lightrag.kg.postgres_impl import (
    PGGraphEdgeWriteLostError,
    PGGraphQueryException,
    PGGraphStorage,
    _is_transient_graph_write_error,
)

pytestmark = pytest.mark.offline

GRAPH = "test_graph"


def _make_graph_storage() -> PGGraphStorage:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "chunk_entity_relation"
    storage.graph_name = GRAPH
    storage.__post_init__()
    storage.db = MagicMock()
    return storage


def _undefined_table(relation: str) -> asyncpg.exceptions.UndefinedTableError:
    return asyncpg.exceptions.UndefinedTableError(
        f'relation "{relation}" does not exist'
    )


def _undecodable(schema: str) -> UnicodeDecodeError:
    """The error asyncpg raises when the server's message is not valid UTF-8."""
    head = f'relation "{schema}.'.encode()
    raw = head + b'\x9c\xff" does not exist'
    return UnicodeDecodeError("utf-8", raw, len(head), len(head) + 1, "invalid")


DANGLING_SHAPES = {
    "undefined_table": lambda: _undefined_table(f"{GRAPH}.é\u0002garbage"),
    "unicode_decode": lambda: _undecodable(GRAPH),
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _wrapped(cause: BaseException, graph_name: str | None = GRAPH):
    exc = PGGraphQueryException({"message": "write failed", "graph_name": graph_name})
    exc.__cause__ = cause
    return exc


@pytest.mark.parametrize("shape", sorted(DANGLING_SHAPES))
def test_dangling_label_error_in_graph_schema_is_transient(shape):
    assert _is_transient_graph_write_error(_wrapped(DANGLING_SHAPES[shape]())) is True


@pytest.mark.parametrize(
    "cause",
    [
        pytest.param(_undefined_table("other_graph.garbage"), id="other-schema"),
        pytest.param(
            _undefined_table(f"{GRAPH}_2.garbage"), id="schema-sharing-a-prefix"
        ),
        pytest.param(_undefined_table("garbage"), id="unqualified"),
        pytest.param(_undefined_table(f"{GRAPH}.base"), id="real-label-base"),
        pytest.param(_undefined_table(f"{GRAPH}.DIRECTED"), id="real-label-DIRECTED"),
        pytest.param(
            _undefined_table(f"{GRAPH}._ag_label_vertex"), id="real-label-vertex"
        ),
        pytest.param(_undefined_table(f"{GRAPH}._ag_label_edge"), id="real-label-edge"),
        pytest.param(
            asyncpg.PostgresError(f'relation "{GRAPH}.garbage" does not exist'),
            id="not-undefined-table-error",
        ),
        pytest.param(_undecodable("other_graph"), id="undecodable-other-schema"),
        pytest.param(
            UnicodeDecodeError("utf-8", b"\x9c some other payload", 0, 1, "invalid"),
            id="undecodable-unrelated",
        ),
    ],
)
def test_other_errors_are_not_transient(cause):
    assert _is_transient_graph_write_error(_wrapped(cause)) is False


@pytest.mark.parametrize("shape", sorted(DANGLING_SHAPES))
def test_dangling_label_error_without_graph_name_is_not_transient(shape):
    """A wrapper that does not name its graph cannot be matched, so it fails fast."""
    cause = DANGLING_SHAPES[shape]()
    assert _is_transient_graph_write_error(_wrapped(cause, graph_name=None)) is False
    bare = PGGraphQueryException("write failed")
    bare.__cause__ = cause
    assert _is_transient_graph_write_error(bare) is False


def test_edge_write_lost_error_keeps_its_graph_name():
    err = PGGraphEdgeWriteLostError(GRAPH, "Alice", "Bob", ("Bob",))
    assert err.graph_name == GRAPH
    assert _is_transient_graph_write_error(err) is False


# ---------------------------------------------------------------------------
# The four retried write paths
# ---------------------------------------------------------------------------


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    """Accepts every statement; ``fetch`` returns a row so edge writes succeed."""

    def __init__(self):
        self.statements: list[str] = []

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql, *args):
        self.statements.append(sql)
        return ""

    async def fetch(self, sql, *args):
        self.statements.append(sql)
        return [{"r": "edge"}]


@dataclass
class _Attempts:
    calls: int = 0
    succeeded: bool = False


@dataclass(frozen=True)
class _WritePath:
    retried_method: str
    install: Callable[
        [PGGraphStorage, Callable[[], BaseException], int, _Attempts], None
    ]
    invoke: Callable[[PGGraphStorage], Awaitable[Any]]


def _fail_then_run_operation(storage, make_cause, failures, attempts):
    """Fake ``db._run_with_retry``: raise ``failures`` times, then run the operation."""

    async def fake_run_with_retry(operation, **_kwargs):
        attempts.calls += 1
        if attempts.calls <= failures:
            raise make_cause()
        conn = _FakeConnection()
        await operation(conn)
        attempts.succeeded = bool(conn.statements)

    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)


def _fail_then_execute(storage, make_cause, failures, attempts):
    """Fake ``db.execute`` (what ``_query`` calls): raise, then succeed."""

    async def fake_execute(*_args, **_kwargs):
        attempts.calls += 1
        if attempts.calls <= failures:
            raise make_cause()
        attempts.succeeded = True

    storage.db.execute = AsyncMock(side_effect=fake_execute)


NODE = ("Alice", {"entity_id": "Alice", "entity_type": "person"})
EDGE = ("Alice", "Bob", {"weight": "1.0"})

WRITE_PATHS = {
    "upsert_node": _WritePath(
        "upsert_node", _fail_then_execute, lambda s: s.upsert_node(*NODE)
    ),
    "upsert_edge": _WritePath(
        "upsert_edge", _fail_then_run_operation, lambda s: s.upsert_edge(*EDGE)
    ),
    "upsert_nodes_batch": _WritePath(
        "_upsert_node_chunk",
        _fail_then_run_operation,
        lambda s: s.upsert_nodes_batch([NODE]),
    ),
    "upsert_edges_batch": _WritePath(
        "_upsert_edge_chunk",
        _fail_then_run_operation,
        lambda s: s.upsert_edges_batch([EDGE]),
    ),
}


def _prepare(monkeypatch, path_name, make_cause, failures):
    path = WRITE_PATHS[path_name]
    monkeypatch.setattr(
        getattr(PGGraphStorage, path.retried_method).retry, "wait", wait_none()
    )
    storage = _make_graph_storage()
    attempts = _Attempts()
    path.install(storage, make_cause, failures, attempts)
    return path, storage, attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(DANGLING_SHAPES))
@pytest.mark.parametrize("path_name", sorted(WRITE_PATHS))
async def test_write_path_retries_dangling_label_error(monkeypatch, path_name, shape):
    path, storage, attempts = _prepare(
        monkeypatch, path_name, DANGLING_SHAPES[shape], failures=10
    )

    with pytest.raises(PGGraphQueryException) as excinfo:
        await path.invoke(storage)

    assert type(excinfo.value.__cause__) is type(DANGLING_SHAPES[shape]())
    assert excinfo.value.graph_name == GRAPH
    assert attempts.calls == 3
    assert attempts.succeeded is False


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(DANGLING_SHAPES))
@pytest.mark.parametrize("path_name", sorted(WRITE_PATHS))
async def test_write_path_recovers_after_one_dangling_label_error(
    monkeypatch, path_name, shape
):
    path, storage, attempts = _prepare(
        monkeypatch, path_name, DANGLING_SHAPES[shape], failures=1
    )

    await path.invoke(storage)

    assert attempts.calls == 2
    assert attempts.succeeded is True


@pytest.mark.asyncio
@pytest.mark.parametrize("path_name", sorted(WRITE_PATHS))
async def test_write_path_fails_fast_on_missing_real_label(monkeypatch, path_name):
    path, storage, attempts = _prepare(
        monkeypatch, path_name, lambda: _undefined_table(f"{GRAPH}.base"), failures=10
    )

    with pytest.raises(PGGraphQueryException) as excinfo:
        await path.invoke(storage)

    assert isinstance(excinfo.value.__cause__, asyncpg.exceptions.UndefinedTableError)
    assert attempts.calls == 1
