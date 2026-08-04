"""Backend factories for the shared doc_status scheduling contract suite.

LR2 §13.1 requires the five built-in backends to share ONE parametrized contract
suite. Five independently written per-backend files were the previous state, and
they drift: a backend passes its own file while violating an invariant only some
other file happens to check.

JSON needs nothing and always runs. The other four need their service; each is
probed once per session and its parameters skip when it is unreachable, so the
suite is useful offline and complete when the services are up (see the module
docstring of ``test_doc_status_scheduling_contract`` for how to start them).
"""

from __future__ import annotations

import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import pytest


class _DummyEmbeddingFunc:
    """doc_status never embeds; the storage dataclasses just require the field."""

    embedding_dim = 4

    async def __call__(self, texts):  # pragma: no cover - never invoked
        return [[0.0] * self.embedding_dim for _ in texts]


def _reachable(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass(frozen=True)
class BackendSpec:
    """One row of the contract-suite parametrization."""

    name: str
    build: Callable[[Any], Awaitable[Any]]
    probe: Callable[[], bool]
    unavailable_hint: str

    def skip_unless_available(self) -> None:
        if not self.probe():
            pytest.skip(f"{self.name} unavailable: {self.unavailable_hint}")


def _unique_workspace() -> str:
    """A fresh workspace per test: the contract covers derived indexes and source
    multimaps, so leakage between cases would produce false Conflicts."""
    return f"lr2ct{uuid.uuid4().hex[:10]}"


@contextmanager
def _workspace_env_isolated(*env_names: str):
    """Neutralise the per-backend ``*_WORKSPACE`` overrides while a storage
    resolves its workspace, then restore them.

    Those env vars take PRIORITY over the constructor's ``workspace=`` argument
    (``"Using REDIS_WORKSPACE environment variable … overriding …"``). So on the
    one kind of machine these tests actually run on — an integration box with a
    configured ``.env`` — ``_unique_workspace()`` was silently discarded and
    every case landed in the SAME workspace: the derived-index and
    source-multimap cases would then report each other's rows as Conflicts, and
    one case's ``drop()`` would wipe another's data.

    The window only has to cover construction and ``initialize()``: both
    ``__post_init__`` (workspace) and the client setup (URI) read the
    environment there and keep the resolved values.
    """
    saved = {name: os.environ.pop(name, None) for name in env_names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _ensure_shared_data() -> None:
    """Every backend reaches shared_storage — JSON for its cross-process data
    proxy, the others for the keyed write locks their index maintenance takes."""
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()


async def _build_json(tmp_path) -> Any:
    from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage

    _ensure_shared_data()

    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=_unique_workspace(),
    )
    await storage.initialize()
    return storage


async def _build_redis(_tmp_path) -> Any:
    from lightrag.kg.redis_impl import RedisDocStatusStorage

    _ensure_shared_data()

    with _workspace_env_isolated("REDIS_WORKSPACE"):
        os.environ.setdefault("REDIS_URI", "redis://localhost:6379")
        storage = RedisDocStatusStorage(
            namespace="doc_status",
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=_unique_workspace(),
        )
        await storage.initialize()
    return storage


async def _build_postgres(_tmp_path) -> Any:
    from lightrag.kg.postgres_impl import ClientManager, PGDocStatusStorage

    _ensure_shared_data()

    # The PG pool is a process-wide singleton that pins its vector settings on
    # first use, so an earlier test in the same session (one that monkeypatched a
    # POSTGRES_HNSW_* / vector env var) makes every later request incompatible.
    # Clearing it is only safe while nobody holds it — assert that rather than
    # assume it, so a genuine leak shows up as a failure instead of a torn pool.
    if ClientManager._instances["vector_signature"] is not None:
        assert ClientManager._instances["ref_count"] == 0, (
            "another test still holds the PG pool; the contract fixture will not "
            "reset a pool that is in use"
        )
        ClientManager._instances["db"] = None
        ClientManager._instances["vector_signature"] = None

    # POSTGRES_WORKSPACE feeds the pool config too, so the isolation has to span
    # the client setup inside initialize(), not just __post_init__.
    with _workspace_env_isolated("POSTGRES_WORKSPACE"):
        storage = PGDocStatusStorage(
            namespace="doc_status",
            global_config={"embedding_batch_num": 8},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=_unique_workspace(),
        )
        await storage.initialize()
    return storage


async def _build_mongo(_tmp_path) -> Any:
    from lightrag.kg.mongo_impl import MongoDocStatusStorage

    _ensure_shared_data()

    with _workspace_env_isolated("MONGODB_WORKSPACE"):
        storage = MongoDocStatusStorage(
            namespace="doc_status",
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=_unique_workspace(),
        )
        await storage.initialize()
    return storage


async def _build_opensearch(_tmp_path) -> Any:
    from lightrag.kg.opensearch_impl import OpenSearchDocStatusStorage

    _ensure_shared_data()

    with _workspace_env_isolated("OPENSEARCH_WORKSPACE"):
        storage = OpenSearchDocStatusStorage(
            namespace="doc_status",
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=_unique_workspace(),
        )
        await storage.initialize()
    return storage


_SERVICE_BACKENDS: tuple[BackendSpec, ...] = (
    BackendSpec(
        name="redis",
        build=_build_redis,
        probe=lambda: _reachable("localhost", 6379),
        unavailable_hint="no Redis on localhost:6379",
    ),
    BackendSpec(
        name="postgres",
        build=_build_postgres,
        probe=lambda: _reachable("localhost", 5432),
        unavailable_hint="no PostgreSQL on localhost:5432",
    ),
    BackendSpec(
        name="mongodb",
        build=_build_mongo,
        probe=lambda: _reachable("localhost", 27017),
        unavailable_hint="no MongoDB on localhost:27017",
    ),
    BackendSpec(
        name="opensearch",
        build=_build_opensearch,
        probe=lambda: _reachable("localhost", 9200),
        unavailable_hint="no OpenSearch on localhost:9200",
    ),
)

_JSON_BACKEND = BackendSpec(
    name="json",
    build=_build_json,
    probe=lambda: True,
    unavailable_hint="never unavailable",
)

# Marked per PARAMETER, not per module: JSON needs no service, so it must run in
# the default offline suite — a contract test that only executes behind
# ``--run-integration`` would go stale between the runs that use it.
BACKEND_PARAMS = [
    pytest.param(_JSON_BACKEND, id=_JSON_BACKEND.name),
    *[
        pytest.param(
            spec,
            id=spec.name,
            marks=[pytest.mark.integration, pytest.mark.requires_db],
        )
        for spec in _SERVICE_BACKENDS
    ],
]

BACKEND_SPECS: tuple[BackendSpec, ...] = (_JSON_BACKEND, *_SERVICE_BACKENDS)
