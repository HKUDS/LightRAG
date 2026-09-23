"""Pytest fixtures shared by the ``lightrag/tools`` CLI tests.

These tests drive real CLI entry points, which read workspace configuration
straight out of the ambient environment. Without neutralization the developer's
own deployment decides what the assertions see.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic_workspace_env(monkeypatch):
    """Pin the workspace-selecting env vars so a local ``.env`` cannot leak in.

    ``migrate_graph_storage.main()`` calls ``load_dotenv(".env",
    override=False)`` before parsing arguments, and ``--workspace`` defaults to
    ``os.getenv("WORKSPACE", "")``. A developer ``.env`` carrying
    ``WORKSPACE=space1`` therefore becomes the default ``--workspace`` for every
    CLI test, which then trips the tool's own workspace cross-checks against
    fakes that resolved to ``""`` — a failure that reproduces only on machines
    with that ``.env``.

    ``delenv`` alone does not hold, for the reason already documented on
    ``tests/conftest.py::_hermetic_mineru_env``: ``override=False`` fills every
    name that is *unset*, so a deleted variable is silently repopulated from
    ``.env`` on the next ``load_dotenv``. An explicit empty string survives it,
    and an empty workspace is exactly the "operator named nothing" state the
    tools treat as neutral.

    ``POSTGRES_WORKSPACE`` is pinned for the same reason and matters more: it
    outranks ``--workspace`` in the backends' resolution order, so a leaked
    value makes ``_precheck_requested_workspace`` refuse runs the test expected
    to proceed. Monkeypatch restores the developer's real values at teardown.
    """
    monkeypatch.setenv("WORKSPACE", "")
    monkeypatch.setenv("POSTGRES_WORKSPACE", "")


@pytest.fixture(autouse=True)
def _hermetic_working_dir(monkeypatch, tmp_path):
    """Point every tool at a private ``WORKING_DIR`` and hand back its claims.

    ``RebuildTool.setup_storages()`` and the clear-storage tool resolve their
    configuration directory from ``WORKING_DIR`` / ``LIGHTRAG_CONFIG_DIR`` and
    take the single-server ``flock`` on it before anything else. Left at the
    default that is the developer's own ``./rag_storage``: a ``lightrag-server``
    or ``lightrag-clear-storage`` running beside the test run holds that lock,
    and every test that drives the real ``setup_storages()`` is refused with
    "already in use" -- a failure caused by a process the test never started.
    Two pytest workers collide on it the same way.

    ``setenv`` rather than ``delenv`` for the ``load_dotenv(override=False)``
    reason documented on ``_hermetic_workspace_env``; the empty
    ``LIGHTRAG_CONFIG_DIR`` is the "use the default under WORKING_DIR" spelling.

    The claim is per process tree and released only by the tools' own
    ``run()``, which these tests bypass, so any claim a test leaves behind is
    given back here. Only claims the test ADDED are touched: a test that
    manipulates the claim table itself restores it in its own ``finally``.
    """
    from lightrag.kg import working_dir_lock as wdl

    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag_storage"))
    monkeypatch.setenv("LIGHTRAG_CONFIG_DIR", "")

    before = set(wdl._claims)
    yield
    for lock_path in set(wdl._claims) - before:
        while lock_path in wdl._claims:
            wdl.release_working_dir_lock(os.path.dirname(lock_path))
