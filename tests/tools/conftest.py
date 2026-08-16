"""Pytest fixtures shared by the ``lightrag/tools`` CLI tests.

These tests drive real CLI entry points, which read workspace configuration
straight out of the ambient environment. Without neutralization the developer's
own deployment decides what the assertions see.
"""

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
