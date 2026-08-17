"""The cross-backend contract fixture really gets a fresh workspace per case.

``_unique_workspace()`` is passed as the constructor's ``workspace=``, but every
service backend gives its ``*_WORKSPACE`` env var HIGHER priority than that
argument. So on the only kind of machine these tests run on — an integration box
with a configured ``.env`` — the unique name was silently discarded and all five
backends' cases shared one workspace: the contract's derived-index and
source-multimap cases would report each other's rows as Conflicts, and one
case's ``drop()`` would wipe another's data.

Redis stands in for all four here: it resolves its workspace in
``__post_init__``, so the precedence is observable without a live service.
"""

from __future__ import annotations

import os

import pytest

from .conftest_scheduling_backends import _unique_workspace, _workspace_env_isolated

pytestmark = pytest.mark.offline


def _resolved_namespace(workspace: str) -> str:
    from lightrag.kg.redis_impl import RedisDocStatusStorage

    class _Embed:
        embedding_dim = 4

        async def __call__(self, texts):  # pragma: no cover - never invoked
            return [[0.0] * 4 for _ in texts]

    storage = RedisDocStatusStorage(
        namespace="doc_status",
        global_config={},
        embedding_func=_Embed(),
        workspace=workspace,
    )
    return storage.final_namespace


def test_the_env_override_really_does_beat_the_constructor(monkeypatch):
    """The premise of the fixture bug, pinned: without isolation the unique
    workspace argument is discarded."""
    monkeypatch.setenv("REDIS_WORKSPACE", "shared_by_everyone")
    assert _resolved_namespace(_unique_workspace()) == "shared_by_everyone_doc_status"


def test_isolation_restores_the_unique_workspace_and_puts_the_env_back(monkeypatch):
    monkeypatch.setenv("REDIS_WORKSPACE", "shared_by_everyone")

    unique = _unique_workspace()
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        assert "REDIS_WORKSPACE" not in os.environ
        assert _resolved_namespace(unique) == f"{unique}_doc_status"

    # Restored, so a later test that depends on the deployment's own workspace
    # is unaffected.
    assert os.environ["REDIS_WORKSPACE"] == "shared_by_everyone"


def test_isolation_leaves_an_unset_variable_unset(monkeypatch):
    monkeypatch.delenv("REDIS_WORKSPACE", raising=False)
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        pass
    assert "REDIS_WORKSPACE" not in os.environ


def test_two_cases_never_collide(monkeypatch):
    monkeypatch.setenv("REDIS_WORKSPACE", "shared_by_everyone")
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        first = _resolved_namespace(_unique_workspace())
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        second = _resolved_namespace(_unique_workspace())
    assert first != second
