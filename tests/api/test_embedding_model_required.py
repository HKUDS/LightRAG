"""The server refuses to start without a named embedding model.

Every vector storage records the embedding model's name beside its vectors,
and that marker is the only thing that detects a later switch to a *different
model of the same dimension* — the one change no dimension check can see, and
the one that returns confidently wrong neighbours with no error anywhere.

A server with no `EMBEDDING_MODEL` can never record that name, so every
container it provisions is unprotected for life. LightRAG supports neither
multi-process configuration propagation nor rolling updates, so an embedding
change is always stop → `lightrag-rebuild-vdb` → restart; a deployment that
cannot say which model wrote its vectors has no safe way through that.

Only the SERVER refuses. The library stays usable without a model name, and
`lightrag-rebuild-vdb` in particular must keep running — it is the way out.
"""

import sys

import pytest

pytestmark = pytest.mark.offline


def _parse_args():
    """Build server args, then publish them before the server is imported.

    ``lightrag.api.lightrag_server``'s dependency graph reads ``global_args``
    at import time, so an uninitialized config makes that import re-run
    ``parse_args()`` against pytest's own argv. Every API test that builds a
    server does this; see ``tests/api/test_workspace_entry_mount.py``.
    """
    from lightrag.api.config import initialize_config, parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)
    return args


@pytest.fixture
def server_args(monkeypatch):
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    return _parse_args


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_create_app_refuses_without_an_embedding_model(
    monkeypatch, server_args, configured
):
    """Unset, empty and whitespace are the same condition: no name to record."""
    if configured is None:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_MODEL", configured)
    args = server_args()

    from lightrag.api.lightrag_server import create_app

    with pytest.raises(SystemExit) as excinfo:
        create_app(args)

    message = str(excinfo.value)
    assert "EMBEDDING_MODEL" in message
    # The message has to carry the way out, not just the refusal.
    assert "lightrag-rebuild-vdb" in message


def test_the_refusal_names_why_a_rolling_change_is_not_an_option(
    monkeypatch, server_args
):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    args = server_args()

    from lightrag.api.lightrag_server import create_app

    with pytest.raises(SystemExit) as excinfo:
        create_app(args)

    assert "rolling" in str(excinfo.value)


def test_an_args_object_without_the_field_is_refused_too():
    """No usable name is no usable name, however it came to be missing.

    ``create_app`` tolerates a missing ``default_ui`` because that one is a
    cosmetic default. This is a safety guard, and a rule with an exemption for
    "the attribute was never set" is a rule with a bypass — so the guard asks
    only whether it has a model name to record, not how the args were built.
    The server's own parser always sets the field, so nothing real is caught by
    the difference.
    """
    from argparse import Namespace

    from lightrag.api.lightrag_server import create_app

    with pytest.raises(SystemExit, match="EMBEDDING_MODEL"):
        create_app(Namespace())
