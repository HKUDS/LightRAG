"""think= is validated against the installed ollama package at server startup.

think= (OLLAMA_LLM_THINK / {ROLE}_OLLAMA_LLM_THINK) is gated on the binding's
declared ollama floor (>=0.5.4). The
check itself is cheap installed-package metadata, but where it fires matters:
raising only inside ``_ollama_model_if_cache`` means a server boots happily and
then dies on the first LLM call -- potentially hours into a document run, once
per role, with the failure surfacing as a pipeline error rather than a
configuration one. Validating at every option-resolution chokepoint in
``create_app`` turns it into a boot-time refusal instead.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.offline


# Cleared so a developer-local .env cannot decide the outcome (lightrag.api.config
# calls load_dotenv at import time).
_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "OLLAMA_LLM_THINK",
    "EXTRACT_OLLAMA_LLM_THINK",
    "KEYWORD_OLLAMA_LLM_THINK",
    "QUERY_OLLAMA_LLM_THINK",
    "VLM_OLLAMA_LLM_THINK",
    "EXTRACT_LLM_BINDING",
    "EXTRACT_LLM_MODEL",
    "EXTRACT_LLM_BINDING_API_KEY",
    "KEYWORD_LLM_BINDING",
    "QUERY_LLM_BINDING",
    "VLM_LLM_BINDING",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")

    import lightrag.api.config as config

    # Restore whatever the singleton held rather than blanking it: a blanked
    # proxy lazily re-parses sys.argv, which under pytest is the pytest command
    # line -- argparse then SystemExits inside whichever unrelated test touches
    # the config next.
    saved_args, saved_initialized = config._global_args, config._initialized
    config._global_args = None
    config._initialized = False
    try:
        yield
    finally:
        config._global_args = saved_args
        config._initialized = saved_initialized


def _create_app(monkeypatch, *, supports_think: bool):
    """Run create_app the way the server does, with storage and the installed
    ollama version both faked out."""
    import lightrag.llm.ollama as ollama_binding
    from lightrag.api.config import initialize_config, parse_args

    monkeypatch.setattr(ollama_binding, "_OLLAMA_SUPPORTS_THINK", supports_think)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    args = parse_args()
    # create_app reads the module-level global_args proxy, which would otherwise
    # lazily re-parse pytest's argv.
    initialize_config(args, force=True)

    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        from lightrag.api.lightrag_server import create_app

        return create_app(args)


def test_global_think_is_refused_at_startup_when_ollama_is_too_old(monkeypatch):
    monkeypatch.setenv("OLLAMA_LLM_THINK", "false")

    with pytest.raises(RuntimeError, match="ollama>=0.5.4"):
        _create_app(monkeypatch, supports_think=False)


def test_role_think_is_refused_at_startup_and_names_the_role(monkeypatch):
    """A role can enable think= while the base binding never touches it, so
    resolution has to be checked per role -- and the message must say which
    one, otherwise there are five variables to go hunting through."""
    monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", "false")

    with pytest.raises(RuntimeError, match=r"LLM role 'extract'"):
        _create_app(monkeypatch, supports_think=False)


def test_cross_provider_role_think_is_refused_at_startup(monkeypatch):
    """The base binding never touching ollama is not a reason to skip the
    check: a role can point at ollama on its own, and its options are resolved
    from the env directly rather than from the base binding's argparse group."""
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EXTRACT_LLM_BINDING", "ollama")
    # parse_args refuses a cross-provider role without its own model/key.
    monkeypatch.setenv("EXTRACT_LLM_MODEL", "qwen3")
    monkeypatch.setenv("EXTRACT_LLM_BINDING_API_KEY", "dummy")
    monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", "false")

    with pytest.raises(RuntimeError, match=r"LLM role 'extract'"):
        _create_app(monkeypatch, supports_think=False)


def test_a_reasoning_level_is_refused_at_startup_too(monkeypatch):
    """think is on/off *or* a named reasoning level, and the level form is the
    one an old package fails on most confusingly (a pydantic error about a
    string, far from the .env line that set it). It must be caught by the same
    startup check, and the message must quote the value."""
    monkeypatch.setenv("OLLAMA_LLM_THINK", "high")

    with pytest.raises(RuntimeError, match="'high'"):
        _create_app(monkeypatch, supports_think=False)


def test_startup_succeeds_when_the_installed_ollama_supports_think(monkeypatch):
    monkeypatch.setenv("OLLAMA_LLM_THINK", "false")
    monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", "false")

    assert _create_app(monkeypatch, supports_think=True) is not None


def test_startup_succeeds_on_an_old_ollama_when_think_is_unconfigured(monkeypatch):
    """0.5.4 is the declared floor, but an in-place upgrade can still leave an
    older package installed (the import-time auto-install only installs a
    missing ollama, it never upgrades an outdated one). Such an environment
    must keep working for everything that doesn't ask for think=."""
    assert _create_app(monkeypatch, supports_think=False) is not None
