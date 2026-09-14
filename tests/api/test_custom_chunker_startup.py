"""Exercise the shared uvicorn/Gunicorn app factory, not just a helper."""

import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightrag.api import config
from lightrag.chunker import plugins, registry

pytestmark = pytest.mark.offline


@pytest.fixture
def startup(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_global_args", None)
    monkeypatch.setattr(config, "_initialized", False)
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DUPLICATES", set())
    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "_failures", [])
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)
    monkeypatch.setenv("CUSTOM_CHUNKER", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("UI_TEMPLATES_DIR", "")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightrag-server",
            "--working-dir",
            str(tmp_path / "rag"),
            "--input-dir",
            str(tmp_path / "inputs"),
        ],
    )
    args = config.initialize_config(force=True)
    from lightrag.api import lightrag_server as server

    captured = []

    class Constructed(Exception):
        pass

    def construct(**kwargs):
        captured.append(kwargs)
        raise Constructed

    monkeypatch.setattr(server, "LightRAG", construct)
    monkeypatch.setattr(server, "check_frontend_build", lambda: (False, False))
    monkeypatch.setattr(server, "check_workspace_frontend_build", lambda: False)
    return server, args, captured, Constructed


@pytest.mark.parametrize("factory", ["create_app", "get_application"])
def test_discovered_chunker_injected_before_constructor(startup, monkeypatch, factory):
    server, args, captured, constructed = startup

    def callback(*args):
        return args

    monkeypatch.setitem(
        sys.modules, "startup_chunker_impl", SimpleNamespace(chunk=callback)
    )

    def register():
        registry.register_chunker(
            registry.ChunkerSpec("acme", "startup_chunker_impl:chunk", "1", "Acme")
        )

    ep = SimpleNamespace(name="acme", value="provider:register", load=lambda: register)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [ep])

    def validate():
        assert registry.registered_chunker_names() == ("acme",)

    monkeypatch.setattr(server, "validate_parser_routing_config", validate)
    args.custom_chunker = "acme"
    with pytest.raises(constructed):
        getattr(server, factory)(args)
    assert captured[0]["chunking_func"](*range(6)) == tuple(range(6))


def test_unset_does_not_override_constructor_default(startup, monkeypatch):
    server, args, captured, constructed = startup
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    with pytest.raises(constructed):
        server.create_app(args)
    assert "chunking_func" not in captured[0]


@pytest.mark.parametrize(
    "selected,outcome",
    [
        (
            "acme",
            "selected chunker 'acme' validated; C and no-selector chunking remain available",
        ),
        ("partial", "CUSTOM_CHUNKER='partial' could not be activated; startup aborted"),
        (
            "",
            "CUSTOM_CHUNKER is unset; built-in callback and existing C admission/fallback are unchanged",
        ),
    ],
)
def test_failed_provider_log_alone_describes_selected_chunker_outcome(
    startup, monkeypatch, caplog, selected, outcome
):
    server, args, captured, constructed = startup
    monkeypatch.setitem(
        sys.modules, "startup_chunker_impl", SimpleNamespace(chunk=lambda *args: args)
    )

    def good():
        registry.register_chunker(
            registry.ChunkerSpec("acme", "startup_chunker_impl:chunk", "1", "Acme")
        )

    def broken():
        registry.register_chunker(
            registry.ChunkerSpec("partial", "absent:chunk", "1", "Partial")
        )
        raise RuntimeError("provider failed")

    entries = [
        SimpleNamespace(name=name, value=f"{name}:register", load=lambda fn=fn: fn)
        for name, fn in [("good", good), ("broken", broken)]
    ]
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: entries)
    args.custom_chunker = selected
    with pytest.raises(ValueError if selected == "partial" else constructed):
        server.create_app(args)
    errors = [
        r.getMessage()
        for r in caplog.records
        if "skipped failed plugin" in r.getMessage()
    ]
    assert len(errors) == 1
    assert "broken:register" in errors[0] and "provider failed" in errors[0]
    assert outcome in errors[0]
    if selected == "partial":
        assert not captured
    elif selected:
        assert captured[0]["chunking_func"](*range(6)) == tuple(range(6))


@pytest.mark.parametrize(
    "name,match", [("missing", "unknown chunker"), ("pkg:func", "import paths")]
)
def test_invalid_selection_fails_before_constructor(startup, monkeypatch, name, match):
    server, args, captured, _ = startup
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    args.custom_chunker = name
    with pytest.raises(ValueError, match=match):
        server.create_app(args)
    assert captured == []


def test_ingress_error_names_config_and_registered_choices(monkeypatch):
    from lightrag.api.routers.document_routes import _validate_custom_chunking_available
    from lightrag.chunker import chunking_by_token_size

    monkeypatch.setattr(registry, "_REGISTRY", {})
    registry.register_chunker(registry.ChunkerSpec("acme", "absent:chunk", "1", "Acme"))
    with pytest.raises(ValueError, match="CUSTOM_CHUNKER.*acme"):
        _validate_custom_chunking_available(
            "C", SimpleNamespace(chunking_func=chunking_by_token_size)
        )


@pytest.mark.parametrize(
    "name,match",
    [
        ("missing-chunker", "unknown chunker"),
        ("pkg:func", "import paths are not supported"),
    ],
)
def test_invalid_cli_selection_exits_nonzero(monkeypatch, name, match):
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("UI_TEMPLATES_DIR", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lightrag.api.config import parse_args; from lightrag.api.lightrag_server import create_app; create_app(parse_args())",
            "--custom-chunker",
            name,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert match in result.stderr
