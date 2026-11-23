"""Shared fixtures for the API server tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from lightrag.api import config


class LightRAGConstructed(Exception):
    """Stops startup once the stubbed LightRAG constructor captured its kwargs."""


# The smallest environment create_app starts with offline. Tests override
# entries through ``prepare_startup(env)``; a value of None deletes the key.
_STARTUP_ENV = {
    "LLM_BINDING": "openai",
    "EMBEDDING_BINDING": "openai",
    "RERANK_BINDING": "null",
    "UI_TEMPLATES_DIR": "",
    "AUTH_ACCOUNTS": "",
}


@pytest.fixture
def prepare_startup(monkeypatch, tmp_path):
    """Parse the server config and replace ``LightRAG`` with a recorder.

    ``prepare_startup(env)`` returns a namespace with ``server`` (the
    ``lightrag_server`` module), ``args`` (the parsed config) and ``captured``
    (the kwargs of every constructor call). ``constructor_kwargs()`` runs
    ``create_app`` up to the constructor and returns the kwargs it received.
    """

    def prepare(env: dict[str, str | None] | None = None) -> SimpleNamespace:
        monkeypatch.setattr(config, "_global_args", None)
        monkeypatch.setattr(config, "_initialized", False)
        for key, value in {**_STARTUP_ENV, **(env or {})}.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
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

        captured: list[dict] = []

        def construct(**kwargs):
            captured.append(kwargs)
            raise LightRAGConstructed

        monkeypatch.setattr(server, "LightRAG", construct)
        monkeypatch.setattr(server, "check_frontend_build", lambda: (False, False))
        monkeypatch.setattr(server, "check_workspace_frontend_build", lambda: False)

        def constructor_kwargs() -> dict:
            with pytest.raises(LightRAGConstructed):
                server.create_app(args)
            assert len(captured) == 1
            return captured[0]

        return SimpleNamespace(
            server=server,
            args=args,
            captured=captured,
            constructor_kwargs=constructor_kwargs,
        )

    return prepare
