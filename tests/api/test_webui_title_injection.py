"""WEBUI_TITLE reaches the browser tab title through the entry HTML.

The WebUI already showed `WEBUI_TITLE` in its header (via /health and the
login response), but the browser tab kept the bundled default because the
`<title>` in the entry HTML is static. The server now publishes the value as
`webuiTitle` in the runtime-config snippet it injects on every HTML response
and assigns it to `document.title` there, so the tab is right from the FIRST
paint — before the SPA boots and without a second title flashing by.

The staging approach (tmp index.html + patched `lightrag_server.__file__`)
mirrors `test_path_prefixes.py::TestRuntimeConfigInjection`.
"""

import json
import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


PLACEHOLDER = "<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"
STATIC_TITLE = "<title>LightRAG</title>"

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


def _entry_html(name: str) -> str:
    """Mirror what Vite emits: static <title> followed by the placeholder."""
    return (
        "<!doctype html><html><head>"
        f"{STATIC_TITLE}"
        f"{PLACEHOLDER}"
        f'<script type="module" crossorigin src="./assets/{name}-X.js"></script>'
        f"</head><body><div id=root>{name}</div></body></html>"
    )


def _stage_build(tmp_path):
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(exist_ok=True)
    (webui_dir / "index.html").write_text(_entry_html("index"), encoding="utf-8")
    (webui_dir / "workspace.html").write_text(
        _entry_html("workspace"), encoding="utf-8"
    )


def _client(tmp_path, monkeypatch, title):
    """Build the app with ``WEBUI_TITLE=title`` (None = variable unset).

    `webui_title` is resolved at server-module import time, so the module
    attribute — not the environment — is what a request-time injection reads.
    """
    _stage_build(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    from lightrag.api.config import parse_args
    from lightrag.api import lightrag_server

    monkeypatch.setattr(
        lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
    )
    monkeypatch.setattr(lightrag_server, "webui_title", title)

    args = parse_args()
    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        app = lightrag_server.create_app(args)
    return TestClient(app)


def _config_payload(body: str) -> dict:
    match = re.search(r"window\.__LIGHTRAG_CONFIG__ = (\{.*?\});", body)
    assert match is not None, body
    return json.loads(match.group(1))


class TestWebuiTitleInjection:
    def test_custom_title_is_published_and_applied(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "My Graph KB")

        body = client.get("/webui/").text
        assert _config_payload(body)["webuiTitle"] == "My Graph KB"
        # Applied inline, in the same <head>, so no bundle boot is needed.
        assert "document.title=window.__LIGHTRAG_CONFIG__.webuiTitle" in body
        # The static tag stays: it is the pre-injection default and the
        # fallback for a deployment that sets no title.
        assert STATIC_TITLE in body

    def test_both_entries_get_the_title(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "My Graph KB")

        for path in ("/webui/", "/workspace/"):
            body = client.get(path).text
            assert _config_payload(body)["webuiTitle"] == "My Graph KB", path

    def test_unset_title_leaves_the_static_default(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, None)

        body = client.get("/webui/").text
        # null, not "": the client tests one falsy value, and the guard in the
        # snippet must not overwrite the tab with an empty title.
        assert _config_payload(body)["webuiTitle"] is None
        assert STATIC_TITLE in body

    def test_empty_title_is_normalized_to_null(self, tmp_path, monkeypatch):
        """`WEBUI_TITLE=` (set but blank) must behave exactly like unset."""
        client = _client(tmp_path, monkeypatch, "")

        assert _config_payload(client.get("/webui/").text)["webuiTitle"] is None

    def test_title_cannot_break_out_of_the_script(self, tmp_path, monkeypatch):
        """A title containing `</script>` must not close the inline script.

        Defense in depth — the value comes from admin config, not user input —
        matching the existing `</` escaping on the prefixes.
        """
        client = _client(tmp_path, monkeypatch, "</script><script>alert(1)</script>")

        body = client.get("/webui/").text
        assert "</script><script>alert(1)" not in body
        assert "<\\/script>" in body
        # Still a single well-formed config script, and the value survives.
        payload = _config_payload(body.replace("<\\/", "</"))
        assert payload["webuiTitle"] == "</script><script>alert(1)</script>"

    def test_quotes_in_title_are_json_escaped(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, 'My "Graph" KB')

        payload = _config_payload(client.get("/webui/").text)
        assert payload["webuiTitle"] == 'My "Graph" KB'
