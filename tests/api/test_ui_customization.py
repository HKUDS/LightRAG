"""Tests for the UI customization bundle (workspace-entry PRD §8).

Covers: strict manifest schema (brand.logo required, explicit null accepted),
locale normalization (case folding, underscore rejection), single-level
fallbacks + default_locale, whole-bundle fail-fast validation, path
containment, MIME sniffing, the public read API (200 customized:false without
a bundle; no-store without ETag; hash-addressed immutable assets), revision /
hash change boundaries, orthogonality to workspace.html availability, and
/health remaining free of customization fields.
"""

import json
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.api.ui_customization import (
    UICustomizationError,
    load_ui_customization_snapshot,
    normalize_locale,
)

PLACEHOLDER = "<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_DEFAULT_UI",
    "ENABLE_API_DOCS",
    "UI_TEMPLATES_DIR",
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
)  # magic + padding; content check is magic-based
SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'


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


def make_bundle(root, manifest=None, *, extra_files=None):
    """Stage a valid two-locale bundle; `manifest` overrides the default."""
    bundle = root / "ui_templates"
    (bundle / "assets").mkdir(parents=True, exist_ok=True)
    (bundle / "locales" / "en").mkdir(parents=True, exist_ok=True)
    (bundle / "locales" / "zh").mkdir(parents=True, exist_ok=True)

    (bundle / "assets" / "logo.svg").write_bytes(SVG_BYTES)
    (bundle / "locales" / "en" / "welcome.md").write_text("# Welcome EN", "utf-8")
    (bundle / "locales" / "en" / "query_empty.md").write_text("Ask EN", "utf-8")
    (bundle / "locales" / "zh" / "welcome.md").write_text("# 欢迎", "utf-8")
    (bundle / "locales" / "zh" / "query_empty.md").write_text("请提问", "utf-8")

    if manifest is None:
        manifest = {
            "schema_version": 1,
            "default_locale": "en",
            "fallbacks": {"ko": ["en"]},
            "brand": {"logo": "assets/logo.svg"},
            "locales": {
                "en": {
                    "welcome": "locales/en/welcome.md",
                    "query_empty": "locales/en/query_empty.md",
                    "logo_alt": "ACME",
                },
                "zh": {
                    "welcome": "locales/zh/welcome.md",
                    "query_empty": "locales/zh/query_empty.md",
                    "logo_alt": "ACME 中文",
                },
            },
        }
    (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    for rel, content in (extra_files or {}).items():
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, "utf-8")
    return bundle


# --------------------------------------------------------------------------- #
# Snapshot loader
# --------------------------------------------------------------------------- #
class TestLocaleNormalization:
    def test_case_folding(self):
        assert normalize_locale("zh-tw") == "zh-TW"
        assert normalize_locale("ZH-hant-TW") == "zh-Hant-TW"
        assert normalize_locale("EN") == "en"

    def test_underscore_rejected(self):
        with pytest.raises(ValueError, match="underscore"):
            normalize_locale("zh_TW")

    @pytest.mark.parametrize(
        "bad", ["", "  ", "a", "1x", "en-", "-en", "en--US", "x" * 40]
    )
    def test_invalid_forms_rejected(self, bad):
        with pytest.raises(ValueError):
            normalize_locale(bad)


class TestBundleValidation:
    def test_valid_bundle_loads(self, tmp_path):
        snapshot = load_ui_customization_snapshot(make_bundle(tmp_path))
        assert set(snapshot.locales) == {"en", "zh"}
        assert snapshot.default_locale == "en"
        assert snapshot.locales["en"].welcome == "# Welcome EN"
        assert snapshot.locales["en"].logo_asset_id == "brand-logo"
        assert len(snapshot.assets) == 1
        assert snapshot.bundle_revision

    def test_explicit_null_logo_accepted(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["brand"]["logo"] = None
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].logo_asset_id is None
        assert snapshot.assets == {}

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda m: m.pop("brand"), "missing required"),
            (lambda m: m["brand"].pop("logo"), "missing required"),
            (lambda m: m.update(schema_version=2), "schema_version"),
            (lambda m: m.update(unknown_key=1), "unknown field"),
            (lambda m: m.update(default_locale="fr"), "not a declared locale"),
            (lambda m: m["locales"]["en"].pop("query_empty"), "missing required"),
            (lambda m: m["locales"]["en"].update(logo_alt=""), "logo_alt"),
            (lambda m: m["locales"]["en"].update(surprise=1), "unknown field"),
            (lambda m: m.update(fallbacks={"ko": []}), "non-empty array"),
            (lambda m: m.update(fallbacks={"ko": ["fr"]}), "not a declared locale"),
            (lambda m: m.update(fallbacks={"ko": ["en", "en"]}), "duplicate target"),
            (
                lambda m: m["locales"].update(
                    {
                        "zh_TW": {
                            "welcome": "locales/zh/welcome.md",
                            "query_empty": "locales/zh/query_empty.md",
                            "logo_alt": "x",
                        }
                    }
                ),
                "underscore",
            ),
            (
                lambda m: m["locales"].update(
                    {
                        "ZH-tw": {
                            "welcome": "locales/zh/welcome.md",
                            "query_empty": "locales/zh/query_empty.md",
                            "logo_alt": "x",
                        }
                    }
                ),
                "normalized form",
            ),
        ],
    )
    def test_schema_violations_fail(self, tmp_path, mutate, match):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        mutate(manifest)
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match=match):
            load_ui_customization_snapshot(bundle)

    def test_missing_manifest_fails(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "manifest.json").unlink()
        with pytest.raises(UICustomizationError, match="manifest.json is missing"):
            load_ui_customization_snapshot(bundle)

    def test_missing_referenced_file_fails(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "locales" / "zh" / "welcome.md").unlink()
        with pytest.raises(UICustomizationError, match="does not exist"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize(
        "path", ["/etc/hostname", "../outside.md", "locales/../../outside.md"]
    )
    def test_escaping_paths_rejected(self, tmp_path, path):
        (tmp_path / "outside.md").write_text("outside", "utf-8")
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["en"]["welcome"] = path
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError):
            load_ui_customization_snapshot(bundle)

    def test_symlink_escape_rejected(self, tmp_path):
        secret = tmp_path / "secret.md"
        secret.write_text("secret", "utf-8")
        bundle = make_bundle(tmp_path)
        link = bundle / "locales" / "en" / "link.md"
        link.symlink_to(secret)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["en"]["welcome"] = "locales/en/link.md"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match="escapes"):
            load_ui_customization_snapshot(bundle)

    def test_template_size_limit(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "locales" / "en" / "welcome.md").write_text("x" * (64 * 1024 + 1))
        with pytest.raises(UICustomizationError, match="byte limit"):
            load_ui_customization_snapshot(bundle)

    def test_logo_mime_checked_by_content(self, tmp_path):
        bundle = make_bundle(tmp_path)
        # .svg extension but garbage content: rejected by content sniffing.
        (bundle / "assets" / "logo.svg").write_bytes(b"GIF89a not allowed")
        with pytest.raises(UICustomizationError, match="not PNG, JPEG, WebP or SVG"):
            load_ui_customization_snapshot(bundle)

    def test_locale_logo_override(self, tmp_path):
        bundle = make_bundle(tmp_path, extra_files={"assets/logo-zh.png": PNG_BYTES})
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["zh"]["logo"] = "assets/logo-zh.png"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].logo_asset_id == "logo-zh"
        assert snapshot.locales["en"].logo_asset_id == "brand-logo"
        mimes = {a.asset_id: a.mime for a in snapshot.assets.values()}
        assert mimes == {"brand-logo": "image/svg+xml", "logo-zh": "image/png"}


class TestRevisionBoundaries:
    def test_deterministic_across_loads(self, tmp_path):
        bundle = make_bundle(tmp_path)
        first = load_ui_customization_snapshot(bundle)
        second = load_ui_customization_snapshot(bundle)
        assert first.bundle_revision == second.bundle_revision
        assert {k: v.sha256 for k, v in first.assets.items()} == {
            k: v.sha256 for k, v in second.assets.items()
        }

    def test_text_change_moves_revision_but_not_logo_hash(self, tmp_path):
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle)
        (bundle / "locales" / "zh" / "welcome.md").write_text("# 新欢迎", "utf-8")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision
        assert set(before.assets) == set(after.assets)  # (hash, id) unchanged

    def test_logo_change_moves_both(self, tmp_path):
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle)
        (bundle / "assets" / "logo.svg").write_bytes(SVG_BYTES + b"<!--v2-->")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision
        assert set(before.assets) != set(after.assets)
        # The semantic asset id is stable; only the hash moved.
        assert {a.asset_id for a in after.assets.values()} == {"brand-logo"}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
class _FakeLightRAG:
    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def get_llm_role_config(self):
        return {}

    async def get_llm_queue_status(self, include_base=True):
        return {}

    async def get_embedding_queue_status(self):
        return {}

    async def get_rerank_queue_status(self):
        return {}


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _stage_frontend(tmp_path, *, workspace=True):
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(exist_ok=True)
    (webui_dir / "index.html").write_text(
        f"<html><head>{PLACEHOLDER}</head><body>index</body></html>", "utf-8"
    )
    if workspace:
        (webui_dir / "workspace.html").write_text(
            f"<html><head>{PLACEHOLDER}</head><body>workspace</body></html>", "utf-8"
        )


def _build_app(tmp_path, monkeypatch, *cli_args):
    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", *cli_args]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(
        lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
    )
    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)
    monkeypatch.setattr(
        lightrag_server, "get_namespace_data", AsyncMock(return_value={"busy": False})
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )
    return lightrag_server.create_app(args)


class TestCustomizationEndpointNoBundle:
    def test_customized_false_with_brand(self, tmp_path, monkeypatch):
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))

        resp = client.get("/ui/customization?locale=zh-TW")
        assert resp.status_code == 200
        data = resp.json()
        assert data["customized"] is False
        assert data["brand"]["title"]
        assert "welcome" not in data
        assert resp.headers["cache-control"] == "no-store"
        assert "etag" not in resp.headers

    def test_asset_endpoint_rejects_everything(self, tmp_path, monkeypatch):
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert (
            client.get("/ui/customization/assets/abc123/brand-logo").status_code == 404
        )


class TestCustomizationEndpointWithBundle:
    def _client(self, tmp_path, monkeypatch, *cli_args, workspace=True):
        bundle = make_bundle(tmp_path)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace)
        return TestClient(_build_app(tmp_path, monkeypatch, *cli_args))

    def test_full_representation(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        resp = client.get("/ui/customization?locale=zh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["customized"] is True
        assert data["locale"] == "zh"
        assert data["fallback_used"] is False
        assert data["direction"] == "ltr"
        assert data["welcome"] == {"format": "markdown", "content": "# 欢迎"}
        assert data["query_empty"]["content"] == "请提问"
        assert data["brand"]["logo_alt"] == "ACME 中文"
        assert data["brand"]["logo_url"].startswith("/ui/customization/assets/")
        assert resp.headers["cache-control"] == "no-store"
        assert "etag" not in resp.headers

    def test_case_normalization_and_underscore_rejection(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        ok = client.get("/ui/customization?locale=ZH")
        assert ok.json()["locale"] == "zh"
        bad = client.get("/ui/customization?locale=zh_TW")
        assert bad.status_code == 400

    def test_fallback_chain_then_default(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        ko = client.get("/ui/customization?locale=ko").json()
        assert (ko["locale"], ko["fallback_used"]) == ("en", True)
        # Unknown locale with no fallback entry → the bundle's own default,
        # never the frontend defaults.
        fr = client.get("/ui/customization?locale=fr").json()
        assert (fr["locale"], fr["fallback_used"]) == ("en", True)
        assert fr["customized"] is True

    def test_rtl_direction_from_registry(self, tmp_path, monkeypatch):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["ar"] = {
            "welcome": "locales/en/welcome.md",
            "query_empty": "locales/en/query_empty.md",
            "logo_alt": "ACME AR",
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization?locale=ar").json()["direction"] == "rtl"

    def test_asset_served_immutable_with_security_headers(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        logo_url = client.get("/ui/customization?locale=en").json()["brand"]["logo_url"]
        resp = client.get(logo_url)
        assert resp.status_code == 200
        assert resp.content == SVG_BYTES
        assert resp.headers["content-type"].startswith("image/svg+xml")
        assert "immutable" in resp.headers["cache-control"]
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" in resp.headers
        # Unknown hash and mismatched id are both rejected.
        parts = logo_url.rsplit("/", 2)
        assert client.get(f"{parts[0]}/{'0' * 64}/{parts[2]}").status_code == 404
        assert client.get(f"{parts[0]}/{parts[1]}/other-id").status_code == 404

    def test_logo_url_keeps_api_prefix(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch, "--api-prefix", "/site01")
        data = client.get("/site01/ui/customization?locale=en").json()
        assert data["brand"]["logo_url"].startswith("/site01/ui/customization/assets/")
        assert client.get(data["brand"]["logo_url"]).status_code == 200

    def test_health_carries_no_customization_fields(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        health = client.get("/health")
        assert "bundle_revision" not in health.text
        assert "customization" not in health.text

    def test_route_audit_includes_mounts_and_customization(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        app = client.app
        paths = set()
        for route in app.routes:
            paths.add(getattr(route, "path", None))
        assert "/webui" in paths
        assert "/workspace" in paths
        assert "/ui/customization" in paths
        assert "/ui/customization/assets/{asset_hash}/{asset_id}" in paths


class TestOrthogonality:
    """workspace.html presence × UI_TEMPLATES_DIR unset/valid/invalid:
    customization behavior depends ONLY on the latter dimension."""

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_invalid_bundle_fails_startup_regardless_of_products(
        self, tmp_path, monkeypatch, workspace_present
    ):
        bundle = make_bundle(tmp_path)
        (bundle / "manifest.json").write_text("{broken", "utf-8")
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace_present)
        with pytest.raises(UICustomizationError):
            _build_app(tmp_path, monkeypatch)

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_valid_bundle_loads_regardless_of_products(
        self, tmp_path, monkeypatch, workspace_present
    ):
        bundle = make_bundle(tmp_path)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace_present)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization?locale=en").json()["customized"] is True

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_unset_is_normal_in_both_product_states(
        self, tmp_path, monkeypatch, workspace_present
    ):
        _stage_frontend(tmp_path, workspace=workspace_present)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization").json()["customized"] is False
