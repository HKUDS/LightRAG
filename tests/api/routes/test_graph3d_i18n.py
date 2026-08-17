"""Test the graph3d i18n.json serving route."""
import sys
sys.argv = [sys.argv[0]]  # strip CLI args before importing router

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.offline


class _StubRag:
    """Minimal stub: the i18n route doesn't use rag at all."""
    pass


from lightrag.api.routers import graph3d_routes  # noqa: E402

app = FastAPI()
app.include_router(graph3d_routes.create_graph3d_routes(_StubRag()))
client = TestClient(app)


def test_i18n_route_returns_json():
    """GET /graph3d/i18n.json returns 200 with application/json."""
    resp = client.get("/graph3d/i18n.json")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    data = resp.json()
    assert "en" in data
    assert "zh" in data


def test_i18n_en_zh_key_parity():
    """English and Chinese key sets must be identical."""
    resp = client.get("/graph3d/i18n.json")
    data = resp.json()
    assert set(data["en"].keys()) == set(data["zh"].keys())


def test_i18n_zh_title_translated():
    """Chinese title must be GK大脑类神经元3D视图 (feature 4)."""
    resp = client.get("/graph3d/i18n.json")
    data = resp.json()
    assert data["zh"]["app.title"] == "GK大脑类神经元3D视图"


def test_i18n_settings_keys_present():
    """Settings panel keys must exist for the new sidebar (feature 1)."""
    resp = client.get("/graph3d/i18n.json")
    data = resp.json()
    for key in (
        "settings.title",
        "settings.language",
        "settings.theme",
        "settings.brightness",
        "settings.panels",
        "settings.view",
        "btn.settings",
        "lang.zh",
        "lang.en",
        "theme.dark",
        "theme.light",
    ):
        assert key in data["en"], f"Missing key: {key}"
        assert key in data["zh"], f"Missing zh key: {key}"
