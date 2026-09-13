"""Tests for authenticated runtime observability routes."""

import importlib
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


_original_argv = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    utils_api = importlib.import_module("lightrag.api.utils_api")
    observability_routes = importlib.import_module(
        "lightrag.api.routers.observability_routes"
    )
finally:
    sys.argv = _original_argv


def make_client(api_key="test-api-key"):
    app = FastAPI()
    app.include_router(observability_routes.create_observability_routes(api_key))
    return TestClient(app)


def test_langfuse_status_requires_credentials_even_when_whitelisted(monkeypatch):
    monkeypatch.setattr(utils_api, "path_is_whitelisted", lambda _scope: True)
    monkeypatch.setattr(
        observability_routes,
        "get_langfuse_tracing_status",
        lambda: {
            "installed": True,
            "configured": True,
            "enabled": True,
            "active": True,
        },
    )

    response = make_client().get("/observability/langfuse/tracing")

    assert response.status_code in {401, 403}


def test_langfuse_status_accepts_api_key(monkeypatch):
    monkeypatch.setattr(
        observability_routes,
        "get_langfuse_tracing_status",
        lambda: {
            "installed": True,
            "configured": True,
            "enabled": False,
            "active": False,
        },
    )

    response = make_client().get(
        "/observability/langfuse/tracing",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "installed": True,
        "configured": True,
        "enabled": False,
        "active": False,
    }


def test_langfuse_toggle_updates_runtime_state(monkeypatch):
    state = {"enabled": False}

    def get_status():
        return {
            "installed": True,
            "configured": True,
            "enabled": state["enabled"],
            "active": state["enabled"],
        }

    def set_enabled(enabled):
        state["enabled"] = enabled

    monkeypatch.setattr(
        observability_routes,
        "get_langfuse_tracing_status",
        get_status,
    )
    monkeypatch.setattr(
        observability_routes,
        "set_langfuse_tracing_enabled",
        set_enabled,
    )

    response = make_client().put(
        "/observability/langfuse/tracing",
        headers={"X-API-Key": "test-api-key"},
        json={"enabled": True},
    )

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert response.json()["active"] is True


def test_langfuse_toggle_rejects_unconfigured_installation(monkeypatch):
    monkeypatch.setattr(
        observability_routes,
        "get_langfuse_tracing_status",
        lambda: {
            "installed": True,
            "configured": False,
            "enabled": False,
            "active": False,
        },
    )

    response = make_client().put(
        "/observability/langfuse/tracing",
        headers={"X-API-Key": "test-api-key"},
        json={"enabled": True},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Langfuse credentials are not configured"}
