"""Sidecar pytest fixtures (auto-loaded; no pytest_plugins)."""

from __future__ import annotations

import logging

import pytest

from tests.sidecar.conftest_query_attachments import build_synthetic_sidecar


@pytest.fixture
def propagate_lightrag_logs():
    """The ``lightrag`` logger sets propagate=False; caplog needs it on."""
    lg = logging.getLogger("lightrag")
    old = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = old


@pytest.fixture
def synthetic_sidecar(tmp_path):
    return build_synthetic_sidecar(tmp_path)


@pytest.fixture
def synthetic_sidecar_uri(synthetic_sidecar: dict) -> str:
    return str(synthetic_sidecar["sidecar_uri"])
