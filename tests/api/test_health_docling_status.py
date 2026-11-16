"""The ``docling`` block of the /health snapshot (``_build_docling_status``).

Under the VLM pipeline the standard OCR options are not in effect and must not
be reported as if they were; a misconfiguration the client would refuse to
parse with (unknown pipeline, VLM without a preset) is surfaced as
``pipeline_error`` instead of a clean-looking snapshot.
"""

from __future__ import annotations

import importlib
import sys

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_server = importlib.import_module("lightrag.api.lightrag_server")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_OCR_KEYS = {"do_ocr", "force_ocr", "ocr_engine", "ocr_lang", "do_formula_enrichment"}


@pytest.fixture(autouse=True)
def _docling_env(monkeypatch):
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
    for name in ("DOCLING_PIPELINE", "DOCLING_VLM_PRESET"):
        monkeypatch.delenv(name, raising=False)


def test_no_endpoint_reports_empty_block(monkeypatch):
    monkeypatch.delenv("DOCLING_ENDPOINT")
    assert _server._build_docling_status() == {"endpoint": "", "options": {}}


def test_standard_pipeline_reports_ocr_options():
    options = _server._build_docling_status()["options"]
    assert options["pipeline"] == "standard"
    assert _OCR_KEYS <= set(options)
    assert "vlm_pipeline_preset" not in options
    assert "pipeline_error" not in options


def test_vlm_pipeline_reports_preset_and_hides_ocr_options(monkeypatch):
    monkeypatch.setenv("DOCLING_PIPELINE", "vlm")
    monkeypatch.setenv("DOCLING_VLM_PRESET", "minicpm_ocr")
    options = _server._build_docling_status()["options"]
    assert options == {"pipeline": "vlm", "vlm_pipeline_preset": "minicpm_ocr"}


def test_vlm_pipeline_without_preset_reports_error(monkeypatch):
    monkeypatch.setenv("DOCLING_PIPELINE", "vlm")
    options = _server._build_docling_status()["options"]
    assert options["pipeline"] == "vlm"
    assert "DOCLING_VLM_PRESET" in options["pipeline_error"]
    assert not (_OCR_KEYS & set(options))


def test_unknown_pipeline_reports_error_only(monkeypatch):
    monkeypatch.setenv("DOCLING_PIPELINE", "fast")
    status = _server._build_docling_status()
    assert status["endpoint"] == "http://docling.test"
    assert set(status["options"]) == {"pipeline_error"}
    assert "fast" in status["options"]["pipeline_error"]
