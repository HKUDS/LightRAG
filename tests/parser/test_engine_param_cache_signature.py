"""Cache-correctness proof for per-file engine params (Phase 2).

A per-file override MUST participate in the raw-bundle cache signature (so an
overridden document does not hit a bundle parsed with different params) AND must
reach the live request.  These tests assert both for MinerU and Docling.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# MinerU
# --------------------------------------------------------------------------- #


def test_mineru_signature_changes_with_page_range(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    from lightrag.parser.external.mineru.cache import current_mineru_options_signature

    base = current_mineru_options_signature()
    ov1 = current_mineru_options_signature({"page_range": "1-3"})
    ov2 = current_mineru_options_signature({"page_range": "1-3,5"})
    assert base != ov1 != ov2 and base != ov2
    # No spurious invalidation: no override == bare env signature.
    assert current_mineru_options_signature(None) == base


def test_mineru_signature_changes_with_language_and_parse_method(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "local")
    from lightrag.parser.external.mineru.cache import current_mineru_options_signature

    base = current_mineru_options_signature()
    assert current_mineru_options_signature({"language": "en"}) != base
    assert current_mineru_options_signature({"local_parse_method": "ocr"}) != base


def test_mineru_options_reflect_override(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    from lightrag.parser.external.mineru.cache import MinerUParserOptions

    opts = MinerUParserOptions.from_env(
        overrides={"language": "en", "page_range": "1-3,5"}
    )
    assert opts.language == "en" and opts.page_ranges == "1-3,5"


def test_mineru_local_bounds_from_page_range_override():
    from lightrag.parser.external.mineru.cache import MinerUParserOptions

    opts = MinerUParserOptions.from_env(
        api_mode="local", overrides={"page_range": "2-4"}
    )
    # local_page_bounds is 0-based.
    assert opts.local_start_page_id == 1 and opts.local_end_page_id == 3


def test_mineru_request_payload_reflects_override(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_OFFICIAL_ENDPOINT", "https://mineru.net")
    monkeypatch.setenv("MINERU_API_TOKEN", "test-token")
    from lightrag.parser.external.mineru.client import MinerURawClient

    client = MinerURawClient(overrides={"language": "en", "page_range": "1-3,5"})
    payload = client._official_payload("doc.pdf")
    assert payload["language"] == "en"
    assert payload["files"][0]["page_ranges"] == "1-3,5"


def test_mineru_local_form_reflects_parse_method(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    from lightrag.parser.external.mineru.client import MinerURawClient

    client = MinerURawClient(overrides={"local_parse_method": "ocr"})
    assert client._local_form_data()["parse_method"] == "ocr"


# --------------------------------------------------------------------------- #
# Docling
# --------------------------------------------------------------------------- #


def test_docling_snapshot_and_signature_reflect_force_ocr():
    from lightrag.parser.external.docling.cache import (
        compute_options_signature,
        snapshot_tunable_env,
    )
    from lightrag.parser.external.docling.client import FIXED_CONSTANTS

    assert snapshot_tunable_env({"force_ocr": False})["DOCLING_FORCE_OCR"] == "false"
    base = compute_options_signature(
        tunable_env=snapshot_tunable_env(), fixed_constants=FIXED_CONSTANTS
    )
    ov = compute_options_signature(
        tunable_env=snapshot_tunable_env({"force_ocr": False}),
        fixed_constants=FIXED_CONSTANTS,
    )
    assert base != ov


def test_docling_client_reflects_force_ocr_override(monkeypatch):
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://localhost:5001")
    from lightrag.parser.external.docling.client import DoclingRawClient

    client = DoclingRawClient(overrides={"force_ocr": False})
    assert client.force_ocr is False
