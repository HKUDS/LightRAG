"""Tests for per-file ENGINE parameters (Phase 2).

Engine params attach to the engine token of a hint / LIGHTRAG_PARSER rule
(``mineru(page_range=1-3,language=en)`` / ``docling(force_ocr=true)``) and ride
the existing ``parse_engine`` field encoded in hint syntax.

The suite's conftest strips ``MINERU_API_MODE`` (tests default to local, which
allows only a single page_range segment), so multi-segment cases explicitly set
official mode via monkeypatch.  ``parser_rules=""`` keeps assertions independent
of any ambient ``LIGHTRAG_PARSER``.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from lightrag.parser.param_schema import (
    normalize_engine_params,
    parse_engine_params,
    render_engine_params,
)
from lightrag.parser.routing import (
    FilenameParserHintError,
    ParserRoutingConfigError,
    decode_parse_engine,
    encode_parse_engine,
    normalize_parser_engine,
    resolve_file_parser_directives,
    resolve_parser_directives,
    validate_parser_routing_config,
)

# Importing document_routes runs argparse over sys.argv at import time;
# neutralise pytest's argv during that first import.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv


# --------------------------------------------------------------------------- #
# parse_engine_params
# --------------------------------------------------------------------------- #


def test_parse_engine_params_basic_and_alias():
    parsed, errors = parse_engine_params(
        "page_range=1-3,language=en", engine="mineru", label="x"
    )
    assert errors == []
    assert parsed == {"page_range": "1-3", "language": "en"}
    # alias pr -> page_range
    parsed, errors = parse_engine_params("pr=2-4", engine="mineru", label="x")
    assert errors == [] and parsed == {"page_range": "2-4"}
    # alias local_pm -> local_parse_method
    parsed, errors = parse_engine_params("local_pm=ocr", engine="mineru", label="x")
    assert errors == [] and parsed == {"local_parse_method": "ocr"}


def test_parse_engine_params_paddleocr_vl_top_level_request_params(monkeypatch):
    # PaddleOCR-VL has its own pageRanges request field; it must not inherit
    # MinerU local-mode's single-segment restriction.
    monkeypatch.delenv("MINERU_API_MODE", raising=False)
    parsed, errors = parse_engine_params(
        "page_range=1-3,pr=5,useOcrForImageBlock=true,"
        "useSealRecognition=false,useDocUnwarping=yes",
        engine="paddleocr_vl",
        label="x",
    )

    assert errors == []
    assert parsed == {
        "page_range": "1-3,5",
        "use_ocr_for_image_block": True,
        "use_seal_recognition": False,
        "use_doc_unwarping": True,
    }


def test_parse_engine_params_paddleocr_vl_accepts_relative_end_page_range():
    parsed, errors = parse_engine_params(
        "page_range=2--2", engine="paddleocr_vl", label="x"
    )

    assert errors == []
    assert parsed == {"page_range": "2--2"}


def test_parse_engine_params_paddleocr_vl_rejects_unregistered_params():
    _parsed, errors = parse_engine_params(
        "batch_id=batch-1,model=PaddleOCR-VL,use_layout_detection=false,"
        "merge_tables=false,visualize=true",
        engine="paddleocr_vl",
        label="x",
    )

    assert errors
    assert any("unknown parameter 'batch_id'" in e for e in errors)
    assert any("unknown parameter 'model'" in e for e in errors)
    assert any("unknown parameter 'use_layout_detection'" in e for e in errors)
    assert any("unknown parameter 'merge_tables'" in e for e in errors)
    assert any("unknown parameter 'visualize'" in e for e in errors)


def test_parse_engine_params_bool_coercion():
    parsed, errors = parse_engine_params("force_ocr=false", engine="docling", label="x")
    assert errors == [] and parsed == {"force_ocr": False}
    parsed, errors = parse_engine_params("ocr=yes", engine="docling", label="x")
    assert errors == [] and parsed == {"force_ocr": True}


def test_parse_engine_params_multi_segment_official(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    parsed, errors = parse_engine_params(
        "page_range=1-3,page_range=5,page_range=7-9", engine="mineru", label="x"
    )
    assert errors == [] and parsed == {"page_range": "1-3,5,7-9"}


def test_parse_engine_params_multi_segment_rejected_local(monkeypatch):
    monkeypatch.delenv("MINERU_API_MODE", raising=False)  # default local
    parsed, errors = parse_engine_params(
        "page_range=1-3,page_range=5", engine="mineru", label="x"
    )
    assert parsed == {} or "page_range" not in parsed
    assert errors and "local supports only a single page" in errors[0]


def test_parse_engine_params_local_parse_method_accepted_local(monkeypatch):
    monkeypatch.delenv("MINERU_API_MODE", raising=False)  # default local
    parsed, errors = parse_engine_params(
        "local_parse_method=ocr", engine="mineru", label="x"
    )
    assert errors == [] and parsed == {"local_parse_method": "ocr"}


def test_parse_engine_params_local_parse_method_rejected_official(monkeypatch):
    # local_parse_method is local-only: official neither sends it nor folds it
    # into the cache key, so accepting it would persist a silent no-op directive.
    monkeypatch.setenv("MINERU_API_MODE", "official")
    parsed, errors = parse_engine_params(
        "local_parse_method=ocr", engine="mineru", label="x"
    )
    assert "local_parse_method" not in parsed
    assert errors and "MINERU_API_MODE=local" in errors[0]


@pytest.mark.parametrize(
    "block,engine,needle",
    [
        ("page_range=1", "legacy", "does not accept parameters"),
        ("page_range=1", "native", "does not accept parameters"),
        ("foo=1", "mineru", "unknown parameter 'foo'"),
        ("local_parse_method=bad", "mineru", "must be one of"),
        ("force_ocr=maybe", "docling", "must be a boolean"),
        ("language=", "mineru", "must be non-empty"),
        ("page_range=1-3,5", "mineru", "page lists must repeat the key"),
    ],
)
def test_parse_engine_params_errors(block, engine, needle):
    _parsed, errors = parse_engine_params(block, engine=engine, label="x")
    assert errors and any(needle in e for e in errors)


# --------------------------------------------------------------------------- #
# normalize_engine_params (resolved-dict path) + encode/decode round-trip
# --------------------------------------------------------------------------- #


def test_normalize_engine_params_coerces_bool_string():
    # A direct caller may pass force_ocr as the string "false" — must coerce.
    norm, errors = normalize_engine_params("docling", {"force_ocr": "false"})
    assert errors == [] and norm == {"force_ocr": False}


def test_normalize_engine_params_accepts_page_range_list_or_string(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    by_list, e1 = normalize_engine_params("mineru", {"page_range": ["1-3", "5"]})
    by_str, e2 = normalize_engine_params("mineru", {"page_range": "1-3,5"})
    assert e1 == [] and e2 == [] and by_list == by_str == {"page_range": "1-3,5"}


def test_normalize_engine_params_accepts_paddleocr_vl_request_params(monkeypatch):
    monkeypatch.delenv("MINERU_API_MODE", raising=False)
    norm, errors = normalize_engine_params(
        "paddleocr_vl",
        {
            "page_range": ["2", "4-6"],
            "useOcrForImageBlock": "true",
            "useSealRecognition": "false",
            "useDocUnwarping": "yes",
        },
    )

    assert errors == []
    assert norm == {
        "page_range": "2,4-6",
        "use_ocr_for_image_block": True,
        "use_seal_recognition": False,
        "use_doc_unwarping": True,
    }


def test_normalize_engine_params_rejects_unregistered_paddleocr_vl_params():
    _norm, errors = normalize_engine_params(
        "paddleocr_vl",
        {
            "batch_id": "batch-1",
            "model": "PaddleOCR-VL",
            "use_layout_detection": "false",
            "merge_tables": "no",
            "visualize": "yes",
        },
    )

    assert errors
    assert any("unknown parameter 'batch_id'" in e for e in errors)
    assert any("unknown parameter 'model'" in e for e in errors)
    assert any("unknown parameter 'use_layout_detection'" in e for e in errors)
    assert any("unknown parameter 'merge_tables'" in e for e in errors)
    assert any("unknown parameter 'visualize'" in e for e in errors)


def test_normalize_engine_params_rejects_unregistered_engine():
    _norm, errors = normalize_engine_params("legacy", {"page_range": "1"})
    assert errors and "does not accept parameters" in errors[0]


def test_encode_decode_round_trip(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    enc = encode_parse_engine("mineru", {"page_range": "1-3,5", "language": "en"})
    # list param emitted as repeated keys; deterministic (sorted)
    assert enc == "mineru(language=en,page_range=1-3,page_range=5)"
    engine, params, errors = decode_parse_engine(enc)
    assert engine == "mineru" and errors == []
    assert params == {"page_range": "1-3,5", "language": "en"}


def test_paddleocr_vl_encode_decode_round_trip():
    enc = encode_parse_engine(
        "paddleocr_vl",
        {"page_range": "1-3,5", "use_doc_unwarping": True},
    )

    assert enc == "paddleocr_vl(page_range=1-3,page_range=5,use_doc_unwarping=true)"
    engine, params, errors = decode_parse_engine(enc)
    assert engine == "paddleocr_vl" and errors == []
    assert params == {"page_range": "1-3,5", "use_doc_unwarping": True}


def test_encode_bare_when_no_params():
    assert encode_parse_engine("mineru", {}) == "mineru"
    assert encode_parse_engine("legacy", None) == "legacy"


def test_render_engine_params_bool_and_list(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    inner, errs = render_engine_params("mineru", {"page_range": "1-3,5"})
    assert errs == [] and inner == "page_range=1-3,page_range=5"
    inner, errs = render_engine_params("docling", {"force_ocr": False})
    assert errs == [] and inner == "force_ocr=false"


def test_normalize_parser_engine_strips_params():
    assert normalize_parser_engine("mineru(page_range=1-3)") == "mineru"
    assert normalize_parser_engine("MINERU(page_range=1-3,language=en)") == "mineru"
    assert normalize_parser_engine("mineru") == "mineru"


def test_decode_malformed_reports_errors():
    _e, _p, errors = decode_parse_engine("mineru(page_range=")
    assert errors and "unbalanced" in errors[0]
    _e, _p, errors = decode_parse_engine("mineru(badkey=1)")
    assert errors and "unknown parameter" in errors[0]


# --------------------------------------------------------------------------- #
# resolve_parser_directives — filename hint + rule + overlay + drop
# --------------------------------------------------------------------------- #


def test_resolve_filename_engine_params(monkeypatch):
    monkeypatch.delenv("MINERU_API_MODE", raising=False)  # local single segment
    d = resolve_parser_directives(
        "paper.[mineru(page_range=1-3,language=en)].pdf",
        parser_rules="",
        require_external_endpoint=False,
    )
    assert d.engine == "mineru"
    assert d.engine_params == {"page_range": "1-3", "language": "en"}
    # selector stays a pure (here empty) string
    assert "(" not in d.process_options


def test_resolve_filename_paddleocr_vl_engine_params():
    d = resolve_parser_directives(
        "paper.[paddleocr_vl(page_range=1-3,useOcrForImageBlock=true)].pdf",
        parser_rules="",
        require_external_endpoint=False,
    )

    assert d.engine == "paddleocr_vl"
    assert d.engine_params == {
        "page_range": "1-3",
        "use_ocr_for_image_block": True,
    }


def test_resolve_rule_engine_params():
    d = resolve_parser_directives(
        "a.pdf",
        parser_rules="pdf:mineru(language=en)",
        require_external_endpoint=False,
    )
    assert d.engine == "mineru" and d.engine_params == {"language": "en"}


def test_resolve_drop_rule_params_when_hint_engine_wins():
    # Rule names mineru(language); filename hint names a different usable engine
    # (docling) which wins -> mineru's engine params are dropped.
    d = resolve_parser_directives(
        "paper.[docling(force_ocr=true)].pdf",
        parser_rules="pdf:mineru(language=en)",
        require_external_endpoint=False,
    )
    assert d.engine == "docling"
    assert d.engine_params == {"force_ocr": True}


@pytest.mark.parametrize(
    "name",
    [
        "x.[mineru(language=)].pdf",  # empty value
        "x.[mineru(local_parse_method=bad)].pdf",  # bad enum
        "x.[docling(force_ocr=maybe)].pdf",  # bad bool
        "x.[legacy(page_range=1)].txt",  # engine declares no params
        "x.[mineru(page_range=1-3,5)].pdf",  # env-style comma list
    ],
)
def test_upload_validation_raises_on_bad_engine_params(name):
    with pytest.raises(FilenameParserHintError):
        resolve_file_parser_directives(
            name, parser_rules="", require_external_endpoint=False
        )


# --------------------------------------------------------------------------- #
# validate_parser_routing_config (startup)
# --------------------------------------------------------------------------- #


def test_validate_rule_engine_params_ok(monkeypatch):
    # mineru requires its endpoint configured before its rule validates.
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    validate_parser_routing_config("pdf:mineru(language=en);*:legacy-R")


@pytest.mark.parametrize(
    "rules,needle",
    [
        ("pdf:legacy(page_range=1)", "does not accept parameters"),
        ("pdf:mineru(foo=1)", "unknown parameter 'foo'"),
        ("pdf:mineru(local_parse_method=bad)", "must be one of"),
        ("pdf:docling(force_ocr=maybe)", "must be a boolean"),
    ],
)
def test_validate_rule_engine_params_rejected(rules, needle):
    with pytest.raises(ParserRoutingConfigError, match=needle):
        validate_parser_routing_config(rules)


# --------------------------------------------------------------------------- #
# Direct-API guard via apipeline_enqueue_documents -> _parse_engine_at
# --------------------------------------------------------------------------- #


def _decode_via_parse_engine_at(raw: str):
    """Exercise the same decode+validate+re-encode that _parse_engine_at runs."""
    engine, params, errs = decode_parse_engine(raw)
    if errs:
        raise ValueError("; ".join(errs))
    return encode_parse_engine(engine, params) if params else engine


def test_direct_parse_engine_field_canonicalises(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    assert (
        _decode_via_parse_engine_at("mineru(page_range=1-3,page_range=5)")
        == "mineru(page_range=1-3,page_range=5)"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "legacy(page_range=1)",  # legacy declares no params
        "mineru(badkey=1)",  # unknown
        "mineru(page_range=",  # malformed
    ],
)
def test_direct_parse_engine_field_rejected(raw):
    with pytest.raises(ValueError):
        _decode_via_parse_engine_at(raw)
