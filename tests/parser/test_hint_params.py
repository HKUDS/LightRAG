"""Tests for parameterised parser hints (``[-R(chunk_ts=800,chunk_ol=80)]``).

Phase 1 wires only ``chunk_token_size``/``chunk_ts`` and
``chunk_overlap_token_size``/``chunk_ol`` through the existing per-document
``chunk_options`` channel.  ``parser_rules=""`` is passed explicitly so the
assertions are independent of any ambient ``LIGHTRAG_PARSER``.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from lightrag.parser.param_schema import (
    parse_chunk_params,
    split_top_level,
    take_paren_block,
)
from lightrag.parser.routing import (
    FilenameParserHintError,
    ParserRoutingConfigError,
    canonicalize_parser_hinted_basename,
    filename_parser_directives,
    resolve_file_parser_directives,
    resolve_parser_directives,
    validate_parser_routing_config,
)

# Importing document_routes runs an argparse over sys.argv at import time;
# neutralise pytest's argv during that first import (subsequent imports hit the
# module cache and skip argparse).  Mirrors tests/api/routes/test_document_routes_chunking.py.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv


# --------------------------------------------------------------------------- #
# param_schema: scanners
# --------------------------------------------------------------------------- #


def test_split_top_level_protects_commas_inside_parens():
    # A comma inside a parameter block must not split the surrounding rules.
    assert split_top_level(
        "pdf:legacy-R(chunk_ts=800,chunk_ol=80);*:legacy-R", ";,"
    ) == ["pdf:legacy-R(chunk_ts=800,chunk_ol=80)", "*:legacy-R"]
    # Legacy comma rule separator still works at depth 0.
    assert split_top_level("a:native-F,b:legacy-R", ";,") == [
        "a:native-F",
        "b:legacy-R",
    ]


def test_take_paren_block():
    assert take_paren_block("R(chunk_ts=800)", 1) == ("chunk_ts=800", 15)
    assert take_paren_block("R", 1) == (None, 1)  # no block
    assert take_paren_block("R(unterminated", 1) == (None, 1)  # unbalanced


# --------------------------------------------------------------------------- #
# param_schema: parse_chunk_params (alias / type / target / repeat)
# --------------------------------------------------------------------------- #


def test_parse_chunk_params_alias_normalised_to_canonical():
    parsed, errors = parse_chunk_params(
        "chunk_ts=800,chunk_ol=80", selector="R", label="x"
    )
    assert errors == []
    assert parsed == {"chunk_token_size": 800, "chunk_overlap_token_size": 80}


def test_parse_chunk_params_canonical_names_accepted():
    parsed, errors = parse_chunk_params(
        "chunk_token_size=1500", selector="V", label="x"
    )
    assert errors == [] and parsed == {"chunk_token_size": 1500}


def test_parse_chunk_params_overlap_rejected_for_vector():
    parsed, errors = parse_chunk_params("chunk_ol=80", selector="V", label="x")
    assert parsed == {}
    assert errors and "not supported for chunk strategy 'V'" in errors[0]


@pytest.mark.parametrize(
    "block,needle",
    [
        ("chunk_ts=abc", "must be an integer"),
        ("chunk_ts=0", "must be >= 1"),
        ("foo=1", "unknown parameter 'foo'"),
        ("chunk_ts=1,chunk_ts=2", "may not be repeated"),
        ("correct_tl", "flag parameters are not supported"),
        ("", "empty parameter"),
    ],
)
def test_parse_chunk_params_errors(block, needle):
    _parsed, errors = parse_chunk_params(block, selector="F", label="x")
    assert errors and any(needle in e for e in errors)


# --------------------------------------------------------------------------- #
# routing: filename hint params
# --------------------------------------------------------------------------- #


def test_filename_hint_selector_stays_pure_and_params_extracted():
    d = resolve_parser_directives(
        "notes.[-R(chunk_ts=800,chunk_ol=80)].md", parser_rules=""
    )
    assert d.engine == "legacy"
    assert d.process_options == "R"  # pure selector, no params
    assert d.chunk_params == {
        "R": {"chunk_token_size": 800, "chunk_overlap_token_size": 80}
    }
    assert d.engine_params == {}


def test_back_compat_two_tuple_and_directives_unaffected_by_params():
    # The legacy 2-tuple accessor must keep returning (engine, pure-selector).
    assert resolve_file_parser_directives(
        "notes.[-R(chunk_ts=800)].md", parser_rules=""
    ) == ("legacy", "R")
    assert filename_parser_directives("notes.[-R(chunk_ts=800)].md") == (None, "R")
    # A param-free hint is completely unchanged.
    assert filename_parser_directives("paper.[native-iet].docx") == ("native", "iet")


def test_canonicalize_strips_parameterised_hint():
    assert (
        canonicalize_parser_hinted_basename("notes.[-R(chunk_ts=800,chunk_ol=80)].md")
        == "notes.md"
    )


def test_engine_params_rejected_with_friendly_error():
    # Lenient classifier ignores it (engine stays resolved, no chunk params)...
    d = resolve_parser_directives("doc.[native-F(chunk_ts=900)].docx", parser_rules="")
    assert d.chunk_params == {"F": {"chunk_token_size": 900}}
    # ...but an engine-level block is a hard error on ingestion.
    with pytest.raises(FilenameParserHintError, match="engine parameters are not"):
        resolve_file_parser_directives(
            "doc.[native(do_ocr=true)-F].docx", parser_rules=""
        )


@pytest.mark.parametrize(
    "name",
    [
        "x.[-V(chunk_ol=80)].md",  # overlap invalid for V
        "x.[-R(chunk_ts=abc)].md",  # non-integer
        "x.[native-R(foo=1)].md",  # unknown parameter
    ],
)
def test_upload_validation_raises_on_bad_filename_params(name):
    with pytest.raises(FilenameParserHintError):
        resolve_file_parser_directives(name, parser_rules="")


# --------------------------------------------------------------------------- #
# routing: LIGHTRAG_PARSER rule params + overlay merge
# --------------------------------------------------------------------------- #


def test_rule_params_with_semicolon_separator_and_comma_in_parens():
    d = resolve_parser_directives(
        "a.md", parser_rules="md:legacy-R(chunk_ts=800,chunk_ol=80);*:legacy-R"
    )
    assert d.engine == "legacy"
    assert d.chunk_params == {
        "R": {"chunk_token_size": 800, "chunk_overlap_token_size": 80}
    }


def test_rule_params_with_comma_separator_and_comma_in_parens():
    # Rule splitting is parenthesis-aware: a comma inside a parameter block is
    # NOT a rule separator, so ',' still separates rules even when a rule carries
    # parameters (the docs recommend ';' but ',' must keep working).
    assert split_top_level(
        "md:legacy-R(chunk_ts=800,chunk_ol=80),*:legacy-R", ";,"
    ) == ["md:legacy-R(chunk_ts=800,chunk_ol=80)", "*:legacy-R"]
    d = resolve_parser_directives(
        "a.md", parser_rules="md:legacy-R(chunk_ts=800,chunk_ol=80),*:legacy-R"
    )
    assert d.engine == "legacy"
    assert d.chunk_params == {
        "R": {"chunk_token_size": 800, "chunk_overlap_token_size": 80}
    }


def test_overlay_rule_then_filename_hint_wins_per_key():
    # Design worked example: rule supplies chunk_ol on P, the filename hint
    # supplies chunk_ts on P; the surviving selector is the filename's "P".
    d = resolve_parser_directives(
        "paper.[-P(chunk_ts=3000)].pdf", parser_rules="pdf:legacy-iteP(chunk_ol=100)"
    )
    assert d.process_options == "P"  # filename options wholesale-override rule
    assert d.chunk_params["P"] == {
        "chunk_overlap_token_size": 100,  # inherited from the rule
        "chunk_token_size": 3000,  # from the filename hint
    }


def test_filename_hint_key_overrides_rule_key():
    d = resolve_parser_directives(
        "paper.[-R(chunk_ts=500)].md", parser_rules="md:legacy-R(chunk_ts=900,chunk_ol=50)"
    )
    assert d.chunk_params["R"] == {
        "chunk_token_size": 500,  # filename wins over the rule's 900
        "chunk_overlap_token_size": 50,  # inherited from the rule
    }


# --------------------------------------------------------------------------- #
# routing: startup validation
# --------------------------------------------------------------------------- #


def test_validate_parser_routing_config_accepts_good_params():
    validate_parser_routing_config(
        "pdf:legacy-R(chunk_ts=800,chunk_ol=80);*:legacy-R"
    )


@pytest.mark.parametrize(
    "rules,needle",
    [
        ("pdf:legacy-V(chunk_ol=80)", "not supported for chunk strategy 'V'"),
        ("pdf:legacy-R(chunk_ts=abc)", "must be an integer"),
        ("pdf:legacy-R(foo=1)", "unknown parameter 'foo'"),
    ],
)
def test_validate_parser_routing_config_rejects_bad_params(rules, needle):
    with pytest.raises(ParserRoutingConfigError, match=needle):
        validate_parser_routing_config(rules)


def test_legacy_comma_rules_without_params_still_valid():
    # Pure back-compat: legacy comma-separated, param-free rules unchanged.
    validate_parser_routing_config("docx:native-iteP,pdf:legacy-R,*:legacy-R")


# --------------------------------------------------------------------------- #
# upload path: filename-hint params reach the enqueued chunk_options
# --------------------------------------------------------------------------- #


def _recording_rag():
    import asyncio  # noqa: F401 - imported for clarity at call sites
    from types import SimpleNamespace

    captured: dict = {}

    async def fake_enqueue(content, **kwargs):
        captured.update(kwargs)
        return "track-xyz"  # non-None -> success path

    async def fake_error(files, track_id):
        captured["error_files"] = files

    rag = SimpleNamespace(
        addon_params={},
        apipeline_enqueue_documents=fake_enqueue,
        apipeline_enqueue_error_documents=fake_error,
    )
    return rag, captured


def test_upload_path_applies_hint_chunk_params(tmp_path):
    import asyncio

    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    rag, captured = _recording_rag()
    f = tmp_path / "notes.[-R(chunk_ts=800,chunk_ol=80)].md"
    f.write_text("hello world", encoding="utf-8")

    ok, _track = asyncio.run(pipeline_enqueue_file(rag, f))

    assert ok is True
    assert "error_files" not in captured
    assert captured["parse_engine"] == "legacy"
    assert captured["process_options"] == "R"  # pure selector forwarded
    sub = captured["chunk_options"]["recursive_character"]
    assert sub["chunk_token_size"] == 800
    assert sub["chunk_overlap_token_size"] == 80


def test_upload_path_without_params_omits_chunk_options(tmp_path):
    import asyncio

    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    rag, captured = _recording_rag()
    f = tmp_path / "plain.[native-R].md"
    f.write_text("hello world", encoding="utf-8")

    ok, _track = asyncio.run(pipeline_enqueue_file(rag, f))

    assert ok is True
    # Legacy behaviour: no per-file chunk_options when the hint has no params.
    assert "chunk_options" not in captured


def test_upload_path_rejects_bad_hint_params(tmp_path):
    import asyncio

    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    rag, captured = _recording_rag()
    f = tmp_path / "bad.[-V(chunk_ol=80)].md"
    f.write_text("hello world", encoding="utf-8")

    ok, _track = asyncio.run(pipeline_enqueue_file(rag, f))

    assert ok is False
    assert captured["error_files"]  # an error document was enqueued
    assert "chunk_options" not in captured


def test_upload_path_rejects_effective_overlap_violation(tmp_path):
    import asyncio

    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    rag, captured = _recording_rag()
    # overlap (100) >= size (50) -> effective-value validation fails
    f = tmp_path / "tiny.[-R(chunk_ts=50,chunk_ol=100)].md"
    f.write_text("hello world", encoding="utf-8")

    ok, _track = asyncio.run(pipeline_enqueue_file(rag, f))

    assert ok is False
    assert captured["error_files"]
    assert "Chunk parameter error" in captured["error_files"][0]["error_description"]
