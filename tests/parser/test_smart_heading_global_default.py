"""Tests for the ``DOCX_SMART_HEADING`` global default and its startup check.

The env switch seeds ``smart_heading=true`` at the lowest precedence of the
``resolve_parser_directives`` engine-param merge (native + .docx only), so it
materializes into the persisted ``parse_engine`` at upload time; an explicit
``native(smart_heading=false)`` rule/hint overrides it. The startup validator
fails fast when the switch (or a ``LIGHTRAG_PARSER`` rule) enables
smart_heading while the pinned spaCy models are missing.

``parser_rules=""`` / explicit rules are passed so assertions are independent
of any ambient ``LIGHTRAG_PARSER``; ``DOCX_SMART_HEADING`` is always set or
deleted explicitly per test.
"""

from __future__ import annotations

import pytest

from lightrag.parser.docx.smart_heading import nlp
from lightrag.parser.routing import (
    ParserRoutingConfigError,
    _validate_smart_heading_max_chars,
    encode_parse_engine,
    resolve_parser_directives,
    seed_smart_heading_param,
    smart_heading_default_enabled,
    validate_smart_heading_dependencies,
)


@pytest.fixture(autouse=True)
def _clean_switch(monkeypatch):
    monkeypatch.delenv("DOCX_SMART_HEADING", raising=False)
    monkeypatch.delenv("DOCX_SMART_HEADING_MAX_CHARS", raising=False)


# --------------------------------------------------------------------------- #
# smart_heading_default_enabled: env parsing
# --------------------------------------------------------------------------- #


def test_switch_defaults_off() -> None:
    assert smart_heading_default_enabled() is False


@pytest.mark.parametrize("raw", ["true", "1", "on", "Yes"])
def test_switch_truthy_values(monkeypatch, raw) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", raw)
    assert smart_heading_default_enabled() is True


def test_switch_falsy_value(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "false")
    assert smart_heading_default_enabled() is False


def test_switch_invalid_value_raises(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "maybe")
    with pytest.raises(ParserRoutingConfigError, match="DOCX_SMART_HEADING"):
        smart_heading_default_enabled()


# --------------------------------------------------------------------------- #
# resolve_parser_directives: seed injection and overrides
# --------------------------------------------------------------------------- #


def test_seed_injected_for_native_docx(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    d = resolve_parser_directives("doc.[native].docx", parser_rules="")
    assert d.engine == "native"
    assert d.engine_params == {"smart_heading": True}
    assert (
        encode_parse_engine(d.engine, d.engine_params) == "native(smart_heading=true)"
    )


def test_switch_off_keeps_bare_encoding(monkeypatch) -> None:
    d = resolve_parser_directives("doc.[native].docx", parser_rules="")
    assert d.engine_params == {}
    assert encode_parse_engine(d.engine, d.engine_params) == "native"


def test_seed_not_injected_for_non_docx_native(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    # markdown routed to native must not carry the seed
    d_md = resolve_parser_directives("notes.[native].md", parser_rules="")
    assert d_md.engine == "native"
    assert "smart_heading" not in d_md.engine_params
    # .textpack defaults to native without any hint — still no seed
    d_tp = resolve_parser_directives("bundle.textpack", parser_rules="")
    assert d_tp.engine == "native"
    assert "smart_heading" not in d_tp.engine_params


def test_seed_not_injected_when_docx_resolves_to_legacy(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    d = resolve_parser_directives("doc.docx", parser_rules="")
    assert d.engine == "legacy"
    assert d.engine_params == {}


def test_hint_false_overrides_seed(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    d = resolve_parser_directives(
        "doc.[native(smart_heading=false)].docx", parser_rules=""
    )
    assert d.engine_params == {"smart_heading": False}
    assert (
        encode_parse_engine(d.engine, d.engine_params) == "native(smart_heading=false)"
    )


def test_rule_false_overrides_seed(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    d = resolve_parser_directives(
        "doc.docx", parser_rules="docx:native(smart_heading=false)"
    )
    assert d.engine == "native"
    assert d.engine_params == {"smart_heading": False}


# --------------------------------------------------------------------------- #
# seed_smart_heading_param: the shared chokepoint helper
# --------------------------------------------------------------------------- #


def test_seed_helper_fills_native_docx(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    params: dict = {}
    seed_smart_heading_param("native", params, "a.docx")
    assert params == {"smart_heading": True}


def test_seed_helper_respects_explicit_false(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    params: dict = {"smart_heading": False}
    seed_smart_heading_param("native", params, "a.docx")
    assert params == {"smart_heading": False}


def test_seed_helper_skips_non_docx_and_non_native(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    params: dict = {}
    seed_smart_heading_param("native", params, "notes.md")
    seed_smart_heading_param("mineru", params, "a.docx")
    assert params == {}


def test_seed_helper_noop_when_switch_off() -> None:
    params: dict = {}
    seed_smart_heading_param("native", params, "a.docx")
    assert params == {}


# --------------------------------------------------------------------------- #
# validate_smart_heading_dependencies: startup fail-fast
# --------------------------------------------------------------------------- #


def test_startup_check_silent_when_not_enabled(monkeypatch) -> None:
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: ["zh_core_web_sm"])
    validate_smart_heading_dependencies(parser_rules="")


def test_startup_check_rule_false_does_not_trigger(monkeypatch) -> None:
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: ["zh_core_web_sm"])
    validate_smart_heading_dependencies(parser_rules="docx:native(smart_heading=false)")


def test_startup_check_env_on_missing_models_raises(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: ["zh_core_web_sm"])
    with pytest.raises(nlp.SmartHeadingNLPError, match="zh_core_web_sm"):
        validate_smart_heading_dependencies(parser_rules="")


def test_startup_check_rule_triggers(monkeypatch) -> None:
    monkeypatch.setattr(
        nlp, "missing_spacy_models", lambda: ["en_core_web_sm", "zh_core_web_sm"]
    )
    with pytest.raises(nlp.SmartHeadingNLPError, match="lightrag-download-cache"):
        validate_smart_heading_dependencies(
            parser_rules="docx:native(smart_heading=true)"
        )


def test_startup_check_non_docx_rule_does_not_trigger(monkeypatch) -> None:
    # smart_heading only takes effect on .docx (markdown warn-and-ignores it),
    # so a non-docx rule must not force the spaCy models at startup.
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: ["zh_core_web_sm"])
    validate_smart_heading_dependencies(parser_rules="md:native(smart_heading=true)")


def test_startup_check_wildcard_rule_triggers(monkeypatch) -> None:
    # A wildcard pattern can match docx and must keep the fail-fast behavior.
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: ["zh_core_web_sm"])
    with pytest.raises(nlp.SmartHeadingNLPError, match="zh_core_web_sm"):
        validate_smart_heading_dependencies(
            parser_rules="do*:native(smart_heading=true)"
        )


def test_startup_check_passes_with_models(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: [])
    validate_smart_heading_dependencies(parser_rules="")


def test_startup_check_surfaces_invalid_switch(monkeypatch) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING", "treu")
    with pytest.raises(ParserRoutingConfigError, match="DOCX_SMART_HEADING"):
        validate_smart_heading_dependencies(parser_rules="")


# --------------------------------------------------------------------------- #
# missing_spacy_models: lightweight probe
# --------------------------------------------------------------------------- #


def test_missing_spacy_models_without_spacy(monkeypatch) -> None:
    real_find_spec = nlp.importlib.util.find_spec
    monkeypatch.setattr(
        nlp.importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            None if name == "spacy" else real_find_spec(name, *a, **k)
        ),
    )
    assert nlp.missing_spacy_models() == ["en_core_web_sm", "zh_core_web_sm"]


def test_missing_spacy_models_reports_uninstalled(monkeypatch) -> None:
    spacy_util = pytest.importorskip("spacy.util")
    monkeypatch.setattr(spacy_util, "is_package", lambda name: name == "en_core_web_sm")
    assert nlp.missing_spacy_models() == ["zh_core_web_sm"]


# --------------------------------------------------------------------------- #
# DOCX_SMART_HEADING_MAX_CHARS: startup validation (loud on invalid, warn tiny)
# --------------------------------------------------------------------------- #


def test_max_chars_unset_and_sane_values_ok(monkeypatch) -> None:
    # Unset -> uses the default; explicit sane values pass silently.
    _validate_smart_heading_max_chars()
    for good in ("180", "90", "300"):
        monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", good)
        _validate_smart_heading_max_chars()


@pytest.mark.parametrize("bad", ["abc", "1.5", "12x"])
def test_max_chars_non_integer_raises(monkeypatch, bad) -> None:
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", bad)
    with pytest.raises(ParserRoutingConfigError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        _validate_smart_heading_max_chars()


@pytest.mark.parametrize("bad", ["2", "1", "0", "-5"])
def test_max_chars_below_minimum_raises(monkeypatch, bad) -> None:
    """A cap < 3 cannot hold the '...' marker (same boundary heading_max_chars
    floors at runtime); reject it loudly at startup."""
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", bad)
    with pytest.raises(ParserRoutingConfigError, match="too small"):
        _validate_smart_heading_max_chars()


def test_max_chars_low_value_warns_not_raises(monkeypatch) -> None:
    """A usable-but-tiny cap (below the title-line width) warns, does not fail."""
    from lightrag.parser import routing

    calls: list = []
    monkeypatch.setattr(routing.logger, "warning", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "50")
    _validate_smart_heading_max_chars()  # no raise
    assert calls, "expected a warning for a below-title-width cap"


def test_startup_check_validates_max_chars_when_enabled(monkeypatch) -> None:
    """The MAX_CHARS check runs (and fails fast) before the spaCy model check
    when smart_heading is enabled."""
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "2")
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: [])
    with pytest.raises(ParserRoutingConfigError, match="too small"):
        validate_smart_heading_dependencies(parser_rules="")


def test_startup_check_ignores_max_chars_when_disabled(monkeypatch) -> None:
    """A bad MAX_CHARS is not surfaced for deployments that never enable
    smart_heading (the knob is never read there)."""
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "2")
    validate_smart_heading_dependencies(parser_rules="")  # switch off -> silent
