"""Malformed numeric values in role-specific provider option overrides.

``BindingOptions.options_dict_for_role`` reads a ``{ROLE}_{BINDING}_{FIELD}``
override for an ``int``/``float`` field by calling ``int(env_raw)`` /
``float(env_raw)`` directly. When that raises (a typo such as writing
``"500.0"`` for an ``int`` field, or ``"0.7x"`` for a ``float`` one), the
surrounding ``except`` falls back to ``base[field_name] = env_raw`` -- the
*raw, unconverted string* -- with no warning. That string then reaches the
provider's request payload (e.g. ``temperature="0.7x"``) instead of a number,
so a config typo surfaces as a confusing request-time API error far from its
cause.

This mirrors the boolean override bug fixed in `96db09602` (an unrecognized
spelling must not silently corrupt the option): an int/float override should
warn and leave the field unset -- same as the base, non-role path already
does through ``get_env_value`` -- rather than smuggle a string into a numeric
field.
"""

from argparse import Namespace

import pytest

from lightrag.llm.binding_options import OpenAILLMOptions

pytestmark = pytest.mark.offline


def test_role_int_override_typo_is_ignored_with_warning(
    monkeypatch, lightrag_log_records
):
    monkeypatch.setenv("EXTRACT_OPENAI_LLM_MAX_COMPLETION_TOKENS", "500.0")

    options = OpenAILLMOptions.options_dict_for_role(Namespace(), "extract")

    assert "max_completion_tokens" not in options
    assert any(
        "EXTRACT_OPENAI_LLM_MAX_COMPLETION_TOKENS" in r.getMessage()
        for r in lightrag_log_records
    )


def test_role_float_override_typo_is_ignored_with_warning(
    monkeypatch, lightrag_log_records
):
    monkeypatch.setenv("EXTRACT_OPENAI_LLM_TEMPERATURE", "0.7x")

    options = OpenAILLMOptions.options_dict_for_role(Namespace(), "extract")

    assert "temperature" not in options
    assert any(
        "EXTRACT_OPENAI_LLM_TEMPERATURE" in r.getMessage() for r in lightrag_log_records
    )


def test_role_int_override_valid_value_still_parses(monkeypatch):
    monkeypatch.setenv("EXTRACT_OPENAI_LLM_MAX_COMPLETION_TOKENS", " 500 ")

    options = OpenAILLMOptions.options_dict_for_role(Namespace(), "extract")

    assert options["max_completion_tokens"] == 500
    assert isinstance(options["max_completion_tokens"], int)


def test_role_float_override_valid_value_still_parses(monkeypatch):
    monkeypatch.setenv("EXTRACT_OPENAI_LLM_TEMPERATURE", " 0.7 ")

    options = OpenAILLMOptions.options_dict_for_role(Namespace(), "extract")

    assert options["temperature"] == pytest.approx(0.7)
    assert isinstance(options["temperature"], float)
