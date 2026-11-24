"""How provider option values are read from the env and the command line.

Booleans: ``get_env_value(..., bool)`` strips the raw value before matching
it, so ``OLLAMA_LLM_LOW_VRAM=" true "`` is read as ``True``. The role-specific
override (``EXTRACT_OLLAMA_LLM_LOW_VRAM``) is parsed separately by
``BindingOptions.options_dict_for_role`` and must read the same value the
same way, and so must the command-line flag (``--ollama-llm-low_vram``).

Role overrides that do not parse (a bool, int, float or JSON field): the
override is ignored with a warning naming the variable, so the role keeps the
base option. Storing the raw string instead would send e.g.
``temperature="0.7x"`` to the provider and fail at request time.
"""

from argparse import ArgumentParser, Namespace

import pytest

from lightrag.llm.binding_options import OllamaLLMOptions, OpenAILLMOptions

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" ON ", True),
        ("  0\n", False),
    ],
)
def test_role_bool_override_ignores_surrounding_whitespace(monkeypatch, raw, expected):
    monkeypatch.setenv("EXTRACT_OLLAMA_LLM_LOW_VRAM", raw)

    options = OllamaLLMOptions.options_dict_for_role(Namespace(), "extract")

    assert options["low_vram"] is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" ON ", True),
        ("  0\n", False),
    ],
)
def test_cli_bool_option_ignores_surrounding_whitespace(monkeypatch, raw, expected):
    monkeypatch.delenv("OLLAMA_LLM_LOW_VRAM", raising=False)
    parser = ArgumentParser()
    OllamaLLMOptions.add_args(parser)

    args = parser.parse_args(["--ollama-llm-low_vram", raw])

    assert args.ollama_llm_low_vram is expected


@pytest.mark.parametrize(
    ("options_cls", "env_name", "raw", "field_name", "base_value"),
    [
        (OllamaLLMOptions, "EXTRACT_OLLAMA_LLM_LOW_VRAM", "ture", "low_vram", True),
        (
            OpenAILLMOptions,
            "EXTRACT_OPENAI_LLM_MAX_COMPLETION_TOKENS",
            "500.0",
            "max_completion_tokens",
            321,
        ),
        (
            OpenAILLMOptions,
            "EXTRACT_OPENAI_LLM_TEMPERATURE",
            "0.7x",
            "temperature",
            0.3,
        ),
        (
            OpenAILLMOptions,
            "EXTRACT_OPENAI_LLM_EXTRA_BODY",
            "not json",
            "extra_body",
            {"base": True},
        ),
        (OpenAILLMOptions, "EXTRACT_OPENAI_LLM_STOP", '{"a": 1}', "stop", ["END"]),
    ],
    ids=["bool", "int", "float", "json-invalid", "json-wrong-shape"],
)
def test_role_override_that_does_not_parse_keeps_base_with_warning(
    monkeypatch,
    lightrag_log_records,
    options_cls,
    env_name,
    raw,
    field_name,
    base_value,
):
    monkeypatch.setenv(env_name, raw)
    args = Namespace(**{f"{options_cls._binding_name}_{field_name}": base_value})

    options = options_cls.options_dict_for_role(args, "extract")

    assert options[field_name] == base_value
    warnings = [
        r.getMessage() for r in lightrag_log_records if env_name in r.getMessage()
    ]
    assert len(warnings) == 1
    # The warning names what applies instead, not the internal sentinel.
    assert "base option or provider default" in warnings[0]


def test_cli_bool_option_rejects_typo(monkeypatch, capsys):
    monkeypatch.delenv("OLLAMA_LLM_LOW_VRAM", raising=False)
    parser = ArgumentParser()
    OllamaLLMOptions.add_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--ollama-llm-low_vram", "ture"])
    assert "OLLAMA_LLM_LOW_VRAM" in capsys.readouterr().err


def test_env_bool_option_typo_keeps_binding_default(monkeypatch, lightrag_log_records):
    monkeypatch.setenv("OLLAMA_LLM_LOW_VRAM", "ture")
    parser = ArgumentParser()
    OllamaLLMOptions.add_args(parser)

    args = parser.parse_args([])

    assert not hasattr(args, "ollama_llm_low_vram")
    assert any("OLLAMA_LLM_LOW_VRAM" in r.getMessage() for r in lightrag_log_records)
