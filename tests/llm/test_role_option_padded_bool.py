"""Whitespace-padded booleans in provider option overrides.

``get_env_value(..., bool)`` strips the raw value before matching it, so
``OLLAMA_LLM_LOW_VRAM=" true "`` is read as ``True``. The role-specific
override (``EXTRACT_OLLAMA_LLM_LOW_VRAM``) is parsed separately by
``BindingOptions.options_dict_for_role`` and must read the same value the
same way instead of silently turning it into ``False``. So must the
command-line flag (``--ollama-llm-low_vram``), which has its own parser.
"""

from argparse import ArgumentParser, Namespace

import pytest

from lightrag.llm.binding_options import OllamaLLMOptions

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" true ", True),
        ("true\n", True),
        ("\tyes\t", True),
        (" 1 ", True),
        (" ON ", True),
        (" false ", False),
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
        (" true ", True),
        ("true\n", True),
        ("\tyes\t", True),
        (" ON ", True),
        (" false ", False),
        ("  0\n", False),
    ],
)
def test_cli_bool_option_ignores_surrounding_whitespace(monkeypatch, raw, expected):
    monkeypatch.delenv("OLLAMA_LLM_LOW_VRAM", raising=False)
    parser = ArgumentParser()
    OllamaLLMOptions.add_args(parser)

    args = parser.parse_args(["--ollama-llm-low_vram", raw])

    assert args.ollama_llm_low_vram is expected
