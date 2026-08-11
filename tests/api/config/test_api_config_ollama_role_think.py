"""OllamaLLMOptions.think: a normal, generically-configurable field.

A thinking-capable Ollama model can spend its entire generation budget on
a hidden reasoning trace before producing any actual output -- for the
extract/keyword roles that can mean entity/relation extraction silently
returns nothing (see issue #3597). think is exposed as a plain per-role
Ollama option (OLLAMA_LLM_THINK / {ROLE}_OLLAMA_LLM_THINK) so a user who
hits this can turn thinking off for the roles that need deterministic
output. The framework doesn't default it -- disabling thinking isn't
universally the right call, it measurably hurts extraction quality on
some smaller models.
"""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args
from lightrag.llm.binding_options import OllamaLLMOptions

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ollama_llm_binding(monkeypatch):
    """parse_args only registers the Ollama option group when LLM_BINDING
    resolves to ollama. That default holds in CI (no .env), but a developer
    .env pointing LLM_BINDING elsewhere would silently strip every option
    asserted below -- making the absence assertions pass for the wrong reason
    and the presence ones fail. Pin the binding instead of inheriting it."""
    monkeypatch.setenv("LLM_BINDING", "ollama")


def test_think_is_absent_when_unset(monkeypatch):
    """Every OllamaLLMOptions field is argparse.SUPPRESS-default until
    configured (confirmed against num_predict too, not think-specific) --
    unset means genuinely absent from the dict, not materialized as the
    dataclass's `True`. That absence is what lets Ollama's own per-model
    default apply when nothing overrides it."""
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.delenv("OLLAMA_LLM_THINK", raising=False)

    args = parse_args()

    assert "think" not in OllamaLLMOptions.options_dict(args)


def test_think_is_configurable_globally_like_any_other_ollama_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("OLLAMA_LLM_THINK", "false")

    args = parse_args()

    assert OllamaLLMOptions.options_dict(args)["think"] is False


def test_think_is_configurable_per_role(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.delenv("OLLAMA_LLM_THINK", raising=False)
    monkeypatch.setenv("QUERY_OLLAMA_LLM_THINK", "false")

    args = parse_args()

    assert OllamaLLMOptions.options_dict_for_role(args, "query")["think"] is False
    assert "think" not in OllamaLLMOptions.options_dict_for_role(args, "extract")
