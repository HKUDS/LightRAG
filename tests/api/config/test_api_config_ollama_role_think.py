"""Per-role default for Ollama's `think` (extended-reasoning) option.

A thinking-capable Ollama model (e.g. gemma4:12b) can spend its entire
generation budget on a hidden reasoning trace before producing any actual
output. For a user-facing query that is a reasonable trade (a bit of extra
latency for a better-considered answer); for the extract/keyword roles it
means entity/relation extraction silently returns nothing -- reported as a
successful, empty knowledge graph, no error raised. See issue #3597.

_apply_ollama_role_think_default() (lightrag/api/lightrag_server.py) forces
think=False specifically for those two roles, but only when the user hasn't
already made an explicit choice via OLLAMA_LLM_THINK or the role's own
{ROLE}_OLLAMA_LLM_THINK.
"""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args
from lightrag.llm.binding_options import OllamaLLMOptions

pytestmark = pytest.mark.offline


# --- OllamaLLMOptions.think itself: a normal, generically-configurable field ---


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


# --- _apply_ollama_role_think_default(): the extract/keyword-specific override ---


def _apply(monkeypatch, role: str, *, extract_think=None, global_think=None):
    monkeypatch.delenv("OLLAMA_LLM_THINK", raising=False)
    monkeypatch.delenv("EXTRACT_OLLAMA_LLM_THINK", raising=False)
    if global_think is not None:
        monkeypatch.setenv("OLLAMA_LLM_THINK", global_think)
    if extract_think is not None:
        monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", extract_think)

    # lightrag.api.lightrag_server transitively imports lightrag.api.auth,
    # whose module-level AuthHandler() singleton reads sys.argv on first
    # import -- patch it before triggering that import, same as
    # test_api_config_ollama_embedding_dim.py.
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _apply_ollama_role_think_default

    options: dict = {"num_predict": 4096}
    _apply_ollama_role_think_default(role, options)
    return options


def test_extract_role_defaults_to_no_thinking(monkeypatch):
    options = _apply(monkeypatch, "extract")

    assert options["think"] is False


def test_keyword_role_defaults_to_no_thinking(monkeypatch):
    options = _apply(monkeypatch, "keyword")

    assert options["think"] is False


def test_query_role_is_not_touched(monkeypatch):
    """query is deliberately left alone -- a slower, better-considered final
    answer is a reasonable trade a user might want, unlike extraction."""
    options = _apply(monkeypatch, "query")

    assert "think" not in options


def test_explicit_role_choice_wins_over_the_default(monkeypatch):
    options = _apply(monkeypatch, "extract", extract_think="true")

    assert "think" not in options


def test_explicit_global_choice_also_wins_over_the_default(monkeypatch):
    """A user who set OLLAMA_LLM_THINK=true globally clearly wants thinking
    on everywhere, extract included -- the role default must not silently
    override that broader intent."""
    options = _apply(monkeypatch, "extract", global_think="true")

    assert "think" not in options
