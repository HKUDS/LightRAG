import sys

import pytest

from lightrag.api.config import parse_args


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "selector,options,expected_binding,expected_option",
    [
        (
            ["--llm-binding=openai"],
            ["--openai-llm-temperature", "0.25"],
            "openai",
            "openai_llm_temperature",
        ),
        (
            ["--llm-binding", "openai"],
            ["--openai-llm-temperature", "0.25"],
            "openai",
            "openai_llm_temperature",
        ),
        (
            ["--llm-binding", "ollama", "--llm-binding=openai"],
            ["--openai-llm-temperature", "0.25"],
            "openai",
            "openai_llm_temperature",
        ),
        (
            ["--llm-binding", "ollama", "--llm-binding", "openai"],
            ["--openai-llm-temperature", "0.25"],
            "openai",
            "openai_llm_temperature",
        ),
        # The final parser accepts unambiguous abbreviations, so the selector
        # pre-parse has to resolve them the same way.
        (
            ["--llm-bind", "openai"],
            ["--openai-llm-temperature", "0.25"],
            "openai",
            "openai_llm_temperature",
        ),
        (
            ["--llm-b=openai"],
            ["--openai-llm-temperature", "0.25"],
            "openai",
            "openai_llm_temperature",
        ),
    ],
)
def test_llm_options_follow_effective_cli_binding(
    monkeypatch, selector, options, expected_binding, expected_option
):
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setattr(sys, "argv", ["lightrag-server", *selector, *options])

    args = parse_args()

    assert args.llm_binding == expected_binding
    assert getattr(args, expected_option) == 0.25


@pytest.mark.parametrize(
    "selector",
    [
        ["--embedding-binding=ollama"],
        ["--embedding-binding", "ollama"],
        ["--embedding-binding", "openai", "--embedding-binding=ollama"],
        ["--embedding-binding", "openai", "--embedding-binding", "ollama"],
        ["--embedding-bind", "ollama"],
    ],
)
def test_embedding_options_follow_effective_cli_binding(monkeypatch, selector):
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        ["lightrag-server", *selector, "--ollama-embedding-num_ctx", "4096"],
    )

    args = parse_args()

    assert args.embedding_binding == "ollama"
    assert args.ollama_embedding_num_ctx == 4096


@pytest.mark.parametrize(
    "selector",
    [
        ["--llm-binding=openai"],
        ["--llm-binding", "ollama", "--llm-binding", "openai"],
        ["--llm-bind", "openai"],
    ],
)
def test_env_only_llm_options_follow_effective_cli_binding(monkeypatch, selector):
    """Provider options set only in the environment must not be dropped.

    When the selector was misread, the OpenAI options were never registered, so
    ``OPENAI_LLM_TEMPERATURE`` was ignored without any error.
    """
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setenv("OPENAI_LLM_TEMPERATURE", "0.3")
    monkeypatch.setattr(sys, "argv", ["lightrag-server", *selector])

    args = parse_args()

    assert args.llm_binding == "openai"
    assert args.openai_llm_temperature == 0.3


def test_env_only_embedding_options_follow_effective_cli_binding(monkeypatch):
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("OLLAMA_EMBEDDING_NUM_CTX", "4096")
    monkeypatch.setattr(sys, "argv", ["lightrag-server", "--embedding-binding=ollama"])

    args = parse_args()

    assert args.embedding_binding == "ollama"
    assert args.ollama_embedding_num_ctx == 4096


@pytest.mark.parametrize(
    "argv",
    [
        ["--llm-binding"],
        ["--llm-binding", "--host", "0.0.0.0"],
        ["--embedding-binding"],
    ],
)
def test_malformed_selector_is_reported_by_the_full_parser(monkeypatch, capsys, argv):
    """A selector without a value is rejected with the server's full usage text,
    not the two-option usage of the selector pre-parse."""
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setattr(sys, "argv", ["lightrag-server", *argv])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "expected one argument" in err
    assert "--working-dir" in err
