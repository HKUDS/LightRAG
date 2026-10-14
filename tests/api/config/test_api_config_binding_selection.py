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
