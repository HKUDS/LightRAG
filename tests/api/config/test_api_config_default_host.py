"""``get_default_host``: each binding's default endpoint and its override.

A binding missing from the table falls back to the Ollama default
(http://localhost:11434), so every request it makes fails at connect time.
The ``DEFAULT_*_ENDPOINT`` sentinels let the provider SDK pick its own regional
or versioned endpoint unless ``LLM_BINDING_HOST`` overrides it.
"""

import pytest

from lightrag.api.config import get_default_host

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("binding", "default"),
    [
        ("gemini", "DEFAULT_GEMINI_ENDPOINT"),
        ("bedrock", "DEFAULT_BEDROCK_ENDPOINT"),
        ("openai_responses", "https://api.openai.com/v1"),
    ],
)
def test_default_host_and_its_override(monkeypatch, binding, default):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    assert get_default_host(binding) == default

    monkeypatch.setenv("LLM_BINDING_HOST", "https://gateway.example/v1")
    assert get_default_host(binding) == "https://gateway.example/v1"
