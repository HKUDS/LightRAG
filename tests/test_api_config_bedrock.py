import pytest

from lightrag.api.config import get_default_host


pytestmark = pytest.mark.offline


def test_bedrock_default_host_uses_sdk_default_endpoint(monkeypatch):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)

    assert get_default_host("aws_bedrock") == "DEFAULT_BEDROCK_ENDPOINT"


def test_bedrock_custom_host_is_returned(monkeypatch):
    monkeypatch.setenv("LLM_BINDING_HOST", "https://proxy.example.com")

    assert get_default_host("aws_bedrock") == "https://proxy.example.com"
