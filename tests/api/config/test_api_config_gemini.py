import pytest

from lightrag.api.config import get_default_host


pytestmark = pytest.mark.offline


def test_gemini_default_host_uses_sdk_default_endpoint(monkeypatch):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)

    assert get_default_host("gemini") == "DEFAULT_GEMINI_ENDPOINT"
