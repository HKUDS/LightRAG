import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_gemini_module(monkeypatch):
    fake_pm = SimpleNamespace(
        is_installed=lambda name: True,
        install=lambda name: None,
    )

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        HttpOptions=FakeHttpOptions,
    )
    fake_genai = SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(kwargs=kwargs))
    fake_google_module = ModuleType("google")
    fake_google_module.genai = fake_genai
    fake_api_exceptions = SimpleNamespace(
        InternalServerError=type("InternalServerError", (Exception,), {}),
        ServiceUnavailable=type("ServiceUnavailable", (Exception,), {}),
        ResourceExhausted=type("ResourceExhausted", (Exception,), {}),
        GatewayTimeout=type("GatewayTimeout", (Exception,), {}),
        BadGateway=type("BadGateway", (Exception,), {}),
        DeadlineExceeded=type("DeadlineExceeded", (Exception,), {}),
        Aborted=type("Aborted", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
    )
    fake_google_api_core = ModuleType("google.api_core")
    fake_google_api_core.exceptions = fake_api_exceptions

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", SimpleNamespace(types=fake_types))
    monkeypatch.setitem(sys.modules, "google.api_core", fake_google_api_core)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", fake_api_exceptions)
    sys.modules.pop("lightrag.llm.gemini", None)

    return importlib.import_module("lightrag.llm.gemini")


@pytest.mark.offline
def test_gemini_maps_schema_response_format_to_response_json_schema(monkeypatch):
    gemini_module = _load_gemini_module(monkeypatch)

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    config = gemini_module._build_generation_config(
        base_config=None,
        system_prompt=None,
        response_format=schema,
    )

    assert config.kwargs["response_mime_type"] == "application/json"
    assert config.kwargs["response_json_schema"] == schema
    assert "response_schema" not in config.kwargs
