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


def _make_fake_gemini_response(regular_text="", thought_text=""):
    parts = []
    if thought_text:
        parts.append(SimpleNamespace(text=thought_text, thought=True))
    if regular_text:
        parts.append(SimpleNamespace(text=regular_text, thought=False))

    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=parts)),
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=2,
            total_token_count=3,
        ),
    )


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


@pytest.mark.offline
def test_gemini_unwraps_openai_json_schema_wrapper(monkeypatch):
    gemini_module = _load_gemini_module(monkeypatch)

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_payload",
            "schema": schema,
        },
    }

    config = gemini_module._build_generation_config(
        base_config=None,
        system_prompt=None,
        response_format=response_format,
    )

    assert config.kwargs["response_mime_type"] == "application/json"
    assert config.kwargs["response_json_schema"] == schema


@pytest.mark.offline
def test_gemini_rejects_typed_response_format(monkeypatch):
    gemini_module = _load_gemini_module(monkeypatch)

    class FakeSchemaModel:
        pass

    with pytest.raises(TypeError, match="typed/Pydantic"):
        gemini_module._validate_gemini_response_format(FakeSchemaModel)


@pytest.mark.offline
def test_gemini_default_service_root_is_not_treated_as_custom_base_url(monkeypatch):
    gemini_module = _load_gemini_module(monkeypatch)
    gemini_module._get_gemini_client.cache_clear()
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    client = gemini_module._get_gemini_client(
        "test-key",
        "https://generativelanguage.googleapis.com",
        1234,
    )

    assert client.kwargs["api_key"] == "test-key"
    assert "http_options" in client.kwargs
    assert client.kwargs["http_options"].kwargs == {"timeout": 1234}


@pytest.mark.offline
def test_gemini_custom_base_url_is_preserved(monkeypatch):
    gemini_module = _load_gemini_module(monkeypatch)
    gemini_module._get_gemini_client.cache_clear()
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    client = gemini_module._get_gemini_client(
        "test-key",
        "https://proxy.example.com",
        1234,
    )

    assert client.kwargs["http_options"].kwargs == {
        "base_url": "https://proxy.example.com",
        "timeout": 1234,
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_streaming_structured_output_disables_cot(monkeypatch):
    gemini_module = _load_gemini_module(monkeypatch)

    fake_stream_response = _make_fake_gemini_response(
        regular_text='{"answer":"ok"}',
        thought_text="this should not be included",
    )

    async def _single_chunk_stream(response):
        yield response

    async def _fake_generate_content_stream(**kwargs):
        return _single_chunk_stream(fake_stream_response)

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=_fake_generate_content_stream
            )
        )
    )

    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    stream = await gemini_module.gemini_complete_if_cache(
        model="gemini-model",
        prompt="hello",
        stream=True,
        enable_cot=True,
        response_format={"type": "json_object"},
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert "".join(chunks) == '{"answer":"ok"}'
