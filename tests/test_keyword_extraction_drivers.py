from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache
from lightrag.llm.lollms import lollms_model_complete, lollms_model_if_cache
from lightrag.llm.ollama import _ollama_model_if_cache, ollama_model_complete


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ollama_response_format_forwards_to_inner():
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "ollama-model"})

    with patch(
        "lightrag.llm.ollama._ollama_model_if_cache",
        AsyncMock(return_value="{}"),
    ) as mocked_complete:
        await ollama_model_complete(
            prompt="hello",
            hashing_kv=hashing_kv,
            response_format={"type": "json_object"},
        )

    assert mocked_complete.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ollama_legacy_keyword_extraction_emits_deprecation_warning():
    """_ollama_model_if_cache is the canonical emission site for the shim."""
    captured_kwargs = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self._client = SimpleNamespace(aclose=AsyncMock())

        async def chat(self, **kwargs):
            captured_kwargs.update(kwargs)
            return {"message": {"content": "{}"}}

    with patch("lightrag.llm.ollama.ollama.AsyncClient", FakeAsyncClient):
        with pytest.warns(DeprecationWarning):
            await _ollama_model_if_cache(
                model="ollama-model",
                prompt="hello",
                keyword_extraction=True,
            )

    assert captured_kwargs["format"] == "json"
    assert "keyword_extraction" not in captured_kwargs
    assert "response_format" not in captured_kwargs


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ollama_complete_forwards_legacy_flag_downstream():
    """ollama_model_complete is a pure forwarder; the shim fires inside _if_cache."""
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "ollama-model"})

    with patch(
        "lightrag.llm.ollama._ollama_model_if_cache",
        AsyncMock(return_value="{}"),
    ) as mocked_complete:
        await ollama_model_complete(
            prompt="hello",
            hashing_kv=hashing_kv,
            keyword_extraction=True,
        )

    assert mocked_complete.await_args.kwargs.get("keyword_extraction") is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ollama_translates_json_object_response_format_to_native_format():
    captured_kwargs = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self._client = SimpleNamespace(aclose=AsyncMock())

        async def chat(self, **kwargs):
            captured_kwargs.update(kwargs)
            return {"message": {"content": "{}"}}

    with patch("lightrag.llm.ollama.ollama.AsyncClient", FakeAsyncClient):
        result = await _ollama_model_if_cache(
            model="ollama-model",
            prompt="hello",
            response_format={"type": "json_object"},
        )

    assert result == "{}"
    assert captured_kwargs["format"] == "json"
    assert "response_format" not in captured_kwargs


@pytest.mark.offline
@pytest.mark.asyncio
async def test_lollms_if_cache_strips_response_format_before_request():
    """lollms_model_if_cache drops response_format; lollms has no JSON mode."""
    captured_requests = []

    class FakeResponse:
        def __init__(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def text(self):
            return "{}"

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def post(self, url, json):
            captured_requests.append(json)
            return FakeResponse()

    with patch("lightrag.llm.lollms.aiohttp.ClientSession", FakeSession):
        result = await lollms_model_if_cache(
            model="lollms-model",
            prompt="hello",
            response_format={"type": "json_object"},
        )

    assert result == "{}"
    assert captured_requests
    assert "response_format" not in captured_requests[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_lollms_if_cache_emits_deprecation_warning():
    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def text(self):
            return "{}"

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def post(self, url, json):
            return FakeResponse()

    with patch("lightrag.llm.lollms.aiohttp.ClientSession", FakeSession):
        with pytest.warns(DeprecationWarning):
            await lollms_model_if_cache(
                model="lollms-model",
                prompt="hello",
                keyword_extraction=True,
            )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_lollms_complete_forwards_legacy_flag_downstream():
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "lollms-model"})

    with patch(
        "lightrag.llm.lollms.lollms_model_if_cache",
        AsyncMock(return_value="{}"),
    ) as mocked_complete:
        await lollms_model_complete(
            prompt="hello",
            hashing_kv=hashing_kv,
            keyword_extraction=True,
        )

    assert mocked_complete.await_args.kwargs.get("keyword_extraction") is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_lmdeploy_strips_response_format_before_generation_config(monkeypatch):
    captured_gen_config_kwargs = {}

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            captured_gen_config_kwargs.update(kwargs)

    class FakeVersion:
        def __lt__(self, other):
            return False

    async def fake_generate(*_args, **_kwargs):
        yield SimpleNamespace(response="{}")

    monkeypatch.setattr(
        "lightrag.llm.lmdeploy.initialize_lmdeploy_pipeline",
        lambda **_kwargs: SimpleNamespace(generate=fake_generate),
    )

    import sys

    sys.modules["lmdeploy"] = SimpleNamespace(
        __version__="0.6.0",
        version_info=FakeVersion(),
        GenerationConfig=FakeGenerationConfig,
    )

    result = await lmdeploy_model_if_cache(
        model="lmdeploy-model",
        prompt="hello",
        response_format={"type": "json_object"},
    )

    assert result == "{}"
    assert "response_format" not in captured_gen_config_kwargs
    assert "keyword_extraction" not in captured_gen_config_kwargs
