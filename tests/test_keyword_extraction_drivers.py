from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache
from lightrag.llm.lollms import lollms_model_complete
from lightrag.llm.ollama import _ollama_model_if_cache, ollama_model_complete


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ollama_keyword_extraction_preserves_explicit_flag():
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

    assert mocked_complete.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


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
async def test_lollms_keyword_extraction_preserves_explicit_flag():
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

    forwarded_kwargs = mocked_complete.await_args.kwargs
    assert "keyword_extraction" not in forwarded_kwargs


@pytest.mark.offline
@pytest.mark.asyncio
async def test_lmdeploy_strips_keyword_extraction_before_generation_config(monkeypatch):
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
        keyword_extraction=True,
    )

    assert result == "{}"
    assert "keyword_extraction" not in captured_gen_config_kwargs
