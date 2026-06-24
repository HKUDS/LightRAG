import numpy as np
import pytest

from lightrag.llm.lmstudio import (
    clear_lmstudio_model_cache,
    is_any_available_model,
    is_auto_embedding_dim,
    probe_lmstudio_embedding_dim,
    resolve_lmstudio_model,
    _normalize_lmstudio_response_format,
)

pytestmark = pytest.mark.offline


def test_is_any_available_model_aliases():
    assert is_any_available_model(None)
    assert is_any_available_model("")
    assert is_any_available_model("   ")
    assert is_any_available_model("any-available")
    assert is_any_available_model("ANY_AVAILABLE")
    assert is_any_available_model("auto")
    assert not is_any_available_model("meta-llama-3.1-8b-instruct")


@pytest.mark.asyncio
async def test_resolve_lmstudio_model_uses_loaded_v0_instance(monkeypatch):
    clear_lmstudio_model_cache()

    async def fake_list_v0(base_url, api_key):
        assert base_url == "http://localhost:1234/v1"
        assert api_key == "lm-studio"
        return [
            {
                "type": "llm",
                "id": "meta-llama-3.1-8b-instruct",
                "loaded_instances": [{"id": "loaded-chat-model"}],
            },
            {
                "type": "embeddings",
                "id": "text-embedding-nomic-embed-text-v1.5",
                "loaded_instances": [{"id": "loaded-embed-model"}],
            },
        ]

    monkeypatch.setattr(
        "lightrag.llm.lmstudio._list_v0_models",
        fake_list_v0,
    )

    llm_model = await resolve_lmstudio_model(
        "any-available",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        purpose="llm",
    )
    embed_model = await resolve_lmstudio_model(
        "",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        purpose="embedding",
    )

    assert llm_model == "loaded-chat-model"
    assert embed_model == "loaded-embed-model"


@pytest.mark.asyncio
async def test_resolve_lmstudio_model_returns_explicit_id_without_lookup(monkeypatch):
    clear_lmstudio_model_cache()
    called = False

    async def fake_list_v0(base_url, api_key):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "lightrag.llm.lmstudio._list_v0_models",
        fake_list_v0,
    )

    resolved = await resolve_lmstudio_model(
        "my-explicit-model",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        purpose="llm",
    )

    assert resolved == "my-explicit-model"
    assert called is False


def test_is_auto_embedding_dim_aliases():
    assert is_auto_embedding_dim(None)
    assert is_auto_embedding_dim("")
    assert is_auto_embedding_dim("auto")
    assert not is_auto_embedding_dim(768)


@pytest.mark.asyncio
async def test_probe_lmstudio_embedding_dim(monkeypatch):
    clear_lmstudio_model_cache()

    async def fake_resolve(model, base_url=None, api_key=None, purpose="llm"):
        assert purpose == "embedding"
        return "loaded-embed-model"

    async def fake_embed(texts, model, base_url=None, api_key=None, embedding_dim=None):
        assert model == "loaded-embed-model"
        assert embedding_dim is None
        return np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    monkeypatch.setattr(
        "lightrag.llm.lmstudio.resolve_lmstudio_model",
        fake_resolve,
    )
    monkeypatch.setattr(
        "lightrag.llm.lmstudio.openai_embed.func",
        fake_embed,
    )

    dim, resolved_model = await probe_lmstudio_embedding_dim(
        "any-available",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )

    assert dim == 4
    assert resolved_model == "loaded-embed-model"


def test_normalize_lmstudio_response_format_maps_json_object_to_text():
    kwargs: dict = {"response_format": {"type": "json_object"}}
    _normalize_lmstudio_response_format(kwargs)
    assert kwargs["response_format"] == {"type": "text"}


def test_normalize_lmstudio_response_format_preserves_json_schema():
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "test", "schema": {"type": "object"}},
    }
    kwargs = {"response_format": schema}
    _normalize_lmstudio_response_format(kwargs)
    assert kwargs["response_format"] == schema
