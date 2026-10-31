"""Deprecated extraction flags stay inside the binding, outside Ollama's API."""

import json
from types import SimpleNamespace
import warnings

import httpx
import pytest

import lightrag.llm.ollama as binding

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("entry", ["direct", "wrapper"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
async def test_extraction_flags_use_real_ollama_signature(
    monkeypatch, entry, stream, keyword, response_format
):
    requests = []
    clients = []
    real_client = binding.ollama.AsyncClient

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {"role": "assistant", "content": "{}"},
                "done": True,
                "done_reason": "stop",
            },
        )

    def client_factory(**kwargs):
        client = real_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(binding.ollama, "AsyncClient", client_factory)
    kwargs = {
        "entity_extraction": True,
        "keyword_extraction": keyword,
        "response_format": response_format,
        "host": "http://ollama.test",
        "stream": stream,
    }
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", DeprecationWarning)
        if entry == "direct":
            result = await binding._ollama_model_if_cache(
                "test-model", "hello", **kwargs
            )
        else:
            cache = SimpleNamespace(global_config={"llm_model_name": "test-model"})
            result = await binding.ollama_model_complete(
                "hello", hashing_kv=cache, **kwargs
            )
        content = "".join([part async for part in result]) if stream else result

    assert content == "{}"
    assert len(requests) == 1
    assert requests[0]["format"] == "json"
    assert "entity_extraction" not in requests[0]
    assert "keyword_extraction" not in requests[0]
    legacy_warnings = [
        notice for notice in notices if "extraction=" in str(notice.message)
    ]
    assert len(legacy_warnings) == (1 if response_format is None else 0)
    assert all(
        "entity_extraction=True" in str(notice.message) for notice in legacy_warnings
    )
    assert all(client._client.is_closed for client in clients)
