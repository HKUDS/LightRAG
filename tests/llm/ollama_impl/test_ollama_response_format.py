"""Offline tests: how the Ollama binding builds the native ``format`` field.

OpenAI-style ``response_format`` is translated into Ollama's ``format``, and
the deprecated extraction flags stay inside the binding, outside Ollama's API:
the real ``AsyncClient.chat`` declares no ``**kwargs``, so a flag that reaches
it is a ``TypeError`` rather than a silently ignored argument.
"""

import json
from types import SimpleNamespace
import warnings

import httpx
import pytest

import lightrag.llm.ollama as binding

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("entry", ["direct", "wrapper"])
@pytest.mark.parametrize(
    ("flags", "response_format", "warned_flag"),
    [
        # An explicit False is still a kwarg the real chat() would reject.
        pytest.param(
            {"entity_extraction": True, "keyword_extraction": False},
            None,
            "entity",
            id="entity",
        ),
        pytest.param({"keyword_extraction": True}, None, "keyword", id="keyword"),
        # entity wins the elif, so keyword must still be consumed.
        pytest.param(
            {"entity_extraction": True, "keyword_extraction": True},
            None,
            "entity",
            id="both",
        ),
        # An explicit response_format consumes both flags without a warning.
        pytest.param(
            {"entity_extraction": True, "keyword_extraction": True},
            {"type": "json_object"},
            None,
            id="both-explicit-format",
        ),
    ],
)
async def test_extraction_flags_use_real_ollama_signature(
    monkeypatch, entry, flags, response_format, warned_flag
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
        **flags,
        "response_format": response_format,
        "host": "http://ollama.test",
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

    assert result == "{}"
    assert len(requests) == 1
    assert requests[0]["format"] == "json"
    assert "entity_extraction" not in requests[0]
    assert "keyword_extraction" not in requests[0]
    legacy_warnings = [
        str(notice.message)
        for notice in notices
        if issubclass(notice.category, DeprecationWarning)
        and "extraction=" in str(notice.message)
    ]
    if warned_flag is None:
        assert legacy_warnings == []
    else:
        assert len(legacy_warnings) == 1
        assert f"{warned_flag}_extraction=True" in legacy_warnings[0]
    assert len(clients) == 1
    assert clients[0]._client.is_closed


_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


@pytest.mark.parametrize(
    ("kwargs", "expected_format"),
    [
        pytest.param(
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer_payload", "schema": _SCHEMA},
                }
            },
            _SCHEMA,
            id="json-schema-unwrapped",
        ),
        pytest.param(
            {"response_format": {"type": "json_schema", "json_schema": _SCHEMA}},
            _SCHEMA,
            id="json-schema-without-schema-key",
        ),
        pytest.param({"response_format": _SCHEMA}, _SCHEMA, id="raw-schema"),
        pytest.param(
            {"format": "json", "response_format": {"type": "json_schema"}},
            "json",
            id="native-format-wins",
        ),
    ],
)
def test_response_format_maps_to_native_format(kwargs, expected_format):
    binding._normalize_ollama_response_format(kwargs)

    assert kwargs == {"format": expected_format}
