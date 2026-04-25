import importlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.llm.bedrock import (
    bedrock_complete,
    bedrock_complete_if_cache,
    bedrock_embed,
)


def _reload_api_modules_if_mocked() -> None:
    """Drop Mock-replaced lightrag.api entries so importlib reloads the real modules.

    Other test files (e.g. test_token_auto_renewal.py) replace
    ``sys.modules["lightrag.api.config"]`` with a Mock at import time. When
    pytest collects those files before ours, any subsequent
    ``from .config import global_args`` inside lightrag_server picks up the
    Mock, which breaks ``create_app`` in create_app_* tests below.
    """
    for modname in (
        "lightrag.api.lightrag_server",
        "lightrag.api.auth",
        "lightrag.api.config",
    ):
        if isinstance(sys.modules.get(modname), Mock):
            sys.modules.pop(modname, None)


class _FakeBedrockClient:
    def __init__(self, captured_calls: list[dict]):
        self._captured_calls = captured_calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **kwargs):
        self._captured_calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
                        }
                    ]
                }
            }
        }


class _FakeSession:
    def __init__(self, captured_calls: list[dict], client_kwargs_calls: list[dict]):
        self._captured_calls = captured_calls
        self._client_kwargs_calls = client_kwargs_calls

    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeBedrockClient(self._captured_calls)


class _FakeReasoningClient(_FakeBedrockClient):
    async def converse(self, **kwargs):
        self._captured_calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "internal thought"}
                            }
                        },
                        {"text": "final answer"},
                    ]
                }
            }
        }


class _FakeReasoningSession(_FakeSession):
    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeReasoningClient(self._captured_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_complete_skips_reasoning_content_block(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    captured_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeReasoningSession(captured_calls, []),
    ):
        result = await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            extra_fields={"reasoning_config": {"type": "enabled"}},
        )

    assert result == "final answer"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_complete_forwards_keyword_extraction_to_if_cache():
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "bedrock-model"})

    with patch(
        "lightrag.llm.bedrock.bedrock_complete_if_cache",
        AsyncMock(return_value="{}"),
    ) as mocked_complete:
        await bedrock_complete(
            prompt="hello",
            hashing_kv=hashing_kv,
            keyword_extraction=True,
        )

    assert mocked_complete.await_args.kwargs["keyword_extraction"] is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_keyword_extraction_does_not_inject_system_prompt(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        result = await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            response_format={"type": "json_object"},
        )

    assert result == '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    assert len(captured_calls) == 1
    assert "system" not in captured_calls[0]
    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_default_endpoint_sentinel_uses_sdk_default(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            endpoint_url="DEFAULT_BEDROCK_ENDPOINT",
        )

    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_empty_endpoint_url_uses_sdk_default(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            endpoint_url="",
        )

    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_custom_endpoint_url_is_forwarded(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            endpoint_url="https://proxy.example.com",
        )

    assert client_kwargs_calls[-1] == {
        "region_name": None,
        "endpoint_url": "https://proxy.example.com",
    }


class _FakeEmbeddingBody:
    async def json(self):
        return {"embedding": [0.1] * 1024}


class _FakeEmbeddingResponse:
    def get(self, key):
        assert key == "body"
        return _FakeEmbeddingBody()


class _FakeEmbeddingClient(_FakeBedrockClient):
    async def invoke_model(self, **_kwargs):
        return _FakeEmbeddingResponse()


class _FakeEmbeddingSession(_FakeSession):
    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeEmbeddingClient(self._captured_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_embed_custom_endpoint_url_is_forwarded(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeEmbeddingSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_embed(
            texts=["hello"],
            endpoint_url="https://proxy.example.com",
        )

    assert client_kwargs_calls[-1] == {
        "region_name": None,
        "endpoint_url": "https://proxy.example.com",
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_embed_default_endpoint_sentinel_uses_sdk_default(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeEmbeddingSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_embed(
            texts=["hello"],
            endpoint_url="DEFAULT_BEDROCK_ENDPOINT",
        )

    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_embed_empty_endpoint_url_uses_sdk_default(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeEmbeddingSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_embed(
            texts=["hello"],
            endpoint_url="",
        )

    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_complete_forwards_explicit_sigv4_client_kwargs(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            aws_region="us-west-2",
            aws_access_key_id="akid",
            aws_secret_access_key="secret",
            aws_session_token="session",
            endpoint_url="https://proxy.example.com",
        )

    assert client_kwargs_calls[-1] == {
        "region_name": "us-west-2",
        "endpoint_url": "https://proxy.example.com",
        "aws_access_key_id": "akid",
        "aws_secret_access_key": "secret",
        "aws_session_token": "session",
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_extra_fields_maps_to_additional_model_request_fields(
    monkeypatch,
):
    monkeypatch.delenv("AWS_REGION", raising=False)
    captured_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, []),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            extra_fields={"reasoning_config": {"type": "enabled"}},
        )

    assert captured_calls[-1]["additionalModelRequestFields"] == {
        "reasoning_config": {"type": "enabled"}
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_empty_extra_fields_is_dropped(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    captured_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, []),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            extra_fields=None,
        )
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            extra_fields={},
        )

    for call in captured_calls:
        assert "additionalModelRequestFields" not in call


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_api_key_is_ignored_and_does_not_mutate_env(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "absk-from-env")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession([], []),
    ):
        with pytest.warns(DeprecationWarning, match="api_key=.*ignored"):
            await bedrock_complete_if_cache(
                model="bedrock-model",
                prompt="hello",
                api_key="absk-should-be-ignored",
                aws_access_key_id="akid",
                aws_secret_access_key="secret",
                aws_session_token="session",
            )

    assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "absk-from-env"
    assert os.environ.get("AWS_ACCESS_KEY_ID") is None
    assert os.environ.get("AWS_SECRET_ACCESS_KEY") is None
    assert os.environ.get("AWS_SESSION_TOKEN") is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_embed_forwards_sigv4_and_ignores_api_key(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeEmbeddingSession([], client_kwargs_calls),
    ):
        with pytest.warns(DeprecationWarning, match="api_key=.*ignored"):
            await bedrock_embed(
                texts=["hello"],
                api_key="absk-embedding-key",
                aws_region="us-east-1",
                aws_access_key_id="akid",
                aws_secret_access_key="secret",
                aws_session_token="session",
            )

    assert client_kwargs_calls[-1] == {
        "region_name": "us-east-1",
        "aws_access_key_id": "akid",
        "aws_secret_access_key": "secret",
        "aws_session_token": "session",
    }
    assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") is None


@pytest.mark.offline
def test_bedrock_auth_docstrings_describe_generic_api_key_behavior():
    assert "AWS_BEARER_TOKEN_BEDROCK" in bedrock_complete_if_cache.__doc__
    assert "LLM_BINDING_API_KEY" in bedrock_complete_if_cache.__doc__
    assert "EMBEDDING_BINDING_API_KEY" in bedrock_embed.func.__doc__


class _FakeLightRAG:
    last_init_kwargs = None
    last_instance = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = dict(kwargs)
        type(self).last_instance = self
        self.role_config_snapshot = {}
        self.queue_status_snapshot = {}

    def register_role_llm_builder(self, _builder) -> None:
        return None

    def set_role_llm_metadata(self, _role: str, **_metadata) -> None:
        return None

    def get_llm_role_config(self):
        return self.role_config_snapshot

    async def get_llm_queue_status(self, include_base=True):
        return self.queue_status_snapshot


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _make_args(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        host="127.0.0.1",
        port=9621,
        log_level="INFO",
        verbose=False,
        cors_origins="*",
        whitelist_paths="/health,/api/*",
        auth_accounts="",
        token_secret=None,
        token_expire_hours=48,
        guest_token_expire_hours=24,
        jwt_algorithm="HS256",
        token_auto_renew=True,
        token_renew_threshold=0.5,
        llm_binding="bedrock",
        embedding_binding="bedrock",
        llm_binding_host="DEFAULT_BEDROCK_ENDPOINT",
        embedding_binding_host="DEFAULT_BEDROCK_ENDPOINT",
        ssl=False,
        ssl_certfile=None,
        ssl_keyfile=None,
        key=None,
        input_dir=str(tmp_path / "inputs"),
        workspace="",
        working_dir=str(tmp_path / "rag_storage"),
        llm_binding_api_key=None,
        embedding_binding_api_key="",
        aws_region="us-east-1",
        aws_access_key_id="global-akid",
        aws_secret_access_key="global-secret",
        aws_session_token="global-session",
        query_aws_region=None,
        query_aws_access_key_id=None,
        query_aws_secret_access_key=None,
        query_aws_session_token=None,
        llm_model="us.amazon.nova-lite-v1:0",
        embedding_model=None,
        embedding_dim=None,
        embedding_send_dim=False,
        embedding_token_limit=None,
        max_async=4,
        summary_max_tokens=512,
        summary_context_size=4096,
        force_llm_summary_on_merge=8,
        chunk_size=1200,
        chunk_overlap_size=100,
        kv_storage="JsonKVStorage",
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        doc_status_storage="JsonDocStatusStorage",
        cosine_threshold=0.2,
        enable_llm_cache_for_extract=True,
        enable_llm_cache=True,
        max_parallel_insert=2,
        max_graph_nodes=1000,
        simulated_model_name="lightrag",
        simulated_model_tag="latest",
        summary_language="English",
        rerank_binding="null",
        rerank_model=None,
        rerank_binding_host=None,
        rerank_binding_api_key=None,
        embedding_func_max_async=8,
        embedding_batch_num=10,
        min_rerank_score=0.0,
        related_chunk_number=5,
        top_k=10,
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_create_app_query_role_uses_bedrock_binding(tmp_path, monkeypatch):
    _reload_api_modules_if_mocked()
    monkeypatch.setattr(sys, "argv", ["pytest"])
    config = importlib.import_module("lightrag.api.config")
    config.initialize_config(_make_args(tmp_path), force=True)
    lightrag_server = importlib.import_module("lightrag.api.lightrag_server")
    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)

    args = _make_args(tmp_path)

    with (
        patch(
            "lightrag.llm.bedrock.bedrock_complete_if_cache",
            AsyncMock(return_value="bedrock-ok"),
        ) as mocked_bedrock,
        patch(
            "lightrag.llm.openai.openai_complete_if_cache",
            AsyncMock(side_effect=AssertionError("OpenAI fallback should not be used")),
        ) as mocked_openai,
    ):
        lightrag_server.create_app(args)
        query_func = _FakeLightRAG.last_init_kwargs["role_llm_configs"]["query"].func
        result = await query_func("hello")

    assert result == "bedrock-ok"
    assert mocked_openai.await_count == 0
    assert mocked_bedrock.await_count == 1
    assert mocked_bedrock.await_args.args[:2] == ("us.amazon.nova-lite-v1:0", "hello")
    assert "api_key" not in mocked_bedrock.await_args.kwargs
    assert (
        mocked_bedrock.await_args.kwargs["endpoint_url"] == "DEFAULT_BEDROCK_ENDPOINT"
    )
    assert mocked_bedrock.await_args.kwargs["aws_region"] == "us-east-1"
    assert mocked_bedrock.await_args.kwargs["aws_access_key_id"] == "global-akid"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_create_app_bedrock_query_role_uses_role_sigv4_credentials(
    tmp_path, monkeypatch
):
    _reload_api_modules_if_mocked()
    monkeypatch.setattr(sys, "argv", ["pytest"])
    config = importlib.import_module("lightrag.api.config")
    config.initialize_config(_make_args(tmp_path), force=True)
    lightrag_server = importlib.import_module("lightrag.api.lightrag_server")
    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)

    args = _make_args(tmp_path)
    args.query_aws_region = "us-west-2"
    args.query_aws_access_key_id = "query-akid"
    args.query_aws_secret_access_key = "query-secret"
    args.query_aws_session_token = "query-session"

    with patch(
        "lightrag.llm.bedrock.bedrock_complete_if_cache",
        AsyncMock(return_value="bedrock-ok"),
    ) as mocked_bedrock:
        lightrag_server.create_app(args)
        query_func = _FakeLightRAG.last_init_kwargs["role_llm_configs"]["query"].func
        await query_func("hello")

    assert mocked_bedrock.await_args.kwargs["aws_region"] == "us-west-2"
    assert mocked_bedrock.await_args.kwargs["aws_access_key_id"] == "query-akid"
    assert mocked_bedrock.await_args.kwargs["aws_secret_access_key"] == "query-secret"
    assert mocked_bedrock.await_args.kwargs["aws_session_token"] == "query-session"


@pytest.mark.offline
def test_create_app_rejects_bedrock_role_api_key(tmp_path, monkeypatch):
    _reload_api_modules_if_mocked()
    monkeypatch.setattr(sys, "argv", ["pytest"])
    config = importlib.import_module("lightrag.api.config")
    config.initialize_config(_make_args(tmp_path), force=True)
    lightrag_server = importlib.import_module("lightrag.api.lightrag_server")
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))

    args = _make_args(tmp_path)
    args.query_llm_binding_api_key = "absk-role"

    with pytest.raises(ValueError, match="does not support role-specific"):
        lightrag_server.create_app(args)


@pytest.mark.offline
def test_health_role_llm_config_uses_runtime_snapshot(tmp_path, monkeypatch):
    _reload_api_modules_if_mocked()
    monkeypatch.setattr(sys, "argv", ["pytest"])
    config = importlib.import_module("lightrag.api.config")
    config.initialize_config(_make_args(tmp_path), force=True)
    lightrag_server = importlib.import_module("lightrag.api.lightrag_server")
    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_args, **_kwargs: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)
    monkeypatch.setattr(
        lightrag_server,
        "get_namespace_data",
        AsyncMock(return_value={"busy": False}),
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )

    app = lightrag_server.create_app(_make_args(tmp_path))
    _FakeLightRAG.last_instance.role_config_snapshot = {
        "query": {
            "binding": "runtime-binding",
            "model": "runtime-model",
            "host": "https://runtime.example/v1",
            "max_async": 9,
            "metadata": {"binding": "runtime-binding"},
        }
    }
    _FakeLightRAG.last_instance.queue_status_snapshot = {
        "query": {"available": True, "rejected_total": 2}
    }

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    role_cfg = body["configuration"]["role_llm_config"]["query"]
    assert role_cfg["binding"] == "runtime-binding"
    assert role_cfg["model"] == "runtime-model"
    assert role_cfg["host"] == "https://runtime.example/v1"
    assert role_cfg["max_async"] == 9
    assert role_cfg["model"] != "us.amazon.nova-lite-v1:0"
    assert body["llm_queue_status"]["query"]["rejected_total"] == 2
