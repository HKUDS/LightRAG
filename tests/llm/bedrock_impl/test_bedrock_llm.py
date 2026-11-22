import importlib
from functools import partial
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.llm.bedrock import (
    bedrock_complete,
    bedrock_complete_if_cache,
    bedrock_embed,
)
from lightrag.llm_roles import ROLES

_ROLE_ATTR_SUFFIXES = (
    "llm_binding",
    "llm_model",
    "llm_binding_host",
    "llm_binding_api_key",
    "llm_max_async",
    "llm_timeout",
    "aws_region",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
)

_API_ENV_VARS_TO_ISOLATE = (
    "AUTH_ACCOUNTS",
    "LIGHTRAG_API_KEY",
    "TOKEN_SECRET",
)


@pytest.fixture(autouse=True)
def _isolate_api_auth_env(monkeypatch):
    """Keep API app tests independent from developer-local .env auth settings."""
    for var in _API_ENV_VARS_TO_ISOLATE:
        monkeypatch.setenv(var, "")


def _reload_api_modules_if_mocked() -> None:
    """Drop cached lightrag.api entries so importlib reloads with isolated env.

    Other test files (e.g. test_token_auto_renewal.py) replace
    ``sys.modules["lightrag.api.config"]`` with a Mock at import time. When
    pytest collects those files before ours, any subsequent
    ``from .config import global_args`` inside lightrag_server picks up the
    Mock, which breaks ``create_app`` in create_app_* tests below.
    """
    for modname in (
        "lightrag.api.lightrag_server",
        "lightrag.api.utils_api",
        "lightrag.api.auth",
        "lightrag.api.config",
    ):
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


class _FakeInterleavedClient(_FakeBedrockClient):
    async def converse(self, **kwargs):
        self._captured_calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "first thought"}
                            }
                        },
                        {"text": "first part. "},
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "second thought"}
                            }
                        },
                        {"text": "second part."},
                    ]
                }
            }
        }


class _FakeInterleavedSession(_FakeSession):
    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeInterleavedClient(self._captured_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_complete_joins_interleaved_text_blocks(monkeypatch):
    """Extended thinking can interleave reasoning and text blocks as
    [reasoning, text, reasoning, text]; every text block must be joined,
    not just the first one found, and every reasoning block skipped.
    """
    monkeypatch.delenv("AWS_REGION", raising=False)
    captured_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeInterleavedSession(captured_calls, []),
    ):
        result = await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            extra_fields={"reasoning_config": {"type": "enabled"}},
        )

    assert result == "first part. second part."
    assert captured_calls[-1]["additionalModelRequestFields"] == {
        "reasoning_config": {"type": "enabled"}
    }


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
@pytest.mark.parametrize(
    "endpoint_url",
    [
        pytest.param("DEFAULT_BEDROCK_ENDPOINT", id="sentinel"),
        pytest.param("", id="empty"),
    ],
)
async def test_bedrock_default_endpoint_sentinel_uses_sdk_default(
    monkeypatch, endpoint_url
):
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
            endpoint_url=endpoint_url,
        )

    assert client_kwargs_calls[-1] == {"region_name": None}


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


class _FakeStreamingBedrockClient(_FakeBedrockClient):
    async def converse_stream(self, **kwargs):
        self._captured_calls.append(kwargs)

        async def _events():
            yield {"contentBlockDelta": {"delta": {"text": "chunk"}}}
            yield {"messageStop": {}}

        return {"stream": _events()}


class _FakeStreamingSession(_FakeSession):
    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeStreamingBedrockClient(self._captured_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_stream_timeout_maps_to_botocore_client_config(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeStreamingSession(captured_calls, client_kwargs_calls),
    ):
        stream = await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            stream=True,
            timeout=240,
        )
        chunks = [chunk async for chunk in stream]

    assert chunks == ["chunk"]
    config = client_kwargs_calls[-1]["config"]
    assert config.connect_timeout == 240
    assert config.read_timeout == 240
    assert "timeout" not in captured_calls[-1]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_no_timeout_keeps_botocore_defaults(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession([], client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
        )

    assert "config" not in client_kwargs_calls[-1]


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("default_llm_timeout", "call_kwargs", "expected_read_timeout"),
    [
        pytest.param(240, {}, 240, id="fallback-to-default-llm-timeout"),
        # A caller-passed timeout (llm_model_kwargs, role kwargs) still wins.
        pytest.param(240, {"timeout": 30}, 30, id="caller-timeout-wins"),
        # LLM_TIMEOUT=0 means the SDK default: botocore's own 60s, never an
        # unbounded wait.
        pytest.param(0, {}, None, id="zero-means-botocore-default"),
        # LLM_TIMEOUT is the client timeout even below botocore's 60s: role
        # timeouts now reach the client, so nothing needs that floor.
        pytest.param(30, {}, 30, id="short-default-applied"),
    ],
)
async def test_bedrock_complete_inherits_default_llm_timeout(
    monkeypatch, default_llm_timeout, call_kwargs, expected_read_timeout
):
    """A direct bedrock_complete call must apply LightRAG's LLM timeout.

    Through a role queue the role wrapper passes the timeout; a direct call
    that passes none would otherwise keep botocore's 60s read timeout, which
    cut any attempt longer than that (GPT-6 Astra extraction on a dense
    chunk).
    """
    monkeypatch.delenv("AWS_REGION", raising=False)
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    hashing_kv = SimpleNamespace(
        global_config={
            "llm_model_name": "bedrock-model",
            "default_llm_timeout": default_llm_timeout,
        }
    )

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete(prompt="hello", hashing_kv=hashing_kv, **call_kwargs)

    assert len(client_kwargs_calls) == 1
    # timeout is not a Converse API parameter and must never leak into the call.
    assert "timeout" not in captured_calls[-1]
    config = client_kwargs_calls[0].get("config")
    if expected_read_timeout is None:
        assert config is None
    else:
        assert config.read_timeout == expected_read_timeout
        assert config.connect_timeout == expected_read_timeout


async def _custom_bedrock_wrapper(prompt, **kwargs):
    return await bedrock_complete(prompt, aws_region="us-east-1", **kwargs)


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_model_func",
    [
        pytest.param(bedrock_complete, id="bedrock_complete"),
        pytest.param(partial(bedrock_complete, aws_region="us-east-1"), id="partial"),
        pytest.param(_custom_bedrock_wrapper, id="custom-wrapper"),
    ],
)
async def test_bedrock_complete_uses_role_timeout_through_role_wrapper(
    monkeypatch, tmp_path, llm_model_func
):
    """Each role's effective timeout must size its Bedrock client.

    An extract role at timeout=180 reaches the client as 180 rather than the
    base 30s or botocore's 60s, also when bedrock_complete sits behind a
    partial or a custom function; a role without its own timeout gets the
    base default_llm_timeout its queue is sized from.
    """
    import numpy as np

    from lightrag import LightRAG, RoleLLMConfig
    from lightrag.utils import EmbeddingFunc

    monkeypatch.delenv("AWS_REGION", raising=False)

    async def _embed(texts):
        return np.zeros((len(texts), 8))

    rag = LightRAG(
        working_dir=str(tmp_path / "bedrock-role-timeout"),
        llm_model_func=llm_model_func,
        llm_model_name="bedrock-model",
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed),
        default_llm_timeout=30,
        role_llm_configs={"extract": RoleLLMConfig(timeout=180)},
    )
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession([], client_kwargs_calls),
    ):
        await rag.role_llm_funcs["extract"]("hello")
        await rag.role_llm_funcs["query"]("hello")

    assert len(client_kwargs_calls) == 2
    assert client_kwargs_calls[0]["config"].read_timeout == 180
    assert client_kwargs_calls[0]["config"].connect_timeout == 180
    assert client_kwargs_calls[1]["config"].read_timeout == 30


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


class _FakeLightRAG:
    last_init_kwargs = None
    last_instance = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = dict(kwargs)
        type(self).last_instance = self
        self.role_config_snapshot = {}
        for role, cfg in (kwargs.get("role_llm_configs") or {}).items():
            metadata = dict(getattr(cfg, "metadata", None) or {})
            self.role_config_snapshot[role] = {
                "binding": metadata.get("binding"),
                "model": metadata.get("model"),
                "host": metadata.get("host"),
                "is_cross_provider": metadata.get("is_cross_provider", False),
                "max_async": getattr(cfg, "max_async", None),
                "timeout": getattr(cfg, "timeout", None),
                "has_model_kwargs": getattr(cfg, "kwargs", None) is not None,
                "metadata": metadata,
            }
        self.queue_status_snapshot = {}
        self.embedding_queue_status_snapshot = {}
        self.rerank_queue_status_snapshot = {}

    def register_role_llm_builder(self, _builder) -> None:
        return None

    def set_role_llm_metadata(self, _role: str, **_metadata) -> None:
        return None

    def get_llm_role_config(self):
        return self.role_config_snapshot

    async def get_llm_queue_status(self, include_base=True):
        return self.queue_status_snapshot

    async def get_embedding_queue_status(self):
        return self.embedding_queue_status_snapshot

    async def get_rerank_queue_status(self):
        return self.rerank_queue_status_snapshot


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _make_args(tmp_path):
    """Server args for the ``create_app`` tests below.

    Derived from the REAL parser and then overridden, NOT hand-rolled: every
    server-consumed config knob added later (LR2 Phase 2's
    ``pipeline_scheduling_page_size`` broke all seven ``create_app`` tests here
    with an ``AttributeError``) exists automatically. ``sys.argv`` is pinned so
    the parse never sees pytest's own arguments, and the auth/env-sensitive
    fields stay pinned below so a developer ``.env`` cannot reach the app.
    """
    original_argv = sys.argv[:]
    sys.argv = ["lightrag-server"]
    try:
        from lightrag.api.config import parse_args

        args = parse_args()
    finally:
        sys.argv = original_argv

    # parse_args() reads per-role LLM env vars (e.g. QUERY_LLM_MODEL) straight
    # from a developer's local .env. Clear them all so these tests exercise
    # only the base llm_binding/llm_model set below, never leaking whatever
    # role overrides happen to be configured on the machine running pytest.
    for spec in ROLES:
        for suffix in _ROLE_ATTR_SUFFIXES:
            setattr(args, f"{spec.name}_{suffix}", None)

    overrides = dict(
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
        # The server refuses to start without a named embedding model, so these
        # tests -- which are about the LLM side -- have to name one. It is the
        # Bedrock binding's OWN default, which carries a known dimension, so
        # they still say nothing about embedding configuration.
        embedding_model="amazon.titan-embed-text-v2:0",
        embedding_dim=None,
        embedding_send_dim=False,
        embedding_token_limit=None,
        embedding_document_prefix=None,
        embedding_document_prefix_configured=False,
        embedding_query_prefix=None,
        embedding_query_prefix_configured=False,
        embedding_prefix_no_prefix_sentinel="NO_PREFIX",
        embedding_prefixes_configured=False,
        embedding_asymmetric=False,
        embedding_asymmetric_configured=False,
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
        vlm_process_enable=False,
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
        llm_timeout=180,
        embedding_timeout=30,
        rerank_max_async=4,
        rerank_timeout=30,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _setup_bedrock_app_modules(monkeypatch, args):
    """Prepare an isolated lightrag_server module for create_app tests."""
    _reload_api_modules_if_mocked()
    monkeypatch.setattr(sys, "argv", ["pytest"])
    config = importlib.import_module("lightrag.api.config")
    config.initialize_config(args, force=True)
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
    return lightrag_server


@pytest.mark.offline
@pytest.mark.asyncio
async def test_create_app_query_role_uses_bedrock_binding(tmp_path, monkeypatch):
    args = _make_args(tmp_path)
    lightrag_server = _setup_bedrock_app_modules(monkeypatch, args)

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
        query_cfg = _FakeLightRAG.last_init_kwargs["role_llm_configs"]["query"]
        query_func = query_cfg.func
        result = await query_func("hello")

    assert query_cfg.metadata["binding"] == "bedrock"
    assert query_cfg.metadata["model"] == "us.amazon.nova-lite-v1:0"
    assert query_cfg.metadata["host"] == "DEFAULT_BEDROCK_ENDPOINT"
    assert query_cfg.metadata["api_key"] is None
    assert query_cfg.metadata["bedrock_aws_options"]["aws_region"] == "us-east-1"
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
    args = _make_args(tmp_path)
    args.query_aws_region = "us-west-2"
    args.query_aws_access_key_id = "query-akid"
    args.query_aws_secret_access_key = "query-secret"
    args.query_aws_session_token = "query-session"
    lightrag_server = _setup_bedrock_app_modules(monkeypatch, args)

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
@pytest.mark.asyncio
async def test_create_app_bedrock_query_role_uses_role_specific_timeout(
    tmp_path, monkeypatch
):
    args = _make_args(tmp_path)
    args.query_llm_timeout = 99
    lightrag_server = _setup_bedrock_app_modules(monkeypatch, args)

    with patch(
        "lightrag.llm.bedrock.bedrock_complete_if_cache",
        AsyncMock(return_value="bedrock-ok"),
    ) as mocked_bedrock:
        lightrag_server.create_app(args)
        query_func = _FakeLightRAG.last_init_kwargs["role_llm_configs"]["query"].func
        await query_func("hello")

    assert mocked_bedrock.await_args.kwargs["timeout"] == 99


@pytest.mark.offline
@pytest.mark.asyncio
async def test_create_app_bedrock_server_timeout_overrides_caller_value(
    tmp_path, monkeypatch
):
    """Caller-passed timeout must not raise a duplicate-keyword TypeError and is
    overridden by the server-configured value, matching the OpenAI wrappers."""
    args = _make_args(tmp_path)
    lightrag_server = _setup_bedrock_app_modules(monkeypatch, args)

    with patch(
        "lightrag.llm.bedrock.bedrock_complete_if_cache",
        AsyncMock(return_value="bedrock-ok"),
    ) as mocked_bedrock:
        lightrag_server.create_app(args)
        base_func = _FakeLightRAG.last_init_kwargs["llm_model_func"]
        query_func = _FakeLightRAG.last_init_kwargs["role_llm_configs"]["query"].func

        await base_func("hello", timeout=5)
        assert mocked_bedrock.await_args.kwargs["timeout"] == 180

        await query_func("hello", timeout=5)
        assert mocked_bedrock.await_args.kwargs["timeout"] == 180


@pytest.mark.offline
@pytest.mark.asyncio
async def test_create_app_keyword_openai_role_forwards_nested_extra_body(
    tmp_path, monkeypatch, caplog, lightrag_caplog
):
    monkeypatch.setenv(
        "KEYWORD_OPENAI_LLM_EXTRA_BODY",
        '{"chat_template_kwargs": {"enable_thinking": false}}',
    )
    args = _make_args(tmp_path)
    args.keyword_llm_binding = "openai"
    args.keyword_llm_model = "xhd/Qwen3.5-35B-A3B"
    args.keyword_llm_binding_host = "https://keyword.example/v1"
    args.keyword_llm_binding_api_key = "keyword-secret"
    lightrag_server = _setup_bedrock_app_modules(monkeypatch, args)

    with (
        caplog.at_level("INFO", logger="lightrag"),
        patch(
            "lightrag.llm.openai.openai_complete_if_cache",
            AsyncMock(
                return_value='{"high_level_keywords":[],"low_level_keywords":[]}'
            ),
        ) as mocked_openai,
    ):
        lightrag_server.create_app(args)
        keyword_cfg = _FakeLightRAG.last_init_kwargs["role_llm_configs"]["keyword"]
        result = await keyword_cfg.func(
            "keyword prompt", response_format={"type": "json_object"}
        )

    assert result == '{"high_level_keywords":[],"low_level_keywords":[]}'
    assert keyword_cfg.metadata["binding"] == "openai"
    assert keyword_cfg.metadata["provider_options"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert mocked_openai.await_count == 1
    assert mocked_openai.await_args.args[:2] == (
        "xhd/Qwen3.5-35B-A3B",
        "keyword prompt",
    )
    kwargs = mocked_openai.await_args.kwargs
    assert kwargs["base_url"] == "https://keyword.example/v1"
    assert kwargs["api_key"] == "keyword-secret"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Role LLM Option:" in messages
    assert " - extract: Bedrock {}" in messages
    assert " - keyword: OpenAI {'extra_body':" in messages
    assert " - query: Bedrock {}" in messages
    assert " - vlm: Bedrock {}" in messages
    assert "chat_template_kwargs" in messages
    assert "reasoning_effort" not in messages
    assert "frequency_penalty" not in messages
    assert "keyword-secret" not in messages


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
    args = _make_args(tmp_path)
    lightrag_server = _setup_bedrock_app_modules(monkeypatch, args)
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

    app = lightrag_server.create_app(args)
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
    _FakeLightRAG.last_instance.embedding_queue_status_snapshot = {
        "available": True,
        "running": 1,
    }
    _FakeLightRAG.last_instance.rerank_queue_status_snapshot = {
        "available": False,
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
    assert body["embedding_queue_status"]["running"] == 1
    assert body["rerank_queue_status"]["available"] is False


@pytest.mark.offline
@pytest.mark.parametrize(
    "pipeline_state, expected_active",
    [
        ({"busy": False}, False),
        ({"busy": True}, True),
        ({"busy": False, "scanning": True}, True),
        ({"busy": False, "destructive_busy": True}, True),
        ({"busy": False, "pending_enqueues": 2}, True),
    ],
)
def test_health_pipeline_active_derivation(
    tmp_path, monkeypatch, pipeline_state, expected_active
):
    args = _make_args(tmp_path)
    lightrag_server = _setup_bedrock_app_modules(monkeypatch, args)
    monkeypatch.setattr(
        lightrag_server,
        "get_namespace_data",
        AsyncMock(return_value=pipeline_state),
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )

    app = lightrag_server.create_app(args)
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_busy"] is bool(pipeline_state.get("busy", False))
    assert body["pipeline_scanning"] is bool(pipeline_state.get("scanning", False))
    assert body["pipeline_destructive_busy"] is bool(
        pipeline_state.get("destructive_busy", False)
    )
    assert body["pipeline_pending_enqueues"] == int(
        pipeline_state.get("pending_enqueues", 0)
    )
    assert body["pipeline_active"] is expected_active


class _FakeTruncatedClient(_FakeBedrockClient):
    def __init__(self, captured_calls: list[dict], stop_reason: str):
        super().__init__(captured_calls)
        self._stop_reason = stop_reason

    async def converse(self, **kwargs):
        self._captured_calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": '{"entities":[{"name":"Ali'}]}},
            "stopReason": self._stop_reason,
        }


class _FakeTruncatedSession(_FakeSession):
    def __init__(self, captured_calls, client_kwargs_calls, stop_reason):
        super().__init__(captured_calls, client_kwargs_calls)
        self._stop_reason = stop_reason

    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeTruncatedClient(self._captured_calls, self._stop_reason)


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "truncated"),
    [("max_tokens", True), ("end_turn", False)],
)
async def test_bedrock_stop_reason_sets_truncation_marker(
    monkeypatch, stop_reason, truncated
):
    """Only Converse stopReason == "max_tokens" flags the response uncacheable."""
    from lightrag.utils import is_truncated_response

    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeTruncatedSession([], [], stop_reason),
    ):
        result = await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="Extract entities",
        )

    assert result == '{"entities":[{"name":"Ali'
    assert is_truncated_response(result) is truncated


class _FakeEmptyResponseClient(_FakeBedrockClient):
    """Converse returned a well-formed envelope with no usable text.

    ``stop_reason`` / ``reasoning`` shape the diagnostics the binding is
    expected to attach.
    """

    stop_reason = "max_tokens"
    reasoning = True

    async def converse(self, **kwargs):
        self._captured_calls.append(kwargs)
        content = []
        if self.reasoning:
            content.append(
                {"reasoningContent": {"reasoningText": {"text": "internal thought"}}}
            )
        content.append({"text": ""})
        return {
            "output": {"message": {"content": content}},
            "stopReason": self.stop_reason,
            "usage": {"outputTokens": 64},
        }


def _empty_response_session(captured_calls, *, stop_reason, reasoning):
    client = _FakeEmptyResponseClient(captured_calls)
    client.stop_reason = stop_reason
    client.reasoning = reasoning
    return SimpleNamespace(client=lambda *_args, **_kwargs: client)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_empty_max_tokens_response_names_the_token_limit(monkeypatch):
    """Bedrock already failed on empty content; what was missing is the cause.

    doc_status.error_msg has to say which knob to turn, not just report
    "empty content" (issue #3601 gap 4).
    """
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_empty_response_session(
            [], stop_reason="max_tokens", reasoning=True
        ),
    ):
        with pytest.raises(Exception) as excinfo:
            await bedrock_complete_if_cache.__wrapped__(
                model="bedrock-model", prompt="Extract"
            )

    message = str(excinfo.value)
    assert "Received empty content from Bedrock API" in message
    assert "stopReason=max_tokens" in message
    assert "outputTokens=64" in message
    # Measured from the reasoning TEXT, not the repr of the block.
    assert f"reasoning_content_len={len('internal thought')}" in message
    assert "budget consumed by reasoning" in message
    assert "BEDROCK_LLM_MAX_TOKENS" in message


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_empty_response_without_token_limit_says_so(monkeypatch):
    """A normal stop with no output is a different diagnosis, not the budget."""
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_empty_response_session(
            [], stop_reason="end_turn", reasoning=False
        ),
    ):
        with pytest.raises(Exception) as excinfo:
            await bedrock_complete_if_cache.__wrapped__(
                model="bedrock-model", prompt="Extract"
            )

    message = str(excinfo.value)
    assert "stopReason=end_turn" in message
    assert "model produced no output" in message
    assert "BEDROCK_LLM_MAX_TOKENS" not in message
