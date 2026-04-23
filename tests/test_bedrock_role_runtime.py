import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import APIRouter


pytestmark = pytest.mark.offline


class _FakeLightRAG:
    last_init_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = dict(kwargs)

    def register_role_llm_builder(self, _builder) -> None:
        return None

    def set_role_llm_metadata(self, _role: str, **_metadata) -> None:
        return None


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
        llm_model="us.amazon.nova-lite-v1:0",
        embedding_model=None,
        embedding_dim=None,
        embedding_send_dim=False,
        embedding_token_limit=None,
        max_async=4,
        summary_max_tokens=512,
        summary_context_size=4096,
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
        top_k=10,
    )


@pytest.mark.asyncio
async def test_create_app_query_role_uses_bedrock_binding(tmp_path, monkeypatch):
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
        query_func = _FakeLightRAG.last_init_kwargs["query_llm_model_func"]
        result = await query_func("hello")

    assert result == "bedrock-ok"
    assert mocked_openai.await_count == 0
    assert mocked_bedrock.await_count == 1
    assert mocked_bedrock.await_args.args[:2] == ("us.amazon.nova-lite-v1:0", "hello")
    assert (
        mocked_bedrock.await_args.kwargs["endpoint_url"] == "DEFAULT_BEDROCK_ENDPOINT"
    )
