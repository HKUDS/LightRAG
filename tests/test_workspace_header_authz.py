"""Regression test for cross-workspace access via LIGHTRAG-WORKSPACE."""

import argparse
import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.api.config import initialize_config

pytestmark = pytest.mark.offline


def _make_args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1", port=9621, working_dir=str(tmp_path / "rag_storage"),
        input_dir=str(tmp_path / "inputs"), timeout=30, max_async=4,
        summary_max_tokens=128, summary_context_size=1024,
        summary_length_recommended=128, log_level="ERROR", verbose=False, key=None,
        ssl=False, ssl_certfile=None, ssl_keyfile=None,
        simulated_model_name="lightrag", simulated_model_tag="latest",
        workspace="production", workers=1, llm_binding="openai",
        llm_binding_host="http://localhost", llm_model="dummy",
        llm_binding_api_key="dummy", embedding_binding="openai",
        embedding_binding_host="http://localhost", embedding_model="dummy",
        embedding_dim=8, embedding_binding_api_key="dummy",
        embedding_token_limit=8192, embedding_func_max_async=4,
        embedding_batch_num=1, embedding_send_dim=False,
        embedding_asymmetric=False, embedding_asymmetric_configured=False,
        embedding_query_prefix=None, embedding_document_prefix=None,
        embedding_query_prefix_configured=False,
        embedding_document_prefix_configured=False, rerank_binding="null",
        rerank_binding_host=None, rerank_binding_api_key=None, rerank_model=None,
        kv_storage="JsonKVStorage", graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        doc_status_storage="JsonDocStatusStorage",
        enable_llm_cache_for_extract=True, enable_llm_cache=True,
        max_parallel_insert=1, max_graph_nodes=1000, summary_language="English",
        entity_types=["organization", "person"], cosine_threshold=0.2,
        top_k=10, related_chunk_number=5, min_rerank_score=0.0,
        chunk_size=1200, chunk_overlap_size=100, cors_origins="*",
        whitelist_paths="/health,/api/*", auth_accounts="", token_secret=None,
        token_expire_hours=48, guest_token_expire_hours=24,
        jwt_algorithm="HS256", token_auto_renew=True, token_renew_threshold=0.5,
        docling=False, document_loading_engine="DEFAULT",
        pdf_decrypt_password=None, force_llm_summary_on_merge=0,
    )


def test_health_rejects_cross_workspace_header(tmp_path):
    args = _make_args(tmp_path)
    os.makedirs(args.input_dir, exist_ok=True)
    initialize_config(args, force=True)

    with patch(
        "lightrag.api.lightrag_server.check_frontend_build",
        return_value=(False, False),
    ), patch("lightrag.api.lightrag_server.LightRAG") as mock_rag_cls:
        rag = mock_rag_cls.return_value
        rag.initialize_storages = AsyncMock()
        rag.check_and_migrate_data = AsyncMock()
        rag.finalize_storages = AsyncMock()
        rag.workspace = args.workspace

        from lightrag.api.lightrag_server import create_app
        from lightrag.kg.shared_storage import (
            finalize_share_data,
            get_namespace_data,
            initialize_pipeline_status,
            initialize_share_data,
        )

        initialize_share_data(workers=1)
        try:
            asyncio.run(initialize_pipeline_status(workspace=args.workspace))
            asyncio.run(initialize_pipeline_status(workspace="secret_internal"))
            secret_status = asyncio.run(
                get_namespace_data("pipeline_status", workspace="secret_internal")
            )
            secret_status["busy"] = True

            client = TestClient(create_app(args))

            direct = client.get(
                "/health", headers={"LIGHTRAG-WORKSPACE": "secret_internal"}
            )
            assert direct.status_code == 403

            sanitized_variant = client.get(
                "/health", headers={"LIGHTRAG-WORKSPACE": "secret-internal"}
            )
            assert sanitized_variant.status_code == 403

            allowed = client.get(
                "/health", headers={"LIGHTRAG-WORKSPACE": args.workspace}
            )
            assert allowed.status_code == 200
            assert allowed.json()["pipeline_busy"] is False
        finally:
            finalize_share_data()
