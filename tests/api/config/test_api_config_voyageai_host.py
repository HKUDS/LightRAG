"""create_optimized_embedding_function(): the voyageai binding forwards
EMBEDDING_BINDING_HOST to voyageai_embed as base_url."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_voyageai_binding_forwards_host_as_base_url():
    with patch.object(sys, "argv", ["lightrag-server"]):
        from lightrag.api.lightrag_server import create_optimized_embedding_function

    args = SimpleNamespace(
        embedding_dim=None,
        embedding_token_limit=None,
        embedding_asymmetric=False,
        embedding_asymmetric_configured=False,
        embedding_query_prefix_configured=False,
        embedding_document_prefix_configured=False,
    )
    embedding_func = create_optimized_embedding_function(
        config_cache=MagicMock(ollama_embedding_options=None),
        binding="voyageai",
        model=None,
        host="https://voyage-gw.example/v1",
        api_key="test-key",
        args=args,
    )

    async def fake_embed(**kwargs):
        return MagicMock(embeddings=[[0.0] * 1024 for _ in kwargs["texts"]])

    fake_client = MagicMock()
    fake_client.embed = fake_embed
    with patch(
        "lightrag.llm.voyageai.voyageai.AsyncClient", return_value=fake_client
    ) as client_cls:
        out = await embedding_func(["hello"])

    assert isinstance(out, np.ndarray)
    assert client_cls.call_args.kwargs["base_url"] == "https://voyage-gw.example/v1"
    assert client_cls.call_args.kwargs["api_key"] == "test-key"
