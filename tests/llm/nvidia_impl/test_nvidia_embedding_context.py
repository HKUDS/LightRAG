from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_nvidia_client():
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.0] * 2048)]

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.embeddings.create = AsyncMock(return_value=response)

    with patch(
        "lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=client
    ) as client_cls:
        yield client, client_cls


@pytest.mark.asyncio
async def test_nvidia_embed_uses_query_input_type(mock_nvidia_client):
    client, _ = mock_nvidia_client

    from lightrag.llm.nvidia_openai import nvidia_openai_embed

    await nvidia_openai_embed(["question"], context="query")

    assert client.embeddings.create.await_args.kwargs["extra_body"]["input_type"] == (
        "query"
    )


@pytest.mark.asyncio
async def test_nvidia_embed_uses_passage_input_type_for_documents(mock_nvidia_client):
    client, _ = mock_nvidia_client

    from lightrag.llm.nvidia_openai import nvidia_openai_embed

    await nvidia_openai_embed(["document"], context="document")

    assert client.embeddings.create.await_args.kwargs["extra_body"]["input_type"] == (
        "passage"
    )


@pytest.mark.asyncio
async def test_nvidia_embed_preserves_explicit_input_type(mock_nvidia_client):
    client, _ = mock_nvidia_client

    from lightrag.llm.nvidia_openai import nvidia_openai_embed

    await nvidia_openai_embed(["question"], context="query", input_type="passage")

    assert client.embeddings.create.await_args.kwargs["extra_body"]["input_type"] == (
        "passage"
    )


def test_nvidia_embed_declares_asymmetric_support():
    from lightrag.llm.nvidia_openai import nvidia_openai_embed

    assert nvidia_openai_embed.supports_asymmetric is True
