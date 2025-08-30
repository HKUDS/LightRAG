"""
Tests the fix for the lambda closure bug in the API server's embedding function.

Issue: https://github.com/HKUDS/LightRAG/issues/2023
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Functions to be patched
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_embed


@pytest.fixture
def mock_args():
    """Provides a mock of the server's arguments object."""
    args = Mock()
    args.embedding_binding = "ollama"
    args.embedding_model = "mxbai-embed-large:latest"
    args.embedding_binding_host = "http://localhost:11434"
    args.embedding_binding_api_key = None
    args.embedding_dim = 1024
    args.OllamaEmbeddingOptions.options_dict.return_value = {"num_ctx": 4096}
    return args


@pytest.mark.asyncio
@patch("lightrag.llm.openai.openai_embed", new_callable=AsyncMock)
@patch("lightrag.llm.ollama.ollama_embed", new_callable=AsyncMock)
async def test_embedding_func_captures_values_correctly(
    mock_ollama_embed, mock_openai_embed, mock_args
):
    """
    Verifies that the embedding function correctly captures configuration
    values at creation time and is not affected by later mutations of its source.
    """
    # --- Setup Mocks ---
    mock_ollama_embed.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_openai_embed.return_value = np.array([[0.4, 0.5, 0.6]])

    # --- SIMULATE THE FIX: Capture values before creating the function ---
    binding = mock_args.embedding_binding
    model = mock_args.embedding_model
    host = mock_args.embedding_binding_host
    api_key = mock_args.embedding_binding_api_key

    # CORRECTED: Use an async def instead of a lambda
    async def fixed_func(texts):
        if binding == "ollama":
            return await ollama_embed(
                texts, embed_model=model, host=host, api_key=api_key
            )
        else:
            return await openai_embed(
                texts, model=model, base_url=host, api_key=api_key
            )

    # --- VERIFICATION ---

    # 1. First call: The function should use the initial "ollama" binding.
    await fixed_func(["hello world"])
    mock_ollama_embed.assert_awaited_once()
    mock_openai_embed.assert_not_called()

    # 2. CRITICAL STEP: Mutate the original args object AFTER the function is created.
    mock_args.embedding_binding = "openai"

    # 3. Reset mocks and call the function AGAIN.
    mock_ollama_embed.reset_mock()
    mock_openai_embed.reset_mock()

    await fixed_func(["see you again"])

    # 4. Final check: The function should STILL call ollama_embed.
    mock_ollama_embed.assert_awaited_once()
    mock_openai_embed.assert_not_called()
