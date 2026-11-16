"""Exercise the Gemini demos' decorated embedding callbacks without a provider request."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs
from tests.examples._demo_source import load_demo_definition

pytestmark = pytest.mark.offline

GEMINI_DEMOS = [
    "lightrag_gemini_demo.py",
    "lightrag_gemini_postgres_demo.py",
    "lightrag_gemini_workspace_demo.py",
]


def _load_embedding_func(demo: str, provider: AsyncMock) -> EmbeddingFunc:
    """Load only the demo's ``embedding_func`` against a stubbed Gemini binding."""
    return load_demo_definition(
        demo,
        "embedding_func",
        {
            "np": np,
            "os": os,
            "wrap_embedding_func_with_attrs": wrap_embedding_func_with_attrs,
            "gemini_embed": SimpleNamespace(func=provider),
            "GEMINI_API_KEY": "test-key",
        },
    )


@pytest.mark.parametrize("demo", GEMINI_DEMOS)
async def test_gemini_demo_forwards_declared_embedding_dimension(demo, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = AsyncMock()
    embedding_func = _load_embedding_func(demo, provider)
    vectors = np.zeros((1, embedding_func.embedding_dim), dtype=np.float32)
    provider.return_value = vectors

    result = await embedding_func(["sample"])

    assert result is vectors
    # Without send_dimensions, Gemini returns its default size rather than the
    # declared one and the wrapper's shape check rejects every batch.
    assert embedding_func.send_dimensions is True
    provider.assert_awaited_once_with(
        ["sample"],
        api_key="test-key",
        # The declared model_name keys the vector space and the configuration
        # binding, so it must name the model that actually produced the vectors.
        model=embedding_func.model_name,
        embedding_dim=embedding_func.embedding_dim,
    )
