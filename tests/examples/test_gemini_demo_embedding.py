"""Exercise the Gemini demos' decorated embedding callbacks without a provider request."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

pytestmark = pytest.mark.offline

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

GEMINI_DEMOS = [
    "lightrag_gemini_demo.py",
    "lightrag_gemini_postgres_demo.py",
    "lightrag_gemini_workspace_demo.py",
]


def _load_embedding_func(path: Path, provider: AsyncMock) -> EmbeddingFunc:
    """Execute only the demo's ``embedding_func`` definition.

    Importing a demo would validate the API key, create directories, apply
    ``nest_asyncio`` or configure logging, so the definition is compiled on its
    own against a namespace that stubs the Gemini binding.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    callback = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "embedding_func"
    )
    namespace = {
        "np": np,
        "os": os,
        "wrap_embedding_func_with_attrs": wrap_embedding_func_with_attrs,
        "gemini_embed": SimpleNamespace(func=provider),
        "GEMINI_API_KEY": "test-key",
    }
    exec(  # noqa: S102 - trusted repository source, see the docstring
        compile(ast.Module(body=[callback], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["embedding_func"]


@pytest.mark.parametrize("demo", GEMINI_DEMOS)
async def test_gemini_demo_forwards_declared_embedding_dimension(demo, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = AsyncMock()
    embedding_func = _load_embedding_func(EXAMPLES_DIR / demo, provider)
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
