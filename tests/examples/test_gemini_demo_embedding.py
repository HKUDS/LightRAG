"""Exercise the demo's decorated embedding callback without a provider request."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.utils import wrap_embedding_func_with_attrs

pytestmark = pytest.mark.offline


async def test_gemini_demo_forwards_declared_embedding_dimension():
    path = Path(__file__).resolve().parents[2] / "examples/lightrag_gemini_demo.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    callback = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "embedding_func"
    )
    vectors = np.zeros((1, 768), dtype=np.float32)
    provider = AsyncMock(return_value=vectors)
    namespace = {
        "np": np,
        "wrap_embedding_func_with_attrs": wrap_embedding_func_with_attrs,
        "gemini_embed": SimpleNamespace(func=provider),
        "GEMINI_API_KEY": "test-key",
    }
    # Loading only this definition avoids the demo's key check and mkdir side effects.
    exec(
        compile(ast.Module(body=[callback], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    result = await namespace["embedding_func"](["sample"])
    assert result is vectors
    provider.assert_awaited_once_with(
        ["sample"],
        api_key="test-key",
        model="models/text-embedding-004",
        embedding_dim=768,
    )
