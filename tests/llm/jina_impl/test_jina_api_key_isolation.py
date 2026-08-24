"""Regression tests for request-local Jina API credentials."""

import base64
import os
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.llm.jina import jina_embed


def _encoded_embedding(values=(0.1, 0.2, 0.3)) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.float32).tobytes()).decode()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_explicit_api_key_is_request_local(monkeypatch):
    """An explicit Jina key must not replace the process-wide default key."""
    monkeypatch.setenv("JINA_API_KEY", "shared-jina-key")
    fetch = AsyncMock(return_value=[{"embedding": _encoded_embedding()}])

    with patch("lightrag.llm.jina.fetch_data", fetch):
        result = await jina_embed.func.__wrapped__(
            texts=["hello"], api_key="request-jina-key", embedding_dim=3
        )

    _, headers, _ = fetch.await_args.args
    assert headers["Authorization"] == "Bearer request-jina-key"
    assert os.environ["JINA_API_KEY"] == "shared-jina-key"
    np.testing.assert_allclose(result, [[0.1, 0.2, 0.3]])
