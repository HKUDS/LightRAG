"""nvidia_openai_embed() request and response handling through the real SDK.

Each test drives the actual ``AsyncOpenAI`` client over an ``httpx``
``MockTransport``, so what is asserted is the JSON that would reach NVIDIA and
the vectors decoded from NVIDIA's JSON, with no network I/O.
"""

import base64
import json

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI

from lightrag.llm import nvidia_openai
from lightrag.llm.nvidia_openai import nvidia_openai_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _row(index, embedding):
    row = {"object": "embedding", "embedding": embedding}
    if index is not None:
        row["index"] = index
    return row


def _mock_sdk(monkeypatch, rows):
    """Route ``AsyncOpenAI`` to a real client that answers with ``rows``."""
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "nvidia/llama-3.2-nv-embedqa-1b-v1",
                "data": rows,
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://nvidia.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(nvidia_openai, "AsyncOpenAI", lambda **kwargs: client)
    return client, requests


@pytest.mark.parametrize(
    ("encode", "response_format"),
    [("float", "float"), ("base64", "base64"), ("base64", "float")],
)
async def test_embedding_encoding_preserves_numeric_vectors(
    monkeypatch, encode, response_format
):
    expected = (np.arange(2 * 2048, dtype=np.float32).reshape(2, 2048) - 1024) / 4096
    client, requests = _mock_sdk(
        monkeypatch,
        [
            _row(
                index,
                base64.b64encode(vector.tobytes()).decode("ascii")
                if response_format == "base64"
                else vector.tolist(),
            )
            for index, vector in enumerate(expected)
        ],
    )
    texts = ["document 0", "document 1"]

    result = await nvidia_openai_embed(texts, encode=encode)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    assert requests[0]["input"] == texts
    assert requests[0]["encoding_format"] == encode
    assert client.is_closed()


async def test_short_base64_vector_fails_dimension_validation(monkeypatch):
    """A base64 vector is decoded as sent, never padded or cut to the declared
    dimension, so a short one reaches the wrapper's dimension check."""
    vector = np.zeros(3, dtype=np.float32)
    client, _ = _mock_sdk(
        monkeypatch, [_row(0, base64.b64encode(vector.tobytes()).decode("ascii"))]
    )

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await nvidia_openai_embed(["document"], encode="base64")
    assert client.is_closed()


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [_row(1, [10.0]), _row(0, [20.0])], [[20.0], [10.0]], id="shuffled"
        ),
        # An endpoint that sends no indices at all keeps its arrival order.
        pytest.param(
            [_row(None, [10.0]), _row(None, [20.0])],
            [[10.0], [20.0]],
            id="unindexed",
        ),
        pytest.param([_row(0, [10.0]), _row(0, [20.0])], None, id="duplicate"),
        pytest.param([_row(1, [10.0]), _row(None, [20.0])], None, id="partial"),
        # Every input position is covered, but one twice: only the row count
        # tells this batch from a valid one.
        pytest.param(
            [_row(0, [10.0]), _row(1, [20.0]), _row(1, [30.0])], None, id="extra-row"
        ),
    ],
)
async def test_embedding_rows_follow_input_indices(monkeypatch, rows, expected):
    client, _ = _mock_sdk(monkeypatch, rows)
    call = nvidia_openai_embed.func(["first", "second"])

    if expected is None:
        with pytest.raises(ValueError, match="invalid embedding indices for 2 texts"):
            await call
    else:
        np.testing.assert_array_equal(await call, expected)
    assert client.is_closed()


@pytest.mark.parametrize(
    ("kwargs", "expected_input_type"),
    [
        pytest.param({"context": "query"}, "query", id="query-context"),
        pytest.param({"context": "document"}, "passage", id="document-context"),
        pytest.param({}, "passage", id="no-context"),
        pytest.param(
            {"context": "query", "input_type": "passage"},
            "passage",
            id="explicit-input-type-wins",
        ),
    ],
)
async def test_input_type_follows_lightrag_context(
    monkeypatch, kwargs, expected_input_type
):
    """NVIDIA's embedqa models embed queries and passages asymmetrically.

    Called through the ``EmbeddingFunc`` wrapper, so the query case also
    proves the wrapper forwards ``context`` at all (``supports_asymmetric``).
    """
    _, requests = _mock_sdk(monkeypatch, [_row(0, [0.0] * 2048)])

    await nvidia_openai_embed(["text"], trunc="END", **kwargs)

    assert requests[0]["input_type"] == expected_input_type
    assert requests[0]["truncate"] == "END"
