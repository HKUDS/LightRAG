"""Bedrock dimension settings must reach the real SDK request boundary."""

import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web

from lightrag.llm.bedrock import bedrock_embed

pytestmark = pytest.mark.offline


@pytest.fixture
async def bedrock_endpoint(
    unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> AsyncIterator[tuple[str, list[dict]]]:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    requests: list[dict] = []

    async def invoke(request: web.Request) -> web.Response:
        body = await request.json()
        model = request.match_info["model"]
        requests.append(
            {
                "body": body,
                "session_token": request.headers.get("X-Amz-Security-Token"),
            }
        )
        if model.startswith("cohere."):
            response = {"embeddings": [[0.1] * 1024 for _ in body["texts"]]}
        else:
            dimension = 1536 if "v1" in model else body.get("dimensions", 1024)
            vector = [0.1] * dimension
            response = {"embedding": vector, "inputTextTokenCount": 1}
            if "v2" in model:
                response["embeddingsByType"] = {"float": vector}
        return web.json_response(response)

    app = web.Application()
    app.router.add_post("/model/{model}/invoke", invoke)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
        yield f"http://127.0.0.1:{unused_tcp_port}", requests
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    ("model", "dimension", "send_dimensions"),
    [
        ("amazon.titan-embed-text-v2:0", 256, True),
        ("amazon.titan-embed-text-v2:0", 512, True),
        ("amazon.titan-embed-text-v2:0", 1024, True),
        ("amazon.titan-embed-text-v2:0", 1024, False),
        ("amazon.titan-embed-text-v1", 1536, True),
        ("cohere.embed-english-v3", 1024, True),
    ],
)
async def test_server_embedding_dimension(
    bedrock_endpoint, model: str, dimension: int, send_dimensions: bool
):
    with patch.object(sys, "argv", ["lightrag-server"]):
        from lightrag.api.lightrag_server import create_embedding_function_from_args

    endpoint, requests = bedrock_endpoint
    args = SimpleNamespace(
        llm_binding="bedrock",
        embedding_binding="bedrock",
        embedding_model=model,
        embedding_binding_host=endpoint,
        embedding_binding_api_key=None,
        embedding_document_prefix=None,
        embedding_query_prefix=None,
        embedding_send_dim=send_dimensions,
        embedding_dim=dimension,
        embedding_token_limit=None,
        embedding_asymmetric=False,
        embedding_asymmetric_configured=False,
        embedding_query_prefix_configured=False,
        embedding_document_prefix_configured=False,
        aws_region="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_session_token="test-session",
    )
    embedding = create_embedding_function_from_args(args)

    result = await embedding(["a document", "another document"])

    assert result.shape == (2, dimension)
    assert len(requests) == (1 if model.startswith("cohere.") else 2)
    if "v2" in model and send_dimensions:
        assert all(r["body"]["dimensions"] == dimension for r in requests)
    else:
        assert all("dimensions" not in r["body"] for r in requests)
    assert all(r["session_token"] == "test-session" for r in requests)


async def test_bedrock_embed_preserves_positional_auth_arguments(bedrock_endpoint):
    endpoint, requests = bedrock_endpoint

    result = await bedrock_embed.func(
        ["a document"],
        "amazon.titan-embed-text-v2:0",
        "test-key",
        "test-secret",
        "test-session",
        "us-east-1",
        None,
        endpoint,
    )

    assert result.shape == (1, 1024)
    assert requests[0]["session_token"] == "test-session"
    assert "dimensions" not in requests[0]["body"]
