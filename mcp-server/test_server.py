import json
from unittest.mock import patch

import pytest

from server import LightRAGAPIClient, _query_payload


def test_query_payload_omits_unset_values():
    payload = _query_payload(
        "What is RAG?",
        "mix",
        "Multiple Paragraphs",
        None,
        5,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        True,
        False,
    )

    assert payload == {
        "query": "What is RAG?",
        "mode": "mix",
        "response_type": "Multiple Paragraphs",
        "chunk_top_k": 5,
        "include_references": True,
        "include_chunk_content": False,
    }


class _FakeResponse:
    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in [json.dumps({"response": "one"}), "", json.dumps({"response": "two"})]:
            yield line


class _FakeStream:
    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    def stream(self, method, path, json):
        return _FakeStream()


@pytest.mark.asyncio
async def test_stream_post_parses_ndjson():
    with patch("server.httpx.AsyncClient", _FakeClient):
        events = [
            event
            async for event in LightRAGAPIClient().stream_post(
                "query/stream", {"query": "What is RAG?"}
            )
        ]

    assert events == [{"response": "one"}, {"response": "two"}]