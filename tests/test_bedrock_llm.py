from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.bedrock import bedrock_complete, bedrock_complete_if_cache


class _FakeBedrockClient:
    def __init__(self, captured_calls: list[dict]):
        self._captured_calls = captured_calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **kwargs):
        self._captured_calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
                        }
                    ]
                }
            }
        }


class _FakeSession:
    def __init__(self, captured_calls: list[dict]):
        self._captured_calls = captured_calls

    def client(self, *_args, **_kwargs):
        return _FakeBedrockClient(self._captured_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_complete_forwards_keyword_extraction_to_if_cache():
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "bedrock-model"})

    with patch(
        "lightrag.llm.bedrock.bedrock_complete_if_cache",
        AsyncMock(return_value="{}"),
    ) as mocked_complete:
        await bedrock_complete(
            prompt="hello",
            hashing_kv=hashing_kv,
            keyword_extraction=True,
        )

    assert mocked_complete.await_args.kwargs["keyword_extraction"] is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_keyword_extraction_does_not_inject_system_prompt():
    captured_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls),
    ):
        result = await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            response_format={"type": "json_object"},
        )

    assert result == '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    assert len(captured_calls) == 1
    assert "system" not in captured_calls[0]
