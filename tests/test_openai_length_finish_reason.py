from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import LengthFinishReasonError

from lightrag.llm.openai import openai_complete_if_cache


@pytest.mark.offline
@pytest.mark.asyncio
async def test_length_finish_reason_falls_back_to_raw_content():
    raw_json = (
        '{"entities":[{"name":"Alice","type":"Person",'
        '"description":"Founder"}],"relationships":[]}'
    )
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    content=raw_json,
                    parsed=None,
                    reasoning_content="",
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                parse=AsyncMock(
                    side_effect=LengthFinishReasonError(completion=completion)
                ),
                create=AsyncMock(),
            )
        ),
        close=AsyncMock(),
    )

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract entities",
            entity_extraction=True,
            max_completion_tokens=128,
        )

    assert result == raw_json
    fake_client.chat.completions.parse.assert_awaited_once()
    fake_client.chat.completions.create.assert_not_called()
    fake_client.close.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_keyword_extraction_uses_json_object_create_mode():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}',
                    parsed=None,
                    reasoning_content="",
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=6,
            total_tokens=11,
        ),
    )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                parse=AsyncMock(),
                create=AsyncMock(return_value=completion),
            )
        ),
        close=AsyncMock(),
    )

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract keywords",
            keyword_extraction=True,
        )

    assert result == '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    fake_client.chat.completions.parse.assert_not_called()
    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }
    fake_client.close.assert_awaited_once()
