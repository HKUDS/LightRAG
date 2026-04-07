from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import LengthFinishReasonError

from lightrag.llm.openai import openai_complete_if_cache


@pytest.mark.offline
@pytest.mark.asyncio
async def test_length_finish_reason_falls_back_to_raw_content():
    raw_json = (
        '{"entities":[{"entity_name":"Alice","entity_type":"Person",'
        '"entity_description":"Founder"}],"relationships":[]}'
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
