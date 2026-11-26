"""Provider prompt-cache counts reach the chat usage tracker."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from lightrag.llm import openai
from lightrag.utils import TokenTracker

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


# Both call sites share one helper, so the absent / reported-zero branches are
# pinned on the non-streaming path; the streaming case proves its call site uses
# that helper.
@pytest.mark.parametrize(
    ("stream", "cached_tokens"),
    [
        pytest.param(False, None, id="absent"),
        pytest.param(False, 0, id="reported-zero"),
        pytest.param(False, 2, id="reported"),
        pytest.param(True, 2, id="stream-reported"),
    ],
)
async def test_openai_chat_reports_cached_prompt_tokens(
    monkeypatch, stream, cached_tokens
):
    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    response = {"id": "completion", "model": "test-model", "created": 0}

    def respond(request):
        if not stream:
            return httpx.Response(
                200,
                json={
                    **response,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )
        chunk = {**response, "object": "chat.completion.chunk"}
        events = [
            {
                **chunk,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {**chunk, "choices": [], "usage": usage},
        ]
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
            + "data: [DONE]\n\n",
        )

    tracker = TokenTracker()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        sdk = AsyncOpenAI(
            api_key="placeholder",
            base_url="https://example.invalid/v1",
            http_client=http_client,
        )
        monkeypatch.setattr(openai, "create_openai_async_client", lambda **_: sdk)
        result = await openai.openai_complete_if_cache(
            "test-model",
            "extract entities",
            stream=stream,
            token_tracker=tracker,
        )
        if stream:
            result = "".join([part async for part in result])

    assert result == "answer"
    counts = tracker.get_usage()
    assert counts["prompt_tokens"] == 4
    assert counts["total_tokens"] == 5
    if cached_tokens is None:
        assert "cached_prompt_tokens" not in counts
    else:
        assert counts["cached_prompt_tokens"] == cached_tokens
