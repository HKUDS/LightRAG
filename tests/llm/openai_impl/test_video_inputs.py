"""Offline regression coverage for video content transport."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm import openai as binding

pytestmark = pytest.mark.offline


@pytest.fixture
def client():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", reasoning_content=None),
            )
        ],
        usage=None,
    )
    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        ),
        close=AsyncMock(),
    )
    with patch.object(binding, "create_openai_async_client", return_value=fake):
        yield fake


@pytest.mark.parametrize(
    "video",
    [
        "https://example.com/clip.mp4",
        "data:video/mp4;base64,AAAA",
        "mm_file://file_id",
        {"url": "mm_file://file_id", "fps": 1, "detail": "default"},
    ],
)
async def test_video_content_preserves_options_and_history(client, video):
    before = deepcopy(video)
    history = [{"role": "assistant", "content": "previous"}]
    result = await binding.openai_complete_if_cache(
        model="video-capable-model",
        prompt="Summarize the clip",
        system_prompt="Describe the scene",
        history_messages=history,
        video_inputs=[video],
    )
    assert result == "ok"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "video_inputs" not in kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "Describe the scene"},
        *history,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize the clip"},
                {
                    "type": "video_url",
                    "video_url": {"url": video} if isinstance(video, str) else video,
                },
            ],
        },
    ]
    assert video == before
    assert history == [{"role": "assistant", "content": "previous"}]


async def test_images_and_multiple_videos_share_user_message(client):
    await binding.openai_complete_if_cache(
        model="video-capable-model",
        prompt="Compare the media",
        image_inputs=[{"base64": "aW1hZ2U=", "mime_type": "image/png"}],
        video_inputs=["mm_file://first", "mm_file://second"],
    )
    content = client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "video_url",
        "video_url",
    ]
    assert content[1]["image_url"]["url"] == "data:image/png;base64,aW1hZ2U="
    assert content[2]["video_url"]["url"] == "mm_file://first"
    assert content[3]["video_url"]["url"] == "mm_file://second"


async def test_empty_video_inputs_preserve_text_only_payload(client):
    await binding.openai_complete_if_cache(
        model="video-capable-model", prompt="Hello", video_inputs=[]
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert "video_inputs" not in kwargs


@pytest.mark.parametrize("video", ["", " ", {}, {"url": None}, {"url": 123}])
async def test_invalid_video_urls_fail_before_request(client, video):
    with pytest.raises(ValueError, match="nonempty url"):
        await binding.openai_complete_if_cache(
            model="video-capable-model", prompt="Describe", video_inputs=[video]
        )
    client.chat.completions.create.assert_not_called()


async def test_invalid_video_type_fails_before_request(client):
    with pytest.raises(TypeError, match="URL strings or dicts"):
        await binding.openai_complete_if_cache(
            model="video-capable-model", prompt="Describe", video_inputs=[123]
        )
    client.chat.completions.create.assert_not_called()
