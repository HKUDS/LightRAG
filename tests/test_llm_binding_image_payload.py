"""Offline tests for image_inputs payload shape per LLM binding.

These tests stub the underlying network clients with ``unittest.mock`` so they
exercise only the message-construction layer that this repository owns.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.offline


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\x5c\xcc\xd9\x9e\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


@pytest.mark.asyncio
async def test_openai_binding_inserts_image_url_content_block():
    from lightrag.llm import openai as openai_mod

    fake_choice = MagicMock()
    fake_choice.message.content = "ok"
    fake_choice.message.reasoning_content = None
    fake_choice.finish_reason = "stop"
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)
    fake_client.close = AsyncMock()

    with patch.object(
        openai_mod, "create_openai_async_client", return_value=fake_client
    ):
        await openai_mod.openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="describe",
            api_key="dummy",
            image_inputs=[PNG_B64],
        )

    _, kwargs = fake_client.chat.completions.create.call_args
    messages = kwargs["messages"]
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "text", "text": "describe"}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_openai_binding_text_only_remains_plain_string():
    from lightrag.llm import openai as openai_mod

    fake_choice = MagicMock()
    fake_choice.message.content = "ok"
    fake_choice.message.reasoning_content = None
    fake_choice.finish_reason = "stop"
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)
    fake_client.close = AsyncMock()

    with patch.object(
        openai_mod, "create_openai_async_client", return_value=fake_client
    ):
        await openai_mod.openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="describe",
            api_key="dummy",
        )

    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["messages"][-1]["content"] == "describe"


@pytest.mark.asyncio
async def test_ollama_binding_attaches_images_to_user_message():
    from lightrag.llm import ollama as ollama_mod

    fake_client = MagicMock()
    fake_client.chat = AsyncMock(return_value={"message": {"content": "ok"}})
    fake_client._client = MagicMock()
    fake_client._client.aclose = AsyncMock()

    with patch.object(ollama_mod.ollama, "AsyncClient", return_value=fake_client):
        await ollama_mod._ollama_model_if_cache(
            model="llava",
            prompt="describe",
            image_inputs=[PNG_B64],
        )

    _, kwargs = fake_client.chat.call_args
    user_msg = kwargs["messages"][-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "describe"
    assert user_msg["images"] == [PNG_B64]


@pytest.mark.asyncio
async def test_anthropic_binding_inserts_image_content_block():
    from lightrag.llm import anthropic as anthropic_mod

    captured: dict[str, Any] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)

            async def empty():
                if False:
                    yield None

            return empty()

    fake_client = MagicMock()
    fake_client.messages = FakeMessages()

    with patch.object(anthropic_mod, "AsyncAnthropic", return_value=fake_client):
        await anthropic_mod.anthropic_complete_if_cache(
            model="claude-3-opus",
            prompt="describe",
            api_key="dummy",
            image_inputs=[PNG_B64],
        )

    user_content = captured["messages"][-1]["content"]
    assert isinstance(user_content, list)
    image_blocks = [b for b in user_content if b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": PNG_B64,
    }
    assert user_content[-1] == {"type": "text", "text": "describe"}


@pytest.mark.asyncio
async def test_lollms_binding_rejects_image_inputs():
    from lightrag.llm import lollms as lollms_mod

    with pytest.raises(NotImplementedError):
        await lollms_mod.lollms_model_if_cache(
            model="unused",
            prompt="hi",
            image_inputs=[PNG_B64],
        )


@pytest.mark.asyncio
async def test_bedrock_binding_forces_non_stream_when_image_present():
    from lightrag.llm import bedrock as bedrock_mod

    captured: dict[str, Any] = {}

    class FakeBedrockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def converse(self, **kwargs):
            captured["mode"] = "converse"
            captured["args"] = kwargs
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
            }

        async def converse_stream(self, **kwargs):
            captured["mode"] = "converse_stream"
            captured["args"] = kwargs
            return {"stream": []}

    class FakeSession:
        def client(self, *_, **__):
            return FakeBedrockClient()

    with patch.object(bedrock_mod.aioboto3, "Session", return_value=FakeSession()):
        await bedrock_mod.bedrock_complete_if_cache(
            "anthropic.claude-3-haiku-20240307-v1:0",
            "describe",
            stream=True,
            image_inputs=[PNG_B64],
            aws_region="us-east-1",
        )

    assert captured["mode"] == "converse"
    user_msg = captured["args"]["messages"][-1]
    image_blocks = [block for block in user_msg["content"] if "image" in block]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image"]["format"] == "png"
    assert image_blocks[0]["image"]["source"]["bytes"] == PNG_BYTES
