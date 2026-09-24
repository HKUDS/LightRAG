"""Regression test for a caller-supplied messages kwarg colliding with the
messages list _ollama_model_if_cache builds internally.

A vision_model_func wired up for multimodal calls may pass messages=[...]
straight through to _ollama_model_if_cache. Since the function has no
explicit messages parameter, that value lands in **kwargs and previously
collided with the internally built messages list at the ollama_client.chat()
call site, raising a TypeError instead of reaching the model.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.ollama import _ollama_model_if_cache

pytestmark = pytest.mark.offline


def _make_fake_client():
    return SimpleNamespace(
        chat=AsyncMock(
            return_value={"message": {"content": "ok"}, "done_reason": "stop"}
        ),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


async def test_caller_supplied_messages_kwarg_does_not_crash():
    """A vision_model_func-style caller passing messages= must not crash."""
    fake_client = _make_fake_client()
    caller_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ]

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(
            model="qwen3-vl",
            prompt="describe this image",
            messages=caller_messages,
        )

    assert result == "ok"
    assert fake_client.chat.call_args.kwargs["messages"] == caller_messages
    # messages must not also appear positionally/duplicated in call_args.args
    assert fake_client.chat.call_args.args == ()
