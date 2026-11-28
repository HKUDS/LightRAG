"""Zhipu response classification: moderation verdicts and empty completions.

Every case asserts the attempt count through the decorated entry point, so a
regression that re-adds one of these errors to the retry predicate is caught.
"""

import importlib
from types import SimpleNamespace

import httpx
import pytest
import zai
from tenacity import wait_none
from zai.core._errors import APIRequestFailedError

from lightrag.exceptions import (
    ContentPolicyViolationError,
    EmptyTruncatedResponseError,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

_SENSITIVE_BODY = {
    "error": {
        "code": "1301",
        "message": (
            "系统检测到输入或生成内容可能包含不安全或敏感内容，"
            "请您避免输入易产生敏感内容的提示语，感谢您的配合。"
        ),
    },
    "contentFilter": [{"level": 1, "role": "user"}],
}


@pytest.fixture
def zhipu_module(monkeypatch):
    module = importlib.import_module("lightrag.llm.zhipu")
    monkeypatch.setattr(module.zhipu_complete_if_cache.retry, "wait", wait_none())
    return module


@pytest.fixture
def install_client(monkeypatch):
    """Install a fake zai client whose chat call returns ``response`` or raises ``error``."""

    def install(*, response=None, error=None):
        calls = []

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

            def close(self):
                pass

            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self.create)
                )

            def create(self, **kwargs):
                calls.append(kwargs)
                if error is not None:
                    raise error
                return response

        monkeypatch.setattr(zai, "ZhipuAiClient", FakeClient)
        return calls

    return install


def _completion(content, finish_reason, *, reasoning_content=""):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, reasoning_content=reasoning_content
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _status_error(body):
    """Build the error zai-sdk raises for a 400 carrying ``body``."""
    request = httpx.Request(
        "POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    )
    if isinstance(body, str):
        response = httpx.Response(400, text=body, request=request)
    else:
        response = httpx.Response(400, json=body, request=request)
    return APIRequestFailedError(
        f"Error code: 400, with error text {response.text}", response=response
    )


@pytest.mark.parametrize(
    "content", ["", "flagged partial text"], ids=["empty", "partial"]
)
async def test_sensitive_finish_reason_raises_content_policy_error(
    zhipu_module, install_client, content
):
    calls = install_client(response=_completion(content, "sensitive"))

    with pytest.raises(
        ContentPolicyViolationError, match="finish_reason=sensitive"
    ) as exc:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert len(calls) == 1
    assert "flagged partial text" not in str(exc.value)


async def test_zai_1301_status_error_raises_content_policy_error(
    zhipu_module, install_client
):
    error = _status_error(_SENSITIVE_BODY)
    calls = install_client(error=error)

    with pytest.raises(ContentPolicyViolationError, match="code 1301") as exc:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert exc.value.__cause__ is error
    assert "敏感内容" in str(exc.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": "1210", "message": "parameter error"}},
        "<html>bad gateway</html>",
    ],
    ids=["other-code", "non-json-body"],
)
async def test_non_1301_status_errors_are_reraised_unchanged(
    zhipu_module, install_client, body
):
    error = _status_error(body)
    calls = install_client(error=error)

    with pytest.raises(APIRequestFailedError) as exc:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert exc.value is error
    assert len(calls) == 1


@pytest.mark.parametrize("shape", ["no-choices", "message-none"])
async def test_malformed_completion_raises_invalid_response_error(
    zhipu_module, install_client, shape
):
    if shape == "no-choices":
        response = SimpleNamespace(choices=[], usage=None)
        expected = "no choices"
    else:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=None, finish_reason="stop")], usage=None
        )
        expected = "no message"
    calls = install_client(response=response)

    with pytest.raises(zhipu_module.InvalidResponseError, match=expected):
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert len(calls) == 1


@pytest.mark.parametrize("content", ["", "   "], ids=["empty", "whitespace"])
async def test_empty_content_raises_invalid_response_error(
    zhipu_module, install_client, content
):
    calls = install_client(response=_completion(content, "stop"))

    with pytest.raises(
        zhipu_module.InvalidResponseError, match="Received empty content"
    ) as exc:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert "finish_reason=stop" in str(exc.value)
    assert len(calls) == 1


@pytest.mark.parametrize("content", ["", "   "], ids=["empty", "whitespace"])
async def test_empty_length_finish_reason_raises_empty_truncated_error(
    zhipu_module, install_client, content
):
    calls = install_client(response=_completion(content, "length"))

    with pytest.raises(EmptyTruncatedResponseError, match="token limit") as exc:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert "finish_reason=length" in str(exc.value)
    assert len(calls) == 1
