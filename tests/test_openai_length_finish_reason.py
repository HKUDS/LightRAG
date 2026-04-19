from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.openai import openai_complete_if_cache


def _make_completion(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
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


def _make_fake_client(completion):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=completion),
            )
        ),
        close=AsyncMock(),
    )


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        return None


def _make_stream_chunk(content=None, reasoning_content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                )
            )
        ]
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_length_finish_reason_returns_raw_content():
    """Truncated responses (finish_reason='length') still yield raw content.

    After the dispatch simplification, we no longer rely on the typed
    ``LengthFinishReasonError`` path — ``create()`` returns the partial
    content unchanged and upstream tolerant JSON parsing handles it.
    """
    raw_json = (
        '{"entities":[{"name":"Alice","type":"Person",'
        '"description":"Founder"}],"relationships":[]}'
    )
    completion = _make_completion(raw_json, finish_reason="length")
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract entities",
            response_format={"type": "json_object"},
            max_completion_tokens=128,
        )

    assert result == raw_json
    fake_client.chat.completions.create.assert_awaited_once()
    fake_client.close.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_object_response_format_forwarded_to_create():
    completion = _make_completion(
        '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    )
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract keywords",
            response_format={"type": "json_object"},
        )

    assert result == '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }
    fake_client.close.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_legacy_entity_extraction_emits_deprecation_warning():
    completion = _make_completion('{"entities":[],"relationships":[]}')
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        with pytest.warns(DeprecationWarning):
            await openai_complete_if_cache(
                model="test-model",
                prompt="Extract entities",
                entity_extraction=True,
            )

    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_legacy_keyword_extraction_emits_deprecation_warning():
    completion = _make_completion('{"high_level_keywords":[],"low_level_keywords":[]}')
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        with pytest.warns(DeprecationWarning):
            await openai_complete_if_cache(
                model="test-model",
                prompt="Extract keywords",
                keyword_extraction=True,
            )

    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_typed_response_format_is_rejected():
    completion = _make_completion("{}")
    fake_client = _make_fake_client(completion)

    class FakeSchemaModel:
        pass

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        with pytest.raises(TypeError, match="typed/Pydantic"):
            await openai_complete_if_cache(
                model="test-model",
                prompt="Extract entities",
                response_format=FakeSchemaModel,
            )

    fake_client.chat.completions.create.assert_not_awaited()
    fake_client.close.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_streaming_structured_output_disables_cot():
    fake_stream = _FakeAsyncStream(
        [
            _make_stream_chunk(reasoning_content="this should not be included"),
            _make_stream_chunk(content='{"answer":"ok"}'),
        ]
    )
    fake_client = _make_fake_client(fake_stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        stream = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract entities",
            stream=True,
            enable_cot=True,
            response_format={"type": "json_object"},
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

    assert "".join(chunks) == '{"answer":"ok"}'
    fake_client.close.assert_awaited_once()
