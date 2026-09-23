import pytest
from unittest.mock import patch

from lightrag.base import QueryParam
from lightrag.operate import (
    _is_unsupported_response_format_error,
    _parse_keywords_payload,
    extract_keywords_only,
)


class _FakeKeywordModel:
    def model_dump(self):
        return {
            "high_level_keywords": ["AI"],
            "low_level_keywords": ["RAG", "Graph"],
        }


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


def _keyword_global_config(
    model: str, binding: str = "openai", keyword_func=None
) -> dict:
    return {
        "addon_params": {"language": "en"},
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {"keyword": keyword_func} if keyword_func else {},
        "llm_cache_identities": {
            "keyword": {
                "role": "keyword",
                "binding": binding,
                "model": model,
                "host": "https://api.example.com/v1",
            }
        },
    }


@pytest.mark.offline
def test_parse_keywords_payload_accepts_model_like_objects():
    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(_FakeKeywordModel())

    assert is_valid is True
    assert hl_keywords == ["AI"]
    assert ll_keywords == ["RAG", "Graph"]


@pytest.mark.offline
def test_parse_keywords_payload_extracts_json_from_wrapped_text():
    result = """
    analysis first
    {"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]}
    trailing note
    """

    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert is_valid is True
    assert hl_keywords == ["AI", "Agents"]
    assert ll_keywords == ["RAG", "LightRAG"]


@pytest.mark.offline
@pytest.mark.parametrize(
    "result",
    [
        '```json\n{"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]}\n```',
        '```json {"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]}```',
    ],
    ids=["multiline-fence", "single-line-fence"],
)
def test_parse_keywords_payload_strips_markdown_fence(result):
    """Keyword parsing shares utils.strip_markdown_code_fence; both multi-line
    and single-line (no interior newline) fences must be stripped."""
    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert is_valid is True
    assert hl_keywords == ["AI", "Agents"]
    assert ll_keywords == ["RAG", "LightRAG"]


@pytest.mark.offline
def test_parse_keywords_payload_warns_when_json_repair_is_used():
    broken_result = (
        '{"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]'
    )

    with patch("lightrag.operate.logger.warning") as mocked_warning:
        is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(broken_result)

    assert is_valid is True
    assert hl_keywords == ["AI", "Agents"]
    assert ll_keywords == ["RAG", "LightRAG"]
    mocked_warning.assert_called_once()
    assert (
        "Keyword extraction response required JSON repair"
        in mocked_warning.call_args[0][0]
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_keywords_only_accepts_empty_keyword_cache_without_requery():
    async def should_not_run(*_args, **_kwargs):
        raise AssertionError(
            "keyword LLM should not be called on a valid empty cache hit"
        )

    param = QueryParam()
    global_config = _keyword_global_config("model-a", keyword_func=should_not_run)

    with patch(
        "lightrag.operate.handle_cache",
        return_value=('{"high_level_keywords":[],"low_level_keywords":[]}', None),
    ):
        hl_keywords, ll_keywords = await extract_keywords_only(
            "hello",
            param,
            global_config,
            hashing_kv=None,
        )

    assert hl_keywords == []
    assert ll_keywords == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_keywords_only_partitions_cache_by_keyword_llm_identity():
    cache = _FakeKVStorage()
    calls = 0

    async def keyword_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            '{"high_level_keywords":["model-'
            + str(calls)
            + '"],"low_level_keywords":["rag"]}'
        )

    param = QueryParam()

    first_hl, first_ll = await extract_keywords_only(
        "same query",
        param,
        _keyword_global_config("model-a", keyword_func=keyword_model),
        hashing_kv=cache,
    )
    second_hl, second_ll = await extract_keywords_only(
        "same query",
        param,
        _keyword_global_config("model-b", keyword_func=keyword_model),
        hashing_kv=cache,
    )

    assert first_hl == ["model-1"]
    assert first_ll == ["rag"]
    assert second_hl == ["model-2"]
    assert second_ll == ["rag"]
    assert calls == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_auto_sends_response_format_by_default():
    """Supported providers must not be downgraded: the first attempt carries the
    API-enforced JSON constraint unless the policy says otherwise."""
    captured_kwargs = {}

    async def keyword_model(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'

    hl_keywords, ll_keywords = await extract_keywords_only(
        "hello",
        QueryParam(),
        _keyword_global_config("model-a", keyword_func=keyword_model),
        hashing_kv=None,
    )

    assert captured_kwargs["response_format"] == {"type": "json_object"}
    assert hl_keywords == ["AI"]
    assert ll_keywords == ["RAG"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_auto_retries_once_without_response_format_on_compat_error():
    """The regression for servers such as LM Studio that reject response_format.

    The first request dies with a 'response_format.type' must be 'json_schema' or
    'text' style 400; auto mode retries the same prompt without the constraint and the
    second response carries the keywords.
    """
    calls = []

    async def keyword_model(*_args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "'response_format.type' must be 'json_schema' or 'text'"
            )
        return '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'

    hl_keywords, ll_keywords = await extract_keywords_only(
        "hello",
        QueryParam(),
        _keyword_global_config("model-a", keyword_func=keyword_model),
        hashing_kv=None,
    )

    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert hl_keywords == ["AI"]
    assert ll_keywords == ["RAG"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_auto_does_not_retry_unrelated_errors():
    """The fallback is only for the compatibility error, never for a generic failure:
    a transient transport error must surface, not silently degrade to a prompt-only
    JSON request that may parse to nothing."""
    calls = []

    async def keyword_model(*_args, **kwargs):
        calls.append(kwargs)
        raise ConnectionResetError("connection reset by peer")

    with pytest.raises(ConnectionResetError, match="connection reset"):
        await extract_keywords_only(
            "hello",
            QueryParam(),
            _keyword_global_config("model-a", keyword_func=keyword_model),
            hashing_kv=None,
        )

    assert len(calls) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_object_always_sends_response_format():
    captured_kwargs = {}

    async def keyword_model(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"high_level_keywords":[],"low_level_keywords":[]}'

    global_config = _keyword_global_config("model-a", keyword_func=keyword_model)
    global_config["keyword_extraction_json_mode"] = "json_object"

    await extract_keywords_only("hello", QueryParam(), global_config, hashing_kv=None)

    assert captured_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_none_never_sends_response_format():
    captured_kwargs = {}

    async def keyword_model(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'

    global_config = _keyword_global_config("model-a", keyword_func=keyword_model)
    global_config["keyword_extraction_json_mode"] = "none"

    hl_keywords, ll_keywords = await extract_keywords_only(
        "hello", QueryParam(), global_config, hashing_kv=None
    )

    assert "response_format" not in captured_kwargs
    # The tolerant parser reads plain JSON text as well as a JSON-mode response.
    assert hl_keywords == ["AI"]
    assert ll_keywords == ["RAG"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_non_json_response_is_not_treated_as_valid_keyword_output():
    """A provider that ignores the constraint and rambles must yield no keywords:
    prose that merely mentions the topic is not structured keyword output, and
    empty keywords — not fabricated ones — are what a caller can act on."""
    async def keyword_model(*_args, **kwargs):
        return "AI and RAG are two interesting topics for retrieval, graph and memory."

    hl_keywords, ll_keywords = await extract_keywords_only(
        "hello",
        QueryParam(),
        _keyword_global_config("model-a", keyword_func=keyword_model),
        hashing_kv=None,
    )

    assert hl_keywords == []
    assert ll_keywords == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_keyword_extraction_json_mode_rejects_unknown_value():
    global_config = _keyword_global_config("model-a", keyword_func=None)
    global_config["keyword_extraction_json_mode"] = "sometimes"

    with pytest.raises(ValueError, match="keyword_extraction_json_mode"):
        await extract_keywords_only(
            "hello", QueryParam(), global_config, hashing_kv=None
        )


@pytest.mark.offline
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # The field itself is rejected — retry without it is safe.
        ("'response_format.type' must be 'json_schema' or 'text'", True),
        ("Invalid response_format: json_object is not supported", True),
        ("response_format json_schema is not supported by this model", True),
        ("json_object mode is not allowed for this deployment", True),
        # Anything without the field in the message must propagate.
        ("Connection reset by peer", False),
        ("Request timed out while waiting for response_format", False),
        ("HTTP 429: rate limit exceeded", False),
        ("invalid JSON in prompt", False),
    ],
)
def test_is_unsupported_response_format_error(message, expected):
    assert _is_unsupported_response_format_error(RuntimeError(message)) is expected

