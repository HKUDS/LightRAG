import pytest
from unittest.mock import patch

from lightrag.base import QueryParam
from lightrag.operate import _parse_keywords_payload, extract_keywords_only


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


_ANSWER = '{"high_level_keywords":["AI"],"low_level_keywords":["RAG","Graph"]}'


@pytest.mark.offline
@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            '```json\n{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}\n```'
            "\n\nWait, I need to correct my format - no markdown code fences"
            f" allowed:\n\n{_ANSWER}\n```",
            id="fence-prose-second-object",
        ),
        pytest.param(f"{_ANSWER}\n{_ANSWER}", id="two-bare-objects"),
    ],
)
def test_parse_keywords_payload_accepts_repeated_objects(result):
    """json_repair returns a list for a reply that repeats its object; the
    first object carrying keywords is the answer."""
    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert is_valid is True
    assert hl_keywords == ["AI"]
    assert ll_keywords[0] == "RAG"


@pytest.mark.offline
@pytest.mark.parametrize(
    "leading_object",
    [
        pytest.param(
            '{"high_level_keywords": ["<high_level_keyword>"],'
            ' "low_level_keywords": ["<low_level_keyword>"]}',
            id="echoed-format-template",
        ),
        pytest.param(
            '{"high_level_keywords": [], "low_level_keywords": []}',
            id="echoed-empty-example",
        ),
        pytest.param('{"note": "thinking"}', id="unrelated-object"),
    ],
)
def test_parse_keywords_payload_skips_leading_object_without_keywords(leading_object):
    """An object that carries no real keyword must not shadow the answer."""
    result = f"The format is {leading_object}. Answer:\n{_ANSWER}"

    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert is_valid is True
    assert hl_keywords == ["AI"]
    assert ll_keywords == ["RAG", "Graph"]


@pytest.mark.offline
def test_parse_keywords_payload_keeps_empty_answer_valid_when_repeated():
    empty = '{"high_level_keywords": [], "low_level_keywords": []}'

    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(f"{empty}\n{empty}")

    assert is_valid is True
    assert hl_keywords == []
    assert ll_keywords == []


@pytest.mark.offline
def test_parse_keywords_payload_drops_template_placeholders():
    """An echoed placeholder is never a keyword, even in a single clean
    object; a real angle-bracket keyword is kept."""
    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(
        '{"high_level_keywords": ["<high_level_keyword>", "HTML"],'
        ' "low_level_keywords": ["<low_level_keyword_2>", "<div>"]}'
    )

    assert is_valid is True
    assert hl_keywords == ["HTML"]
    assert ll_keywords == ["<div>"]


@pytest.mark.offline
def test_keyword_template_placeholder_pattern_covers_the_prompt_template():
    """Every keyword of the prompt's format template must be recognised as a
    placeholder, or an echoed template comes back as keywords."""
    import json

    from lightrag.operate import _KEYWORD_TEMPLATE_PLACEHOLDER
    from lightrag.prompt import PROMPTS

    placeholders = [
        keyword
        for example in PROMPTS["keywords_extraction_examples"]
        for keywords in json.loads(example).values()
        for keyword in keywords
    ]

    assert placeholders
    for placeholder in placeholders:
        assert _KEYWORD_TEMPLATE_PLACEHOLDER.fullmatch(placeholder), placeholder
    assert _KEYWORD_TEMPLATE_PLACEHOLDER.fullmatch("<low_level_keyword_2>")
    assert not _KEYWORD_TEMPLATE_PLACEHOLDER.fullmatch("<div>")


@pytest.mark.offline
@pytest.mark.parametrize(
    "result",
    [
        pytest.param('["AI", "RAG"]', id="array-of-strings"),
        pytest.param(f"[{_ANSWER}]", id="array-of-one-object"),
        pytest.param(f"[{_ANSWER},{_ANSWER}", id="truncated-array-of-objects"),
        pytest.param(f"Result: [{_ANSWER},{_ANSWER}", id="prose-then-truncated-array"),
    ],
)
def test_parse_keywords_payload_rejects_top_level_array(result):
    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert is_valid is False
    assert hl_keywords == []
    assert ll_keywords == []


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
