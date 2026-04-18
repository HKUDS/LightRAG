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
    assert "Keyword extraction response required JSON repair" in mocked_warning.call_args[0][0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_keywords_only_accepts_empty_keyword_cache_without_requery():
    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("model_func should not be called on a valid empty cache hit")

    param = QueryParam(model_func=should_not_run)
    global_config = {"addon_params": {"language": "en"}}

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
