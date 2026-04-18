import pytest

from lightrag.operate import _parse_keywords_payload


class _FakeKeywordModel:
    def model_dump(self):
        return {
            "high_level_keywords": ["AI"],
            "low_level_keywords": ["RAG", "Graph"],
        }


@pytest.mark.offline
def test_parse_keywords_payload_accepts_model_like_objects():
    hl_keywords, ll_keywords = _parse_keywords_payload(_FakeKeywordModel())

    assert hl_keywords == ["AI"]
    assert ll_keywords == ["RAG", "Graph"]


@pytest.mark.offline
def test_parse_keywords_payload_extracts_json_from_wrapped_text():
    result = """
    analysis first
    {"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]}
    trailing note
    """

    hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert hl_keywords == ["AI", "Agents"]
    assert ll_keywords == ["RAG", "LightRAG"]
