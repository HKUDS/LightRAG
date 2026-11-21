"""Query metadata reports the keywords the search used, one entry per keyword.

``_normalize_keyword_list`` keeps a list-shaped keyword payload per item "so
multi-word phrases that legitimately contain commas are not broken apart".
``kg_query`` then joins each list with ``", "`` for the search, and
``_build_query_context`` rebuilt ``metadata["keywords"]`` by splitting that
string on ``", "`` again, which breaks exactly those keywords apart.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import operate
from lightrag.base import QueryParam
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline


class _Characters:
    """One token per character: no encoding data to load."""

    def encode(self, content):
        return list(map(ord, content))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


async def test_keyword_containing_a_comma_is_reported_whole(monkeypatch):
    high_level = ["Treaty negotiations, 1871"]
    low_level = ["Washington, D.C.", "Hamilton Fish"]
    keyword_model = AsyncMock(
        return_value=json.dumps(
            {"high_level_keywords": high_level, "low_level_keywords": low_level}
        )
    )
    config = {
        "tokenizer": Tokenizer("characters", _Characters()),
        "role_llm_funcs": {"query": AsyncMock(), "keyword": keyword_model},
    }
    search = {
        "final_entities": [
            {
                "entity_name": "Hamilton Fish",
                "entity_type": "person",
                "description": "Secretary of State",
                "source_id": "chunk-1",
            }
        ],
        "final_relations": [],
        "vector_chunks": [],
        "chunk_tracking": {},
        "query_embedding": None,
    }
    monkeypatch.setattr(operate, "_perform_kg_search", AsyncMock(return_value=search))
    monkeypatch.setattr(operate, "_merge_all_chunks", AsyncMock(return_value=[]))

    result = await operate.kg_query(
        "Who negotiated the treaty signed in Washington?",
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        QueryParam(mode="hybrid", only_need_context=True, enable_rerank=False),
        config,
    )

    assert result.raw_data["metadata"]["keywords"] == {
        "high_level": high_level,
        "low_level": low_level,
    }
