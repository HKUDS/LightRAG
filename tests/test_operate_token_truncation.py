"""``_apply_token_truncation`` truncates entities_context and relations_context
against separate budgets (max_entity_tokens / max_relation_tokens), with no
cross-filtering step afterward. A relation can therefore survive its own
budget while one of its endpoints gets cut from entities_context by the
entity budget -- shipping a relationship the LLM (and the structured API
response's raw_data["data"]["relationships"]) can't resolve against any
entity in the accompanying list.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _apply_token_truncation
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    """1:1 char-to-token mapping so token budgets are easy to reason about."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _tokenizer():
    return Tokenizer("dummy", _CharTokenizer())


@pytest.mark.asyncio
async def test_relation_referencing_a_truncated_entity_is_dropped():
    final_entities = [
        {"entity_name": "A", "entity_type": "PERSON", "description": "short"},
        {"entity_name": "B", "entity_type": "PERSON", "description": "x" * 500},
    ]
    final_relations = [
        {"src_id": "A", "tgt_id": "B", "description": "A relates to B"},
    ]
    search_result = {
        "final_entities": final_entities,
        "final_relations": final_relations,
    }
    # Entity budget only fits "A"; relation budget is generous, so the A-B
    # relation would survive its own truncation untouched.
    query_param = QueryParam(max_entity_tokens=60, max_relation_tokens=100_000)
    global_config = {"tokenizer": _tokenizer()}

    result = await _apply_token_truncation(search_result, query_param, global_config)

    assert [e["entity"] for e in result["entities_context"]] == ["A"]
    assert result["relations_context"] == []
    assert result["filtered_relations"] == []


@pytest.mark.asyncio
async def test_relation_between_surviving_entities_is_kept():
    """Negative twin: a relation whose endpoints both survive truncation
    must not be dropped by the fix."""
    final_entities = [
        {"entity_name": "A", "entity_type": "PERSON", "description": "short"},
        {"entity_name": "B", "entity_type": "PERSON", "description": "also short"},
    ]
    final_relations = [
        {"src_id": "A", "tgt_id": "B", "description": "A relates to B"},
    ]
    search_result = {
        "final_entities": final_entities,
        "final_relations": final_relations,
    }
    query_param = QueryParam(max_entity_tokens=100_000, max_relation_tokens=100_000)
    global_config = {"tokenizer": _tokenizer()}

    result = await _apply_token_truncation(search_result, query_param, global_config)

    assert {e["entity"] for e in result["entities_context"]} == {"A", "B"}
    assert [(r["entity1"], r["entity2"]) for r in result["relations_context"]] == [
        ("A", "B")
    ]
    assert [(r["src_id"], r["tgt_id"]) for r in result["filtered_relations"]] == [
        ("A", "B")
    ]
