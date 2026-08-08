import pytest
from lightrag.base import QueryParam
from lightrag.operate import _apply_token_truncation
from lightrag.utils import Tokenizer, TokenizerInterface, convert_to_user_format


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]):
        return "".join(chr(t) for t in tokens)


@pytest.mark.asyncio
async def test_apply_token_truncation_bidirectional_relation_matching():
    search_result = {
        "final_entities": [],
        "final_relations": [
            {
                "src_tgt": ("NodeA", "NodeB"),
                "description": "Relation between NodeA and NodeB",
                "keywords": "kw1",
                "weight": 2.0,
                "created_at": 1000,
            },
            {
                "src_id": "NodeD",
                "tgt_id": "NodeC",
                "description": "Reversed edge direction relation",
                "keywords": "kw2",
                "weight": 3.0,
                "created_at": 1000,
            },
        ],
        "vector_chunks": [],
        "chunk_tracking": {},
        "query_embedding": None,
    }

    query_param = QueryParam(
        mode="global",
        max_relation_tokens=1000,
        max_entity_tokens=1000,
    )

    global_config = {
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "max_relation_tokens": 1000,
        "max_entity_tokens": 1000,
    }

    result = await _apply_token_truncation(search_result, query_param, global_config)

    filtered_relations = result["filtered_relations"]
    relation_id_to_original = result["relation_id_to_original"]

    assert len(filtered_relations) == 2, (
        f"Expected 2 filtered relations, got {len(filtered_relations)}"
    )
    assert ("NodeD", "NodeC") in relation_id_to_original
    assert ("NodeC", "NodeD") in relation_id_to_original

    # Verify convert_to_user_format retrieves original DB data in both direction queries
    user_format_reverse = convert_to_user_format(
        entities_context=[],
        relations_context=[
            {
                "entity1": "NodeC",
                "entity2": "NodeD",
                "description": "Truncated desc",
                "created_at": "2026-01-01",
                "file_path": "f.txt",
            }
        ],
        chunks=[],
        references=[],
        query_mode="global",
        relation_id_to_original=relation_id_to_original,
    )

    rel_out = user_format_reverse["data"]["relationships"][0]
    assert rel_out["description"] == "Reversed edge direction relation"
    assert rel_out["weight"] == 3.0
