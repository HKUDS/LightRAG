"""Entity extraction receives document dates without changing chunk content."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lightrag.operate import extract_entities
from lightrag.utils import TokenLimitTruncationTally, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

_DOCUMENT_MARKER = "---Document Context---"
_SECTION_MARKER = "---Section Context---"


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _FullDocs:
    def __init__(self, records: dict[str, dict]):
        self.records = records
        self.requested_ids: list[str] = []

    async def get_by_ids(self, ids: list[str]):
        self.requested_ids = ids
        return [self.records.get(doc_id) for doc_id in ids]


def _config(use_json: bool):
    response = '{"entities": [], "relationships": []}' if use_json else "<|COMPLETE|>"
    llm = AsyncMock(return_value=response)
    return {
        "role_llm_funcs": {
            "extract": llm,
            "keyword": llm,
            "query": llm,
            "vlm": llm,
        },
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "entity_extraction_use_json": use_json,
        "addon_params": {},
        "tokenizer": Tokenizer("document-date-test", _CharTokenizer()),
        "llm_model_max_async": 1,
    }, llm


def _chunks():
    return {
        "chunk-2018": {
            "tokens": 16,
            "content": "Alice led Acme.",
            "full_doc_id": "doc-2018",
            "chunk_order_index": 0,
            "heading": {
                "level": 2,
                "heading": "Leadership",
                "parent_headings": ["Organization"],
            },
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_json", [False, True], ids=["text", "json"])
async def test_extraction_prompt_includes_document_and_section_context(use_json):
    config, llm = _config(use_json)
    chunks = _chunks()
    full_docs = _FullDocs(
        {"doc-2018": {"content": "full text", "document_date": "2018-10-01"}}
    )

    await extract_entities(
        chunks=chunks,
        global_config=config,
        full_docs_storage=full_docs,
    )

    prompt = llm.await_args_list[0].args[0]
    assert _DOCUMENT_MARKER in prompt
    assert "Document date: 2018-10-01" in prompt
    assert _SECTION_MARKER in prompt
    assert "Organization → Leadership" in prompt
    assert prompt.index(_DOCUMENT_MARKER) < prompt.index(_SECTION_MARKER)
    assert prompt.index(_SECTION_MARKER) < prompt.rindex("---Input Text---")
    assert full_docs.requested_ids == ["doc-2018"]
    assert "document_date" not in chunks["chunk-2018"]


@pytest.mark.asyncio
async def test_old_document_without_date_keeps_extraction_prompt_byte_identical():
    undated_config, undated_llm = _config(use_json=False)
    legacy_config, legacy_llm = _config(use_json=False)

    await extract_entities(
        chunks=_chunks(),
        global_config=undated_config,
        full_docs_storage=_FullDocs({"doc-2018": {"content": "full text"}}),
    )
    await extract_entities(chunks=_chunks(), global_config=legacy_config)

    undated_prompt = undated_llm.await_args_list[0].args[0]
    legacy_prompt = legacy_llm.await_args_list[0].args[0]
    assert undated_prompt == legacy_prompt
    assert _DOCUMENT_MARKER not in undated_prompt
    assert _SECTION_MARKER in undated_prompt


@pytest.mark.asyncio
async def test_legacy_positional_truncation_tally_keeps_its_meaning():
    config, llm = _config(use_json=False)
    tally = TokenLimitTruncationTally()

    await extract_entities(_chunks(), config, None, None, None, None, tally)

    prompt = llm.await_args_list[0].args[0]
    assert _DOCUMENT_MARKER not in prompt
    assert _SECTION_MARKER in prompt
