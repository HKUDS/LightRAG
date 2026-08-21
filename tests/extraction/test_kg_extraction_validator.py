"""Tests for the optional per-chunk extraction-quality hook (#3691).

``LightRAG.kg_extraction_validator`` is the last word before the merge:
whatever it drops never reaches ``merge_nodes_and_edges``, the vector
stores, or any ``source_id`` chain. ``None`` (the default) must leave the
pipeline byte-identical.
"""

from unittest.mock import AsyncMock

import pytest

from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


_CHUNK_CONTENTS = {
    "chunk-alpha": "Alpha content about Alice.",
    "chunk-bravo": "Bravo content about Bob.",
}


def _extraction_result(name: str) -> str:
    return f"(entity<|#|>{name}<|#|>CONCEPT<|#|>Description of {name})<|COMPLETE|>"


async def _fake_extract(prompt: str, *args, **kwargs) -> str:
    for key, content in _CHUNK_CONTENTS.items():
        if content in prompt:
            return _extraction_result(key.split("-", 1)[1].upper())
    raise AssertionError(f"prompt carried no known chunk content: {prompt[:200]!r}")


def _make_chunks() -> dict[str, dict]:
    return {
        key: {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": index,
            "file_path": f"{key}.md",
        }
        for index, (key, content) in enumerate(_CHUNK_CONTENTS.items())
    }


def _make_global_config(extract_func, validator=None, max_async: int = 3) -> dict:
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": tokenizer,
        "llm_model_max_async": max_async,
        "kg_extraction_validator": validator,
    }


def _collect_names(chunk_results) -> list[str]:
    return [
        name
        for maybe_nodes, _ in chunk_results
        for name in maybe_nodes
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_validator_leaves_results_unchanged():
    from lightrag.operate import extract_entities

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(AsyncMock(side_effect=_fake_extract)),
    )
    assert sorted(_collect_names(chunk_results)) == ["ALPHA", "BRAVO"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_sync_validator_receives_chunk_key_text_and_shapes():
    from lightrag.operate import extract_entities

    seen: list[tuple[str, str, list[str], list[tuple[str, str]]]] = []

    def validator(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        seen.append((chunk_key, chunk_text, list(maybe_nodes), list(maybe_edges)))
        return maybe_nodes, maybe_edges

    await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=validator
        ),
    )

    assert sorted(item[0] for item in seen) == sorted(_CHUNK_CONTENTS)
    by_key = {item[0]: item for item in seen}
    assert by_key["chunk-alpha"][1] == _CHUNK_CONTENTS["chunk-alpha"]
    # The shapes the hook receives are exactly the merge inputs.
    assert by_key["chunk-alpha"][2] == ["ALPHA"]
    assert by_key["chunk-alpha"][3] == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_sync_validator_filters_entities():
    from lightrag.operate import extract_entities

    def drop_alpha(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        maybe_nodes.pop("ALPHA", None)
        return maybe_nodes, maybe_edges

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=drop_alpha
        ),
    )
    assert _collect_names(chunk_results) == ["BRAVO"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_async_validator_is_awaited():
    from lightrag.operate import extract_entities

    async def drop_everything(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        return {}, {}

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=drop_everything
        ),
    )
    assert _collect_names(chunk_results) == []
    # Empty results still produce one entry per chunk — the merge treats an
    # empty extraction as "nothing from this chunk", not a failure.
    assert len(chunk_results) == len(_CHUNK_CONTENTS)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_hook_runs_after_extraction_before_return():
    """The hook is the LAST word per chunk: a validator that inspects
    maybe_nodes sees exactly what the merge would have received."""
    from lightrag.operate import extract_entities

    received: dict[str, dict] = {}

    def capture(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        received[chunk_key] = {
            name: [record["source_id"] for record in records]
            for name, records in maybe_nodes.items()
        }
        return maybe_nodes, maybe_edges

    chunks = _make_chunks()
    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=capture
        ),
    )

    for chunk_key in chunks:
        for name, source_ids in received[chunk_key].items():
            assert source_ids == [chunk_key], (
                "the hook must see the merge-shaped records, source_id included"
            )
