"""Regression tests for ``extract_entities`` chunk_results ordering.

``asyncio.wait`` returns ``done`` as a set whose iteration order derives from
object identity (``Task`` inherits ``object.__hash__``), so it is neither
completion order nor creation order, and it is unstable across processes.
Collecting results while walking ``done`` therefore produced a different
permutation of ``chunk_results`` on every run of the same document.

That permutation is not cosmetic: ``merge_nodes_and_edges`` feeds it into the
order-preserving ``source_id`` / ``file_path`` dedup (no sort to fall back on)
and into ``apply_source_ids_limit``, which past the cap truncates a *different
subset* of chunks. Two ingests of one document could thus persist different
chunk references, and different descriptions with them.

``test_chunk_results_follow_ordered_chunks`` forces ``done`` to iterate in
reverse and asserts ``chunk_results[i]`` still maps to ``ordered_chunks[i]``.
Against the pre-fix code it fails with the order exactly reversed.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


# Chunk key -> chunk content. Contents are distinguishable so the fake LLM can
# answer per chunk, and the entity name is derived from the key so a result can
# be traced back to the chunk it came from.
_CHUNK_CONTENTS = {
    "chunk-alpha": "Alpha content.",
    "chunk-bravo": "Bravo content.",
    "chunk-charlie": "Charlie content.",
}


def _entity_name(chunk_key: str) -> str:
    return chunk_key.split("-", 1)[1].upper()


def _extraction_result(chunk_key: str) -> str:
    name = _entity_name(chunk_key)
    return f"(entity<|#|>{name}<|#|>CONCEPT<|#|>Description of {name})<|COMPLETE|>"


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


async def _fake_extract(prompt: str, *args, **kwargs) -> str:
    """Return a per-chunk extraction result, keyed off the chunk content
    embedded in the prompt."""
    for key, content in _CHUNK_CONTENTS.items():
        if content in prompt:
            return _extraction_result(key)
    raise AssertionError(f"prompt carried no known chunk content: {prompt[:200]!r}")


def _make_global_config(extract_func, max_async: int = 3) -> dict:
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
    }


@pytest.fixture
def _reverse_done_order(monkeypatch):
    """Make ``asyncio.wait`` hand back ``done`` in reverse creation order.

    A real set only guarantees *some* order, which makes the bug impossible to
    pin down in a test. Reversing is the deterministic worst case: code that
    still returns results in input order cannot be passing by luck.
    """
    real_wait = asyncio.wait

    async def _reversed_wait(fs, **kwargs):
        done, pending = await real_wait(fs, **kwargs)
        completed_in_creation_order = [task for task in fs if task in done]
        return list(reversed(completed_in_creation_order)), pending

    monkeypatch.setattr(asyncio, "wait", _reversed_wait)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_chunk_results_follow_ordered_chunks(_reverse_done_order):
    """``chunk_results[i]`` must correspond to ``ordered_chunks[i]`` even when
    the completed-task container iterates in an unrelated order."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks()
    chunk_results = await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(AsyncMock(side_effect=_fake_extract)),
    )

    assert len(chunk_results) == len(chunks)

    # source_id is the field the ordering actually reaches persisted state
    # through, so assert on it rather than on the entity name.
    observed_source_ids = [
        record["source_id"]
        for maybe_nodes, _maybe_edges in chunk_results
        for records in maybe_nodes.values()
        for record in records
    ]
    assert observed_source_ids == list(chunks)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_first_exception_survives_pending_tasks():
    """Fail-fast is unchanged: the chunk's own exception propagates, not an
    ``InvalidStateError`` from touching a task that never completed.

    This guards the reason the exception scan and the result materialisation
    are two separate passes. Folding them into one loop over ``tasks`` would
    call ``.exception()`` on still-pending tasks, and the surrounding
    ``except Exception`` would latch that ``InvalidStateError`` as the first
    exception, masking the real failure.
    """
    from lightrag.operate import extract_entities

    async def _explode_on_alpha(prompt: str, *args, **kwargs) -> str:
        if _CHUNK_CONTENTS["chunk-alpha"] in prompt:
            raise RuntimeError("chunk alpha exploded")
        # The siblings stay pending until fail-fast cancels them.
        await asyncio.Event().wait()
        raise AssertionError("unreachable: sibling should have been cancelled")

    with pytest.raises(Exception) as excinfo:
        await extract_entities(
            chunks=_make_chunks(),
            global_config=_make_global_config(AsyncMock(side_effect=_explode_on_alpha)),
        )

    message = str(excinfo.value)
    assert "chunk alpha exploded" in message
    assert "InvalidStateError" not in message
