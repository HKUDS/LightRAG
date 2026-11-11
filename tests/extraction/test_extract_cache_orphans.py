"""No extraction cache row may outlive the only reference that reaches it (#3833).

An ``extract`` cache row carries the chunk text verbatim plus the entities
pulled from it, and the ONLY thing that ever finds it again is the owning
chunk's ``llm_cache_list``. The row used to be written first and the key
attached once at the end of the chunk, so anything cutting that gap short left
a row nothing could reach: a sibling chunk's exception cancelling the task
through ``extract_entities``' ``FIRST_EXCEPTION`` wait, a hard kill, or a
swallowed storage error in the attach.

The reference is now recorded BEFORE the row is written, so the leftover state
flips to a dangling reference, which every reader tolerates. These tests pin the
invariant itself -- every ``extract`` row in the cache is referenced by some
chunk -- rather than the code path that currently maintains it.
"""

import asyncio

import pytest

from lightrag.operate import extract_entities
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


_ALPHA = "Alpha content about Alice."
_BRAVO = "Bravo content about Bob."
_CHUNK_CONTENTS = {"chunk-alpha": _ALPHA, "chunk-bravo": _BRAVO}


def _extraction_result(name: str) -> str:
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


def _make_global_config(extract_func, *, max_gleaning: int = 0) -> dict:
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": max_gleaning,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("dummy", DummyTokenizer()),
        "llm_model_max_async": 3,
        "kg_extraction_validator": None,
        "enable_llm_cache_for_entity_extract": True,
    }


class _FakeKV:
    """Minimal BaseKVStorage stand-in: enough for the cache-key attach path."""

    def __init__(self, rows: dict | None = None, global_config: dict | None = None):
        self.data = dict(rows or {})
        # The cache path reads its gates off the storage itself.
        self.global_config = global_config or {}

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, rows: dict):
        self.data.update(rows)

    async def index_done_callback(self):
        return None


def _cache_gates() -> dict:
    return {
        "enable_llm_cache": True,
        "enable_llm_cache_for_entity_extract": True,
    }


def _referenced_keys(text_chunks: _FakeKV) -> set[str]:
    return {
        key
        for row in text_chunks.data.values()
        for key in (row.get("llm_cache_list") or [])
    }


def _extract_rows(llm_cache: _FakeKV) -> set[str]:
    return {
        key
        for key, row in llm_cache.data.items()
        if isinstance(row, dict) and row.get("cache_type") == "extract"
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_extract_cache_row_survives_a_cancelled_sibling():
    """The flagship #3833 scenario: a sibling raises while this chunk is mid-flight.

    ``extract_entities`` waits with ``FIRST_EXCEPTION`` and cancels the pending
    tasks, so alpha is cancelled after its cache row is durable. Under the old
    order alpha's key lived only in an in-memory collector that the
    cancellation discarded, leaving the row unreachable forever.
    """
    chunks = _make_chunks()
    text_chunks = _FakeKV({key: dict(value) for key, value in chunks.items()})

    row_written = asyncio.Event()
    # Only the cancellation unparks alpha, so it can never reach any
    # end-of-chunk step: the invariant has to hold from inside the write.
    never = asyncio.Event()

    class _ParkingCache(_FakeKV):
        async def upsert(self, rows: dict):
            self.data.update(rows)
            if any(
                isinstance(row, dict) and row.get("cache_type") == "extract"
                for row in rows.values()
            ):
                row_written.set()
                await never.wait()

    llm_cache = _ParkingCache(global_config=_cache_gates())

    async def fake_llm(prompt: str, *args, **kwargs) -> str:
        if _BRAVO in prompt:
            # Detonate only once alpha's row is durable, so the cancellation
            # lands in exactly the window this change is about.
            await row_written.wait()
            raise RuntimeError("bravo exploded")
        if _ALPHA in prompt:
            return _extraction_result("ALPHA")
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    with pytest.raises(Exception, match="bravo exploded"):
        await asyncio.wait_for(
            extract_entities(
                chunks=chunks,
                global_config=_make_global_config(fake_llm),
                llm_response_cache=llm_cache,
                text_chunks_storage=text_chunks,
            ),
            timeout=10,
        )

    rows = _extract_rows(llm_cache)
    assert rows, "fixture must actually write an extract cache row"
    orphaned = rows - _referenced_keys(text_chunks)
    assert not orphaned, (
        "extract cache rows are unreachable — no chunk's llm_cache_list names "
        f"them, so delete_llm_cache can never remove them: {sorted(orphaned)}"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_every_extract_row_is_referenced_on_the_happy_path():
    """Each call's key is attached exactly once, gleaning included."""
    chunks = _make_chunks()
    text_chunks = _FakeKV({key: dict(value) for key, value in chunks.items()})
    llm_cache = _FakeKV(global_config=_cache_gates())

    calls: list[str] = []

    async def fake_llm(prompt: str, *args, **kwargs) -> str:
        for key, content in _CHUNK_CONTENTS.items():
            if content in prompt:
                calls.append(key)
                return _extraction_result(key.split("-", 1)[1].upper())
        # The gleaning prompt carries the prior result as history rather than
        # the chunk text, so it is identified by its own template instead.
        if "last extraction task" in prompt:
            calls.append("gleaning")
            return _extraction_result("GLEANED")
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(fake_llm, max_gleaning=1),
        llm_response_cache=llm_cache,
        text_chunks_storage=text_chunks,
    )

    # One initial call plus one gleaning call per chunk.
    assert len(calls) == 4, calls
    rows = _extract_rows(llm_cache)
    assert rows == _referenced_keys(text_chunks)
    for chunk_id, row in text_chunks.data.items():
        attached = row.get("llm_cache_list") or []
        assert len(attached) == 2, (chunk_id, attached)
        assert len(set(attached)) == len(attached), (chunk_id, attached)
