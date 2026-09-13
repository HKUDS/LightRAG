"""Operator-visible reporting when the extraction cache is skipped (#3833).

Since the reference-before-row change, ``use_llm_func_with_cache`` refuses to
write a cache row whose reference it could not record on the owning chunk --
the safe direction, because the alternative is a row holding document text that
nothing can ever delete. But it means the extraction cache silently stopped
working for those chunks: the next run re-calls the LLM for them. Left in the
server log alone, an operator watching the WebUI never learns that.

The reporting contract pinned here mirrors the one for token-limit truncation:

- every skipped write is logged with chunk + file identity (server log,
  unbounded, one line per event);
- ``pipeline_status`` gets the FIRST event immediately plus exactly ONE
  aggregate at the end of the stage -- never one line per chunk, because
  ``history_messages`` is a bounded ring and an unwritable ``text_chunks``
  storage skips on every chunk of the document;
- the aggregate is published on every exit, including the one where a chunk
  raises, because it rides the same ``finally`` as the truncation summary.

Nothing is recorded on the document: unlike truncation, the trigger is a
storage that cannot be written rather than a property of the content.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.operate import extract_entities
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _FakeCache:
    def __init__(self):
        self.global_config = {"enable_llm_cache_for_entity_extract": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def get_by_ids(self, keys):
        return [self._store.get(k) for k in keys]

    async def upsert(self, entries):
        self._store.update(entries)

    async def index_done_callback(self):
        return None


class _ChunkStore(_FakeCache):
    """text_chunks double seeded with the chunk rows extraction will attach to."""

    def __init__(self, rows: dict):
        super().__init__()
        self._store = {key: dict(value) for key, value in rows.items()}


class _UnwritableChunks(_ChunkStore):
    """Its writes fail, so the cache reference cannot be recorded."""

    async def upsert(self, entries):
        raise RuntimeError("text_chunks write is down")


def _make_chunks(count: int = 1, file_path: str = "reports/annual.md") -> dict:
    content = "Test content."
    return {
        f"chunk-{index:03d}": {
            "tokens": len(content),
            "content": f"{content} {index}",
            "full_doc_id": "doc-001",
            "chunk_order_index": index,
            "file_path": file_path,
        }
        for index in range(1, count + 1)
    }


def _make_global_config(extract_func=None, *, gleaning: int = 0) -> dict:
    extract_func = extract_func or AsyncMock(return_value=_EXTRACTION_RESULT)
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": gleaning,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "llm_model_max_async": 1,
        "enable_llm_cache_for_entity_extract": True,
    }


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    """``lightrag.utils.logger`` disables propagation; restore it so caplog
    can see WARNING records emitted from inside ``lightrag.operate``."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


@pytest.fixture
def status_messages(monkeypatch) -> list[str]:
    """Capture what extraction publishes to ``pipeline_status``.

    ``PipelineStatusLogger`` is instantiated inside ``extract_entities``, so
    patching the class in ``operate``'s namespace is the available seam.
    """
    captured: list[str] = []

    class _CaptureLogger:
        def __init__(self, pipeline_status):
            self.pipeline_status = pipeline_status

        def log(self, *messages):
            captured.extend(messages)

    monkeypatch.setattr(operate, "PipelineStatusLogger", _CaptureLogger)
    return captured


def _skip_lines(messages: list[str]) -> list[str]:
    return [m for m in messages if "cache write" in m.lower()]


@pytest.mark.asyncio
async def test_the_first_skip_is_published_with_chunk_and_file_identity(
    status_messages,
):
    chunks = _make_chunks(1)
    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(),
        llm_response_cache=_FakeCache(),
        text_chunks_storage=_UnwritableChunks(chunks),
    )

    lines = _skip_lines(status_messages)
    # One first-occurrence line plus one end-of-stage aggregate.
    assert len(lines) == 2, status_messages
    assert "chunk-001" in lines[0]
    assert "reports/annual.md" in lines[0]
    assert "1 of 1 chunks" in lines[1]
    assert "recomputed" in lines[1]


@pytest.mark.asyncio
async def test_status_lines_do_not_grow_with_the_number_of_skipped_chunks(
    status_messages,
):
    """An unwritable text_chunks skips on every chunk; the ring must survive it."""
    chunks = _make_chunks(6)
    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(),
        llm_response_cache=_FakeCache(),
        text_chunks_storage=_UnwritableChunks(chunks),
    )

    lines = _skip_lines(status_messages)
    assert len(lines) == 2, lines
    assert "6 of 6 chunks" in lines[1]


@pytest.mark.asyncio
async def test_every_occurrence_still_reaches_the_server_log(
    status_messages, _propagate_lightrag_logger, caplog
):
    chunks = _make_chunks(3)
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        await extract_entities(
            chunks=chunks,
            global_config=_make_global_config(),
            llm_response_cache=_FakeCache(),
            text_chunks_storage=_UnwritableChunks(chunks),
        )

    per_chunk = [
        record.getMessage()
        for record in caplog.records
        if "cache write skipped for chunk" in record.getMessage()
    ]
    assert len(per_chunk) == 3, per_chunk
    for index in (1, 2, 3):
        assert any(f"chunk-{index:03d}" in message for message in per_chunk)


@pytest.mark.asyncio
async def test_a_run_whose_cache_writes_all_land_reports_nothing(status_messages):
    """No false alarm: a writable chunk store records every reference."""
    chunks = _make_chunks(2)
    text_chunks = _ChunkStore(chunks)
    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(),
        llm_response_cache=_FakeCache(),
        text_chunks_storage=text_chunks,
    )

    assert _skip_lines(status_messages) == []
    # Guard the fixture: the references really were recorded.
    attached = [
        key
        for row in text_chunks._store.values()
        for key in (row.get("llm_cache_list") or [])
    ]
    assert len(attached) == 2, attached


@pytest.mark.asyncio
async def test_the_summary_is_published_when_a_chunk_raises_midway(status_messages):
    """The aggregate rides the same ``finally`` as the truncation summary."""
    chunks = _make_chunks(2)
    calls: list[str] = []

    async def flaky_llm(prompt: str, *args, **kwargs) -> str:
        calls.append(prompt)
        if len(calls) > 1:
            raise RuntimeError("second chunk exploded")
        return _EXTRACTION_RESULT

    with pytest.raises(Exception, match="second chunk exploded"):
        await extract_entities(
            chunks=chunks,
            global_config=_make_global_config(flaky_llm),
            llm_response_cache=_FakeCache(),
            text_chunks_storage=_UnwritableChunks(chunks),
        )

    lines = _skip_lines(status_messages)
    assert len(lines) == 2, lines
    assert "1 of 2 chunks" in lines[1]
