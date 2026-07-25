"""Caller-supplied ``doc_status.created_at`` (LR2 Phase 4-e, §8.3.A/§8.5).

``created_at`` is the leading component of the immutable scheduling sort key
``(created_at, id)``, so ``apipeline_enqueue_documents`` is the ONLY place it
can be set.  ``/scan`` supplies each discovered file's ``st_mtime`` there so a
bulk scan of an existing corpus drains oldest-file-first; upload/text keep
``now()``.

Covered here:

* a supplied timestamp lands on the row AND drives the keyset page order
  (arrival order and mtime order are deliberately opposed in the fixture);
* the timestamp stays out of ``full_docs`` — it is doc_status data;
* omitting it stamps ``now()`` (upload/text contract, unchanged);
* offsets are normalised to UTC because the sort key is compared as a string;
* a naive / malformed / mis-counted timestamp fails the enqueue instead of
  landing a permanently mis-sorted row.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _chunking(
    tokenizer,
    content,
    split_by_character,
    split_by_character_only,
    chunk_overlap_token_size,
    chunk_token_size,
) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"ca-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _doc_id(name: str) -> str:
    return compute_mdhash_id(name, prefix="doc-")


def test_supplied_created_at_orders_the_sweep_by_file_age(tmp_path):
    """The three files arrive newest-first but must be swept oldest-first: the
    keyset page order follows the supplied mtimes, not the arrival order."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            base = datetime(2021, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
            # Arrival order (newest → oldest) is the reverse of file age.
            arrivals = [
                ("newest.txt", base + timedelta(days=2)),
                ("middle.txt", base + timedelta(days=1)),
                ("oldest.txt", base),
            ]
            for name, stamp in arrivals:
                await rag.apipeline_enqueue_documents(
                    input=f"body of {name}",
                    file_paths=name,
                    created_at=stamp.isoformat(),
                )

            for name, stamp in arrivals:
                row = await rag.doc_status.get_by_id(_doc_id(name))
                assert row["created_at"] == stamp.isoformat(), name
                # updated_at is still the write time, not the file's age.
                assert row["updated_at"] != stamp.isoformat(), name

            page = await rag.doc_status.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=10
            )
            assert [record.file_path for record in page.docs.values()] == [
                "oldest.txt",
                "middle.txt",
                "newest.txt",
            ]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_supplied_created_at_never_reaches_full_docs(tmp_path):
    """created_at is doc_status data: the full_docs payload must not grow a
    scheduling field just because the caller passed one."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await rag.apipeline_enqueue_documents(
                input="body",
                file_paths="only.txt",
                created_at="2021-03-04T05:06:07+00:00",
            )
            row = await rag.full_docs.get_by_id(_doc_id("only.txt"))
            assert row is not None
            assert "created_at" not in row
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_omitted_created_at_stamps_now(tmp_path):
    """Upload/text contract, unchanged: no timestamp supplied → now()."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            before = datetime.now(timezone.utc)
            await rag.apipeline_enqueue_documents(input="body", file_paths="now.txt")
            after = datetime.now(timezone.utc)
            row = await rag.doc_status.get_by_id(_doc_id("now.txt"))
            created = datetime.fromisoformat(row["created_at"])
            assert before <= created <= after
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_created_at_is_normalised_to_utc(tmp_path):
    """Two rows written with different offsets must sort by instant, not by
    offset — the file-backed and Redis backends compare the key as a string,
    so the stored form is always ``+00:00``."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await rag.apipeline_enqueue_documents(
                input="body",
                file_paths="shanghai.txt",
                created_at="2021-03-04T13:06:07+08:00",
            )
            row = await rag.doc_status.get_by_id(_doc_id("shanghai.txt"))
            assert row["created_at"] == "2021-03-04T05:06:07+00:00"
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_created_at_broadcast_applies_to_every_input(tmp_path):
    """A single string broadcasts like parse_engine / process_options do."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await rag.apipeline_enqueue_documents(
                input=["body a", "body b"],
                file_paths=["a.txt", "b.txt"],
                created_at="2021-03-04T05:06:07+00:00",
            )
            for name in ("a.txt", "b.txt"):
                row = await rag.doc_status.get_by_id(_doc_id(name))
                assert row["created_at"] == "2021-03-04T05:06:07+00:00", name
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "bad_value, message",
    [
        ("2021-03-04T05:06:07", "must carry a UTC offset"),
        ("not-a-timestamp", "Malformed created_at"),
        ("", "non-empty ISO-8601 string"),
        (1614834367, "non-empty ISO-8601 string"),
    ],
)
def test_unusable_created_at_fails_the_enqueue(tmp_path, bad_value, message):
    """created_at is immutable once written, so a value that cannot be ordered
    against the existing rows must be refused at the door."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            with pytest.raises(ValueError, match=message):
                await rag.apipeline_enqueue_documents(
                    input="body",
                    file_paths="bad.txt",
                    created_at=bad_value,
                )
            assert await rag.doc_status.get_by_id(_doc_id("bad.txt")) is None
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_created_at_count_must_match_the_inputs(tmp_path):
    """A misaligned list would silently give docs someone else's age."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            with pytest.raises(ValueError, match="Number of created_at timestamps"):
                await rag.apipeline_enqueue_documents(
                    input=["body a", "body b"],
                    file_paths=["a.txt", "b.txt"],
                    created_at=["2021-03-04T05:06:07+00:00"],
                )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
