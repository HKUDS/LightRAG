"""``doc_status.created_at`` is the immutable first-persistence timestamp.

Filesystem age is a pre-persistence scan concern. The scan's disk spool orders
new files by mtime before enqueue; ``apipeline_enqueue_documents`` then stamps
every new row with the current UTC time. No caller can inject a filesystem or
other external timestamp into persistent creation metadata.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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


def test_created_at_is_the_first_persistence_time(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            before = datetime.now(timezone.utc)
            await rag.apipeline_enqueue_documents(input="body", file_paths="now.txt")
            after = datetime.now(timezone.utc)

            row = await rag.doc_status.get_by_id(_doc_id("now.txt"))
            created = datetime.fromisoformat(row["created_at"])
            assert before <= created <= after
            assert row["updated_at"] >= row["created_at"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_created_at_records_enqueue_order_for_the_scheduler(tmp_path):
    """After scan has mtime-sorted its candidates, sequential persistence fixes
    that order into the ordinary ``(created_at, id)`` scheduling key."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            arrivals = ["oldest.txt", "middle.txt", "newest.txt"]
            for name in arrivals:
                await rag.apipeline_enqueue_documents(
                    input=f"body of {name}",
                    file_paths=name,
                )
                # Make the assertion independent of platform clock resolution.
                await asyncio.sleep(0.001)

            page = await rag.doc_status.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=10
            )
            assert [record.file_path for record in page.docs.values()] == arrivals
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_created_at_never_reaches_full_docs(tmp_path):
    """Creation/scheduling metadata belongs to doc_status alone."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await rag.apipeline_enqueue_documents(input="body", file_paths="only.txt")
            row = await rag.full_docs.get_by_id(_doc_id("only.txt"))
            assert row is not None
            assert "created_at" not in row
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
