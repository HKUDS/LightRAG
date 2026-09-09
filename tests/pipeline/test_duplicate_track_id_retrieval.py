"""A duplicate-document record must be retrievable by the CURRENT track_id
used for the duplicate upload attempt, not only by the original document's
track_id.

Regression test for https://github.com/HKUDS/LightRAG/issues/2468: a client
uploading a duplicate got back a track_id from the enqueue call, but querying
``/documents/track_status/{track_id}`` (backed by ``aget_docs_by_track_id``)
for that exact track_id returned nothing — the duplicate's FAILED record
existed in ``doc_status``, but not under any track_id the client could ever
query for, since a client only ever learns the track_id handed back from ITS
OWN request.

``apipeline_enqueue_documents`` stamps every duplicate record it creates with
the CURRENT call's ``track_id`` (see the "Handle duplicate documents" block in
``lightrag/pipeline.py``), so this pins the behavior end-to-end through both
duplicate-detection paths: an exact file-path repeat (``duplicate_kind ==
"filename"``) and a different file_path with identical content
(``duplicate_kind == "content_hash"``).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer

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
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"trk-{uuid4().hex[:8]}",
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


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


def test_filename_duplicate_is_retrievable_by_its_own_track_id(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await rag.apipeline_enqueue_documents(
                input="the original body",
                file_paths="report.txt",
                track_id="track-original",
            )
            # Same file_path again -> duplicate_kind == "filename".
            await rag.apipeline_enqueue_documents(
                input="different body, same file name",
                file_paths="report.txt",
                track_id="track-dup-filename",
            )

            original_docs = await rag.aget_docs_by_track_id("track-original")
            assert len(original_docs) == 1
            (original_status,) = original_docs.values()
            assert original_status.status == DocStatus.PENDING

            dup_docs = await rag.aget_docs_by_track_id("track-dup-filename")
            assert len(dup_docs) == 1, (
                "the duplicate upload's own track_id must resolve to its "
                "FAILED duplicate record, not come back empty"
            )
            (dup_status,) = dup_docs.values()
            assert dup_status.track_id == "track-dup-filename"
            assert dup_status.status == DocStatus.FAILED
            assert dup_status.metadata.get("is_duplicate") is True
            assert dup_status.metadata.get("duplicate_kind") == "filename"
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_content_hash_duplicate_is_retrievable_by_its_own_track_id(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            shared_body = "identical content, different file names"
            await rag.apipeline_enqueue_documents(
                input=shared_body,
                file_paths="first.txt",
                track_id="track-original",
            )
            # Different file_path, identical content -> duplicate_kind == "content_hash".
            await rag.apipeline_enqueue_documents(
                input=shared_body,
                file_paths="second.txt",
                track_id="track-dup-content",
            )

            dup_docs = await rag.aget_docs_by_track_id("track-dup-content")
            assert len(dup_docs) == 1, (
                "the duplicate upload's own track_id must resolve to its "
                "FAILED duplicate record, not come back empty"
            )
            (dup_status,) = dup_docs.values()
            assert dup_status.track_id == "track-dup-content"
            assert dup_status.status == DocStatus.FAILED
            assert dup_status.metadata.get("is_duplicate") is True
            assert dup_status.metadata.get("duplicate_kind") == "content_hash"

            # The original keeps answering under its own track_id, untouched.
            original_docs = await rag.aget_docs_by_track_id("track-original")
            assert len(original_docs) == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
