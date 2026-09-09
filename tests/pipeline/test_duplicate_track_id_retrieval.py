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

Both halves of that contract are pinned, because the fix is a trade: the
top-level ``track_id`` field now answers the duplicate upload, so the original's
track_id is displaced into ``metadata.original_track_id`` (alongside
``original_doc_id``). Losing that metadata would trade one unanswerable lookup
for another — the client could reach its own record but no longer name the
document it collided with.
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


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"trk-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _assert_both_track_ids_resolve(
    rag: LightRAG,
    *,
    original_track_id: str,
    dup_track_id: str,
    dup_kind: str,
) -> None:
    """Each of the two uploads must be answerable under the track_id ITS OWN
    call was handed, and the duplicate must name the document it collided with.
    """
    original_docs = await rag.aget_docs_by_track_id(original_track_id)
    assert len(original_docs) == 1, (
        "the original document's track_id must answer with only its own record "
        "— a duplicate stamped with the original's track_id lands here instead"
    )
    original_doc_id, original_status = next(iter(original_docs.items()))
    assert original_status.track_id == original_track_id
    assert original_status.status == DocStatus.PENDING

    dup_docs = await rag.aget_docs_by_track_id(dup_track_id)
    assert len(dup_docs) == 1, (
        "the duplicate upload's own track_id must resolve to its "
        "FAILED duplicate record, not come back empty"
    )
    (dup_status,) = dup_docs.values()
    assert dup_status.track_id == dup_track_id
    assert dup_status.status == DocStatus.FAILED
    assert dup_status.metadata.get("is_duplicate") is True
    assert dup_status.metadata.get("duplicate_kind") == dup_kind
    # The displaced original identity: without these the client reaches its own
    # record but can no longer name what it collided with.
    assert dup_status.metadata.get("original_doc_id") == original_doc_id
    assert dup_status.metadata.get("original_track_id") == original_track_id


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

            await _assert_both_track_ids_resolve(
                rag,
                original_track_id="track-original",
                dup_track_id="track-dup-filename",
                dup_kind="filename",
            )
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

            await _assert_both_track_ids_resolve(
                rag,
                original_track_id="track-original",
                dup_track_id="track-dup-content",
                dup_kind="content_hash",
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
