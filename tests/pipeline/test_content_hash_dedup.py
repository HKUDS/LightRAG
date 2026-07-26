"""content-hash duplicate detection is bounded (LR2 Phase 2.5) and never cyclic.

``get_duplicate_doc_by_content_hash`` used to fall back to a full
``get_docs_by_statuses(list(DocStatus))`` scan whenever the indexed lookup
returned the current doc itself (the common case: a unique/original doc whose
own content_hash is already persisted). Phase 2.5 pushes the self-exclusion
INTO the query via ``get_doc_by_content_hash(..., exclude_doc_id=...)`` and
deletes the fallback, so the check is a single bounded lookup — verified here
against the JSON backend end-to-end, including that the fallback is gone.

Also pinned here: a row that is itself marked ``metadata.is_duplicate`` is a
POINTER at a content holder, not a holder, so it must never be reported as the
original of the very document it points at. Answering "yes, a duplicate — of that
row" closes an is_duplicate cycle whose shared canonical source ends up with no
primary at all, which the repair endpoint cannot settle.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id
from lightrag.utils_pipeline import (
    get_duplicate_doc_by_content_hash,
    get_existing_doc_by_content_hash,
)

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 8

    async def __call__(self, texts):
        return [[0.0] * 8 for _ in texts]


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
        workspace=f"chd-{uuid4().hex[:8]}",
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


async def _storage(tmp_path, rows: dict) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()
    async with storage._storage_lock:
        storage._data.update(rows)
    return storage


def _row(content_hash: str, created_at: str, metadata: dict | None = None) -> dict:
    return {
        "content_summary": "s",
        "content_length": 3,
        "file_path": "f.txt",
        "status": DocStatus.PROCESSED,
        "created_at": created_at,
        "updated_at": created_at,
        "metadata": metadata if metadata is not None else {},
        "error_msg": None,
        "chunks_list": [],
        "content_hash": content_hash,
    }


def _demoted_row(content_hash: str, created_at: str, original_doc_id: str) -> dict:
    """A row a source-conflict repair demoted: content and status kept, its claim
    on the canonical source replaced by a pointer at the primary that won."""
    return _row(
        content_hash,
        created_at,
        {"is_duplicate": True, "original_doc_id": original_doc_id},
    )


def test_get_doc_by_content_hash_excludes_self(tmp_path):
    async def _run():
        storage = await _storage(
            tmp_path,
            {
                "doc-early": _row("hashX", "2026-01-01T00:00:00"),
                "doc-late": _row("hashX", "2026-02-01T00:00:00"),
                "doc-solo": _row("hashY", "2026-01-01T00:00:00"),
            },
        )
        # Without exclusion the earliest holder wins.
        got = await storage.get_doc_by_content_hash("hashX")
        assert got is not None and got[0] == "doc-early"
        # Excluding the earliest yields the OTHER holder, not None.
        got = await storage.get_doc_by_content_hash("hashX", exclude_doc_id="doc-early")
        assert got is not None and got[0] == "doc-late"
        # Excluding the sole holder of a hash yields None (no other match).
        assert (
            await storage.get_doc_by_content_hash("hashY", exclude_doc_id="doc-solo")
            is None
        )

    asyncio.run(_run())


def test_duplicate_check_finds_other_and_ignores_self(tmp_path):
    async def _run():
        storage = await _storage(
            tmp_path,
            {
                "doc-orig": _row("dup", "2026-01-01T00:00:00"),
                "doc-copy": _row("dup", "2026-02-01T00:00:00"),
                "doc-unique": _row("uniq", "2026-01-01T00:00:00"),
            },
        )
        # The later copy finds the original.
        match = await get_duplicate_doc_by_content_hash(storage, "dup", "doc-copy")
        assert match is not None and match[0] == "doc-orig"
        # The original, checking itself, finds the copy (another holder exists).
        match = await get_duplicate_doc_by_content_hash(storage, "dup", "doc-orig")
        assert match is not None and match[0] == "doc-copy"
        # A unique doc (only itself holds the hash) → no duplicate.
        assert (
            await get_duplicate_doc_by_content_hash(storage, "uniq", "doc-unique")
            is None
        )

    asyncio.run(_run())


def test_a_row_demoted_in_favour_of_this_doc_is_not_its_duplicate(tmp_path):
    """Fix-proof: the post-parse check returned the demoted row, so the primary an
    operator explicitly kept was marked FAILED-duplicate OF the row they demoted —
    both rows ``is_duplicate=true`` naming each other and their shared canonical
    source left with no primary at all, which no repair can settle (the endpoint
    only accepts a current primary candidate).

    No race needed: a PENDING upload is an ordinary conflict candidate, so the
    repair commits while the kept primary still has to be parsed, and the demoted
    row keeps the content hash that the parse then matches.
    """

    async def _run():
        storage = await _storage(
            tmp_path,
            {
                # The repair kept doc-primary and demoted doc-demoted.
                "doc-demoted": _demoted_row(
                    "dup", "2026-01-01T00:00:00", "doc-primary"
                ),
                "doc-primary": _row("dup", "2026-02-01T00:00:00"),
            },
        )
        # The exclusion lives in the backend query, both halves of it: the row
        # being processed AND a row that merely points at it.
        assert (
            await storage.get_doc_by_content_hash("dup", exclude_doc_id="doc-primary")
            is None
        )
        assert (
            await get_duplicate_doc_by_content_hash(storage, "dup", "doc-primary")
            is None
        )

    asyncio.run(_run())


def test_a_pointer_row_does_not_block_re_ingesting_its_missing_original(tmp_path):
    """Enqueue side of the same rule: a duplicate row naming a doc id that no
    longer exists must not answer for it, or content whose primary row was deleted
    could never be re-uploaded — a doc id is derived deterministically from the
    file name, so the re-upload asks about the very id the pointer names."""

    async def _run():
        storage = await _storage(
            tmp_path,
            {"doc-pointer": _demoted_row("dup", "2026-01-01T00:00:00", "doc-gone")},
        )
        # Without the candidate id, the lookup is unchanged: SOME row holds the hash.
        match = await get_existing_doc_by_content_hash(storage, "dup")
        assert match is not None and match[0] == "doc-pointer"
        # Re-uploading the same bytes derives doc-gone again → admitted.
        assert (
            await get_existing_doc_by_content_hash(
                storage, "dup", candidate_doc_id="doc-gone"
            )
            is None
        )
        # A different new document IS a genuine duplicate of that content.
        assert (
            await get_existing_doc_by_content_hash(
                storage, "dup", candidate_doc_id="doc-other"
            )
            is not None
        )

    asyncio.run(_run())


def test_a_pointer_row_does_not_hide_a_genuine_third_holder(tmp_path):
    """Skipping the pointer must not STOP the search.

    Fix-proof: the skip started life as a post-filter over the single row the
    backend returned, so when the earliest holder was a pointer back at the
    asking document the answer became None even though a real third holder
    existed — the asking document was then ingested as an original and its
    content was duplicated in the graph. The exclusion belongs where the rows are
    selected, so the query keeps walking to the earliest surviving holder.

    The third holder here is created AFTER the asking document, the ordering that
    a "look once more, excluding the pointer" fix would still get wrong (that
    lookup returns the asking row itself and reads it as "no other holder").
    """

    async def _run():
        storage = await _storage(
            tmp_path,
            {
                "doc-pointer": _demoted_row("dup", "2026-01-01T00:00:00", "doc-asking"),
                "doc-asking": _row("dup", "2026-02-01T00:00:00"),
                "doc-third": _row("dup", "2026-03-01T00:00:00"),
            },
        )
        match = await get_duplicate_doc_by_content_hash(storage, "dup", "doc-asking")
        assert match is not None and match[0] == "doc-third"

        # Same for the enqueue leg, where the asking id does not exist yet.
        del storage._data["doc-asking"]
        match = await get_existing_doc_by_content_hash(
            storage, "dup", candidate_doc_id="doc-asking"
        )
        assert match is not None and match[0] == "doc-third"

    asyncio.run(_run())


def test_a_duplicate_pointing_at_a_third_document_is_still_a_duplicate(tmp_path):
    """The guard is narrow on purpose. When the matched row points somewhere ELSE,
    the content really does exist under another document, and dropping the match
    would fail open — re-admitting a genuine content duplicate."""

    async def _run():
        storage = await _storage(
            tmp_path,
            {
                "doc-pointer": _demoted_row("dup", "2026-01-01T00:00:00", "doc-orig"),
                "doc-orig": _row("dup", "2026-01-02T00:00:00"),
            },
        )
        match = await get_duplicate_doc_by_content_hash(storage, "dup", "doc-new")
        assert match is not None and match[0] == "doc-pointer"

    asyncio.run(_run())


def test_enqueue_re_admits_a_document_whose_pointer_row_outlived_it(tmp_path):
    """The enqueue leg, end to end through ``apipeline_enqueue_documents``.

    Fix-proof: with only a pointer row left (``a.txt`` demoted in favour of
    ``b.txt``, then ``b.txt`` deleted), re-uploading ``b.txt`` was rejected as a
    content-hash duplicate OF THE POINTER — so the file could never be ingested
    again, and each attempt only added another FAILED duplicate record naming a
    document that does not exist.
    """

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            body = "the shared body"
            await rag.apipeline_enqueue_documents(input=body, file_paths="a.txt")
            demoted_id = compute_mdhash_id("a.txt", prefix="doc-")
            revived_id = compute_mdhash_id("b.txt", prefix="doc-")

            # A source-conflict repair kept b.txt and demoted a.txt; b.txt was
            # then deleted, leaving only the pointer.
            row = dict(await rag.doc_status.get_by_id(demoted_id))
            row["metadata"] = {"is_duplicate": True, "original_doc_id": revived_id}
            await rag.doc_status.upsert({demoted_id: row})

            await rag.apipeline_enqueue_documents(input=body, file_paths="b.txt")

            revived = await rag.doc_status.get_by_id(revived_id)
            assert revived is not None, "b.txt was not enqueued at all"
            assert revived["status"] == DocStatus.PENDING.value
            assert not (revived.get("metadata") or {}).get("is_duplicate")
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_duplicate_check_never_full_scans_all_statuses(tmp_path):
    """The Phase 2.5 guarantee: the dedup path no longer calls
    get_docs_by_statuses (the removed O(entire-store) fallback)."""

    async def _run():
        storage = await _storage(
            tmp_path, {"doc-unique": _row("uniq", "2026-01-01T00:00:00")}
        )

        async def _forbidden(*args, **kwargs):
            raise AssertionError(
                "get_duplicate_doc_by_content_hash must not full-scan "
                "doc_status (the removed fallback)"
            )

        storage.get_docs_by_statuses = _forbidden

        # Unique doc → the pre-Phase-2.5 code would hit the fallback here.
        assert (
            await get_duplicate_doc_by_content_hash(storage, "uniq", "doc-unique")
            is None
        )

    asyncio.run(_run())


def test_the_repair_and_duplicate_marking_are_linearized_by_the_source_lock(tmp_path):
    """LR2 §5.5: a source-conflict commit must be mutually exclusive with every
    writer that changes the key's candidate set — the processing stage's duplicate
    marking included.

    Fix-proof: the marking held no source-key lock, so it could land between the
    repair's demotions and its post-commit verification. The repair then reported
    "storage unavailable, retry" (503) for storage that was perfectly available,
    with irreversible demotions already written — a half-applied operation
    reported as a transient failure.

    What the lock buys is ORDER, not a permanent Unique: the marking here is
    blocked for the whole commit, the repair verifies Unique and returns, and only
    then does the marking run — leaving the key Absent as an ordinary content-dedup
    transition AFTER a completed operation. (Marking first is the other
    linearization: the repair then refuses before demoting anything, covered by
    tests/tools/test_source_conflict_repair_cli.py.)
    """

    async def _run():
        from lightrag.base import DocProcessingStatus
        from lightrag.tools.source_conflict_repair import repair_one_conflict

        rag = await _build_rag(tmp_path)
        try:
            # a.pdf is claimed by two primaries; doc-third legitimately holds the
            # same content under its own source, so the marking has a real target.
            async with rag.doc_status._storage_lock:
                rag.doc_status._data.update(
                    {
                        "doc-keep": _row("dup", "2026-01-01T00:00:00"),
                        "doc-lose": _row("dup", "2026-01-02T00:00:00"),
                        "doc-third": _row("dup", "2026-01-03T00:00:00"),
                    }
                )
                rag.doc_status._data["doc-keep"]["file_path"] = "a.pdf"
                # Unparsed: no content_hash yet, which is why the commit's
                # pre-check cannot see the collision the parse is about to find —
                # the documented residual that makes the lock the only guard here.
                rag.doc_status._data["doc-keep"]["content_hash"] = ""
                rag.doc_status._data["doc-lose"]["file_path"] = "a.pdf"
                rag.doc_status._data["doc-lose"]["content_hash"] = "other"
                rag.doc_status._data["doc-third"]["file_path"] = "c.pdf"
            # The kept primary must have content, or the commit refuses it for
            # that reason instead (its own test covers that).
            await rag.full_docs.upsert({"doc-keep": {"content": "body"}})

            status_doc = DocProcessingStatus(
                content_summary="s",
                content_length=3,
                file_path="a.pdf",
                status=DocStatus.PARSING,
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
                content_hash="dup",
            )

            async def _mark():
                return await rag._mark_duplicate_after_parse(
                    doc_id="doc-keep",
                    status_doc=status_doc,
                    file_path="a.pdf",
                    content_hash="dup",
                    content_length=3,
                )

            marking: list[asyncio.Task] = []
            original = rag.doc_status.repair_source_conflict

            async def _commit_then_let_the_marking_try(key, **kwargs):
                result = await original(key, **kwargs)
                if not kwargs.get("dry_run", True):
                    # Inside the repair's lock, after the demotions: give the
                    # marking every chance to interleave before verification.
                    task = asyncio.create_task(_mark())
                    marking.append(task)
                    for _ in range(50):
                        await asyncio.sleep(0)
                    assert not task.done(), (
                        "the duplicate marking interleaved between the demotions "
                        "and the verification"
                    )
                return result

            rag.doc_status.repair_source_conflict = _commit_then_let_the_marking_try

            # No exception: the commit's own verification saw the key settled.
            result = await repair_one_conflict(
                rag.doc_status,
                "a.pdf",
                "doc-keep",
                workspace=rag.workspace,
                full_docs=rag.full_docs,
                apply=True,
            )
            assert result.committed is True

            assert await marking[0] is True  # runs once the lock is released
            demoted = await rag.doc_status.get_by_id("doc-lose")
            assert demoted["metadata"]["original_doc_id"] == "doc-keep"
            marked = await rag.doc_status.get_by_id("doc-keep")
            assert marked["metadata"]["original_doc_id"] == "doc-third"
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
