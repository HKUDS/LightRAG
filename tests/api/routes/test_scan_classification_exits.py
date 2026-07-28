"""Typed source resolution: the seven scan classification exits (LR2 Phase 4-d, §8.3).

Scan identity comes from ``resolve_doc_source_strict``, so a canonical basename
is no longer assumed to belong to exactly one primary row: two candidates are a
conflict an operator repairs by doc id, and a unique candidate is classified by
its status plus ``metadata.source_file`` — the physical name that created it.
Each exit is asserted through its OBSERVABLE effects: what is enqueued, what is
archived, what is deleted, and what the bounded job counters say.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import (  # noqa: E402
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.constants import PARSED_DIR_NAME  # noqa: E402
from lightrag.exceptions import StorageCapabilityError  # noqa: E402

DocumentManager = _document_routes.DocumentManager
run_scanning_process = _document_routes.run_scanning_process
classify_scan_file = _document_routes.classify_scan_file
_ScanFileClass = _document_routes._ScanFileClass

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    yield


class _ResolverDocStatus:
    """doc_status double driven by an explicit ``{canonical_key: resolution}`` map."""

    def __init__(
        self,
        resolutions: dict,
        rows: dict | None = None,
        *,
        delete_error: Exception | None = None,
    ):
        self.resolutions = resolutions
        self.rows = rows or {}
        self.deleted_ids: list[str] = []
        self.hydrated: list[str] = []
        self.delete_error = delete_error

    async def resolve_doc_source_strict(self, canonical_source_key):
        return self.resolutions.get(canonical_source_key, SourceAbsent())

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        self.hydrated.extend(doc_ids)
        return {doc_id: self.rows[doc_id] for doc_id in doc_ids if doc_id in self.rows}

    async def delete(self, ids):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_ids.extend(ids)


class _ClassifyRag:
    def __init__(self, resolutions, rows=None, full_docs=None, delete_error=None):
        self.workspace = f"scanexit-{uuid4().hex[:8]}"
        self.doc_status = _ResolverDocStatus(
            resolutions, rows, delete_error=delete_error
        )
        self.full_docs = full_docs if full_docs is not None else _StrictFullDocs({})
        self.process_calls = 0

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1


class _StrictFullDocs:
    supports_strict_point_reads = True

    def __init__(self, docs_by_id, *, raise_on_strict=False):
        self.docs_by_id = docs_by_id
        self.raise_on_strict = raise_on_strict

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)

    async def get_by_id_strict(self, doc_id):
        if self.raise_on_strict:
            raise RuntimeError("strict point read boom")
        return self.docs_by_id.get(doc_id)


class _WeakFullDocs:
    """A backend WITHOUT strict point reads (the base-class default)."""

    supports_strict_point_reads = False

    def __init__(self, docs_by_id):
        self.docs_by_id = docs_by_id

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)

    async def get_by_id_strict(self, doc_id):
        raise StorageCapabilityError("no strict point reads")


def _record(doc_id: str, status: DocStatus) -> DocSchedulingRecord:
    return DocSchedulingRecord(
        id=doc_id,
        status=status,
        created_at="",
        updated_at="",
        file_path=Path(doc_id).name,
        track_id=None,
        has_custom_chunk_journal=False,
    )


def _row(doc_id: str, status: DocStatus, metadata: dict) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary="",
        content_length=0,
        file_path=Path(doc_id).name,
        status=status,
        created_at="",
        updated_at="",
        metadata=metadata,
    )


def _unique(doc_id: str, status: DocStatus) -> SourceUnique:
    return SourceUnique(doc_id=doc_id, doc=_record(doc_id, status))


# ---------------------------------------------------------------------------
# Pure classification (no side effects)
# ---------------------------------------------------------------------------


def test_absent_is_claimed_new():
    async def _run():
        rag = _ClassifyRag({})
        decision = await classify_scan_file(rag, Path("fresh.txt"), "fresh.txt")
        assert decision.kind is _ScanFileClass.CLAIMED_NEW
        assert rag.doc_status.hydrated == []  # nothing to hydrate

    asyncio.run(_run())


def test_conflict_reports_bounded_candidate_sample():
    async def _run():
        rag = _ClassifyRag(
            {
                "dup.txt": SourceConflict(
                    candidate_count=3, sample_doc_ids=("doc-a", "doc-b")
                )
            }
        )
        decision = await classify_scan_file(rag, Path("dup.txt"), "dup.txt")
        assert decision.kind is _ScanFileClass.SOURCE_CONFLICT
        assert "doc-a, doc-b" in decision.detail
        assert "3 primary documents" in decision.detail
        # No identity read: a conflict is not resolvable into one row.
        assert rag.doc_status.hydrated == []

    asyncio.run(_run())


def test_processed_wins_over_alias_comparison():
    """§8.3.C precedes F/G: a PROCESSED row means archive, even for a physical
    name that differs from the row's ``source_file``."""

    async def _run():
        rag = _ClassifyRag(
            {"done.txt": _unique("doc-done", DocStatus.PROCESSED)},
            rows={
                "doc-done": _row(
                    "doc-done", DocStatus.PROCESSED, {"source_file": "done.txt"}
                )
            },
        )
        decision = await classify_scan_file(rag, Path("done.[native].txt"), "done.txt")
        assert decision.kind is _ScanFileClass.PROCESSED
        assert decision.doc_id == "doc-done"

    asyncio.run(_run())


def test_failed_without_confirmed_content_is_a_stale_stub():
    async def _run():
        rag = _ClassifyRag(
            {"broken.txt": _unique("doc-broken", DocStatus.FAILED)},
            rows={
                "doc-broken": _row(
                    "doc-broken", DocStatus.FAILED, {"source_file": "broken.txt"}
                )
            },
            full_docs=_StrictFullDocs({}),  # confirmed absent
        )
        decision = await classify_scan_file(rag, Path("broken.txt"), "broken.txt")
        assert decision.kind is _ScanFileClass.STALE_STUB
        assert decision.doc_id == "doc-broken"

    asyncio.run(_run())


@pytest.mark.parametrize(
    "full_docs",
    [
        _WeakFullDocs({}),  # capability gap
        _StrictFullDocs({}, raise_on_strict=True),  # read failure
    ],
    ids=["no_strict_capability", "strict_read_failed"],
)
def test_unconfirmed_absence_never_deletes_the_stub(full_docs):
    """§8.3.D: only a CONFIRMED absence may drop a FAILED row. A capability gap
    or a failed read must fall through to a non-destructive exit — a best-effort
    miss would delete the row of a document whose content actually exists."""

    async def _run():
        rag = _ClassifyRag(
            {"broken.txt": _unique("doc-broken", DocStatus.FAILED)},
            rows={
                "doc-broken": _row(
                    "doc-broken", DocStatus.FAILED, {"source_file": "broken.txt"}
                )
            },
            full_docs=full_docs,
        )
        decision = await classify_scan_file(rag, Path("broken.txt"), "broken.txt")
        assert decision.kind is _ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE
        assert rag.doc_status.deleted_ids == []

    asyncio.run(_run())


def test_missing_source_file_is_identity_unknown_not_an_alias():
    """§8.3.E: ``None`` is not evidence of a different physical file (custom-ID /
    legacy / non-scan-origin rows carry no source identity)."""

    async def _run():
        rag = _ClassifyRag(
            {"legacy.txt": _unique("custom-id", DocStatus.PENDING)},
            rows={"custom-id": _row("custom-id", DocStatus.PENDING, {})},
        )
        decision = await classify_scan_file(rag, Path("legacy.txt"), "legacy.txt")
        assert decision.kind is _ScanFileClass.SOURCE_IDENTITY_UNKNOWN
        assert "records no source file" in decision.detail

    asyncio.run(_run())


def test_same_physical_source_resumes_and_a_different_one_is_an_alias():
    async def _run():
        rows = {
            "doc-x": _row(
                "doc-x", DocStatus.PARSING, {"source_file": "report.[native].txt"}
            )
        }
        rag = _ClassifyRag(
            {"report.txt": _unique("doc-x", DocStatus.PARSING)}, rows=rows
        )

        same = await classify_scan_file(rag, Path("report.[native].txt"), "report.txt")
        assert same.kind is _ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE

        alias = await classify_scan_file(rag, Path("report.txt"), "report.txt")
        assert alias.kind is _ScanFileClass.ALIAS_DUPLICATE
        assert "report.[native].txt" in alias.detail

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# End-to-end effects through run_scanning_process
# ---------------------------------------------------------------------------


def _scan_rig(tmp_path, monkeypatch, rag):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(scan_enqueue_batch_size=8)
    )
    doc_manager = DocumentManager(str(tmp_path))
    batched: list[Path] = []

    async def _capture_batch(_rag, candidates, _track_id):
        batched.extend(c.path for c in candidates)
        return len(candidates)

    monkeypatch.setattr(_document_routes, "pipeline_enqueue_scan_batch", _capture_batch)
    return doc_manager, batched


def test_source_conflict_keeps_the_file_and_never_enqueues(tmp_path, monkeypatch):
    """§8.3.B: no enqueue, no doc_status delete, and — unlike every other
    "skip" — NO archive: the operator needs the file where it is."""

    async def _run():
        rag = _ClassifyRag(
            {
                "shared.txt": SourceConflict(
                    candidate_count=None, sample_doc_ids=("doc-1", "doc-2")
                )
            }
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        conflicted = doc_manager.input_dir / "shared.txt"
        conflicted.write_text("body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-conflict")

        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert conflicted.exists()
        assert not (doc_manager.input_dir / PARSED_DIR_NAME).exists()

    asyncio.run(_run())


def test_alias_duplicate_is_archived_and_the_row_is_untouched(tmp_path, monkeypatch):
    """§8.3.G: a different physical file behind the same canonical key is
    archived rather than enqueued — enqueuing it would mint a ``dup-*`` row."""

    async def _run():
        rag = _ClassifyRag(
            {"report.txt": _unique("doc-x", DocStatus.PENDING)},
            rows={
                "doc-x": _row(
                    "doc-x", DocStatus.PENDING, {"source_file": "report.[native].txt"}
                )
            },
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        alias = doc_manager.input_dir / "report.txt"
        alias.write_text("alias body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-alias")

        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert not alias.exists()
        assert (doc_manager.input_dir / PARSED_DIR_NAME / "report.txt").exists()

    asyncio.run(_run())


def test_stale_stub_is_deleted_then_enqueued_as_new(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag(
            {"fixed.txt": _unique("doc-stub", DocStatus.FAILED)},
            rows={
                "doc-stub": _row(
                    "doc-stub", DocStatus.FAILED, {"source_file": "fixed.txt"}
                )
            },
            full_docs=_StrictFullDocs({}),
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        fixed = doc_manager.input_dir / "fixed.txt"
        fixed.write_text("fixed body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-stub")

        assert rag.doc_status.deleted_ids == ["doc-stub"]
        assert batched == [fixed]
        assert fixed.exists()  # enqueue is mocked; the file is not archived

    asyncio.run(_run())


def test_failed_stub_delete_is_counted_as_an_error_not_a_resume(tmp_path, monkeypatch):
    """A STALE_STUB whose ``doc_status`` delete FAILS is an error, nothing else.

    It used to also be tallied under ``resume_same_physical_source`` and added to
    the run summary's ``resuming`` count — reporting a document as being advanced
    when nothing can advance it: the row has no ``full_docs`` content, so the
    resume path can never move it. The file is simply left for the next scan."""

    async def _run():
        rag = _ClassifyRag(
            {"fixed.txt": _unique("doc-stub", DocStatus.FAILED)},
            rows={
                "doc-stub": _row(
                    "doc-stub", DocStatus.FAILED, {"source_file": "fixed.txt"}
                )
            },
            full_docs=_StrictFullDocs({}),
            delete_error=RuntimeError("doc_status delete boom"),
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        fixed = doc_manager.input_dir / "fixed.txt"
        fixed.write_text("fixed body", encoding="utf-8")

        reported: dict = {}

        class _Reporter:
            enabled = False

            def count(self, key, amount=1):
                reported[key] = reported.get(key, 0) + amount

            def sample(self, *_a, **_k):
                return None

            def flush(self):
                return None

            def renew(self):
                return None

            def finish(self, *_a, **_k):
                return None

        monkeypatch.setattr(
            _document_routes, "_ScanJobReporter", lambda *_a: _Reporter()
        )

        await run_scanning_process(rag, doc_manager, "track-stub-fail")

        # Not enqueued, not archived, row kept (delete failed) — and counted as
        # an error ONLY.
        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert fixed.exists()
        assert reported.get("errors") == 1
        assert _ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE.value not in reported
        assert _ScanFileClass.STALE_STUB.value not in reported

    asyncio.run(_run())


def test_identity_unknown_keeps_both_the_file_and_the_row(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag(
            {"legacy.txt": _unique("custom-id", DocStatus.PENDING)},
            rows={"custom-id": _row("custom-id", DocStatus.PENDING, {})},
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        legacy = doc_manager.input_dir / "legacy.txt"
        legacy.write_text("legacy body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-unknown")

        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert legacy.exists()
        assert not (doc_manager.input_dir / PARSED_DIR_NAME).exists()
        # The scan still ends with its single processing drive (§8.1).
        assert rag.process_calls == 1

    asyncio.run(_run())
