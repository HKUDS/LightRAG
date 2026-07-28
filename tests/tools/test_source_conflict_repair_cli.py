"""Offline source-conflict repair tool (LR2 Phase 4-f, §5.5).

Runs against a real ``JsonDocStatusStorage`` so the tool is exercised through the
same contract every backend implements: bounded listing, a dry-run that mutates
nothing, and a commit whose compare-and-set token comes from that dry-run rather
than from a fabricated placeholder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lightrag.base import SourceConflict, SourceUnique
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.tools.source_conflict_repair import (
    collect_source_conflicts,
    repair_one_conflict,
)

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


class _AnyContent:
    """full_docs double that confirms content for every id.

    A COMMIT requires one: an unverified primary is refused (the demotions are
    irreversible), so every ``apply=True`` call below passes a full_docs handle.
    The refusals themselves have their own tests.
    """

    supports_strict_point_reads = True

    def __init__(self, contents=None):
        self.contents = contents
        self.reads: list[str] = []

    async def get_by_id_strict(self, doc_id):
        self.reads.append(doc_id)
        if self.contents is None:
            return {"content": "body"}
        return self.contents.get(doc_id)


def _row(file_path: str, status: str = "pending") -> dict:
    return {
        "status": status,
        "content_summary": "summary",
        "content_length": 3,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "file_path": file_path,
    }


async def _storage(tmp_path, rows: dict) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="conflict-cli",
    )
    await storage.initialize()
    async with storage._storage_lock:
        storage._data.update(rows)
    return storage


@pytest.mark.asyncio
async def test_listing_reports_only_multi_primary_keys(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _row("a.pdf"),
            "doc-2": _row("a.pdf", "failed"),
            "solo": _row("b.pdf"),
        },
    )
    conflicts = await collect_source_conflicts(storage, limit=10)

    assert [c.canonical_source_key for c in conflicts] == ["a.pdf"]
    assert conflicts[0].candidate_count == 2


@pytest.mark.asyncio
async def test_listing_walks_every_page_when_asked(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _row("a.pdf"),
            "doc-2": _row("a.pdf"),
            "doc-3": _row("b.pdf"),
            "doc-4": _row("b.pdf"),
            "doc-5": _row("c.pdf"),
            "doc-6": _row("c.pdf"),
        },
    )
    first_page = await collect_source_conflicts(storage, limit=2)
    assert len(first_page) == 2

    every = await collect_source_conflicts(storage, limit=2, all_pages=True)
    assert [c.canonical_source_key for c in every] == ["a.pdf", "b.pdf", "c.pdf"]


@pytest.mark.asyncio
async def test_dry_run_mutates_nothing(tmp_path):
    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    result = await repair_one_conflict(storage, "a.pdf", "doc-2")

    assert result.committed is False
    assert result.candidate_count == 2
    assert result.demoted_sample_doc_ids == ("doc-1",)
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)
    assert (await storage.get_by_id("doc-1")).get("metadata") is None


@pytest.mark.asyncio
async def test_apply_commits_with_the_token_from_its_own_dry_run(tmp_path):
    """The tool automates the copy-paste, not the guard: the commit still carries
    a count/fingerprint it just read, so the backend can refuse a moved set."""
    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    seen: list[dict] = []
    original = storage.repair_source_conflict

    async def _recording(key, **kwargs):
        seen.append(kwargs)
        return await original(key, **kwargs)

    storage.repair_source_conflict = _recording

    result = await repair_one_conflict(
        storage, "a.pdf", "doc-2", full_docs=_AnyContent(), apply=True
    )

    assert result.committed is True
    assert [call["dry_run"] for call in seen] == [True, False]
    # The commit echoed the dry-run's token rather than a placeholder.
    assert seen[1]["expected_candidate_count"] == 2
    assert seen[1]["expected_candidate_fingerprint"] == result.fingerprint

    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
    demoted = await storage.get_by_id("doc-1")
    assert demoted["metadata"]["is_duplicate"] is True
    assert demoted["metadata"]["original_doc_id"] == "doc-2"
    # Content and status survive: only the source claim was dropped.
    assert demoted["status"] == "pending"
    assert demoted["content_summary"] == "summary"


@pytest.mark.asyncio
async def test_unknown_primary_is_refused_before_any_write(tmp_path):
    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    with pytest.raises(ValueError, match="not a current primary candidate"):
        await repair_one_conflict(
            storage, "a.pdf", "doc-typo", full_docs=_AnyContent(), apply=True
        )

    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_commit_holds_the_enqueue_serialize_lock(tmp_path):
    """Fix-proof for the contract gap: no backend can block a NEW primary being
    inserted between the repair's re-read and its demotions, so the commit must
    hold the SAME workspace lock the enqueue critical section holds. Without it,
    an enqueue could interleave and the repair would still report committed=True
    on a key that is still in conflict.
    """
    import asyncio

    from lightrag.constants import ENQUEUE_SERIALIZE_LOCK_NAMESPACE
    from lightrag.kg.shared_storage import get_namespace_lock

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    workspace = "enqueue-ws"
    enqueue_ran: list[str] = []

    # Stand in for the enqueue critical section (filter_keys → dedup → upsert).
    async def _enqueue_critical_section():
        async with get_namespace_lock(
            ENQUEUE_SERIALIZE_LOCK_NAMESPACE, workspace=workspace
        ):
            enqueue_ran.append("enqueue")

    original = storage.repair_source_conflict
    observed_during_repair: list[list[str]] = []

    async def _observing(key, **kwargs):
        # Sampled inside the backend call — precisely the re-read → demote span
        # the CAS cannot protect.
        if not kwargs.get("dry_run", True):
            await asyncio.sleep(0)
            observed_during_repair.append(list(enqueue_ran))
        return await original(key, **kwargs)

    storage.repair_source_conflict = _observing

    async with get_namespace_lock(
        ENQUEUE_SERIALIZE_LOCK_NAMESPACE, workspace=workspace
    ):
        # The enqueue lock is held, so a correctly-locked commit cannot start.
        commit = asyncio.create_task(
            repair_one_conflict(
                storage,
                "a.pdf",
                "doc-2",
                workspace=workspace,
                full_docs=_AnyContent(),
                apply=True,
            )
        )
        await asyncio.sleep(0.05)
        assert not commit.done(), (
            "the commit ran while the enqueue-serialize lock was held — the "
            "phantom-insert window is still open"
        )

    result = await commit
    assert result.committed is True
    # And an enqueue cannot slip inside the backend's re-read → demote span.
    await _enqueue_critical_section()
    assert observed_during_repair == [[]]
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"


@pytest.mark.asyncio
async def test_the_commit_registers_itself_as_in_flight_ingress(tmp_path):
    """The repair holds a weight-0 pending-enqueue reservation for its duration.

    That single fact is what excludes the writers the enqueue-serialize lock
    cannot: ``_acquire_destructive_busy`` and the scan reservation both already
    list ``pending_enqueues`` in their own reject_when, so a clear/delete or a
    scan cannot start while a repair is mid-commit — no new lock and no new
    lock-order edge. Weight 0 so the repair never consumes admission capacity.
    """
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        initialize_pipeline_status,
    )

    workspace = "reservation-ws"
    await initialize_pipeline_status(workspace=workspace)
    pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    seen: list[dict] = []
    original = storage.repair_source_conflict

    async def _observing(key, **kwargs):
        # Sampled inside the backend call: the span a concurrent delete would
        # otherwise be free to interleave with.
        seen.append(
            {
                "pending": pipeline_status.get("pending_enqueues", 0),
                "weights": [
                    meta.get("weight")
                    for meta in dict(
                        pipeline_status.get("pending_enqueue_tokens", {})
                    ).values()
                ],
            }
        )
        return await original(key, **kwargs)

    storage.repair_source_conflict = _observing
    result = await repair_one_conflict(
        storage,
        "a.pdf",
        "doc-1",
        workspace=workspace,
        full_docs=_AnyContent(),
        apply=True,
    )
    assert result.committed is True

    assert seen and all(sample["pending"] == 1 for sample in seen), seen
    assert all(sample["weights"] == [0] for sample in seen), seen
    # Released afterwards, so it cannot wedge later uploads / scans / deletes.
    assert pipeline_status.get("pending_enqueues", 0) == 0
    assert dict(pipeline_status.get("pending_enqueue_tokens", {})) == {}


@pytest.mark.asyncio
async def test_a_destructive_job_in_flight_refuses_the_commit(tmp_path):
    """The other direction of the same fence: a clear/delete already running can
    DELETE the primary the operator is about to keep, so the repair refuses
    rather than committing demotions around it."""
    from lightrag.exceptions import StorageControlPlaneError
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
        initialize_pipeline_status,
    )

    workspace = "destructive-ws"
    await initialize_pipeline_status(workspace=workspace)
    pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)
    lock = get_namespace_lock("pipeline_status", workspace=workspace)
    async with lock:
        pipeline_status["destructive_busy"] = True

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    with pytest.raises(StorageControlPlaneError, match="clear/delete"):
        await repair_one_conflict(
            storage,
            "a.pdf",
            "doc-1",
            workspace=workspace,
            full_docs=_AnyContent(),
            apply=True,
        )
    # Nothing demoted.
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_scan_classification_and_a_manual_freeze_also_refuse(tmp_path):
    """``scanning_exclusive`` deletes stale FAILED stubs and the manual exclusive
    reset rewrites ``file_path`` (so a placeholder row can ENTER this candidate
    set without passing the enqueue critical section). Both are refused for the
    repair's duration."""
    from lightrag.exceptions import StorageControlPlaneError
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
        initialize_pipeline_status,
    )

    for flag, expected in (
        ("scanning_exclusive", "scan is classifying"),
        ("manual_freeze_requested", "manual retry"),
    ):
        workspace = f"fence-{flag}"
        await initialize_pipeline_status(workspace=workspace)
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        async with get_namespace_lock("pipeline_status", workspace=workspace):
            pipeline_status[flag] = True

        storage = await _storage(
            tmp_path / flag, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
        )
        with pytest.raises(StorageControlPlaneError, match=expected):
            await repair_one_conflict(
                storage,
                "a.pdf",
                "doc-1",
                workspace=workspace,
                full_docs=_AnyContent(),
                apply=True,
            )


@pytest.mark.asyncio
async def test_an_offline_repair_proceeds_but_says_it_is_unguarded(tmp_path, caplog):
    """The CLI runs with no pipeline_status of its own, so the caller-side
    exclusion is inert. Refusing would break the stopped-server case — the CLI's
    whole purpose — so it proceeds, but it must SAY so.

    An uninitialised pipeline_status proves only that THIS process has none; a
    server may be running in another process whose shared state this one cannot
    see. Inferring "no server" from a local miss is the same mistake as inferring
    "no duplicate" from a swallowed read.
    """
    import logging

    from lightrag.utils import logger

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            result = await repair_one_conflict(
                storage,
                "a.pdf",
                "doc-1",
                workspace="never-initialised-ws",
                full_docs=_AnyContent(),
                apply=True,
            )
    finally:
        logger.propagate = False

    assert result.committed is True
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-1"

    assert "without pipeline exclusion" in caplog.text
    assert "ANOTHER process" in caplog.text  # the boundary, not just "no server"
    assert "source_conflicts/repair" in caplog.text  # names the guarded route


@pytest.mark.asyncio
async def test_a_guarded_repair_does_not_warn(tmp_path, caplog):
    """The complement: with a real pipeline_status the exclusion is live, so the
    caveat must not fire — a warning on every in-server repair would train
    operators to ignore it."""
    import logging

    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import logger

    workspace = "guarded-ws"
    await initialize_pipeline_status(workspace=workspace)
    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            await repair_one_conflict(
                storage,
                "a.pdf",
                "doc-1",
                workspace=workspace,
                full_docs=_AnyContent(),
                apply=True,
            )
    finally:
        logger.propagate = False
    assert "without pipeline exclusion" not in caplog.text


@pytest.mark.asyncio
async def test_a_contentless_primary_is_refused_offline_too(tmp_path):
    """Same refusal on the CLI path, when it is given full_docs to check with."""
    from lightrag.exceptions import SourceConflictPrimaryUnusableError

    class _FullDocs:
        supports_strict_point_reads = True

        def __init__(self, contents):
            self.contents = contents

        async def get_by_id_strict(self, doc_id):
            return self.contents.get(doc_id)

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    with pytest.raises(SourceConflictPrimaryUnusableError, match="doc-2"):
        await repair_one_conflict(
            storage,
            "a.pdf",
            "doc-2",
            workspace="content-ws",
            full_docs=_FullDocs({"doc-1": {"content": "x"}}),
            apply=True,
        )
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_an_unverifiable_primary_refuses_instead_of_committing(tmp_path):
    """Fix-proof: a strict read that RAISED warned and committed the demotions
    anyway. The demotions are irreversible — a repair only demotes, and a key with
    no candidate left cannot be repaired at all — so an unverified primary must
    fail closed as a storage-control-plane error (503 on the endpoint) and leave
    the conflict exactly as it was, for the operator to retry."""
    from lightrag.exceptions import StorageControlPlaneError

    class _UnreachableFullDocs:
        supports_strict_point_reads = True

        async def get_by_id_strict(self, doc_id):
            raise ConnectionError("full_docs is down")

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    with pytest.raises(StorageControlPlaneError, match="doc-2"):
        await repair_one_conflict(
            storage,
            "a.pdf",
            "doc-2",
            workspace="unverifiable-ws",
            full_docs=_UnreachableFullDocs(),
            apply=True,
        )
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_the_cli_refuses_to_apply_when_full_docs_cannot_be_opened(
    tmp_path, monkeypatch, capsys
):
    """The CLI's own fail-open: ``full_docs.initialize()`` failing used to print a
    warning and run the commit with the contentless-primary check disabled — the
    same irreversible demotion on an unverified primary, one layer up. Listing and
    dry-runs still work, since they modify nothing."""
    import lightrag
    from lightrag.tools import source_conflict_repair as tool

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )
    committed: list[bool] = []
    real_repair = storage.repair_source_conflict

    async def _recording_repair(*args, **kwargs):
        committed.append(not kwargs.get("dry_run", True))
        return await real_repair(*args, **kwargs)

    storage.repair_source_conflict = _recording_repair

    class _BrokenFullDocs:
        async def initialize(self):
            raise ConnectionError("full_docs socket refused")

    class _FakeRAG:
        def __init__(self, **kwargs):
            self.workspace = kwargs.get("workspace") or "cli-refuse-ws"
            self.doc_status = storage
            self.full_docs = _BrokenFullDocs()

    async def _keep_the_storage_open():
        """The tool finalizes doc_status on its way out; this test still has to
        read the conflict afterwards to prove nothing was demoted."""

    monkeypatch.setattr(lightrag, "LightRAG", _FakeRAG)
    monkeypatch.setattr(storage, "finalize", _keep_the_storage_open, raising=False)

    args = SimpleNamespace(
        command="repair",
        workspace="cli-refuse-ws",
        source="a.pdf",
        primary="doc-2",
        apply=True,
    )
    assert await tool._async_main(args) is False
    assert committed == []  # nothing reached the backend, not even a dry-run
    assert "Refused" in capsys.readouterr().out
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_a_commit_without_full_docs_is_refused(tmp_path):
    """Fix-proof: omitting ``full_docs`` used to skip the contentless-primary
    check and commit anyway, so a library caller (and the CLI when full_docs
    could not be opened) could hand a canonical source to an unprocessable stub
    irreversibly. A commit now requires the handle; dry-runs still do not."""
    from lightrag.exceptions import StorageControlPlaneError

    storage = await _storage(
        tmp_path, {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    )

    # Dry-run: mutates nothing, so it needs no verification.
    assert (await repair_one_conflict(storage, "a.pdf", "doc-2")).committed is False

    with pytest.raises(StorageControlPlaneError, match="no full_docs handle"):
        await repair_one_conflict(
            storage, "a.pdf", "doc-2", workspace="no-full-docs-ws", apply=True
        )
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_a_primary_that_duplicates_another_source_is_refused_offline_too(
    tmp_path,
):
    """The processing stage holds no source-key lock, so a primary whose content
    already lives under a different source is refused BEFORE the irreversible
    demotions — the same shape as the contentless-stub refusal, on the CLI path."""
    from lightrag.exceptions import SourceConflictPrimaryUnusableError

    rows = {
        "doc-1": _row("a.pdf"),
        "doc-2": _row("a.pdf", "failed"),
        "doc-elsewhere": _row("other.pdf", "processed"),
    }
    rows["doc-2"]["content_hash"] = "hash-x"
    rows["doc-elsewhere"]["content_hash"] = "hash-x"
    storage = await _storage(tmp_path, rows)

    with pytest.raises(SourceConflictPrimaryUnusableError, match="doc-elsewhere"):
        await repair_one_conflict(
            storage,
            "a.pdf",
            "doc-2",
            workspace="twin-ws",
            full_docs=_AnyContent(),
            apply=True,
        )
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)


@pytest.mark.asyncio
async def test_a_document_already_marked_duplicate_is_refused_before_any_demotion(
    tmp_path,
):
    """The other linearization the source-key lock establishes (stability test —
    this direction was already refused; what is new is that the lock guarantees
    the two operations cannot overlap, so this is the ONLY thing the repair can
    see once the marking has won).

    Marking wins → the primary is no longer a candidate → the commit refuses
    before demoting anything, instead of writing irreversible demotions and then
    reporting a 503 about storage that was fine.
    """
    rows = {"doc-1": _row("a.pdf"), "doc-2": _row("a.pdf", "failed")}
    # doc-2 lost its claim the way _mark_duplicate_after_parse takes it away.
    rows["doc-2"]["metadata"] = {
        "is_duplicate": True,
        "duplicate_kind": "content_hash",
        "original_doc_id": "doc-elsewhere",
    }
    storage = await _storage(tmp_path, rows)

    with pytest.raises(ValueError, match="not a current primary candidate"):
        await repair_one_conflict(
            storage,
            "a.pdf",
            "doc-2",
            workspace="linearized-ws",
            full_docs=_AnyContent(),
            apply=True,
        )
    # doc-1 keeps its claim: nothing was demoted on the way to the refusal.
    assert (await storage.get_by_id("doc-1")).get("metadata") in (None, {})


@pytest.mark.asyncio
async def test_the_absent_recovery_hint_matches_what_actually_happened(tmp_path):
    """Fix-proof: the hint said "delete the leftover row and re-upload to claim the
    source again", which does not work while the content's new holder still exists
    — a doc id is derived from the file name, so the re-upload is deduplicated
    against that holder and the key stays Absent. The hint now reads the row and
    names the real condition."""
    from lightrag.exceptions import StorageControlPlaneError
    from lightrag.tools.source_conflict_repair import verify_repair_outcome

    rows = {"doc-keep": _row("a.pdf", "failed")}
    rows["doc-keep"]["metadata"] = {
        "is_duplicate": True,
        "duplicate_kind": "content_hash",
        "original_doc_id": "doc-holder",
    }
    storage = await _storage(tmp_path, rows)
    result = SimpleNamespace(demoted_sample_doc_ids=("doc-lose",))

    with pytest.raises(StorageControlPlaneError) as raised:
        await verify_repair_outcome(storage, "a.pdf", "doc-keep", result)

    message = str(raised.value)
    assert "doc-holder" in message
    assert "NOT enough" in message  # deleting the leftover row does not suffice
    assert "cannot be re-run" in message

    # The complement: a primary that simply vanished IS recoverable by re-ingest.
    async with storage._storage_lock:
        del storage._data["doc-keep"]
    with pytest.raises(StorageControlPlaneError) as gone:
        await verify_repair_outcome(storage, "a.pdf", "doc-keep", result)
    assert "re-ingest the file" in str(gone.value)
