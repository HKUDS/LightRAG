#!/usr/bin/env python3
"""Offline source-conflict listing / repair (LR2 §5.5).

Two or more *primary* documents can end up claiming the same canonical source
key (a hint-stripped basename): custom-ID inserts, legacy ids, and historical
basename collisions all predate the typed source resolver. A scan that meets one
of these classifies the file ``source_conflict`` and refuses to act — it does not
enqueue, delete or archive anything, because picking a winner automatically could
silently retire the document an operator actually wanted to keep.

This tool is the offline half of the explicit repair flow (the online half is
``GET /documents/source_conflicts`` + ``POST /documents/source_conflicts/repair``).
It never chooses a winner: the operator names the document that keeps the source,
every other candidate is marked ``metadata.is_duplicate=true`` with
``original_doc_id=<primary>``, and **no content is ever deleted** — the demoted
rows keep their status and their ``full_docs`` entry, they only lose their claim
on the canonical source.

"Offline" means OUT OF BAND — not through the HTTP API — and NOT "the server must
be stopped". Which matters, because the protection you get depends on where the
call runs, and only one of the two callers gets all of it:

===========================  =============================================
caller                       caller-side exclusion
===========================  =============================================
``POST …/repair`` (online)   real: the endpoint runs inside a server worker,
  or a library call from       so the keyed lock, the enqueue-serialize lock
  inside the server            and the pending-enqueue reservation all live
                               in the shared state every worker sees
standalone CLI               NONE of it: ``initialize_share_data(workers=1)``
                               builds process-LOCAL asyncio locks and a local
                               ``_shared_dicts``, and this process never
                               initialises ``pipeline_status``, so the locks
                               exclude only this process and the reservation
                               is a no-op
===========================  =============================================

Cross-process exclusion is not reachable from here: it would need the server's
Manager (its address and authkey are private to the server) or a persisted
liveness marker, which is the storage-verifiable fencing token the design
excludes. So:

**Run ``--apply`` while the server is stopped or quiesced, or use the HTTP
endpoint instead.** A CLI commit against a live server is not silently unsafe —
it is guarded by the CAS and by the post-commit re-resolve, so a concurrent
change makes the repair FAIL LOUDLY rather than corrupt the key — but "fails
loudly" is the guarantee, not "cannot interleave".

Safety model:

- listing and ``repair`` without ``--apply`` mutate nothing;
- ``--apply`` re-reads the candidate set and commits only when it still matches
  the dry-run it just performed (compare-and-set), so a concurrent
  enqueue/delete fails the repair instead of being overwritten. This is the one
  guarantee that holds for BOTH callers, because it lives in the storage;
- ``verify_repair_outcome`` re-resolves the key after the commit and raises
  unless it is now unique on the chosen primary — the backstop that turns any
  interleaving the caller-side locks did not cover into a reported failure;
- the caller-side locks (per the table above) additionally exclude every
  in-deployment writer of this candidate set for the whole commit — a NEW primary
  being INSERTED (enqueue-serialize lock), clear/delete + scan classification +
  manual reset (the reservation), and the processing stage's duplicate marking
  (the keyed source lock, which ``_mark_duplicate_after_parse`` takes too, LR2
  §5.5) — for in-server callers only;
- a repair interrupted half-way needs no reconciliation: the finished rows
  already express ``duplicate`` and the rest still resolve as a conflict, so
  simply run it again.

``doc_status`` and ``full_docs`` are opened — no vector store, graph store, LLM
or embedding model is touched.

Usage (CLI — honors WORKING_DIR / WORKSPACE and the LIGHTRAG_* storage env vars
from ``.env``, same as the server)::

    python -m lightrag.tools.source_conflict_repair list [--limit 50] [--all]
    python -m lightrag.tools.source_conflict_repair repair \\
        --source report.docx --primary doc-abc123 [--apply]

Usage (library — for a deployment that builds its own LightRAG)::

    conflicts = await collect_source_conflicts(rag.doc_status, limit=50)
    result = await repair_one_conflict(
        rag.doc_status,
        "report.docx",
        "doc-abc123",
        workspace=rag.workspace,
        full_docs=rag.full_docs,  # required for a commit
        apply=True,
    )
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from lightrag.base import CURSOR_END, CURSOR_START, SourceConflictSummary
from lightrag.constants import ENQUEUE_SERIALIZE_LOCK_NAMESPACE
from lightrag.exceptions import (
    SourceConflictPrimaryUnusableError,
    SourceConflictRepairCASError,
    StorageCapabilityError,
    StorageControlPlaneError,
)
from lightrag.utils import logger
from lightrag.utils_pipeline import (
    doc_status_field,
    get_duplicate_doc_by_content_hash,
    source_candidate_set_lock,
)

# Pages are bounded on both sides: the backend projects a bounded sample per
# key, and --all stops after this many pages so a pathological workspace cannot
# turn a listing into an unbounded walk.
_MAX_PAGES = 1000


# Pipeline states that must not overlap a source-conflict COMMIT, evaluated when
# the repair registers its pending-enqueue reservation. Each one owns a writer
# that can change the candidate set the repair just re-read, and none of them can
# be excluded by the enqueue-serialize lock:
#
# * ``destructive_busy`` — a clear/delete can DELETE the primary the operator
#   chose to keep, leaving the key with no primary and the demoted rows pointing
#   at an id that no longer exists (and the repair only demotes, so there is no
#   supported way back);
# * ``scanning_exclusive`` — scan classification deletes stale FAILED stubs, one
#   of which could be that primary;
# * ``manual_freeze_requested`` — the exclusive FAILED reset rewrites
#   ``file_path`` from ``resolve_doc_file_path``, so a row whose doc_status path
#   is a placeholder can ENTER this candidate set without ever passing the
#   enqueue critical section.
#
# Symmetry is what makes this cheap: the scan and destructive acquires already
# list ``pending_enqueues`` in their own reject_when, so holding a reservation
# for the repair's duration buys the other direction for free — no new lock, no
# new lock-order edge.
_REPAIR_INGRESS_FENCES: tuple[tuple[str, str], ...] = (
    (
        "destructive_busy",
        "A clear/delete job is in flight; it may remove one of these documents. "
        "Wait for it to finish, then repair.",
    ),
    (
        "scanning_exclusive",
        "A scan is classifying files and may delete stale failed stubs. Wait for "
        "the classification phase to finish, then repair.",
    ),
    (
        "manual_freeze_requested",
        "A manual retry is draining the pipeline for its exclusive reset, which "
        "can change which documents claim this source. Wait for it to finish, "
        "then repair.",
    ),
)


@asynccontextmanager
async def _repair_ingress_reservation(workspace: str):
    """Register the repair as in-flight ingress work for its whole duration.

    A weighted-0 entry in ``pending_enqueue_tokens``: it charges nothing against
    the admission capacity but makes ``pending_enqueues`` non-zero, which is what
    the scan and destructive acquires already refuse on. The fences above are
    evaluated in the same atomic step, so the exclusion holds in both directions.

    An uninitialised ``pipeline_status`` does NOT prove there is no server: it
    proves only that THIS process has none, and a standalone CLI never
    initialises one even when a server is running elsewhere (its shared state is
    a different process's). So the repair proceeds — refusing would break the
    stopped-server case, which is the CLI's whole purpose — but it says so, at
    the moment it actually applies, rather than leaving the operator to infer it
    from the module docstring. The CAS and the post-commit re-resolve are what
    keep that case honest.
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import (
        acquire_enqueue_reservation,
        get_namespace_data,
        get_namespace_lock,
        release_token_set_reservation,
    )

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=workspace
        )
    except PipelineNotInitializedError:
        logger.warning(
            "Source-conflict repair is committing without pipeline exclusion: "
            "this process has no pipeline_status, which means either no server "
            "is running OR one is running in ANOTHER process whose shared state "
            "this one cannot see. Concurrent clear/delete, scan classification "
            "or a manual retry are therefore NOT excluded — the compare-and-set "
            "and the post-commit re-resolve will fail the repair loudly if one "
            "interleaves. Prefer POST /documents/source_conflicts/repair while a "
            "server is running."
        )
        yield
        return

    pipeline_status_lock = get_namespace_lock("pipeline_status", workspace=workspace)
    token = f"source-repair-{uuid.uuid4().hex}"
    result = await acquire_enqueue_reservation(
        pipeline_status,
        pipeline_status_lock,
        token=token,
        reject_when=_REPAIR_INGRESS_FENCES,
        weight=0,
        capacity=0,
    )
    if not result.acquired:
        raise StorageControlPlaneError(result.message)
    try:
        yield
    finally:
        await release_token_set_reservation(
            workspace,
            tokens_key="pending_enqueue_tokens",
            token=token,
        )


@asynccontextmanager
async def source_conflict_repair_lock(workspace: str, canonical_source_key: str):
    """Hold the caller-side locks a source-conflict COMMIT requires.

    ``repair_source_conflict`` re-reads the candidate set and demotes the losers;
    no backend can stop a new primary being INSERTED in between (``FOR UPDATE``
    locks only the rows it saw, a snapshot transaction never sees a phantom, and
    Redis/OpenSearch have no transaction at all), so excluding that insert is the
    caller's job — see ``DocStatusStorage.repair_source_conflict``.

    Three things, in this TOTAL order — outermost first:

    1. the workspace's enqueue-serialize lock — the one that excludes the
       phantom, because it is the same lock the enqueue critical section
       (``filter_keys`` → dedup → ``doc_status.upsert``) holds;
    2. a keyed lock on the canonical source key (:func:`source_candidate_set_lock`)
       — serializes concurrent repairs of the SAME key without blocking other
       keys, and, since LR2 §5.5 requires exclusion against every writer that can
       change this key's candidate set, it is the SAME lock the processing stage's
       duplicate marking now takes (``_mark_duplicate_after_parse``);
    3. a pending-enqueue reservation (see :func:`_repair_ingress_reservation`),
       which excludes the writers the enqueue-serialize lock cannot: clear/delete,
       scan classification, and the manual exclusive reset.

    This order used to be keyed → enqueue_serialize, with a note that anyone
    extending the keyed ``DocSource`` lock to another writer MUST flip it first,
    because it was INVERTED relative to the enqueue path (which holds
    enqueue_serialize and would naturally reach for a keyed source lock inside
    it). Extending it to the duplicate-marking path is exactly that change, so
    the flip happened with it: ``enqueue_serialize → DocSource → pipeline_status``
    is now the total order every holder of more than one agrees on, and it is the
    order the enqueue path could adopt without a cycle. The marking path takes
    only (2) and, inside it, ``pipeline_status`` for its owner check — same
    direction, no cycle.

    Every one of these is PROCESS-LOCAL machinery, so the scope is one
    deployment — its worker processes included, because they share the Manager
    the master created — and nothing beyond it. Two consequences, the second of
    which is easy to miss:

    * a repair run against a database another deployment is writing to keeps the
      phantom window;
    * **the standalone CLI is such an "other deployment"**. It calls
      ``initialize_share_data(workers=1)``, so these locks are asyncio objects in
      its own process and the reservation finds no ``pipeline_status`` at all.
      Everything here holds for the HTTP endpoint and for a library call inside a
      server worker; for the CLI it is inert, and the CAS plus
      :func:`verify_repair_outcome` are the whole guarantee (see the module
      docstring).
    """
    # Imported lazily: the CLI initializes shared storage in main(), and
    # importing at module scope would bind before that happens.
    from lightrag.kg.shared_storage import get_namespace_lock

    async with get_namespace_lock(
        ENQUEUE_SERIALIZE_LOCK_NAMESPACE, workspace=workspace
    ):
        async with source_candidate_set_lock(workspace, canonical_source_key):
            async with _repair_ingress_reservation(workspace):
                yield


async def collect_source_conflicts(
    doc_status: Any,
    *,
    limit: int = 50,
    all_pages: bool = False,
) -> list[SourceConflictSummary]:
    """Collect one page of conflicts, or every page when ``all_pages``.

    ``limit`` is the page size; the returned list is bounded by the number of
    conflicting keys, which is an operator-scale quantity (each one needs a
    human decision) — not by the document count.
    """
    collected: list[SourceConflictSummary] = []
    position = CURSOR_START
    for _ in range(_MAX_PAGES):
        page = await doc_status.list_source_conflicts_page(
            limit=limit, position=position
        )
        collected.extend(page.conflicts)
        if not all_pages or page.next_position is CURSOR_END:
            break
        position = page.next_position
    return collected


async def repair_one_conflict(
    doc_status: Any,
    canonical_source_key: str,
    primary_doc_id: str,
    *,
    workspace: str = "",
    full_docs: Any = None,
    apply: bool = False,
) -> Any:
    """Dry-run one repair and, with ``apply``, commit it with that fresh token.

    The commit echoes the count/fingerprint the dry-run just returned, so the
    backend still fails closed if the candidate set changes in between — the CAS
    is not bypassed by automating the two steps, only the copy-paste is.

    The commit additionally runs under the caller-side locks the storage contract
    requires: a keyed lock on the canonical source key (serializing concurrent
    repairs of the same key) plus the workspace's enqueue-serialize lock, which
    is what actually excludes a new primary being INSERTED between the backend's
    re-read and its demotions — no backend can block that phantom on its own. A
    phantom that lands BEFORE the commit is already caught by the CAS (a new
    primary changes both the count and the fingerprint); the locks close the one
    window the CAS cannot see, inside the backend call itself.
    ``workspace`` must be the LightRAG instance's workspace (``rag.workspace``),
    NOT ``doc_status.workspace``: a backend may override its own workspace from
    the environment, and a lock keyed on the wrong one excludes nothing.

    ``full_docs`` is REQUIRED for a commit (``apply=True``): it is what the
    contentless-primary refusal reads, and an unverified primary is refused with
    :class:`StorageControlPlaneError` rather than granted the source — see
    :func:`refuse_a_contentless_primary` for why the asymmetry is not close. It
    cannot live in the backend: a doc_status storage has no handle on full_docs.
    Dry-runs (and ``collect_source_conflicts``) mutate nothing and need none.
    """
    if not apply:
        return await doc_status.repair_source_conflict(
            canonical_source_key,
            primary_doc_id=primary_doc_id,
            # Ignored in dry-run mode by contract.
            expected_candidate_count=0,
            expected_candidate_fingerprint="",
            dry_run=True,
        )
    # The dry-run stays INSIDE the lock, but its token is never refreshed: the
    # CAS must still be able to refuse. A candidate set that moved before the
    # lock was taken SHOULD 409 rather than be silently accepted.
    async with source_conflict_repair_lock(workspace, canonical_source_key):
        # Inside the lock: the row could have lost its content — or gained a
        # content-hash twin — between the operator reading the listing and this
        # commit.
        await refuse_an_unusable_primary(
            doc_status, full_docs, canonical_source_key, primary_doc_id
        )
        dry = await doc_status.repair_source_conflict(
            canonical_source_key,
            primary_doc_id=primary_doc_id,
            expected_candidate_count=0,
            expected_candidate_fingerprint="",
            dry_run=True,
        )
        result = await doc_status.repair_source_conflict(
            canonical_source_key,
            primary_doc_id=primary_doc_id,
            expected_candidate_count=dry.candidate_count,
            expected_candidate_fingerprint=dry.fingerprint,
            dry_run=False,
        )
        # Verify the OUTCOME, still inside the locks, instead of inferring it
        # from "the demotions were committed". ``committed=True`` only claims
        # that the demotions named in the result landed; the key being unique is a
        # separate claim, so it is checked rather than assumed.
        #
        # Every in-deployment writer of this candidate set is now excluded for the
        # whole span: enqueue by the enqueue-serialize lock, clear/delete + scan
        # classification + manual reset by the reservation, and the processing
        # stage's duplicate marking by the keyed source lock it now takes too
        # (LR2 §5.5). So this verification is a backstop, not the primary guard —
        # which is what it must be, because two things remain outside every lock
        # here: the standalone CLI (its locks are process-local and it has no
        # pipeline_status at all) and any second deployment writing the same
        # database.
        await verify_repair_outcome(
            doc_status, canonical_source_key, primary_doc_id, result
        )
        return result


async def refuse_a_contentless_primary(full_docs: Any, primary_doc_id: str) -> None:
    """Refuse to hand a canonical source to a row with no ``full_docs`` content.

    Such a row is an unprocessable stub, and keeping it is self-defeating twice
    over. It can never be processed, so the source key ends up owned by a
    document that will never exist; and it is exactly what scan classification
    deletes on sight (``_ScanFileClass.STALE_STUB`` fires only when
    ``_confirm_full_docs_absent`` is True), so the next scan would delete the
    primary and leave the demoted rows pointing at a missing id — with no way
    back, since a repair only demotes and refuses a ``primary_doc_id`` that is
    not a current candidate.

    Checking here is what removes that interaction at the root, and it is
    strictly better than locking the scan's delete path against the repair: it
    costs one strict read on an operator-invoked path instead of a lock on scan
    classification, and it also catches the plain operator mistake of choosing a
    stub when no scan is running at all.

    A CONFIRMED absence refuses with
    :class:`SourceConflictPrimaryUnusableError` (409 — pick another primary).
    EVERY other answer than "it has content" refuses with
    :class:`StorageControlPlaneError` (503 — retry once storage is back): a read
    that failed, a backend with no strict point reads, and no ``full_docs`` handle
    at all. Unverified is not the same as verified, and the asymmetry is not
    close: the demotions are irreversible (a repair only demotes, and a key with
    no candidate left cannot be repaired again), while a 503 costs the operator a
    retry of an operation that has no deadline — the conflict has been sitting
    there since before they looked at it.

    The no-strict-reads branch used to warn and proceed, reasoning that scan
    classification cannot delete the stub on such a backend either
    (``_confirm_full_docs_absent`` returns ``None`` for the same reason, so the
    STALE_STUB exit never fires). That covers only ONE of the two harms: a stub
    can never be processed, so handing it the canonical source leaves the key
    owned by a document that will never exist, whatever the scan does. Every
    in-tree KV backend declares ``supports_strict_point_reads = True``, so this
    only refuses a third-party backend that cannot answer the question.
    """
    if full_docs is None:
        raise StorageControlPlaneError(
            f"Cannot confirm whether {primary_doc_id} has full_docs content: no "
            f"full_docs handle was provided. A source-conflict COMMIT requires "
            f"one — its demotions are irreversible, so an unverified primary is "
            f"refused rather than granted the source."
        )
    if not getattr(full_docs, "supports_strict_point_reads", False):
        raise StorageControlPlaneError(
            f"{type(full_docs).__name__} has no strict point reads, so it cannot "
            f"confirm whether {primary_doc_id} has content; refusing to commit "
            f"irreversible demotions on an unverified primary. A contentless stub "
            f"can never be processed, so it would own the source forever."
        )
    try:
        content = await full_docs.get_by_id_strict(primary_doc_id)
    except Exception as read_error:
        raise StorageControlPlaneError(
            f"Could not confirm whether {primary_doc_id} has full_docs content "
            f"({read_error}); refusing to commit irreversible demotions on an "
            f"unverified primary. Retry once full_docs is reachable."
        ) from read_error
    if content:
        return
    raise SourceConflictPrimaryUnusableError(
        f"Document {primary_doc_id} has no full_docs content: it is an "
        f"unprocessable stub, so it cannot own a canonical source. Pick a "
        f"primary that has content — a scan would delete this row and leave the "
        f"demoted documents pointing at an id that no longer exists.",
        reason=SourceConflictPrimaryUnusableError.REASON_NO_CONTENT,
    )


async def refuse_a_primary_whose_content_lives_elsewhere(
    doc_status: Any, canonical_source_key: str, primary_doc_id: str
) -> None:
    """Refuse a primary the PROCESSING stage is already destined to demote.

    The commit DOES exclude the processing stage for its whole span — the keyed
    source lock it holds is the same one ``_mark_duplicate_after_parse`` takes
    (LR2 §5.5) — but exclusion only ORDERS the two operations; it cannot keep a
    primary usable after it releases. A marking that was already destined to
    demote this row simply lands the moment the lock is free, and the key the
    operator was just told is settled has no primary again. So the lock and this
    refusal answer different questions, and this one belongs where the
    contentless-stub refusal already sits: BEFORE the irreversible demotions, by
    refusing a primary that cannot keep the source.

    A primary whose content hash is already held by a document under a DIFFERENT
    canonical source is exactly that: when it is parsed, the post-parse duplicate
    check will mark it FAILED-duplicate of that holder and delete its
    ``full_docs`` body, and the key it was just given ends with no primary — the
    state the repair endpoint cannot settle, since it only accepts a current
    primary candidate.

    Bounded on purpose, and partial by construction:

    * a primary with no ``content_hash`` yet (an unparsed PENDING candidate — an
      ordinary conflict candidate) cannot be checked at all: nobody knows what
      its content will be. If it later turns out to duplicate another document,
      the outcome is the ordinary content-dedup steady state (the content lives
      under the other document, and this key has no primary because the file's
      content is not its own), and no content is lost. That residual is
      irreducible without parsing the document inside the repair;
    * a match that claims THIS canonical source is not a reason to refuse — it is
      one of the candidates this repair is about to demote (or one an earlier
      repair already demoted). The lookup returns only the earliest holder, so a
      same-key holder that sorts first also hides any further one: the check
      abstains rather than guess, which is the same partiality with the same
      residual as the case above.

    The lookup is fail-closed by contract (``get_doc_by_content_hash`` raises
    rather than reporting a transport failure as "no holder"), and the strict
    hydration of the primary's own row propagates too, so an unreadable state
    refuses the commit instead of skipping the check.
    """
    rows = await doc_status.get_full_docs_by_ids([primary_doc_id], strict=True)
    primary_row = rows.get(primary_doc_id)
    if primary_row is None:
        # Not a candidate any more; the backend's own check reports that, with
        # the message that sends the operator to re-list the conflict.
        return
    content_hash = doc_status_field(primary_row, "content_hash", "") or ""
    if not content_hash:
        return
    match = await get_duplicate_doc_by_content_hash(
        doc_status, content_hash, primary_doc_id
    )
    if match is None:
        return
    holder_doc_id, holder_row = match
    if doc_status_field(holder_row, "file_path", "") == canonical_source_key:
        return
    raise SourceConflictPrimaryUnusableError(
        f"Document {primary_doc_id} holds the same content as {holder_doc_id}, "
        f"which claims a different source. Processing marks a content duplicate "
        f"FAILED and deletes its content, so {primary_doc_id} cannot keep "
        f"'{canonical_source_key}' — the key would end up with no primary at all, "
        f"and no repair can settle that. Pick another primary, or settle "
        f"{holder_doc_id} first.",
        reason=SourceConflictPrimaryUnusableError.REASON_CONTENT_ELSEWHERE,
        holder_doc_id=holder_doc_id,
    )


async def refuse_an_unusable_primary(
    doc_status: Any,
    full_docs: Any,
    canonical_source_key: str,
    primary_doc_id: str,
) -> None:
    """Every pre-commit refusal, in one call the commit paths cannot half-apply.

    Both checks exist for the same reason — a repair only demotes, so a primary
    that cannot keep the source leaves a key no repair can settle — and both must
    run on BOTH commit paths (the endpoint and the library/CLI helper). Bundling
    them here is what keeps one path from silently growing a check the other
    lacks.
    """
    await refuse_a_contentless_primary(full_docs, primary_doc_id)
    await refuse_a_primary_whose_content_lives_elsewhere(
        doc_status, canonical_source_key, primary_doc_id
    )


async def _absent_key_recovery_hint(doc_status: Any, primary_doc_id: str) -> str:
    """What can actually be done about a key left with no primary.

    Reads the kept primary's row so the advice matches its state, because the two
    states have DIFFERENT recoveries and the wrong one wastes the operator's time:

    * marked a content duplicate — deleting this row is not enough. Its content
      belongs to the named document, and a doc id is derived from the file name,
      so re-uploading the same bytes is deduplicated against that document again
      and the key stays Absent. The source can only be reclaimed once that holder
      is deleted (or the file's content actually changes) — and if the holder is
      the right home for the content, the honest end state is that this file has
      no separate document at all;
    * gone — a concurrent delete removed it, so re-ingesting the file recreates a
      primary and the key resolves again.

    Best-effort by design: this runs on a path that is already raising, so a read
    failure must not replace the reported outcome with a read error. It degrades
    to naming both possibilities.
    """
    try:
        rows = await doc_status.get_full_docs_by_ids([primary_doc_id], strict=True)
    except Exception as read_error:  # already failing; never mask the outcome
        logger.warning(
            f"Could not read {primary_doc_id} to explain the unsettled repair: "
            f"{read_error}"
        )
        rows = {}
    row = rows.get(primary_doc_id)
    preamble = (
        "The repair cannot be re-run: with no candidate left there is no conflict "
        "to settle and the endpoint refuses any primary as 'not a current "
        "candidate'."
    )
    if row is None:
        return (
            f"{preamble} {primary_doc_id} is gone (a concurrent delete, or a read "
            f"that could not confirm it): re-ingest the file to recreate a primary "
            f"for this source, or re-list the conflicts to see the current state."
        )
    metadata = doc_status_field(row, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("is_duplicate"):
        holder = metadata.get("original_doc_id") or "another document"
        return (
            f"{preamble} {primary_doc_id} is now marked a duplicate of {holder}, so "
            f"its content lives there. Deleting {primary_doc_id} is NOT enough to "
            f"reclaim the source: a doc id is derived from the file name, so "
            f"re-uploading the same bytes is deduplicated against {holder} again. "
            f"Reclaiming it needs {holder} deleted or this file's content actually "
            f"changed — and if {holder} is the right home for that content, this "
            f"source having no document of its own is the correct end state."
        )
    return (
        f"{preamble} {primary_doc_id} still exists and is not marked a duplicate, "
        f"so it lost the source some other way (a rewritten file_path, an external "
        f"writer): re-list the conflicts and inspect that row before acting."
    )


async def verify_repair_outcome(
    doc_status: Any,
    canonical_source_key: str,
    primary_doc_id: str,
    result: Any,
) -> None:
    """Re-resolve the key after a commit and raise unless it is now UNIQUE on
    ``primary_doc_id``.

    Raising rather than warning: an operator repairing a conflict needs to know
    the key is settled, and a repair that "succeeded" while leaving the key
    Absent (the kept primary was deleted meanwhile) or still Conflicting (a
    candidate re-appeared) is exactly the outcome they would otherwise act on as
    if it were fixed. The demotions themselves have already been committed and
    are not rolled back — they are individually correct; what failed is the
    end state, so the message says which.

    The recovery is per-outcome, and only two of the three are "run it again":
    with a primary left (wrong one, or several) the key still has candidates, so
    a fresh dry-run → commit settles it. With NO primary left there is nothing to
    repair — this function is the one place that knows that, so it is the one
    place that must not send the operator back to an endpoint that will answer
    "not a current primary candidate".
    """
    from lightrag.base import SourceConflict, SourceUnique

    resolution = await doc_status.resolve_doc_source_strict(canonical_source_key)
    if isinstance(resolution, SourceUnique) and resolution.doc_id == primary_doc_id:
        return
    rerun = (
        "Re-run the repair (fresh dry-run, then commit) once concurrent writers "
        "to these documents have stopped."
    )
    if isinstance(resolution, SourceUnique):
        detail = (
            f"the surviving primary is {resolution.doc_id}, not the requested "
            f"{primary_doc_id}"
        )
        recovery = rerun
    elif isinstance(resolution, SourceConflict):
        detail = (
            "the key is STILL in conflict "
            f"(sample: {', '.join(resolution.sample_doc_ids) or 'unavailable'})"
        )
        recovery = rerun
    else:
        detail = (
            f"the key now resolves to NO primary at all — {primary_doc_id} was "
            "deleted or marked a duplicate by a concurrent writer"
        )
        recovery = await _absent_key_recovery_hint(doc_status, primary_doc_id)
    raise StorageControlPlaneError(
        f"Source conflict repair for '{canonical_source_key}' committed its "
        f"demotions ({list(result.demoted_sample_doc_ids) or 'none'}) but the key "
        f"is not settled: {detail}. {recovery}"
    )


def _print_conflicts(conflicts: list[SourceConflictSummary]) -> None:
    if not conflicts:
        print("No source conflicts found.")
        return
    print(f"{len(conflicts)} conflicting canonical source key(s):")
    for conflict in conflicts:
        count = (
            conflict.candidate_count
            if conflict.candidate_count is not None
            else "2+ (exact count unavailable)"
        )
        print(f"  {conflict.canonical_source_key}: {count} primary candidates")
        for doc_id in conflict.sample_doc_ids:
            print(f"    - {doc_id}")
        if conflict.candidate_count is not None and conflict.candidate_count > len(
            conflict.sample_doc_ids
        ):
            remaining = conflict.candidate_count - len(conflict.sample_doc_ids)
            print(f"    … {remaining} more (sample is bounded)")
    print(
        "\nSettle one with:\n"
        "  python -m lightrag.tools.source_conflict_repair repair "
        "--source <key> --primary <doc_id> [--apply]"
    )


def _print_repair(result: Any) -> None:
    verb = "Committed" if result.committed else "Would demote (dry-run)"
    print(f"Source: {result.canonical_source_key}")
    print(f"Keeping primary: {result.primary_doc_id}")
    print(f"Primary candidates observed: {result.candidate_count}")
    print(f"Candidate fingerprint: {result.fingerprint}")
    print(f"{verb}: {list(result.demoted_sample_doc_ids) or 'nothing'}")
    if not result.committed:
        print("\nNothing was modified. Re-run with --apply to commit.")
    else:
        print(
            "\nDemoted rows keep their content and status; they are now marked "
            "metadata.is_duplicate=true with original_doc_id="
            f"{result.primary_doc_id}."
        )


async def _async_main(args: argparse.Namespace) -> bool:
    import numpy as np

    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_share_data
    from lightrag.utils import EmbeddingFunc

    async def _noop_llm(*_args, **_kwargs) -> str:
        raise RuntimeError("source_conflict_repair never calls the LLM")

    async def _noop_embed(texts: list[str]) -> "np.ndarray":
        raise RuntimeError("source_conflict_repair never embeds")

    initialize_share_data(workers=1)
    rag = LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        workspace=args.workspace,
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=8192,
            func=_noop_embed,
        ),
    )
    # doc_status plus full_docs: still no vector/graph store, no LLM and no
    # embedding model — full_docs is a plain KV, and reading it is what lets the
    # repair refuse a primary that has no content (see
    # refuse_a_contentless_primary).
    doc_status = rag.doc_status
    await doc_status.initialize()
    full_docs = rag.full_docs
    full_docs_error: Exception | None = None
    try:
        await full_docs.initialize()
    except Exception as init_error:
        full_docs_error = init_error
        full_docs = None
    if full_docs_error is not None:
        # Listing and dry-runs mutate nothing, so they still run — but a COMMIT
        # without this check is the fail-open the check exists to remove: an
        # unverified primary that turns out to be a contentless stub gets the
        # source key, a later scan deletes it, and the demotions this command
        # already wrote cannot be undone. Refuse the command instead.
        if args.command == "repair" and args.apply:
            print(
                f"Refused: full_docs is unavailable ({full_docs_error}), so the "
                "repair cannot confirm the chosen primary has content. Retry once "
                "it is reachable (listing and dry-runs still work)."
            )
            await doc_status.finalize()
            return False
        print(
            f"Warning: full_docs is unavailable ({full_docs_error}); a commit "
            "would be refused, but this command modifies nothing."
        )
    try:
        if args.command == "list":
            _print_conflicts(
                await collect_source_conflicts(
                    doc_status, limit=args.limit, all_pages=args.all
                )
            )
            return True
        result = await repair_one_conflict(
            doc_status,
            args.source,
            args.primary,
            # rag.workspace, not doc_status.workspace: a backend may override
            # its own from the environment, and the enqueue path locks on this one.
            workspace=rag.workspace,
            full_docs=full_docs,
            apply=args.apply,
        )
        _print_repair(result)
        return True
    except ValueError as not_a_candidate:
        print(f"Refused: {not_a_candidate}")
        print("Run 'list' again — the candidate set may have changed.")
        return False
    except SourceConflictRepairCASError as cas_error:
        print(f"Refused (candidate set changed under the repair CAS): {cas_error}")
        print("Nothing was modified. Run the command again.")
        return False
    except StorageCapabilityError as capability_error:
        print(f"Unsupported by the configured doc_status backend: {capability_error}")
        return False
    except StorageControlPlaneError as storage_error:
        print(f"Storage control plane unavailable: {storage_error}")
        return False
    finally:
        if full_docs is not None:
            await full_docs.finalize()
        await doc_status.finalize()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
    parser = argparse.ArgumentParser(
        description=(
            "List and repair canonical source keys claimed by more than one "
            "primary document (LightRAG doc_status)."
        )
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("WORKSPACE", ""),
        help="Workspace to operate on (default: WORKSPACE env / the default one)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List source conflicts")
    list_parser.add_argument(
        "--limit", type=int, default=50, help="Page size (default: 50)"
    )
    list_parser.add_argument(
        "--all", action="store_true", help="Walk every page, not just the first"
    )

    repair_parser = subparsers.add_parser(
        "repair", help="Settle one conflict by naming the surviving primary"
    )
    repair_parser.add_argument(
        "--source", required=True, help="Canonical source key to repair"
    )
    repair_parser.add_argument(
        "--primary", required=True, help="Document ID to keep as the single primary"
    )
    repair_parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit the demotions (default: dry-run, nothing is modified)",
    )

    args = parser.parse_args()
    if args.command == "list" and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if not asyncio.run(_async_main(args)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
