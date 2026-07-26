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

Safety model:

- listing and ``repair`` without ``--apply`` mutate nothing;
- ``--apply`` re-reads the candidate set and commits only when it still matches
  the dry-run it just performed (compare-and-set), so a concurrent enqueue/delete
  fails the repair instead of being overwritten; the commit runs under the
  canonical-key + enqueue-serialize locks, which is what keeps a NEW primary from
  being inserted between that re-read and the demotions;
- a repair interrupted half-way needs no reconciliation: the finished rows
  already express ``duplicate`` and the rest still resolve as a conflict, so
  simply run it again.

Only ``doc_status`` is opened — no vector store, graph store, LLM or embedding
model is touched, so this is safe to run against a live deployment's database
while the server is stopped or running.

Usage (CLI — honors WORKING_DIR / WORKSPACE and the LIGHTRAG_* storage env vars
from ``.env``, same as the server)::

    python -m lightrag.tools.source_conflict_repair list [--limit 50] [--all]
    python -m lightrag.tools.source_conflict_repair repair \\
        --source report.docx --primary doc-abc123 [--apply]

Usage (library — for a deployment that builds its own LightRAG)::

    conflicts = await collect_source_conflicts(rag.doc_status, limit=50)
    result = await repair_one_conflict(
        rag.doc_status, "report.docx", "doc-abc123", apply=True
    )
"""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from lightrag.base import CURSOR_END, CURSOR_START, SourceConflictSummary
from lightrag.constants import (
    ENQUEUE_SERIALIZE_LOCK_NAMESPACE,
    SOURCE_CONFLICT_LOCK_NAMESPACE,
)
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageCapabilityError,
    StorageControlPlaneError,
)

# Pages are bounded on both sides: the backend projects a bounded sample per
# key, and --all stops after this many pages so a pathological workspace cannot
# turn a listing into an unbounded walk.
_MAX_PAGES = 1000


@asynccontextmanager
async def source_conflict_repair_lock(workspace: str, canonical_source_key: str):
    """Hold the caller-side locks a source-conflict COMMIT requires.

    ``repair_source_conflict`` re-reads the candidate set and demotes the losers;
    no backend can stop a new primary being INSERTED in between (``FOR UPDATE``
    locks only the rows it saw, a snapshot transaction never sees a phantom, and
    Redis/OpenSearch have no transaction at all), so excluding that insert is the
    caller's job — see ``DocStatusStorage.repair_source_conflict``.

    Two locks, always in this order:

    * a keyed lock on the canonical source key — serializes concurrent repairs of
      the SAME key without blocking repairs of other keys;
    * the workspace's enqueue-serialize lock — the one that actually excludes the
      phantom, because it is the same lock the enqueue critical section
      (``filter_keys`` → dedup → ``doc_status.upsert``) holds.

    Lock order is keyed → namespace, and the enqueue path never takes the keyed
    lock, so no cycle exists. Scope is one deployment (its worker processes
    included), not several deployments sharing a database — a repair run against
    a database that another deployment is writing to keeps the phantom window.
    """
    # Imported lazily: the CLI initializes shared storage in main(), and
    # importing at module scope would bind before that happens.
    from lightrag.kg.shared_storage import (
        get_namespace_lock,
        get_storage_keyed_lock,
    )

    keyed_namespace = (
        f"{workspace}:{SOURCE_CONFLICT_LOCK_NAMESPACE}"
        if workspace
        else SOURCE_CONFLICT_LOCK_NAMESPACE
    )
    async with get_storage_keyed_lock(
        [canonical_source_key], namespace=keyed_namespace, enable_logging=False
    ):
        async with get_namespace_lock(
            ENQUEUE_SERIALIZE_LOCK_NAMESPACE, workspace=workspace
        ):
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
        # that the demotions named in the result landed; it cannot claim the key
        # is now unique, because the locks here exclude a concurrent ENQUEUE and
        # nothing else — a concurrent delete of the kept primary, a
        # processing-stage duplicate marking, or a scan stale-stub deletion all
        # mutate this candidate set without taking them (see
        # DocStatusStorage.repair_source_conflict). Those paths are single-doc
        # and cannot be excluded from here, so the honest move is to re-resolve
        # and report what is actually true.
        await verify_repair_outcome(
            doc_status, canonical_source_key, primary_doc_id, result
        )
        return result


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
    """
    from lightrag.base import SourceConflict, SourceUnique

    resolution = await doc_status.resolve_doc_source_strict(canonical_source_key)
    if isinstance(resolution, SourceUnique) and resolution.doc_id == primary_doc_id:
        return
    if isinstance(resolution, SourceUnique):
        detail = (
            f"the surviving primary is {resolution.doc_id}, not the requested "
            f"{primary_doc_id}"
        )
    elif isinstance(resolution, SourceConflict):
        detail = (
            "the key is STILL in conflict "
            f"(sample: {', '.join(resolution.sample_doc_ids) or 'unavailable'})"
        )
    else:
        detail = (
            f"the key now resolves to NO primary at all — {primary_doc_id} was "
            "removed by a concurrent delete or duplicate marking"
        )
    raise StorageControlPlaneError(
        f"Source conflict repair for '{canonical_source_key}' committed its "
        f"demotions ({list(result.demoted_sample_doc_ids) or 'none'}) but the key "
        f"is not settled: {detail}. Re-run the repair once concurrent writers to "
        f"these documents have stopped."
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
    # Only doc_status is initialized: this tool must not require a reachable
    # vector/graph store to fix a doc_status bookkeeping problem.
    doc_status = rag.doc_status
    await doc_status.initialize()
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
