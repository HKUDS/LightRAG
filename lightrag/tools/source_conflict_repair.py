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
- ``--apply`` re-reads the candidate set under the backend's repair lock and
  commits only when it still matches the dry-run it just performed
  (compare-and-set), so a concurrent enqueue/delete fails the repair instead of
  being overwritten;
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
from typing import Any

from lightrag.base import CURSOR_END, CURSOR_START, SourceConflictSummary
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageCapabilityError,
    StorageControlPlaneError,
)

# Pages are bounded on both sides: the backend projects a bounded sample per
# key, and --all stops after this many pages so a pathological workspace cannot
# turn a listing into an unbounded walk.
_MAX_PAGES = 1000


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
    apply: bool = False,
) -> Any:
    """Dry-run one repair and, with ``apply``, commit it with that fresh token.

    The commit echoes the count/fingerprint the dry-run just returned, so the
    backend still fails closed if the candidate set changes in between — the CAS
    is not bypassed by automating the two steps, only the copy-paste is.
    """
    dry = await doc_status.repair_source_conflict(
        canonical_source_key,
        primary_doc_id=primary_doc_id,
        # Ignored in dry-run mode by contract; the tokens below come from it.
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )
    if not apply:
        return dry
    return await doc_status.repair_source_conflict(
        canonical_source_key,
        primary_doc_id=primary_doc_id,
        expected_candidate_count=dry.candidate_count,
        expected_candidate_fingerprint=dry.fingerprint,
        dry_run=False,
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
            doc_status, args.source, args.primary, apply=args.apply
        )
        _print_repair(result)
        return True
    except ValueError as not_a_candidate:
        print(f"Refused: {not_a_candidate}")
        print("Run 'list' again — the candidate set may have changed.")
        return False
    except SourceConflictRepairCASError as cas_error:
        print(f"Refused (candidate set changed under the repair lock): {cas_error}")
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
