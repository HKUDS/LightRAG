# Source-Conflict Listing / Repair

Offline companion to the typed source resolver (LR2 §5.5).

A canonical source key — a filename with any parser `[hint]` stripped — is
expected to belong to exactly one **primary** document. History can break that:
custom-ID inserts, legacy ids, and basename collisions predate the strict
resolver. When two or more primary rows claim one key, the resolver returns a
*conflict* and `/documents/scan` refuses to act on that file: it is not
enqueued, no record is deleted and nothing is archived, because picking a winner
automatically could silently retire the document you wanted to keep.

This tool (and the equivalent HTTP endpoints) is how an operator settles such a
conflict explicitly.

- The winner is never chosen for you: you name the `primary_doc_id` to keep.
- Every other candidate is marked `metadata.is_duplicate=true` with
  `original_doc_id=<primary>`. **No content is ever deleted** — the demoted rows
  keep their status and their `full_docs` entry; they only lose their claim on
  the canonical source, which is what makes the resolver return a single
  primary afterwards.
- Only `doc_status` is opened. No vector store, graph store, LLM or embedding
  model is touched, so a doc_status bookkeeping problem never requires a
  reachable Milvus/Neo4j/LLM endpoint to fix.

## Usage

CLI (honors `WORKING_DIR` / `WORKSPACE` and the `LIGHTRAG_*` storage variables
from `.env`, exactly like the server):

```bash
# What is conflicting?
python -m lightrag.tools.source_conflict_repair list [--limit 50] [--all]

# Plan a repair (dry-run: nothing is modified)
python -m lightrag.tools.source_conflict_repair repair \
    --source report.docx --primary doc-abc123

# Commit it
python -m lightrag.tools.source_conflict_repair repair \
    --source report.docx --primary doc-abc123 --apply
```

`--workspace` overrides the `WORKSPACE` env var. Exit code is `1` when a repair
is refused (unknown primary, candidate set changed, backend cannot repair).

Library (for a deployment that builds its own `LightRAG`):

```python
from lightrag.tools.source_conflict_repair import (
    collect_source_conflicts,
    repair_one_conflict,
)

conflicts = await collect_source_conflicts(rag.doc_status, limit=50)
result = await repair_one_conflict(
    rag.doc_status, "report.docx", "doc-abc123", apply=True
)
```

## Online equivalent

```
GET  /documents/source_conflicts?limit=50&cursor=<next_cursor>
POST /documents/source_conflicts/repair
     {"canonical_source_key": "...", "primary_doc_id": "...",
      "expected_candidate_count": N, "expected_candidate_fingerprint": "...",
      "dry_run": false}
```

The HTTP flow is two explicit calls: a dry-run returns the
`candidate_count`/`fingerprint` pair, and the commit echoes both back. The CLI
performs the same two steps for you — it automates the copy-paste, not the
guard.

## Safety

- Listing and a repair without `--apply` mutate nothing.
- A commit re-reads the candidate set under the backend's repair lock and
  proceeds only when the count and fingerprint still match the dry-run
  (compare-and-set), so a concurrent enqueue / delete / repair fails the commit
  instead of being overwritten. A refused commit changes nothing; re-run it.
- A repair interrupted half-way needs no reconciliation and leaves no marker
  behind: the rows it finished already express `duplicate`, the rest still
  resolve as a conflict. Run the command again.
- Repeating a completed repair is safe: the same request is refused (its token
  described the old candidate set) and a fresh run is a no-op with one
  candidate and nothing to demote.
