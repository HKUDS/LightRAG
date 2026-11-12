# Atomic PostgreSQL legacy vector migration

When a model-suffixed vector table is empty for the effective workspace,
LightRAG copies legacy rows and verifies their count in one PostgreSQL
repeatable-read transaction on one connection. The copy uses server-side
`INSERT ... SELECT` with explicit source column names, avoiding Python vector
serialization and independently committed 500-row batches.

An insert error, verification failure or cancellation before commit rolls back
the transaction's new rows. Retryable connection failures retry the complete
operation, not an individual batch. Serialization failures fail safely; this
change does not expand the global retry policy.

## Commit ambiguity and existing targets

A disconnect or cancellation during commit can leave its outcome unknown to
the caller. Do not assume it rolled back. A subsequent attempt checks the target
inside its transaction. If that workspace is already populated, it preserves
it without copying again. This avoids overwriting committed data after a lost
commit response, but does not certify an existing target as complete.

Older versions could leave partially copied targets. This change cannot
identify those retrospectively and does not automatically backfill, overwrite
or delete them. Preserve both source and target, stop all writers and reconcile
such targets offline. Do not use a destructive clear/rebuild as a shortcut:
backend drop can also remove the retained legacy source.

## Operational boundaries

Stop writers to the source and target during migration. Repeatable read gives
source counts and copying one consistent database snapshot; it does not capture
later source changes, fence old runtimes, or provide a graph/KV/vector snapshot
across stores. Other workspaces are outside the copy's workspace filter.
`POSTGRES_WORKSPACE` can override the requested workspace, so verify the effective
database, table and workspace before maintenance.

One large transaction can generate substantial WAL, retain old row versions,
maintain indexes and hold locks until commit. Server-side copying reduces client
memory and round trips, not total database work. Budget database capacity and
migration time, retain verified backups, and inspect failures before retrying.
No migration marker column or new metadata table is introduced.

Source column names are explicitly mapped into the target, independent of column
order. Target defaults apply to columns absent from the source; incompatible
schemas or types fail rather than silently dropping source fields. Deployment
validation should include actual PostgreSQL/pgvector versions, vector/halfvec
casts, representative volume, concurrent failures and cancellation. Unit tests
with transaction fakes are not a substitute for those server checks.
