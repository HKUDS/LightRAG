# PostgreSQL legacy vector migration

When a model-suffixed vector table holds no rows for a workspace and the
legacy (unsuffixed) table does, `PGVectorStorage.setup_table` copies that
workspace's rows across at startup. This document states how the copy stays
safe to interrupt, what it leaves behind when it is, and what was rejected.

## The defect this replaces

The copy used to read 500 rows at a time into Python and write each batch in
its own transaction. A failure after the first batch left a partly filled
target, and the next startup took "the target has rows" to mean "the migration
is done": it logged a warning and served an incomplete vector index from then
on. Nothing healed it. The same code could not migrate a `VECTOR` legacy table
into a `HALFVEC` target at all, because the decoded vector objects were not
valid `halfvec` parameters.

## Mechanism

1. **Marker first.** Before the first row is copied, a row keyed by
   `(target_table, workspace)` is committed to `LIGHTRAG_VDB_MIGRATION_STATE`.
   The table is created on first use and is not part of `TABLES`, so a
   deployment that never migrates never has it.
2. **Segmented server-side copy.** The workspace is walked by keyset
   pagination on the legacy primary key, `LEGACY_VECTOR_MIGRATION_BATCH_SIZE`
   rows per statement. Each statement is an `INSERT ... SELECT ... ON CONFLICT
   (workspace, id) DO NOTHING` with explicitly named columns and commits on its
   own. No vector is transferred to the client; `vector` to `halfvec` uses
   pgvector's assignment cast.
3. **Verification by identity.** After the last segment an anti-join counts
   legacy rows of the workspace that are absent from the target. Anything but
   zero raises `DataMigrationError`.
4. **Marker last.** Only after verification is the marker row deleted.

At startup the marker outranks the row count: a workspace whose marker is
present is resumed however many rows its target already holds. Resuming
restarts the walk from the beginning; rows already copied conflict away.

Every step is idempotent, so the shared connection retry applies to each
statement unchanged, including a statement whose `COMMIT` response was lost.

## Why not one transaction

A single transaction over the whole workspace is atomic and needs no marker,
but its cost grows with the table: the WAL of every row and of the vector
index maintenance is held until commit, old row versions are retained, and a
late failure rolls all of it back and starts from zero. Segments bound each of
those to one batch and keep the work already done.

## Accepted residues

- **A partly filled target between a failure and the next startup.** It heals
  on the next startup, which resumes the copy. Startup fails until the copy
  verifies, so the partial target is never served by this version.
- **A marker that outlives its purpose.** If the workspace's legacy rows are
  cleared (`drop()`), or the legacy table is removed, before a resume runs, the
  marker row stays. It is inert: `setup_table` returns before consulting it when
  the legacy side has nothing for the workspace.
- **Partial targets left by older versions.** They carry no marker and cannot
  be told apart from a completed migration followed by deletions, so they are
  neither backfilled nor cleared. The only signal is a warning when the target
  holds fewer rows than the legacy table. To repair one, stop the server and
  rebuild the vectors with `lightrag-rebuild-vdb`. Deleting the workspace's rows
  from the suffixed table also makes the next startup migrate again, but it
  discards every vector written since the interrupted migration.
- **Downgrade during an interrupted migration.** A version that predates the
  marker ignores it and accepts the partial target. Finish the migration before
  downgrading.

## Boundaries

- Only one process migrates at a time: storage initialization runs under the
  data-init lock, and the other workers wait. Independent server instances that
  start together are still safe, because every statement is idempotent; they
  duplicate work rather than data.
- The legacy table is assumed quiescent. Rows added to it by an old runtime
  while the copy runs may or may not be picked up.
- A legacy column that the target table lacks fails the copy instead of being
  dropped. Columns only the target has take their defaults.
- The legacy rows are never deleted by the migration. Removing them after
  checking the result remains a manual step.
