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
   deployment that never migrates never has it. Only a fresh migration writes
   the marker, and only while the target workspace is still empty; a resume
   never re-creates it.
2. **Segmented server-side copy.** The workspace is walked by keyset
   pagination on the legacy primary key, `LEGACY_VECTOR_MIGRATION_BATCH_SIZE`
   rows per statement. Each statement is an `INSERT ... SELECT ... ON CONFLICT
   (workspace, id) DO NOTHING` with explicitly named columns and commits on its
   own. No vector is transferred to the client; `vector` to `halfvec` uses
   pgvector's assignment cast. The statement takes a share lock on the marker
   row and copies only while that row exists.
3. **Verification by identity.** After the last segment an anti-join counts
   legacy rows of the workspace that are absent from the target. Anything but
   zero raises `DataMigrationError` and leaves the marker in place.
4. **Marker last.** The marker row is deleted by the verification statement
   itself, and only when nothing is missing.

At startup the marker outranks the row count: a workspace whose marker is
present is resumed however many rows its target already holds. Resuming
restarts the walk from the beginning; rows already copied conflict away.

Every step is idempotent, so the shared connection retry applies to each
statement unchanged, including a statement whose `COMMIT` response was lost.

## The marker is the fence between server instances

The data-init lock serializes the workers of one server. It does not reach a
second server instance started against the same database, and such an instance
finds the first one's marker and joins the copy. That is harmless while both
only copy. It stops being harmless the moment one of them finishes, starts
serving and deletes a document: a copy still running elsewhere would read the
deleted rows from the unchanged legacy table and write them back.

The marker row closes this. Every segment holds a share lock on it for the
length of its transaction, so the statement that deletes the marker waits for
the segments in flight, and a segment that starts afterwards finds no marker
and copies nothing. Whoever clears the marker ends the migration for every
instance. An instance that finds the marker gone stops without verifying: rows
missing from the target at that point may have been deleted on purpose, so
their absence is not a failure.

Two rules follow, and breaking either reopens the hole: no legacy row may be
copied outside a statement that holds the marker, and nothing but a fresh
migration into an empty target may create it.

A session-level advisory lock was rejected for this. It would tie the whole
migration to one connection, it does not survive a transaction-pooling proxy,
and losing that connection would drop the lock while the copy carried on.

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

- Within one server only one process migrates: storage initialization runs
  under the data-init lock, and the other workers wait. Independent server
  instances may copy at the same time; they duplicate work, not data, and the
  marker fence above stops the slower ones once the first has finished.
- A workspace whose target is empty and whose legacy rows still exist is
  migrated, whatever happened before. Deleting every document one by one after
  a migration therefore brings the legacy rows back on the next startup, unless
  the legacy rows were removed as the migration log asks. `drop()` clears both
  sides and is not affected. This predates the segmented copy.
- The legacy table is assumed quiescent. Rows added to it by an old runtime
  while the copy runs may or may not be picked up.
- A legacy column that the target table lacks fails the copy instead of being
  dropped. Columns only the target has take their defaults.
- The legacy rows are never deleted by the migration. Removing them after
  checking the result remains a manual step.
