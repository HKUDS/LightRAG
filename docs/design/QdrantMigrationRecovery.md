# Qdrant interrupted legacy migration

Qdrant's model-suffixed collection is shared by workspaces. Legacy migration
copies each workspace in batches of 500 points and keeps the legacy collection.
Previously, a failure after the first batch left some points in the new target.
The next startup treated any nonempty target workspace as already migrated and
served only that partial index.

## Persistent incomplete state

Every newly copied point carries the reserved boolean payload field
`_lightrag_pending_migration=true`. The flag is part of the same upsert as its
vector and payload, so a successful or partially applied batch leaves evidence
on the target itself, not just in the failed process. Legacy values under this
reserved key are overwritten; applications must not use it for their own data.

A boolean payload index is provisioned before checking this field, including on
existing collections, so strict-mode servers need not allow unindexed filtering.
Before accepting an existing target, startup checks this field within the
effective workspace. Any pending point refuses startup, including when the
legacy collection was subsequently removed. An unreadable check also fails
startup. Other workspaces in the shared collection are not refused or changed.

After every source batch is copied and the existing exact-count verification
passes, migration removes the flag with a workspace-scoped, awaited payload
update and verifies no pending flags remain. Failure during flag removal remains
incomplete until no flags remain. If the server completed removal but the reply
was lost, a later startup may succeed: the data verification preceded removal.
No legacy data is deleted by this completion step.

This is a durable incomplete-migration indication, not an embedding-space
provenance marker, transaction, distributed lock, or automatic recovery. The
existing vector-space contract is unchanged.

## Recovery without destroying the source

1. Stop **all** runtimes, SDK processes, and maintenance jobs that can write the
   source or target workspace. Take and verify backups of both. Do not clear
   the pending field to force startup.
2. Identify the exact Qdrant endpoint, target collection, legacy source, and
   **effective** workspace from the startup configuration and logs.
   `QDRANT_WORKSPACE` overrides the LightRAG workspace; collection names alone
   do not identify a tenant. Verify that the retained legacy source is complete
   and usable for this workspace before discarding anything.
3. Inspect the target workspace. If it contains any unmarked points, it may
   contain post-migration writes or another writer's changes. Preserve it and
   reconcile offline; do not automatically replace those records from legacy.
4. For a never-published target whose **every** workspace point is still pending,
   remove only those points using Qdrant's workspace-filtered delete. Keep the
   collection and every sibling workspace intact. **Do not call LightRAG's
   `storage.drop()` or clear-storage/rebuild recovery as a shortcut:** its drop
   also cleans the retained legacy data.
5. With writers still stopped, initialize again. The empty target workspace is
   copied from the preserved legacy source. Verify completeness, then start the
   runtime. Keep the source backup until the result has been checked.

The following is the narrowly scoped reset in step 4, after steps 1–3 have been
completed. `client` is the operator's configured Qdrant client; `target` and
`workspace` must be the verified physical destination and effective workspace.
This intentionally deletes incomplete target copies and does not select legacy:

```python
from qdrant_client import models

scope = models.Filter(must=[models.FieldCondition(
    key="workspace_id", match=models.MatchValue(value=workspace)
)])
pending = models.Filter(must=[*scope.must, models.FieldCondition(
    key="_lightrag_pending_migration", match=models.MatchValue(value=True)
)])
total = client.count(target, count_filter=scope, exact=True).count
incomplete = client.count(target, count_filter=pending, exact=True).count
if not total or total != incomplete:
    raise RuntimeError("Target is empty or has unmarked data; preserve and inspect it")
client.delete(target, points_selector=models.FilterSelector(filter=scope), wait=True)
if client.count(target, count_filter=scope, exact=True).count:
    raise RuntimeError("Target reset was incomplete; do not start the runtime")
```

These checks do not replace stopped writers: count followed by delete is not a
transaction, and concurrent writes could invalidate the operator's inspection.

## Boundaries

Old versions did not write this flag, so an already-existing unmarked partial
migration cannot be distinguished from a legitimate populated target. Such
workspaces need offline source/target reconciliation. Existing unmarked targets
retain their previous behavior rather than being silently overwritten.

The marker does not fence old binaries, detect source writes during copying,
or prove a cross-store snapshot. Concurrent writers must remain stopped. The
existing migration count check is not a graph-authoritative relation-ID rebuild
or content equality proof. This change does not implement the broader relation
format migration, shadow-target publication, or cross-backend cutover protocol.

Tests use the official persistent local Qdrant client, reopen it after failure,
and cover completion, interrupted batches, marker cleanup, and tenant isolation.
They do not establish distributed Qdrant replica or power-loss behavior.
