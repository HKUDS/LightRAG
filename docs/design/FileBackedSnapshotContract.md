# File-Backed Snapshot Contract

Read this before changing `lightrag/kg/nano_vector_db_impl.py`,
`lightrag/kg/faiss_impl.py`, `lightrag/kg/json_kv_impl.py`, or
`lightrag/kg/file_fingerprint.py`.

Three storages keep their data in process memory and persist it by rewriting a
whole file. Two of them — `NanoVectorDBStorage` and `FaissVectorDBStorage` —
share one design and are documented together here. The third, `JsonKVStorage`,
uses a *fundamentally different* cross-process model and is covered in its own
section; compare carefully before changing either side.

The graph store, `NetworkXStorage`, is the fourth file-backed storage and the
one that behaves differently on a write conflict. Its contract is
[NetworkXSingleWriterContract.md](NetworkXSingleWriterContract.md), and the
comparison between the two is
[the section below](#why-these-backends-never-decline-a-stale-write).

## Scope

All four file-backed storages are supported for **small-scale testing and
validation only**. Production deployments run server-backed storages. The cost
of a whole-file rewrite is therefore not a consideration, and no change to these
files may be justified by — or blocked on — it.

## The vector backends

`NanoVectorDBStorage` and `FaissVectorDBStorage`.

### Storage model

`NanoVectorDBStorage` keeps a single `NanoVectorDB` instance in process memory
and serializes its full state to one JSON file at
`working_dir/[workspace/]vdb_<namespace>.json`.

`FaissVectorDBStorage` splits its state across two fields — `self._index` (the
Faiss index) and `self._id_to_meta` (`dict[int_faiss_id, metadata]`) — and two
files per `(workspace, namespace)`:

* `working_dir/[workspace/]faiss_index_<namespace>.index` — the Faiss index,
  binary, written by `faiss.write_index`.
* `…<namespace>.index.meta.json` — `_id_to_meta` as JSON, **without** the
  `__vector__` field; vectors are reconstructed from the index on load.

Cosine similarity is obtained by storing L2-normalized vectors in an
`IndexFlatIP` (inner-product search over normalized vectors is cosine).

In both cases the file(s) are the **only** cross-process synchronization surface
— no shared memory, no message bus, no network channel. Cross-process visibility
is mediated by (a) an atomic file write at commit time and (b) a per-namespace
`storage_updated` flag distributed through `lightrag.kg.shared_storage`.

**Faiss cross-file atomicity is not guaranteed.** The two `atomic_write` renames
in `_save_faiss_index` are independent, so a crash between them can leave
`.index` and `.meta.json` referring to different snapshots. `_load_faiss_index`
tolerates both directions: `meta > index` rows are dropped silently;
`index > meta` — the more dangerous case — is logged as a warning but **not**
auto-repaired, so orphan vectors remain in the loaded index, unreachable through
custom-id lookups. Repair semantics (truncate index vs rebuild meta) are
deliberately left to a follow-up.

### Concurrency invariants

The code is correct *only* while all three hold.

1. **Single writer per workspace.** The document pipeline's `busy` /
   `destructive_busy` flags (see
   [PipelineConcurrencyContract.md](PipelineConcurrencyContract.md)) guarantee
   at most one process performs `upsert` / `delete` / `index_done_callback` at
   any time. Every other process is read-only with respect to this storage.
2. **Eventual consistency is sufficient.** Read-only processes only need to
   observe the writer's data *after* the writer's `index_done_callback`
   completes. Reads landing in the gap between a writer's in-memory mutation and
   its commit may legitimately return the pre-update snapshot.
3. **Mutations are synchronous and stay on the event loop.** Under a
   single-threaded asyncio loop, `client.upsert` / `client.query` /
   `client.delete` (Nano) and `index.add` / `index.remove_ids` /
   `self._id_to_meta` mutations (Faiss) cannot be preempted by another
   coroutine, which gives them implicit mutual exclusion. This is why the
   methods do not hold `_storage_lock` while calling into them.

   Faiss has one part that runs in a worker thread, `_save_faiss_index`, and it
   is compatible with the invariant because it only READS: it takes its
   `self._id_to_meta` snapshot on the loop before offloading, and reads
   `self._index` while holding `_storage_lock`, which excludes every mutation
   above. The only unlocked Faiss access, `query`'s `index.search`, is a read as
   well. Moving a MUTATION — or an unsnapshotted dict iteration — into the pool
   would break the invariant and would require widening the lock scope instead.

### Cross-process sync protocol

Two independent tests decide whether this process holds a current snapshot,
OR-ed at the one place that asks (`_reload_client_from_disk_locked` /
`_reload_index_from_disk_locked`). They are not redundant — their blind spots do
not overlap. The mechanism lives in `lightrag.kg.file_fingerprint`; the canonical
prose is the same-named section of
[NetworkXSingleWriterContract.md](NetworkXSingleWriterContract.md#cross-process-sync-protocol).

* **Authoritative channel — the file fingerprint.** `(st_mtime_ns, st_size)`
  against what this process recorded when it last loaded or wrote the file.
  State, not an event: nothing consumes it and a failed notification cannot lose
  it. Blind spot: two commits inside one filesystem timestamp tick with
  identical sizes.
* **Accelerator channel — the `storage_updated` flag.** Read first, because a
  `True` value already answers the question. `set_all_update_flags` publishes it
  with one Manager RPC per process, so a partial publication leaves a peer
  unnotified — that loss is its blind spot, and it is what the file channel
  exists for. In exchange it covers the fingerprint's tick collision, which needs
  a healthy, fast-committing system.

**Faiss: both files must have moved** for a change to count. The publication
renames them one at a time, so a pair where only one moved does not describe a
single state, and reloading THAT is the corruption vector — `_load_faiss_index`
binds every in-range metadata row to whatever vector the other file now holds. A
partial change therefore reports "no change" and this process keeps the older
self-consistent snapshot until the writer's retry completes the set (see
`file_fingerprint.peer_commit_detected`).

#### Writer side (`index_done_callback` / `finalize`)

1. Atomically write the in-memory state to disk (`atomic_write` swaps a tmp file
   into place; for Faiss, per file — cross-file atomicity is best-effort).
2. Record the fingerprint of what was just written, so this process does not
   later read its own save as a peer's. Done inside `_save_to_disk_locked` /
   `_save_faiss_index` — before the caller's bookkeeping, because that publishes
   through the manager and can fail, while this is a local `stat`.
3. `set_all_update_flags` flips every process's `storage_updated` flag
   (including the writer's own), then reset the writer's own flag to `False`.

**Faiss, on a FAILED save, adopts the pair anyway.** Step 1 is two
`atomic_write` calls, so a failure between them publishes a MISMATCHED pair — a
new `.index` beside the previous `.meta.json` — which is this process's own
doing, not a peer's. The in-memory index plus the redo logs are the authority
the retry writes from; reloading the mismatched pair instead would bind one
row's metadata to another's vector, and the replay would then delete the wrong
one. This is the opposite direction from `NetworkXStorage`, which reloads after a
failed save because it has no redo log, making its in-memory graph the
untrustworthy side. (It does not reach that reload by invalidating its
fingerprint — the file did not move, so the fingerprint still describes it — but
through a process-local `_recovery_reload_pending` flag.)

Adoption covers the failing writer only. Every OTHER process is covered by the
both-files-must-move rule above: they never recorded this pair, so nothing local
tells them the publication was interrupted; only the shape of the change on disk
does.

#### Reader and writer side

1. Inside `_storage_lock`, test the flag; if it is `False`, test the fingerprint.
2. On either, **fully reload**. Nano re-parses the entire JSON file and rebuilds
   a fresh in-memory matrix; Faiss re-inits `self._index` from `IndexFlatIP`,
   clears `self._id_to_meta`, and calls `_load_faiss_index` to re-parse both
   files. Neither library has an incremental sync API.
3. Record the new fingerprint **and** reset the flag, whichever channel fired.

### Why these backends never decline a stale write

`NetworkXStorage` refuses to save when the file has moved past its snapshot,
because it has no way to keep the mutation. Here the write path is
**reload-then-replay**: `index_done_callback` reloads the peer's snapshot and
`_flush_pending_locked` replays the pending buffer and the `_unsaved_upserts` /
`_unsaved_deletes` redo logs on top of it, so both sides survive and there is
nothing to report as a failure.

Replay is idempotent: an unchanged file matches by fingerprint and writes
nothing, and a genuinely newer row under the same id is declined by
`_resident_supersedes_redo`.

Consequently the flush-failure propagation `NetworkXStorage` needs — a declined
commit must not be acknowledged as durable, see `LightRAG._flush_storages` — has
no counterpart here.

The reason the graph store cannot do the same is recorded in
[its contract](NetworkXSingleWriterContract.md#why-not-reload-then-replay-as-the-vector-backends-do):
these buffers hold complete rows keyed by id, so replaying is last-writer-wins
per id, which is already their contract; graph payloads are accumulate-over-read
and replaying one drops the evidence a peer accumulated.

### Accepted residues

See *Consistency without transactions* in `AGENTS.md` — a documented residue is a
decision, an undocumented one is a defect.

* The tick collision above, **and** the notification lost for this process: the
  peer commit is not observed. Recovery: the next commit by any process flips
  the flag and changes the fingerprint; the redo logs mean this process's own
  rows are replayed rather than lost either way.
* A `stat` this process cannot perform (for Faiss, on either file): the file
  channel reports "no change" and the fence degrades to the flag alone, i.e. to
  the behaviour that predates it. See `kg.file_fingerprint` for why the other
  direction is worse.

### Lock scope

`_storage_lock` is a per-`(namespace, workspace)` keyed lock spanning both
intra-process coroutines and inter-process workers. It wraps only the *reload*
and *commit* critical sections, not every `client.xxx` call. Operating outside
the lock is safe *because of invariant 3* — if that premise is ever broken
(mutations moved to a thread pool, or the library swapped for an async one), the
lock scope must be widened to cover the mutation/read itself.

Faiss wraps, specifically: `_get_index` reload checks; pending-buffer mutations
in `upsert` and pending-buffer reads in `get_by_id` / `get_by_ids` /
`get_vectors_by_ids`; the single critical section in `index_done_callback` and
`finalize` (reload → flush → save → notify); the pending-cancel + rebuild
sections in `delete` / `delete_entity_relation`; and the entire `drop` body.

The lock is **non-reentrant**, so `_flush_pending_locked` /
`_remove_faiss_ids_locked` / `_save_faiss_index` /
`_reload_index_from_disk_locked` all require the caller to already hold it and
never re-enter via `_get_index` — the last of these runs its writes in a worker
thread, where re-entering would deadlock on a lock the caller already holds.

**Caveat — synchronous `client_storage` reads.** `client_storage` is a
synchronous property and does not go through `_get_index` / `_get_client`, so in
a reader process it can return data older than the latest committed snapshot
until some other method triggers a reload. The async read methods funnel through
the reload check after consulting the pending buffer, so they observe the latest
on-disk snapshot.

### Deferred-embedding protocol

`upsert` does **not** call the embedding model. It only buffers a pending doc
(content-bearing record + `vector=None`) in the minimal `self._pending_upserts`
area, overwriting any prior pending doc for the same id — which also clears a
temp vector a previous `get_vectors_by_ids` may have cached. The model is called
once per id at flush time (`_flush_pending_locked`), so repeated upserts of the
same id, and many small upsert calls, embed only once. `OpenSearchVectorDBStorage`
carries the same protocol.

Embedding runs **inside `_storage_lock`** during the flush, not in `upsert`:
under the single-writer invariant this keeps the content used for embedding
consistent with the record written to disk, and prevents a destructive op from
interleaving between embed and write. The lock is non-reentrant, so
`_flush_pending_locked` requires the caller to already hold it and operates on
the client / index directly.

**Faiss vector invariant:** once a `_PendingFaissDoc.vector` is set it is an
**already-L2-normalized float32 1D ndarray** — both flush and lazy
`get_vectors_by_ids` normalize the entire batch with `faiss.normalize_L2` before
caching back, so a later flush can `vstack` and `index.add` without
re-normalizing.

Reads are read-your-writes: `get_by_id` / `get_by_ids` / `get_vectors_by_ids`
consult `_pending_upserts` first, then fall back to the materialized store.
`get_vectors_by_ids` lazily embeds a pending doc on demand and caches the vector
back for the next flush. `query` and `client_storage` see only data already
materialized — unflushed pending data is intentionally not queryable.

A flush failure (embedding error, count mismatch, or save IO error) raises
through `index_done_callback`; the pending buffer is preserved. If only the save
failed, the flushed docs have moved into the `_unsaved_upserts` redo log (record
+ cached vector) and the dirty flag stays `True`: a later commit or `finalize`
reloads whatever another writer committed meanwhile and replays the logged rows
on top — **without re-embedding**. The log is cleared only once a save lands.

### Deferred-delete protocol

`delete` mirrors the buffering above: it cancels any pending upsert for the id
and queues the id in `self._pending_deletes` instead of removing the row.
`_flush_pending_locked` applies every queued id in **one** `client.delete` call,
and does so **before** materializing pending upserts, so an id that was deleted
and re-upserted in the same batch ends up with exactly the new row.

The motivation is cost: `NanoVectorDB.delete` rebuilds the entire matrix
(`np.delete`) on every call, and the entity/relation merge stage deletes the
stale forward/reverse rows once per relation — so an eager delete copied the
whole matrix per merged relation.

**Two buffers, because the flush and the save can fail independently:**

* `_pending_deletes` — queued, not applied to the client.
* `_unsaved_deletes` — applied to the client but not yet on disk, kept as
  `id -> a fingerprint of the removed row`. This is a redo log, not a pending
  buffer.

The redo log exists because a removal that reached the client can still be
undone: if the save fails, `index_done_callback`'s unconditional reload replaces
the client with the on-disk snapshot and the row returns. Replaying the log after
that reload removes it again, so the reload stays lossless for deletes. Upserts
carry the mirror log, so a materialized-but-unsaved upsert is replayed after the
same reload.

#### Replay matches on the row, not on the id alone

Ids are content hashes, so another writer can publish a *new* row under an id we
removed, and deleting by id would destroy it. The log therefore stores
`_row_fingerprint` of the row it removed — a digest of the whole stored record,
stable across a save/reload — and a replay removes only a row that still matches
it.

A whole-second `__created_at__` is not enough on its own: `upsert` stamps
`int(time.time())`, so a rewrite inside the same second as the removed row
carries the same timestamp. A rewrite *identical* in content to the row removed
is still a distinct row version: every write also stamps its own `__write_seq__`
token (see `write_seq`), so it fingerprints differently and the replay preserves
it — whether we wrote it or another writer did. Only a row written before the
token existed is indistinguishable from the row we removed, and the replay
removes it.

#### Ordering rule for the upsert replay

An id whose stored row already fingerprints equal to the logged record has
nothing to redo. Any *other* row under that id is ordered against ours by the
`__write_seq__` token, or by whole-second `__created_at__` when one of them
predates it: a strictly newer one is left alone and our redo entry is dropped
(another writer legitimately superseded us — a reprocess of the same document
under the same content-hash id — and reverting it would undo a completed
reprocess), while an older one is overwritten by ours. Only a tie no token can
break still falls back to being overwritten by the replay.

A removal request evicts the id's redo entry (`delete` / `delete_entity` /
`delete_entity_relation`), or the replay would resurrect the row the removal just
took out.

The read paths apply that same ordering rule over the same find-all row set
(`_resolve_resident_rows`), so read-your-writes never reports a row the replay is
about to decline to restore — not even in a corrupt store where one id carries
several rows.

Ids that matched no row are not logged at all — there is nothing to persist for
them, and replaying them could only hit a row they were never meant to touch. The
log is cleared once a save lands, and an aborting batch keeps it:
`drop_pending_index_ops` discards buffered work, not removals that already
reached the client.

`delete_entity` / `delete_entity_relation` stay eager — they are off the merge
hot path — but their removals are applied-and-unsaved just the same, so they are
recorded in the log too.

#### Why the two buffers are scoped differently

`_pending_deletes` holds a *request*: the caller named an id, so the flush
removes whatever row carries that id — the by-id contract every server-backed
backend implements, and the one purge relies on to leave nothing behind.
Version-scoping a request would silently skip a delete the caller asked for, and
pinning the version at `delete` time would also put back the per-call `O(rows)`
lookup this protocol exists to remove.

`_unsaved_deletes` holds a *record* of a removal that already happened, so it is
version-scoped: replaying it must not remove a row that has since taken the id's
place.

The read-your-writes paths mirror both rules. `get_by_id` / `get_by_ids` /
`get_vectors_by_ids` consult the buffers after `_pending_upserts`: a queued id
reads as absent, while a logged one hides only the row the entry names — a
replacement the replay would preserve stays readable. An `upsert` cancels a
*queued* delete for the same id (`client.upsert` overwrites the row in place, so
applying it first would be redundant work) but never the redo log: the buffered
row may be discarded by an aborting batch before it materializes, and dropping
the entry then would leave the reload nothing to replay. `query` and
`client_storage` are unchanged: they read the materialized index, so a queued
delete still surfaces there until the flush — the same contract the Qdrant and
PostgreSQL buffers document.

Both buffers are in-memory only: they are dropped by `drop` and lost on a crash
before the flush.

#### Being replayable is what lets `finalize` reload before it retries a save

Without the logs it had to choose between skipping the reload (saving a
pre-commit snapshot over another writer's rows) and reloading (dropping its own
unsaved changes). With both logs, every retry path reloads first and replays on
top. The reload is unconditional: `finalize` runs it even when both logs are
empty and only the dirty flag is set, since a bare dirty flag still means the
in-memory state may be stale relative to disk. What a replay deliberately does
*not* restore is a row another writer has since superseded with a strictly newer
one.

### Non-pipeline write paths

The pipeline's `busy` gate serializes `upsert` / `delete` /
`index_done_callback` called from the document ingestion and purge flows. These
entry points are **not** serialized by it and must be guarded externally:

* `drop` — gated by the API layer (the `/documents/clear` endpoint takes the
  pipeline busy reservation before invoking it).
* `delete_entity` / `delete_entity_relation` — reached from the `utils_graph.py`
  admin flows, which the WebUI exercises through the `/graph/*` endpoints.
  Admin-vs-pipeline is guarded by `check_pipeline_busy_or_raise`. Admin-vs-admin
  is not, and deliberately stays that way: these backends replay rather than
  decline, so an overlapping pair loses nothing, which is why the workspace-wide
  admin lock that `NetworkXStorage` requires is not requested here. See
  [the admin write paths](NetworkXSingleWriterContract.md#admin-write-paths).

A flush here publishes every pending upsert buffered in this instance, not only
the caller's, so a co-tenant's unfinished sequence can become durable on someone
else's commit. Unlike the graph store these classes lose nothing to it (a peer
commit is reloaded and the buffers replayed on top), but the *timing* is still
not the caller's to choose.

## `JsonKVStorage` — shared in-memory state, no reload path

This class uses a fundamentally different cross-process model from the three
above, which keep one in-memory copy per process and reconcile via file reloads.

### Storage model

`self._data` is **not** a per-process dict — it is the value returned by
`get_namespace_data(namespace, workspace=...)`, i.e. a reference into
`shared_storage._shared_dicts`. In multiprocess mode this is a
`multiprocessing.Manager().dict()` proxy that every worker sees the **same
instance** of; in single-process mode it degrades to a plain `dict`. Either way,
a mutation in any process is *immediately* visible to every other process —
there is no reload needed.

The on-disk file at `working_dir/[workspace/]kv_store_<namespace>.json` exists
for durability only. It is the source of truth at startup and the target of
`index_done_callback` flushes, but is **not** part of the steady-state
read/write path.

### First-time load (`initialize`)

`try_initialize_namespace` is a global init lock that returns `True` to exactly
one process per `(namespace, workspace)`. That process reads the JSON file and
populates `self._data` under `_storage_lock`. Other processes skip the load —
they will see the data through the same shared proxy.

### Reversed flag semantics

Anyone writing (`upsert` / `delete` / `drop`):

1. Mutate `self._data` under `_storage_lock` (same lock, same dict, all
   processes see the change immediately).
2. Call `set_all_update_flags` to mark **every** process's `storage_updated`
   flag `True`. Here `True` means *"there is dirty data that still needs to be
   flushed"*, **not** *"there is fresher data on disk that I need to reload"* as
   in the file-backed classes.

Commit (`index_done_callback`):

1. Under `_storage_lock`, if `storage_updated.value` is `True`, snapshot
   `self._data` and write it to disk via `write_json` (atomic).
2. `clear_all_update_flags` — wipe every process's flag back to `False`. Because
   the in-memory state is already consistent across processes, there is nothing
   for the *other* processes to do; the clear is just a "the dirty data has been
   persisted" signal.

### Lock scope

Unlike the file-backed classes, which only lock reload/commit critical sections,
this class **holds `_storage_lock` over every `self._data` access** — read or
write — because the underlying `Manager().dict()` is not free-threaded across
processes.

Two places intentionally do work outside the lock for latency reasons:

* `upsert` performs its per-key timestamp prep loop inside the lock but yields to
  the event loop via `_cooperative_yield` between keys (safe: `NamespaceLock` is
  non-reentrant, so siblings blocked on it stay blocked).
* `JsonDocStatusStorage.upsert` prepares its caller-supplied dict outside the
  lock — it only mutates the input, not the shared store.

### Commit granularity — a commit publishes the whole namespace

`index_done_callback` snapshots the entire `_data` dict and rewrites the whole
JSON file. There is no scoped or transactional commit, and adding one was
considered and rejected for the file-backed storages. So **any writer's flush
durably publishes every other writer's pending in-memory mutation in this
namespace.**

This matters most for the chunk-tracking namespaces (`entity_chunks` /
`relation_chunks`), whose rows are the authoritative attribution carriers behind
`_purge_kg_contributions`: a row and the graph object it describes live in
different stores with no transaction between them, and the forbidden ordering is
the object durable without the row. `utils_graph._persist_graph_updates` commits
the rows first for exactly that reason; a co-tenant's flush can still publish a
row early, which lands in the benign direction. The full residue is in
[the dirty-graph backstop section](NetworkXSingleWriterContract.md#accepted-residue-crash).

### Who can write

Pipeline `busy` still serializes the document ingest / purge flows, but the
*file-flush trigger* is symmetric: any process whose `storage_updated.value` is
`True` when `index_done_callback` fires will perform the write. In a
single-writer pipeline this is always the same process; if you ever permit
multiple writers, two processes may race to flush the same in-memory state —
that race is safe (both flush the same shared dict, `write_json` is atomic per
file) but wasteful, and the `clear_all_update_flags` after each flush means
subsequent re-flushes are no-ops.

### Caveats vs the file-backed implementations

* **No reload path.** If something writes to the on-disk file out of band, this
  class will not pick it up until restart. The file is only ever written by
  `index_done_callback` and read once in `initialize`.
* **No `_get_*` entry method.** Adding one would be wrong — there is nothing to
  "get fresher than", since the in-memory state is already the shared,
  authoritative view.
* **`write_json` may sanitize.** If sanitization happens, the on-disk JSON
  differs from what was in memory; the callback re-reads the cleaned file back
  into `self._data` under the same lock so the shared view stays consistent with
  disk.

Because nothing here can discard an uncommitted write, this class carries no
`requires_single_writer` capability flag — see
[the admin write paths](NetworkXSingleWriterContract.md#admin-write-paths).

### Non-pipeline write paths

* `drop` — destructive, **not** serialized by this storage class. Currently
  gated by the API layer (`/documents/clear`); any new caller must hold the
  pipeline `busy` reservation.
* `upsert` / `delete` invoked from non-pipeline admin flows (cache management,
  etc.) — safe under the shared-lock model, but consumers should still respect
  the pipeline gate to avoid interleaving with batched ingest work.
