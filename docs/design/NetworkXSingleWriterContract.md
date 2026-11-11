# NetworkX Single-Writer Contract

Read this before touching `lightrag/kg/networkx_impl.py`, `lightrag/kg/file_fingerprint.py`,
or any caller of `index_done_callback` on the graph store.

`NetworkXStorage` keeps the whole knowledge graph in process memory and publishes
it by rewriting one GraphML file. That shape makes every cross-process concern —
visibility, lost updates, partial state — a property of *when the file is read and
written*, not of a database. This document is the reasoning behind the code; the
docstrings in the module state the rules and point here.

## Scope

This backend — like `JsonKVStorage` / `JsonDocStatusStorage` /
`NanoVectorDBStorage` / `FaissVectorDBStorage` — is supported for **small-scale
testing and validation only**. Production deployments run a server-backed graph
store. Write throughput of the whole-file commit path is therefore not a
consideration, and no change to this file may be justified by — or blocked on —
it.

## Storage model

A single `networkx.Graph` instance lives in process memory; its full state is
serialized to one GraphML file at `working_dir/[workspace/]graph_<namespace>.graphml`.
That GraphML file is the **only** cross-process synchronization surface — there is
no shared memory, no message bus, and no network channel between processes.
Cross-process visibility is mediated by (a) an atomic file write at commit time
and (b) a per-namespace `storage_updated` flag distributed through
`lightrag.kg.shared_storage`.

## Concurrency invariants

The code in this module is correct *only* while all three hold.

1. **Single writer per workspace.** The document pipeline's `busy` /
   `destructive_busy` flags (see
   [PipelineConcurrencyContract.md](PipelineConcurrencyContract.md)) guarantee at
   most one process performs `upsert_*` / `delete_*` / `remove_*` /
   `index_done_callback` at any time. Every other process is read-only. The admin
   flows in `utils_graph.py` and `ainsert_custom_kg` are brought under the same
   invariant by `LightRAG._admin_write_gate`, which this class requests through
   `requires_single_writer` — see [Admin write paths](#admin-write-paths).
2. **Eventual consistency is sufficient.** Read-only processes only need to
   observe the writer's data *after* the writer's `index_done_callback`
   completes. Reads landing in the gap between a writer's in-memory mutation and
   its commit may legitimately return the pre-update snapshot.
3. **networkx operations are fully synchronous.** Under a single-threaded asyncio
   event loop, `graph.add_node` / `graph.remove_node` / `graph.degree` / etc.
   cannot be preempted by another coroutine, which gives them implicit mutual
   exclusion over `self._graph`. This is why the module's methods do not hold
   `_storage_lock` while calling into `graph`. The one place that is NOT on the
   event loop is the GraphML serialization, which runs in the storage-IO pool —
   see [Commit gate](#commit-gate) for what re-establishes the exclusion there.

## Commit gate

`index_done_callback` hands `self._graph` to a worker thread, which iterates
`graph._node` / `graph._adj` for the length of the write. Invariant 3 does not
cover that: a coroutine on the loop could mutate the graph mid-iteration, tearing
the snapshot or raising `RuntimeError: dictionary changed size during iteration`
from inside the writer.

`_commit_gate` is an `asyncio.Event`, set except while this process is
serializing. The committer clears it inside `_storage_lock` and restores it in a
`finally` (every path, cancellation included — leaking a cleared gate once
deadlocks every later graph operation in this workspace). `_get_graph` waits on
it.

**Holding `_storage_lock` across the write is NOT sufficient on its own**, which
is why the gate exists as well. Releasing a `NamespaceLock` runs the release on a
fresh task and awaits a shield, so `__aexit__` always suspends at least once. A
mutator that has read `self._graph` and is suspended in that release is past the
lock and would resume — and mutate — while the worker thread is iterating. The
gate is therefore checked AFTER `_get_graph`'s lock block, i.e. after the last
suspension point: from the `is_set()` check through the caller's synchronous
`graph.add_node()` there is no `await`, so no commit can start in the middle of a
mutation.

The gate sits at `_get_graph`, the single choke point every read and write goes
through, rather than in the seven mutators. Reads are gated too. That is
deliberate and free: a reader already has to pass `_storage_lock`, which the
committer holds throughout, so the gate adds no wait it did not already have —
and a `for_write=True` variant would buy nothing while forking the semantics of
the one choke point.

## Cross-process sync protocol

The fence rests on **two** independent tests, OR-ed at every point that decides
whether this process holds a current snapshot. They are not redundant: their
blind spots do not overlap.

* **Authoritative channel — the file fingerprint.** `(st_mtime_ns, st_size)` of
  the GraphML file, compared against what this process recorded when it last
  loaded or wrote it (`_loaded_fingerprint`). It is *state*, not an event:
  nothing consumes it, nothing can lose it, and it matches the storage model
  above, where the file is the only cross-process synchronization surface. Blind
  spot: two commits landing inside one filesystem timestamp tick with an
  identical file size.
* **Accelerator channel — the `storage_updated` flag.** Distributed through
  `lightrag.kg.shared_storage`. Read first, because a `True` value already
  answers the question. It is an *event*: the reader consumes it, and
  `set_all_update_flags` publishes it with one Manager RPC per process, so it can
  be lost — for one process, for several, or for all. Blind spot: exactly that
  loss. It covers the fingerprint's tick collision, because a collision needs a
  healthy, fast-committing system.

### Writer side (`index_done_callback`)

1. `write_nx_graph` atomically writes the GraphML file (`atomic_write` lays a tmp
   file beside the target and renames it into place — readers either see the
   previous file in full or the new file in full, never a torn write).
2. Record the fingerprint of the file just written, so this process does not
   later mistake its own commit for a peer's. A local `stat`, done before step 3
   because it cannot fail with the manager.
3. `set_all_update_flags` flips every process's `storage_updated` flag (including
   the writer's own), then reset the writer's own flag to `False`.

### Reader side (any method that goes through `_get_graph`)

1. Inside `_storage_lock`, test the flag; if it is `False`, test the fingerprint.
2. On either, **fully reload** `self._graph` from disk via `load_nx_graph`.
   networkx GraphML has no incremental sync API, so the entire file is re-parsed.
3. One reload, one post-condition, whichever channel fired: record the new
   fingerprint, clear the flag, and clear any pending recovery reload
   (`_reload_locked` — do not open-code any of the three).

### Writer side, before saving

The same two tests. A writer holding a snapshot the file has moved past
**declines** (returns `False`) instead of writing: its save serializes the whole
graph, so proceeding would overwrite the peer commit it never saw.

A declined commit must reach its caller as a failure, because declining DISCARDS
this process's pending mutation. All three callers do that:

* `utils_graph._commit_graph_or_raise` raises on the `False` (`adelete_by_entity`,
  `_edit_entity_impl`, `_merge_entities_impl`);
* `LightRAG._flush_storages`'s `_flush_one` turns it into `IndexFlushError`
  (pipeline paths);
* `utils_graph._persist_graph_updates` raises `_declined_commit_error` for the
  graph store — and only for it, since the vector stores carry no such fence
  (`aedit_relation` / `acreate_entity` / `acreate_relation`).

Without the second one the fence would only swap which side loses data — the
peer's commit preserved, this document marked PROCESSED with its graph writes
dropped and nothing to recover them from. As a failure it heals instead: the
document goes FAILED and its reprocessing re-extracts and re-writes the work.

The third was the last one to get there: it discarded the return value until the
ordering rework that introduced the write-ahead tracking rows. Before that a
decline through the admin create and edit paths was silent — the graph mutation
discarded, the same operation's vector and tracking rows durable, and the caller
told it succeeded.

### Ordering rule that keeps the fingerprint honest

It is sampled **before** the file is read, never after. A fingerprint sampled
after the parse can belong to a newer file than the one now in memory, and
recording it would suppress the reload that newer file needs — the one way this
fence could *introduce* a lost write. Sampling early can only cost a redundant
reload, which is harmless.

A sample that fails outright (`UNREADABLE`) is not a third option for a reload:
`_reload_locked` refuses to load at all, because it could neither trust nor
record what it read. See its docstring for both halves of that, and
`file_fingerprint.adopted` for the residue that remains where a `None`
fingerprint is still recorded.

Single-process mode skips the fingerprint test entirely
(`file_fingerprint.fence_enabled()`): there is no peer that could have committed,
so a divergent file means an external edit, and reloading for it would discard
this process's own uncommitted mutations.

Both sites test one more thing first, and it is not a channel: see
[Recovery reload](#recovery-reload) for the process-local condition that says "my
own memory is unpersisted" rather than "a peer committed".

### Accepted residues

See *Consistency without transactions* in `AGENTS.md` — a documented residue is a
decision, an undocumented one is a defect.

* Two commits inside one filesystem timestamp tick, with an identical file size,
  **and** the notification lost for this process: this process does not observe
  the second one. Recovery: the next commit by any process both flips the flag
  and changes the fingerprint. Timestamp granularity is kernel-dependent (~10 µs
  on Linux ≥ 6.13 multigrain timestamps, 1–4 ms on older kernels, coarser on
  ext3 / HFS+ / FAT), while the deployment shape this fence exists for — a peer
  becoming the next writer through a later request — separates the two commits by
  at least a request round trip.
* A `stat` this process cannot perform (permissions, EIO): the fingerprint
  channel reports "no change" and the fence degrades to the flag alone, i.e. to
  the behaviour that predates it. Failing towards a reload instead would install
  an empty graph from a file it cannot read, and the next commit would serialize
  that over the real one.
* The *same* failure inside a **reload** is refused rather than absorbed:
  `_reload_locked` raises without loading. Both of the alternatives lose data.
  Loading blind installs an empty graph (`load_nx_graph` gates on
  `os.path.exists`, false for any stat failure) that the next commit writes over
  the real file; loading and recording `adopted(UNREADABLE)` — i.e. `None`,
  against which every state is a divergence — makes the NEXT `_get_graph` reload
  again and discard whatever was mutated in between, after which the commit
  SUCCEEDS without it and the document is marked PROCESSED. Refusing costs the
  batch, which the FAILED path reprocesses.
* What remains of that failure is the writer adopting its OWN commit or drop
  through `_record_fingerprint` (and `initialize`'s first load), which records
  `None`. That costs one redundant reload of a graph that already equals the
  file, plus — if the `stat` recovers in time for the divergence branch — one
  `_missed_notification_reloads` increment for a peer commit that never happened.
  Bounded, self-healing on the next readable `stat`, and biased towards over-
  rather than under-reporting. What is NOT optional is that the redundant reload
  land BEFORE the next mutation, which is why `_get_graph` treats a `None`
  fingerprint as unresolved: a readable sample settles it through the divergence
  test, and one that is `UNREADABLE` — reporting "no change", so no channel would
  act on it — makes the call REFUSE instead of serving a snapshot a mutation
  would then land on, whose reload would arrive mid-batch and drop it. See
  `file_fingerprint.adopted` for why the obvious fix to the `None` itself is
  recorded there as an option rather than taken.

## Recovery reload

Process-local, and NOT a third fence channel.

`_recovery_reload_pending` is a plain `bool` on the instance, and the rule for it
is **armed when the reload becomes owed, cleared only by one that completes** —
never "armed when a reload fails".

Owed is the earlier moment and the safe one: it is the branch entry, before the
sample is classified and before anything is loaded. `_reload_locked`'s third
post-condition clears it, so the happy path needs no bookkeeping, and
*everything* that can go wrong in between leaves the record standing — including
`_count_unannounced_peer_commit_locked`, which reads `storage_updated.value` over
the Manager and can raise before any reload is attempted. Two earlier attempts at
this were both too late: arming only in the failed-SAVE handler left the other
four sites unarmed, and moving it into `_reload_locked`'s failure paths still
missed everything that raises before the call. Both had the same consequence — an
unreadable `stat` blinds the channels, and the next commit publishes work already
reported as failed over a peer's durable commit.

What the state means, whichever site armed it: this process holds a graph that
does not match the file, and the operation that wanted it failed, so what the
graph holds belongs to work already reported as failed. It is tested first at
both sites the two channels are tested at: `_get_graph` discards the divergent
graph before serving it, and `index_done_callback` declines rather than
publishing mutations that belong to a batch already reported as failed.
`_reload_locked` clears it, as its third post-condition.

It states a different fact from either channel, and the distinction is the point.
The channels answer *"did a peer commit?"* — the file has moved on and this
process's snapshot is behind it. This answers *"do I still owe myself a reload?"*
— whatever the file did, this process failed to converge on it and cannot say
what its own graph represents. Same remedy, and it outranks both channels because
a reload discharges all three while only this one is certain.

Armed after a failed save, that means "my memory is unpersisted": the file has
not moved and it is memory that is wrong. Armed after a failed reload at either
channel, it means the opposite — the file moved and this process could not
follow. The remedy does not care, and neither does the danger: in both, the graph
holds work already reported as failed, and publishing it is what must not happen.

### Why not `storage_updated`, which carried this before

* **Accuracy.** That flag means "a peer committed". Nothing here committed, and
  no peer is involved. Every log line downstream of it said "modifications by
  another process" for an event that had none.
* **Clean evidence.** `_missed_notification_reloads` counts lost notifications,
  and it is the instrument the two-channel fence leaves behind to decide whether
  the writer-side `os.utime` monotonicity option is ever needed. Arming the
  *file* channel for recovery — which the previous code did, by invalidating
  `_loaded_fingerprint` — made a failed flag write show up as a lost notification
  that never happened. A process-local test cannot overcount that way.

  It can *under*count, though, and that is what
  `_count_unannounced_peer_commit_locked` exists for: this test wins over both
  channels, and one reload discharges all of them, so a peer commit that arrived
  unannounced *while* recovery was pending would be handled correctly and never
  counted. Both recovery branches therefore classify the peer channel before they
  reload — and so does the reload that would otherwise arm the flag, the
  failed-save handler's own recovery reload. Overcounting and undercounting are
  both defects in an instrument a later decision rests on, which is why that one
  helper is the single increment site for every branch that counts — and why it
  deduplicates by `(st_mtime_ns, st_size)`: a reload that raises leaves the
  fingerprint and this flag untouched, so without that the same peer commit is
  re-counted by every later call, without bound.
* **Arming cannot fail.** In multiprocess mode `storage_updated` is a
  `Manager().Value` proxy, so arming it was an RPC to the very process whose
  outage may be why the reload just failed. That needed its own failure path, its
  own best-effort log, and an ordering argument against the fingerprint half. An
  attribute write needs none of them, and covers single-process mode too — where
  the file channel is gated off entirely, so the flag was the only thing arming
  recovery at all.

  **Arming**, precisely — not the recovery path, which still reads
  `storage_updated.value` to classify and writes it in `_reload_locked`, both
  Manager RPCs. A manager still down when recovery is attempted raises out of
  `_get_graph` / `index_done_callback` with the flag **still armed**, which is the
  outcome to want: the divergence stays visible and the next call retries. What
  the change buys is that *recording* the divergence no longer depends on the
  thing that may have caused it — the old code could lose the fact itself, and
  then nothing later would retry.

The fingerprint is deliberately *not* invalidated when this is armed: the save
failed, so the file is untouched and the recorded fingerprint still describes it
correctly. Saying otherwise would be a second lie in the opposite direction.
Untouched *by this process*, that is — `index_done_callback` releases the lock
between its fence block and its save, so the recovery reload in its failure
handler classifies the peer channel before it adopts the file, like the branches
above.

`drop` clears it: the mutation it protects is destroyed with everything else, and
memory matches the file again. That clear is load-bearing, not cosmetic — the
flag is sticky and is tested by `index_done_callback`, so leaving it set would
make the first commit after a clear decline and discard fresh work.

## Lock scope

`_storage_lock` is a per-`(namespace, workspace)` keyed lock spanning both
intra-process coroutines and inter-process workers. It wraps only the *reload*
and *commit* critical sections, not every `graph.xxx` call. Operating on `graph`
outside the lock is safe *because of invariant 3* plus the commit gate, which
covers the one case invariant 3 does not: the serialization running in a worker
thread. If invariant 3 is broken further — `graph.xxx` itself moved to a thread
pool, or networkx swapped for an async graph library — the gate is no longer
enough either and the lock scope must be widened to cover the mutation/read
itself.

## Commit granularity — a commit publishes the whole namespace

`index_done_callback` serializes `self._graph` in full and renames the result over
the GraphML file. There is no scoped or transactional commit, and none is planned
(scoped commits were considered and rejected for the file-backed storages
explicitly). Two consequences the callers have to live with:

* **Any writer's flush durably publishes every other writer's pending in-memory
  mutation in this namespace**, including partial state its author had not
  finished. A caller that stages its own writes to guarantee an ordering
  guarantees it only for its own objects; a co-tenant's half-applied sequence
  rides along.
* The pipeline tolerates this, because `doc_status` plus idempotent FAILED
  reprocessing rewrites whatever a restart finds half-applied. The admin flows
  below have no equivalent, which is what makes their residue worth stating
  rather than assuming away.

## Admin write paths

The pipeline's `busy` gate serializes mutation calls reached through the document
ingestion and purge flows. The following entry points are not reached through it,
and each is serialized by something else:

* `drop` — gated by the API layer (the `/documents/clear` endpoint takes the
  pipeline busy reservation before invoking it).
* `delete_node` / `remove_nodes` / `remove_edges` / `upsert_node` / `upsert_edge`
  and the batch variants when invoked from the `utils_graph.py` admin flows
  (`adelete_by_entity` / `adelete_by_relation` / the create, edit and merge
  paths) or from `ainsert_custom_kg` — gated by `LightRAG._admin_write_gate`,
  which every one of those eight public writers runs inside.

For the admin flows invariant 1 is met **by the gate**, and this class asks for it
through `requires_single_writer = True` (the only storage that does):

* Admin-vs-admin is guarded by a workspace-wide admin lock
  (`{workspace}:GraphAdmin` / `"admin"`, cross-process). A second admin write
  WAITS for the first, bounded by `ADMIN_WRITE_LOCK_ACQUIRE_TIMEOUT`, and is
  refused with 409 only on expiry. The lock covers mutate AND commit, so no peer
  admin commit can land between two of one flow's `_get_graph` calls.
* Admin-vs-pipeline is guarded in both directions. An admin write holds the
  pipeline `busy` reservation (`kind="admin"`, no `destructive_busy`) for its
  duration, so a pipeline start arriving meanwhile is deferred into the ingress
  mailbox and driven once the gate releases; a pipeline already running refuses
  the admin write with 409, as the router's snapshot check already did and still
  does (kept as an early refusal before any embedding work). The hold is bounded
  by `admin_write_max_hold_seconds`.

Full mechanics of the gate itself are in
[PipelineConcurrencyContract.md](PipelineConcurrencyContract.md).

### Why the workspace admin lock is back after being dropped

An earlier revision dropped it, arguing that with the two-channel fence in place
the losing writer of an overlapping pair *declines* and gets a loud 500. That
holds only if `index_done_callback` is the next fence test after the peer commit
— and for every admin flow it is not: an intervening `_get_graph` (a read, or the
flow's next mutator call) catches the peer commit first, DISCARDS the mutations
already applied, and the commit that follows finds no divergence and succeeds
without them. No decline, no 500 — exactly the silent loss the drop said could no
longer happen. Reachable in one process too: two `LightRAG` instances on one
workspace each register their own update flag, and one's commit makes the other
reload.

### Why not reload-then-replay, as the vector backends do

`NanoVectorDBStorage` / `FaissVectorDBStorage` reload the peer snapshot and replay
their pending buffer and redo logs on top (`_flush_pending_locked`), so they lose
nothing and report nothing. Their pending buffers hold complete rows keyed by id,
so replaying is last-writer-wins per id, which is already their contract.

Graph payloads are accumulate-over-read — `source_id` is an evidence set merged
from what the writer read, and `weight` is floored by the evidence count it read
— so replaying one over a peer's newer state drops the evidence the peer
accumulated and republishes a `weight` computed against a stale source set,
silently breaking the relation-weight contract; a replayed delete removes an
object the peer may have re-created. Replay would turn a loss the fence can still
see into one nothing can. **Do not re-open this without addressing the
accumulate-over-read point.** What was kept from that proposal is the loud
assertion: see [Dirty-graph backstop](#dirty-graph-backstop).

`JsonKVStorage` / `JsonDocStatusStorage` are neither case: they have no reload
path at all (their data is a shared Manager mapping, already cross-process
consistent), so nothing there can discard an uncommitted write and they carry no
capability flag.

## Dirty-graph backstop

The gate makes the mid-flow discard unreachable by enumeration of its callers,
and an enumeration rots. `_graph_dirty` records that `self._graph` holds
mutations no commit has published (set by every mutator, cleared when the graph
is replaced from disk or a commit or drop lands). A reload that would replace a
dirty graph still replaces it — the reload is correct, and the coroutine that
triggered it may be an innocent reader across the writer's embedding `await` —
but logs at ERROR and arms `_dirty_discard_pending`.

The enforcement point is the COMMIT: `index_done_callback` refuses with
`GraphMutationsDiscardedError` while that flag is set, clearing it in the same
step so a later commit is not blocked forever (a sticky refusal would make every
later commit fail). The writer therefore fails loud — a pipeline batch takes the
FAILED path and is reprocessed, an admin request gets a 500 — instead of
succeeding without its mutations. If the refusing commit finds the graph dirty
again (mutations applied after the discarding reload, belonging to the operation
now failing), it also arms `_recovery_reload_pending` so the next `_get_graph`
discards them rather than a later commit publishing them.

**Exempt, and they must stay so:** the recovery reload
(`_recovery_reload_pending`), which exists precisely to discard mutations of a
batch already reported as failed, and the decline paths of `index_done_callback`,
which already report the loss to their caller.

**NOT exempt**, contrary to how this backstop was first specified: a notification
whose sampled fingerprint equals the loaded one. That was specified as a
self-notification, which replaces nothing a peer wrote — but an equal fingerprint
does not identify a self-notification. A same-tick, same-size peer commit is the
file channel's documented blind spot, and a set flag is the evidence that the
channel is blind right now (the flag exists to cover exactly that collision).
Disarming on the blind channel's verdict is how a peer-replaced dirty graph would
be discarded silently.

### Accepted residue

The refusing commit is not necessarily the operation whose mutations were
discarded: if that writer is already gone, the next unrelated commit refuses once
and clears the flag. That is loud, it converges, and it is strictly better than
the silent success it replaces. Under the gate the branch is unreachable, so the
residue is unreachable with it. The one variant the gate cannot cover — two admin
requests a second apart on different workers, where a lost notification lets the
second mutate a stale snapshot — belongs to the two-channel fence, and this
backstop is its second line.

### Accepted residue (crash)

A hard process exit inside an admin write can leave a chunk-tracking row whose
graph object never became durable; the gate does not change that. It is harmless
to queries; it cannot be inherited as evidence by a later object, because the
explicit creation paths reset attribution; and it is repairable offline with the
chunk-tracking rebuild tool. The dead admin owner's `busy` reservation is
reclaimed without fencing the workspace (`kind="admin"` is re-runnable).

The mirror state — a graph object durable without its tracking row — is the
forbidden one, and the creation paths in `utils_graph` keep it out of reach by
writing and committing the tracking row **before** calling `upsert_node` /
`upsert_edge` at all. Ordering only the flushes would not be enough here: an
uncommitted `upsert_node` still sits in the process-wide in-memory graph, where
the next flush publishes it.

The edit paths reach the same guarantee by a different route, because their delta
can also REMOVE ids and a narrowed row landing ahead of the graph write is itself
the over-deleting state: `aedit_relation` and `_edit_entity_impl`'s non-rename
branch stage the row as grow-then-shrink — superset row, graph write, final row —
so the durable row is never a strict subset of the durable evidence.

### Requirement on new callers

Any new caller of these mutators must make the tracking row durable BEFORE the
mutation call, not merely before the flush, and must run inside
`LightRAG._admin_write_gate` (or the pipeline's `busy` reservation) so it is
serialized against every other writer on the workspace. A caller that bypasses
both trips the dirty-graph backstop above: loud, not silent.

## Attribute validation

Why this backend validates, and why the rule is narrower than the caller
contract.

The `upsert_*` methods reject an attribute name or value XML cannot encode
(`validate_xml_attributes`) **before** touching `self._graph`. This backend needs
the guard more than the others because of the shape above, not because its
callers are less trustworthy: the mutation happens in memory, the serialization
that would reject the value happens later in `index_done_callback`, and nothing
rolls the mutation back. One unencodable value therefore stops *all* persistence
for the life of the process — `write_nx_graph` serializes the whole graph, so
every later flush by any caller re-hits the same failure while reads keep
succeeding. Validating first converts that into a failed single write. See
GHSA-c922-pw4m-4wcv.

The rule is exactly "can GraphML encode this", not the portable contract in
`BaseGraphStorage.upsert_node`. `NaN`, `inf` and an integer past int64 are all
refused by that contract (the Neo4j driver cannot pack them) but round-trip
through GraphML unchanged — so a workspace can already hold one, and every
rewrite path spreads a fetched object's stored attributes back into the upsert
payload. Enforcing the portable rule here would make those objects permanently
unmodifiable; enforcing it where caller input enters (`utils_graph`) costs
nothing. The portable bounds are not this backend's to police.

Names get the same XML rule and nothing more. GraphML writes them into the XML
`attr.name` field, so an unencodable *name* breaks the write exactly like an
unencodable value — but `a.b`, `$set`, `display-name` and `has space` all
round-trip, so refusing those would strand a node whose stored names predate this
validation while preventing nothing. Rules about names a backend *interprets*
belong to the backends that interpret them (MongoDB's `$set` paths).

The XML name rule is safe to apply to a rewrite payload for the reason a portable
rule would not be: a name it rejects can never have been persisted here, because
the write that would have stored it failed.

The batch variants validate the entire batch before applying any of it: rejecting
halfway would leave the earlier items in the in-memory graph, which is exactly
the partial-mutation state the guard exists to prevent.

All four methods validate *after* `_get_graph()` — their only await — and then
mutate with nothing awaited in between. That ordering is what makes the guard
airtight rather than advisory: by invariant 3 a synchronous run cannot be
preempted, so a caller that retains the mapping it passed in has no window in
which to add a value after the check and before `add_node` / `add_edge` consumes
it.

## Implementation differences from `NanoVectorDBStorage`

Same design, different surface.

* **The fence is shared, its verdict is not.** All three test the file
  fingerprint as well as the flag, through the same `lightrag.kg.file_fingerprint`
  helpers. What differs is what happens once a peer commit is detected on a WRITE
  path: this class **declines** the save (it has no way to keep the mutation),
  while the vector backends reload and replay. The flush-failure propagation a
  decline requires (`LightRAG._flush_storages`) is therefore this class's concern
  alone.
* No `client_storage` property — there is no equivalent live reference being
  exposed to callers, so NanoVectorDB's "do-not-retain-across-await" caveat does
  not apply here.
* `write_nx_graph` passes the tmp path directly to `nx.write_graphml`, so the
  writer needs no equivalent of NanoVectorDB's "temporarily reassign
  `storage_file`" trick.
* Mutation surface is finer-grained (`upsert_node` / `upsert_edge` /
  `upsert_nodes_batch` / `upsert_edges_batch` / `delete_node` / `remove_nodes` /
  `remove_edges`); each goes through `_get_graph` once and then operates
  synchronously on `self._graph`.
